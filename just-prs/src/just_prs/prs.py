"""Core PRS computation engine: variant matching, dosage computation, weighted sum."""

import math
from pathlib import Path

import polars as pl
from eliot import log_message, start_action

from just_prs.models import PRSResult
from just_prs.scoring import DEFAULT_CACHE_DIR, load_scoring, parse_scoring_file
from just_prs.vcf import compute_dosage_expr, read_genotypes


def _normalize_genotype_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Ensure genotype LazyFrame has the columns expected by compute_prs.

    polars-bio produces ``start`` while compute_prs joins on ``pos``.
    This renames ``start`` → ``pos`` when the caller passes an external
    LazyFrame that hasn't been through ``read_genotypes()``.
    """
    cols = lf.collect_schema().names()
    if "pos" not in cols and "start" in cols:
        lf = lf.rename({"start": "pos"})
    return lf


def _resolve_scoring(
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str,
    cache_dir: Path,
) -> pl.LazyFrame:
    """Resolve a scoring file argument into a LazyFrame.

    Accepts a Path to a local file, a PGS ID string, or an existing LazyFrame.
    """
    if isinstance(scoring_file, pl.LazyFrame):
        return scoring_file
    if isinstance(scoring_file, Path):
        return parse_scoring_file(scoring_file)
    if isinstance(scoring_file, str) and scoring_file.upper().startswith("PGS"):
        return load_scoring(scoring_file, cache_dir=cache_dir, genome_build=genome_build)
    return parse_scoring_file(Path(scoring_file))


def _normalize_scoring_columns(scoring_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize scoring file columns to use harmonized position columns when available.

    Harmonized files from PGS Catalog have hm_chr and hm_pos columns that should
    be preferred over the original chr_name and chr_position.
    """
    columns = scoring_lf.collect_schema().names()

    rename_exprs: list[pl.Expr] = []

    if "hm_chr" in columns and "hm_pos" in columns:
        rename_exprs.append(
            pl.col("hm_chr").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chr_name_norm")
        )
        rename_exprs.append(pl.col("hm_pos").cast(pl.Int64).alias("chr_pos_norm"))
    elif "chr_name" in columns and "chr_position" in columns:
        rename_exprs.append(
            pl.col("chr_name").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chr_name_norm")
        )
        rename_exprs.append(pl.col("chr_position").cast(pl.Int64).alias("chr_pos_norm"))
    else:
        raise ValueError(
            f"Scoring file must have (hm_chr, hm_pos) or (chr_name, chr_position). "
            f"Found columns: {columns}"
        )

    if "effect_allele" not in columns:
        raise ValueError(f"Scoring file must have 'effect_allele' column. Found: {columns}")
    if "effect_weight" not in columns:
        raise ValueError(f"Scoring file must have 'effect_weight' column. Found: {columns}")

    rename_exprs.append(pl.col("effect_allele").cast(pl.Utf8))
    rename_exprs.append(pl.col("effect_weight").cast(pl.Float64))

    if "other_allele" in columns:
        rename_exprs.append(pl.col("other_allele").cast(pl.Utf8))

    if "allelefrequency_effect" in columns:
        rename_exprs.append(
            pl.col("allelefrequency_effect").cast(pl.Float64, strict=False)
        )

    return scoring_lf.select(rename_exprs)


def _compute_theoretical_stats(
    scoring_df: pl.DataFrame,
) -> tuple[float, float, int] | None:
    """Compute theoretical PRS mean and SD from allele frequencies in the scoring file.

    Under Hardy-Weinberg equilibrium and independent loci:
      E[dosage_i] = 2 * p_i
      Var[dosage_i] = 2 * p_i * (1 - p_i)
      E[PRS] = sum(w_i * 2 * p_i)
      Var[PRS] = sum(w_i^2 * 2 * p_i * (1 - p_i))

    Returns (mean, std, n_variants_with_freq) or None if allelefrequency_effect
    column is absent or has no valid values.
    """
    if "allelefrequency_effect" not in scoring_df.columns:
        return None

    valid = scoring_df.filter(
        pl.col("allelefrequency_effect").is_not_null()
        & pl.col("effect_weight").is_not_null()
        & (pl.col("allelefrequency_effect") > 0.0)
        & (pl.col("allelefrequency_effect") < 1.0)
    )
    if valid.height == 0:
        return None

    agg = valid.select(
        (pl.col("effect_weight") * 2.0 * pl.col("allelefrequency_effect"))
        .sum()
        .alias("mean"),
        (
            pl.col("effect_weight").pow(2)
            * 2.0
            * pl.col("allelefrequency_effect")
            * (1.0 - pl.col("allelefrequency_effect"))
        )
        .sum()
        .alias("variance"),
    )
    row = agg.row(0, named=True)
    mean = float(row["mean"])
    variance = float(row["variance"])
    std = math.sqrt(variance) if variance > 0 else 0.0
    return mean, std, valid.height


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def compute_prs(
    vcf_path: Path | str,
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    pgs_id: str = "unknown",
    trait_reported: str | None = None,
    genotypes_lf: pl.LazyFrame | None = None,
) -> PRSResult:
    """Compute a polygenic risk score for a single VCF against a scoring file.

    Algorithm:
    1. Read genotypes from VCF (chrom, pos, ref, alt, GT)
    2. Parse/load scoring file (chr_name, chr_position, effect_allele, effect_weight)
    3. Normalize chromosome names (strip 'chr' prefix)
    4. Inner join on (chrom == chr_name, pos == chr_position)
    5. Compute dosage of effect allele from GT
    6. PRS = sum(effect_weight * dosage)
    7. If allelefrequency_effect is present, compute theoretical mean/SD
       and estimate population percentile.

    Args:
        vcf_path: Path to VCF file (ignored when *genotypes_lf* is provided)
        scoring_file: Path to scoring file, PGS ID string, or pre-loaded LazyFrame
        genome_build: Genome build for downloading scoring files
        cache_dir: Cache directory for downloaded scoring files
        pgs_id: PGS ID for result labeling
        trait_reported: Trait name for result labeling
        genotypes_lf: Pre-built genotypes LazyFrame with columns
            ``chrom, pos, ref, alt, GT`` (``start`` is accepted as an
            alias for ``pos``).  When provided, *vcf_path* is not read —
            useful for passing a normalized parquet via
            ``pl.scan_parquet()``.

    Returns:
        PRSResult with computed score, match statistics, and optionally
        theoretical distribution stats and percentile.
    """
    with start_action(
        action_type="prs:compute",
        vcf_path=str(vcf_path),
        pgs_id=pgs_id,
        genome_build=genome_build,
    ):
        if genotypes_lf is None:
            genotypes_lf = read_genotypes(vcf_path)
        else:
            genotypes_lf = _normalize_genotype_columns(genotypes_lf)
        scoring_lf = _resolve_scoring(scoring_file, genome_build, cache_dir)
        scoring_norm = _normalize_scoring_columns(scoring_lf)

        scoring_df = scoring_norm.collect()
        variants_total = scoring_df.height

        joined = genotypes_lf.join(
            scoring_df.lazy(),
            left_on=["chrom", "pos"],
            right_on=["chr_name_norm", "chr_pos_norm"],
            how="inner",
        )

        joined = joined.with_columns(
            compute_dosage_expr(
                gt_col="GT",
                ref_col="ref",
                alt_col="alt",
                effect_allele_col="effect_allele",
            )
        )

        joined = joined.with_columns(
            (pl.col("effect_weight") * pl.col("dosage")).alias("weighted_dosage")
        )

        matched_df = joined.collect()
        variants_matched = len(matched_df)

        if variants_matched == 0:
            prs_score = 0.0
        else:
            prs_score = matched_df["weighted_dosage"].sum()
            if prs_score is None:
                prs_score = 0.0

        match_rate = variants_matched / variants_total if variants_total > 0 else 0.0

        has_freqs = False
        theoretical_mean: float | None = None
        theoretical_std: float | None = None
        percentile: float | None = None

        stats = _compute_theoretical_stats(scoring_df)
        if stats is not None:
            mean, std, n_with_freq = stats
            has_freqs = True
            theoretical_mean = mean
            theoretical_std = std
            if std > 0:
                z = (float(prs_score) - mean) / std
                percentile = round(_norm_cdf(z) * 100.0, 2)
            log_message(
                message_type="prs:theoretical_stats",
                pgs_id=pgs_id,
                variants_with_frequency=n_with_freq,
                variants_total=variants_total,
                theoretical_mean=mean,
                theoretical_std=std,
                percentile=percentile,
            )

        return PRSResult(
            pgs_id=pgs_id,
            score=float(prs_score),
            variants_matched=int(variants_matched),
            variants_total=int(variants_total),
            match_rate=float(match_rate),
            trait_reported=trait_reported,
            has_allele_frequencies=has_freqs,
            theoretical_mean=theoretical_mean,
            theoretical_std=theoretical_std,
            percentile=percentile,
        )


def compute_prs_batch(
    vcf_path: Path | str,
    pgs_ids: list[str],
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[PRSResult]:
    """Compute multiple PRS scores for a single VCF file.

    Args:
        vcf_path: Path to VCF file
        pgs_ids: List of PGS Catalog score IDs
        genome_build: Genome build
        cache_dir: Cache directory for downloaded scoring files

    Returns:
        List of PRSResult, one per PGS ID
    """
    from just_prs.catalog import PGSCatalogClient

    with start_action(
        action_type="prs:compute_batch",
        vcf_path=str(vcf_path),
        pgs_ids=pgs_ids,
        genome_build=genome_build,
    ):
        results: list[PRSResult] = []

        with PGSCatalogClient() as client:
            for pgs_id in pgs_ids:
                score_info = client.get_score(pgs_id)
                result = compute_prs(
                    vcf_path=vcf_path,
                    scoring_file=pgs_id,
                    genome_build=genome_build,
                    cache_dir=cache_dir,
                    pgs_id=pgs_id,
                    trait_reported=score_info.trait_reported,
                )
                results.append(result)

        return results
