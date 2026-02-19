"""Core PRS computation engine: variant matching, dosage computation, weighted sum."""

from pathlib import Path

import polars as pl
from eliot import start_action

from just_prs.models import PRSResult
from just_prs.scoring import DEFAULT_CACHE_DIR, load_scoring, parse_scoring_file
from just_prs.vcf import compute_dosage_expr, read_genotypes


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

    return scoring_lf.select(rename_exprs)


def compute_prs(
    vcf_path: Path | str,
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    pgs_id: str = "unknown",
    trait_reported: str | None = None,
) -> PRSResult:
    """Compute a polygenic risk score for a single VCF against a scoring file.

    Algorithm:
    1. Read genotypes from VCF (chrom, pos, ref, alt, GT)
    2. Parse/load scoring file (chr_name, chr_position, effect_allele, effect_weight)
    3. Normalize chromosome names (strip 'chr' prefix)
    4. Inner join on (chrom == chr_name, pos == chr_position)
    5. Compute dosage of effect allele from GT
    6. PRS = sum(effect_weight * dosage)

    Args:
        vcf_path: Path to VCF file
        scoring_file: Path to scoring file, PGS ID string, or pre-loaded LazyFrame
        genome_build: Genome build for downloading scoring files
        cache_dir: Cache directory for downloaded scoring files
        pgs_id: PGS ID for result labeling
        trait_reported: Trait name for result labeling

    Returns:
        PRSResult with computed score and match statistics
    """
    with start_action(
        action_type="prs:compute",
        vcf_path=str(vcf_path),
        pgs_id=pgs_id,
        genome_build=genome_build,
    ):
        genotypes_lf = read_genotypes(vcf_path)
        scoring_lf = _resolve_scoring(scoring_file, genome_build, cache_dir)
        scoring_norm = _normalize_scoring_columns(scoring_lf)

        variants_total = scoring_norm.select(pl.len()).collect().item()

        joined = genotypes_lf.join(
            scoring_norm,
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

        return PRSResult(
            pgs_id=pgs_id,
            score=float(prs_score),
            variants_matched=int(variants_matched),
            variants_total=int(variants_total),
            match_rate=float(match_rate),
            trait_reported=trait_reported,
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
