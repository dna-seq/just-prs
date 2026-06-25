"""Core PRS computation engine: variant matching, dosage computation, weighted sum."""

from __future__ import annotations

import enum
import math
from pathlib import Path

import duckdb
import polars as pl
from eliot import log_message, start_action


class PRSEngine(str, enum.Enum):
    """PRS computation engine selection."""
    POLARS = "polars"
    DUCKDB = "duckdb"


class GenotypeInputMode(str, enum.Enum):
    """How absent scoring loci should be interpreted."""
    AUTO = "auto"
    VARIANT_ONLY = "variant_only"
    ALL_SITES = "all_sites"
    PLINK_PRESENT_ONLY = "plink_present_only"

from just_prs.chip_coverage import Chip, chip_typed_positions
from just_prs.models import PRSResult
from just_prs.scoring import DEFAULT_CACHE_DIR, load_scoring, parse_scoring_file
from just_prs.vcf import compute_dosage_expr, read_genotypes

# Restoration scope: which absent scoring positions may be hom-ref filled.
#   False  -> off (no filling)
#   True   -> the whole reference-allele universe (WGS; the universe index IS the set)
#   Chip   -> chip-typed positions (array; eligible = chip set ∩ universe)
#   Path / pl.DataFrame -> a custom (chrom,pos) set (embedder escape hatch)
RestorationScope = bool | Chip | Path | pl.DataFrame


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


DOSAGE_WEIGHT_COLUMNS = ("dosage_0_weight", "dosage_1_weight", "dosage_2_weight")


def is_dosage_weight_format(columns: list[str]) -> bool:
    """Check if a scoring file uses per-dosage-level weights (GenoBoost format)."""
    return all(c in columns for c in DOSAGE_WEIGHT_COLUMNS) and "effect_weight" not in columns


def _normalize_scoring_columns(scoring_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize scoring file columns to use harmonized position columns when available.

    Harmonized files from PGS Catalog have hm_chr and hm_pos columns that should
    be preferred over the original chr_name and chr_position.

    Supports two weight formats:
    - Standard additive: ``effect_weight`` column
    - Per-dosage (GenoBoost): ``dosage_0_weight``, ``dosage_1_weight``,
      ``dosage_2_weight`` columns (non-linear scoring model)
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

    dosage_weight = is_dosage_weight_format(columns)

    if dosage_weight:
        for col in DOSAGE_WEIGHT_COLUMNS:
            rename_exprs.append(pl.col(col).cast(pl.Float64))
    elif "effect_weight" in columns:
        rename_exprs.append(pl.col("effect_weight").cast(pl.Float64))
    else:
        raise ValueError(
            f"Scoring file must have 'effect_weight' or dosage weight columns "
            f"(dosage_0_weight, dosage_1_weight, dosage_2_weight). Found: {columns}"
        )

    rename_exprs.append(pl.col("effect_allele").cast(pl.Utf8))

    if "other_allele" in columns:
        rename_exprs.append(pl.col("other_allele").cast(pl.Utf8))

    if "reference_allele" in columns:
        rename_exprs.append(pl.col("reference_allele").cast(pl.Utf8))
    else:
        rename_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("reference_allele"))

    if "allelefrequency_effect" in columns:
        rename_exprs.append(
            pl.col("allelefrequency_effect").cast(pl.Float64, strict=False)
        )

    return scoring_lf.select(rename_exprs)


def _normalize_genotype_input_mode(mode: str | GenotypeInputMode) -> GenotypeInputMode:
    """Validate and normalize genotype input mode values."""
    if isinstance(mode, GenotypeInputMode):
        return mode
    try:
        return GenotypeInputMode(str(mode))
    except ValueError as exc:
        valid = ", ".join(m.value for m in GenotypeInputMode)
        raise ValueError(f"Unknown genotype_input_mode {mode!r}; expected one of: {valid}") from exc


def _infer_genotype_input_mode(genotypes_lf: pl.LazyFrame) -> GenotypeInputMode:
    """Best-effort mode detection from normalized genotype rows.

    Most uploaded VCFs are variant-only. gVCF/all-sites inputs commonly contain
    reference blocks (e.g. ``<NON_REF>`` alleles or ``RefCall`` filters), so we
    detect those conservatively and otherwise default to variant-only semantics.
    """
    cols = genotypes_lf.collect_schema().names()
    sample_exprs: list[pl.Expr] = []
    if "alt" in cols:
        sample_exprs.append(pl.col("alt").cast(pl.Utf8).str.contains("NON_REF", literal=True).any().alias("has_non_ref"))
    if "filter" in cols:
        sample_exprs.append(pl.col("filter").cast(pl.Utf8).str.contains("RefCall", literal=True).any().alias("has_refcall"))
    if not sample_exprs:
        return GenotypeInputMode.VARIANT_ONLY
    sample = genotypes_lf.select(sample_exprs).limit(1).collect()
    if any(bool(sample[col][0]) for col in sample.columns):
        return GenotypeInputMode.ALL_SITES
    return GenotypeInputMode.VARIANT_ONLY


def _resolve_genotype_input_mode(
    mode: str | GenotypeInputMode,
    genotypes_lf: pl.LazyFrame,
) -> GenotypeInputMode:
    """Resolve ``auto`` into an executable genotype input mode."""
    normalized = _normalize_genotype_input_mode(mode)
    if normalized == GenotypeInputMode.AUTO:
        return _infer_genotype_input_mode(genotypes_lf)
    return normalized


def _gt_no_call_expr(gt_col: str = "GT") -> pl.Expr:
    """Return an expression identifying missing/no-call diploid genotypes."""
    gt = pl.col(gt_col).cast(pl.Utf8)
    normalized = gt.str.replace_all(r"\|", "/")
    parts = normalized.str.split("/")
    a0 = parts.list.get(0, null_on_oob=True)
    a1 = parts.list.get(1, null_on_oob=True)
    return gt.is_null() | (gt == ".") | a0.is_null() | a1.is_null() | (a0 == ".") | (a1 == ".")


def _compute_theoretical_stats(
    scoring_lf: pl.LazyFrame,
    schema_names: list[str] | None = None,
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
    if schema_names is None:
        schema_names = scoring_lf.collect_schema().names()

    if "allelefrequency_effect" not in schema_names:
        return None
    if "effect_weight" not in schema_names:
        return None

    agg = (
        scoring_lf.filter(
            pl.col("allelefrequency_effect").is_not_null()
            & pl.col("effect_weight").is_not_null()
            & (pl.col("allelefrequency_effect") > 0.0)
            & (pl.col("allelefrequency_effect") < 1.0)
        )
        .select(
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
            pl.len().alias("n_valid"),
        )
        .collect()
    )

    n_valid = int(agg["n_valid"][0])
    if n_valid == 0:
        return None

    mean = float(agg["mean"][0])
    variance = float(agg["variance"][0])
    std = math.sqrt(variance) if variance > 0 else 0.0
    return mean, std, n_valid


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _normalize_restoration_scope(
    scope: RestorationScope | None,
    cache_dir: Path,
    genome_build: str,
) -> bool | pl.LazyFrame:
    """Resolve a user-facing scope into ``True`` (whole universe), ``False`` (off),
    or a ``(chrom, pos)`` LazyFrame (restricted set).

    - ``False``/``None`` -> ``False`` (no restoration).
    - ``True`` -> ``True`` (the universe itself is the eligible set; no duplication).
    - ``Chip`` -> the chip's typed positions for ``genome_build`` (GSA ships both
      A2/GRCh38 and A1/GRCh37 manifests); if the chip has no manifest for that build
      it degrades to ``False`` + a log, never silently mis-fills.
    - ``Path`` / ``pl.DataFrame`` -> a custom set (accepts ``chrom`` or ``chr_norm``).
    """
    if scope is False or scope is None:
        return False
    if scope is True:
        return True
    if isinstance(scope, Chip):
        try:
            positions = chip_typed_positions(scope, cache_dir, build=genome_build)
        except (NotImplementedError, ValueError) as exc:
            log_message(
                message_type="prs:restoration_scope_unavailable",
                chip=str(scope),
                genome_build=genome_build,
                reason=str(exc),
            )
            return False
        return positions.lazy().pipe(_select_chrom_pos)
    if isinstance(scope, pl.DataFrame):
        return scope.lazy().pipe(_select_chrom_pos)
    if isinstance(scope, (str, Path)):
        return pl.scan_parquet(scope).pipe(_select_chrom_pos)
    raise ValueError(f"Unsupported restoration scope: {scope!r}")


def _select_chrom_pos(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize a position-set frame to unique ``(chrom, pos)`` (accepts chr_norm)."""
    cols = lf.collect_schema().names()
    chrom_col = "chrom" if "chrom" in cols else ("chr_norm" if "chr_norm" in cols else None)
    if chrom_col is None or "pos" not in cols:
        raise ValueError(f"Position set must have (chrom|chr_norm, pos); got {cols}")
    return lf.select(
        pl.col(chrom_col).cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chrom"),
        pl.col("pos").cast(pl.Int64),
    ).unique()


def _apply_reference_resolution(
    scoring_norm: pl.LazyFrame,
    universe_path: Path | str | None,
    scope: bool | pl.LazyFrame,
) -> pl.LazyFrame:
    """Fill a missing ``reference_allele`` from the precomputed REF-universe parquet,
    restricted to the eligible ``scope``.

    ``scope`` is the *normalized* form from ``_normalize_restoration_scope``:
    ``False`` -> no fill; ``True`` -> any universe position eligible (WGS); a
    ``(chrom,pos)`` LazyFrame -> only positions in that set ∩ universe (array/chip).
    Adds a ``ref_resolved_source`` column (``panel``/``fasta``/null) for accounting;
    the column is always present so downstream aggregation can reference it. Only
    positions whose ``reference_allele`` was null/empty are filled; existing values win.
    """
    schema = scoring_norm.collect_schema().names()
    if scope is False or universe_path is None:
        if "ref_resolved_source" not in schema:
            scoring_norm = scoring_norm.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("ref_resolved_source")
            )
        return scoring_norm

    universe = (
        pl.scan_parquet(universe_path)
        .filter(pl.col("ref").is_not_null())
        .select(
            pl.col("chrom").cast(pl.Utf8),
            pl.col("pos").cast(pl.Int64),
            pl.col("ref").cast(pl.Utf8),
            pl.col("ref_source").cast(pl.Utf8),
        )
    )
    if isinstance(scope, pl.LazyFrame):
        # Restrict the eligible REF set to the scope (chip ∩ universe).
        universe = universe.join(scope, on=["chrom", "pos"], how="semi")
    universe = universe.select(
        pl.col("chrom").alias("_u_chrom"),
        pl.col("pos").alias("_u_pos"),
        pl.col("ref").alias("_u_ref"),
        pl.col("ref_source").alias("_u_src"),
    )

    ref_unknown = pl.col("reference_allele").is_null() | (
        pl.col("reference_allele").str.len_chars() == 0
    )
    fill_mask = ref_unknown & pl.col("_u_ref").is_not_null()
    return (
        scoring_norm.join(
            universe,
            left_on=["chr_name_norm", "chr_pos_norm"],
            right_on=["_u_chrom", "_u_pos"],
            how="left",
        )
        .with_columns(
            pl.when(fill_mask)
            .then(pl.col("_u_ref"))
            .otherwise(pl.col("reference_allele"))
            .alias("reference_allele"),
            pl.when(fill_mask)
            .then(pl.col("_u_src"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("ref_resolved_source"),
        )
        .drop("_u_ref", "_u_src")
    )


def compute_prs(
    vcf_path: Path | str,
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    pgs_id: str = "unknown",
    trait_reported: str | None = None,
    genotypes_lf: pl.LazyFrame | None = None,
    genotype_input_mode: str | GenotypeInputMode = GenotypeInputMode.AUTO,
    maf_fill: bool = False,
    reference_restoration: RestorationScope = False,
    reference_universe_path: Path | str | None = None,
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
        genotype_input_mode: How absent scoring loci are interpreted:
            ``auto`` (default), ``variant_only``, ``all_sites``, or
            ``plink_present_only``.
        maf_fill: When True and the scoring file has ``allelefrequency_effect``,
            substitute ``dosage = 2 * MAF`` for absent variants that would
            otherwise be unscorable. Tracked as ``variants_maf_filled``.

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

        schema_names = scoring_norm.collect_schema().names()
        variants_total = scoring_norm.select(pl.len()).collect().item()
        resolved_mode = _resolve_genotype_input_mode(genotype_input_mode, genotypes_lf)

        # Reference-allele resolution only affects the variant-only absent-locus
        # path; in other modes absent loci are never imputed hom-ref.
        restoration_scope = (
            _normalize_restoration_scope(reference_restoration, cache_dir, genome_build)
            if resolved_mode == GenotypeInputMode.VARIANT_ONLY
            else False
        )
        scoring_norm = _apply_reference_resolution(
            scoring_norm, reference_universe_path, restoration_scope
        )

        if resolved_mode == GenotypeInputMode.VARIANT_ONLY:
            joined = scoring_norm.join(
                genotypes_lf,
                left_on=["chr_name_norm", "chr_pos_norm"],
                right_on=["chrom", "pos"],
                how="left",
            )
        else:
            joined = genotypes_lf.join(
                scoring_norm,
                left_on=["chrom", "pos"],
                right_on=["chr_name_norm", "chr_pos_norm"],
                how="inner",
            )

        joined = joined.with_columns(
            pl.col("GT").is_not_null().alias("is_present"),
            _gt_no_call_expr().alias("is_no_call"),
            compute_dosage_expr(
                gt_col="GT",
                ref_col="ref",
                alt_col="alt",
                effect_allele_col="effect_allele",
            )
        )

        dosage_weight = DOSAGE_WEIGHT_COLUMNS[0] in schema_names
        has_maf_col = "allelefrequency_effect" in schema_names
        do_maf_fill = maf_fill and has_maf_col and not dosage_weight

        # Per-variant weight mass for C_wt (weight-mass coverage). Standard additive
        # scores use |effect_weight|; per-dosage (GenoBoost) scores have no single beta,
        # so use the largest absolute per-dosage weight as the mass surrogate.
        if dosage_weight:
            variant_mass_expr = pl.max_horizontal(
                pl.col("dosage_0_weight").abs(),
                pl.col("dosage_1_weight").abs(),
                pl.col("dosage_2_weight").abs(),
            )
        else:
            variant_mass_expr = pl.col("effect_weight").abs()
        weight_mass_total = float(
            scoring_norm.select(variant_mass_expr.sum().alias("m")).collect().item() or 0.0
        )

        if resolved_mode == GenotypeInputMode.VARIANT_ONLY:
            ref_known = pl.col("reference_allele").is_not_null() & (pl.col("reference_allele").str.len_chars() > 0)
            absent = pl.col("is_present").not_()

            dosage_chain = (
                pl.when(pl.col("is_present") & pl.col("is_no_call").not_())
                .then(pl.col("dosage"))
                .when(absent & ref_known & (pl.col("effect_allele") == pl.col("reference_allele")))
                .then(pl.lit(2))
                .when(absent & ref_known)
                .then(pl.lit(0))
            )
            if do_maf_fill:
                maf_available = pl.col("allelefrequency_effect").is_not_null() & (pl.col("allelefrequency_effect") > 0.0) & (pl.col("allelefrequency_effect") < 1.0)
                dosage_chain = dosage_chain.when(absent & ref_known.not_() & maf_available).then(
                    (2.0 * pl.col("allelefrequency_effect")).cast(pl.Float64)
                )
            dosage_chain = dosage_chain.otherwise(pl.lit(None, dtype=pl.Float64))

            joined = joined.with_columns(dosage_chain.alias("resolved_dosage"))

            if do_maf_fill:
                joined = joined.with_columns(
                    (absent & ref_known.not_() & pl.col("allelefrequency_effect").is_not_null() & (pl.col("allelefrequency_effect") > 0.0) & (pl.col("allelefrequency_effect") < 1.0) & pl.col("resolved_dosage").is_not_null())
                    .alias("is_maf_filled")
                )
            else:
                joined = joined.with_columns(pl.lit(False).alias("is_maf_filled"))
        else:
            joined = joined.with_columns(
                pl.when(pl.col("is_no_call"))
                .then(pl.lit(None, dtype=pl.Int64))
                .otherwise(pl.col("dosage"))
                .alias("resolved_dosage"),
                pl.lit(False).alias("is_maf_filled"),
            )

        if dosage_weight:
            joined = joined.with_columns(
                pl.when(pl.col("resolved_dosage") == 0)
                .then(pl.col("dosage_0_weight"))
                .when(pl.col("resolved_dosage") == 1)
                .then(pl.col("dosage_1_weight"))
                .when(pl.col("resolved_dosage") == 2)
                .then(pl.col("dosage_2_weight"))
                .otherwise(pl.lit(0.0))
                .alias("weighted_dosage")
            )
        else:
            joined = joined.with_columns(
                (pl.col("effect_weight") * pl.col("resolved_dosage").fill_null(0)).alias("weighted_dosage")
            )

        absent_expr = pl.col("is_present").not_()
        ref_known_expr = pl.col("reference_allele").is_not_null() & (pl.col("reference_allele").str.len_chars() > 0)
        agg = joined.select(
            pl.col("is_present").cast(pl.Int64).sum().alias("variants_observed"),
            (pl.col("is_present") & pl.col("is_no_call").not_()).cast(pl.Int64).sum().alias("observed_called"),
            (absent_expr & ref_known_expr).cast(pl.Int64).sum().alias("variants_assumed_hom_ref"),
            (absent_expr & ref_known_expr.not_() & pl.col("is_maf_filled").not_()).cast(pl.Int64).sum().alias("variants_unscorable_absent"),
            (pl.col("is_present") & pl.col("is_no_call")).cast(pl.Int64).sum().alias("variants_no_call"),
            pl.col("is_maf_filled").cast(pl.Int64).sum().alias("variants_maf_filled"),
            (absent_expr & (pl.col("ref_resolved_source") == "panel")).cast(pl.Int64).sum().alias("variants_ref_resolved_panel"),
            (absent_expr & (pl.col("ref_resolved_source") == "fasta")).cast(pl.Int64).sum().alias("variants_ref_resolved_fasta"),
            pl.col("weighted_dosage").sum().alias("prs_score"),
            (
                pl.when(pl.col("resolved_dosage").is_not_null())
                .then(variant_mass_expr)
                .otherwise(0.0)
            ).sum().alias("weight_mass_matched"),
        ).collect()

        variants_observed = int(agg["variants_observed"][0] or 0)
        variants_assumed_hom_ref = int(agg["variants_assumed_hom_ref"][0] or 0)
        variants_unscorable_absent = int(agg["variants_unscorable_absent"][0] or 0)
        variants_no_call = int(agg["variants_no_call"][0] or 0)
        variants_maf_filled = int(agg["variants_maf_filled"][0] or 0)
        variants_ref_resolved_panel = int(agg["variants_ref_resolved_panel"][0] or 0)
        variants_ref_resolved_fasta = int(agg["variants_ref_resolved_fasta"][0] or 0)
        variants_matched = int(agg["observed_called"][0] or 0) + variants_assumed_hom_ref + variants_maf_filled

        if variants_matched == 0:
            prs_score = 0.0
        else:
            prs_score = float(agg["prs_score"][0] or 0.0)

        match_rate = variants_matched / variants_total if variants_total > 0 else 0.0
        weight_mass_matched = float(agg["weight_mass_matched"][0] or 0.0)
        weight_mass_coverage = (
            weight_mass_matched / weight_mass_total if weight_mass_total > 0 else None
        )

        has_freqs = False
        theoretical_mean: float | None = None
        theoretical_std: float | None = None
        percentile: float | None = None
        percentile_method: str | None = None
        z_score: float | None = None
        reference_mean: float | None = None
        reference_std: float | None = None

        stats = _compute_theoretical_stats(scoring_norm, schema_names)
        if stats is not None:
            mean, std, n_with_freq = stats
            has_freqs = True
            theoretical_mean = mean
            theoretical_std = std
            if std > 0:
                z = (prs_score - mean) / std
                percentile = round(_norm_cdf(z) * 100.0, 2)
                percentile_method = "theoretical"
                z_score = z
                reference_mean = mean
                reference_std = std
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
            score=prs_score,
            variants_matched=variants_matched,
            variants_total=int(variants_total),
            match_rate=float(match_rate),
            variants_observed=variants_observed,
            variants_assumed_hom_ref=variants_assumed_hom_ref,
            variants_unscorable_absent=variants_unscorable_absent,
            variants_no_call=variants_no_call,
            variants_maf_filled=variants_maf_filled,
            variants_ref_resolved_panel=variants_ref_resolved_panel,
            variants_ref_resolved_fasta=variants_ref_resolved_fasta,
            weight_mass_matched=weight_mass_matched,
            weight_mass_total=weight_mass_total,
            weight_mass_coverage=weight_mass_coverage,
            genotype_input_mode=resolved_mode.value,
            trait_reported=trait_reported,
            has_allele_frequencies=has_freqs,
            theoretical_mean=theoretical_mean,
            theoretical_std=theoretical_std,
            percentile=percentile,
            percentile_method=percentile_method,
            z_score=z_score,
            reference_mean=reference_mean,
            reference_std=reference_std,
        )


_DUCKDB_DOSAGE_SQL = """\
CASE
    WHEN g.GT IS NULL OR g.GT = './.' OR g.GT = '.' THEN 0
    WHEN s.effect_allele = g.alt THEN
        (CASE WHEN split_part(replace(replace(g.GT, '|', '/'), './', '0/'), '/', 1) = '1' THEN 1 ELSE 0 END
       + CASE WHEN split_part(replace(replace(g.GT, '|', '/'), './', '0/'), '/', 2) = '1' THEN 1 ELSE 0 END)
    WHEN s.effect_allele = g.ref THEN
        (CASE WHEN split_part(replace(replace(g.GT, '|', '/'), './', '0/'), '/', 1) = '0' THEN 1 ELSE 0 END
       + CASE WHEN split_part(replace(replace(g.GT, '|', '/'), './', '0/'), '/', 2) = '0' THEN 1 ELSE 0 END)
    ELSE 0
END"""

_DUCKDB_NO_CALL_SQL = """\
(g.GT IS NULL OR g.GT = './.' OR g.GT = '.'
 OR split_part(replace(g.GT, '|', '/'), '/', 1) = '.'
 OR split_part(replace(g.GT, '|', '/'), '/', 2) = '.')"""

_DUCKDB_REFERENCE_KNOWN_SQL = """\
(s.reference_allele IS NOT NULL AND length(s.reference_allele) > 0)"""

_DUCKDB_RESOLVED_DOSAGE_PRESENT_ONLY = f"""\
CASE
    WHEN {_DUCKDB_NO_CALL_SQL} THEN NULL
    ELSE ({_DUCKDB_DOSAGE_SQL})
END"""

_DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY = f"""\
CASE
    WHEN g.GT IS NOT NULL AND NOT ({_DUCKDB_NO_CALL_SQL}) THEN ({_DUCKDB_DOSAGE_SQL})
    WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} AND s.effect_allele = s.reference_allele THEN 2
    WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} THEN 0
    ELSE NULL
END"""

_DUCKDB_MAF_AVAILABLE_SQL = """\
(s.allelefrequency_effect IS NOT NULL AND s.allelefrequency_effect > 0.0 AND s.allelefrequency_effect < 1.0)"""

_DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY_MAF = f"""\
CASE
    WHEN g.GT IS NOT NULL AND NOT ({_DUCKDB_NO_CALL_SQL}) THEN ({_DUCKDB_DOSAGE_SQL})
    WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} AND s.effect_allele = s.reference_allele THEN 2
    WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} THEN 0
    WHEN g.GT IS NULL AND NOT ({_DUCKDB_REFERENCE_KNOWN_SQL}) AND {_DUCKDB_MAF_AVAILABLE_SQL} THEN 2.0 * s.allelefrequency_effect
    ELSE NULL
END"""

_DUCKDB_WEIGHTED_DOSAGE_ADDITIVE = f"""\
s.effect_weight * resolved_dosage"""

_DUCKDB_WEIGHTED_DOSAGE_GENOBOOST = """\
CASE resolved_dosage
    WHEN 0 THEN s.dosage_0_weight
    WHEN 1 THEN s.dosage_1_weight
    WHEN 2 THEN s.dosage_2_weight
    ELSE 0.0
END"""


_DEFAULT_DUCKDB_MEMORY_PERCENT = 75


def _resolve_duckdb_memory_limit() -> str:
    """Compute DuckDB per-connection memory limit.

    Resolution order:
      1. ``PRS_DUCKDB_MEMORY_LIMIT`` env var (e.g. ``"8GB"``) — used as-is.
      2. ``PRS_DUCKDB_MEMORY_PERCENT`` env var — percentage of total RAM.
      3. Default: 75% of total RAM.
    """
    import os

    import psutil

    explicit = os.environ.get("PRS_DUCKDB_MEMORY_LIMIT", "").strip()
    if explicit:
        return explicit

    total_bytes = psutil.virtual_memory().total
    pct_str = os.environ.get("PRS_DUCKDB_MEMORY_PERCENT", "").strip()
    pct = int(pct_str) if pct_str else _DEFAULT_DUCKDB_MEMORY_PERCENT
    limit_bytes = int(total_bytes * pct / 100)
    limit_gb = max(limit_bytes / (1024**3), 1.0)
    return f"{limit_gb:.1f}GB"


def compute_prs_duckdb(
    vcf_path: Path | str,
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    pgs_id: str = "unknown",
    trait_reported: str | None = None,
    genotypes_parquet: Path | str | None = None,
    genotypes_lf: pl.LazyFrame | None = None,
    memory_limit: str | None = None,
    genotype_input_mode: str | GenotypeInputMode = GenotypeInputMode.AUTO,
    maf_fill: bool = False,
    reference_restoration: RestorationScope = False,
    reference_universe_path: Path | str | None = None,
) -> PRSResult:
    """Compute a polygenic risk score using DuckDB for the join and aggregation.

    Functionally equivalent to ``compute_prs()`` but uses DuckDB SQL instead of
    polars for the variant-matching join and weighted-sum aggregation. DuckDB can
    spill to disk under memory pressure, making this more robust for large scoring
    files on low-memory machines.

    Either *genotypes_parquet* (preferred — DuckDB reads the file directly) or
    *genotypes_lf* (materialized to an Arrow table and registered with DuckDB)
    must be provided. If neither is given, the VCF is read via polars-bio and
    materialized to a temporary Arrow table.

    Args:
        vcf_path: Path to VCF file (used only when neither genotypes arg is provided)
        scoring_file: Path to scoring file, PGS ID string, or pre-loaded LazyFrame
        genome_build: Genome build for downloading scoring files
        cache_dir: Cache directory for downloaded scoring files
        pgs_id: PGS ID for result labeling
        trait_reported: Trait name for result labeling
        genotypes_parquet: Path to normalized genotypes parquet (best for DuckDB)
        genotypes_lf: Pre-built genotypes LazyFrame (collected to Arrow for DuckDB)
        memory_limit: DuckDB memory limit (e.g. ``"2GB"``). Falls back to
            ``PRS_DUCKDB_MEMORY_LIMIT`` / ``PRS_DUCKDB_MEMORY_PERCENT`` env vars,
            then 75% of total RAM.
        genotype_input_mode: How absent scoring loci are interpreted:
            ``auto`` (default), ``variant_only``, ``all_sites``, or
            ``plink_present_only``.
        maf_fill: When True and the scoring file has ``allelefrequency_effect``,
            substitute ``dosage = 2 * MAF`` for absent variants that would
            otherwise be unscorable. Tracked as ``variants_maf_filled``.

    Returns:
        PRSResult with computed score, match statistics, and optionally
        theoretical distribution stats and percentile.
    """
    with start_action(
        action_type="prs:compute_duckdb",
        vcf_path=str(vcf_path),
        pgs_id=pgs_id,
        genome_build=genome_build,
    ):
        scoring_lf = _resolve_scoring(scoring_file, genome_build, cache_dir)
        scoring_norm = _normalize_scoring_columns(scoring_lf)
        # Fill any missing reference_allele from the precomputed REF universe. Applied
        # unconditionally (adds a null ref_resolved_source when off) so the SQL can
        # reference s.ref_resolved_source; only the variant-only branch consumes it.
        restoration_scope = _normalize_restoration_scope(
            reference_restoration, cache_dir, genome_build
        )
        scoring_norm = _apply_reference_resolution(
            scoring_norm, reference_universe_path, restoration_scope
        )

        schema_names = scoring_norm.collect_schema().names()
        scoring_df = scoring_norm.collect()
        variants_total = scoring_df.height

        dosage_weight = DOSAGE_WEIGHT_COLUMNS[0] in schema_names
        weighted_sql = _DUCKDB_WEIGHTED_DOSAGE_GENOBOOST if dosage_weight else _DUCKDB_WEIGHTED_DOSAGE_ADDITIVE
        # Per-variant weight mass for C_wt: |effect_weight| (additive) or the largest
        # absolute per-dosage weight (GenoBoost). ``variant_mass_sql`` is aliased ``s.``
        # for the join CTE; ``mass_total_sql`` runs against the bare ``scoring`` table.
        if dosage_weight:
            variant_mass_sql = "greatest(abs(s.dosage_0_weight), abs(s.dosage_1_weight), abs(s.dosage_2_weight))"
            mass_total_sql = "greatest(abs(dosage_0_weight), abs(dosage_1_weight), abs(dosage_2_weight))"
        else:
            variant_mass_sql = "abs(s.effect_weight)"
            mass_total_sql = "abs(effect_weight)"

        mem_limit = memory_limit or _resolve_duckdb_memory_limit()
        conn = duckdb.connect(config={"memory_limit": mem_limit})
        try:
            conn.execute("SET arrow_large_buffer_size = true")
            conn.register("scoring", scoring_df.to_arrow())

            weight_mass_total = float(
                conn.execute(
                    f"SELECT COALESCE(SUM({mass_total_sql}), 0.0) FROM scoring"
                ).fetchone()[0]
            )

            if genotypes_parquet is not None:
                geno_from = f"read_parquet('{genotypes_parquet}')"
                geno_mode_lf = _normalize_genotype_columns(pl.scan_parquet(genotypes_parquet))
            elif genotypes_lf is not None:
                geno_lf = _normalize_genotype_columns(genotypes_lf)
                conn.register("genotypes_tbl", geno_lf.collect().to_arrow())
                geno_from = "genotypes_tbl"
                geno_mode_lf = geno_lf
            else:
                geno_lf = read_genotypes(vcf_path)
                conn.register("genotypes_tbl", geno_lf.collect().to_arrow())
                geno_from = "genotypes_tbl"
                geno_mode_lf = geno_lf

            resolved_mode = _resolve_genotype_input_mode(genotype_input_mode, geno_mode_lf)
            has_maf_col_ddb = "allelefrequency_effect" in schema_names
            do_maf_fill_ddb = maf_fill and has_maf_col_ddb and not dosage_weight

            if resolved_mode == GenotypeInputMode.VARIANT_ONLY:
                join_sql = f"""
                    FROM scoring s
                    LEFT JOIN {geno_from} g
                      ON g.chrom = s.chr_name_norm AND g.pos = s.chr_pos_norm
                """
                if do_maf_fill_ddb:
                    resolved_dosage_sql = _DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY_MAF
                    counter_sql = f"""
                        SUM(CASE WHEN g.GT IS NOT NULL THEN 1 ELSE 0 END) AS variants_observed,
                        SUM(CASE WHEN g.GT IS NOT NULL AND NOT ({_DUCKDB_NO_CALL_SQL}) THEN 1 ELSE 0 END) AS observed_called,
                        SUM(CASE WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} THEN 1 ELSE 0 END) AS variants_assumed_hom_ref,
                        SUM(CASE WHEN g.GT IS NULL AND NOT ({_DUCKDB_REFERENCE_KNOWN_SQL}) AND NOT ({_DUCKDB_MAF_AVAILABLE_SQL}) THEN 1 ELSE 0 END) AS variants_unscorable_absent,
                        SUM(CASE WHEN g.GT IS NOT NULL AND {_DUCKDB_NO_CALL_SQL} THEN 1 ELSE 0 END) AS variants_no_call,
                        SUM(CASE WHEN g.GT IS NULL AND NOT ({_DUCKDB_REFERENCE_KNOWN_SQL}) AND {_DUCKDB_MAF_AVAILABLE_SQL} THEN 1 ELSE 0 END) AS variants_maf_filled,
                        SUM(CASE WHEN g.GT IS NULL AND s.ref_resolved_source = 'panel' THEN 1 ELSE 0 END) AS variants_ref_resolved_panel,
                        SUM(CASE WHEN g.GT IS NULL AND s.ref_resolved_source = 'fasta' THEN 1 ELSE 0 END) AS variants_ref_resolved_fasta
                    """
                else:
                    resolved_dosage_sql = _DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY
                    counter_sql = f"""
                        SUM(CASE WHEN g.GT IS NOT NULL THEN 1 ELSE 0 END) AS variants_observed,
                        SUM(CASE WHEN g.GT IS NOT NULL AND NOT ({_DUCKDB_NO_CALL_SQL}) THEN 1 ELSE 0 END) AS observed_called,
                        SUM(CASE WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} THEN 1 ELSE 0 END) AS variants_assumed_hom_ref,
                        SUM(CASE WHEN g.GT IS NULL AND NOT ({_DUCKDB_REFERENCE_KNOWN_SQL}) THEN 1 ELSE 0 END) AS variants_unscorable_absent,
                        SUM(CASE WHEN g.GT IS NOT NULL AND {_DUCKDB_NO_CALL_SQL} THEN 1 ELSE 0 END) AS variants_no_call,
                        0 AS variants_maf_filled,
                        SUM(CASE WHEN g.GT IS NULL AND s.ref_resolved_source = 'panel' THEN 1 ELSE 0 END) AS variants_ref_resolved_panel,
                        SUM(CASE WHEN g.GT IS NULL AND s.ref_resolved_source = 'fasta' THEN 1 ELSE 0 END) AS variants_ref_resolved_fasta
                    """
            else:
                join_sql = f"""
                    FROM {geno_from} g
                    JOIN scoring s
                      ON g.chrom = s.chr_name_norm AND g.pos = s.chr_pos_norm
                """
                resolved_dosage_sql = _DUCKDB_RESOLVED_DOSAGE_PRESENT_ONLY
                counter_sql = f"""
                    COUNT(*) AS variants_observed,
                    SUM(CASE WHEN NOT ({_DUCKDB_NO_CALL_SQL}) THEN 1 ELSE 0 END) AS observed_called,
                    0 AS variants_assumed_hom_ref,
                    0 AS variants_unscorable_absent,
                    SUM(CASE WHEN {_DUCKDB_NO_CALL_SQL} THEN 1 ELSE 0 END) AS variants_no_call,
                    0 AS variants_maf_filled,
                    0 AS variants_ref_resolved_panel,
                    0 AS variants_ref_resolved_fasta
                """

            row = conn.execute(f"""
                WITH resolved AS (
                    SELECT
                        s.*,
                        g.GT,
                        {resolved_dosage_sql} AS resolved_dosage,
                        {variant_mass_sql} AS variant_mass,
                        {counter_sql}
                    {join_sql}
                    GROUP BY ALL
                )
                SELECT
                    COALESCE(SUM({weighted_sql}), 0.0) AS prs_score,
                    COALESCE(SUM(observed_called), 0) AS observed_called,
                    COALESCE(SUM(variants_observed), 0) AS variants_observed,
                    COALESCE(SUM(variants_assumed_hom_ref), 0) AS variants_assumed_hom_ref,
                    COALESCE(SUM(variants_unscorable_absent), 0) AS variants_unscorable_absent,
                    COALESCE(SUM(variants_no_call), 0) AS variants_no_call,
                    COALESCE(SUM(variants_maf_filled), 0) AS variants_maf_filled,
                    COALESCE(SUM(variants_ref_resolved_panel), 0) AS variants_ref_resolved_panel,
                    COALESCE(SUM(variants_ref_resolved_fasta), 0) AS variants_ref_resolved_fasta,
                    COALESCE(SUM(CASE WHEN resolved_dosage IS NOT NULL THEN variant_mass ELSE 0.0 END), 0.0) AS weight_mass_matched
                FROM resolved s
            """).fetchone()

            prs_score = float(row[0])
            observed_called = int(row[1])
            variants_observed = int(row[2])
            variants_assumed_hom_ref = int(row[3])
            variants_unscorable_absent = int(row[4])
            variants_no_call = int(row[5])
            variants_maf_filled = int(row[6])
            variants_ref_resolved_panel = int(row[7])
            variants_ref_resolved_fasta = int(row[8])
            weight_mass_matched = float(row[9] or 0.0)
            variants_matched = observed_called + variants_assumed_hom_ref + variants_maf_filled

            has_freqs = False
            theoretical_mean: float | None = None
            theoretical_std: float | None = None
            percentile: float | None = None
            percentile_method: str | None = None
            z_score: float | None = None
            reference_mean: float | None = None
            reference_std: float | None = None

            if "allelefrequency_effect" in schema_names and "effect_weight" in schema_names:
                stats_row = conn.execute("""
                    SELECT
                        SUM(effect_weight * 2.0 * allelefrequency_effect) AS mean,
                        SUM(POWER(effect_weight, 2) * 2.0
                            * allelefrequency_effect * (1.0 - allelefrequency_effect)) AS variance,
                        COUNT(*) AS n_valid
                    FROM scoring
                    WHERE allelefrequency_effect IS NOT NULL
                      AND effect_weight IS NOT NULL
                      AND allelefrequency_effect > 0.0
                      AND allelefrequency_effect < 1.0
                """).fetchone()

                n_valid = int(stats_row[2])
                if n_valid > 0:
                    mean = float(stats_row[0])
                    variance = float(stats_row[1])
                    std = math.sqrt(variance) if variance > 0 else 0.0
                    has_freqs = True
                    theoretical_mean = mean
                    theoretical_std = std
                    if std > 0:
                        z = (prs_score - mean) / std
                        percentile = round(_norm_cdf(z) * 100.0, 2)
                        percentile_method = "theoretical"
                        z_score = z
                        reference_mean = mean
                        reference_std = std
                    log_message(
                        message_type="prs:theoretical_stats",
                        pgs_id=pgs_id,
                        variants_with_frequency=n_valid,
                        variants_total=variants_total,
                        theoretical_mean=mean,
                        theoretical_std=std,
                        percentile=percentile,
                    )
        finally:
            conn.close()

        match_rate = variants_matched / variants_total if variants_total > 0 else 0.0
        weight_mass_coverage = (
            weight_mass_matched / weight_mass_total if weight_mass_total > 0 else None
        )

        return PRSResult(
            pgs_id=pgs_id,
            score=prs_score,
            variants_matched=variants_matched,
            variants_total=variants_total,
            match_rate=match_rate,
            variants_observed=variants_observed,
            variants_assumed_hom_ref=variants_assumed_hom_ref,
            variants_unscorable_absent=variants_unscorable_absent,
            variants_no_call=variants_no_call,
            variants_maf_filled=variants_maf_filled,
            variants_ref_resolved_panel=variants_ref_resolved_panel,
            variants_ref_resolved_fasta=variants_ref_resolved_fasta,
            weight_mass_matched=weight_mass_matched,
            weight_mass_total=weight_mass_total,
            weight_mass_coverage=weight_mass_coverage,
            genotype_input_mode=resolved_mode.value,
            trait_reported=trait_reported,
            has_allele_frequencies=has_freqs,
            theoretical_mean=theoretical_mean,
            theoretical_std=theoretical_std,
            percentile=percentile,
            percentile_method=percentile_method,
            z_score=z_score,
            reference_mean=reference_mean,
            reference_std=reference_std,
        )


_CORRUPT_PARQUET_MARKERS = (
    "out of specification",
    "invalid thrift",
    "metadata size",
    "footer",
    "not a parquet",
)


def _is_corrupt_parquet_error(exc: BaseException) -> bool:
    """Check if an exception looks like a corrupt parquet read."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _CORRUPT_PARQUET_MARKERS)


def _remove_scoring_parquet_cache(
    pgs_id: str, cache_dir: Path, genome_build: str,
) -> bool:
    """Delete a corrupt scoring parquet cache file. Returns True if deleted."""
    from just_prs.scoring import scoring_parquet_path

    parquet = scoring_parquet_path(pgs_id, cache_dir, genome_build)
    if parquet.exists():
        try:
            parquet.unlink()
            return True
        except OSError:
            pass
    return False


def compute_prs_batch(
    vcf_path: Path | str,
    pgs_ids: list[str],
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    genotype_input_mode: str | GenotypeInputMode = GenotypeInputMode.AUTO,
    engine: PRSEngine | str = PRSEngine.DUCKDB,
    genotypes_lf: pl.LazyFrame | None = None,
    memory_limit: str | None = None,
    reference_restoration: RestorationScope = False,
    reference_universe_path: Path | str | None = None,
) -> "PRSBatchResult":
    """Compute multiple PRS scores for a single VCF file.

    Memory-safe: uses DuckDB engine by default (spill-to-disk), runs
    ``gc.collect()`` after each score, continues on per-score errors
    instead of crashing, and auto-retries once on corrupt parquet caches.

    Args:
        vcf_path: Path to VCF file
        pgs_ids: List of PGS Catalog score IDs
        genome_build: Genome build
        cache_dir: Cache directory for downloaded scoring files
        genotype_input_mode: How absent scoring loci are interpreted.
        engine: Computation engine — DUCKDB (default, spill-to-disk)
            or POLARS (in-memory).
        genotypes_lf: Pre-built genotypes LazyFrame. When provided,
            vcf_path is not re-read on each iteration.
        memory_limit: DuckDB per-connection memory limit (e.g. "4GB").
            Only used when engine is DUCKDB. Defaults to env var or
            75% of RAM.

    Returns:
        PRSBatchResult with successful results and per-ID outcome tracking.
    """
    import gc

    from just_prs.catalog import PGSCatalogClient
    from just_prs.models import PRSBatchOutcome, PRSBatchResult

    if isinstance(engine, str):
        engine = PRSEngine(engine)

    with start_action(
        action_type="prs:compute_batch",
        vcf_path=str(vcf_path),
        pgs_ids=pgs_ids,
        genome_build=genome_build,
        engine=engine.value,
    ):
        results: list[PRSResult] = []
        outcomes: list[PRSBatchOutcome] = []
        failed_ids: list[str] = []

        with PGSCatalogClient() as client:
            for pgs_id in pgs_ids:
                attempts = 1
                try:
                    score_info = client.get_score(pgs_id)
                    trait = score_info.trait_reported

                    if engine == PRSEngine.DUCKDB:
                        result = compute_prs_duckdb(
                            vcf_path=vcf_path,
                            scoring_file=pgs_id,
                            genome_build=genome_build,
                            cache_dir=cache_dir,
                            pgs_id=pgs_id,
                            trait_reported=trait,
                            genotypes_parquet=str(vcf_path) if (not genotypes_lf and str(vcf_path).endswith(".parquet")) else None,
                            genotypes_lf=genotypes_lf,
                            memory_limit=memory_limit,
                            genotype_input_mode=genotype_input_mode,
                            reference_restoration=reference_restoration,
                            reference_universe_path=reference_universe_path,
                        )
                    else:
                        result = compute_prs(
                            vcf_path=vcf_path,
                            scoring_file=pgs_id,
                            genome_build=genome_build,
                            cache_dir=cache_dir,
                            pgs_id=pgs_id,
                            trait_reported=trait,
                            genotypes_lf=genotypes_lf,
                            genotype_input_mode=genotype_input_mode,
                            reference_restoration=reference_restoration,
                            reference_universe_path=reference_universe_path,
                        )

                    results.append(result)
                    outcomes.append(PRSBatchOutcome(
                        pgs_id=pgs_id, status="ok", attempts=attempts,
                    ))

                except Exception as exc:
                    if _is_corrupt_parquet_error(exc):
                        removed = _remove_scoring_parquet_cache(
                            pgs_id, cache_dir, genome_build,
                        )
                        if removed:
                            attempts = 2
                            try:
                                log_message(
                                    message_type="prs:batch_cache_repair",
                                    pgs_id=pgs_id,
                                )
                                score_info = client.get_score(pgs_id)
                                trait = score_info.trait_reported
                                if engine == PRSEngine.DUCKDB:
                                    result = compute_prs_duckdb(
                                        vcf_path=vcf_path,
                                        scoring_file=pgs_id,
                                        genome_build=genome_build,
                                        cache_dir=cache_dir,
                                        pgs_id=pgs_id,
                                        trait_reported=trait,
                                        genotypes_parquet=str(vcf_path) if (not genotypes_lf and str(vcf_path).endswith(".parquet")) else None,
                                        genotypes_lf=genotypes_lf,
                                        memory_limit=memory_limit,
                                        genotype_input_mode=genotype_input_mode,
                                    )
                                else:
                                    result = compute_prs(
                                        vcf_path=vcf_path,
                                        scoring_file=pgs_id,
                                        genome_build=genome_build,
                                        cache_dir=cache_dir,
                                        pgs_id=pgs_id,
                                        trait_reported=trait,
                                        genotypes_lf=genotypes_lf,
                                        genotype_input_mode=genotype_input_mode,
                                    )
                                results.append(result)
                                outcomes.append(PRSBatchOutcome(
                                    pgs_id=pgs_id, status="cache_repaired",
                                    attempts=attempts,
                                ))
                                gc.collect()
                                continue
                            except Exception as retry_exc:
                                log_message(
                                    message_type="prs:batch_retry_failed",
                                    pgs_id=pgs_id,
                                    error=str(retry_exc),
                                )
                                failed_ids.append(pgs_id)
                                outcomes.append(PRSBatchOutcome(
                                    pgs_id=pgs_id, status="failed",
                                    error=str(retry_exc), attempts=attempts,
                                ))
                                gc.collect()
                                continue

                    log_message(
                        message_type="prs:batch_score_failed",
                        pgs_id=pgs_id,
                        error=str(exc),
                    )
                    failed_ids.append(pgs_id)
                    outcomes.append(PRSBatchOutcome(
                        pgs_id=pgs_id, status="failed",
                        error=str(exc), attempts=attempts,
                    ))

                gc.collect()

        return PRSBatchResult(
            results=results,
            outcomes=outcomes,
            n_total=len(pgs_ids),
            n_ok=len(results),
            n_failed=len(failed_ids),
            failed_ids=failed_ids,
        )
