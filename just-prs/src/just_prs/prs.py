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


def compute_prs(
    vcf_path: Path | str,
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    pgs_id: str = "unknown",
    trait_reported: str | None = None,
    genotypes_lf: pl.LazyFrame | None = None,
    genotype_input_mode: str | GenotypeInputMode = GenotypeInputMode.AUTO,
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
        if resolved_mode == GenotypeInputMode.VARIANT_ONLY:
            ref_known = pl.col("reference_allele").is_not_null() & (pl.col("reference_allele").str.len_chars() > 0)
            absent = pl.col("is_present").not_()
            joined = joined.with_columns(
                pl.when(pl.col("is_present") & pl.col("is_no_call").not_())
                .then(pl.col("dosage"))
                .when(absent & ref_known & (pl.col("effect_allele") == pl.col("reference_allele")))
                .then(pl.lit(2))
                .when(absent & ref_known)
                .then(pl.lit(0))
                .otherwise(pl.lit(None, dtype=pl.Int64))
                .alias("resolved_dosage")
            )
        else:
            joined = joined.with_columns(
                pl.when(pl.col("is_no_call"))
                .then(pl.lit(None, dtype=pl.Int64))
                .otherwise(pl.col("dosage"))
                .alias("resolved_dosage")
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
            (absent_expr & ref_known_expr.not_()).cast(pl.Int64).sum().alias("variants_unscorable_absent"),
            (pl.col("is_present") & pl.col("is_no_call")).cast(pl.Int64).sum().alias("variants_no_call"),
            pl.col("weighted_dosage").sum().alias("prs_score"),
        ).collect()

        variants_observed = int(agg["variants_observed"][0] or 0)
        variants_assumed_hom_ref = int(agg["variants_assumed_hom_ref"][0] or 0)
        variants_unscorable_absent = int(agg["variants_unscorable_absent"][0] or 0)
        variants_no_call = int(agg["variants_no_call"][0] or 0)
        variants_matched = int(agg["observed_called"][0] or 0) + variants_assumed_hom_ref

        if variants_matched == 0:
            prs_score = 0.0
        else:
            prs_score = float(agg["prs_score"][0] or 0.0)

        match_rate = variants_matched / variants_total if variants_total > 0 else 0.0

        has_freqs = False
        theoretical_mean: float | None = None
        theoretical_std: float | None = None
        percentile: float | None = None
        percentile_method: str | None = None

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
            genotype_input_mode=resolved_mode.value,
            trait_reported=trait_reported,
            has_allele_frequencies=has_freqs,
            theoretical_mean=theoretical_mean,
            theoretical_std=theoretical_std,
            percentile=percentile,
            percentile_method=percentile_method,
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

        schema_names = scoring_norm.collect_schema().names()
        scoring_df = scoring_norm.collect()
        variants_total = scoring_df.height

        dosage_weight = DOSAGE_WEIGHT_COLUMNS[0] in schema_names
        weighted_sql = _DUCKDB_WEIGHTED_DOSAGE_GENOBOOST if dosage_weight else _DUCKDB_WEIGHTED_DOSAGE_ADDITIVE

        mem_limit = memory_limit or _resolve_duckdb_memory_limit()
        conn = duckdb.connect(config={"memory_limit": mem_limit})
        try:
            conn.execute("SET arrow_large_buffer_size = true")
            conn.register("scoring", scoring_df.to_arrow())

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
            if resolved_mode == GenotypeInputMode.VARIANT_ONLY:
                join_sql = f"""
                    FROM scoring s
                    LEFT JOIN {geno_from} g
                      ON g.chrom = s.chr_name_norm AND g.pos = s.chr_pos_norm
                """
                resolved_dosage_sql = _DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY
                counter_sql = f"""
                    SUM(CASE WHEN g.GT IS NOT NULL THEN 1 ELSE 0 END) AS variants_observed,
                    SUM(CASE WHEN g.GT IS NOT NULL AND NOT ({_DUCKDB_NO_CALL_SQL}) THEN 1 ELSE 0 END) AS observed_called,
                    SUM(CASE WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} THEN 1 ELSE 0 END) AS variants_assumed_hom_ref,
                    SUM(CASE WHEN g.GT IS NULL AND NOT ({_DUCKDB_REFERENCE_KNOWN_SQL}) THEN 1 ELSE 0 END) AS variants_unscorable_absent,
                    SUM(CASE WHEN g.GT IS NOT NULL AND {_DUCKDB_NO_CALL_SQL} THEN 1 ELSE 0 END) AS variants_no_call
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
                    SUM(CASE WHEN {_DUCKDB_NO_CALL_SQL} THEN 1 ELSE 0 END) AS variants_no_call
                """

            row = conn.execute(f"""
                WITH resolved AS (
                    SELECT
                        s.*,
                        g.GT,
                        {resolved_dosage_sql} AS resolved_dosage,
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
                    COALESCE(SUM(variants_no_call), 0) AS variants_no_call
                FROM resolved s
            """).fetchone()

            prs_score = float(row[0])
            observed_called = int(row[1])
            variants_observed = int(row[2])
            variants_assumed_hom_ref = int(row[3])
            variants_unscorable_absent = int(row[4])
            variants_no_call = int(row[5])
            variants_matched = observed_called + variants_assumed_hom_ref

            has_freqs = False
            theoretical_mean: float | None = None
            theoretical_std: float | None = None
            percentile: float | None = None
            percentile_method: str | None = None

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
            genotype_input_mode=resolved_mode.value,
            trait_reported=trait_reported,
            has_allele_frequencies=has_freqs,
            theoretical_mean=theoretical_mean,
            theoretical_std=theoretical_std,
            percentile=percentile,
            percentile_method=percentile_method,
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
