"""Cleanup pipeline for PGS Catalog metadata.

Pure-function pipeline that transforms raw DataFrames from ftp.download_metadata_sheet()
into cleaned LazyFrames with normalized genome builds, snake_case column names,
parsed metric strings, and only the columns needed for PRS computation and search.
"""

import re

import polars as pl

# ---------------------------------------------------------------------------
# Genome build normalization
# ---------------------------------------------------------------------------

BUILD_NORMALIZATION: dict[str, str] = {
    "GRCh37": "GRCh37",
    "hg19": "GRCh37",
    "hg37": "GRCh37",
    "GRCh38": "GRCh38",
    "hg38": "GRCh38",
    "NCBI36": "GRCh36",
    "hg18": "GRCh36",
    "NCBI35": "GRCh36",
    "NR": "NR",
}

# Columns retained by clean_scores (raw PGS Catalog names -> snake_case)
_SCORES_COLUMN_RENAME: dict[str, str] = {
    "Polygenic Score (PGS) ID": "pgs_id",
    "PGS Name": "name",
    "Reported Trait": "trait_reported",
    "Mapped Trait(s) (EFO label)": "trait_efo",
    "Mapped Trait(s) (EFO ID)": "trait_efo_id",
    "Original Genome Build": "genome_build",
    "Number of Variants": "n_variants",
    "Type of Variant Weight": "weight_type",
    "PGS Publication (PGP) ID": "pgp_id",
    "Publication (PMID)": "pmid",
    "FTP link": "ftp_link",
    "Release Date": "release_date",
}

_PERF_COLUMN_RENAME: dict[str, str] = {
    "PGS Performance Metric (PPM) ID": "ppm_id",
    "Evaluated Score": "pgs_id",
    "PGS Sample Set (PSS)": "pss_id",
    "PGS Publication (PGP) ID": "pgp_id",
    "Reported Trait": "trait_reported",
    "Covariates Included in the Model": "covariates",
    "Publication (PMID)": "pmid",
    "Publication (doi)": "doi",
    "Hazard Ratio (HR)": "hr_raw",
    "Odds Ratio (OR)": "or_raw",
    "Beta": "beta_raw",
    "Area Under the Receiver-Operating Characteristic Curve (AUROC)": "auroc_raw",
    "Concordance Statistic (C-index)": "cindex_raw",
    "Other Metric(s)": "other_raw",
}

_EVAL_COLUMN_RENAME: dict[str, str] = {
    "PGS Sample Set (PSS)": "pss_id",
    "Polygenic Score (PGS) ID": "pgs_id",
    "Number of Individuals": "n_individuals",
    "Number of Cases": "n_cases",
    "Number of Controls": "n_controls",
    "Broad Ancestry Category": "ancestry_broad",
    "Country of Recruitment": "country",
    "Cohort(s)": "cohorts",
}

# Regex for parsing metric strings: "1.55 [1.52,1.58]" or "-0.7 (0.15)" or "1.41"
_METRIC_RE = re.compile(
    r"^(?P<estimate>-?[\d.]+)"
    r"(?:\s*\[(?P<ci_lower>-?[\d.]+)\s*,\s*(?P<ci_upper>-?[\d.]+)\])?"
    r"(?:\s*\((?P<se>-?[\d.]+)\))?"
    r"$"
)


def normalize_genome_build(lf: pl.LazyFrame, col: str = "genome_build") -> pl.LazyFrame:
    """Map all genome build variants to canonical GRCh37/GRCh38/GRCh36/NR."""
    return lf.with_columns(
        pl.col(col).replace_strict(BUILD_NORMALIZATION, default=pl.col(col)).alias(col)
    )


def rename_score_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename raw PGS Catalog score columns to short snake_case names.

    Only keeps columns present in _SCORES_COLUMN_RENAME; drops everything else.
    """
    current_cols = set(lf.collect_schema().names())
    rename_map = {k: v for k, v in _SCORES_COLUMN_RENAME.items() if k in current_cols}
    return lf.select([pl.col(old).alias(new) for old, new in rename_map.items()])


def clean_scores(df: pl.DataFrame) -> pl.LazyFrame:
    """Full cleanup pipeline for the scores metadata sheet.

    Renames columns, normalizes genome builds, and selects only the fields
    useful for PRS computation and search.
    """
    lf = rename_score_columns(df.lazy())
    lf = normalize_genome_build(lf, col="genome_build")
    return lf


def parse_metric_string(value: str | None) -> dict[str, float | None]:
    """Parse a PGS Catalog metric string into estimate, ci_lower, ci_upper, se.

    Handles formats like:
        "1.55 [1.52,1.58]"  -> estimate=1.55, ci_lower=1.52, ci_upper=1.58
        "-0.7 (0.15)"       -> estimate=-0.7, se=0.15
        "1.41"              -> estimate=1.41
    """
    if value is None:
        return {"estimate": None, "ci_lower": None, "ci_upper": None, "se": None}
    m = _METRIC_RE.match(value.strip())
    if m is None:
        return {"estimate": None, "ci_lower": None, "ci_upper": None, "se": None}
    return {
        "estimate": float(m.group("estimate")),
        "ci_lower": float(m.group("ci_lower")) if m.group("ci_lower") else None,
        "ci_upper": float(m.group("ci_upper")) if m.group("ci_upper") else None,
        "se": float(m.group("se")) if m.group("se") else None,
    }


def _parse_metric_column(df: pl.DataFrame, raw_col: str, prefix: str) -> pl.DataFrame:
    """Parse a raw metric string column into structured numeric columns."""
    if raw_col not in df.columns:
        return df

    parsed = [parse_metric_string(v) for v in df[raw_col].to_list()]
    return df.with_columns(
        pl.Series(f"{prefix}_estimate", [p["estimate"] for p in parsed], dtype=pl.Float64),
        pl.Series(f"{prefix}_ci_lower", [p["ci_lower"] for p in parsed], dtype=pl.Float64),
        pl.Series(f"{prefix}_ci_upper", [p["ci_upper"] for p in parsed], dtype=pl.Float64),
        pl.Series(f"{prefix}_se", [p["se"] for p in parsed], dtype=pl.Float64),
    )


def rename_performance_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename raw PGS Catalog performance metric columns to snake_case."""
    current_cols = set(lf.collect_schema().names())
    rename_map = {k: v for k, v in _PERF_COLUMN_RENAME.items() if k in current_cols}
    keep_extra = [c for c in lf.collect_schema().names() if c not in _PERF_COLUMN_RENAME]
    selects = [pl.col(old).alias(new) for old, new in rename_map.items()]
    selects.extend([pl.col(c) for c in keep_extra])
    return lf.select(selects)


def rename_eval_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename raw PGS Catalog evaluation sample set columns to snake_case."""
    current_cols = set(lf.collect_schema().names())
    rename_map = {k: v for k, v in _EVAL_COLUMN_RENAME.items() if k in current_cols}
    keep_extra = [c for c in lf.collect_schema().names() if c not in _EVAL_COLUMN_RENAME]
    selects = [pl.col(old).alias(new) for old, new in rename_map.items()]
    selects.extend([pl.col(c) for c in keep_extra])
    return lf.select(selects)


def clean_performance_metrics(
    perf_df: pl.DataFrame,
    eval_df: pl.DataFrame,
) -> pl.LazyFrame:
    """Full cleanup pipeline for performance metrics.

    Renames columns, parses metric strings into numeric columns, joins with
    evaluation sample sets for sample size and ancestry, and produces a flat
    table with parsed metrics per performance evaluation row.

    Args:
        perf_df: Raw performance_metrics DataFrame from download_metadata_sheet
        eval_df: Raw evaluation_sample_sets DataFrame from download_metadata_sheet

    Returns:
        Cleaned LazyFrame with parsed metrics and evaluation context
    """
    perf_renamed = rename_performance_columns(perf_df.lazy()).collect()

    for raw_col, prefix in [
        ("or_raw", "or"),
        ("hr_raw", "hr"),
        ("beta_raw", "beta"),
        ("auroc_raw", "auroc"),
        ("cindex_raw", "cindex"),
    ]:
        perf_renamed = _parse_metric_column(perf_renamed, raw_col, prefix)

    eval_renamed = rename_eval_columns(eval_df.lazy())

    eval_agg = eval_renamed.group_by("pss_id").agg(
        pl.col("n_individuals").sum().alias("n_individuals"),
        pl.col("ancestry_broad").first().alias("ancestry_broad"),
    ).collect()

    result = perf_renamed.join(eval_agg, on="pss_id", how="left")

    drop_cols = [c for c in ["or_raw", "hr_raw", "beta_raw", "auroc_raw", "cindex_raw", "other_raw",
                              "PGS Performance: Other Relevant Information"] if c in result.columns]
    if drop_cols:
        result = result.drop(drop_cols)

    return result.lazy()


def best_performance_per_score(perf_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Select the single best performance row per PGS ID.

    Prefers rows with the largest evaluation sample size, with a bonus for
    European-ancestry cohorts (matching PGS Catalog convention).
    """
    df = perf_lf.collect()

    n_ind = df["n_individuals"].fill_null(0).to_list()
    ancestry = df["ancestry_broad"].fill_null("").to_list()
    scores = [
        n + (1_000_000 if "european" in a.lower() else 0)
        for n, a in zip(n_ind, ancestry)
    ]
    df = df.with_columns(pl.Series("_rank_score", scores, dtype=pl.Int64))

    best = df.sort("_rank_score", descending=True).group_by("pgs_id").first()
    return best.drop("_rank_score").lazy()
