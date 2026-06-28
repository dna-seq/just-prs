"""Cleanup pipeline for PGS Catalog metadata.

Pure-function pipeline that transforms raw DataFrames from ftp.download_metadata_sheet()
into cleaned LazyFrames with normalized genome builds, snake_case column names,
parsed metric strings, and only the columns needed for PRS computation and search.
"""

import json
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
    Idempotent: if columns are already in snake_case target form, passes through unchanged.
    """
    current_cols = set(lf.collect_schema().names())
    target_cols = set(_SCORES_COLUMN_RENAME.values())
    already_renamed = target_cols & current_cols
    if already_renamed and not (set(_SCORES_COLUMN_RENAME.keys()) & current_cols):
        return lf.select([pl.col(c) for c in _SCORES_COLUMN_RENAME.values() if c in current_cols])
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
        pl.col("n_cases").sum().alias("n_cases"),
        pl.col("n_controls").sum().alias("n_controls"),
        pl.col("ancestry_broad").first().alias("ancestry_broad"),
    ).collect()

    result = perf_renamed.join(eval_agg, on="pss_id", how="left")

    drop_cols = [c for c in ["or_raw", "hr_raw", "beta_raw", "auroc_raw", "cindex_raw", "other_raw",
                              "PGS Performance: Other Relevant Information"] if c in result.columns]
    if drop_cols:
        result = result.drop(drop_cols)

    return result.lazy()


_PUBLICATIONS_COLUMN_RENAME: dict[str, str] = {
    "PGS Publication (PGP) ID": "pgp_id",
    "PGS Publication/Study (PGP) ID": "pgp_id",
    "First Author": "first_author",
    "PubMed ID (PMID)": "pmid",
    "Digital Object Identifier (DOI)": "doi",
    "digital object identifier (doi)": "doi",
    "Title": "title",
    "Author(s)": "authors",
    "Authors": "authors",
    "Journal": "journal",
    "Journal Name": "journal",
    "Publication Date": "date_publication",
}


def rename_publications_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename raw PGS Catalog publications columns to snake_case."""
    current_cols = set(lf.collect_schema().names())
    target_cols = set(_PUBLICATIONS_COLUMN_RENAME.values())
    already_renamed = target_cols & current_cols
    if already_renamed and not (set(_PUBLICATIONS_COLUMN_RENAME.keys()) & current_cols):
        return lf.select([pl.col(c) for c in _PUBLICATIONS_COLUMN_RENAME.values() if c in current_cols])
    rename_map = {k: v for k, v in _PUBLICATIONS_COLUMN_RENAME.items() if k in current_cols}
    return lf.select([pl.col(old).alias(new) for old, new in rename_map.items()])


def clean_publications(df: pl.DataFrame) -> pl.LazyFrame:
    """Full cleanup pipeline for the publications metadata sheet.

    Renames columns to snake_case and selects only the fields useful for
    linking PGS scores/performance to their source papers.
    """
    return rename_publications_columns(df.lazy())


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


# ---------------------------------------------------------------------------
# Development-sample ancestry (score_development_samples sheet)
# ---------------------------------------------------------------------------

# Raw score_development_samples columns -> snake_case (only those we use).
_DEV_SAMPLES_COLUMN_RENAME: dict[str, str] = {
    "Polygenic Score (PGS) ID": "pgs_id",
    "Stage of PGS Development": "stage_raw",
    "Number of Individuals": "n_individuals",
    "Broad Ancestry Category": "ancestry_broad",
    "Ancestry (e.g. French, Chinese)": "ancestry_free",
    "Additional Ancestry Description": "ancestry_additional",
}

# Stage labels in the raw score_development_samples sheet.
_DEV_STAGE_GWAS = "Source of Variant Associations (GWAS)"
_DEV_STAGE_TRAINING = "Score Development/Training"


def rename_dev_samples_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename raw score_development_samples columns to snake_case (keep only used)."""
    current_cols = set(lf.collect_schema().names())
    rename_map = {k: v for k, v in _DEV_SAMPLES_COLUMN_RENAME.items() if k in current_cols}
    return lf.select([pl.col(old).alias(new) for old, new in rename_map.items()])


def _split_broad_ancestries(label: str) -> list[str]:
    """Split a PGS Catalog broad-ancestry label into individual categories.

    The "Broad Ancestry Category" field can list several comma-separated
    categories for an admixed cohort (e.g. "European, East Asian").
    """
    return [part.strip() for part in label.split(",") if part.strip()]


def _dominant_ancestry_distribution(
    labels: list[str | None], weights: list[float]
) -> tuple[str | None, dict[str, float]]:
    """Sample-weighted distribution over broad-ancestry label buckets.

    Returns ``(dominant_bucket, {bucket: fraction})``. Buckets are the raw label
    strings (admixed labels are kept intact). Rows with a null/blank label are
    ignored; when every weight is non-positive (sample sizes unreported), the
    present buckets are weighted equally. Dominant ties break alphabetically for
    determinism.
    """
    bucket_w: dict[str, float] = {}
    for label, w in zip(labels, weights):
        if label is None or not label.strip():
            continue
        bucket_w[label.strip()] = bucket_w.get(label.strip(), 0.0) + max(w, 0.0)
    if not bucket_w:
        return None, {}
    total = sum(bucket_w.values())
    if total <= 0:
        bucket_w = {k: 1.0 for k in bucket_w}
        total = float(len(bucket_w))
    dist = {k: v / total for k, v in bucket_w.items()}
    dominant = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return dominant, dist


def _summarize_dev_samples(pgs_id: str, sub: pl.DataFrame) -> dict[str, object]:
    """Collapse one PGS ID's development-sample rows into a single summary dict."""
    stages = sub["stage_raw"].to_list()
    labels = sub["ancestry_broad"].to_list()
    n_list = [float(n) if n is not None else 0.0 for n in sub["n_individuals"].to_list()]

    dominant, dist = _dominant_ancestry_distribution(labels, n_list)

    anc_set: set[str] = set()
    for label in labels:
        if label:
            anc_set.update(_split_broad_ancestries(label))
    ancestries = sorted(anc_set)

    def _stage_n(stage: str) -> int:
        return int(sum(n for n, s in zip(n_list, stages) if s == stage))

    def _stage_dominant(stage: str) -> str | None:
        s_labels = [lbl for lbl, s in zip(labels, stages) if s == stage]
        s_w = [n for n, s in zip(n_list, stages) if s == stage]
        return _dominant_ancestry_distribution(s_labels, s_w)[0]

    gwas_n = _stage_n(_DEV_STAGE_GWAS)
    train_n = _stage_n(_DEV_STAGE_TRAINING)

    dist_json = (
        json.dumps({k: round(v, 4) for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))})
        if dist
        else None
    )

    return {
        "pgs_id": pgs_id,
        "dev_ancestry_broad": dominant,
        "dev_ancestries": ancestries,
        "dev_n_ancestries": len(ancestries),
        "dev_is_multi_ancestry": len(ancestries) > 1,
        "dev_ancestry_distribution": dist_json,
        "dev_sample_size": max(gwas_n, train_n),
        "dev_gwas_sample_size": gwas_n,
        "dev_training_sample_size": train_n,
        "gwas_ancestry_broad": _stage_dominant(_DEV_STAGE_GWAS),
        "training_ancestry_broad": _stage_dominant(_DEV_STAGE_TRAINING),
    }


_DEV_ANCESTRY_SCHEMA: dict[str, pl.DataType] = {
    "pgs_id": pl.Utf8,
    "dev_ancestry_broad": pl.Utf8,
    "dev_ancestries": pl.List(pl.Utf8),
    "dev_n_ancestries": pl.Int64,
    "dev_is_multi_ancestry": pl.Boolean,
    "dev_ancestry_distribution": pl.Utf8,
    "dev_sample_size": pl.Int64,
    "dev_gwas_sample_size": pl.Int64,
    "dev_training_sample_size": pl.Int64,
    "gwas_ancestry_broad": pl.Utf8,
    "training_ancestry_broad": pl.Utf8,
}


def clean_score_development_samples(df: pl.DataFrame) -> pl.LazyFrame:
    """Aggregate the score_development_samples sheet to one row per PGS ID.

    Surfaces the *development* (training) ancestry — distinct from the
    *evaluation* ancestry that ``clean_performance_metrics`` already exposes —
    so callers can judge score x sample x panel ancestry coherence (F19). The
    raw sheet carries up to ~75 rows per PGS spanning two stages — "Source of
    Variant Associations (GWAS)" (discovery) and "Score Development/Training"
    (tuning) — each optionally broken down by ancestry. They collapse into the
    columns described by :data:`_DEV_ANCESTRY_SCHEMA`:

    - ``dev_ancestry_broad`` — dominant broad ancestry, sample-size weighted
      across all stages (the headline for a quick coherence check)
    - ``dev_ancestries`` / ``dev_n_ancestries`` / ``dev_is_multi_ancestry`` —
      distinct broad ancestries present (admixed labels split), cardinality, flag
    - ``dev_ancestry_distribution`` — JSON ``{broad_ancestry: fraction}``
      (sample-weighted, descending) — the structured input a coherence veto reads
    - ``dev_sample_size`` — development cohort size (the larger of the GWAS and
      training stage totals, avoiding cross-stage double counting)
    - ``dev_gwas_sample_size`` / ``dev_training_sample_size`` — per-stage totals
    - ``gwas_ancestry_broad`` / ``training_ancestry_broad`` — dominant per stage
    """
    renamed = rename_dev_samples_columns(df.lazy()).collect()
    for col, dtype in (("stage_raw", pl.Utf8), ("n_individuals", pl.Float64), ("ancestry_broad", pl.Utf8)):
        if col not in renamed.columns:
            renamed = renamed.with_columns(pl.lit(None, dtype=dtype).alias(col))
    if "pgs_id" not in renamed.columns or renamed.height == 0:
        return pl.DataFrame(schema=_DEV_ANCESTRY_SCHEMA).lazy()

    records: list[dict[str, object]] = []
    for key, sub in renamed.group_by("pgs_id", maintain_order=True):
        pid = key[0] if isinstance(key, tuple) else key
        records.append(_summarize_dev_samples(str(pid), sub))

    return pl.DataFrame(records, schema=_DEV_ANCESTRY_SCHEMA).lazy()
