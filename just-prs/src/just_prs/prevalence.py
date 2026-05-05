"""Prevalence data sourcing, caching, and consolidation.

Merges hand-curated seed prevalence data with automated GWAS Catalog cohort
fractions and PGS Catalog evaluation sample set fractions into a single
per-EFO-trait prevalence table.

Three tiers:
  1. Seed CSV (highest quality, ~50 traits) — hand-curated from WHO/CDC/literature
  2. GWAS Catalog cohort fractions (automated, broad) — biased but available
  3. PGS eval cohort fractions (last resort) — from best_performance n_cases/n_controls
"""

import logging
from pathlib import Path

import polars as pl
from eliot import start_action

from just_prs.hf import (
    DEFAULT_HF_CATALOG_REPO,
    HF_DATA_PREFIX,
    _configure_hf_timeouts,
    _resolve_token,
)
from just_prs.ontology import (
    ONTOLOGY_ALIAS_COLUMNS,
    build_efo_alias_map,
    colon_trait_id,
    enrich_with_trait_aliases,
    normalize_trait_id,
)

logger = logging.getLogger(__name__)

_SEED_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "trait_prevalence_seed.csv"

_PREVALENCE_SCHEMA = {
    "efo_id": pl.Utf8,
    "trait_label": pl.Utf8,
    "prevalence": pl.Float64,
    "prevalence_lower": pl.Float64,
    "prevalence_upper": pl.Float64,
    "prevalence_type": pl.Utf8,
    "sex": pl.Utf8,
    "ancestry": pl.Utf8,
    "age_range": pl.Utf8,
    "source": pl.Utf8,
    "source_detail": pl.Utf8,
    "xref_mondo": pl.Utf8,
    "xref_icd10": pl.Utf8,
    "confidence": pl.Utf8,
    **ONTOLOGY_ALIAS_COLUMNS,
}


def load_seed_prevalence(path: Path | None = None) -> pl.LazyFrame:
    """Load the hand-curated seed prevalence CSV.

    Args:
        path: Path to the seed CSV. Defaults to the bundled data file.

    Returns:
        LazyFrame with columns matching the prevalence table schema.
    """
    csv_path = path or _SEED_CSV_PATH
    if not csv_path.exists():
        logger.warning("Seed prevalence CSV not found at %s", csv_path)
        return pl.LazyFrame(schema=_PREVALENCE_SCHEMA)

    df = pl.read_csv(csv_path, null_values=["", "NA"])
    for col_name, dtype in _PREVALENCE_SCHEMA.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

    df = df.with_columns(pl.lit("high").alias("confidence"))
    return df.select(list(_PREVALENCE_SCHEMA.keys())).lazy()


def _gwas_cohort_prevalence(gwas_summary_df: pl.DataFrame) -> pl.LazyFrame:
    """Convert GWAS trait summary to prevalence table rows.

    Uses the case fraction from the largest GWAS study per EFO trait as a
    biased but available prevalence proxy.
    """
    if gwas_summary_df.height == 0 or "case_fraction" not in gwas_summary_df.columns:
        return pl.LazyFrame(schema=_PREVALENCE_SCHEMA)

    result = gwas_summary_df.select(
        pl.col("efo_id"),
        pl.col("efo_label").alias("trait_label"),
        pl.col("case_fraction").alias("prevalence"),
        pl.lit(None, dtype=pl.Float64).alias("prevalence_lower"),
        pl.lit(None, dtype=pl.Float64).alias("prevalence_upper"),
        pl.lit("cohort").alias("prevalence_type"),
        pl.lit(None, dtype=pl.Utf8).alias("sex"),
        pl.lit(None, dtype=pl.Utf8).alias("ancestry"),
        pl.lit(None, dtype=pl.Utf8).alias("age_range"),
        pl.lit("gwas_catalog_cohort").alias("source"),
        pl.when(pl.col("study_accession").is_not_null())
        .then(pl.lit("GWAS Catalog study ") + pl.col("study_accession"))
        .otherwise(pl.lit("GWAS Catalog"))
        .alias("source_detail"),
        pl.lit(None, dtype=pl.Utf8).alias("xref_mondo"),
        pl.lit(None, dtype=pl.Utf8).alias("xref_icd10"),
        pl.lit("low").alias("confidence"),
        pl.col("efo_id").alias("canonical_efo_id"),
        pl.lit(None, dtype=pl.Utf8).alias("mapped_from_id"),
        pl.lit("direct").alias("mapping_source"),
    )
    return result.lazy()


def _pgs_eval_cohort_prevalence(best_performance_lf: pl.LazyFrame, scores_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Derive cohort case fractions from PGS Catalog evaluation sample sets.

    Uses n_cases / (n_cases + n_controls) from best_performance joined with
    scores to get EFO trait IDs.
    """
    bp = best_performance_lf.select(
        "pgs_id", "n_cases", "n_controls"
    ).filter(
        pl.col("n_cases").is_not_null() & pl.col("n_controls").is_not_null()
        & (pl.col("n_cases") > 0) & (pl.col("n_controls") > 0)
    ).collect()

    if bp.height == 0:
        return pl.LazyFrame(schema=_PREVALENCE_SCHEMA)

    bp = bp.with_columns(
        (pl.col("n_cases").cast(pl.Float64) / (pl.col("n_cases") + pl.col("n_controls")).cast(pl.Float64))
        .alias("case_fraction")
    )

    scores = scores_lf.select("pgs_id", "trait_efo_id", "trait_efo").collect()
    joined = bp.join(scores, on="pgs_id", how="inner")

    if joined.height == 0:
        return pl.LazyFrame(schema=_PREVALENCE_SCHEMA)

    per_efo = (
        joined
        .sort("case_fraction", descending=False)
        .group_by("trait_efo_id")
        .first()
    )

    result = per_efo.select(
        pl.col("trait_efo_id").alias("efo_id"),
        pl.col("trait_efo").alias("trait_label"),
        pl.col("case_fraction").alias("prevalence"),
        pl.lit(None, dtype=pl.Float64).alias("prevalence_lower"),
        pl.lit(None, dtype=pl.Float64).alias("prevalence_upper"),
        pl.lit("cohort").alias("prevalence_type"),
        pl.lit(None, dtype=pl.Utf8).alias("sex"),
        pl.lit(None, dtype=pl.Utf8).alias("ancestry"),
        pl.lit(None, dtype=pl.Utf8).alias("age_range"),
        pl.lit("pgs_eval_cohort").alias("source"),
        pl.lit("PGS Catalog evaluation sample sets").alias("source_detail"),
        pl.lit(None, dtype=pl.Utf8).alias("xref_mondo"),
        pl.lit(None, dtype=pl.Utf8).alias("xref_icd10"),
        pl.lit("low").alias("confidence"),
        pl.col("trait_efo_id").alias("canonical_efo_id"),
        pl.lit(None, dtype=pl.Utf8).alias("mapped_from_id"),
        pl.lit("direct").alias("mapping_source"),
    )
    return result.lazy()


def query_ols_xrefs(efo_id: str) -> dict[str, str | None]:
    """Query the EBI OLS4 API for cross-references of an EFO term.

    Returns a dict with optional mondo_id, icd10_code, snomed_id keys.
    """
    result: dict[str, str | None] = {"mondo_id": None, "icd10_code": None, "snomed_id": None}
    normalized = normalize_trait_id(efo_id) or efo_id
    aliases = build_efo_alias_map([normalized], allow_network=True).get(normalized, [])
    for alias in aliases:
        curie = colon_trait_id(alias)
        if curie.startswith("MONDO:") and result["mondo_id"] is None:
            result["mondo_id"] = curie
        if curie.startswith("ICD10:") and result["icd10_code"] is None:
            result["icd10_code"] = curie.split(":", 1)[1]
        if curie.startswith("SNOMEDCT:") and result["snomed_id"] is None:
            result["snomed_id"] = curie.split(":", 1)[1]
    return result


def build_efo_xrefs(efo_ids: list[str], cache_dir: Path | None = None) -> pl.DataFrame:
    """Batch query OLS for cross-references of a list of EFO IDs.

    Results are cached per EFO ID in cache_dir to avoid re-querying.

    Args:
        efo_ids: List of EFO IDs to query.
        cache_dir: Directory for per-ID JSON caches. If None, no caching.

    Returns:
        DataFrame with columns: efo_id, xref_mondo, xref_icd10, xref_snomed.
    """
    rows: list[dict[str, str | None]] = []
    alias_map = build_efo_alias_map(efo_ids, cache_dir=cache_dir, allow_network=True)
    for efo_id in efo_ids:
        aliases = alias_map.get(efo_id, [])

        rows.append({
            "efo_id": efo_id,
            "xref_mondo": next((colon_trait_id(a) for a in aliases if a.startswith("MONDO_")), None),
            "xref_icd10": None,
            "xref_snomed": None,
        })

    return pl.DataFrame(rows)


def build_prevalence_table(
    scores_lf: pl.LazyFrame,
    best_performance_lf: pl.LazyFrame,
    gwas_summary_df: pl.DataFrame | None = None,
    seed_path: Path | None = None,
    xref_cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Merge all prevalence tiers into a single table.

    Priority: Seed (Tier 1) > GWAS cohort (Tier 2) > PGS eval cohort (Tier 3).
    For each EFO ID, the highest-confidence row wins.

    Args:
        scores_lf: Cleaned scores LazyFrame (for EFO trait IDs).
        best_performance_lf: Best performance LazyFrame (for n_cases/n_controls).
        gwas_summary_df: GWAS trait summary DataFrame. None to skip Tier 2.
        seed_path: Path to seed CSV. None for default bundled file.

    Returns:
        Combined prevalence DataFrame.
    """
    with start_action(action_type="prevalence:build_table"):
        seed_lf = load_seed_prevalence(seed_path)
        tiers: list[pl.LazyFrame] = [seed_lf]

        if gwas_summary_df is not None and gwas_summary_df.height > 0:
            gwas_lf = _gwas_cohort_prevalence(gwas_summary_df)
            tiers.append(gwas_lf)

        pgs_lf = _pgs_eval_cohort_prevalence(best_performance_lf, scores_lf)
        tiers.append(pgs_lf)

        all_rows = pl.concat(tiers, how="vertical_relaxed").collect()

        if all_rows.height == 0:
            return pl.DataFrame(schema=_PREVALENCE_SCHEMA)

        confidence_order = {"high": 0, "moderate": 1, "low": 2}
        all_rows = all_rows.with_columns(
            pl.col("confidence")
            .replace_strict(confidence_order, default=3)
            .alias("_conf_rank")
        )
        result = (
            all_rows
            .sort("_conf_rank")
            .group_by("efo_id")
            .first()
            .drop("_conf_rank")
        )

        enriched = enrich_with_trait_aliases(
            result,
            cache_dir=xref_cache_dir,
            allow_network=xref_cache_dir is not None,
        )
        enriched = enriched.with_columns(
            pl.col("confidence")
            .replace_strict(confidence_order, default=3)
            .alias("_conf_rank")
        )
        return (
            enriched
            .sort(["_conf_rank", "mapping_source"])
            .group_by("efo_id")
            .first()
            .drop("_conf_rank")
        )


def pull_prevalence_from_hf(
    local_dir: Path,
    repo_id: str = DEFAULT_HF_CATALOG_REPO,
    token: str | None = None,
) -> Path | None:
    """Download trait_prevalence.parquet from the HF catalog repo.

    Args:
        local_dir: Directory to save the downloaded file.
        repo_id: HuggingFace dataset repository ID.
        token: HF API token.

    Returns:
        Path to the downloaded file, or None if not available.
    """
    import shutil

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    resolved_token = _resolve_token(token)
    hf_path = f"{HF_DATA_PREFIX}/metadata/trait_prevalence.parquet"

    with start_action(action_type="prevalence:pull_from_hf", repo_id=repo_id):
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                repo_type="dataset",
                local_dir=local_dir,
                token=resolved_token,
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            logger.debug("trait_prevalence.parquet not found on HF (%s)", repo_id)
            return None

        target = local_dir / "trait_prevalence.parquet"
        hf_cached = Path(path)
        if hf_cached != target:
            shutil.copy2(hf_cached, target)
        return target


def push_prevalence_to_hf(
    parquet_path: Path,
    repo_id: str = DEFAULT_HF_CATALOG_REPO,
    token: str | None = None,
) -> None:
    """Upload trait_prevalence.parquet to the HF catalog repo.

    Args:
        parquet_path: Local path to the prevalence parquet file.
        repo_id: HuggingFace dataset repository ID.
        token: HF API token.
    """
    from huggingface_hub import HfApi

    resolved_token = _resolve_token(token)
    with start_action(action_type="prevalence:push_to_hf", repo_id=repo_id):
        _configure_hf_timeouts()
        api = HfApi(token=resolved_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"{HF_DATA_PREFIX}/metadata/trait_prevalence.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
