"""Dagster Software-Defined Assets for the PRS reference distribution pipeline.

This pipeline computes population-level Polygenic Risk Score (PRS) distributions
using a reference panel (default: 1000 Genomes) and PGS Catalog scoring files.
The output is a per-score, per-superpopulation percentile table that lets
end-users compare their personal PRS against a global reference.

Asset lineage (left to right):

  [external]                        [download]       [compute]              [upload]
  ebi_pgs_catalog_reference_panel → reference_panel → reference_scores ──→ hf_prs_percentiles
  ebi_pgs_catalog_scoring_files  ──────────────────↗                     ↗
                                    raw_pgs_metadata → cleaned_pgs_metadata

hf_prs_percentiles enriches the raw distribution statistics with cleaned
PGS Catalog metadata (trait names, EFO terms, performance metrics) before
pushing to HuggingFace, creating a cross-pipeline dependency.

Each computed asset logs its source URL in output metadata.
SourceAssets (external group) are visualisation-only nodes that document remote
data Dagster observes but does not create.
"""

import os
from pathlib import Path

import polars as pl
from dagster import (
    AssetDep,
    AssetExecutionContext,
    AssetIn,
    Output,
    SourceAsset,
    asset,
)

from just_prs.ftp import PGS_FTP_BASE, list_all_pgs_ids
from just_prs.hf import DEFAULT_HF_PERCENTILES_REPO, push_reference_distributions
from just_prs.reference import (
    DEFAULT_PANEL,
    REFERENCE_PANELS,
    REFERENCE_PANEL_URL,
    compute_reference_prs_batch,
    download_reference_panel,
    enrich_distributions,
)
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource
from prs_pipeline.runtime import resource_tracker


# ---------------------------------------------------------------------------
# Source assets — external data that Dagster observes but does not create
# ---------------------------------------------------------------------------

ebi_pgs_catalog_reference_panel = SourceAsset(
    key="ebi_pgs_catalog_reference_panel",
    group_name="external",
    description=(
        "PGS Catalog 1000 Genomes reference panel (pgsc_1000G_v1.tar.zst, ~7 GB) "
        "hosted on the EBI PGS Catalog FTP server. "
        "Contains binary genotype files (.pgen/.pvar/.psam) for 2,504 individuals "
        "across 5 superpopulations (AFR, AMR, EAS, EUR, SAS). "
        "This is the baseline population panel against which all PRS scores are computed."
    ),
    metadata={"url": REFERENCE_PANEL_URL},
)

ebi_pgs_catalog_scoring_files = SourceAsset(
    key="ebi_pgs_catalog_scoring_files",
    group_name="external",
    description=(
        "PGS Catalog scoring files and metadata hosted on the EBI PGS Catalog FTP server "
        "(https://www.pgscatalog.org). "
        "Includes harmonised scoring files for all 5,000+ published Polygenic Scores "
        "(variant weights mapped to GRCh37/GRCh38), plus metadata CSVs describing "
        "each score's trait, development method, and validation performance."
    ),
    metadata={"url": PGS_FTP_BASE},
)

# ---------------------------------------------------------------------------
# Download asset — fetch reference panel into local cache
# ---------------------------------------------------------------------------

@asset(
    group_name="download",
    description=(
        "Downloads the reference panel from the EBI FTP server and extracts the "
        "binary genotype files (.pgen/.pvar/.psam) into the local cache. "
        "This only runs once — subsequent materializations skip if files already exist."
    ),
)
def reference_panel(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[Path]:
    """Download the reference panel tarball from the EBI PGS Catalog FTP and extract it."""
    cache_dir = cache_dir_resource.get_path()
    panel = os.environ.get("PRS_PIPELINE_PANEL", DEFAULT_PANEL)

    with resource_tracker("reference_panel", context=context):
        dest = download_reference_panel(cache_dir=cache_dir, panel=panel)

    context.log.info(f"Reference panel ready at: {dest}")
    context.add_output_metadata({
        "ref_dir": str(dest),
        "exists": dest.exists(),
        "panel": panel,
        "source_url": REFERENCE_PANELS[panel]["url"],
    })
    return Output(dest)


# ---------------------------------------------------------------------------
# Compute asset — batch score all PGS IDs and aggregate distributions
# ---------------------------------------------------------------------------

@asset(
    group_name="compute",
    ins={"ref_dir": AssetIn("reference_panel")},
    description=(
        "Scores all PGS IDs (or a test subset) against the reference panel in a "
        "single process using pgenlib + polars. Failures are logged and tracked "
        "but do not abort the batch. Produces per-sample scores, aggregated "
        "per-superpopulation distribution statistics, and a quality report. "
        "The distributions parquet is the input for hf_prs_percentiles."
    ),
)
def reference_scores(
    context: AssetExecutionContext,
    ref_dir: Path,
    cache_dir_resource: CacheDirResource,
) -> Output[pl.DataFrame]:
    """Score all PGS IDs against the reference panel and aggregate distributions."""
    import random

    cache_dir = cache_dir_resource.get_path()
    panel = os.environ.get("PRS_PIPELINE_PANEL", DEFAULT_PANEL)
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()

    all_pgs_ids = list_all_pgs_ids()

    if test_spec:
        if test_spec.startswith("random:"):
            n = int(test_spec.split(":", 1)[1])
            pgs_ids = sorted(random.sample(all_pgs_ids, min(n, len(all_pgs_ids))))
            context.log.info(
                f"TEST MODE: randomly selected {len(pgs_ids)} PGS IDs from {len(all_pgs_ids)} total."
            )
        else:
            requested = [s.strip() for s in test_spec.split(",") if s.strip()]
            valid = set(all_pgs_ids)
            pgs_ids = [pid for pid in requested if pid in valid]
            invalid = [pid for pid in requested if pid not in valid]
            if invalid:
                context.log.warning(f"TEST MODE: ignoring unknown PGS IDs: {invalid}")
            context.log.info(
                f"TEST MODE: using {len(pgs_ids)} specified PGS IDs (from {len(all_pgs_ids)} total)."
            )
    else:
        pgs_ids = all_pgs_ids

    is_test = bool(test_spec)
    output_subdir = "test" if is_test else None

    context.log.info(f"Batch scoring {len(pgs_ids)} PGS IDs against {panel} panel.")

    with resource_tracker("reference_scores", context=context):
        result = compute_reference_prs_batch(
            pgs_ids=pgs_ids,
            ref_dir=ref_dir,
            cache_dir=cache_dir,
            genome_build="GRCh38",
            panel=panel,
            skip_existing=True,
            output_subdir=output_subdir,
        )

    n_ok = sum(1 for o in result.outcomes if o.status == "ok")
    n_failed = sum(1 for o in result.outcomes if o.status == "failed")
    n_problematic = sum(1 for o in result.outcomes if o.status in ("low_match", "zero_variance"))

    scores_dir = cache_dir / "reference_scores" / panel
    percentiles_dir = cache_dir / "percentiles"
    if output_subdir:
        percentiles_dir = percentiles_dir / output_subdir
    context.log.info(
        f"Scoring complete: {n_ok} ok, {n_failed} failed, {n_problematic} problematic. "
        f"Per-PGS scores: {scores_dir}/<PGS_ID>/scores.parquet  "
        f"Distributions: {percentiles_dir / f'{panel}_distributions.parquet'}"
    )

    context.add_output_metadata({
        "panel": panel,
        "n_pgs_ids": len(pgs_ids),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_problematic": n_problematic,
        "n_distributions": result.distributions_df.height,
        "n_pgs_with_distributions": result.distributions_df["pgs_id"].n_unique() if result.distributions_df.height > 0 else 0,
        "per_pgs_scores_dir": str(scores_dir),
        "percentiles_dir": str(percentiles_dir),
        "engine": "polars",
        "test_mode": is_test,
    })
    return Output(result.distributions_df)


# ---------------------------------------------------------------------------
# Upload asset — push to HuggingFace (represents the HF dataset as an asset)
# ---------------------------------------------------------------------------

@asset(
    group_name="upload",
    ins={"distributions": AssetIn("reference_scores")},
    deps=[AssetDep("cleaned_pgs_metadata")],
    description=(
        "Enriches the raw distribution statistics with cleaned PGS Catalog metadata "
        "(trait names, EFO terms, best performance metrics like AUROC/OR/C-index, "
        "ancestry info) and uploads the enriched parquet to the HuggingFace dataset "
        "just-dna-seq/prs-percentiles. This makes the population-level PRS percentiles "
        "self-contained and publicly available. End-users of the just-prs library pull "
        "this dataset via PRSCatalog.percentile() to compare their personal scores "
        "against the reference panel. This asset represents the final published artefact."
    ),
)
def hf_prs_percentiles(
    context: AssetExecutionContext,
    distributions: pl.DataFrame,
    cache_dir_resource: CacheDirResource,
    hf_resource: HuggingFaceResource,
) -> Output[str]:
    """Enrich distributions with metadata and upload to HuggingFace."""
    repo_id = hf_resource.percentiles_repo
    token = hf_resource.get_token()
    panel = os.environ.get("PRS_PIPELINE_PANEL", DEFAULT_PANEL)
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
    is_test = bool(test_spec)

    cache_dir = cache_dir_resource.get_path()
    metadata_dir = cache_dir / "metadata"
    percentiles_dir = cache_dir / "percentiles"
    if is_test:
        percentiles_dir = percentiles_dir / "test"
    percentiles_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = percentiles_dir / f"{panel}_distributions.parquet"

    with resource_tracker("hf_prs_percentiles", context=context):
        enriched = enrich_distributions(distributions, metadata_dir)
        n_metadata_cols = len(enriched.columns) - len(distributions.columns)
        context.log.info(
            f"Enriched distributions with {n_metadata_cols} metadata columns "
            f"({distributions.height} rows, {len(enriched.columns)} total columns)."
        )

        enriched.write_parquet(parquet_path)

        if is_test:
            context.log.info(
                f"TEST MODE: skipping HuggingFace push. "
                f"Results saved locally:\n"
                f"  Distributions: {parquet_path}\n"
                f"  Quality report: {percentiles_dir / f'{panel}_quality.parquet'}"
            )
            context.add_output_metadata({
                "panel": panel,
                "test_mode": True,
                "hf_push_skipped": True,
                "distributions_path": str(parquet_path),
                "quality_path": str(percentiles_dir / f"{panel}_quality.parquet"),
                "n_rows": enriched.height,
                "n_columns": len(enriched.columns),
                "n_metadata_columns_added": n_metadata_cols,
                "n_pgs_ids": enriched["pgs_id"].n_unique() if enriched.height > 0 else 0,
            })
            return Output(str(parquet_path))

        push_reference_distributions(
            parquet_path=parquet_path,
            repo_id=repo_id,
            token=token,
            panel=panel,
        )

    url = f"https://huggingface.co/datasets/{repo_id}"
    context.log.info(f"Pushed {panel}_distributions.parquet to {url}")
    context.log.info(
        f"Results saved locally:\n"
        f"  Distributions: {parquet_path}\n"
        f"  Quality report: {percentiles_dir / f'{panel}_quality.parquet'}\n"
        f"  HuggingFace: {url}"
    )
    context.add_output_metadata({
        "repo_id": repo_id,
        "url": url,
        "panel": panel,
        "distributions_path": str(parquet_path),
        "quality_path": str(percentiles_dir / f"{panel}_quality.parquet"),
        "n_rows": enriched.height,
        "n_columns": len(enriched.columns),
        "n_metadata_columns_added": n_metadata_cols,
        "n_pgs_ids": enriched["pgs_id"].n_unique() if enriched.height > 0 else 0,
    })
    return Output(url)
