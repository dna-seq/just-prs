"""Dagster Definitions for the PRS reference panel and metadata pipelines."""

import dagster as dg

from prs_pipeline.assets import (
    ebi_reference_panel_fingerprint,
    ebi_pgs_catalog_reference_panel,
    ebi_pgs_catalog_scoring_files,
    ebi_scoring_files_fingerprint,
    hf_prs_percentiles,
    reference_panel,
    reference_scores,
    scoring_files,
    scoring_files_parquet,
)
from prs_pipeline.metadata_assets import (
    cleaned_pgs_metadata,
    hf_pgs_catalog,
    hf_polygenic_risk_scores,
    raw_pgs_metadata,
)
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource
from prs_pipeline.sensors import make_all_sensors
from prs_pipeline.utils import resource_summary_hook

download_reference_data = dg.define_asset_job(
    name="download_reference_data",
    selection=["ebi_reference_panel_fingerprint", "reference_panel"],
    description="Download the reference panel from the EBI FTP server.",
    hooks={resource_summary_hook},
)

score_and_push = dg.define_asset_job(
    name="score_and_push",
    selection=[
        "ebi_scoring_files_fingerprint", "scoring_files",
        "scoring_files_parquet", "reference_scores",
        "raw_pgs_metadata", "cleaned_pgs_metadata",
        "hf_prs_percentiles",
    ],
    description=(
        "Score all PGS IDs against the reference panel in a single batch, "
        "download and clean metadata, enrich distributions, and push to HuggingFace."
    ),
    hooks={resource_summary_hook},
)

full_pipeline = dg.define_asset_job(
    name="full_pipeline",
    selection=[
        "ebi_reference_panel_fingerprint", "ebi_scoring_files_fingerprint",
        "reference_panel", "scoring_files", "scoring_files_parquet",
        "reference_scores",
        "raw_pgs_metadata", "cleaned_pgs_metadata",
        "hf_prs_percentiles",
    ],
    description=(
        "Full pipeline: download reference panel, batch-score all PGS IDs, "
        "download and clean PGS Catalog metadata, enrich distributions with "
        "metadata, and push to HuggingFace. Triggered automatically by "
        "run_pipeline_on_startup sensor when assets are unmaterialized."
    ),
    hooks={resource_summary_hook},
)

catalog_pipeline = dg.define_asset_job(
    name="catalog_pipeline",
    selection=[
        "ebi_scoring_files_fingerprint",
        "scoring_files", "scoring_files_parquet",
        "raw_pgs_metadata", "cleaned_pgs_metadata",
        "hf_pgs_catalog",
    ],
    description=(
        "Build and push the combined PGS Catalog dataset to HuggingFace "
        "(just-dna-seq/pgs-catalog): download scoring files, convert to parquet, "
        "download and clean metadata, then upload both to HF."
    ),
    hooks={resource_summary_hook},
)

metadata_pipeline = dg.define_asset_job(
    name="metadata_pipeline",
    selection=[
        "ebi_scoring_files_fingerprint",
        "raw_pgs_metadata", "cleaned_pgs_metadata", "hf_polygenic_risk_scores",
    ],
    description=(
        "End-to-end metadata pipeline: download raw PGS Catalog sheets from EBI FTP, "
        "run cleanup pipeline, push cleaned parquets to HuggingFace."
    ),
    hooks={resource_summary_hook},
)

clean_and_push_metadata = dg.define_asset_job(
    name="clean_and_push_metadata",
    selection=["cleaned_pgs_metadata", "hf_polygenic_risk_scores"],
    description=(
        "Re-run cleanup pipeline on already-cached raw metadata and push to HuggingFace. "
        "Use this when raw sheets are already present in the cache."
    ),
    hooks={resource_summary_hook},
)

_assets = [
    ebi_pgs_catalog_reference_panel,
    ebi_pgs_catalog_scoring_files,
    ebi_reference_panel_fingerprint,
    ebi_scoring_files_fingerprint,
    scoring_files,
    scoring_files_parquet,
    reference_panel,
    reference_scores,
    hf_prs_percentiles,
    raw_pgs_metadata,
    cleaned_pgs_metadata,
    hf_polygenic_risk_scores,
    hf_pgs_catalog,
]
_resources = {
    "cache_dir_resource": CacheDirResource(),
    "hf_resource": HuggingFaceResource(),
}
_unresolved_jobs = [
    download_reference_data,
    score_and_push,
    full_pipeline,
    catalog_pipeline,
    metadata_pipeline,
    clean_and_push_metadata,
]


def _build_definitions() -> dg.Definitions:
    """Resolve unresolved asset jobs and build the final Definitions.

    The temporary Definitions used for resolution is a local variable so
    Dagster's module scanner only finds one Definitions object (the returned one).
    """
    tmp = dg.Definitions(assets=_assets, resources=_resources)
    asset_graph = tmp.resolve_asset_graph()
    resolved_jobs = [
        uj.resolve(asset_graph=asset_graph, resource_defs=_resources)
        for uj in _unresolved_jobs
    ]
    jobs_by_name = {j.name: j for j in resolved_jobs}

    return dg.Definitions(
        assets=_assets,
        sensors=make_all_sensors(
            full_pipeline_job=jobs_by_name["full_pipeline"],
            catalog_pipeline_job=jobs_by_name["catalog_pipeline"],
            score_and_push_job=jobs_by_name["score_and_push"],
        ),
        resources=_resources,
        jobs=resolved_jobs,
    )


defs = _build_definitions()


def main() -> None:
    """Launch the Dagster webserver for the PRS pipeline."""
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "-m", "dagster", "dev", "-m", "prs_pipeline.definitions"],
        check=True,
    )
