"""Dagster Definitions for the PRS reference panel and metadata pipelines."""

import dagster as dg

from prs_pipeline.assets import (
    ebi_pgs_catalog_reference_panel,
    ebi_pgs_catalog_scoring_files,
    hf_prs_percentiles,
    reference_panel,
    reference_scores,
)
from prs_pipeline.metadata_assets import (
    cleaned_pgs_metadata,
    hf_polygenic_risk_scores,
    raw_pgs_metadata,
)
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource
from prs_pipeline.sensors import run_pipeline_on_startup
from prs_pipeline.utils import resource_summary_hook

download_reference_data = dg.define_asset_job(
    name="download_reference_data",
    selection=["reference_panel"],
    description="Download the reference panel from the EBI FTP server.",
    hooks={resource_summary_hook},
)

score_and_push = dg.define_asset_job(
    name="score_and_push",
    selection=[
        "reference_scores",
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
        "reference_panel", "reference_scores",
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

defs = dg.Definitions(
    assets=[
        ebi_pgs_catalog_reference_panel,
        ebi_pgs_catalog_scoring_files,
        reference_panel,
        reference_scores,
        hf_prs_percentiles,
        raw_pgs_metadata,
        cleaned_pgs_metadata,
        hf_polygenic_risk_scores,
    ],
    sensors=[run_pipeline_on_startup],
    resources={
        "cache_dir_resource": CacheDirResource(),
        "hf_resource": HuggingFaceResource(),
    },
    jobs=[
        download_reference_data,
        score_and_push,
        full_pipeline,
        dg.define_asset_job(
            name="metadata_pipeline",
            selection=["raw_pgs_metadata", "cleaned_pgs_metadata", "hf_polygenic_risk_scores"],
            description=(
                "End-to-end metadata pipeline: download raw PGS Catalog sheets from EBI FTP, "
                "run cleanup pipeline, push cleaned parquets to HuggingFace."
            ),
            hooks={resource_summary_hook},
        ),
        dg.define_asset_job(
            name="clean_and_push_metadata",
            selection=["cleaned_pgs_metadata", "hf_polygenic_risk_scores"],
            description=(
                "Re-run cleanup pipeline on already-cached raw metadata and push to HuggingFace. "
                "Use this when raw sheets are already present in the cache."
            ),
            hooks={resource_summary_hook},
        ),
    ],
)


def main() -> None:
    """Launch the Dagster webserver for the PRS pipeline."""
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "-m", "dagster", "dev", "-m", "prs_pipeline.definitions"],
        check=True,
    )
