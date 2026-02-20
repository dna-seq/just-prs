"""Dagster Definitions for the PRS reference panel and metadata pipelines."""

import dagster as dg

from prs_pipeline.assets import (
    PGS_IDS_PARTITIONS,
    ebi_pgs_catalog,
    ebi_reference_panel,
    hf_prs_percentiles,
    per_pgs_scores,
    pgs_id_partitions,
    reference_distributions,
    reference_panel,
)
from prs_pipeline.metadata_assets import (
    cleaned_pgs_metadata,
    hf_polygenic_risk_scores,
    raw_pgs_metadata,
)
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource, Plink2Resource
from prs_pipeline.sensors import build_pipeline_sensors

download_reference_data = dg.define_asset_job(
    name="download_reference_data",
    selection=[
        "reference_panel",
        "pgs_id_partitions",
    ],
    description=(
        "Step 1: Download the 1000G reference panel and register PGS ID partitions. "
        "The score_all_partitions_sensor will automatically trigger scoring for all partitions."
    ),
)

per_pgs_scores_job = dg.define_asset_job(
    name="per_pgs_scores_job",
    selection=["per_pgs_scores"],
    partitions_def=PGS_IDS_PARTITIONS,
    description=(
        "Step 2 (auto-triggered by sensor): Run PLINK2 --score for PGS ID partitions. "
        "Triggered automatically after download_reference_data succeeds."
    ),
)

aggregate_and_push = dg.define_asset_job(
    name="aggregate_and_push",
    selection=["reference_distributions", "hf_prs_percentiles"],
    description=(
        "Step 3 (auto-triggered by sensor): Aggregate all per-PGS scores into "
        "reference distributions and push to HuggingFace."
    ),
)

defs = dg.Definitions(
    assets=[
        ebi_reference_panel,
        ebi_pgs_catalog,
        reference_panel,
        pgs_id_partitions,
        per_pgs_scores,
        reference_distributions,
        hf_prs_percentiles,
        raw_pgs_metadata,
        cleaned_pgs_metadata,
        hf_polygenic_risk_scores,
    ],
    resources={
        "cache_dir_resource": CacheDirResource(),
        "plink2_resource": Plink2Resource(),
        "hf_resource": HuggingFaceResource(),
    },
    jobs=[
        download_reference_data,
        per_pgs_scores_job,
        aggregate_and_push,
        dg.define_asset_job(
            name="metadata_pipeline",
            selection=["raw_pgs_metadata", "cleaned_pgs_metadata", "hf_polygenic_risk_scores"],
            description=(
                "End-to-end metadata pipeline: download raw PGS Catalog sheets from EBI FTP, "
                "run cleanup pipeline, push cleaned parquets to HuggingFace."
            ),
        ),
        dg.define_asset_job(
            name="clean_and_push_metadata",
            selection=["cleaned_pgs_metadata", "hf_polygenic_risk_scores"],
            description=(
                "Re-run cleanup pipeline on already-cached raw metadata and push to HuggingFace. "
                "Use this when raw sheets are already present in the cache."
            ),
        ),
    ],
    sensors=build_pipeline_sensors(per_pgs_scores_job, aggregate_and_push),
)


def main() -> None:
    """Launch the Dagster webserver for the PRS pipeline."""
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "-m", "dagster", "dev", "-m", "prs_pipeline.definitions"],
        check=True,
    )
