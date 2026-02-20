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

defs = dg.Definitions(
    assets=[
        # External source assets (EBI FTP data)
        ebi_reference_panel,
        ebi_pgs_catalog,
        # Download — reference panel pipeline
        reference_panel,
        pgs_id_partitions,
        # Compute — reference panel pipeline
        per_pgs_scores,
        reference_distributions,
        # Upload — reference panel pipeline
        hf_prs_percentiles,
        # Download — metadata pipeline
        raw_pgs_metadata,
        # Compute — metadata pipeline
        cleaned_pgs_metadata,
        # Upload — metadata pipeline
        hf_polygenic_risk_scores,
    ],
    resources={
        "cache_dir_resource": CacheDirResource(),
        "plink2_resource": Plink2Resource(),
        "hf_resource": HuggingFaceResource(),
    },
    jobs=[
        dg.define_asset_job(
            name="download_reference_data",
            selection=[
                "reference_panel",
                "pgs_id_partitions",
            ],
            description=(
                "Download the 1000G reference panel and register PGS ID partitions. "
                "Run this first, then launch per_pgs_scores via backfill."
            ),
        ),
        dg.define_asset_job(
            name="per_pgs_scores_job",
            selection=["per_pgs_scores"],
            partitions_def=PGS_IDS_PARTITIONS,
            description=(
                "Run PLINK2 --score for selected PGS ID partitions. "
                "Launch as a backfill to process multiple partitions."
            ),
        ),
        dg.define_asset_job(
            name="aggregate_and_push",
            selection=["reference_distributions", "hf_prs_percentiles"],
            description="Re-aggregate already-computed per-PGS scores and push to HuggingFace.",
        ),
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
)


def main() -> None:
    """Launch the Dagster webserver for the PRS pipeline."""
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "-m", "dagster", "dev", "-m", "prs_pipeline.definitions"],
        check=True,
    )
