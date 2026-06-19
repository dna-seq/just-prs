"""Dagster Definitions for the PRS reference panel and metadata pipelines."""

import dagster as dg
from dagster import in_process_executor

from prs_pipeline.assets import (
    chip_coverage,
    ebi_reference_panel_fingerprint,
    ebi_pgs_catalog_reference_panel,
    ebi_pgs_catalog_scoring_files,
    ebi_scoring_files_fingerprint,
    hf_chip_coverage,
    hf_ld_proxy_table,
    hf_prs_percentiles,
    illumina_gsa_manifest,
    ld_proxy_table,
    reference_percentile_audit,
    reference_panel,
    reference_scores,
    scoring_files,
    scoring_files_parquet,
)
from prs_pipeline.metadata_assets import (
    cleaned_pgs_metadata,
    gwas_studies,
    hf_pgs_catalog,
    hf_pgs_catalog_risk_metadata,
    raw_pgs_metadata,
    trait_heritability,
    trait_prevalence,
)
from prs_pipeline.checks import ALL_ASSET_CHECKS
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource
from prs_pipeline.sensors import make_all_sensors
from prs_pipeline.utils import resource_summary_hook

download_reference_data = dg.define_asset_job(
    name="download_reference_data",
    selection=["ebi_reference_panel_fingerprint", "reference_panel"],
    description="Download the reference panel from the EBI FTP server.",
    hooks={resource_summary_hook},
    executor_def=in_process_executor,
)

_score_and_push_assets = dg.AssetSelection.assets(
    "ebi_scoring_files_fingerprint", "scoring_files",
    "scoring_files_parquet", "reference_scores",
    "raw_pgs_metadata", "cleaned_pgs_metadata",
    "gwas_studies", "trait_prevalence", "trait_heritability",
    "hf_pgs_catalog_risk_metadata", "hf_prs_percentiles",
)

score_and_push = dg.define_asset_job(
    name="score_and_push",
    selection=_score_and_push_assets | dg.AssetSelection.checks_for_assets(
        "reference_scores", "hf_prs_percentiles", "cleaned_pgs_metadata",
    ),
    description=(
        "Score all PGS IDs against the reference panel in a single batch, "
        "download and clean metadata, enrich distributions, and push to HuggingFace. "
        "Includes data quality checks on distributions and metadata."
    ),
    hooks={resource_summary_hook},
    executor_def=in_process_executor,
)

chip_coverage_pipeline = dg.define_asset_job(
    name="chip_coverage_pipeline",
    selection=dg.AssetSelection.assets(
        "scoring_files", "scoring_files_parquet",
        "chip_coverage", "hf_chip_coverage",
    ) | dg.AssetSelection.checks_for_assets("chip_coverage"),
    description=(
        "Compute consumer-chip (Illumina GSA) coverage of all PGS scoring files "
        "and publish chip_coverage.parquet to HuggingFace. Lightweight: reuses "
        "cached scoring parquets and does not touch the reference panel."
    ),
    hooks={resource_summary_hook},
    executor_def=in_process_executor,
)

reference_percentile_audit_job = dg.define_asset_job(
    name="reference_percentile_audit_job",
    selection=dg.AssetSelection.assets("reference_percentile_audit"),
    description=(
        "Audit cached or HuggingFace reference percentile distributions and write "
        "quality sidecars without recomputing reference scores."
    ),
    hooks={resource_summary_hook},
    executor_def=in_process_executor,
)

ld_proxy_pipeline = dg.define_asset_job(
    name="ld_proxy_pipeline",
    selection=dg.AssetSelection.assets(
        "reference_panel", "scoring_files", "scoring_files_parquet",
        "ld_proxy_table", "hf_ld_proxy_table",
    ) | dg.AssetSelection.checks_for_assets("ld_proxy_table"),
    description=(
        "Build LD-proxy lookup tables for consumer genotyping arrays (GSA v3) "
        "from the 1000G reference panel and publish to HuggingFace. "
        "Currently produces a GRCh38 table. Includes schema validation checks."
    ),
    hooks={resource_summary_hook},
    executor_def=in_process_executor,
)

_full_pipeline_assets = dg.AssetSelection.assets(
    "ebi_reference_panel_fingerprint", "ebi_scoring_files_fingerprint",
    "reference_panel", "scoring_files", "scoring_files_parquet",
    "chip_coverage", "hf_chip_coverage",
    "reference_scores",
    "raw_pgs_metadata", "cleaned_pgs_metadata",
    "gwas_studies", "trait_prevalence", "trait_heritability",
    "hf_pgs_catalog", "hf_pgs_catalog_risk_metadata", "hf_prs_percentiles",
)

full_pipeline = dg.define_asset_job(
    name="full_pipeline",
    selection=_full_pipeline_assets | dg.AssetSelection.checks_for_assets(
        "reference_scores", "hf_prs_percentiles", "cleaned_pgs_metadata", "chip_coverage",
    ),
    description=(
        "Full pipeline: download reference panel, batch-score all PGS IDs, "
        "download and clean PGS Catalog metadata, enrich distributions with "
        "metadata, and push to HuggingFace. Includes data quality checks. "
        "Triggered automatically by run_pipeline_on_startup sensor when "
        "assets are unmaterialized."
    ),
    hooks={resource_summary_hook},
    executor_def=in_process_executor,
)

catalog_pipeline = dg.define_asset_job(
    name="catalog_pipeline",
    selection=dg.AssetSelection.assets(
        "ebi_scoring_files_fingerprint",
        "scoring_files", "scoring_files_parquet",
        "raw_pgs_metadata", "cleaned_pgs_metadata",
        "gwas_studies", "trait_prevalence", "trait_heritability",
        "hf_pgs_catalog", "hf_pgs_catalog_risk_metadata",
    ) | dg.AssetSelection.checks_for_assets("cleaned_pgs_metadata"),
    description=(
        "Build and push the combined PGS Catalog dataset to HuggingFace "
        "(just-dna-seq/pgs-catalog): download scoring files, convert to parquet, "
        "download and clean metadata, then upload both to HF. "
        "Includes metadata quality checks."
    ),
    hooks={resource_summary_hook},
    executor_def=in_process_executor,
)

metadata_pipeline = dg.define_asset_job(
    name="metadata_pipeline",
    selection=dg.AssetSelection.assets(
        "ebi_scoring_files_fingerprint",
        "raw_pgs_metadata", "cleaned_pgs_metadata",
    ) | dg.AssetSelection.checks_for_assets("cleaned_pgs_metadata"),
    description=(
        "End-to-end metadata pipeline: download raw PGS Catalog sheets from EBI FTP "
        "and run the cleanup pipeline. Includes metadata quality checks. "
        "Metadata is published to HuggingFace via the catalog_pipeline (hf_pgs_catalog "
        "asset) together with scoring files."
    ),
    hooks={resource_summary_hook},
    executor_def=in_process_executor,
)

_assets = [
    ebi_pgs_catalog_reference_panel,
    ebi_pgs_catalog_scoring_files,
    illumina_gsa_manifest,
    ebi_reference_panel_fingerprint,
    ebi_scoring_files_fingerprint,
    scoring_files,
    scoring_files_parquet,
    chip_coverage,
    hf_chip_coverage,
    hf_ld_proxy_table,
    ld_proxy_table,
    reference_percentile_audit,
    reference_panel,
    reference_scores,
    hf_prs_percentiles,
    raw_pgs_metadata,
    cleaned_pgs_metadata,
    gwas_studies,
    trait_prevalence,
    trait_heritability,
    hf_pgs_catalog,
    hf_pgs_catalog_risk_metadata,
]
_asset_checks = ALL_ASSET_CHECKS
_resources = {
    "cache_dir_resource": CacheDirResource(),
    "hf_resource": HuggingFaceResource(),
}
_unresolved_jobs = [
    download_reference_data,
    score_and_push,
    full_pipeline,
    catalog_pipeline,
    chip_coverage_pipeline,
    ld_proxy_pipeline,
    reference_percentile_audit_job,
    metadata_pipeline,
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
        asset_checks=_asset_checks,
        sensors=make_all_sensors(
            full_pipeline_job=jobs_by_name["full_pipeline"],
            catalog_pipeline_job=jobs_by_name["catalog_pipeline"],
            score_and_push_job=jobs_by_name["score_and_push"],
            reference_percentile_audit_job=jobs_by_name["reference_percentile_audit_job"],
            ld_proxy_pipeline_job=jobs_by_name["ld_proxy_pipeline"],
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
