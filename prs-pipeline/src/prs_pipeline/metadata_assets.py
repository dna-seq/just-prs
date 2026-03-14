"""Dagster Software-Defined Assets for the PGS Catalog metadata pipeline.

This pipeline downloads, cleans, and publishes PGS Catalog metadata — the tables
that describe *what* each Polygenic Score measures (trait, method, publication) and
*how well* it performs (AUROC, OR, C-index evaluated on specific cohorts).

Asset lineage (left to right):

  [download]          [compute]              [upload]
  raw_pgs_metadata → cleaned_pgs_metadata → hf_polygenic_risk_scores
                                           ↘ hf_prs_percentiles (in assets.py)
                                           ↘ hf_pgs_catalog (+ scoring_files_parquet dep)

cleaned_pgs_metadata feeds into three upload assets: hf_polygenic_risk_scores
(metadata-only HF repo), hf_prs_percentiles (enriched distributions), and
hf_pgs_catalog (combined metadata + scoring parquets at just-dna-seq/pgs-catalog).
SourceAsset ebi_pgs_catalog_scoring_files is declared in assets.py as a
visualisation-only node documenting the FTP origin.
"""

import os
from pathlib import Path

import polars as pl
from dagster import AssetDep, AssetExecutionContext, Output, asset
from eliot import start_action

from just_prs.cleanup import best_performance_per_score, clean_performance_metrics, clean_scores
from just_prs.ftp import PGS_METADATA_BASE, download_metadata_sheet
from just_prs.hf import DEFAULT_HF_REPO, push_cleaned_parquets, push_pgs_catalog
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource
from prs_pipeline.runtime import resource_tracker


def _no_cache() -> bool:
    """Return True if PRS_PIPELINE_NO_CACHE is set (user passed --no-cache)."""
    return os.environ.get("PRS_PIPELINE_NO_CACHE", "").strip().lower() in {"1", "true", "yes"}


@asset(
    group_name="download",
    deps=[AssetDep("ebi_scoring_files_fingerprint")],
    description=(
        "Downloads three bulk metadata CSV sheets from the PGS Catalog FTP server "
        "and converts them to Parquet: scores (what each PGS measures), "
        "performance_metrics (AUROC, OR, etc. from validation studies), and "
        "evaluation_sample_sets (cohort details for each validation). "
        "These raw sheets contain verbose column names and inconsistent genome "
        "build labels that are normalised by the downstream cleaned_pgs_metadata asset."
    ),
)
def raw_pgs_metadata(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[Path]:
    """Fetch bulk metadata CSV sheets from the PGS Catalog FTP and save as Parquet."""
    cache_dir = cache_dir_resource.get_path()
    raw_dir = cache_dir / "metadata" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    overwrite = _no_cache()
    if overwrite:
        context.log.info("NO-CACHE: will re-download all metadata sheets from EBI FTP.")

    with resource_tracker("raw_pgs_metadata", context=context):
        sheet_meta: dict[str, int] = {}
        for sheet_name in ("scores", "performance_metrics", "evaluation_sample_sets"):
            dest = raw_dir / f"{sheet_name}.parquet"
            with start_action(action_type="pipeline:download_metadata_sheet", sheet=sheet_name):
                df = download_metadata_sheet(sheet_name, dest, overwrite=overwrite)
            sheet_meta[sheet_name] = df.height
            context.log.info(f"Downloaded {sheet_name}: {df.height:,} rows → {dest}")

    context.add_output_metadata({
        "raw_dir": str(raw_dir),
        "source_url": PGS_METADATA_BASE,
        **{f"n_{k}": v for k, v in sheet_meta.items()},
    })
    return Output(raw_dir)


@asset(
    group_name="compute",
    deps=[AssetDep("raw_pgs_metadata")],
    description=(
        "Cleans and normalises the raw PGS Catalog metadata into analysis-ready Parquets. "
        "Renames verbose column headers to snake_case, normalises genome build labels "
        "(hg19/hg37 → GRCh37, hg38 → GRCh38), and parses metric strings like "
        "'1.55 [1.52,1.58]' into numeric estimate/CI columns. "
        "Produces three files: scores.parquet (score metadata), performance.parquet "
        "(all validation results joined with evaluation cohort details), and "
        "best_performance.parquet (one best-performing row per PGS ID, preferring "
        "the largest European-ancestry cohort)."
    ),
)
def cleaned_pgs_metadata(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[Path]:
    """Apply the cleanup pipeline to raw PGS Catalog metadata Parquets."""
    cache_dir = cache_dir_resource.get_path()
    raw_dir = cache_dir / "metadata" / "raw"
    metadata_dir = cache_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with resource_tracker("cleaned_pgs_metadata", context=context):
        with start_action(action_type="pipeline:clean_pgs_metadata"):
            scores_df = pl.read_parquet(raw_dir / "scores.parquet")
            perf_df = pl.read_parquet(raw_dir / "performance_metrics.parquet")
            eval_df = pl.read_parquet(raw_dir / "evaluation_sample_sets.parquet")

            scores_lf = clean_scores(scores_df)
            perf_lf = clean_performance_metrics(perf_df, eval_df)
            best_perf_lf = best_performance_per_score(perf_lf)

            scores_out = scores_lf.collect()
            perf_out = perf_lf.collect()
            best_perf_out = best_perf_lf.collect()

            scores_out.write_parquet(metadata_dir / "scores.parquet")
            perf_out.write_parquet(metadata_dir / "performance.parquet")
            best_perf_out.write_parquet(metadata_dir / "best_performance.parquet")

    context.log.info(
        f"Cleaned metadata: {scores_out.height:,} scores, "
        f"{perf_out.height:,} performance rows, "
        f"{best_perf_out.height:,} best-performance rows."
    )
    context.add_output_metadata({
        "metadata_dir": str(metadata_dir),
        "n_scores": scores_out.height,
        "n_performance": perf_out.height,
        "n_best_performance": best_perf_out.height,
        "n_unique_pgs_ids": scores_out["pgs_id"].n_unique(),
    })
    return Output(metadata_dir)


@asset(
    group_name="upload",
    deps=[AssetDep("cleaned_pgs_metadata")],
    description=(
        "Uploads the cleaned PGS Catalog metadata (scores, performance, "
        "best_performance Parquets) to the HuggingFace dataset "
        "just-dna-seq/polygenic_risk_scores. This is the public dataset that "
        "the just-prs library pulls on first use via PRSCatalog so that users "
        "get pre-cleaned metadata without needing to run the cleanup pipeline locally."
    ),
)
def hf_polygenic_risk_scores(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
    hf_resource: HuggingFaceResource,
) -> Output[str]:
    """Push cleaned metadata Parquets to HuggingFace and return the repo URL."""
    cache_dir = cache_dir_resource.get_path()
    metadata_dir = cache_dir / "metadata"
    repo_id = hf_resource.metadata_repo
    token = hf_resource.get_token()

    with resource_tracker("hf_polygenic_risk_scores", context=context):
        with start_action(action_type="pipeline:push_cleaned_parquets", repo_id=repo_id):
            push_cleaned_parquets(local_dir=metadata_dir, repo_id=repo_id, token=token)

    url = f"https://huggingface.co/datasets/{repo_id}"
    context.log.info(f"Pushed cleaned metadata parquets to {url}")

    dist_df = pl.read_parquet(metadata_dir / "scores.parquet")
    context.add_output_metadata({
        "repo_id": repo_id,
        "url": url,
        "n_scores": dist_df.height,
    })
    return Output(url)


@asset(
    group_name="upload",
    deps=[AssetDep("cleaned_pgs_metadata"), AssetDep("scoring_files_parquet")],
    description=(
        "Uploads the full PGS Catalog dataset — cleaned metadata parquets and "
        "all scoring file parquets — to the HuggingFace dataset "
        "just-dna-seq/pgs-catalog. Metadata goes under data/metadata/ and "
        "scoring files under data/scores/. Generates a dataset card with "
        "release statistics and timestamp."
    ),
)
def hf_pgs_catalog(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
    hf_resource: HuggingFaceResource,
) -> Output[str]:
    """Push scoring parquets and cleaned metadata to a combined HF dataset."""
    cache_dir = cache_dir_resource.get_path()
    metadata_dir = cache_dir / "metadata"
    scores_dir = cache_dir / "scores"
    repo_id = hf_resource.catalog_repo
    token = hf_resource.get_token()

    scoring_parquets = sorted(scores_dir.glob("*_hmPOS_*.parquet"))
    scoring_parquets = [p for p in scoring_parquets if p.name != "conversion_failures.parquet"]
    n_scoring = len(scoring_parquets)

    with resource_tracker("hf_pgs_catalog", context=context):
        with start_action(action_type="pipeline:push_pgs_catalog", repo_id=repo_id):
            push_pgs_catalog(
                metadata_dir=metadata_dir,
                scores_dir=scores_dir,
                repo_id=repo_id,
                token=token,
            )

    url = f"https://huggingface.co/datasets/{repo_id}"
    context.log.info(f"Pushed PGS Catalog ({n_scoring} scoring parquets + metadata) to {url}")

    scores_df = pl.read_parquet(metadata_dir / "scores.parquet")
    context.add_output_metadata({
        "repo_id": repo_id,
        "url": url,
        "n_metadata_scores": scores_df.height,
        "n_scoring_parquets": n_scoring,
    })
    return Output(url)
