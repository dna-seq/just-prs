"""Dagster Software-Defined Assets for the PGS Catalog metadata pipeline.

Asset lineage (left to right):

  [external]         [download]          [compute]              [upload]
  ebi_pgs_catalog → raw_pgs_metadata → cleaned_pgs_metadata → hf_polygenic_risk_scores

SourceAsset ebi_pgs_catalog is declared in assets.py and shared between pipelines.
"""

from pathlib import Path

import polars as pl
from dagster import AssetDep, AssetExecutionContext, Output, asset
from eliot import start_action

from just_prs.cleanup import best_performance_per_score, clean_performance_metrics, clean_scores
from just_prs.ftp import download_metadata_sheet
from just_prs.hf import DEFAULT_HF_REPO, push_cleaned_parquets
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource


@asset(
    group_name="download",
    deps=["ebi_pgs_catalog"],
    description=(
        "Download raw PGS Catalog metadata sheets from EBI FTP into local cache: "
        "scores, performance_metrics, and evaluation_sample_sets."
    ),
)
def raw_pgs_metadata(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[Path]:
    """Fetch the three bulk metadata CSV sheets from EBI FTP and save as parquet.

    Writes to <cache>/metadata/raw/:
      scores.parquet, performance_metrics.parquet, evaluation_sample_sets.parquet

    Skips sheets that are already cached.
    """
    cache_dir = cache_dir_resource.get_path()
    raw_dir = cache_dir / "metadata" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sheet_meta: dict[str, int] = {}
    for sheet_name in ("scores", "performance_metrics", "evaluation_sample_sets"):
        dest = raw_dir / f"{sheet_name}.parquet"
        with start_action(action_type="pipeline:download_metadata_sheet", sheet=sheet_name):
            df = download_metadata_sheet(sheet_name, dest)
        sheet_meta[sheet_name] = df.height
        context.log.info(f"Downloaded {sheet_name}: {df.height:,} rows → {dest}")

    context.add_output_metadata({
        "raw_dir": str(raw_dir),
        **{f"n_{k}": v for k, v in sheet_meta.items()},
    })
    return Output(raw_dir)


@asset(
    group_name="compute",
    deps=[AssetDep("raw_pgs_metadata")],
    description=(
        "Run the cleanup pipeline on raw PGS Catalog metadata to produce "
        "scores.parquet, performance.parquet, and best_performance.parquet."
    ),
)
def cleaned_pgs_metadata(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[Path]:
    """Apply the cleanup pipeline to the raw FTP parquets.

    Reads from <cache>/metadata/raw/ and writes cleaned parquets to <cache>/metadata/:
      scores.parquet          — renamed columns, normalized genome builds
      performance.parquet     — parsed metric strings, joined with evaluation samples
      best_performance.parquet — one row per PGS ID (largest sample, EUR-preferred)
    """
    cache_dir = cache_dir_resource.get_path()
    raw_dir = cache_dir / "metadata" / "raw"
    metadata_dir = cache_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

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
        "HuggingFace dataset just-dna-seq/polygenic_risk_scores — "
        "uploads cleaned metadata parquets (scores, performance, best_performance). "
        "End users pull these via PRSCatalog._load_all()."
    ),
)
def hf_polygenic_risk_scores(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
    hf_resource: HuggingFaceResource,
) -> Output[str]:
    """Push cleaned metadata parquets to the HuggingFace dataset repo.

    Reads the three cleaned parquets from <cache>/metadata/ and uploads them
    along with a generated dataset card README to the configured metadata repo.
    Returns the HF dataset URL.
    """
    cache_dir = cache_dir_resource.get_path()
    metadata_dir = cache_dir / "metadata"
    repo_id = hf_resource.metadata_repo
    token = hf_resource.get_token()

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
