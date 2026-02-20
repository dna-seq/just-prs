"""Dagster Software-Defined Assets for the PRS reference panel pipeline.

Asset lineage (left to right):

  [external]           [download]               [compute]                  [upload]
  ebi_reference_panel  reference_panel  →  per_pgs_scores  →  reference_distributions  →  hf_prs_percentiles
  ebi_pgs_catalog      pgs_id_partitions ↗                 ↗

SourceAssets (external group) are standalone visualization nodes.
Each computed asset documents its source URL in output metadata.
"""

from pathlib import Path

import polars as pl
from dagster import (
    AssetDep,
    AssetExecutionContext,
    AssetIn,
    DynamicPartitionsDefinition,
    Output,
    SourceAsset,
    asset,
)
from eliot import start_action

from just_prs.ftp import PGS_FTP_BASE, list_all_pgs_ids
from just_prs.hf import DEFAULT_HF_PERCENTILES_REPO, push_reference_distributions
from just_prs.reference import (
    REFERENCE_PANEL_URL,
    aggregate_distributions,
    compute_reference_prs_plink2,
    download_reference_panel,
)
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource, Plink2Resource

PGS_IDS_PARTITIONS = DynamicPartitionsDefinition(name="pgs_ids")

# ---------------------------------------------------------------------------
# Source assets — external data that Dagster observes but does not create
# ---------------------------------------------------------------------------

ebi_reference_panel = SourceAsset(
    key="ebi_reference_panel",
    group_name="external",
    description=(
        "PGS Catalog 1000 Genomes reference panel hosted at EBI FTP (~7 GB). "
        "Contains PLINK2 binary genotype files (.pgen/.pvar/.psam) for 2,504 individuals "
        "across 5 superpopulations (AFR, AMR, EAS, EUR, SAS)."
    ),
    metadata={"url": REFERENCE_PANEL_URL},
)

ebi_pgs_catalog = SourceAsset(
    key="ebi_pgs_catalog",
    group_name="external",
    description=(
        "PGS Catalog bulk data at EBI FTP: harmonized scoring files for all 5,000+ PGS IDs "
        "and metadata CSVs (scores, performance metrics, evaluation sample sets)."
    ),
    metadata={"url": PGS_FTP_BASE},
)

# ---------------------------------------------------------------------------
# Download assets — fetch external data into local cache
# ---------------------------------------------------------------------------

@asset(
    group_name="download",
    description="Download and extract the 1000G reference panel (~7 GB, cached after first run).",
)
def reference_panel(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[Path]:
    """Download pgsc_1000G_v1.tar.zst from EBI FTP and extract into the cache directory.

    Skips download if already extracted. Re-materialization only re-downloads
    when the cache directory is missing or overwrite is forced.
    """
    cache_dir = cache_dir_resource.get_path()
    dest = download_reference_panel(cache_dir=cache_dir)
    context.add_output_metadata({
        "ref_dir": str(dest),
        "exists": dest.exists(),
        "source_url": REFERENCE_PANEL_URL,
    })
    return Output(dest)


@asset(
    group_name="download",
    description="Fetch all PGS IDs from EBI FTP and register them as dynamic partitions for per_pgs_scores.",
)
def pgs_id_partitions(
    context: AssetExecutionContext,
) -> Output[list[str]]:
    """Fetch the complete PGS ID list from pgs_scores_list.txt and register as dynamic partitions.

    Materialize this asset first so per_pgs_scores has a partition per PGS ID.
    """
    pgs_ids = list_all_pgs_ids()
    context.instance.add_dynamic_partitions(PGS_IDS_PARTITIONS.name, pgs_ids)
    context.log.info(f"Registered {len(pgs_ids)} PGS ID partitions.")
    context.add_output_metadata({
        "n_pgs_ids": len(pgs_ids),
        "source_url": f"{PGS_FTP_BASE}/pgs_scores_list.txt",
    })
    return Output(pgs_ids)

# ---------------------------------------------------------------------------
# Compute assets — run PLINK2 and aggregate distributions
# ---------------------------------------------------------------------------

@asset(
    group_name="compute",
    partitions_def=PGS_IDS_PARTITIONS,
    op_tags={"dagster/concurrency_key": "plink2"},
    ins={"ref_dir": AssetIn("reference_panel")},
    deps=["pgs_id_partitions"],
    description="Run PLINK2 --score on all 2,504 1000G individuals for one PGS ID (one partition per score).",
)
def per_pgs_scores(
    context: AssetExecutionContext,
    ref_dir: Path,
    cache_dir_resource: CacheDirResource,
    plink2_resource: Plink2Resource,
) -> Output[Path | None]:
    """Run PLINK2 --score on the reference panel for the current PGS ID partition.

    Downloads the harmonized scoring file from EBI FTP if not already cached,
    then runs PLINK2 against the reference panel.
    Output: per-partition parquet at cache_dir/reference_scores/{pgs_id}/scores.parquet
    Columns: pgs_id, iid, superpop, population, score
    """
    if not context.has_partition_key:
        raise RuntimeError(
            "per_pgs_scores must be run as a partitioned execution (one PGS ID per partition). "
            "Use a backfill or launch individual partitions — do not run it in a non-partitioned job."
        )
    pgs_id = context.partition_key
    cache_dir = cache_dir_resource.get_path()
    plink2_bin = plink2_resource.get_bin()

    scores_cache = cache_dir / "scores"
    scores_cache.mkdir(parents=True, exist_ok=True)

    out_dir = cache_dir / "reference_scores" / pgs_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result_parquet = out_dir / "scores.parquet"
    if result_parquet.exists():
        context.log.info(f"Skipping {pgs_id} — already computed at {result_parquet}")
        context.add_output_metadata({"pgs_id": pgs_id, "cached": True, "path": str(result_parquet)})
        return Output(result_parquet)

    from just_prs.scoring import download_scoring_file
    with start_action(action_type="pipeline:download_scoring", pgs_id=pgs_id):
        scoring_file = download_scoring_file(
            pgs_id=pgs_id,
            output_dir=scores_cache,
            genome_build="GRCh38",
        )

    scores_df = compute_reference_prs_plink2(
        pgs_id=pgs_id,
        scoring_file=scoring_file,
        ref_dir=ref_dir,
        out_dir=out_dir,
        plink2_bin=plink2_bin,
        genome_build="GRCh38",
    )

    if scores_df is None:
        context.log.warning(f"PLINK2 scoring failed for {pgs_id}, skipping.")
        context.add_output_metadata({"pgs_id": pgs_id, "failed": True})
        return Output(None)

    scores_df.write_parquet(result_parquet)
    context.add_output_metadata({
        "pgs_id": pgs_id,
        "n_samples": scores_df.height,
        "path": str(result_parquet),
    })
    return Output(result_parquet)


@asset(
    group_name="compute",
    deps=[AssetDep("per_pgs_scores")],
    description="Aggregate all per-PGS score parquets into distribution statistics per superpopulation.",
)
def reference_distributions(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[pl.DataFrame]:
    """Collect all per-PGS parquets and aggregate into reference_distributions.parquet.

    Groups by (pgs_id, superpopulation) → mean, std, n, median, p5, p25, p75, p95.
    Depends on per_pgs_scores (all partitions must be materialized first).

    Output written to cache_dir/percentiles/reference_distributions.parquet.
    Also returned as a DataFrame so the downstream HF upload asset can receive it
    via AssetIn without filesystem coupling.
    """
    cache_dir = cache_dir_resource.get_path()
    scores_root = cache_dir / "reference_scores"

    parquet_files = list(scores_root.rglob("scores.parquet"))
    context.log.info(f"Found {len(parquet_files)} per-PGS score parquets to aggregate.")

    if not parquet_files:
        raise RuntimeError(
            f"No per-PGS score parquets found under {scores_root}. "
            "Materialize per_pgs_scores partitions first."
        )

    dfs: list[pl.DataFrame] = []
    for p in parquet_files:
        df = pl.read_parquet(p)
        if df.height > 0:
            dfs.append(df)

    all_scores = pl.concat(dfs, how="diagonal_relaxed")
    dist_df = aggregate_distributions(all_scores)

    percentiles_dir = cache_dir / "percentiles"
    percentiles_dir.mkdir(parents=True, exist_ok=True)
    out_path = percentiles_dir / "reference_distributions.parquet"
    dist_df.write_parquet(out_path)

    context.add_output_metadata({
        "n_rows": dist_df.height,
        "n_pgs_ids": dist_df["pgs_id"].n_unique(),
        "superpopulations": sorted(dist_df["superpopulation"].unique().to_list()),
        "path": str(out_path),
    })
    return Output(dist_df)

# ---------------------------------------------------------------------------
# Upload asset — push to HuggingFace (represents the HF dataset as an asset)
# ---------------------------------------------------------------------------

@asset(
    group_name="upload",
    ins={"distributions": AssetIn("reference_distributions")},
    description=(
        "HuggingFace dataset just-dna-seq/prs-percentiles — "
        "receives reference_distributions.parquet and uploads it. "
        "End users pull this via PRSCatalog.reference_distributions()."
    ),
)
def hf_prs_percentiles(
    context: AssetExecutionContext,
    distributions: pl.DataFrame,
    cache_dir_resource: CacheDirResource,
    hf_resource: HuggingFaceResource,
) -> Output[str]:
    """Upload reference_distributions.parquet to just-dna-seq/prs-percentiles on HuggingFace.

    Writes the DataFrame to cache_dir/percentiles/ first (ensures the local copy
    is in sync with what was uploaded), then pushes to HF.
    Returns the repo URL as the asset value.
    """
    repo_id = hf_resource.percentiles_repo
    token = hf_resource.get_token()

    cache_dir = cache_dir_resource.get_path()
    percentiles_dir = cache_dir / "percentiles"
    percentiles_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = percentiles_dir / "reference_distributions.parquet"
    distributions.write_parquet(parquet_path)

    push_reference_distributions(
        parquet_path=parquet_path,
        repo_id=repo_id,
        token=token,
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    context.log.info(f"Pushed reference_distributions.parquet to {url}")
    context.add_output_metadata({
        "repo_id": repo_id,
        "url": url,
        "n_rows": distributions.height,
        "n_pgs_ids": distributions["pgs_id"].n_unique(),
    })
    return Output(url)
