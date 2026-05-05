"""Dagster Software-Defined Assets for the PGS Catalog metadata pipeline.

This pipeline downloads, cleans, and publishes PGS Catalog metadata — the tables
that describe *what* each Polygenic Score measures (trait, method, publication) and
*how well* it performs (AUROC, OR, C-index evaluated on specific cohorts).

Asset lineage (left to right):

  [download]          [compute]              [upload]
  raw_pgs_metadata → cleaned_pgs_metadata → hf_pgs_catalog (+ scoring_files_parquet dep)
                                           ↘ hf_prs_percentiles (in assets.py)

cleaned_pgs_metadata feeds into two upload assets: hf_pgs_catalog (combined
metadata + scoring parquets at just-dna-seq/pgs-catalog) and hf_prs_percentiles
(enriched distributions). The just-prs library pulls cleaned metadata from the
pgs-catalog repo on first use via PRSCatalog.
SourceAsset ebi_pgs_catalog_scoring_files is declared in assets.py as a
visualisation-only node documenting the FTP origin.
"""

import os
from pathlib import Path

import polars as pl
from dagster import AssetDep, AssetExecutionContext, Output, asset
from eliot import start_action

from just_prs.cleanup import best_performance_per_score, clean_performance_metrics, clean_publications, clean_scores
from just_prs.ftp import PGS_METADATA_BASE, download_metadata_sheet
from just_prs.gwas import build_gwas_trait_summary, download_gwas_studies, download_gwas_trait_mappings
from just_prs.heritability import (
    build_heritability_table,
    download_efo_ukb_mappings,
    download_gwas_atlas_heritability,
    download_pan_ukbb_heritability,
    map_gwas_atlas_to_efo,
    push_heritability_to_hf,
)
from just_prs.hf import push_pgs_catalog
from just_prs.ontology import alias_coverage_metadata
from just_prs.prevalence import build_prevalence_table, push_prevalence_to_hf
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
        for sheet_name in ("scores", "performance_metrics", "evaluation_sample_sets", "publications"):
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
            pub_df = pl.read_parquet(raw_dir / "publications.parquet")

            scores_lf = clean_scores(scores_df)
            perf_lf = clean_performance_metrics(perf_df, eval_df)
            best_perf_lf = best_performance_per_score(perf_lf)
            pub_lf = clean_publications(pub_df)

            scores_out = scores_lf.collect()
            perf_out = perf_lf.collect()
            best_perf_out = best_perf_lf.collect()
            pub_out = pub_lf.collect()

            scores_out.write_parquet(metadata_dir / "scores.parquet")
            perf_out.write_parquet(metadata_dir / "performance.parquet")
            best_perf_out.write_parquet(metadata_dir / "best_performance.parquet")
            pub_out.write_parquet(metadata_dir / "publications.parquet")

    context.log.info(
        f"Cleaned metadata: {scores_out.height:,} scores, "
        f"{perf_out.height:,} performance rows, "
        f"{best_perf_out.height:,} best-performance rows, "
        f"{pub_out.height:,} publications."
    )
    context.add_output_metadata({
        "metadata_dir": str(metadata_dir),
        "n_scores": scores_out.height,
        "n_performance": perf_out.height,
        "n_best_performance": best_perf_out.height,
        "n_publications": pub_out.height,
        "n_unique_pgs_ids": scores_out["pgs_id"].n_unique(),
    })
    return Output(metadata_dir)


@asset(
    group_name="download",
    deps=[AssetDep("ebi_scoring_files_fingerprint")],
    description=(
        "Downloads the GWAS Catalog bulk studies TSV and trait-to-EFO mapping file, "
        "parses case/control counts from free-text sample descriptions, joins with "
        "EFO trait IDs, and produces a per-EFO-trait summary with the largest study's "
        "case fraction. Cached as gwas_studies.parquet."
    ),
)
def gwas_studies(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[Path]:
    """Download GWAS Catalog bulk data and build per-trait summary."""
    cache_dir = cache_dir_resource.get_path()
    gwas_dir = cache_dir / "metadata" / "raw" / "gwas"
    gwas_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = cache_dir / "metadata"

    overwrite = _no_cache()

    with resource_tracker("gwas_studies", context=context):
        context.log.info("Downloading GWAS Catalog bulk studies TSV...")
        studies_df = download_gwas_studies(gwas_dir / "gwas_studies_raw.parquet", overwrite=overwrite)
        context.log.info(f"Downloaded {studies_df.height:,} GWAS studies.")

        context.log.info("Downloading GWAS Catalog trait mappings...")
        mappings_df = download_gwas_trait_mappings(gwas_dir / "gwas_trait_mappings.parquet", overwrite=overwrite)
        context.log.info(f"Downloaded {mappings_df.height:,} trait mappings.")

        summary_df = build_gwas_trait_summary(studies_df, mappings_df)
        out_path = metadata_dir / "gwas_studies.parquet"
        summary_df.write_parquet(out_path)
        context.log.info(f"Built GWAS trait summary: {summary_df.height:,} EFO traits with case/control data.")

    context.add_output_metadata({
        "n_raw_studies": studies_df.height,
        "n_trait_mappings": mappings_df.height,
        "n_efo_traits_with_counts": summary_df.height,
        "output_path": str(out_path),
    })
    return Output(out_path)


@asset(
    group_name="compute",
    deps=[AssetDep("cleaned_pgs_metadata"), AssetDep("gwas_studies")],
    description=(
        "Merges prevalence data from three tiers into a single per-EFO-trait table: "
        "(1) hand-curated seed CSV, (2) GWAS Catalog cohort fractions, "
        "(3) PGS Catalog evaluation cohort fractions. "
        "Higher-confidence tiers take priority when multiple sources cover the same trait."
    ),
)
def trait_prevalence(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[Path]:
    """Build the merged trait prevalence table."""
    cache_dir = cache_dir_resource.get_path()
    metadata_dir = cache_dir / "metadata"
    out_path = metadata_dir / "trait_prevalence.parquet"

    with resource_tracker("trait_prevalence", context=context):
        scores_lf = pl.scan_parquet(metadata_dir / "scores.parquet")
        best_perf_lf = pl.scan_parquet(metadata_dir / "best_performance.parquet")

        gwas_summary_df: pl.DataFrame | None = None
        gwas_path = metadata_dir / "gwas_studies.parquet"
        if gwas_path.exists():
            gwas_summary_df = pl.read_parquet(gwas_path)
            context.log.info(f"Loaded GWAS trait summary: {gwas_summary_df.height:,} traits.")

        prevalence_df = build_prevalence_table(
            scores_lf=scores_lf,
            best_performance_lf=best_perf_lf,
            gwas_summary_df=gwas_summary_df,
            xref_cache_dir=metadata_dir / "raw" / "ontology_xrefs",
        )
        prevalence_df.write_parquet(out_path)
        context.log.info(f"Built prevalence table: {prevalence_df.height:,} traits.")

    n_by_source: dict[str, int] = {}
    if "source" in prevalence_df.columns and prevalence_df.height > 0:
        for row in prevalence_df.group_by("source").len().iter_rows(named=True):
            n_by_source[row["source"]] = row["len"]

    context.add_output_metadata({
        "n_traits": prevalence_df.height,
        "output_path": str(out_path),
        **alias_coverage_metadata(prevalence_df, prefix="prevalence"),
        **{f"n_source_{k}": v for k, v in n_by_source.items()},
    })
    return Output(out_path)


@asset(
    group_name="download",
    deps=[AssetDep("cleaned_pgs_metadata")],
    description=(
        "Downloads SNP heritability estimates from Pan-UK Biobank (~7,200 traits × 6 ancestries) "
        "and GWAS Atlas (~4,700 GWAS). Maps traits to EFO IDs via the EBISPOT EFO-UKB "
        "master mapping file. Produces trait_heritability.parquet with multiple rows "
        "per trait (one per ancestry × source × method) for transparent comparison."
    ),
)
def trait_heritability(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[Path]:
    """Download and merge heritability data from Pan-UKBB and GWAS Atlas."""
    cache_dir = cache_dir_resource.get_path()
    metadata_dir = cache_dir / "metadata"
    raw_dir = metadata_dir / "raw" / "heritability"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = metadata_dir / "trait_heritability.parquet"

    overwrite = _no_cache()

    with resource_tracker("trait_heritability", context=context):
        context.log.info("Downloading EBISPOT EFO-UKB mappings...")
        mappings_df = download_efo_ukb_mappings(
            raw_dir / "efo_ukb_mappings.parquet", overwrite=overwrite,
        )
        context.log.info(f"EFO-UKB mappings: {mappings_df.height:,} rows.")

        context.log.info("Downloading Pan-UKBB heritability data...")
        pan_ukbb_df = download_pan_ukbb_heritability(
            raw_dir / "pan_ukbb_h2.parquet",
            mappings_df=mappings_df,
            overwrite=overwrite,
        )
        context.log.info(
            f"Pan-UKBB heritability: {pan_ukbb_df.height:,} rows, "
            f"{pan_ukbb_df['efo_id'].n_unique():,} EFO traits."
        )

        context.log.info("Downloading GWAS Atlas heritability data...")
        atlas_df = download_gwas_atlas_heritability(
            raw_dir / "gwas_atlas_h2.parquet", overwrite=overwrite,
        )

        gwas_mappings_path = metadata_dir / "raw" / "gwas" / "gwas_trait_mappings.parquet"
        if gwas_mappings_path.exists():
            atlas_df = map_gwas_atlas_to_efo(atlas_df, gwas_mappings_path)
            n_mapped = atlas_df.filter(pl.col("efo_id").is_not_null()).height
            context.log.info(
                f"GWAS Atlas heritability: {atlas_df.height:,} rows, "
                f"{n_mapped:,} mapped to EFO."
            )
        else:
            context.log.info(
                f"GWAS Atlas heritability: {atlas_df.height:,} rows "
                "(no GWAS trait mappings available for EFO linking)."
            )

        combined_df = build_heritability_table(
            pan_ukbb_df=pan_ukbb_df,
            gwas_atlas_df=atlas_df,
            requested_traits_df=pl.read_parquet(metadata_dir / "scores.parquet").select(
                "trait_efo_id", "trait_efo"
            ),
            efo_mappings_df=mappings_df,
            xref_cache_dir=raw_dir / "ontology_xrefs",
        )
        combined_df.write_parquet(out_path)
        context.log.info(
            f"Combined heritability table: {combined_df.height:,} rows, "
            f"{combined_df['efo_id'].n_unique():,} EFO traits."
        )

    n_by_source: dict[str, int] = {}
    if "source" in combined_df.columns and combined_df.height > 0:
        for row in combined_df.group_by("source").len().iter_rows(named=True):
            n_by_source[row["source"]] = row["len"]

    context.add_output_metadata({
        "n_total_rows": combined_df.height,
        "n_efo_traits": combined_df["efo_id"].n_unique(),
        "n_efo_ukb_mappings": mappings_df.height,
        "output_path": str(out_path),
        **alias_coverage_metadata(combined_df, prefix="heritability"),
        **{f"n_source_{k}": v for k, v in n_by_source.items()},
    })
    return Output(out_path)


@asset(
    group_name="upload",
    deps=[AssetDep("trait_prevalence"), AssetDep("trait_heritability")],
    description=(
        "Uploads trait-level risk metadata to the HuggingFace PGS Catalog dataset: "
        "trait_prevalence.parquet and trait_heritability.parquet under data/metadata/. "
        "These tables let PRSCatalog compute absolute-risk estimates from a user's "
        "PRS z-score using prevalence and h²-liability data."
    ),
)
def hf_pgs_catalog_risk_metadata(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
    hf_resource: HuggingFaceResource,
) -> Output[str]:
    """Push prevalence and heritability tables used for absolute-risk estimates."""
    cache_dir = cache_dir_resource.get_path()
    metadata_dir = cache_dir / "metadata"
    repo_id = hf_resource.catalog_repo
    token = hf_resource.get_token()
    prevalence_path = metadata_dir / "trait_prevalence.parquet"
    heritability_path = metadata_dir / "trait_heritability.parquet"

    with resource_tracker("hf_pgs_catalog_risk_metadata", context=context):
        with start_action(action_type="pipeline:push_pgs_catalog_risk_metadata", repo_id=repo_id):
            push_prevalence_to_hf(prevalence_path, repo_id=repo_id, token=token)
            push_heritability_to_hf(heritability_path, repo_id=repo_id, token=token)

    prevalence_df = pl.read_parquet(prevalence_path)
    heritability_df = pl.read_parquet(heritability_path)
    url = f"https://huggingface.co/datasets/{repo_id}/tree/main/data/metadata"
    context.log.info(f"Pushed prevalence and heritability risk metadata to {url}")
    context.add_output_metadata({
        "repo_id": repo_id,
        "url": url,
        "prevalence_path": "data/metadata/trait_prevalence.parquet",
        "heritability_path": "data/metadata/trait_heritability.parquet",
        "n_prevalence_traits": prevalence_df.height,
        "n_heritability_rows": heritability_df.height,
        "n_heritability_efo_traits": heritability_df["efo_id"].n_unique(),
        **alias_coverage_metadata(prevalence_df, prefix="prevalence"),
        **alias_coverage_metadata(heritability_df, prefix="heritability"),
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
