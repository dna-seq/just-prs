"""Dagster Software-Defined Assets for the PRS reference distribution pipeline.

This pipeline computes population-level Polygenic Risk Score (PRS) distributions
using a reference panel (default: 1000 Genomes) and PGS Catalog scoring files.
The output is a per-score, per-superpopulation percentile table that lets
end-users compare their personal PRS against a global reference.

Asset lineage (left to right):

  [external]                        [download]                         [compute]                     [upload]
  ebi_pgs_catalog_reference_panel    ebi_reference_panel_fingerprint → reference_panel ──────→ reference_scores ──→ hf_prs_percentiles
  ebi_pgs_catalog_scoring_files  →   ebi_scoring_files_fingerprint  → scoring_files → scoring_files_parquet ↗       ↗
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
from typing import Any

import polars as pl
from dagster import (
    AssetDep,
    AssetExecutionContext,
    AssetIn,
    Output,
    SourceAsset,
    asset,
)

from just_prs.ftp import PGS_FTP_BASE, PGS_SCORES_LIST_URL, bulk_download_scoring_files, list_all_pgs_ids
from just_prs.hf import DEFAULT_HF_PERCENTILES_REPO, push_reference_distributions
from just_prs.reference import (
    DEFAULT_PANEL,
    REFERENCE_PANELS,
    REFERENCE_PANEL_URL,
    compute_reference_prs_batch,
    download_reference_panel,
    enrich_distributions,
)
from prs_pipeline.fingerprint import fingerprint_http_resource
from prs_pipeline.resources import CacheDirResource, HuggingFaceResource
from prs_pipeline.runtime import resource_tracker


def _no_cache() -> bool:
    """Return True if PRS_PIPELINE_NO_CACHE is set (user passed --no-cache)."""
    return os.environ.get("PRS_PIPELINE_NO_CACHE", "").strip().lower() in {"1", "true", "yes"}


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


@asset(
    group_name="download",
    description=(
        "Computes a stable fingerprint for the remote reference panel tarball "
        "from HTTP metadata (ETag / Last-Modified / Content-Length). "
        "Downstream assets depend on this fingerprint for freshness checks "
        "without depending directly on SourceAssets."
    ),
)
def ebi_reference_panel_fingerprint(context: AssetExecutionContext) -> Output[str]:
    """Fingerprint the remote reference panel artifact for dependency tracking."""
    panel = os.environ.get("PRS_PIPELINE_PANEL", DEFAULT_PANEL)
    panel_url = REFERENCE_PANELS[panel]["url"]
    fingerprint, metadata = fingerprint_http_resource(panel_url)
    context.add_output_metadata({"panel": panel, **metadata})
    return Output(fingerprint)


@asset(
    group_name="download",
    description=(
        "Computes a fingerprint for the remote PGS scoring manifest "
        "(pgs_scores_list.txt) using HTTP metadata plus body SHA256. "
        "Used as a freshness dependency for metadata/scoring assets."
    ),
)
def ebi_scoring_files_fingerprint(context: AssetExecutionContext) -> Output[str]:
    """Fingerprint the remote scoring index for downstream freshness dependencies."""
    fingerprint, metadata = fingerprint_http_resource(PGS_SCORES_LIST_URL, include_body_hash=True)
    context.add_output_metadata(metadata)
    return Output(fingerprint)

# ---------------------------------------------------------------------------
# Download asset — bulk-fetch all scoring .txt.gz files from EBI FTP
# ---------------------------------------------------------------------------

@asset(
    group_name="download",
    deps=[AssetDep("ebi_scoring_files_fingerprint")],
    description=(
        "Bulk-downloads all harmonized PGS scoring files (.txt.gz) from the "
        "EBI FTP server into the local cache. Uses direct HTTPS URLs (no REST "
        "API calls) with concurrent connections. Files that already exist on "
        "disk are skipped. The reference_scores asset depends on this to "
        "ensure all scoring files are available before computation begins."
    ),
)
def scoring_files(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[int]:
    """Download all PGS scoring .txt.gz files from EBI FTP."""
    cache_dir = cache_dir_resource.get_path()
    scores_dir = cache_dir / "scores"
    genome_build = "GRCh38"

    context.log.info("Fetching PGS ID list from EBI FTP...")
    pgs_ids = list_all_pgs_ids()
    context.log.info(f"Found {len(pgs_ids)} PGS IDs. Starting bulk download to {scores_dir}")

    def _dagster_progress(payload: dict[str, int]) -> None:
        completed = payload.get("completed", 0)
        total = payload.get("total", 0)
        downloaded = payload.get("downloaded", 0)
        cached = payload.get("cached", 0)
        parquet_cached = payload.get("parquet_cached", 0)
        failed = payload.get("failed", 0)
        percent = (completed / total * 100) if total > 0 else 0
        context.log.info(
            f"Download progress: {completed}/{total} ({percent:.1f}%), "
            f"downloaded={downloaded}, cached={cached}, parquet_cached={parquet_cached}, failed={failed}"
        )

    with resource_tracker("scoring_files", context=context):
        result = bulk_download_scoring_files(
            pgs_ids=pgs_ids,
            output_dir=scores_dir,
            genome_build=genome_build,
            progress_callback=_dagster_progress,
        )

    context.log.info(
        f"Bulk download complete: {result.downloaded} downloaded, "
        f"{result.cached} cached, {result.parquet_cached} parquet_cached, "
        f"{result.failed} failed (out of {result.total} total)."
    )
    if result.failed_ids:
        context.log.warning(
            f"Failed to download {result.failed} scoring files: "
            f"{result.failed_ids[:20]}{'...' if result.failed > 20 else ''}"
        )

    n_ok = result.total - result.failed
    coverage_ratio = n_ok / result.total if result.total > 0 else 0.0
    context.add_output_metadata({
        "n_total": result.total,
        "n_ok": n_ok,
        "n_failed": result.failed,
        "n_cached": result.cached + result.parquet_cached,
        "coverage_ratio": round(coverage_ratio, 4),
        "downloaded": result.downloaded,
        "cached": result.cached,
        "parquet_cached": result.parquet_cached,
        "scores_dir": str(scores_dir),
        "genome_build": genome_build,
    })
    return Output(n_ok)


# ---------------------------------------------------------------------------
# Compute asset — convert scoring .txt.gz files to parquet caches
# ---------------------------------------------------------------------------

@asset(
    group_name="compute",
    deps=[AssetDep("scoring_files")],
    description=(
        "Converts all downloaded PGS scoring .txt.gz files to parquet caches "
        "with spec-driven schema overrides and zstd-9 compression. "
        "By default keeps the original .txt.gz files; set env var "
        "PRS_PIPELINE_DELETE_GZ=1 to delete them after verified conversion "
        "(saves ~5.5 GB for the full catalog). "
        "Failures are tracked per-file without aborting the loop and written "
        "to a conversion_failures.parquet report for post-hoc error analysis. "
        "The reference_scores asset depends on this to ensure all scoring "
        "files are available as parquet before batch computation begins."
    ),
)
def scoring_files_parquet(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[int]:
    """Convert all scoring .txt.gz files to parquet caches."""
    import datetime

    from eliot import log_message as _log

    from just_prs.scoring import _scoring_parquet_cache_path, parse_scoring_file

    cache_dir = cache_dir_resource.get_path()
    scores_dir = cache_dir / "scores"
    delete_gz = os.environ.get("PRS_PIPELINE_DELETE_GZ", "0") == "1"

    if delete_gz:
        context.log.info("PRS_PIPELINE_DELETE_GZ=1: will delete .txt.gz after verified conversion")

    gz_files = sorted(scores_dir.glob("*_hmPOS_*.txt.gz"))
    total = len(gz_files)
    context.log.info(f"Found {total} scoring .txt.gz files in {scores_dir}")

    converted = 0
    already_cached = 0
    failed = 0
    deleted_gz_count = 0
    total_gz_bytes = 0
    total_parquet_bytes = 0
    failures: list[dict[str, str]] = []
    progress_every = max(total // 20, 50)
    
    import time
    last_log_time = time.monotonic()

    force = _no_cache()
    if force:
        context.log.info("NO-CACHE: will re-parse all .txt.gz files even if parquet cache exists.")

    with resource_tracker("scoring_files_parquet", context=context):
        for i, gz_path in enumerate(gz_files):
            pgs_id = gz_path.name.split("_hmPOS_")[0] if "_hmPOS_" in gz_path.name else gz_path.stem
            parquet_path = _scoring_parquet_cache_path(gz_path)
            gz_size = gz_path.stat().st_size
            total_gz_bytes += gz_size

            if parquet_path.exists() and not force:
                already_cached += 1
                total_parquet_bytes += parquet_path.stat().st_size
                if delete_gz and gz_path.exists():
                    gz_path.unlink()
                    deleted_gz_count += 1
            else:
                try:
                    lf = parse_scoring_file(gz_path)
                    row_count = lf.select(pl.len()).collect().item()

                    if not parquet_path.exists() or row_count == 0:
                        raise RuntimeError(
                            f"Parquet not created or empty after parse (rows={row_count})"
                        )
                    total_parquet_bytes += parquet_path.stat().st_size
                    converted += 1

                    if delete_gz:
                        gz_path.unlink()
                        deleted_gz_count += 1
                except Exception as exc:
                    failed += 1
                    failures.append({
                        "pgs_id": pgs_id,
                        "gz_path": str(gz_path),
                        "error": str(exc),
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    })
                    _log(
                        message_type="scoring:parquet_conversion_failed",
                        pgs_id=pgs_id,
                        error=str(exc),
                    )

            current_time = time.monotonic()
            should_log = ((i + 1) % progress_every == 0) or (current_time - last_log_time > 15.0) or ((i + 1) == total)
            
            if should_log:
                last_log_time = current_time
                context.log.info(
                    f"Parquet conversion: {i + 1}/{total} "
                    f"(converted={converted}, cached={already_cached}, failed={failed}, "
                    f"deleted_gz={deleted_gz_count})"
                )

    failure_report_path = ""
    if failures:
        failure_df = pl.DataFrame(failures)
        report_path = scores_dir / "conversion_failures.parquet"
        failure_df.write_parquet(report_path)
        failure_report_path = str(report_path)
        context.log.warning(
            f"{failed} conversions failed. Report: {report_path}"
        )

    n_ok = converted + already_cached
    coverage_ratio = n_ok / total if total > 0 else 0.0
    context.add_output_metadata({
        "n_total": total,
        "n_ok": n_ok,
        "n_failed": failed,
        "n_cached": already_cached,
        "coverage_ratio": round(coverage_ratio, 4),
        "converted": converted,
        "deleted_gz_files": deleted_gz_count,
        "delete_gz_enabled": delete_gz,
        "failure_report_path": failure_report_path,
        "total_gz_bytes": total_gz_bytes,
        "total_parquet_bytes": total_parquet_bytes,
        "compression_ratio": round(total_parquet_bytes / max(total_gz_bytes, 1), 3),
        "scores_dir": str(scores_dir),
    })
    return Output(n_ok)


# ---------------------------------------------------------------------------
# Download asset — fetch reference panel into local cache
# ---------------------------------------------------------------------------

@asset(
    group_name="download",
    deps=[AssetDep("ebi_reference_panel_fingerprint")],
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
    deps=[AssetDep("scoring_files_parquet")],
    description=(
        "Scores all PGS IDs (or a test subset) against the reference panel in a "
        "single process using pgenlib + polars. Depends on the scoring_files_parquet "
        "asset which converts all .txt.gz files to parquet first, so this asset "
        "reads pre-parsed parquet caches instead of decompressing gzip on every ID. "
        "Failures are logged and tracked but do not abort the batch. "
        "Produces per-sample scores, aggregated per-superpopulation distribution "
        "statistics, and a quality report. "
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
    import psutil

    cache_dir = cache_dir_resource.get_path()
    panel = os.environ.get("PRS_PIPELINE_PANEL", DEFAULT_PANEL)
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()

    context.log.info("Fetching PGS ID list from EBI FTP...")
    all_pgs_ids = list_all_pgs_ids()
    context.log.info(f"Fetched {len(all_pgs_ids)} PGS IDs from EBI catalog.")

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
    progress_every_raw = os.environ.get("PRS_PIPELINE_PROGRESS_EVERY", "10").strip()
    progress_every = max(int(progress_every_raw), 1) if progress_every_raw else 10
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)

    scores_dir = cache_dir / "reference_scores" / panel
    existing_cached = 0
    if scores_dir.exists():
        existing_cached = sum(1 for p in scores_dir.rglob("scores.parquet") if p.is_file())

    context.log.info(
        f"Batch scoring {len(pgs_ids)} PGS IDs against {panel} panel. "
        f"Found {existing_cached} already cached scores on disk."
    )

    def _dagster_progress(payload: dict[str, Any]) -> None:
        nonlocal peak_rss_mb
        processed = int(payload.get("processed", 0))
        total = int(payload.get("total", 0))
        if total <= 0:
            return

        rss_mb = process.memory_info().rss / (1024 * 1024)
        peak_rss_mb = max(peak_rss_mb, rss_mb)
        cpu_percent = process.cpu_percent(interval=None)
        last_status = str(payload.get("last_status", "n/a"))

        if last_status == "panel_resolved":
            panel_sec = payload.get("panel_resolve_sec", "?")
            pvar_variants = payload.get("pvar_variants", "?")
            n_samples = payload.get("n_samples", "?")
            context.log.info(
                f"Panel resolved in {panel_sec}s: "
                f"{pvar_variants} variants, {n_samples} samples. "
                f"rss={rss_mb:.1f}MB, cpu={cpu_percent:.1f}%. "
                f"Starting scoring loop for {total} PGS IDs..."
            )
            return

        percent = 100.0 * processed / total
        ok = int(payload.get("ok", 0))
        failed = int(payload.get("failed", 0))
        problematic = int(payload.get("problematic", 0))
        cached = int(payload.get("cached", 0))
        rate_per_sec = float(payload.get("rate_per_sec", 0.0))
        eta_sec = payload.get("eta_sec")
        eta_text = "unknown"
        if isinstance(eta_sec, (int, float)):
            eta_text = f"{int(eta_sec)}s"
        recent_ids: list[str] = payload.get("recent_ids", [])
        ids_text = ",".join(recent_ids) if recent_ids else str(payload.get("last_pgs_id", "n/a"))
        context.log.info(
            "Scoring progress: "
            f"{processed}/{total} ({percent:.1f}%), "
            f"ok={ok}, failed={failed}, problematic={problematic}, cached={cached}, "
            f"rate={rate_per_sec:.2f}/s, eta={eta_text}, "
            f"rss={rss_mb:.1f}MB, peak_rss={peak_rss_mb:.1f}MB, cpu={cpu_percent:.1f}%, "
            f"ids=[{ids_text}] ({last_status})"
        )

    skip_existing = not _no_cache()
    if not skip_existing:
        context.log.info("NO-CACHE: will recompute all PGS scores, ignoring cached results.")

    with resource_tracker("reference_scores", context=context):
        result = compute_reference_prs_batch(
            pgs_ids=pgs_ids,
            ref_dir=ref_dir,
            cache_dir=cache_dir,
            genome_build="GRCh38",
            panel=panel,
            skip_existing=skip_existing,
            output_subdir=output_subdir,
            progress_callback=_dagster_progress,
            progress_every=progress_every,
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

    n_total = len(pgs_ids)
    n_cached = sum(1 for o in result.outcomes if o.status == "ok" and o.elapsed_sec is None)
    coverage_ratio = n_ok / n_total if n_total > 0 else 0.0
    context.add_output_metadata({
        "panel": panel,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_cached": n_cached,
        "n_problematic": n_problematic,
        "coverage_ratio": round(coverage_ratio, 4),
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

    min_coverage_str = os.environ.get("PRS_PIPELINE_MIN_COVERAGE", "0.90").strip()
    min_coverage = float(min_coverage_str) if min_coverage_str else 0.90

    with resource_tracker("hf_prs_percentiles", context=context):
        enriched = enrich_distributions(distributions, metadata_dir)
        n_metadata_cols = len(enriched.columns) - len(distributions.columns)
        context.log.info(
            f"Enriched distributions with {n_metadata_cols} metadata columns "
            f"({distributions.height} rows, {len(enriched.columns)} total columns)."
        )

        n_scored = enriched["pgs_id"].n_unique() if enriched.height > 0 else 0
        try:
            all_pgs_ids = list_all_pgs_ids()
            n_catalog_total = len(all_pgs_ids)
        except Exception as exc:
            context.log.warning(f"Could not fetch EBI catalog for coverage check: {exc}")
            n_catalog_total = n_scored

        coverage_ratio = n_scored / n_catalog_total if n_catalog_total > 0 else 0.0
        n_missing = n_catalog_total - n_scored
        if coverage_ratio < min_coverage:
            context.log.warning(
                f"Coverage below threshold: {n_scored}/{n_catalog_total} "
                f"({coverage_ratio:.1%} < {min_coverage:.0%}). "
                f"Missing {n_missing} PGS IDs. Uploading partial data anyway."
            )
        else:
            context.log.info(
                f"Coverage check passed: {n_scored}/{n_catalog_total} ({coverage_ratio:.1%})."
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
                "n_scored": n_scored,
                "n_catalog_total": n_catalog_total,
                "n_missing": n_missing,
                "coverage_ratio": round(coverage_ratio, 4),
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
        "n_scored": n_scored,
        "n_catalog_total": n_catalog_total,
        "n_missing": n_missing,
        "coverage_ratio": round(coverage_ratio, 4),
    })
    return Output(url)
