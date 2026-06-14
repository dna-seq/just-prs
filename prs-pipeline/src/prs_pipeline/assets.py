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

import json
import os
from datetime import datetime, timezone
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

from just_prs.chip_coverage import CHIPS, CHIPS_BY_ID, compute_chip_coverage
from just_prs.ftp import PGS_FTP_BASE, PGS_SCORES_LIST_URL, bulk_download_scoring_files, list_all_pgs_ids
from just_prs.hf import (
    DEFAULT_HF_PERCENTILES_REPO,
    pull_reference_distributions,
    push_reference_audit_sidecars,
    push_chip_coverage,
    push_reference_distributions,
)
from just_prs.reference import (
    DEFAULT_PANEL,
    REFERENCE_PANELS,
    REFERENCE_PANEL_URL,
    compute_reference_prs_batch,
    download_reference_panel,
    enrich_distributions,
    reference_distribution_audit_issues,
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

illumina_gsa_manifest = SourceAsset(
    key="illumina_gsa_manifest",
    group_name="external",
    description=(
        "Illumina Global Screening Array v3.0 manifest (A2 = GRCh38 coordinates, "
        "~70 MB zip, ~648K typed markers). The GSA is the platform underlying "
        "23andMe v5 and AncestryDNA v2 consumer genotyping chips. Used to compute, "
        "per PGS score, how many variants are directly typed on the array vs require "
        "imputation. Coordinates are GRCh38 so they intersect the harmonized GRCh38 "
        "scoring files without liftover."
    ),
    metadata={"url": CHIPS_BY_ID["gsa_v3"]["manifest_url"]},
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

            if gz_size == 0:
                failures.append({
                    "pgs_id": pgs_id,
                    "gz_path": str(gz_path),
                    "error": f"Scoring file {gz_path} is 0 bytes (corrupt download)",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
                failed += 1
                gz_path.unlink()
                continue

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
# Compute asset — consumer-chip coverage of scoring files
# ---------------------------------------------------------------------------

@asset(
    group_name="compute",
    deps=[AssetDep("scoring_files_parquet")],
    description=(
        "Computes, per PGS score and per consumer genotyping chip (Illumina GSA "
        "v3 = 23andMe v5 / AncestryDNA v2 platform), how many of the score's "
        "variants are directly typed on the array vs require imputation. "
        "Intersects the GRCh38 GSA manifest positions against the GRCh38 "
        "harmonized scoring parquets (the same hm_chr/hm_pos used by compute_prs). "
        "Output chip_coverage.parquet lets the UI label each PRS 'array-ready' or "
        "'imputation-required' for a given chip before a user even uploads data."
    ),
)
def chip_coverage(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
) -> Output[pl.DataFrame]:
    """Compute per-PGS, per-chip direct-typing coverage and cache the parquet."""
    cache_dir = cache_dir_resource.get_path()
    scores_dir = cache_dir / "scores"
    coverage_dir = cache_dir / "percentiles"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    out_path = coverage_dir / "chip_coverage.parquet"

    def _progress(payload: dict[str, int]) -> None:
        context.log.info(
            f"Chip coverage: chip {payload['chip_index'] + 1}/{payload['n_chips']}, "
            f"{payload['completed']}/{payload['total']} scores"
        )

    with resource_tracker("chip_coverage", context=context):
        df = compute_chip_coverage(
            scores_dir=scores_dir,
            cache_dir=cache_dir,
            chips=CHIPS,
            progress_callback=_progress,
        )
        df.write_parquet(out_path)

    n_scores = df["pgs_id"].n_unique() if df.height > 0 else 0
    # Per-chip summary: how many scores are array-ready (array_ready bool) vs need imputation.
    array_ready_by_chip: dict[str, int] = {}
    for chip in df["chip"].unique().to_list() if df.height > 0 else []:
        chip_df = df.filter(pl.col("chip") == chip)
        array_ready_by_chip[chip] = int(chip_df.filter(pl.col("array_ready")).height)

    context.add_output_metadata({
        "n_rows": df.height,
        "n_scores": n_scores,
        "n_chips": len(CHIPS),
        "chips": ", ".join(c["chip"] for c in CHIPS),
        "median_coverage_ratio": round(float(df["coverage_ratio"].median()), 4) if df.height > 0 else 0.0,
        "array_ready_by_chip": str(array_ready_by_chip),
        "n_array_ready_total": int(df.filter(pl.col("array_ready")).height) if df.height > 0 else 0,
        "coverage_path": str(out_path),
    })
    return Output(df)


# ---------------------------------------------------------------------------
# Upload asset — publish chip coverage to HuggingFace
# ---------------------------------------------------------------------------

@asset(
    group_name="upload",
    ins={"coverage": AssetIn("chip_coverage")},
    description=(
        "Uploads chip_coverage.parquet to the HuggingFace dataset "
        "just-dna-seq/prs-percentiles. Published so the just-prs UI can label, per "
        "consumer chip, which PRS models are usable on raw array data without "
        "imputation. Named after the destination per lineage convention."
    ),
)
def hf_chip_coverage(
    context: AssetExecutionContext,
    coverage: pl.DataFrame,
    cache_dir_resource: CacheDirResource,
    hf_resource: HuggingFaceResource,
) -> Output[str]:
    """Publish the chip coverage parquet to HuggingFace."""
    repo_id = hf_resource.percentiles_repo
    token = hf_resource.get_token()
    cache_dir = cache_dir_resource.get_path()
    parquet_path = cache_dir / "percentiles" / "chip_coverage.parquet"

    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
    if test_spec:
        context.log.info("TEST MODE: skipping HuggingFace push of chip_coverage.parquet.")
        context.add_output_metadata({
            "test_mode": True, "hf_push_skipped": True,
            "coverage_path": str(parquet_path), "n_rows": coverage.height,
        })
        return Output(str(parquet_path))

    with resource_tracker("hf_chip_coverage", context=context):
        push_chip_coverage(parquet_path=parquet_path, repo_id=repo_id, token=token)

    url = f"https://huggingface.co/datasets/{repo_id}"
    context.log.info(f"Pushed chip_coverage.parquet to {url}")
    context.add_output_metadata({
        "repo_id": repo_id, "url": url,
        "coverage_path": str(parquet_path),
        "n_rows": coverage.height,
        "n_scores": coverage["pgs_id"].n_unique() if coverage.height > 0 else 0,
    })
    return Output(url)


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
        f"Distributions: {percentiles_dir / f'{panel}_distributions.parquet'}  "
        f"Distribution issues: {percentiles_dir / f'{panel}_distribution_quality_issues.parquet'}"
    )
    n_distribution_errors = result.distribution_issues_df.filter(pl.col("severity") == "ERROR").height
    n_distribution_warnings = result.distribution_issues_df.filter(pl.col("severity") == "WARN").height
    if result.distribution_issues_df.height > 0:
        examples = result.distribution_issues_df.select(
            "pgs_id", "superpopulation", "severity", "issue", "recommended_action"
        ).head(20).to_dicts()
        context.log.warning(
            "Distribution-quality issues detected: "
            f"errors={n_distribution_errors}, warnings={n_distribution_warnings}, "
            f"examples={examples}"
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
        "n_distribution_quality_errors": n_distribution_errors,
        "n_distribution_quality_warnings": n_distribution_warnings,
        "coverage_ratio": round(coverage_ratio, 4),
        "n_distributions": result.distributions_df.height,
        "n_pgs_with_distributions": result.distributions_df["pgs_id"].n_unique() if result.distributions_df.height > 0 else 0,
        "per_pgs_scores_dir": str(scores_dir),
        "percentiles_dir": str(percentiles_dir),
        "distribution_quality_issues_path": str(percentiles_dir / f"{panel}_distribution_quality_issues.parquet"),
        "engine": "polars",
        "test_mode": is_test,
    })
    return Output(result.distributions_df)


# ---------------------------------------------------------------------------
# Helper — enrich distributions with precomputed absolute risk
# ---------------------------------------------------------------------------


def _enrich_with_absolute_risk(
    enriched: pl.DataFrame,
    metadata_dir: Path,
    context: "AssetExecutionContext",
) -> pl.DataFrame:
    """Add absolute risk columns for key percentile z-scores.

    For each row (pgs_id × superpopulation), computes absolute risk at the
    mean z-score (0.0) using whichever effect size is available (OR preferred,
    then AUROC). Requires trait_prevalence.parquet in metadata_dir.
    """
    from just_prs.absolute_risk import estimate_absolute_risk

    prevalence_path = metadata_dir / "trait_prevalence.parquet"
    if not prevalence_path.exists():
        context.log.info("No trait_prevalence.parquet found; skipping absolute risk enrichment.")
        return enriched

    if "trait_efo_id" not in enriched.columns:
        context.log.info("No trait_efo_id column in enriched data; skipping absolute risk enrichment.")
        return enriched

    prevalence_df = pl.read_parquet(prevalence_path)
    if prevalence_df.height == 0:
        context.log.info("Empty prevalence table; skipping absolute risk enrichment.")
        return enriched

    prev_map: dict[str, tuple[float, str, str, str]] = {}
    for row in prevalence_df.iter_rows(named=True):
        efo_id = row.get("efo_id")
        prev = row.get("prevalence")
        if efo_id and prev and 0 < prev < 1.0:
            prev_map[efo_id] = (
                float(prev),
                row.get("source", ""),
                row.get("prevalence_type", "lifetime"),
                row.get("confidence", "moderate"),
            )

    abs_risk_at_mean: list[float | None] = []
    abs_risk_method: list[str | None] = []
    abs_risk_prevalence: list[float | None] = []

    for row in enriched.iter_rows(named=True):
        efo_ids_raw = row.get("trait_efo_id", "")
        or_est = row.get("or_estimate")
        auroc_est = row.get("auroc_estimate")

        if not efo_ids_raw:
            abs_risk_at_mean.append(None)
            abs_risk_method.append(None)
            abs_risk_prevalence.append(None)
            continue

        efo_ids = [e.strip() for e in str(efo_ids_raw).split(",")]
        prev_data = None
        for eid in efo_ids:
            if eid in prev_map:
                prev_data = prev_map[eid]
                break

        if prev_data is None:
            abs_risk_at_mean.append(None)
            abs_risk_method.append(None)
            abs_risk_prevalence.append(None)
            continue

        prevalence, prev_source, prev_type, confidence = prev_data
        result = estimate_absolute_risk(
            z_score=0.0,
            prevalence=prevalence,
            or_estimate=float(or_est) if or_est is not None else None,
            auroc_estimate=float(auroc_est) if auroc_est is not None else None,
            prevalence_source=prev_source,
            prevalence_type=prev_type,
            confidence=confidence,
        )

        if result is not None:
            abs_risk_at_mean.append(result.absolute_risk)
            abs_risk_method.append(result.method)
            abs_risk_prevalence.append(result.population_prevalence)
        else:
            abs_risk_at_mean.append(None)
            abs_risk_method.append(None)
            abs_risk_prevalence.append(None)

    enriched = enriched.with_columns(
        pl.Series("abs_risk_at_mean", abs_risk_at_mean, dtype=pl.Float64),
        pl.Series("abs_risk_method", abs_risk_method, dtype=pl.Utf8),
        pl.Series("abs_risk_prevalence", abs_risk_prevalence, dtype=pl.Float64),
    )

    n_with_risk = sum(1 for v in abs_risk_at_mean if v is not None)
    context.log.info(
        f"Absolute risk enrichment: {n_with_risk}/{enriched.height} rows "
        f"have precomputed absolute risk at mean z-score."
    )
    return enriched


def _write_distribution_audit_summary(
    *,
    issue_df: pl.DataFrame,
    path: Path,
    panel: str,
    input_rows: int,
    published_rows: int,
    input_pgs_ids: int,
    published_pgs_ids: int,
    fully_removed_pgs_ids: list[str],
) -> dict[str, Any]:
    """Write a compact JSON audit summary for the HF percentiles dataset."""
    issue_counts_by_severity = (
        {
            row["severity"]: row["len"]
            for row in issue_df.group_by("severity").len().iter_rows(named=True)
        }
        if issue_df.height > 0
        else {}
    )
    issue_counts_by_type = (
        {
            row["issue"]: row["len"]
            for row in issue_df.group_by("issue").len().iter_rows(named=True)
        }
        if issue_df.height > 0
        else {}
    )
    affected_pgs_ids = sorted(issue_df["pgs_id"].unique().to_list()) if issue_df.height > 0 else []
    quarantined_rows = (
        issue_df.select("pgs_id", "superpopulation").unique().height
        if issue_df.height > 0
        else 0
    )
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel": panel,
        "input_rows": input_rows,
        "published_rows": published_rows,
        "quarantined_distribution_rows": quarantined_rows,
        "input_pgs_ids": input_pgs_ids,
        "published_pgs_ids": published_pgs_ids,
        "fully_removed_pgs_ids": fully_removed_pgs_ids,
        "issue_rows": issue_df.height,
        "issue_counts_by_severity": issue_counts_by_severity,
        "issue_counts_by_type": issue_counts_by_type,
        "affected_pgs_ids": affected_pgs_ids,
        "policy": (
            "Rows with ERROR distribution quality issues are quarantined from the "
            "published distributions parquet. WARN rows remain published but are "
            "retained in the issue parquet as an audit/debugging sidecar."
        ),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


@asset(
    group_name="compute",
    description=(
        "Audit cached or HuggingFace reference percentile distributions without "
        "recomputing reference scores. Writes {panel}_distribution_quality_issues.parquet "
        "and {panel}_distribution_audit_summary.json, using {panel}_quality.parquet "
        "when available to flag missing match metadata, low match rate, stale aggregates, "
        "and other suspicious reference percentile rows."
    ),
)
def reference_percentile_audit(
    context: AssetExecutionContext,
    cache_dir_resource: CacheDirResource,
    hf_resource: HuggingFaceResource,
) -> Output[str]:
    """Run the quality-aware reference percentile audit as a standalone asset."""
    panel = os.environ.get("PRS_PIPELINE_PANEL", DEFAULT_PANEL)
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
    is_test = bool(test_spec)
    cache_dir = cache_dir_resource.get_path()
    percentiles_dir = cache_dir / "percentiles"
    if is_test:
        percentiles_dir = percentiles_dir / "test"
    percentiles_dir.mkdir(parents=True, exist_ok=True)

    dist_path = percentiles_dir / f"{panel}_distributions.parquet"
    quality_path = percentiles_dir / f"{panel}_quality.parquet"
    issue_report_path = percentiles_dir / f"{panel}_distribution_quality_issues.parquet"
    audit_summary_path = percentiles_dir / f"{panel}_distribution_audit_summary.json"

    with resource_tracker("reference_percentile_audit", context=context):
        if not dist_path.exists() and not is_test:
            context.log.info(
                f"No local {dist_path.name}; pulling reference distributions from {hf_resource.percentiles_repo}."
            )
            pull_reference_distributions(
                percentiles_dir,
                repo_id=hf_resource.percentiles_repo,
                token=hf_resource.get_token(),
                panel=panel,
            )

        if not dist_path.exists():
            raise FileNotFoundError(
                f"Reference distributions not found: {dist_path}. "
                "Run reference_scores/hf_prs_percentiles first or pull from HuggingFace."
            )

        distributions = pl.read_parquet(dist_path)
        quality_df = pl.read_parquet(quality_path) if quality_path.exists() else None
        issue_df = reference_distribution_audit_issues(distributions, quality_df)
        issue_df.write_parquet(issue_report_path)

        error_issue_df = issue_df.filter(pl.col("severity") == "ERROR")
        warning_issue_df = issue_df.filter(pl.col("severity") == "WARN")
        affected_error_ids = (
            sorted(error_issue_df["pgs_id"].unique().to_list())
            if error_issue_df.height > 0
            else []
        )
        affected_warning_ids = (
            sorted(warning_issue_df["pgs_id"].unique().to_list())
            if warning_issue_df.height > 0
            else []
        )
        published_pgs_ids = distributions["pgs_id"].n_unique() if distributions.height > 0 else 0
        warn_only_ids = sorted(set(affected_warning_ids) - set(affected_error_ids))
        passed_pgs_ids = max(published_pgs_ids - len(set(affected_error_ids) | set(affected_warning_ids)), 0)
        audit_summary = _write_distribution_audit_summary(
            issue_df=issue_df,
            path=audit_summary_path,
            panel=panel,
            input_rows=distributions.height,
            published_rows=distributions.height,
            input_pgs_ids=published_pgs_ids,
            published_pgs_ids=published_pgs_ids,
            fully_removed_pgs_ids=affected_error_ids,
        )

        context.log.info(
            "Reference percentile audit summary: "
            f"passed={passed_pgs_ids}, warnings={len(warn_only_ids)}, failed={len(affected_error_ids)}, "
            f"issue_rows={issue_df.height}, error_rows={error_issue_df.height}, warning_rows={warning_issue_df.height}."
        )
        if issue_df.height > 0:
            issue_counts = audit_summary["issue_counts_by_type"]
            context.log.info(f"Reference percentile audit issue counts: {issue_counts}")
            examples = issue_df.select(
                "pgs_id", "superpopulation", "severity", "issue", "recommended_action"
            ).head(20).to_dicts()
            context.log.info(f"Reference percentile audit examples: {examples}")

        hf_audit_uploaded = False
        if not is_test:
            hf_token = hf_resource.get_token()
            if hf_token:
                push_reference_audit_sidecars(
                    quality_report_path=quality_path if quality_path.exists() else None,
                    issue_report_path=issue_report_path,
                    audit_summary_path=audit_summary_path,
                    repo_id=hf_resource.percentiles_repo,
                    token=hf_token,
                    panel=panel,
                )
                hf_audit_uploaded = True
                context.log.info(
                    f"Uploaded reference percentile audit sidecars to HuggingFace dataset "
                    f"{hf_resource.percentiles_repo}."
                )
            else:
                context.log.warning(
                    "HF_TOKEN is not set; reference percentile audit sidecars were written locally "
                    "but not uploaded to HuggingFace."
                )

    context.add_output_metadata({
        "panel": panel,
        "test_mode": is_test,
        "hf_audit_uploaded": hf_audit_uploaded,
        "hf_repo": hf_resource.percentiles_repo,
        "distributions_path": str(dist_path),
        "quality_path": str(quality_path) if quality_path.exists() else "",
        "distribution_quality_issues_path": str(issue_report_path),
        "distribution_audit_summary_path": str(audit_summary_path),
        "n_distribution_rows": distributions.height,
        "n_pgs_ids": published_pgs_ids,
        "n_passed_pgs_ids": passed_pgs_ids,
        "n_warning_pgs_ids": len(warn_only_ids),
        "n_failed_pgs_ids": len(affected_error_ids),
        "n_issue_rows": issue_df.height,
        "n_error_rows": error_issue_df.height,
        "n_warning_rows": warning_issue_df.height,
        "n_error_pgs_ids": len(affected_error_ids),
        "issue_counts_by_type": audit_summary["issue_counts_by_type"],
    })
    return Output(str(audit_summary_path))


# ---------------------------------------------------------------------------
# Upload asset — push to HuggingFace (represents the HF dataset as an asset)
# ---------------------------------------------------------------------------

@asset(
    group_name="upload",
    ins={"distributions": AssetIn("reference_scores")},
    deps=[AssetDep("cleaned_pgs_metadata"), AssetDep("trait_prevalence")],
    description=(
        "Enriches the raw distribution statistics with cleaned PGS Catalog metadata "
        "(trait names, EFO terms, best performance metrics like AUROC/OR/C-index, "
        "ancestry info) and precomputed absolute risk estimates at key percentile "
        "buckets, then uploads the enriched parquet to the HuggingFace dataset "
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
        issue_report_path = percentiles_dir / f"{panel}_distribution_quality_issues.parquet"
        audit_summary_path = percentiles_dir / f"{panel}_distribution_audit_summary.json"
        input_rows = distributions.height
        input_pgs_ids_set = set(distributions["pgs_id"].unique().to_list()) if distributions.height > 0 else set()
        quality_path = percentiles_dir / f"{panel}_quality.parquet"
        quality_df = pl.read_parquet(quality_path) if quality_path.exists() else None
        issue_df = reference_distribution_audit_issues(distributions, quality_df)
        issue_df.write_parquet(issue_report_path)
        n_quarantined_rows = 0
        n_error_rows = issue_df.filter(pl.col("severity") == "ERROR").height
        n_warn_rows = issue_df.filter(pl.col("severity") == "WARN").height
        error_issue_df = issue_df.filter(pl.col("severity") == "ERROR")
        if error_issue_df.height > 0:
            bad_ids = error_issue_df.select("pgs_id").unique()
            bad_keys = distributions.join(bad_ids, on="pgs_id", how="semi").select(
                "pgs_id", "superpopulation"
            ).unique()
            n_quarantined_rows = bad_keys.height
            distributions = distributions.join(
                bad_ids,
                on="pgs_id",
                how="anti",
            )
            context.log.warning(
                "Quarantined untrustworthy reference distribution rows before publication: "
                f"{bad_keys.height} unique pgs_id/superpopulation rows "
                f"({n_error_rows} error issue rows; {n_warn_rows} warning issue rows retained in sidecar). "
                f"Full issue report: {issue_report_path}"
            )

        published_pgs_ids_set = set(distributions["pgs_id"].unique().to_list()) if distributions.height > 0 else set()
        fully_removed_pgs_ids = sorted(input_pgs_ids_set - published_pgs_ids_set)
        audit_summary = _write_distribution_audit_summary(
            issue_df=issue_df,
            path=audit_summary_path,
            panel=panel,
            input_rows=input_rows,
            published_rows=distributions.height,
            input_pgs_ids=len(input_pgs_ids_set),
            published_pgs_ids=len(published_pgs_ids_set),
            fully_removed_pgs_ids=fully_removed_pgs_ids,
        )

        enriched = enrich_distributions(distributions, metadata_dir)
        n_metadata_cols = len(enriched.columns) - len(distributions.columns)
        context.log.info(
            f"Enriched distributions with {n_metadata_cols} metadata columns "
            f"({distributions.height} rows, {len(enriched.columns)} total columns)."
        )

        enriched = _enrich_with_absolute_risk(enriched, metadata_dir, context)

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
                f"  Quality report: {percentiles_dir / f'{panel}_quality.parquet'}\n"
                f"  Distribution issues: {percentiles_dir / f'{panel}_distribution_quality_issues.parquet'}\n"
                f"  Audit summary: {audit_summary_path}"
            )
            context.add_output_metadata({
                "panel": panel,
                "test_mode": True,
                "hf_push_skipped": True,
                "distributions_path": str(parquet_path),
                "quality_path": str(percentiles_dir / f"{panel}_quality.parquet"),
                "distribution_quality_issues_path": str(percentiles_dir / f"{panel}_distribution_quality_issues.parquet"),
                "distribution_audit_summary_path": str(audit_summary_path),
                "n_quarantined_distribution_rows": n_quarantined_rows,
                "n_distribution_quality_error_rows": n_error_rows,
                "n_distribution_quality_warning_rows": n_warn_rows,
                "n_distribution_quality_issue_rows": audit_summary["issue_rows"],
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
            quality_report_path=percentiles_dir / f"{panel}_quality.parquet",
            issue_report_path=issue_report_path,
            audit_summary_path=audit_summary_path,
        )

    url = f"https://huggingface.co/datasets/{repo_id}"
    context.log.info(f"Pushed {panel}_distributions.parquet to {url}")
    context.log.info(
        f"Results saved locally:\n"
        f"  Distributions: {parquet_path}\n"
        f"  Quality report: {percentiles_dir / f'{panel}_quality.parquet'}\n"
        f"  Distribution issues: {percentiles_dir / f'{panel}_distribution_quality_issues.parquet'}\n"
        f"  Audit summary: {audit_summary_path}\n"
        f"  HuggingFace: {url}"
    )
    context.add_output_metadata({
        "repo_id": repo_id,
        "url": url,
        "panel": panel,
        "distributions_path": str(parquet_path),
        "quality_path": str(percentiles_dir / f"{panel}_quality.parquet"),
        "distribution_quality_issues_path": str(percentiles_dir / f"{panel}_distribution_quality_issues.parquet"),
        "distribution_audit_summary_path": str(audit_summary_path),
        "n_quarantined_distribution_rows": n_quarantined_rows,
        "n_distribution_quality_error_rows": n_error_rows,
        "n_distribution_quality_warning_rows": n_warn_rows,
        "n_distribution_quality_issue_rows": audit_summary["issue_rows"],
        "n_rows": enriched.height,
        "n_columns": len(enriched.columns),
        "n_metadata_columns_added": n_metadata_cols,
        "n_scored": n_scored,
        "n_catalog_total": n_catalog_total,
        "n_missing": n_missing,
        "coverage_ratio": round(coverage_ratio, 4),
    })
    return Output(url)
