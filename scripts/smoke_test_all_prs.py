#!/usr/bin/env python3
"""Smoke test: compute PRS for all catalog scores against every VCF in data/input/.

Replicates the full UI compute flow via ``enrich_prs_result()``:
  normalize VCF -> detect build -> scan genotypes as LazyFrame ->
  compute each PRS -> enrich with percentile, quality, absolute risk,
  heritability, reference status, per-population percentiles.

Skips gracefully if data/input/ is empty or missing.
Writes per-genome parquet results to data/output/smoke_test/.

Usage:
    # Run all genomes (skip complete, recompute incomplete/missing)
    uv run python scripts/smoke_test_all_prs.py

    # Run only specific VCF files
    uv run python scripts/smoke_test_all_prs.py john.vcf.gz advocate.vcf.gz

    # Force rerun of specific files (delete cached output first)
    uv run python scripts/smoke_test_all_prs.py --force john.vcf.gz

    # Force rerun everything
    uv run python scripts/smoke_test_all_prs.py --force

    # Targeted retry: rerun specific PGS IDs across all (or specified) VCFs
    # Merges results back into the existing parquet — other cached rows are preserved
    uv run python scripts/smoke_test_all_prs.py --pgs-ids PGS004776,PGS004778,PGS004780

    # Serial mode (no threading) — use to isolate hangs or debug specific IDs
    uv run python scripts/smoke_test_all_prs.py --serial
    uv run python scripts/smoke_test_all_prs.py --serial --pgs-ids PGS004776,PGS004778

    # Custom thread count
    uv run python scripts/smoke_test_all_prs.py --threads 8
"""

import argparse
import json
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as futures_wait
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import polars as pl

from just_prs import PRSCatalog, EnrichedPRSResult, enrich_prs_result, normalize_vcf, VcfFilterConfig
from just_prs.prs import compute_prs
from just_prs.vcf import detect_genome_build

INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output/smoke_test")
NORMALIZED_DIR = OUTPUT_DIR / "normalized"

VCF_EXTENSIONS = (".vcf", ".vcf.gz", ".bgz")
DEFAULT_WORKERS = 16

# Seconds with no completions before printing a "still waiting" heartbeat.
HEARTBEAT_INTERVAL = 30.0
# Seconds a single future may run before it is considered stalled (reported once).
TASK_TIMEOUT = 300.0

# Set by the SIGINT handler; workers check this to exit early.
_SHUTDOWN = threading.Event()


def _install_sigint_handler() -> None:
    """Replace the default SIGINT handler with one that sets _SHUTDOWN and exits cleanly.

    ThreadPoolExecutor swallows KeyboardInterrupt when the main thread is
    blocked inside as_completed(). The custom handler unblocks it by setting
    the event so the iteration loop can bail out, then re-raises after a short
    grace period.
    """
    def _handler(sig: int, frame: Any) -> None:  # type: ignore[type-arg]
        if _SHUTDOWN.is_set():
            print("\n[interrupted] Force exit.", flush=True)
            sys.exit(1)
        _SHUTDOWN.set()
        print("\n[interrupted] Stopping after current batch… (Ctrl-C again to force)", flush=True)

    signal.signal(signal.SIGINT, _handler)


def find_vcfs(input_dir: Path, names: list[str] | None = None) -> list[Path]:
    if not input_dir.is_dir():
        return []
    all_vcfs = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.name.endswith(VCF_EXTENSIONS)
    )
    if not names:
        return all_vcfs
    name_set = set(names)
    matched = [f for f in all_vcfs if f.name in name_set]
    missing = name_set - {f.name for f in matched}
    if missing:
        print(f"  [warn] Not found in {input_dir}/: {', '.join(sorted(missing))}")
    return matched


def _genome_name(vcf_path: Path) -> str:
    return vcf_path.stem.replace(".", "_")


def _result_path(vcf_path: Path) -> Path:
    return OUTPUT_DIR / f"{_genome_name(vcf_path)}_results.parquet"


def _validate_cached_results(result_path: Path, expected_ids: set[str]) -> tuple[bool, str]:
    """Check if cached results are complete and match the current catalog.

    Returns (is_valid, reason).
    """
    if not result_path.exists():
        return False, "missing"
    try:
        df = pl.scan_parquet(result_path).select("pgs_id").collect()
    except Exception as exc:
        return False, f"corrupt ({type(exc).__name__})"
    cached_ids = set(df.get_column("pgs_id").to_list())
    if cached_ids == expected_ids:
        return True, "complete"
    extra = cached_ids - expected_ids
    missing = expected_ids - cached_ids
    parts: list[str] = []
    if missing:
        parts.append(f"{len(missing)} missing")
    if extra:
        parts.append(f"{len(extra)} extra")
    return False, f"id mismatch ({', '.join(parts)}; cached {len(cached_ids)}, expected {len(expected_ids)})"


def normalize_genome(vcf_path: Path, out_dir: Path) -> tuple[Path, str]:
    """Normalize a VCF and detect its genome build."""
    build = detect_genome_build(vcf_path)
    if build is None:
        build = "GRCh38"
        print(f"  [warn] Could not detect build for {vcf_path.name}, defaulting to {build}")

    parquet_path = out_dir / f"{vcf_path.stem.replace('.', '_')}.parquet"
    if parquet_path.exists():
        print(f"  [cache] Already normalized: {parquet_path.name} (build={build})")
        return parquet_path, build

    config = VcfFilterConfig(pass_filters=["PASS", "."])
    normalize_vcf(vcf_path, parquet_path, config=config)
    print(f"  [done] Normalized to {parquet_path.name} (build={build})")
    return parquet_path, build


def _enriched_to_parquet_row(enriched: EnrichedPRSResult, genome: str) -> dict[str, Any]:
    """Convert an EnrichedPRSResult to a flat dict suitable for parquet."""
    d = enriched.model_dump()
    d["genome"] = genome
    d["heritability_metrics"] = json.dumps(d["heritability_metrics"])
    d["risk_estimates_by_method"] = json.dumps(d["risk_estimates_by_method"])
    d["risk_estimate_methods"] = json.dumps(d["risk_estimate_methods"])
    d["error"] = None
    return d


def _error_row(genome: str, pgs_id: str, build: str, error: str) -> dict[str, Any]:
    """Create an error placeholder row with the same schema."""
    d = EnrichedPRSResult(pgs_id=pgs_id, genome_build=build).model_dump()
    d["genome"] = genome
    d["heritability_metrics"] = "[]"
    d["risk_estimates_by_method"] = "{}"
    d["risk_estimate_methods"] = "[]"
    d["error"] = error
    return d


def _process_single_pgs(
    pgs_id: str,
    vcf_path: Path,
    genotypes_lf: pl.LazyFrame,
    catalog: PRSCatalog,
    best_perf_df: pl.DataFrame,
    build: str,
    cache: Path,
    genome_name: str,
) -> dict[str, Any]:
    """Process a single PGS ID: compute + enrich. Thread-safe."""
    info = catalog.score_info_row(pgs_id)
    trait = info["trait_reported"] if info else None

    result = compute_prs(
        vcf_path=vcf_path,
        scoring_file=pgs_id,
        genome_build=build,
        cache_dir=cache,
        pgs_id=pgs_id,
        trait_reported=trait,
        genotypes_lf=genotypes_lf,
    )

    enriched = enrich_prs_result(
        result,
        catalog,
        best_perf_df,
        genome_build=build,
        selected_ancestry="EUR",
        compute_all_populations=True,
    )

    return _enriched_to_parquet_row(enriched, genome_name)


def _run_serial(
    pgs_ids: list[str],
    vcf_path: Path,
    genotypes_lf: pl.LazyFrame,
    catalog: PRSCatalog,
    best_perf_df: pl.DataFrame,
    build: str,
    cache: Path,
    genome_name: str,
    flush: "Callable[[list[dict[str, Any]]], None]",
    debug: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """Run PGS IDs one at a time in the main thread. Returns (rows, n_ok, n_fail)."""
    rows: list[dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    total = len(pgs_ids)
    t0 = time.monotonic()

    for i, pgs_id in enumerate(pgs_ids, 1):
        if _SHUTDOWN.is_set():
            flush(rows)
            break

        if debug:
            print(f"  [{i}/{total}] → {pgs_id} ...", flush=True)
        t_item = time.monotonic()

        try:
            row = _process_single_pgs(
                pgs_id, vcf_path, genotypes_lf, catalog,
                best_perf_df, build, cache, genome_name,
            )
            rows.append(row)
            n_ok += 1
            if debug:
                dt = time.monotonic() - t_item
                score = row.get("score")
                score_str = f"  score={score:.6f}" if score is not None else ""
                print(f"  [{i}/{total}] ✓ {pgs_id} ({dt:.1f}s){score_str}", flush=True)
        except Exception as e:
            rows.append(_error_row(genome_name, pgs_id, build, f"{type(e).__name__}: {e}"))
            n_fail += 1
            dt = time.monotonic() - t_item
            print(f"  [error] {pgs_id} ({dt:.1f}s): {type(e).__name__}: {e}", flush=True)

        milestone = 1 if debug else 50
        if i % milestone == 0 or i == total:
            elapsed = time.monotonic() - t0
            rate = i / elapsed if elapsed > 0 else 0
            if not debug:
                print(f"  [{i}/{total}] {rate:.1f} scores/s", flush=True)
            flush(rows)

    return rows, n_ok, n_fail


def _run_parallel(
    pgs_ids: list[str],
    vcf_path: Path,
    genotypes_lf: pl.LazyFrame,
    catalog: PRSCatalog,
    best_perf_df: pl.DataFrame,
    build: str,
    cache: Path,
    genome_name: str,
    max_workers: int,
    flush: "Callable[[list[dict[str, Any]]], None]",
) -> tuple[list[dict[str, Any]], int, int]:
    """Run PGS IDs in parallel via ThreadPoolExecutor. Returns (rows, n_ok, n_fail)."""
    rows: list[dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    n_done = 0
    total = len(pgs_ids)
    lock = threading.Lock()
    t0 = time.monotonic()

    future_start: dict[Any, float] = {}
    stall_reported: set[Any] = set()
    last_completion_t = time.monotonic()
    last_heartbeat_t = time.monotonic()

    def _progress() -> None:
        nonlocal n_done
        n_done += 1
        if n_done % 200 == 0 or n_done == total:
            elapsed = time.monotonic() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  [{n_done}/{total}] {rate:.1f} scores/s", flush=True)
            flush(rows)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Any, str] = {}
        for pgs_id in pgs_ids:
            f = pool.submit(
                _process_single_pgs,
                pgs_id, vcf_path, genotypes_lf, catalog,
                best_perf_df, build, cache, genome_name,
            )
            futures[f] = pgs_id
            future_start[f] = time.monotonic()

        pending = dict(futures)

        while pending:
            if _SHUTDOWN.is_set():
                for f in pending:
                    f.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                with lock:
                    flush(rows)
                return rows, n_ok, n_fail

            done, _ = futures_wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            now = time.monotonic()

            if done:
                last_completion_t = now
                last_heartbeat_t = now

            for future in done:
                pgs_id = pending.pop(future)
                future_start.pop(future, None)
                try:
                    row = future.result()
                    rows.append(row)
                    n_ok += 1
                except Exception as e:
                    rows.append(_error_row(genome_name, pgs_id, build, f"{type(e).__name__}: {e}"))
                    n_fail += 1
                    if n_fail <= 5:
                        print(f"  [error] {pgs_id}: {type(e).__name__}: {e}", flush=True)
                    elif n_fail == 6:
                        print("  [error] (suppressing further error details)", flush=True)
                with lock:
                    _progress()

            # Heartbeat: periodic progress when nothing is completing.
            if now - last_heartbeat_t >= HEARTBEAT_INTERVAL:
                elapsed = now - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                silence = now - last_completion_t
                print(
                    f"  [heartbeat] {n_done}/{total} done, {len(pending)} pending "
                    f"({rate:.1f} scores/s, {silence:.0f}s since last completion)",
                    flush=True,
                )
                last_heartbeat_t = now
                with lock:
                    flush(rows)

            # Stall detection: report each stalled future exactly once.
            for f in list(pending):
                if f in future_start and f not in stall_reported:
                    age = now - future_start[f]
                    if age > TASK_TIMEOUT:
                        stall_reported.add(f)
                        print(
                            f"  [stall] {pending[f]} has been running for {age:.0f}s — "
                            f"possible hang (use --serial to isolate)",
                            flush=True,
                        )

    return rows, n_ok, n_fail


def run_genome(
    vcf_path: Path,
    catalog: PRSCatalog,
    pgs_ids: list[str],
    best_perf_df: pl.DataFrame,
    build: str,
    force: bool = False,
    serial: bool = False,
    debug: bool = False,
    max_workers: int = DEFAULT_WORKERS,
    targeted_ids: list[str] | None = None,
) -> pl.DataFrame | None:
    """Run PRS scores for a single genome.

    Saves results incrementally on every progress milestone and every
    heartbeat, so a killed run can be resumed cheaply on the next invocation.

    When ``targeted_ids`` is set, only those IDs are (re)computed and the
    results are merged into the existing cached parquet — all other rows are
    preserved.  This allows cheap targeted retries without discarding prior
    work.

    When resuming after an interrupted run (partial cache detected), only the
    missing IDs are computed — no need to rerun completed work.
    """
    genome_name = _genome_name(vcf_path)
    result_path = _result_path(vcf_path)
    expected_ids = set(pgs_ids)

    print(f"\n{'='*60}")
    print(f"Processing: {vcf_path.name} ({len(pgs_ids)} scores, build={build})")

    # --- Targeted retry mode ---
    if targeted_ids is not None:
        run_ids = [p for p in targeted_ids if p in expected_ids]
        unknown = set(targeted_ids) - expected_ids
        if unknown:
            print(f"  [warn] Targeted IDs not in catalog for {build}: {', '.join(sorted(unknown))}")
        if not run_ids:
            print(f"  [skip] No valid targeted IDs for this build.")
            return None

        cached_rows = _load_cached_rows(result_path, exclude_ids=set(run_ids))
        if cached_rows is not None:
            print(f"  [targeted] Retrying {len(run_ids)} IDs, preserving {len(cached_rows)} cached rows")
        else:
            cached_rows = []
            print(f"  [targeted] Retrying {len(run_ids)} IDs (no existing cache)")

        parquet_path, _ = normalize_genome(vcf_path, NORMALIZED_DIR)
        genotypes_lf = pl.scan_parquet(parquet_path)
        cache = catalog._cache_dir / "scores"

        flush = _make_flush(result_path, cached_rows)
        mode_label = "serial+debug" if (serial and debug) else "serial" if serial else f"{max_workers} threads"
        print(f"  Computing {len(run_ids)} targeted scores ({mode_label})...")

        if serial:
            new_rows, n_ok, n_fail = _run_serial(
                run_ids, vcf_path, genotypes_lf, catalog, best_perf_df, build, cache, genome_name, flush, debug=debug,
            )
        else:
            new_rows, n_ok, n_fail = _run_parallel(
                run_ids, vcf_path, genotypes_lf, catalog, best_perf_df, build, cache, genome_name, max_workers, flush,
            )

        flush(new_rows)  # final save
        total_rows = len(cached_rows) + len(new_rows)
        print(f"  Targeted: {n_ok} ok, {n_fail} failed → merged into {result_path.name} ({total_rows} total rows)")
        return pl.read_parquet(result_path) if result_path.exists() else None

    # --- Normal full-genome mode ---
    if force:
        if result_path.exists():
            print(f"  [force] Deleting cached results")
            result_path.unlink()
        cached_rows = []
        run_ids = list(pgs_ids)
    else:
        valid, reason = _validate_cached_results(result_path, expected_ids)
        if valid:
            print(f"  [skip] Cache valid ({reason}): {result_path.name}")
            return pl.read_parquet(result_path)

        # Partial cache: resume by running only the missing IDs.
        cached_rows = _load_cached_rows(result_path, exclude_ids=set()) or []
        cached_ids = {r["pgs_id"] for r in cached_rows}
        run_ids = [p for p in pgs_ids if p not in cached_ids]

        if cached_rows:
            print(
                f"  [resume] Partial cache: {len(cached_ids)} done, "
                f"{len(run_ids)} remaining ({reason})"
            )
        elif result_path.exists():
            print(f"  [stale] Recomputing: {reason}")

    parquet_path, _ = normalize_genome(vcf_path, NORMALIZED_DIR)
    genotypes_lf = pl.scan_parquet(parquet_path)
    cache = catalog._cache_dir / "scores"

    if not run_ids:
        print(f"  [skip] Nothing left to compute.")
        return pl.read_parquet(result_path) if result_path.exists() else None

    flush = _make_flush(result_path, cached_rows)
    mode_label = "serial+debug" if (serial and debug) else "serial" if serial else f"{max_workers} threads"
    print(f"  Computing PRS for {len(run_ids)} scores ({mode_label})...")

    t0 = time.monotonic()
    if serial:
        rows, n_ok, n_fail = _run_serial(
            run_ids, vcf_path, genotypes_lf, catalog, best_perf_df, build, cache, genome_name, flush, debug=debug,
        )
    else:
        rows, n_ok, n_fail = _run_parallel(
            run_ids, vcf_path, genotypes_lf, catalog, best_perf_df, build, cache, genome_name, max_workers, flush,
        )

    flush(rows)  # final save
    elapsed = time.monotonic() - t0
    print(f"  Finished {genome_name}: {n_ok} ok, {n_fail} failed, {elapsed:.0f}s")
    return pl.read_parquet(result_path) if result_path.exists() else None


def _load_cached_rows(
    result_path: Path,
    exclude_ids: set[str],
) -> list[dict[str, Any]] | None:
    """Load existing result rows, excluding the given IDs.

    Returns None if the file does not exist or is corrupt.
    """
    if not result_path.exists():
        return None
    try:
        df = pl.read_parquet(result_path)
        if exclude_ids:
            df = df.filter(~pl.col("pgs_id").is_in(exclude_ids))
        return df.to_dicts()
    except Exception as exc:
        print(f"  [warn] Could not read existing cache ({exc}); starting fresh")
        return None


def _make_flush(
    result_path: Path,
    cached_rows: list[dict[str, Any]],
) -> "Callable[[list[dict[str, Any]]], None]":
    """Return a thread-safe flush function that merges cached + new rows and writes to disk."""
    flush_lock = threading.Lock()
    last_n: list[int] = [0]  # mutable int to track last flushed count

    def flush(new_rows: list[dict[str, Any]]) -> None:
        with flush_lock:
            total = len(cached_rows) + len(new_rows)
            if total == last_n[0]:
                return  # nothing new since last flush
            all_rows = cached_rows + new_rows
            pl.DataFrame(all_rows).write_parquet(result_path)
            last_n[0] = total
            print(f"  [saved] {total} rows ({len(new_rows)} new + {len(cached_rows)} cached)", flush=True)

    return flush


def _rebuild_combined(vcfs: list[Path]) -> None:
    """Rebuild all_results.parquet from per-genome parquets."""
    dfs: list[pl.DataFrame] = []
    for vcf_path in vcfs:
        rp = _result_path(vcf_path)
        if rp.exists():
            try:
                dfs.append(pl.read_parquet(rp))
            except Exception:
                pass
    if not dfs:
        print("\nNo per-genome results to combine.")
        return

    combined = pl.concat(dfs)
    combined_path = OUTPUT_DIR / "all_results.parquet"
    combined.write_parquet(combined_path)

    n_genomes = combined.get_column("genome").n_unique()
    n_scores = combined.height
    n_ok = combined.filter(pl.col("error").is_null()).height
    n_fail = combined.filter(pl.col("error").is_not_null()).height
    print(f"\n  Combined: {n_genomes} genomes, {n_scores} rows ({n_ok} ok, {n_fail} failed)")
    print(f"  Written:  {combined_path}")


def main() -> None:
    _install_sigint_handler()

    parser = argparse.ArgumentParser(
        description="Smoke test: compute PRS for all catalog scores against VCFs.",
    )
    parser.add_argument(
        "files", nargs="*",
        help="Specific VCF filenames to process (default: all in data/input/)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cached output exists and is complete (ignored with --pgs-ids)",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="Disable threading — run scores one at a time (safe for debugging hangs)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help=(
            "Implies --serial. Print '[start]' and '[done]' for every individual PGS ID "
            "so you can see exactly which one is hanging. Also saves after every item."
        ),
    )
    parser.add_argument(
        "--pgs-ids",
        help=(
            "Comma-separated PGS IDs to (re)compute, e.g. PGS004776,PGS004778. "
            "Results are merged into the existing per-genome parquet; all other "
            "cached rows are preserved.  Combine with specific VCF filenames to "
            "target a subset of genomes."
        ),
    )
    args = parser.parse_args()

    debug = args.debug
    if debug:
        args.serial = True  # --debug implies --serial

    targeted_ids: list[str] | None = None
    if args.pgs_ids:
        targeted_ids = [p.strip() for p in args.pgs_ids.split(",") if p.strip()]
        print(f"Targeted retry: {len(targeted_ids)} PGS IDs — {', '.join(targeted_ids[:10])}"
              + (f" … and {len(targeted_ids)-10} more" if len(targeted_ids) > 10 else ""))

    vcfs = find_vcfs(INPUT_DIR, args.files or None)
    if not vcfs:
        print("No VCF files found in data/input/ — skipping smoke test.")
        sys.exit(0)

    print(f"Found {len(vcfs)} VCF files in {INPUT_DIR}/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    catalog = PRSCatalog()

    print("Loading PGS Catalog scores...")
    pgs_ids_by_build: dict[str, list[str]] = {}
    for build in ("GRCh37", "GRCh38"):
        ids = (
            catalog.scores(genome_build=build)
            .select("pgs_id")
            .collect()
            .get_column("pgs_id")
            .to_list()
        )
        pgs_ids_by_build[build] = ids
        print(f"  {build}: {len(ids)} scores")

    print("Loading best performance metrics...")
    best_perf_df = catalog.best_performance().collect()
    print(f"  {best_perf_df.height} performance rows loaded")

    # Pre-warm reference distribution cache so threads don't race on HF refresh.
    print("Pre-warming reference distribution cache...")
    catalog.reference_distributions(panel="1000g")
    print("  Reference distributions loaded")

    # Detect build per VCF upfront so we can validate caches before computing.
    vcf_builds: list[tuple[Path, str]] = []
    for vcf_path in vcfs:
        build = detect_genome_build(vcf_path)
        if build is None:
            build = "GRCh38"
        vcf_builds.append((vcf_path, build))

    t_start = time.monotonic()
    n_skipped = 0
    n_computed = 0

    for vcf_path, build in vcf_builds:
        if _SHUTDOWN.is_set():
            break

        pgs_ids = pgs_ids_by_build.get(build, [])
        if not pgs_ids:
            print(f"\n  [skip] {vcf_path.name}: no PGS IDs for build {build}")
            continue
        try:
            if targeted_ids is not None:
                run_genome(
                    vcf_path, catalog, pgs_ids, best_perf_df, build,
                    serial=args.serial, debug=debug, max_workers=args.threads,
                    targeted_ids=targeted_ids,
                )
                n_computed += 1
            else:
                rp = _result_path(vcf_path)
                was_cached = rp.exists() and not args.force
                valid = False
                if was_cached:
                    valid, _ = _validate_cached_results(rp, set(pgs_ids))

                run_genome(
                    vcf_path, catalog, pgs_ids, best_perf_df, build,
                    force=args.force, serial=args.serial, debug=debug, max_workers=args.threads,
                )
                if was_cached and valid:
                    n_skipped += 1
                else:
                    n_computed += 1
        except Exception:
            print(f"  [FATAL] {vcf_path.name} failed entirely:")
            traceback.print_exc()

    # Always rebuild combined from all per-genome parquets (including previously cached).
    all_vcfs = find_vcfs(INPUT_DIR)
    _rebuild_combined(all_vcfs)

    total_elapsed = time.monotonic() - t_start
    print(f"\n{'='*60}")
    print("SMOKE TEST COMPLETE")
    print(f"  Computed: {n_computed}  Skipped: {n_skipped}  Time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
