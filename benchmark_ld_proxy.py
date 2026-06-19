"""Benchmark per-PGS LD-proxy construction with several large scoring files.

Measures wall time, peak RSS, and CPU% to validate memory-efficient chunked
strategy before running the full pipeline.

Usage:
    uv run python benchmark_ld_proxy.py                  # selected PGS IDs
    uv run python benchmark_ld_proxy.py --workers 2       # override thread count
    uv run python benchmark_ld_proxy.py --chunk-bp 500000 # override chunk span
"""

import argparse
import os
import time
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LD-proxy table build")
    parser.add_argument("--workers", type=int, default=2, help="Thread pool size (default: 2)")
    parser.add_argument("--chunk-bp", type=int, default=1_000_000, help="Max chunk span in bp (default: 1000000)")
    parser.add_argument("--build", default="GRCh38", help="Genome build (default: GRCh38)")
    parser.add_argument(
        "--pgs-ids",
        default="PGS003162,PGS003157,PGS003177,PGS003182",
        help="Comma-separated PGS IDs to benchmark.",
    )
    args = parser.parse_args()

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 * 1024)

    from just_prs.ld_proxy import build_ld_proxy_batch

    scores_dir = Path.home() / ".cache/just-prs/scores"
    ref_dir = Path.home() / ".cache/just-prs/reference_panel/pgsc_1000G_v1"
    cache_dir = Path.home() / ".cache/just-prs"

    pgs_ids = [pgs_id.strip().upper() for pgs_id in args.pgs_ids.split(",") if pgs_id.strip()]

    print(f"Benchmarking per-PGS LD proxies for {len(pgs_ids)} PGS files...")
    existing_ids = []
    for pgs_id in pgs_ids:
        p = scores_dir / f"{pgs_id}_hmPOS_{args.build}.parquet"
        if not p.exists():
            print(f"  SKIP {pgs_id} — parquet not found")
            continue
        existing_ids.append(pgs_id)
        print(f"  {pgs_id}: {p}")

    if not existing_ids:
        print("No scoring parquets found!")
        return

    rss_after_setup = process.memory_info().rss / (1024 * 1024)
    print(f"\nMemory after setup: {rss_after_setup:.0f} MB (delta: {rss_after_setup - rss_before:.0f} MB)")

    print(f"\n{'='*60}")
    print(f"Starting LD-proxy table build:")
    print(f"  Build: {args.build}")
    print(f"  Workers: {args.workers}")
    print(f"  Chunk span: {args.chunk_bp:,} bp")
    print(f"  PGS IDs: {', '.join(existing_ids)}")
    print(f"{'='*60}\n")

    peak_rss = rss_after_setup
    cpu_samples = []

    def progress_cb(info: dict) -> None:
        nonlocal peak_rss
        current_rss = process.memory_info().rss / (1024 * 1024)
        cpu = process.cpu_percent(interval=0)
        peak_rss = max(peak_rss, current_rss)
        cpu_samples.append(cpu)
        if "pgs_id" in info:
            print(
                f"  [{info['index']}/{info['total']}] {info['pgs_id']} "
                f"status={info['status']} | RSS: {current_rss:.0f} MB "
                f"(peak: {peak_rss:.0f} MB) | CPU: {cpu:.0f}%"
            )
        elif "chromosome" in info:
            print(
                f"  chr batch [{info['chrom_index']}/{info['n_chromosomes']}] "
                f"chr{info['chromosome']}: {info['n_proxies_found']} proxies so far | "
                f"RSS: {current_rss:.0f} MB (peak: {peak_rss:.0f} MB) | CPU: {cpu:.0f}%"
            )

    process.cpu_percent(interval=None)
    t0 = time.monotonic()

    result = build_ld_proxy_batch(
        pgs_ids=existing_ids,
        chip="gsa_v3",
        build=args.build,
        ref_dir=ref_dir,
        cache_dir=cache_dir,
        panel="1000g",
        max_workers=args.workers,
        chunk_size_bp=args.chunk_bp,
        progress_callback=progress_cb,
    )

    elapsed = time.monotonic() - t0
    final_rss = process.memory_info().rss / (1024 * 1024)
    peak_rss = max(peak_rss, final_rss)
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  PGS IDs total:     {result.n_total:,}")
    print(f"  OK:                {result.n_ok:,}")
    print(f"  Cached:            {result.n_cached:,}")
    print(f"  Failed:            {result.n_failed:,}")
    print(f"  Coverage:          {result.coverage_ratio:.1%}")
    print(f"  Output dir:        {result.output_dir}")
    print(f"  Wall time:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Peak RSS:          {peak_rss:.0f} MB ({peak_rss/1024:.1f} GB)")
    print(f"  RSS delta:         {peak_rss - rss_before:.0f} MB")
    print(f"  Avg CPU%:          {avg_cpu:.0f}%")
    print(f"  Workers:           {args.workers}")
    print(f"  Chunk span:        {args.chunk_bp:,} bp")


if __name__ == "__main__":
    main()
