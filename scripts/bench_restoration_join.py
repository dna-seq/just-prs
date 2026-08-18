"""Measure one restored DuckDB score: wall time, peak RSS, and the counters.

Run one PGS ID per process so the reported peak RSS is that score's own high-water
mark (glibc does not return freed arena pages, so successive scores in one process
inherit the first one's peak).

    uv run python scripts/bench_restoration_join.py PGS000014 --sample antonkulaga
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import psutil
import typer

from just_prs.prs import compute_prs_duckdb, prepare_reference_universe
from just_prs.scoring import resolve_cache_dir

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pgs_id: str = typer.Argument(..., help="PGS ID to score."),
    sample: str = typer.Option("antonkulaga", help="Normalized sample parquet stem."),
    genome_build: str = typer.Option("GRCh38"),
    memory_limit: str = typer.Option("8GB", help="DuckDB memory limit."),
    restoration: bool = typer.Option(True, help="Enable reference restoration."),
    as_json: bool = typer.Option(False, "--json", help="Emit a JSON line."),
) -> None:
    cache = resolve_cache_dir()
    scoring = cache / "scores" / f"{pgs_id}_hmPOS_{genome_build}.parquet"
    genotypes = cache / "normalized" / f"{sample}.parquet"
    universe = cache / "reference" / "reference_allele_universe.parquet"
    for path in (scoring, genotypes):
        if not path.exists():
            raise typer.BadParameter(f"missing {path}")

    proc = psutil.Process()
    peak = [proc.memory_info().rss / 1e6]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss / 1e6)
            stop.wait(0.05)

    watcher = threading.Thread(target=sample_rss, daemon=True)
    watcher.start()

    handle = (
        prepare_reference_universe(universe, genome_build=genome_build)
        if restoration and universe.exists()
        else None
    )
    started = time.perf_counter()
    result = compute_prs_duckdb(
        vcf_path=genotypes,
        scoring_file=scoring,
        genome_build=genome_build,
        pgs_id=pgs_id,
        genotypes_parquet=genotypes,
        reference_restoration=handle is not None,
        reference_universe=handle,
        memory_limit=memory_limit,
    )
    elapsed = time.perf_counter() - started
    stop.set()
    watcher.join()

    payload = {
        "pgs_id": pgs_id,
        "sample": sample,
        "restoration": handle is not None,
        "seconds": round(elapsed, 2),
        "peak_rss_mb": round(peak[0]),
        "score": result.score,
        "variants_total": result.variants_total,
        "variants_matched": result.variants_matched,
        "variants_observed": result.variants_observed,
        "variants_assumed_hom_ref": result.variants_assumed_hom_ref,
        "variants_unscorable_absent": result.variants_unscorable_absent,
        "variants_ref_resolved_panel": result.variants_ref_resolved_panel,
        "variants_ref_resolved_fasta": result.variants_ref_resolved_fasta,
    }
    if as_json:
        typer.echo(json.dumps(payload))
        return
    typer.echo(
        f"{pgs_id} restoration={payload['restoration']} "
        f"{payload['seconds']}s peak={payload['peak_rss_mb']} MB "
        f"score={result.score:.6f} "
        f"matched={result.variants_matched:,}/{result.variants_total:,} "
        f"panel={result.variants_ref_resolved_panel:,} "
        f"fasta={result.variants_ref_resolved_fasta:,} "
        f"unscorable={result.variants_unscorable_absent:,}"
    )


if __name__ == "__main__":
    app()
