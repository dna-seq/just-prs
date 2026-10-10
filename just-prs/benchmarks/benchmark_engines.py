"""Engine benchmark: Polars vs DuckDB vs PLINK2 for PRS computation on personal genome.

Downloads a real personal genome VCF (Zenodo), normalizes it, then scores 100+
PGS IDs with three engines and compares runtime, peak memory, and result quality.

Engines benchmarked
-------------------
polars  — just-prs compute_prs() with a pre-normalized parquet LazyFrame as input.
           Uses polars-bio + polars expressions for variant matching and dosage.
duckdb  — DuckDB SQL join of the normalized parquet with the scoring DataFrame.
           Dosage is computed inline with DuckDB SQL expressions.
plink2  — PLINK2 binary, pre-converted pgen dataset (chr:pos:ref:alt IDs).
           Amortizes VCF→pgen conversion over the full benchmark run.

Inputs are made identical for all three engines
-----------------------------------------------
- Only PASS (or ".") records: just-prs reads ``normalize_vcf(pass_filters=["PASS", "."])``
  and the pgen is converted with ``--vcf-filter``, so DeepVariant RefCall 0/0 sites
  are excluded from both. (Without it PLINK2 counted them as two REF copies.)
- Autosomes only (``--autosomes-only``, default): PLINK2 converts ``--chr 1-22``
  and the just-prs engines read the same records filtered to chromosomes 1-22.
- The PLINK2 score input lists every ``hm_inferOtherAllele`` candidate and sums
  liftover-duplicate (variant, allele) pairs (``_prepare_plink2_score_input``).
- The PGS IDs are an explicit list (``--pgs-ids``), written to ``pgs_ids.txt``.

Outputs (``--output``): long and per-PGS CSVs, the exact ID list, scoring-file
checksums, and PLINK2's ``.log`` plus captured stdout/stderr for every ID.

Memory measurement
------------------
polars / duckdb: process RSS delta (peak RSS during call minus RSS before call),
                 sampled every 10 ms via psutil. Captures native Rust/C++ allocations
                 that tracemalloc misses.
plink2:          subprocess peak RSS sampled every 100 ms via psutil — the full
                 process footprint of the PLINK2 binary.

Usage
-----
    uv run python just-prs/benchmarks/benchmark_engines.py
    uv run python just-prs/benchmarks/benchmark_engines.py --pgs-ids PGS000001-PGS000050
    uv run python just-prs/benchmarks/benchmark_engines.py --output data/output/benchmarks
"""

from __future__ import annotations

import hashlib
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import httpx
import polars as pl
import psutil
import typer
from rich.console import Console
from rich.table import Table

from just_prs.normalize import VcfFilterConfig, normalize_vcf
from just_prs.prs import compute_prs, compute_prs_duckdb
from just_prs.scoring import ensure_scoring_file, read_scoring_header, resolve_cache_dir
from just_prs.reference import _prepare_plink2_score_input

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VCF_URL = "https://zenodo.org/api/records/18370498/files/antonkulaga.vcf/content"
VCF_FILENAME = "antonkulaga.vcf"
VCF_MD5 = "21598da4f77d8237577662ab89e8a55d"

# The PGS IDs of the published benchmark (Paper 1, Table 5): PGS000006 to
# PGS000106 without PGS000085, after the three warm-up IDs.
DEFAULT_PGS_IDS = "PGS000006-PGS000106"
DEFAULT_EXCLUDE = "PGS000085"
DEFAULT_WARMUP_IDS = "PGS000001,PGS000002,PGS000003"

PLINK2_BASE_URL = "https://s3.amazonaws.com/plink2-assets"
PLINK2_RELEASE_PATH = "alpha6"
PLINK2_VERSION_DATE = "20260228"
PLINK2_PLATFORM_MAP: dict[tuple[str, str], str] = {
    ("Linux", "x86_64"): f"plink2_linux_x86_64_{PLINK2_VERSION_DATE}.zip",
    ("Linux", "aarch64"): f"plink2_linux_aarch64_{PLINK2_VERSION_DATE}.zip",
    ("Darwin", "arm64"): f"plink2_mac_arm64_{PLINK2_VERSION_DATE}.zip",
    ("Darwin", "x86_64"): f"plink2_mac_{PLINK2_VERSION_DATE}.zip",
    ("Windows", "AMD64"): f"plink2_win64_{PLINK2_VERSION_DATE}.zip",
}

GENOME_BUILD = "GRCh38"

# ---------------------------------------------------------------------------
# Data class for a single engine result
# ---------------------------------------------------------------------------


@dataclass
class EngineResult:
    pgs_id: str
    engine: str
    elapsed_sec: float
    peak_memory_mb: float
    score: float | None
    variants_matched: int
    variants_total: int
    variants_allele_mismatch: int = 0
    error: str | None = None

    @property
    def match_rate(self) -> float:
        return self.variants_matched / self.variants_total if self.variants_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Memory tracking helpers
# ---------------------------------------------------------------------------


def _track_subprocess_peak_memory(pid: int) -> float:
    """Sample child process RSS every 100 ms until it exits. Returns peak MiB."""
    peak = 0.0
    try:
        p = psutil.Process(pid)
        while p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
            try:
                rss = p.memory_info().rss / 1_048_576
                if rss > peak:
                    peak = rss
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)
    except psutil.NoSuchProcess:
        pass
    return peak


T = TypeVar("T")


def measure_engine(fn: Callable[[], T]) -> tuple[T, float, float]:
    """Run *fn* with RSS-based memory tracking.

    Returns (result, elapsed_sec, rss_delta_mb).

    Measures process RSS before and peak RSS during execution via a background
    sampling thread. This captures native allocations from polars (Rust) and
    DuckDB (C++) that tracemalloc misses.
    """
    import gc
    gc.collect()

    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    peak_rss = rss_before
    stop_event = threading.Event()

    def _sampler() -> None:
        nonlocal peak_rss
        while not stop_event.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
            except psutil.NoSuchProcess:
                break
            stop_event.wait(0.01)

    sampler = threading.Thread(target=_sampler, daemon=True)
    sampler.start()

    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0

    stop_event.set()
    sampler.join(timeout=1.0)

    delta_mb = (peak_rss - rss_before) / 1_048_576
    return result, elapsed, max(delta_mb, 0.0)


# ---------------------------------------------------------------------------
# Download / setup helpers
# ---------------------------------------------------------------------------


def download_vcf(cache_dir: Path) -> Path:
    """Download the benchmark VCF from Zenodo. Cached after first download."""
    vcf_path = cache_dir / VCF_FILENAME
    if vcf_path.exists():
        console.print(f"[green]VCF already cached:[/green] {vcf_path}")
        return vcf_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Downloading VCF from Zenodo → {vcf_path}[/cyan]")
    with httpx.Client(timeout=None, follow_redirects=True) as client:
        with client.stream("GET", VCF_URL) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with vcf_path.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        console.print(
                            f"  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct:.0f}%)",
                            end="\r",
                        )
    console.print()
    return vcf_path


def normalize_vcf_cached(vcf_path: Path, cache_dir: Path) -> Path:
    """Normalize the VCF to parquet. Result is cached next to the VCF."""
    parquet_path = cache_dir / (vcf_path.stem + "_normalized.parquet")
    if parquet_path.exists():
        console.print(f"[green]Normalized parquet already cached:[/green] {parquet_path}")
        return parquet_path

    console.print(f"[cyan]Normalizing VCF → {parquet_path}[/cyan]")
    config = VcfFilterConfig(pass_filters=["PASS", "."])
    normalize_vcf(vcf_path, parquet_path, config=config)
    console.print(f"[green]Normalized parquet written: {parquet_path.stat().st_size / 1e6:.0f} MB[/green]")
    return parquet_path


def download_plink2(cache_dir: Path) -> Path | None:
    """Download the PLINK2 binary. Returns None if unsupported platform."""
    system = platform.system()
    machine = platform.machine()
    filename = PLINK2_PLATFORM_MAP.get((system, machine))
    if filename is None:
        console.print(f"[yellow]PLINK2 auto-download not supported for {system}/{machine}.[/yellow]")
        return None

    binary_name = "plink2.exe" if system == "Windows" else "plink2"
    binary_path = cache_dir / binary_name
    if binary_path.exists():
        return binary_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{PLINK2_BASE_URL}/{PLINK2_RELEASE_PATH}/{filename}"
    zip_path = cache_dir / filename
    console.print(f"[cyan]Downloading PLINK2 binary → {binary_path}[/cyan]")
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    fh.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(binary_name, cache_dir)
    if system != "Windows":
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    zip_path.unlink(missing_ok=True)
    console.print(f"[green]PLINK2 binary ready: {binary_path}[/green]")
    return binary_path


def resolve_plink2() -> Path | None:
    """Find PLINK2: system PATH first, then auto-download."""
    sys_path = shutil.which("plink2")
    if sys_path:
        return Path(sys_path)
    cache_dir = resolve_cache_dir() / "plink2"
    return download_plink2(cache_dir)


def parse_pgs_ids(spec: str) -> list[str]:
    """Expand ``"PGS000006-PGS000106,PGS000200"`` into an ordered ID list."""
    ids: list[str] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        m = re.fullmatch(r"PGS(\d{6})-PGS(\d{6})", part, flags=re.IGNORECASE)
        if m:
            ids.extend(f"PGS{n:06d}" for n in range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ids.append(part.upper())
    return list(dict.fromkeys(ids))


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vcf_to_pgen(vcf_path: Path, plink2_bin: Path, out_dir: Path, log_dir: Path) -> Path:
    """Convert the VCF to a PASS-only, autosomal PLINK2 pgen with chr:pos:ref:alt IDs.

    ``--vcf-filter`` keeps the records just-prs reads (FILTER PASS or "."); the
    unfiltered conversion counted DeepVariant RefCall 0/0 sites as two REF copies.
    The prefix differs from the old unfiltered ``benchmark_pgen`` so a cached
    unfiltered pgen is never reused. Returns the path prefix.
    """
    prefix = out_dir / "benchmark_pgen_pass"
    pgen_file = Path(str(prefix) + ".pgen")
    if not pgen_file.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]Converting VCF → PASS-only PGEN → {prefix}[/cyan]")
        cmd = [
            str(plink2_bin),
            "--vcf", str(vcf_path),
            "--vcf-filter",
            "--set-all-var-ids", "@:#:$r:$a",
            "--new-id-max-allele-len", "10000",
            "--chr", "1-22",  # chrX needs sex info; the just-prs inputs are cut to 1-22 too
            "--make-pgen",
            "--out", str(prefix),
            "--threads", "4",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]PLINK2 VCF→PGEN failed:[/red]\n{result.stderr[-2000:]}")
            raise RuntimeError("PLINK2 VCF→PGEN conversion failed")
        console.print("[green]PGEN conversion done.[/green]")
    else:
        console.print(f"[green]PGEN already cached:[/green] {prefix}.pgen")
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(str(prefix) + ".log"), log_dir / "vcf_to_pgen.log")
    return prefix


def autosome_parquet(normalized_parquet: Path) -> Path:
    """The normalized genotypes restricted to chromosomes 1-22 (PLINK2's ``--chr 1-22``)."""
    out = normalized_parquet.with_name(normalized_parquet.stem + "_autosomes.parquet")
    if out.exists() and out.stat().st_mtime >= normalized_parquet.stat().st_mtime:
        return out
    autosomes = [str(c) for c in range(1, 23)]
    pl.scan_parquet(normalized_parquet).filter(pl.col("chrom").is_in(autosomes)).sink_parquet(out)
    return out


def resolve_scoring_files(
    pgs_ids: list[str],
    scores_dir: Path,
    genome_build: str = GENOME_BUILD,
) -> tuple[list[tuple[str, Path]], dict[str, str]]:
    """Resolve each ID to its cached harmonized parquet. Returns (found, {id: error})."""
    found: list[tuple[str, Path]] = []
    failed: dict[str, str] = {}
    for pgs_id in pgs_ids:
        try:
            found.append((pgs_id, ensure_scoring_file(pgs_id, scores_dir, genome_build=genome_build)))
        except Exception as exc:  # recorded and reported, not silently dropped
            failed[pgs_id] = f"{type(exc).__name__}: {exc}"
    console.print(f"[green]Scoring files: {len(found)} resolved, {len(failed)} failed[/green]")
    for pgs_id, err in failed.items():
        console.print(f"  [yellow]{pgs_id}: {err}[/yellow]")
    return found, failed


# ---------------------------------------------------------------------------
# Polars engine
# ---------------------------------------------------------------------------


def score_polars(
    pgs_id: str,
    scoring_file: Path,
    genotypes_lf: pl.LazyFrame,
) -> tuple[float | None, int, int, int]:
    """Score with the polars engine. Returns (score, matched, total, allele_mismatch)."""
    result = compute_prs(
        vcf_path="__unused__",
        scoring_file=scoring_file,
        genome_build=GENOME_BUILD,
        pgs_id=pgs_id,
        genotypes_lf=genotypes_lf,
    )
    return result.score, result.variants_matched, result.variants_total, result.variants_allele_mismatch


# ---------------------------------------------------------------------------
# DuckDB engine (uses production compute_prs_duckdb from just_prs.prs)
# ---------------------------------------------------------------------------


def score_duckdb(
    pgs_id: str,
    scoring_file: Path,
    normalized_parquet: Path,
) -> tuple[float | None, int, int, int]:
    """Score with the DuckDB engine. Returns (score, matched, total, allele_mismatch)."""
    result = compute_prs_duckdb(
        vcf_path="__unused__",
        scoring_file=scoring_file,
        genome_build=GENOME_BUILD,
        pgs_id=pgs_id,
        genotypes_parquet=normalized_parquet,
    )
    return result.score, result.variants_matched, result.variants_total, result.variants_allele_mismatch


# ---------------------------------------------------------------------------
# PLINK2 engine
# ---------------------------------------------------------------------------


def _read_plink2_score(out_prefix: Path) -> tuple[float | None, int]:
    """SCORE*_SUM from the .sscore and the matched-variant count from the .log."""
    sscore = Path(str(out_prefix) + ".sscore")
    score: float | None = None
    if sscore.exists():
        df = pl.read_csv(sscore, separator="\t")
        df = df.rename({c: c.lstrip("#") for c in df.columns})
        score_col = next((c for c in df.columns if "SCORE" in c.upper() and "SUM" in c.upper()), None)
        if score_col is not None and df.height > 0:
            score = float(df[score_col][0])
    variants_matched = 0
    log_path = Path(str(out_prefix) + ".log")
    if log_path.exists():
        m = re.search(r"--score:\s+(\d+)\s+variants? processed", log_path.read_text())
        if m:
            variants_matched = int(m.group(1))
    return score, variants_matched


def score_plink2(
    pgs_id: str,
    scoring_file: Path,
    pgen_prefix: Path,
    plink2_bin: Path,
    tmp_dir: Path,
    log_dir: Path | None = None,
) -> tuple[tuple[float | None, int, int, int], float, float, str | None]:
    """Score with PLINK2, tracking subprocess peak RSS.

    Returns ``((score, matched, total, 0), elapsed, peak_mb, error)``. PLINK2's
    stdout/stderr and ``.log`` are copied to *log_dir* when given.
    """
    variants_total = parse_scoring_file_height(scoring_file)
    score_input = _prepare_plink2_score_input(scoring_file, pgs_id, tmp_dir)
    if score_input is None:
        return (None, 0, variants_total, 0), 0.0, 0.0, "no score input"

    out_prefix = tmp_dir / pgs_id
    cmd = [
        str(plink2_bin),
        "--pfile", str(pgen_prefix),
        "--score", str(score_input), "1", "2", "3", "header", "no-mean-imputation",
        "cols=scoresums",  # SUM output, directly comparable with polars/duckdb
        "--out", str(out_prefix),
        "--threads", "4",
    ]
    console_path = tmp_dir / f"{pgs_id}.console.txt"
    t0 = time.perf_counter()
    with console_path.open("w") as console_fh:
        proc = subprocess.Popen(cmd, stdout=console_fh, stderr=subprocess.STDOUT)
        peak_mb = _track_subprocess_peak_memory(proc.pid)
        proc.wait()
    elapsed = time.perf_counter() - t0

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(console_path, log_dir / f"{pgs_id}.console.txt")
        log_path = Path(str(out_prefix) + ".log")
        if log_path.exists():
            shutil.copy2(log_path, log_dir / f"{pgs_id}.log")

    score, variants_matched = _read_plink2_score(out_prefix)
    error = None
    if proc.returncode != 0 or score is None:
        tail = console_path.read_text().strip().splitlines()[-1:] or [""]
        error = f"exit {proc.returncode}: {tail[0]}"
        score = None
    return (score, variants_matched, variants_total, 0), elapsed, peak_mb, error


def parse_scoring_file_height(scoring_file: Path) -> int:
    from just_prs.scoring import parse_scoring_file

    return int(parse_scoring_file(scoring_file).select(pl.len()).collect().item())


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_agreement(results_df: pl.DataFrame) -> pl.DataFrame:
    """Compute pairwise Pearson r between all engine pairs per pgs_id.

    Returns a summary DataFrame with correlation and max-abs-diff between engines.
    """
    engines = results_df["engine"].unique().sort().to_list()
    rows: list[dict[str, object]] = []

    scores_wide = (
        results_df
        .filter(pl.col("score").is_not_null())
        .pivot(values="score", index="pgs_id", on="engine", aggregate_function="first")
    )

    for i, eng_a in enumerate(engines):
        for eng_b in engines[i + 1:]:
            if eng_a not in scores_wide.columns or eng_b not in scores_wide.columns:
                continue
            pair_df = scores_wide.select("pgs_id", eng_a, eng_b).drop_nulls()
            if pair_df.height < 3:
                continue
            corr = pair_df.select(pl.corr(eng_a, eng_b)).item()
            max_diff = (
                pair_df.select((pl.col(eng_a) - pl.col(eng_b)).abs().max()).item()
            )
            rows.append({
                "engine_a": eng_a,
                "engine_b": eng_b,
                "n_pgs": pair_df.height,
                "pearson_r": round(corr, 8) if corr is not None else None,
                "max_abs_diff": max_diff,
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def print_summary_table(results_df: pl.DataFrame) -> None:
    """Print per-engine timing and memory statistics using Rich."""
    agg = (
        results_df
        .filter(pl.col("error").is_null())
        .group_by("engine")
        .agg(
            pl.col("elapsed_sec").mean().alias("mean_elapsed_sec"),
            pl.col("elapsed_sec").median().alias("median_elapsed_sec"),
            pl.col("elapsed_sec").min().alias("min_elapsed_sec"),
            pl.col("elapsed_sec").max().alias("max_elapsed_sec"),
            pl.col("peak_memory_mb").mean().alias("mean_peak_mb"),
            pl.col("peak_memory_mb").max().alias("max_peak_mb"),
            pl.col("match_rate").mean().alias("mean_match_rate"),
            pl.len().alias("n_pgs"),
        )
        .sort("mean_elapsed_sec")
    )

    table = Table(title="Engine Benchmark Summary", show_header=True, header_style="bold magenta")
    table.add_column("Engine", style="cyan", no_wrap=True)
    table.add_column("N PGS", justify="right")
    table.add_column("Mean elapsed (s)", justify="right")
    table.add_column("Median elapsed (s)", justify="right")
    table.add_column("Min (s)", justify="right")
    table.add_column("Max (s)", justify="right")
    table.add_column("Mean peak RAM (MB)*", justify="right")
    table.add_column("Max peak RAM (MB)*", justify="right")
    table.add_column("Mean match rate", justify="right")

    for row in agg.iter_rows(named=True):
        table.add_row(
            row["engine"],
            str(row["n_pgs"]),
            f"{row['mean_elapsed_sec']:.3f}",
            f"{row['median_elapsed_sec']:.3f}",
            f"{row['min_elapsed_sec']:.3f}",
            f"{row['max_elapsed_sec']:.3f}",
            f"{row['mean_peak_mb']:.1f}",
            f"{row['max_peak_mb']:.1f}",
            f"{row['mean_match_rate']:.3f}",
        )

    console.print(table)


def print_validation_table(validation_df: pl.DataFrame) -> None:
    """Print pairwise engine agreement statistics."""
    table = Table(title="Cross-Engine Score Agreement", show_header=True, header_style="bold green")
    table.add_column("Engine A", style="cyan")
    table.add_column("Engine B", style="cyan")
    table.add_column("N PGS IDs", justify="right")
    table.add_column("Pearson r", justify="right")
    table.add_column("Max |diff|", justify="right")

    for row in validation_df.iter_rows(named=True):
        r_val = row["pearson_r"]
        r_str = f"{r_val:.8f}" if r_val is not None else "N/A"
        diff_val = row["max_abs_diff"]
        diff_str = f"{diff_val:.4e}" if diff_val is not None else "N/A"
        color = "green" if (r_val is not None and r_val > 0.999) else "red"
        table.add_row(
            row["engine_a"],
            row["engine_b"],
            str(row["n_pgs"]),
            f"[{color}]{r_str}[/{color}]",
            diff_str,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def _engine_result(
    pgs_id: str,
    engine: str,
    run: Callable[[], tuple[float | None, int, int, int]],
) -> EngineResult:
    try:
        (score, matched, total, mismatch), elapsed, peak = measure_engine(run)
    except Exception as exc:
        return EngineResult(
            pgs_id=pgs_id, engine=engine, elapsed_sec=0.0, peak_memory_mb=0.0,
            score=None, variants_matched=0, variants_total=0, error=f"{type(exc).__name__}: {exc}",
        )
    return EngineResult(
        pgs_id=pgs_id, engine=engine, elapsed_sec=elapsed, peak_memory_mb=peak,
        score=score, variants_matched=matched, variants_total=total,
        variants_allele_mismatch=mismatch,
    )


def per_pgs_table(results_df: pl.DataFrame) -> pl.DataFrame:
    """One row per PGS ID: each engine's score and matched count, and the deltas."""
    wide = results_df.pivot(
        values=["score", "variants_matched", "error"], index="pgs_id", on="engine",
        aggregate_function="first",
    )
    totals = results_df.group_by("pgs_id", maintain_order=True).agg(
        pl.col("variants_total").max(),
        pl.col("variants_allele_mismatch").filter(pl.col("engine") == "duckdb").first(),
    )
    out = totals.join(wide, on="pgs_id", how="left")
    if {"score_polars", "score_duckdb"} <= set(out.columns):
        out = out.with_columns((pl.col("score_duckdb") - pl.col("score_polars")).alias("delta_duckdb_polars"))
    if {"score_polars", "score_plink2"} <= set(out.columns):
        out = out.with_columns((pl.col("score_polars") - pl.col("score_plink2")).alias("delta_polars_plink2"))
    return out


@app.command()
def run_benchmark(
    pgs_ids: str = typer.Option(
        DEFAULT_PGS_IDS,
        help="Comma-separated PGS IDs and ranges (PGS000006-PGS000106) to benchmark.",
    ),
    exclude: str = typer.Option(DEFAULT_EXCLUDE, help="Comma-separated IDs/ranges to leave out."),
    warmup_ids: str = typer.Option(
        DEFAULT_WARMUP_IDS, help="IDs scored once before timing starts (not counted)."
    ),
    output: Path = typer.Option(
        Path("data/output/benchmarks"),
        help="Directory to write benchmark results (CSV, Parquet, PLINK2 logs).",
    ),
    skip_plink2: bool = typer.Option(False, help="Skip the PLINK2 engine."),
    autosomes_only: bool = typer.Option(
        True,
        "--autosomes-only/--all-chromosomes",
        help="Score chromosomes 1-22 only in every engine, as PLINK2's pgen holds.",
    ),
    cache_dir: Path = typer.Option(
        None,
        help="Cache directory for the VCF and pgen data. Defaults to <just-prs cache>/benchmark.",
    ),
    scores_dir: Path = typer.Option(
        None, help="Scoring-file cache. Defaults to <just-prs cache>/scores."
    ),
) -> None:
    """Benchmark Polars, DuckDB, and PLINK2 PRS engines on a real personal genome."""
    if cache_dir is None:
        cache_dir = resolve_cache_dir() / "benchmark"
    if scores_dir is None:
        scores_dir = resolve_cache_dir() / "scores"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / "plink2_logs"

    excluded = set(parse_pgs_ids(exclude))
    benchmark_ids = [i for i in parse_pgs_ids(pgs_ids) if i not in excluded]
    warmup_list = parse_pgs_ids(warmup_ids)
    (output / "pgs_ids.txt").write_text("\n".join(benchmark_ids) + "\n")
    console.print(f"Benchmark IDs: {len(benchmark_ids)} ({benchmark_ids[0]} … {benchmark_ids[-1]})")

    # ------------------------------------------------------------------
    # Step 1: Download + normalize VCF
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 1: Download + normalize VCF[/bold blue]")
    vcf_path = download_vcf(cache_dir / "vcf")
    vcf_md5 = file_md5(vcf_path)
    if vcf_md5 != VCF_MD5:
        console.print(f"[red]VCF md5 {vcf_md5} != expected {VCF_MD5}[/red]")
        raise typer.Exit(1)
    normalized_parquet = normalize_vcf_cached(vcf_path, cache_dir / "vcf")
    if autosomes_only:
        normalized_parquet = autosome_parquet(normalized_parquet)
    genotypes_lf = pl.scan_parquet(normalized_parquet)

    n_variants = genotypes_lf.select(pl.len()).collect().item()
    console.print(f"Genotype input: [bold]{n_variants:,}[/bold] PASS records ({normalized_parquet.name})")

    # ------------------------------------------------------------------
    # Step 2: Setup PLINK2 (optional)
    # ------------------------------------------------------------------
    plink2_bin: Path | None = None
    pgen_prefix: Path | None = None
    plink2_version = ""

    if not skip_plink2:
        console.rule("[bold blue]Step 2: Setup PLINK2[/bold blue]")
        plink2_bin = resolve_plink2()
        if plink2_bin is None:
            console.print("[yellow]PLINK2 not available — skipping PLINK2 engine.[/yellow]")
            skip_plink2 = True
        else:
            plink2_version = subprocess.run(
                [str(plink2_bin), "--version"], capture_output=True, text=True
            ).stdout.strip()
            console.print(f"PLINK2: {plink2_bin} ({plink2_version})")
            pgen_prefix = vcf_to_pgen(vcf_path, plink2_bin, cache_dir / "pgen", log_dir)

    # ------------------------------------------------------------------
    # Step 3: Scoring files
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 3: Scoring files[/bold blue]")
    benchmark_pairs, scoring_failures = resolve_scoring_files(benchmark_ids, scores_dir)
    warmup_pairs, _ = resolve_scoring_files(warmup_list, scores_dir)
    scoring_rows = []
    for pgs_id, sf in benchmark_pairs:
        header = read_scoring_header(sf)
        scoring_rows.append({
            "pgs_id": pgs_id,
            "file": sf.name,
            "md5": file_md5(sf),
            "format_version": header.get("format_version"),
            "HmPOS_build": header.get("HmPOS_build"),
            "HmPOS_date": header.get("HmPOS_date"),
            "variants_number": header.get("variants_number"),
        })
    for pgs_id, err in scoring_failures.items():
        scoring_rows.append({"pgs_id": pgs_id, "file": None, "md5": None, "error": err})
    pl.DataFrame(scoring_rows).write_csv(output / "scoring_files.csv")

    # ------------------------------------------------------------------
    # Step 4: Warm-up (not counted)
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 4: Warm-up[/bold blue]")
    for pgs_id, sf in warmup_pairs:
        console.print(f"  Warm-up {pgs_id} ...")
        score_polars(pgs_id, sf, genotypes_lf)
        score_duckdb(pgs_id, sf, normalized_parquet)
        if plink2_bin and pgen_prefix:
            with tempfile.TemporaryDirectory() as td:
                score_plink2(pgs_id, sf, pgen_prefix, plink2_bin, Path(td))
    console.print("[green]Warm-up complete.[/green]")

    # ------------------------------------------------------------------
    # Step 5: Benchmark
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 5: Benchmark[/bold blue]")
    all_results: list[EngineResult] = []

    for idx, (pgs_id, sf) in enumerate(benchmark_pairs):
        console.print(f"  [{idx + 1}/{len(benchmark_pairs)}] {pgs_id}", end="  ")
        for engine, run in (
            ("polars", lambda _sf=sf, _pgs=pgs_id: score_polars(_pgs, _sf, genotypes_lf)),
            ("duckdb", lambda _sf=sf, _pgs=pgs_id: score_duckdb(_pgs, _sf, normalized_parquet)),
        ):
            res = _engine_result(pgs_id, engine, run)
            all_results.append(res)
            console.print(f"{engine}={'ERR' if res.error else f'{res.elapsed_sec:.2f}s'} ", end="")

        if not skip_plink2 and plink2_bin and pgen_prefix:
            with tempfile.TemporaryDirectory() as td:
                (score, matched, total, _), elapsed, peak, error = score_plink2(
                    pgs_id, sf, pgen_prefix, plink2_bin, Path(td), log_dir=log_dir
                )
            all_results.append(EngineResult(
                pgs_id=pgs_id, engine="plink2", elapsed_sec=elapsed, peak_memory_mb=peak,
                score=score, variants_matched=matched, variants_total=total, error=error,
            ))
            console.print(f"plink2={'ERR' if error else f'{elapsed:.2f}s'}", end="")
        console.print()

    # ------------------------------------------------------------------
    # Step 6: Results and validation
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 6: Results[/bold blue]")

    results_df = pl.DataFrame([
        {
            "pgs_id": r.pgs_id,
            "engine": r.engine,
            "elapsed_sec": r.elapsed_sec,
            "peak_memory_mb": r.peak_memory_mb,
            "score": r.score,
            "variants_matched": r.variants_matched,
            "variants_total": r.variants_total,
            "variants_allele_mismatch": r.variants_allele_mismatch,
            "match_rate": r.match_rate,
            "error": r.error,
        }
        for r in all_results
    ])

    print_summary_table(results_df)

    validation_df = validate_agreement(results_df)
    print_validation_table(validation_df)

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 7: Save results[/bold blue]")
    results_df.write_parquet(output / "benchmark_results.parquet")
    results_df.write_csv(output / "benchmark_results.csv")
    per_pgs_table(results_df).write_csv(output / "benchmark_per_pgs.csv")
    validation_df.write_parquet(output / "benchmark_validation.parquet")
    validation_df.write_csv(output / "benchmark_validation.csv")

    import sys
    from importlib.metadata import version

    import just_prs

    src_dir = Path(just_prs.__file__).resolve().parent
    git = ["git", "-C", str(src_dir)]
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run([*git, "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout.strip())
    meta = {
        "just_prs_version": version("just-prs"),
        "just_prs_source": str(src_dir),
        "just_prs_git_commit": commit or "unknown",
        "just_prs_source_dirty": dirty,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "n_pgs_benchmarked": len(benchmark_pairs),
        "n_pgs_scoring_failed": len(scoring_failures),
        "warmup_ids": ",".join(p for p, _ in warmup_pairs),
        "vcf_url": VCF_URL,
        "vcf_md5": vcf_md5,
        "genotype_input": normalized_parquet.name,
        "genotype_records": n_variants,
        "autosomes_only": autosomes_only,
        "genome_build": GENOME_BUILD,
        "plink2_version": plink2_version or "skipped",
        "plink2_pgen": "--vcf-filter --chr 1-22 --set-all-var-ids @:#:$r:$a",
        "plink2_score": "header no-mean-imputation cols=scoresums",
        "memory_method_python": "rss_delta_peak_minus_baseline",
        "memory_method_plink2": "subprocess_peak_rss",
    }
    pl.DataFrame([{"key": k, "value": str(v)} for k, v in meta.items()]).write_csv(
        output / "benchmark_metadata.csv"
    )

    console.print(f"\n[green]Results saved to {output}[/green]: benchmark_results.csv, "
                  "benchmark_per_pgs.csv, benchmark_validation.csv, benchmark_metadata.csv, "
                  "scoring_files.csv, pgs_ids.txt, plink2_logs/")
    console.print("\n[dim]* RAM: polars/duckdb = process RSS delta (peak - baseline); plink2 = subprocess peak RSS[/dim]")


if __name__ == "__main__":
    app()
