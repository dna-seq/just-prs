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

Memory measurement
------------------
polars / duckdb: tracemalloc peak (Python heap allocations) — captures in-process
                 allocations without OS page-cache noise.
plink2:          subprocess peak RSS sampled every 100 ms via psutil — the full
                 process footprint of the PLINK2 binary.

Usage
-----
    uv run python just-prs/benchmarks/benchmark_engines.py
    uv run python just-prs/benchmarks/benchmark_engines.py --n-pgs 50
    uv run python just-prs/benchmarks/benchmark_engines.py --output data/output/benchmarks
"""

from __future__ import annotations

import platform
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import tracemalloc
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import duckdb
import httpx
import polars as pl
import psutil
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from just_prs.normalize import VcfFilterConfig, normalize_vcf
from just_prs.prs import _normalize_scoring_columns, compute_prs
from just_prs.scoring import download_scoring_file, parse_scoring_file, resolve_cache_dir
from just_prs.reference import _prepare_plink2_score_input

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VCF_URL = "https://zenodo.org/api/records/18370498/files/antonkulaga.vcf/content"
VCF_FILENAME = "antonkulaga.vcf"

PLINK2_BASE_URL = "https://s3.amazonaws.com/plink2-assets"
PLINK2_VERSION_DATE = "20260110"
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


def measure_engine(
    fn: Callable[[], tuple[float | None, int, int]],
) -> tuple[tuple[float | None, int, int], float, float]:
    """Run *fn* with tracemalloc memory tracking.

    Returns (result, elapsed_sec, peak_heap_mb).

    Uses tracemalloc to capture peak Python heap allocations during the call.
    This is stable across runs (no OS page-cache noise) and accurately reflects
    the in-process memory cost of each engine.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak_bytes / 1_048_576


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
    url = f"{PLINK2_BASE_URL}/{filename}"
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


def vcf_to_pgen(vcf_path: Path, plink2_bin: Path, out_dir: Path) -> Path:
    """Convert VCF to PLINK2 pgen format with chr:pos:ref:alt IDs.

    This is a one-time setup step for the PLINK2 engine benchmarks.
    Returns the path prefix (without .pgen/.pvar/.psam extension).
    """
    prefix = out_dir / "benchmark_pgen"
    pgen_file = Path(str(prefix) + ".pgen")
    if pgen_file.exists():
        console.print(f"[green]PGEN already cached:[/green] {prefix}.pgen")
        return prefix

    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Converting VCF → PGEN format → {prefix}[/cyan]")
    cmd = [
        str(plink2_bin),
        "--vcf", str(vcf_path),
        "--set-all-var-ids", "@:#:$r:$a",
        "--new-id-max-allele-len", "10000",
        "--chr", "1-22",  # autosomes only; skip chrX/Y sex-info complications
        "--make-pgen",
        "--out", str(prefix),
        "--threads", "4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]PLINK2 VCF→PGEN failed:[/red]\n{result.stderr[-2000:]}")
        raise RuntimeError("PLINK2 VCF→PGEN conversion failed")
    console.print(f"[green]PGEN conversion done.[/green]")
    return prefix


def download_scoring_files(
    n: int,
    cache_dir: Path,
    genome_build: str = GENOME_BUILD,
) -> list[tuple[str, Path]]:
    """Download N scoring files from PGS Catalog. Returns list of (pgs_id, path) tuples."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, Path]] = []
    attempted = 0
    pgs_num = 1

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as prog:
        task = prog.add_task(f"Downloading scoring files (target: {n})", total=None)
        while len(results) < n and pgs_num <= 1000:
            pgs_id = f"PGS{pgs_num:06d}"
            pgs_num += 1
            attempted += 1
            prog.update(task, description=f"Downloading scoring files ({len(results)}/{n}) — trying {pgs_id}")
            try:
                path = download_scoring_file(pgs_id, cache_dir, genome_build=genome_build)
                # Verify it can be parsed
                df = _normalize_scoring_columns(parse_scoring_file(path)).collect()
                if df.height > 0:
                    results.append((pgs_id, path))
            except Exception:
                pass  # skip unavailable or unparseable scoring files
    console.print(f"[green]Collected {len(results)} scoring files (attempted {attempted})[/green]")
    return results


# ---------------------------------------------------------------------------
# Polars engine
# ---------------------------------------------------------------------------


def score_polars(
    pgs_id: str,
    scoring_file: Path,
    genotypes_lf: pl.LazyFrame,
) -> tuple[float | None, int, int]:
    """Score with our polars engine. Returns (score, variants_matched, variants_total)."""
    result = compute_prs(
        vcf_path="__unused__",
        scoring_file=scoring_file,
        genome_build=GENOME_BUILD,
        pgs_id=pgs_id,
        genotypes_lf=genotypes_lf,
    )
    return result.score, result.variants_matched, result.variants_total


# ---------------------------------------------------------------------------
# DuckDB engine
# ---------------------------------------------------------------------------

_DOSAGE_SQL = """
    CASE
        WHEN GT IS NULL OR GT = './.' OR GT = '.' THEN 0
        WHEN effect_allele = alt THEN
            (CASE WHEN split_part(replace(replace(GT, '|', '/'), './', '0/'), '/', 1) = '1' THEN 1 ELSE 0 END +
             CASE WHEN split_part(replace(replace(GT, '|', '/'), './', '0/'), '/', 2) = '1' THEN 1 ELSE 0 END)
        WHEN effect_allele = ref THEN
            (CASE WHEN split_part(replace(replace(GT, '|', '/'), './', '0/'), '/', 1) = '0' THEN 1 ELSE 0 END +
             CASE WHEN split_part(replace(replace(GT, '|', '/'), './', '0/'), '/', 2) = '0' THEN 1 ELSE 0 END)
        ELSE 0
    END
"""


def score_duckdb(
    pgs_id: str,
    scoring_file: Path,
    normalized_parquet: Path,
) -> tuple[float | None, int, int]:
    """Score with DuckDB SQL engine. Returns (score, variants_matched, variants_total)."""
    scoring_df = _normalize_scoring_columns(parse_scoring_file(scoring_file)).collect()
    variants_total = scoring_df.height

    conn = duckdb.connect()
    conn.register("scoring", scoring_df)

    row = conn.execute(f"""
        SELECT
            SUM(effect_weight * ({_DOSAGE_SQL})) AS prs_score,
            COUNT(*) AS variants_matched
        FROM read_parquet('{normalized_parquet}') g
        JOIN scoring s
          ON g.chrom = s.chr_name_norm AND g.pos = s.chr_pos_norm
    """).fetchone()
    conn.close()

    prs_score = float(row[0]) if row[0] is not None else 0.0
    variants_matched = int(row[1]) if row[1] is not None else 0
    return prs_score, variants_matched, variants_total


# ---------------------------------------------------------------------------
# PLINK2 engine
# ---------------------------------------------------------------------------


def score_plink2(
    pgs_id: str,
    scoring_file: Path,
    pgen_prefix: Path,
    plink2_bin: Path,
    tmp_dir: Path,
) -> tuple[float | None, int, int]:
    """Score with PLINK2 binary. Returns (score, variants_matched, variants_total)."""
    scoring_df = _normalize_scoring_columns(parse_scoring_file(scoring_file)).collect()
    variants_total = scoring_df.height

    score_input = _prepare_plink2_score_input(scoring_file, pgs_id, tmp_dir)
    if score_input is None:
        return None, 0, variants_total

    out_prefix = tmp_dir / pgs_id
    cmd = [
        str(plink2_bin),
        "--pfile", str(pgen_prefix),
        "--score", str(score_input), "1", "2", "3", "header", "no-mean-imputation",
        "cols=scoresums",  # force SUM output for direct comparison with polars/duckdb
        "--out", str(out_prefix),
        "--threads", "4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    sscore = Path(str(out_prefix) + ".sscore")
    if result.returncode != 0 or not sscore.exists():
        return None, 0, variants_total

    df = pl.read_csv(sscore, separator="\t")
    df = df.rename({c: c.lstrip("#") for c in df.columns})

    score_col = next(
        (c for c in df.columns if "SCORE" in c.upper() and "SUM" in c.upper()), None
    )
    if score_col is None:
        score_col = next(
            (c for c in df.columns if "SCORE" in c.upper() and "AVG" in c.upper()), None
        )
    if score_col is None or df.height == 0:
        return None, 0, variants_total

    score = float(df[score_col].to_list()[0])

    # Extract variants matched from PLINK2 log
    log_path = Path(str(out_prefix) + ".log")
    variants_matched = 0
    if log_path.exists():
        log_text = log_path.read_text()
        import re
        m = re.search(r"--score:\s+(\d+)\s+valid predictor", log_text)
        if m:
            variants_matched = int(m.group(1))
        else:
            m = re.search(r"--score:\s+(\d+)\s+variant", log_text)
            if m:
                variants_matched = int(m.group(1))

    return score, variants_matched, variants_total


# ---------------------------------------------------------------------------
# PLINK2 memory tracking (subprocess)
# ---------------------------------------------------------------------------


def score_plink2_with_memory(
    pgs_id: str,
    scoring_file: Path,
    pgen_prefix: Path,
    plink2_bin: Path,
    tmp_dir: Path,
) -> tuple[tuple[float | None, int, int], float, float]:
    """Score with PLINK2 and track subprocess peak memory. Returns (result, elapsed, peak_mb)."""
    scoring_df = _normalize_scoring_columns(parse_scoring_file(scoring_file)).collect()
    variants_total = scoring_df.height

    score_input = _prepare_plink2_score_input(scoring_file, pgs_id, tmp_dir)
    if score_input is None:
        return (None, 0, variants_total), 0.0, 0.0

    out_prefix = tmp_dir / pgs_id
    cmd = [
        str(plink2_bin),
        "--pfile", str(pgen_prefix),
        "--score", str(score_input), "1", "2", "3", "header", "no-mean-imputation",
        "cols=scoresums",  # force SUM output for direct comparison with polars/duckdb
        "--out", str(out_prefix),
        "--threads", "4",
    ]

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak_mb = _track_subprocess_peak_memory(proc.pid)
    proc.wait()
    elapsed = time.perf_counter() - t0

    sscore = Path(str(out_prefix) + ".sscore")
    if proc.returncode != 0 or not sscore.exists():
        return (None, 0, variants_total), elapsed, peak_mb

    df = pl.read_csv(sscore, separator="\t")
    df = df.rename({c: c.lstrip("#") for c in df.columns})

    score_col = next(
        (c for c in df.columns if "SCORE" in c.upper() and "SUM" in c.upper()), None
    )
    if score_col is None:
        score_col = next(
            (c for c in df.columns if "SCORE" in c.upper() and "AVG" in c.upper()), None
        )
    if score_col is None or df.height == 0:
        return (None, 0, variants_total), elapsed, peak_mb

    score = float(df[score_col].to_list()[0])

    log_path = Path(str(out_prefix) + ".log")
    variants_matched = 0
    if log_path.exists():
        import re
        log_text = log_path.read_text()
        m = re.search(r"--score:\s+(\d+)\s+valid predictor", log_text)
        if m:
            variants_matched = int(m.group(1))
        else:
            m = re.search(r"--score:\s+(\d+)\s+variant", log_text)
            if m:
                variants_matched = int(m.group(1))

    return (score, variants_matched, variants_total), elapsed, peak_mb


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


@app.command()
def run_benchmark(
    n_pgs: int = typer.Option(100, help="Number of PGS IDs to benchmark."),
    output: Path = typer.Option(
        Path("data/output/benchmarks"),
        help="Directory to write benchmark results (CSV, Parquet).",
    ),
    skip_plink2: bool = typer.Option(False, help="Skip the PLINK2 engine."),
    cache_dir: Path = typer.Option(
        None,
        help="Cache directory for VCF, scoring files, and pgen data. "
             "Defaults to just-prs cache dir.",
    ),
    n_warmup: int = typer.Option(3, help="Number of PGS IDs to use as a warm-up (not counted)."),
) -> None:
    """Benchmark Polars, DuckDB, and PLINK2 PRS engines on a real personal genome."""
    if cache_dir is None:
        cache_dir = resolve_cache_dir() / "benchmark"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Download + normalize VCF
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 1: Download + normalize VCF[/bold blue]")
    vcf_path = download_vcf(cache_dir / "vcf")
    normalized_parquet = normalize_vcf_cached(vcf_path, cache_dir / "vcf")
    genotypes_lf = pl.scan_parquet(normalized_parquet)

    n_variants = genotypes_lf.select(pl.len()).collect().item()
    console.print(f"Normalized parquet: [bold]{n_variants:,}[/bold] variants")

    # ------------------------------------------------------------------
    # Step 2: Setup PLINK2 (optional)
    # ------------------------------------------------------------------
    plink2_bin: Path | None = None
    pgen_prefix: Path | None = None
    plink2_tmp: Path = cache_dir / "plink2_tmp"
    plink2_tmp.mkdir(parents=True, exist_ok=True)

    if not skip_plink2:
        console.rule("[bold blue]Step 2: Setup PLINK2[/bold blue]")
        plink2_bin = resolve_plink2()
        if plink2_bin is None:
            console.print("[yellow]PLINK2 not available — skipping PLINK2 engine.[/yellow]")
            skip_plink2 = True
        else:
            pgen_dir = cache_dir / "pgen"
            pgen_dir.mkdir(parents=True, exist_ok=True)
            pgen_prefix = vcf_to_pgen(vcf_path, plink2_bin, pgen_dir)

    # ------------------------------------------------------------------
    # Step 3: Download scoring files
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 3: Download scoring files[/bold blue]")
    scoring_cache = cache_dir / "scores"
    total_needed = n_pgs + n_warmup
    scoring_files = download_scoring_files(total_needed, scoring_cache)

    if len(scoring_files) < n_warmup + 1:
        console.print(f"[red]Only {len(scoring_files)} scoring files available — need at least {n_warmup + 1}.[/red]")
        raise typer.Exit(1)

    warmup_pairs = scoring_files[:n_warmup]
    benchmark_pairs = scoring_files[n_warmup:n_warmup + n_pgs]
    console.print(f"Warm-up: {len(warmup_pairs)} PGS IDs | Benchmark: {len(benchmark_pairs)} PGS IDs")

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

        # --- Polars engine ---
        try:
            (score, matched, total), elapsed, peak = measure_engine(
                lambda _sf=sf, _pgs=pgs_id: score_polars(_pgs, _sf, genotypes_lf)
            )
            all_results.append(EngineResult(
                pgs_id=pgs_id, engine="polars",
                elapsed_sec=elapsed, peak_memory_mb=peak,
                score=score, variants_matched=matched, variants_total=total,
            ))
            console.print(f"polars={elapsed:.2f}s ", end="")
        except Exception as exc:
            all_results.append(EngineResult(
                pgs_id=pgs_id, engine="polars",
                elapsed_sec=0.0, peak_memory_mb=0.0,
                score=None, variants_matched=0, variants_total=0,
                error=str(exc),
            ))
            console.print("polars=ERR ", end="")

        # --- DuckDB engine ---
        try:
            (score, matched, total), elapsed, peak = measure_engine(
                lambda _sf=sf, _pgs=pgs_id: score_duckdb(_pgs, _sf, normalized_parquet)
            )
            all_results.append(EngineResult(
                pgs_id=pgs_id, engine="duckdb",
                elapsed_sec=elapsed, peak_memory_mb=peak,
                score=score, variants_matched=matched, variants_total=total,
            ))
            console.print(f"duckdb={elapsed:.2f}s ", end="")
        except Exception as exc:
            all_results.append(EngineResult(
                pgs_id=pgs_id, engine="duckdb",
                elapsed_sec=0.0, peak_memory_mb=0.0,
                score=None, variants_matched=0, variants_total=0,
                error=str(exc),
            ))
            console.print("duckdb=ERR ", end="")

        # --- PLINK2 engine ---
        if not skip_plink2 and plink2_bin and pgen_prefix:
            try:
                with tempfile.TemporaryDirectory() as td:
                    (score, matched, total), elapsed, peak = score_plink2_with_memory(
                        pgs_id, sf, pgen_prefix, plink2_bin, Path(td)
                    )
                if score is not None:
                    all_results.append(EngineResult(
                        pgs_id=pgs_id, engine="plink2",
                        elapsed_sec=elapsed, peak_memory_mb=peak,
                        score=score, variants_matched=matched, variants_total=total,
                    ))
                    console.print(f"plink2={elapsed:.2f}s", end="")
                else:
                    all_results.append(EngineResult(
                        pgs_id=pgs_id, engine="plink2",
                        elapsed_sec=elapsed, peak_memory_mb=peak,
                        score=None, variants_matched=0, variants_total=total,
                        error="PLINK2 produced no score",
                    ))
                    console.print("plink2=NOSCORE", end="")
            except Exception as exc:
                all_results.append(EngineResult(
                    pgs_id=pgs_id, engine="plink2",
                    elapsed_sec=0.0, peak_memory_mb=0.0,
                    score=None, variants_matched=0, variants_total=0,
                    error=str(exc),
                ))
                console.print("plink2=ERR", end="")

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
    results_path = output / "benchmark_results.parquet"
    results_csv = output / "benchmark_results.csv"
    validation_path = output / "benchmark_validation.parquet"
    validation_csv = output / "benchmark_validation.csv"

    results_df.write_parquet(results_path)
    results_df.write_csv(results_csv)
    validation_df.write_parquet(validation_path)
    validation_df.write_csv(validation_csv)

    # Save system metadata for reproducibility
    import sys
    mem_gb = psutil.virtual_memory().total / 1024**3
    cpu_count = psutil.cpu_count(logical=True)
    meta = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count_logical": cpu_count,
        "total_ram_gb": round(mem_gb, 1),
        "n_pgs_benchmarked": len(benchmark_pairs),
        "n_pgs_warmup": len(warmup_pairs),
        "vcf_url": VCF_URL,
        "genome_build": GENOME_BUILD,
        "plink2_available": str(not skip_plink2),
        "memory_method_python": "tracemalloc_peak_heap",
        "memory_method_plink2": "subprocess_peak_rss",
    }
    meta_df = pl.DataFrame([{"key": k, "value": str(v)} for k, v in meta.items()])
    meta_df.write_csv(output / "benchmark_metadata.csv")

    console.print(f"\n[green]Results saved:[/green]")
    console.print(f"  {results_path}")
    console.print(f"  {results_csv}")
    console.print(f"  {validation_path}")
    console.print(f"  {validation_csv}")
    console.print(f"  {output / 'benchmark_metadata.csv'}")
    console.print(f"\n[dim]* RAM: polars/duckdb = tracemalloc peak heap; plink2 = subprocess peak RSS[/dim]")


if __name__ == "__main__":
    app()
