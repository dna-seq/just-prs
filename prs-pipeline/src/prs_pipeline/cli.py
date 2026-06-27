"""CLI for prs-pipeline: launch or run the Dagster reference-panel pipeline."""

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(help="PRS reference-panel pipeline (Dagster)")
console = Console()

_DEFAULT_HOST = os.environ.get("PRS_PIPELINE_HOST", "0.0.0.0")
_DEFAULT_PORT = int(os.environ.get("PRS_PIPELINE_PORT", "3010"))
_GRPC_CLEANUP_TIMEOUT = 5


def _kill_stale_dagster_children() -> None:
    """SIGKILL any leftover dagster gRPC server child processes."""
    import psutil

    current = psutil.Process()
    for child in current.children(recursive=True):
        cmdline = " ".join(child.cmdline())
        if "dagster" in cmdline and "grpc" in cmdline:
            console.print(f"[dim]Cleaning up stale gRPC subprocess (PID {child.pid})...[/dim]")
            child.kill()
            child.wait(timeout=_GRPC_CLEANUP_TIMEOUT)


def _execute_job(resolved_job: "dagster.JobDefinition") -> "dagster.ExecuteInProcessResult":  # type: ignore[name-defined]
    """Run a Dagster job in-process, handling gRPC server cleanup timeouts.

    Dagster's internal gRPC code-server subprocess sometimes hangs during
    teardown (subprocess.TimeoutExpired). The job itself completes fine;
    the timeout only affects the cleanup of the child process. We catch it
    and force-kill the stale subprocess so the CLI exits cleanly.
    """
    from dagster import DagsterInstance

    with DagsterInstance.get() as instance:
        result = resolved_job.execute_in_process(instance=instance)

    _kill_stale_dagster_children()
    return result


def _find_project_root() -> Path:
    """Walk upward from cwd to find the uv workspace root (contains [tool.uv.workspace])."""
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists() and "[tool.uv.workspace]" in pyproject.read_text():
            return candidate
    return current


def _setup_dagster_home() -> Path:
    """Set DAGSTER_HOME to a project-relative path and create dagster.yaml."""
    project_root = _find_project_root()
    dagster_home = project_root / "data" / "output" / "dagster"
    dagster_home.mkdir(parents=True, exist_ok=True)
    os.environ["DAGSTER_HOME"] = str(dagster_home)

    yaml_path = dagster_home / "dagster.yaml"
    if not yaml_path.exists():
        yaml_path.write_text("telemetry:\n  enabled: false\n")
        console.print(f"[dim]Created {yaml_path}[/dim]")

    return dagster_home


def _kill_port(port: int) -> None:
    """Kill any process listening on the given TCP port."""
    result = subprocess.run(
        ["lsof", "-t", f"-iTCP:{port}"],
        capture_output=True, text=True,
    )
    pids = [int(p) for p in result.stdout.strip().splitlines() if p.strip()]
    if pids:
        console.print(f"[yellow]Port {port} in use by PIDs {pids}, terminating...[/yellow]")
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        result2 = subprocess.run(
            ["lsof", "-t", f"-iTCP:{port}"],
            capture_output=True, text=True,
        )
        for pid in [int(p) for p in result2.stdout.strip().splitlines() if p.strip()]:
            os.kill(pid, signal.SIGKILL)


def _cancel_orphaned_runs() -> None:
    """Cancel any runs stuck in STARTED or NOT_STARTED from a previous session."""
    from dagster import DagsterInstance, DagsterRunStatus, RunsFilter

    with DagsterInstance.get() as instance:
        stuck = instance.get_run_records(
            filters=RunsFilter(statuses=[DagsterRunStatus.STARTED, DagsterRunStatus.NOT_STARTED])
        )
        for record in stuck:
            console.print(
                f"[yellow]Cancelling orphaned run {record.dagster_run.run_id[:8]}...[/yellow]"
            )
            instance.report_run_canceled(
                record.dagster_run,
                message="Orphaned run from previous session",
            )


def _set_pipeline_env(
    panel: str,
    test: int = 0,
    test_ids: str | None = None,
    no_cache: bool = False,
) -> bool:
    """Set PRS_PIPELINE_* env vars. Returns True if in test mode."""
    os.environ["PRS_PIPELINE_PANEL"] = panel
    if no_cache:
        os.environ["PRS_PIPELINE_NO_CACHE"] = "1"
        console.print("[yellow]NO-CACHE MODE: all assets will re-download and recompute.[/yellow]")
    else:
        os.environ.pop("PRS_PIPELINE_NO_CACHE", None)
    if test > 0:
        os.environ["PRS_PIPELINE_TEST_IDS"] = f"random:{test}"
        console.print(f"[yellow]TEST MODE: will score {test} randomly selected PGS IDs.[/yellow]")
        return True
    if test_ids:
        os.environ["PRS_PIPELINE_TEST_IDS"] = test_ids
        ids = [s.strip() for s in test_ids.split(",") if s.strip()]
        console.print(f"[yellow]TEST MODE: will score {len(ids)} specified PGS IDs: {ids}[/yellow]")
        return True
    return False


def _print_reference_audit_summary(panel: str) -> None:
    """Print the compact audit summary for headless CLI users."""
    from just_prs.scoring import resolve_cache_dir

    cache_dir = Path(os.environ.get("PRS_CACHE_DIR", "")) if os.environ.get("PRS_CACHE_DIR") else resolve_cache_dir()
    percentiles_dir = cache_dir / "percentiles"
    if os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip():
        percentiles_dir = percentiles_dir / "test"
    summary_path = percentiles_dir / f"{panel}_distribution_audit_summary.json"
    if not summary_path.exists():
        console.print(f"[yellow]Audit summary file not found: {summary_path}[/yellow]")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    input_pgs_ids = int(summary.get("input_pgs_ids", 0) or 0)
    failed_pgs_ids = len(summary.get("fully_removed_pgs_ids", []) or [])
    issue_counts_by_severity = summary.get("issue_counts_by_severity", {})
    warning_issue_rows = (
        int(issue_counts_by_severity.get("WARN", 0) or 0)
        if isinstance(issue_counts_by_severity, dict)
        else 0
    )
    error_issue_rows = (
        int(issue_counts_by_severity.get("ERROR", 0) or 0)
        if isinstance(issue_counts_by_severity, dict)
        else 0
    )
    issue_counts = summary.get("issue_counts_by_type", {})
    passed_or_warning_pgs_ids = max(input_pgs_ids - failed_pgs_ids, 0)

    console.print("\n[bold]Reference percentile audit summary[/bold]")
    console.print(f"  Panel: [cyan]{summary.get('panel', panel)}[/cyan]")
    console.print(f"  PGS IDs passing quarantine: [green]{passed_or_warning_pgs_ids}[/green]")
    console.print(f"  PGS IDs failed/quarantined: [red]{failed_pgs_ids}[/red]")
    console.print(f"  Issue rows: [red]{error_issue_rows} error[/red], [yellow]{warning_issue_rows} warning[/yellow]")
    if isinstance(issue_counts, dict) and issue_counts:
        console.print(f"  Issue counts by type: {issue_counts}")
    console.print(f"  Summary file: [dim]{summary_path}[/dim]")


@app.command()
def run(
    test: Annotated[int, typer.Option(help="Pick N random PGS IDs instead of all.")] = 0,
    test_ids: Annotated[Optional[str], typer.Option(help="Comma-separated PGS IDs to score.")] = None,
    panel: Annotated[str, typer.Option(help="Reference panel (1000g or hgdp_1kg).")] = "1000g",
    job: Annotated[str, typer.Option(help="Job to run: full_pipeline, download_reference_data, score_and_push, catalog_pipeline, metadata_pipeline, ld_proxy_pipeline, reference_allele_pipeline, reference_percentile_audit_job.")] = "full_pipeline",
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Ignore on-disk caches and re-download/recompute everything.")] = False,
    headless: Annotated[bool, typer.Option("--headless", help="Run in-process without Dagster UI.")] = False,
    host: Annotated[str, typer.Option(help="Bind address for the Dagster webserver (UI mode only).")] = _DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port for the Dagster webserver (UI mode only).")] = _DEFAULT_PORT,
) -> None:
    """Run a pipeline job with Dagster UI by default.

    \b
    Default behavior launches Dagster UI so you can monitor execution
    live. The startup sensor submits the selected job automatically.

    Use ``--headless`` to run in-process without UI.
    Use ``--no-cache`` to force a full re-run.
    """
    if test and test_ids:
        console.print("[red]Cannot use --test and --test-ids together.[/red]")
        raise typer.Exit(code=1)

    dagster_home = _setup_dagster_home()
    is_test_run = _set_pipeline_env(panel, test, test_ids, no_cache=no_cache)

    os.environ["PRS_PIPELINE_STARTUP_JOB"] = job

    if headless:
        _cancel_orphaned_runs()
        console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
        console.print("[yellow]HEADLESS MODE — no Dagster UI.[/yellow]")

        from prs_pipeline.definitions import defs

        resolved_job = defs.get_job_def(job)
        console.print(f"\n[bold]Running job: {job}[/bold]")
        console.print(f"[dim]{resolved_job.description or ''}[/dim]\n")

        result = _execute_job(resolved_job)

        if result.success:
            if job == "reference_percentile_audit_job":
                _print_reference_audit_summary(panel)
            console.print(f"\n[green bold]Job '{job}' completed successfully.[/green bold]")
        else:
            console.print(f"\n[red bold]Job '{job}' failed.[/red bold]")
            for event in result.all_events:
                if event.is_failure:
                    console.print(f"  [red]{event.message}[/red]")
            raise typer.Exit(code=1)
        return

    if no_cache or is_test_run:
        os.environ["PRS_PIPELINE_STARTUP_REQUEST_ID"] = uuid.uuid4().hex

    _kill_port(port)
    _cancel_orphaned_runs()
    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
    console.print(f"[bold green]Dagster UI:[/bold green] http://{host}:{port}")
    console.print(f"[bold]Job '{job}' will be submitted automatically on startup.[/bold]\n")

    dagster_bin = str(Path(sys.executable).parent / "dagster")
    os.execvp(dagster_bin, [
        "dagster", "dev",
        "-m", "prs_pipeline.definitions",
        "--host", host,
        "--port", str(port),
    ])


@app.command()
def launch(
    host: Annotated[str, typer.Option(help="Bind address for the Dagster webserver.")] = _DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port for the Dagster webserver.")] = _DEFAULT_PORT,
    test: Annotated[int, typer.Option(help="Pick N random PGS IDs instead of all.")] = 0,
    test_ids: Annotated[Optional[str], typer.Option(help="Comma-separated PGS IDs to score.")] = None,
    panel: Annotated[str, typer.Option(help="Reference panel (1000g or hgdp_1kg).")] = "1000g",
) -> None:
    """Start the Dagster UI webserver.

    \b
    Launches the Dagster dev server with full monitoring UI.
    The startup sensor will automatically submit the full_pipeline job
    if any assets are unmaterialized. If all assets are already cached,
    no run is submitted. You can trigger jobs manually from the UI.

    Equivalent to ``pipeline run`` without ``--headless``.
    """
    if test and test_ids:
        console.print("[red]Cannot use --test and --test-ids together.[/red]")
        raise typer.Exit(code=1)

    dagster_home = _setup_dagster_home()
    _set_pipeline_env(panel, test, test_ids)
    _kill_port(port)
    _cancel_orphaned_runs()

    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
    console.print(f"[bold]Dagster UI:[/bold] http://{host}:{port}")
    console.print(f"[dim]The startup sensor will submit jobs only if assets are missing.[/dim]")
    console.print(f"[dim]Use the UI to trigger jobs, or run 'pipeline run' in another terminal.[/dim]\n")

    dagster_bin = str(Path(sys.executable).parent / "dagster")
    os.execvp(dagster_bin, [
        "dagster", "dev",
        "-m", "prs_pipeline.definitions",
        "--host", host,
        "--port", str(port),
    ])


@app.command()
def catalog(
    panel: Annotated[str, typer.Option(help="Reference panel (1000g or hgdp_1kg).")] = "1000g",
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Ignore on-disk caches and re-download/recompute everything.")] = False,
    headless: Annotated[bool, typer.Option("--headless", help="Run catalog_pipeline in-process without Dagster UI.")] = False,
    host: Annotated[str, typer.Option(help="Bind address for the Dagster webserver (UI mode only).")] = _DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port for the Dagster webserver (UI mode only).")] = _DEFAULT_PORT,
) -> None:
    """Run the catalog pipeline with Dagster UI by default.

    \b
    Default behavior launches Dagster UI and sets startup target to
    ``catalog_pipeline`` so you can monitor execution live.

    Use ``--headless`` to run in-process without UI.
    """
    dagster_home = _setup_dagster_home()
    _set_pipeline_env(panel, no_cache=no_cache)
    os.environ["PRS_PIPELINE_STARTUP_JOB"] = "catalog_pipeline"

    if headless:
        _cancel_orphaned_runs()
        console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
        console.print("[bold]Job:[/bold] catalog_pipeline (scoring parquets + metadata → HF)\n")

        from prs_pipeline.definitions import defs

        resolved_job = defs.get_job_def("catalog_pipeline")
        console.print(f"[dim]{resolved_job.description or ''}[/dim]\n")

        result = _execute_job(resolved_job)

        if result.success:
            console.print("\n[green bold]Job 'catalog_pipeline' completed successfully.[/green bold]")
        else:
            console.print("\n[red bold]Job 'catalog_pipeline' failed.[/red bold]")
            for event in result.all_events:
                if event.is_failure:
                    console.print(f"  [red]{event.message}[/red]")
            raise typer.Exit(code=1)
        return

    if no_cache:
        os.environ["PRS_PIPELINE_STARTUP_REQUEST_ID"] = uuid.uuid4().hex

    _kill_port(port)
    _cancel_orphaned_runs()
    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
    console.print(f"[bold green]Dagster UI:[/bold green] http://{host}:{port}")
    console.print("[bold]Job 'catalog_pipeline' will be submitted automatically on startup.[/bold]\n")

    dagster_bin = str(Path(sys.executable).parent / "dagster")
    os.execvp(dagster_bin, [
        "dagster", "dev",
        "-m", "prs_pipeline.definitions",
        "--host", host,
        "--port", str(port),
    ])


@app.command()
def audit(
    panel: Annotated[str, typer.Option(help="Reference panel (1000g or hgdp_1kg).")] = "1000g",
    test: Annotated[bool, typer.Option("--test", help="Audit test-run outputs under percentiles/test.")] = False,
    headless: Annotated[bool, typer.Option("--headless", help="Run reference_percentile_audit_job in-process without Dagster UI.")] = False,
    host: Annotated[str, typer.Option(help="Bind address for the Dagster webserver (UI mode only).")] = _DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port for the Dagster webserver (UI mode only).")] = _DEFAULT_PORT,
) -> None:
    """Audit reference percentile distributions with Dagster UI by default.

    \b
    This runs the ``reference_percentile_audit_job`` Dagster job, which audits
    cached or HuggingFace-pulled percentile parquets and writes the audit
    sidecars without recomputing reference scores.
    """
    dagster_home = _setup_dagster_home()
    _set_pipeline_env(panel, test_ids="audit-test" if test else None)
    os.environ["PRS_PIPELINE_STARTUP_JOB"] = "reference_percentile_audit_job"

    if headless:
        _cancel_orphaned_runs()
        console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
        console.print("[bold]Job:[/bold] reference_percentile_audit_job (audit cached/HF percentiles)\n")

        from prs_pipeline.definitions import defs

        resolved_job = defs.get_job_def("reference_percentile_audit_job")
        console.print(f"[dim]{resolved_job.description or ''}[/dim]\n")

        result = _execute_job(resolved_job)

        if result.success:
            _print_reference_audit_summary(panel)
            console.print("\n[green bold]Job 'reference_percentile_audit_job' completed successfully.[/green bold]")
        else:
            console.print("\n[red bold]Job 'reference_percentile_audit_job' failed.[/red bold]")
            for event in result.all_events:
                if event.is_failure:
                    console.print(f"  [red]{event.message}[/red]")
            raise typer.Exit(code=1)
        return

    os.environ["PRS_PIPELINE_STARTUP_REQUEST_ID"] = uuid.uuid4().hex
    _kill_port(port)
    _cancel_orphaned_runs()
    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
    console.print(f"[bold green]Dagster UI:[/bold green] http://{host}:{port}")
    console.print("[bold]Job 'reference_percentile_audit_job' will be submitted automatically on startup.[/bold]\n")

    dagster_bin = str(Path(sys.executable).parent / "dagster")
    os.execvp(dagster_bin, [
        "dagster", "dev",
        "-m", "prs_pipeline.definitions",
        "--host", host,
        "--port", str(port),
    ])


@app.command(name="reference-allele")
def reference_allele(
    panel: Annotated[str, typer.Option(help="Reference panel (1000g or hgdp_1kg).")] = "1000g",
    headless: Annotated[bool, typer.Option("--headless", help="Run reference_allele_pipeline in-process without Dagster UI.")] = False,
    host: Annotated[str, typer.Option(help="Bind address for the Dagster webserver (UI mode only).")] = _DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port for the Dagster webserver (UI mode only).")] = _DEFAULT_PORT,
) -> None:
    """Precompute + publish the reference-allele universe (Dagster UI by default).

    \b
    Runs the ``reference_allele_pipeline`` job: faidx the GRCh38 FASTA + reference
    panel .pvar to resolve REF at every catalog scoring position lacking a
    reference_allele, write percentiles/reference_allele_universe.parquet, and push
    it to HuggingFace. Recovers genome-wide WGS coverage at runtime without shipping
    the 3 GB genome. Memory-sensitive (parses the FASTA + 75M-row pvar) — prefer a
    machine with adequate RAM.
    """
    dagster_home = _setup_dagster_home()
    _set_pipeline_env(panel)
    os.environ["PRS_PIPELINE_STARTUP_JOB"] = "reference_allele_pipeline"

    if headless:
        _cancel_orphaned_runs()
        console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
        console.print("[bold]Job:[/bold] reference_allele_pipeline (precompute REF universe → HF)\n")

        from prs_pipeline.definitions import defs

        resolved_job = defs.get_job_def("reference_allele_pipeline")
        console.print(f"[dim]{resolved_job.description or ''}[/dim]\n")

        result = _execute_job(resolved_job)
        if result.success:
            console.print("\n[green bold]Job 'reference_allele_pipeline' completed successfully.[/green bold]")
        else:
            console.print("\n[red bold]Job 'reference_allele_pipeline' failed.[/red bold]")
            for event in result.all_events:
                if event.is_failure:
                    console.print(f"  [red]{event.message}[/red]")
            raise typer.Exit(code=1)
        return

    os.environ["PRS_PIPELINE_STARTUP_REQUEST_ID"] = uuid.uuid4().hex
    _kill_port(port)
    _cancel_orphaned_runs()
    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
    console.print(f"[bold green]Dagster UI:[/bold green] http://{host}:{port}")
    console.print("[bold]Job 'reference_allele_pipeline' will be submitted automatically on startup.[/bold]\n")

    dagster_bin = str(Path(sys.executable).parent / "dagster")
    os.execvp(dagster_bin, [
        "dagster", "dev",
        "-m", "prs_pipeline.definitions",
        "--host", host,
        "--port", str(port),
    ])


@app.command(name="ld-proxy")
def ld_proxy(
    panel: Annotated[str, typer.Option(help="Reference panel (1000g or hgdp_1kg).")] = "1000g",
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Rebuild LD proxy tables from scratch.")] = False,
    headless: Annotated[bool, typer.Option("--headless", help="Run ld_proxy_pipeline in-process without Dagster UI.")] = False,
    full_catalog: Annotated[bool, typer.Option("--full-catalog", help="Build LD proxies for the full PGS catalog via the resumable per-PGS batch.")] = False,
    limit_targets: Annotated[Optional[int], typer.Option("--limit", "--limit-targets", help="Pilot mode: build LD proxies for the first N PGS IDs via the resumable per-PGS batch.")] = None,
    pgs_ids: Annotated[Optional[str], typer.Option("--pgs-ids", help="Comma-separated PGS IDs to build LD proxies for.")] = None,
    workers: Annotated[Optional[int], typer.Option("--workers", help="LD worker threads. Default: PRS_LD_MAX_WORKERS or 1.")] = None,
    memory_limit_gb: Annotated[Optional[float], typer.Option("--memory-limit-gb", help="Cooperative RSS limit for LD build; aborts before system OOM.")] = None,
    memory_limit_percent: Annotated[Optional[int], typer.Option("--memory-limit-percent", help="RSS limit as percent of system RAM. Default: 65.")] = None,
    duckdb_memory_limit: Annotated[Optional[str], typer.Option("--duckdb-memory-limit", help="DuckDB memory limit for LD staging queries. Default: 4GB or 25% of LD cap.")] = None,
    target_batch_size: Annotated[Optional[int], typer.Option("--target-batch-size", help="Untyped target positions per submitted LD batch. Default: 10000.")] = None,
    max_targets_per_chunk: Annotated[Optional[int], typer.Option("--max-targets-per-chunk", help="Target variants per in-memory correlation chunk. Default: 256.")] = None,
    chunk_size_bp: Annotated[Optional[int], typer.Option("--chunk-size-bp", help="Max genomic span per in-memory target chunk. Default: 250000.")] = None,
    host: Annotated[str, typer.Option(help="Bind address for the Dagster webserver (UI mode only).")] = _DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port for the Dagster webserver (UI mode only).")] = _DEFAULT_PORT,
) -> None:
    """Build LD-proxy tables for consumer genotyping arrays.

    \b
    Computes LD-proxy lookup tables for GSA v3 from the 1000G reference panel,
    then uploads to HuggingFace. The current pipeline builds GRCh38 only;
    GRCh37 should be added when build-matched typed positions are available.

    Default behavior launches Dagster UI. Use ``--headless`` for in-process.
    """
    scoped_modes = sum([
        1 if full_catalog else 0,
        1 if limit_targets is not None else 0,
        1 if pgs_ids else 0,
    ])
    if scoped_modes == 0:
        console.print(
            "[red bold]Refusing to launch a full-catalog LD-proxy build by default.[/red bold]\n"
            "LD proxy is a reference-panel correlation search, not a lightweight "
            "coverage table. Use [cyan]--pgs-ids PGS000001[/cyan] for a complete "
            "one-score run, [cyan]--limit N[/cyan] for a bounded pilot, "
            "or [cyan]--full-catalog[/cyan] when you intentionally want the full catalog."
        )
        raise typer.Exit(code=2)
    if scoped_modes > 1:
        console.print("[red]Use only one of --pgs-ids, --limit, or --full-catalog.[/red]")
        raise typer.Exit(code=2)

    dagster_home = _setup_dagster_home()
    _set_pipeline_env(panel, no_cache=no_cache)
    os.environ["PRS_PIPELINE_STARTUP_JOB"] = "ld_proxy_pipeline"
    if full_catalog:
        os.environ["PRS_LD_FULL_CATALOG"] = "1"
        os.environ.pop("PRS_LD_LIMIT_TARGETS", None)
        os.environ.pop("PRS_LD_PGS_IDS", None)
    elif pgs_ids:
        ids = [pid.strip().upper() for pid in pgs_ids.split(",") if pid.strip()]
        os.environ["PRS_LD_PGS_IDS"] = ",".join(ids)
        os.environ.pop("PRS_LD_FULL_CATALOG", None)
        os.environ.pop("PRS_LD_LIMIT_TARGETS", None)
    else:
        os.environ["PRS_LD_LIMIT_TARGETS"] = str(limit_targets)
        os.environ.pop("PRS_LD_FULL_CATALOG", None)
        os.environ.pop("PRS_LD_PGS_IDS", None)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("PRS_LD_BLAS_THREADS", "1"))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("PRS_LD_BLAS_THREADS", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("PRS_LD_BLAS_THREADS", "1"))
    if workers is not None:
        os.environ["PRS_LD_MAX_WORKERS"] = str(workers)
    if memory_limit_gb is not None:
        os.environ["PRS_LD_MEMORY_LIMIT_GB"] = str(memory_limit_gb)
    if memory_limit_percent is not None:
        os.environ["PRS_LD_MEMORY_LIMIT_PERCENT"] = str(memory_limit_percent)
    if duckdb_memory_limit is not None:
        os.environ["PRS_LD_DUCKDB_MEMORY_LIMIT"] = duckdb_memory_limit
    if target_batch_size is not None:
        os.environ["PRS_LD_TARGET_BATCH_SIZE"] = str(target_batch_size)
    if max_targets_per_chunk is not None:
        os.environ["PRS_LD_MAX_TARGETS_PER_CHUNK"] = str(max_targets_per_chunk)
    if chunk_size_bp is not None:
        os.environ["PRS_LD_CHUNK_SIZE_BP"] = str(chunk_size_bp)

    if full_catalog:
        console.print(
            "[yellow]Full-catalog LD proxy runs are resumable per-PGS batches, "
            "but should still be launched under OS containment, e.g. "
            "systemd-run --user --scope -p MemoryMax=24G uv run pipeline ld-proxy --full-catalog[/yellow]"
        )

    console.print(
        "[dim]LD guardrails: "
        f"workers={os.environ.get('PRS_LD_MAX_WORKERS', '1')}, "
        f"memory_limit_gb={os.environ.get('PRS_LD_MEMORY_LIMIT_GB', 'auto')}, "
        f"memory_limit_percent={os.environ.get('PRS_LD_MEMORY_LIMIT_PERCENT', '65')}, "
        f"duckdb_memory_limit={os.environ.get('PRS_LD_DUCKDB_MEMORY_LIMIT', 'auto')}, "
        f"pgs_ids={os.environ.get('PRS_LD_PGS_IDS', '')}, "
        f"limit_targets={os.environ.get('PRS_LD_LIMIT_TARGETS', 'full-catalog')}, "
        f"target_batch_size={os.environ.get('PRS_LD_TARGET_BATCH_SIZE', '10000')}, "
        f"max_targets_per_chunk={os.environ.get('PRS_LD_MAX_TARGETS_PER_CHUNK', '256')}, "
        f"chunk_size_bp={os.environ.get('PRS_LD_CHUNK_SIZE_BP', '250000')}[/dim]"
    )

    if headless:
        _cancel_orphaned_runs()
        console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
        console.print("[bold]Job:[/bold] ld_proxy_pipeline (build LD-proxy tables → HF)\n")

        from prs_pipeline.definitions import defs

        resolved_job = defs.get_job_def("ld_proxy_pipeline")
        console.print(f"[dim]{resolved_job.description or ''}[/dim]\n")

        result = _execute_job(resolved_job)

        if result.success:
            console.print("\n[green bold]Job 'ld_proxy_pipeline' completed successfully.[/green bold]")
        else:
            console.print("\n[red bold]Job 'ld_proxy_pipeline' failed.[/red bold]")
            for event in result.all_events:
                if event.is_failure:
                    console.print(f"  [red]{event.message}[/red]")
            raise typer.Exit(code=1)
        return

    if no_cache:
        os.environ["PRS_PIPELINE_STARTUP_REQUEST_ID"] = uuid.uuid4().hex

    _kill_port(port)
    _cancel_orphaned_runs()
    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
    console.print(f"[bold green]Dagster UI:[/bold green] http://{host}:{port}")
    console.print("[bold]Job 'ld_proxy_pipeline' will be submitted automatically on startup.[/bold]\n")

    dagster_bin = str(Path(sys.executable).parent / "dagster")
    os.execvp(dagster_bin, [
        "dagster", "dev",
        "-m", "prs_pipeline.definitions",
        "--host", host,
        "--port", str(port),
    ])


@app.command()
def clean(
    dagster_home: Annotated[Optional[str], typer.Option(help="Override DAGSTER_HOME path.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be cleaned without doing it.")] = False,
) -> None:
    """Cancel stuck runs from the Dagster DB."""
    from dagster import DagsterInstance, DagsterRunStatus, RunsFilter

    project_root = _find_project_root()
    home = Path(dagster_home) if dagster_home else project_root / "data" / "output" / "dagster"
    os.environ["DAGSTER_HOME"] = str(home)

    with DagsterInstance.get() as instance:
        cancel_statuses = [
            DagsterRunStatus.QUEUED,
            DagsterRunStatus.NOT_STARTED,
            DagsterRunStatus.STARTED,
        ]
        cancelled = 0
        for status in cancel_statuses:
            records = instance.get_run_records(
                filters=RunsFilter(statuses=[status]),
            )
            for rec in records:
                if not dry_run:
                    instance.report_run_canceled(
                        rec.dagster_run,
                        message="Cancelled by pipeline clean command",
                    )
                cancelled += 1

        if cancelled:
            label = "Would cancel" if dry_run else "Cancelled"
            console.print(f"[yellow]{label} {cancelled} queued/in-progress runs.[/yellow]")
        else:
            console.print("[green]No stuck runs found.[/green]")

        if dry_run:
            console.print("\n[dim]Run without --dry-run to execute cleanup.[/dim]")


@app.command()
def status(
    panel: Annotated[str, typer.Option("--panel", help="Reference panel to check.")] = "1000g",
    cache_dir: Annotated[Optional[str], typer.Option("--cache-dir", help="Override cache directory.")] = None,
    test: Annotated[bool, typer.Option("--test", help="Show status of test run outputs.")] = False,
) -> None:
    """Show scoring status by reading the quality report parquet."""
    import polars as pl
    from prs_pipeline.resources import CacheDirResource

    resource = CacheDirResource(cache_dir=cache_dir or "")
    base = resource.get_path()
    percentiles_dir = base / "percentiles"
    if test:
        percentiles_dir = percentiles_dir / "test"
    quality_path = percentiles_dir / f"{panel}_quality.parquet"

    if not quality_path.exists():
        console.print(f"[yellow]No quality report found at {quality_path}.[/yellow]")
        if test:
            console.print("Run 'pipeline run --test N' first.")
        else:
            console.print("Run 'pipeline run' or 'prs reference score-batch' first.")
        raise typer.Exit(code=1)

    df = pl.read_parquet(quality_path)
    dist_path = percentiles_dir / f"{panel}_distributions.parquet"
    scores_dir = base / "reference_scores" / panel

    label = f"panel {panel} — TEST" if test else f"panel {panel}"
    console.print(f"\n[bold]Scoring status for {label} ({df.height} PGS IDs):[/bold]")
    console.print(f"  Quality report:  {quality_path}")
    console.print(f"  Distributions:   {dist_path}")
    console.print(f"  Per-PGS scores:  {scores_dir}/{{PGS_ID}}/scores.parquet\n")

    status_counts = df.group_by("status").len().sort("len", descending=True)
    for row in status_counts.iter_rows(named=True):
        style = {"ok": "green", "failed": "red", "low_match": "yellow", "zero_variance": "yellow"}.get(
            row["status"], "dim"
        )
        console.print(f"  [{style}]{row['status']}:[/{style}] {row['len']}")

    failed = df.filter(pl.col("status") == "failed")
    if failed.height > 0:
        console.print(f"\n[red bold]Failed IDs ({failed.height}):[/red bold]")
        for row in failed.head(30).iter_rows(named=True):
            err = (row.get("error") or "unknown")[:100]
            console.print(f"  {row['pgs_id']}  [dim]{err}[/dim]")
        if failed.height > 30:
            console.print(f"  [dim]... and {failed.height - 30} more[/dim]")

        all_failed_ids = failed["pgs_id"].to_list()
        console.print(f"\n[dim]Failed IDs for --pgs-ids:[/dim]")
        console.print(f"  {','.join(all_failed_ids[:20])}")
        if len(all_failed_ids) > 20:
            console.print(f"  [dim]... ({len(all_failed_ids)} total)[/dim]")


@app.command(name="ancestry-model")
def ancestry_model(
    panels: Annotated[str, typer.Option(help="Comma-separated panels to build (1000g,hgdp_1kg).")] = "1000g,hgdp_1kg",
    builds: Annotated[str, typer.Option(help="Comma-separated builds (GRCh38 only; GRCh37 samples lift to GRCh38 at inference).")] = "GRCh38",
    headless: Annotated[bool, typer.Option("--headless", help="Run ancestry_model_pipeline in-process without Dagster UI.")] = False,
    host: Annotated[str, typer.Option(help="Bind address for the Dagster webserver (UI mode only).")] = _DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port for the Dagster webserver (UI mode only).")] = _DEFAULT_PORT,
) -> None:
    """Build + publish the sample-ancestry reference-PCA model (Dagster UI by default).

    \b
    Runs the ``ancestry_model_pipeline`` job: plink2 QC + LD-prune (build-time only) of
    the PGS Catalog reference panels, numpy SVD, KNN leave-one-out validation, then
    publish the small per-(panel,build) model artifacts to HuggingFace. Runtime ancestry
    inference is pure-Python and never needs plink2. Set --panels/--builds to limit scope
    (the HGDP+1kGP panel is a ~16 GB download on first build).
    """
    dagster_home = _setup_dagster_home()
    os.environ["PRS_ANCESTRY_PANELS"] = panels
    os.environ["PRS_ANCESTRY_BUILDS"] = builds
    os.environ["PRS_PIPELINE_STARTUP_JOB"] = "ancestry_model_pipeline"

    if headless:
        _cancel_orphaned_runs()
        console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
        console.print(f"[bold]Job:[/bold] ancestry_model_pipeline (panels={panels}, builds={builds})\n")

        from prs_pipeline.definitions import defs

        resolved_job = defs.get_job_def("ancestry_model_pipeline")
        console.print(f"[dim]{resolved_job.description or ''}[/dim]\n")

        result = _execute_job(resolved_job)
        if result.success:
            console.print("\n[green bold]Job 'ancestry_model_pipeline' completed successfully.[/green bold]")
        else:
            console.print("\n[red bold]Job 'ancestry_model_pipeline' failed.[/red bold]")
            for event in result.all_events:
                if event.is_failure:
                    console.print(f"  [red]{event.message}[/red]")
            raise typer.Exit(code=1)
        return

    os.environ["PRS_PIPELINE_STARTUP_REQUEST_ID"] = uuid.uuid4().hex
    _kill_port(port)
    _cancel_orphaned_runs()
    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
    console.print(f"[bold green]Dagster UI:[/bold green] http://{host}:{port}")
    console.print("[bold]Job 'ancestry_model_pipeline' will be submitted automatically on startup.[/bold]\n")

    dagster_bin = str(Path(sys.executable).parent / "dagster")
    os.execvp(dagster_bin, [
        "dagster", "dev",
        "-m", "prs_pipeline.definitions",
        "--host", host,
        "--port", str(port),
    ])
