"""CLI for prs-pipeline: launch or run the Dagster reference-panel pipeline."""

import os
import signal
import subprocess
import sys
import time
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


@app.command()
def run(
    test: Annotated[int, typer.Option(help="Pick N random PGS IDs instead of all.")] = 0,
    test_ids: Annotated[Optional[str], typer.Option(help="Comma-separated PGS IDs to score.")] = None,
    panel: Annotated[str, typer.Option(help="Reference panel (1000g or hgdp_1kg).")] = "1000g",
    job: Annotated[str, typer.Option(help="Job to run: full_pipeline, download_reference_data, score_and_push, catalog_pipeline, metadata_pipeline.")] = "full_pipeline",
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
    _set_pipeline_env(panel, test, test_ids, no_cache=no_cache)

    os.environ["PRS_PIPELINE_STARTUP_JOB"] = job
    os.environ["PRS_PIPELINE_FORCE_RUN"] = "1"

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
            console.print(f"\n[green bold]Job '{job}' completed successfully.[/green bold]")
        else:
            console.print(f"\n[red bold]Job '{job}' failed.[/red bold]")
            for event in result.all_events:
                if event.is_failure:
                    console.print(f"  [red]{event.message}[/red]")
            raise typer.Exit(code=1)
        return

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
    os.environ["PRS_PIPELINE_FORCE_RUN"] = "1"

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
