"""CLI for prs-pipeline: launch or run the Dagster reference-panel pipeline."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(help="PRS reference-panel pipeline (Dagster)")
console = Console()


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
) -> bool:
    """Set PRS_PIPELINE_* env vars. Returns True if in test mode."""
    os.environ["PRS_PIPELINE_PANEL"] = panel
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
    job: Annotated[str, typer.Option(help="Job to run: full_pipeline, download_reference_data, score_and_push, metadata_pipeline.")] = "full_pipeline",
) -> None:
    """Execute a pipeline job directly in-process (no Dagster UI).

    This is the simplest way to run the pipeline. It materializes assets
    in dependency order, skipping any that are already up-to-date on disk.
    """
    if test and test_ids:
        console.print("[red]Cannot use --test and --test-ids together.[/red]")
        raise typer.Exit(code=1)

    dagster_home = _setup_dagster_home()
    is_test = _set_pipeline_env(panel, test, test_ids)
    _cancel_orphaned_runs()

    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")

    from prs_pipeline.definitions import defs

    resolved_job = defs.get_job_def(job)
    console.print(f"\n[bold]Running job: {job}[/bold]")
    console.print(f"[dim]{resolved_job.description or ''}[/dim]\n")

    result = resolved_job.execute_in_process(
        instance=__import__("dagster").DagsterInstance.get(),
    )

    if result.success:
        console.print(f"\n[green bold]Job '{job}' completed successfully.[/green bold]")
    else:
        console.print(f"\n[red bold]Job '{job}' failed.[/red bold]")
        for event in result.all_events:
            if event.is_failure:
                console.print(f"  [red]{event.message}[/red]")
        raise typer.Exit(code=1)


@app.command()
def launch(
    host: Annotated[str, typer.Option(help="Bind address for the Dagster webserver.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Port for the Dagster webserver.")] = 3010,
    test: Annotated[int, typer.Option(help="Pick N random PGS IDs instead of all.")] = 0,
    test_ids: Annotated[Optional[str], typer.Option(help="Comma-separated PGS IDs to score.")] = None,
    panel: Annotated[str, typer.Option(help="Reference panel (1000g or hgdp_1kg).")] = "1000g",
    run_now: Annotated[
        bool,
        typer.Option(
            "--run-now/--no-run-now",
            help=(
                "Request an immediate full_pipeline run on startup even if assets are "
                "already materialized."
            ),
        ),
    ] = True,
) -> None:
    """Start the Dagster UI webserver with auto-triggered pipeline.

    \b
    Launches the Dagster dev server with full monitoring UI.
    By default, launch requests one immediate ``full_pipeline`` run via
    the startup sensor. Use ``--no-run-now`` to only start the UI without
    submitting a run. You can also trigger jobs manually from the UI, or
    use ``pipeline run`` for headless execution.
    """
    if test and test_ids:
        console.print("[red]Cannot use --test and --test-ids together.[/red]")
        raise typer.Exit(code=1)

    dagster_home = _setup_dagster_home()
    _set_pipeline_env(panel, test, test_ids)
    if run_now:
        os.environ["PRS_PIPELINE_FORCE_RUN_ON_STARTUP"] = "1"
        os.environ["PRS_PIPELINE_STARTUP_RUN_KEY"] = f"pipeline_startup_{int(time.time())}"
        console.print("[yellow]Launch will force one full_pipeline run on startup.[/yellow]")
    else:
        os.environ["PRS_PIPELINE_FORCE_RUN_ON_STARTUP"] = "0"
    _kill_port(port)
    _cancel_orphaned_runs()

    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")
    console.print(f"[bold]Dagster UI:[/bold] http://{host}:{port}")
    console.print(f"[dim]Use the UI to trigger jobs, or run 'pipeline run' in another terminal.[/dim]\n")

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
