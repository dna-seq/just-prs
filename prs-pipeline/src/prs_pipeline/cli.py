"""CLI for prs-pipeline: launch the Dagster reference-panel pipeline."""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from rich.console import Console

console = Console()


def _find_project_root() -> Path:
    """Walk upward from cwd to find the uv workspace root (contains [tool.uv.workspace])."""
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists() and "[tool.uv.workspace]" in pyproject.read_text():
            return candidate
    return current


def _ensure_dagster_yaml(dagster_home: Path) -> None:
    """Write dagster.yaml with telemetry disabled and sensors enabled."""
    yaml_path = dagster_home / "dagster.yaml"
    if yaml_path.exists():
        return
    yaml_path.write_text(
        "telemetry:\n"
        "  enabled: false\n"
        "\n"
        "sensors:\n"
        "  use_threads: true\n"
        "  num_workers: 4\n"
    )
    console.print(f"[dim]Created {yaml_path}[/dim]")


def _kill_port(port: int) -> None:
    """Kill any process listening on the given TCP port (SIGTERM, then SIGKILL)."""
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


def _wait_for_port(host: str, port: int, timeout: int = 60) -> bool:
    """Block until a TCP connection to host:port succeeds, or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def _run_job_in_background(job_name: str) -> None:
    """Execute a named asset job in a background thread via execute_in_process."""
    from dagster import DagsterInstance
    from prs_pipeline.definitions import defs

    job_def = defs.get_job_def(job_name)

    def _execute() -> None:
        with DagsterInstance.get() as instance:
            console.print(f"[green]Starting job '{job_name}' in background...[/green]")
            result = job_def.execute_in_process(instance=instance)
            if result.success:
                console.print(f"[green]Job '{job_name}' completed successfully.[/green]")
            else:
                console.print(f"[red]Job '{job_name}' failed.[/red]")

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()


def launch(host: str = "0.0.0.0", port: int = 3010) -> None:
    """Start Dagster and run the full reference panel pipeline end-to-end.

    One command triggers the entire Map-Reduce flow:
      1. Start Dagster dev server with sensors enabled.
      2. Submit ``download_reference_data`` (downloads reference panel + registers PGS IDs).
      3. ``score_all_partitions_sensor`` auto-triggers per_pgs_scores for all ~5000 PGS IDs.
      4. ``aggregate_when_done_sensor`` auto-triggers aggregation + HuggingFace upload when done.

    The Dagster UI at http://{host}:{port} shows live progress for all stages.
    """
    project_root = _find_project_root()
    dagster_home = project_root / "data" / "output" / "dagster"
    dagster_home.mkdir(parents=True, exist_ok=True)
    os.environ["DAGSTER_HOME"] = str(dagster_home)
    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")

    _ensure_dagster_yaml(dagster_home)
    _kill_port(port)
    _cancel_orphaned_runs()

    dagster_bin = str(Path(sys.executable).parent / "dagster")
    console.print(f"Starting Dagster dev at http://{host}:{port} ...")
    proc = subprocess.Popen(
        [dagster_bin, "dev", "-m", "prs_pipeline.definitions",
         "--host", host, "--port", str(port)],
    )

    def _forward_signal(sig: int, _frame: object) -> None:
        proc.send_signal(sig)

    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)

    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    if _wait_for_port(connect_host, port, timeout=90):
        console.print(f"[green]Dagster UI ready at http://{host}:{port}[/green]")
        console.print(
            "[bold]Full pipeline flow:[/bold]\n"
            "  1. download_reference_data  -> downloads 1000G panel + registers PGS IDs\n"
            "  2. score_all_partitions_sensor -> auto-launches ~5000 PLINK2 scoring runs\n"
            "  3. aggregate_when_done_sensor  -> auto-aggregates + pushes to HuggingFace\n"
        )
        _run_job_in_background("download_reference_data")
    else:
        console.print("[yellow]Timed out waiting for Dagster webserver -- UI may still be starting.[/yellow]")

    proc.wait()
