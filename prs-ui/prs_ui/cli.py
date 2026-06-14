"""CLI for prs-ui: launch the Reflex dev server."""

import os
import platform
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

console = Console()

_DEFAULT_HOST = "0.0.0.0"
_FRONTEND_PORT_ENV = "PRS_UI_PORT"
_BACKEND_PORT_ENV = "PRS_UI_BACKEND_PORT"
_DATA_DIR_ENV = "PRS_UI_DATA_DIR"
_PRESELECT_ENABLED_ENV = "PRS_UI_PRESELECT_ENABLED"
_PRESELECT_QUERY_ENV = "PRS_UI_PRESELECT_QUERY"

# Reflex bundles bun on first run and shells out to `unzip` to extract it on
# Linux (this includes WSL). Fresh WSL/Ubuntu installs frequently lack `unzip`,
# producing a cryptic ``SystemPackageMissingError`` deep inside Reflex. We detect
# it up front and print an actionable message instead.
_REQUIRED_SYSTEM_BINARIES = ("unzip",)


def _is_wsl() -> bool:
    """Return True when running under Windows Subsystem for Linux."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    return "microsoft" in platform.release().lower()


def _check_system_dependencies() -> None:
    """Verify required system binaries exist before handing off to Reflex.

    Reflex needs ``unzip`` to install its bundled ``bun`` runtime on Linux/WSL.
    When missing, exit early with a clear ``apt install`` hint rather than the
    opaque ``SystemPackageMissingError`` traceback Reflex would otherwise raise.
    """
    if platform.system() == "Windows":
        return

    missing = [name for name in _REQUIRED_SYSTEM_BINARIES if shutil.which(name) is None]
    if not missing:
        return

    packages = " ".join(missing)
    context = "WSL" if _is_wsl() else "this system"
    console.print(
        f"[bold red]Missing required system package(s):[/bold red] {packages}"
    )
    console.print(
        f"Reflex needs them to install its bundled bun runtime, but they are "
        f"not installed on {context}."
    )
    console.print("Install them with your package manager, for example:")
    console.print(f"  [bold]sudo apt-get update && sudo apt-get install -y {packages}[/bold]")
    console.print("Then re-run [bold]uv run ui[/bold].")
    sys.exit(1)


def _env_port(env_name: str) -> int | None:
    """Read port from env var, returning None if unset (lets Reflex auto-increment)."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError as exc:
        msg = f"{env_name} must be an integer port, got {raw!r}"
        raise ValueError(msg) from exc
    if port < 1 or port > 65535:
        msg = f"{env_name} must be between 1 and 65535, got {port}"
        raise ValueError(msg)
    return port


def _project_root_from(start: Path) -> Path | None:
    """Return the nearest uv workspace root at or above ``start``."""
    for parent in (start, *start.parents):
        if (parent / "uv.lock").exists() and (parent / "pyproject.toml").exists():
            return parent
    return None


def _default_data_dir(invocation_dir: Path) -> Path:
    """Default runtime data directory, stable across Reflex's cwd switch."""
    root = _project_root_from(invocation_dir)
    return (root or invocation_dir) / "data"


def _launch_reflex() -> None:
    """Start the Reflex dev server after environment setup.

    When ``PRS_UI_PORT`` / ``PRS_UI_BACKEND_PORT`` are set, those exact
    ports are used (no auto-increment).  When unset, Reflex picks the
    default (3000/8000) and auto-increments to the next free port if
    busy — so multiple Reflex apps can run side-by-side.
    """
    _check_system_dependencies()

    invocation_dir = Path.cwd()
    os.environ.setdefault(_DATA_DIR_ENV, str(_default_data_dir(invocation_dir)))

    prs_ui_dir = Path(__file__).resolve().parent.parent
    os.chdir(prs_ui_dir)

    host = os.environ.get("PRS_UI_HOST", _DEFAULT_HOST)
    frontend_port = _env_port(_FRONTEND_PORT_ENV)
    backend_port = _env_port(_BACKEND_PORT_ENV)

    console.print(f"[dim]Starting PRS UI from {prs_ui_dir}[/dim]")
    if frontend_port:
        console.print(f"[dim]Frontend port pinned to {frontend_port}[/dim]")
    if backend_port:
        console.print(f"[dim]Backend port pinned to {backend_port}[/dim]")
    console.print(f"[dim]Runtime data directory: {os.environ[_DATA_DIR_ENV]}[/dim]")

    from reflex import constants
    from reflex.reflex import _run
    from reflex_base.config import environment

    environment.REFLEX_COMPILE_CONTEXT.set(constants.CompileContext.RUN)
    _run(
        env=constants.Env.DEV,
        frontend_port=frontend_port,
        backend_port=backend_port,
        backend_host=host,
    )


def launch_ui() -> None:
    """Start the general Reflex dev server for prs-ui."""
    load_dotenv()
    os.environ.pop(_PRESELECT_ENABLED_ENV, None)
    _launch_reflex()


def launch_preselect_ui() -> None:
    """Start the UI with configured test VCF and score preselection enabled."""
    load_dotenv()
    os.environ[_PRESELECT_ENABLED_ENV] = "1"
    os.environ.setdefault(_PRESELECT_QUERY_ENV, "Type 1 diabetes (T1D)")

    preselect_vcf = os.environ.get("PRS_UI_PRESELECT_VCF", "").strip()
    preselect_query = os.environ.get(_PRESELECT_QUERY_ENV, "").strip()
    if preselect_vcf:
        console.print(f"[dim]Preselect VCF:[/dim] {preselect_vcf}")
    if preselect_query:
        console.print(f"[dim]Preselect query:[/dim] {preselect_query}")
    _launch_reflex()
