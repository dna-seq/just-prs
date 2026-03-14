"""CLI for prs-ui: launch the Reflex dev server."""

import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

console = Console()

_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 3000


def launch_ui() -> None:
    """Start the Reflex dev server for prs-ui."""
    load_dotenv()

    host = os.environ.get("PRS_UI_HOST", _DEFAULT_HOST)
    port = os.environ.get("PRS_UI_PORT", str(_DEFAULT_PORT))

    prs_ui_dir = Path(__file__).resolve().parent.parent
    console.print(f"[dim]Starting PRS UI from {prs_ui_dir}[/dim]")
    console.print(f"[bold green]PRS UI:[/bold green] http://{host}:{port}")
    os.chdir(prs_ui_dir)
    from reflex.reflex import cli as reflex_cli
    reflex_cli(["run"])
