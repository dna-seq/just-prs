"""CLI for prs-ui: launch the Reflex dev server."""

import os
from pathlib import Path

from rich.console import Console

console = Console()


def launch_ui() -> None:
    """Start the Reflex dev server for prs-ui."""
    prs_ui_dir = Path(__file__).resolve().parent.parent
    console.print(f"Starting PRS UI from {prs_ui_dir} ...")
    os.chdir(prs_ui_dir)
    from reflex.reflex import cli as reflex_cli
    reflex_cli(["run"])
