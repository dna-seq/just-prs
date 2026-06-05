"""PRS UI: Reflex app for browsing PGS Catalog metadata and scoring files."""

import traceback

import reflex as rx
from reflex_base.event import EventSpec
from reflex.utils import console
from rich.markup import escape

from prs_ui.grid_style import data_grid_scroll_css
from prs_ui.pages.compute import compute_panel
from prs_ui.pages.metadata import metadata_panel
from prs_ui.pages.scoring import scoring_panel
from prs_ui.state import (
    AppState,
    ComputeGridState,
    GenomicGridState,
    TraitBrowserState,
)


def _safe_backend_exception_handler(exception: Exception) -> EventSpec:
    """Log backend exceptions without letting Rich parse traceback text as markup."""
    from reflex_components_sonner.toast import toast

    error = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    console.error(f"[Reflex Backend Exception]\n {escape(error)}\n")
    return toast(
        "An error occurred.",
        level="error",
        fallback_to_alert=True,
        description=f"{type(exception).__name__}: {exception}\nSee logs for details.",
        position="top-center",
        id="backend_error",
        style={"width": "500px", "white-space": "pre-wrap"},
    )


def index() -> rx.Component:
    return rx.box(
        data_grid_scroll_css(),
        rx.vstack(
            rx.hstack(
                rx.heading("PGS Catalog Browser", size="5"),
                rx.spacer(),
                rx.badge(AppState.status_message, variant="soft", size="2"),
                width="100%",
                align="center",
                padding_x="16px",
                padding_y="12px",
            ),
            rx.separator(),
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger(
                        rx.hstack(
                            rx.icon("calculator", size=14),
                            rx.text("Compute PRS"),
                            rx.tooltip(
                                rx.icon("info", size=13, color="gray"),
                                content="Upload genomic data (VCF or consumer array) once, then compute custom Polygenic Risk Scores by selecting individual PRS models or whole medical traits.",
                            ),
                            spacing="2",
                            align="center",
                        ),
                        value="compute",
                    ),
                    rx.tabs.trigger(
                        rx.hstack(
                            rx.icon("table", size=14),
                            rx.text("Metadata Sheets"),
                            rx.tooltip(
                                rx.icon("info", size=13, color="gray"),
                                content="Explore raw metadata sheets from the PGS Catalog database (including scores, performance metrics, and publication details).",
                            ),
                            spacing="2",
                            align="center",
                        ),
                        value="metadata",
                    ),
                    rx.tabs.trigger(
                        rx.hstack(
                            rx.icon("file-text", size=14),
                            rx.text("Scoring File"),
                            rx.tooltip(
                                rx.icon("info", size=13, color="gray"),
                                content="Inspect variant weights, alleles, and genomic positions directly from the PGS Catalog for any loaded model ID.",
                            ),
                            spacing="2",
                            align="center",
                        ),
                        value="scoring",
                    ),
                ),
                rx.tabs.content(
                    rx.box(compute_panel(), padding_top="12px"),
                    value="compute",
                ),
                rx.tabs.content(
                    rx.box(metadata_panel(), padding_top="12px"),
                    value="metadata",
                ),
                rx.tabs.content(
                    rx.box(scoring_panel(), padding_top="12px"),
                    value="scoring",
                ),
                value=AppState.active_tab,
                on_change=AppState.set_active_tab,  # type: ignore[arg-type]
                width="100%",
            ),
            width="100%",
            spacing="0",
        ),
        padding="16px",
        width="100%",
        min_height="100vh",
    )


app = rx.App(backend_exception_handler=_safe_backend_exception_handler)
app.add_page(
    index,
    title="PGS Catalog Browser",
    on_load=[
        GenomicGridState.initialize_source,
        ComputeGridState.initialize,
        TraitBrowserState.initialize_traits,
    ],
)


def main() -> None:
    """CLI entrypoint: launch the Reflex dev server."""
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent.parent)
    from reflex.reflex import cli
    cli(["run"])
