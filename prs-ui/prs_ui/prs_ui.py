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
from prs_ui.pages.traits import traits_panel
from prs_ui.state import AppState, ComputeGridState, TraitBrowserState


def _tab_label(icon: str, label: str, description: str) -> rx.Component:
    """Tab label with a compact explanation for non-specialist users."""
    return rx.hstack(
        rx.icon(icon, size=14),
        rx.text(label, size="2"),
        rx.tooltip(
            rx.icon("info", size=12, color="gray"),
            content=description,
        ),
        spacing="1",
        align="center",
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
                        _tab_label(
                            "calculator",
                            "Compute PRS",
                            "Upload a VCF, choose PGS Catalog scores, and compute "
                            "polygenic risk scores for one genome.",
                        ),
                        value="compute",
                    ),
                    rx.tabs.trigger(
                        _tab_label(
                            "layers",
                            "Browse by Trait",
                            "Start from a trait or disease name instead of a specific "
                            "PGS ID. The app finds related scores and computes them together.",
                        ),
                        value="traits",
                    ),
                    rx.tabs.trigger(
                        _tab_label(
                            "table",
                            "Metadata Sheets",
                            "Browse the raw PGS Catalog tables, such as score summaries, "
                            "publications, evaluation cohorts, and performance metadata.",
                        ),
                        value="metadata",
                    ),
                    rx.tabs.trigger(
                        _tab_label(
                            "file-text",
                            "Scoring File",
                            "Inspect the variant weights for one PGS ID and genome build. "
                            "This is the recipe the app uses to calculate a score.",
                        ),
                        value="scoring",
                    ),
                ),
                rx.tabs.content(
                    rx.box(compute_panel(), padding_top="12px"),
                    value="compute",
                ),
                rx.tabs.content(
                    rx.box(traits_panel(), padding_top="12px"),
                    value="traits",
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
    on_load=[ComputeGridState.initialize, TraitBrowserState.initialize_traits],
)


def main() -> None:
    """CLI entrypoint: launch the Reflex dev server."""
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent.parent)
    from reflex.reflex import cli
    cli(["run"])
