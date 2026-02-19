"""PRS UI: Reflex app for browsing PGS Catalog metadata and scoring files."""

import reflex as rx

from prs_ui.pages.compute import compute_panel
from prs_ui.pages.metadata import metadata_panel
from prs_ui.pages.scoring import scoring_panel
from prs_ui.state import AppState, ComputeGridState


def index() -> rx.Component:
    return rx.box(
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
                            "Compute PRS",
                            spacing="1",
                            align="center",
                        ),
                        value="compute",
                    ),
                    rx.tabs.trigger("Metadata Sheets", value="metadata"),
                    rx.tabs.trigger("Scoring File", value="scoring"),
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


app = rx.App(
    theme=rx.theme(appearance="light", accent_color="blue"),
)
app.add_page(index, title="PGS Catalog Browser", on_load=ComputeGridState.initialize)


def main() -> None:
    """CLI entrypoint: launch the Reflex dev server."""
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent.parent)
    from reflex.reflex import cli
    cli(["run"])
