"""Scoring file viewer: PGS ID input, genome build selector, and scrollable grid."""

import reflex as rx
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar

from prs_ui.state import MetadataGridState


def scoring_panel() -> rx.Component:
    """Scoring file browsing panel: PGS ID input + genome build + scrollable grid."""
    return rx.vstack(
        rx.hstack(
            rx.input(
                value=MetadataGridState.pgs_id_input,
                on_change=MetadataGridState.set_pgs_id,
                placeholder="PGS000001",
                width="200px",
                size="2",
            ),
            rx.select(
                ["GRCh37", "GRCh38"],
                value=MetadataGridState.genome_build,
                on_change=MetadataGridState.set_genome_build,
                size="2",
            ),
            rx.button(
                "Load Scoring File",
                on_click=MetadataGridState.load_scoring,
                loading=MetadataGridState.lf_grid_loading,
                color_scheme="green",
                size="2",
            ),
            spacing="3",
            align="center",
        ),
        rx.cond(
            MetadataGridState.lf_grid_loaded,
            rx.vstack(
                lazyframe_grid_stats_bar(MetadataGridState),
                lazyframe_grid(
                    MetadataGridState,
                    height="calc(100vh - 260px)",
                    density="compact",
                    column_header_height=56,
                ),
                width="100%",
                spacing="2",
            ),
            rx.cond(
                MetadataGridState.lf_grid_loading,
                rx.center(rx.spinner(size="3"), padding="60px"),
                rx.center(
                    rx.text(
                        "Enter a PGS ID and click Load to stream variant weights.",
                        color="gray",
                        size="3",
                    ),
                    padding="60px",
                ),
            ),
        ),
        width="100%",
        spacing="3",
    )
