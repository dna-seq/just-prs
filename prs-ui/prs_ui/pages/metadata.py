"""Metadata browser page: tab buttons for each of the 7 PGS Catalog sheets."""

import reflex as rx
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar

from prs_ui.state import AppState, SHEET_LABELS, SHEET_NAMES


def _sheet_button(sheet: str) -> rx.Component:
    label = SHEET_LABELS.get(sheet, sheet)
    return rx.button(
        label,
        on_click=AppState.load_sheet(sheet),
        variant=rx.cond(AppState.selected_sheet == sheet, "solid", "outline"),
        color_scheme="blue",
        size="2",
    )


def metadata_panel() -> rx.Component:
    """Full metadata browsing panel: sheet selector buttons + grid."""
    return rx.vstack(
        rx.hstack(
            *[_sheet_button(s) for s in SHEET_NAMES],
            rx.separator(orientation="vertical", size="2"),
            rx.button(
                rx.icon("download", size=14),
                "Download Selected",
                on_click=AppState.download_selected_scoring_files,
                loading=AppState.lf_grid_loading,
                disabled=AppState.selected_pgs_ids.length() == 0,  # type: ignore[operator]
                color_scheme="green",
                size="2",
            ),
            wrap="wrap",
            spacing="2",
            align="center",
        ),
        rx.cond(
            AppState.lf_grid_loaded,
            rx.vstack(
                lazyframe_grid_stats_bar(AppState),
                lazyframe_grid(
                    AppState,
                    height="calc(100vh - 260px)",
                    density="compact",
                    column_header_height=56,
                    checkbox_selection=True,
                    on_row_selection_model_change=AppState.handle_scores_row_selection,
                ),
                width="100%",
                spacing="2",
            ),
            rx.cond(
                AppState.lf_grid_loading,
                rx.center(rx.spinner(size="3"), padding="60px"),
                rx.center(
                    rx.text(
                        "Select a metadata sheet above to load it.",
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
