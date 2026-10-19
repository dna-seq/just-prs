"""Curated tab: browse score direction, review status, and trait clusters."""

import reflex as rx
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar

from prs_ui.curation_browser import (
    CLUSTER_DETAIL_FIELDS,
    CLUSTER_DETAIL_LABELS,
    SCORE_DETAIL_FIELDS,
    SCORE_DETAIL_LABELS,
)
from prs_ui.grid_style import data_grid_scroll_container
from prs_ui.state import CurationGridState


def _view_button(view: str, label: str) -> rx.Component:
    return rx.button(
        label,
        on_click=CurationGridState.set_curation_view(view),
        variant=rx.cond(CurationGridState.curation_view == view, "solid", "outline"),
        color_scheme="blue",
        size="2",
    )


def _status_button(status: str, label: str) -> rx.Component:
    return rx.button(
        label,
        on_click=CurationGridState.set_curation_status(status),
        variant=rx.cond(CurationGridState.curation_status == status, "solid", "outline"),
        color_scheme="gray",
        size="2",
    )


def _toolbar() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Resolved scores passed the quote and polarity checks. Human verified means a "
            "person confirmed the quote. Unreviewed means only the automated checks have "
            "passed. Needs review is still open: a missing quote, low-confidence evidence, "
            "or an unverified correction. Expand a row to read the quote. Select rows and "
            "send them to By PRS to compute those models.",
            size="2",
            color="gray",
            max_width="960px",
        ),
        rx.hstack(
            _view_button("scores", "Resolved"),
            _view_button("review", "Needs review"),
            _view_button("clusters", "Clusters"),
            rx.cond(
                CurationGridState.curation_view != "clusters",
                rx.hstack(
                    rx.separator(orientation="vertical", size="2"),
                    _status_button("all", "All"),
                    _status_button("human_verified", "Human verified"),
                    _status_button("agent_proposed", "Unreviewed"),
                    spacing="2",
                    align="center",
                ),
            ),
            rx.separator(orientation="vertical", size="2"),
            rx.button(
                "Reload",
                on_click=CurationGridState.load_curation,
                variant="outline",
                size="2",
            ),
            rx.button(
                "Use in By PRS",
                on_click=CurationGridState.use_selected_in_prs,
                color_scheme="green",
                size="2",
                disabled=(
                    (CurationGridState.curation_selected_ids.length() == 0)
                    | (CurationGridState.curation_view == "clusters")
                ),
            ),
            rx.button(
                "Clear",
                on_click=CurationGridState.clear_curation_selection,
                variant="outline",
                size="2",
                disabled=CurationGridState.curation_selected_ids.length() == 0,
            ),
            rx.cond(
                CurationGridState.curation_selected_ids.length() > 0,
                rx.badge(
                    CurationGridState.curation_selected_ids.length(),
                    " selected",
                    color_scheme="green",
                    size="2",
                ),
            ),
            wrap="wrap",
            spacing="2",
            align="center",
        ),
        rx.hstack(
            rx.badge(
                CurationGridState.n_verified.to(str) + " human verified",
                color_scheme="green",
                variant="soft",
                size="1",
            ),
            rx.badge(
                CurationGridState.n_unreviewed.to(str) + " unreviewed",
                color_scheme="orange",
                variant="soft",
                size="1",
            ),
            rx.badge(
                CurationGridState.n_review.to(str) + " need review",
                color_scheme="orange",
                variant="soft",
                size="1",
            ),
            rx.text(CurationGridState.curation_message, size="1", color="gray"),
            wrap="wrap",
            spacing="2",
            align="center",
        ),
        spacing="3",
        align="start",
        width="100%",
        flex_shrink="0",
    )


def _score_grid() -> rx.Component:
    return lazyframe_grid(
        CurationGridState,
        height="100%",
        density="compact",
        column_header_height=56,
        checkbox_selection=True,
        show_filter_presets=False,
        detail_columns=SCORE_DETAIL_FIELDS,
        detail_labels=SCORE_DETAIL_LABELS,
        detail_height="auto",
        detail_renderers={"sources": {"type": "link_list", "underline": True, "separator": " · "}},
    )


def _cluster_grid() -> rx.Component:
    return lazyframe_grid(
        CurationGridState,
        height="100%",
        density="compact",
        column_header_height=56,
        checkbox_selection=False,
        show_filter_presets=False,
        detail_columns=CLUSTER_DETAIL_FIELDS,
        detail_labels=CLUSTER_DETAIL_LABELS,
        detail_height="auto",
    )


def _loaded_grid() -> rx.Component:
    return rx.box(
        rx.cond(
            CurationGridState.curation_error != "",
            rx.box(
                rx.text("Curation files are inconsistent.", weight="bold", size="2"),
                rx.text(
                    CurationGridState.curation_error,
                    size="1",
                    white_space="pre-wrap",
                ),
                max_height="160px",
                overflow_y="auto",
                width="100%",
                flex_shrink="0",
            ),
        ),
        lazyframe_grid_stats_bar(CurationGridState),
        rx.box(
            data_grid_scroll_container(
                rx.cond(
                    CurationGridState.curation_view == "clusters",
                    _cluster_grid(),
                    _score_grid(),
                ),
            ),
            flex="1 1 0%",
            min_height="0",
            overflow="hidden",
            width="100%",
        ),
        display="flex",
        flex_direction="column",
        flex="1 1 0%",
        min_height="0",
        width="100%",
        gap="8px",
    )


def curation_panel() -> rx.Component:
    """Browse compiled curation: resolved scores, the review queue, and clusters."""
    return rx.box(
        _toolbar(),
        rx.cond(
            CurationGridState.curation_loaded,
            _loaded_grid(),
            rx.cond(
                CurationGridState.curation_error != "",
                rx.box(
                    rx.text("Curation files did not compile.", weight="bold", size="2"),
                    rx.text(
                        CurationGridState.curation_error,
                        size="1",
                        white_space="pre-wrap",
                    ),
                    max_height="280px",
                    overflow_y="auto",
                    width="100%",
                ),
                rx.center(
                    rx.vstack(
                        rx.spinner(size="3"),
                        rx.text(CurationGridState.curation_message, size="2", color="gray"),
                        spacing="3",
                        align="center",
                    ),
                    padding="48px",
                ),
            ),
        ),
        display="flex",
        flex_direction="column",
        gap="12px",
        height="calc(100vh - 390px)",
        min_height="480px",
        width="100%",
    )
