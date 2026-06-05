"""Trait selection grid for the "By Trait" mode of the Compute PRS workbench.

This module exposes a single reusable ``trait_selector`` component: a datagrid
of traits (grouped from PGS Catalog scores) with Select/Clear controls.  The
VCF source, genome-build selector, compute button, and results are provided by
the shared :func:`prs_ui.components.prs_workbench`, so this file no longer owns
any upload/normalization/results UI.
"""

import reflex as rx

from prs_ui.grid_style import data_grid_scroll_container
from prs_ui.state import GenomicGridState, TraitBrowserState
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar


def trait_selector(state: type[rx.State] = TraitBrowserState) -> rx.Component:
    """Trait selection grid with Select/Clear buttons and selection badges."""
    selection_ready = (state.prs_genotypes_path != "") & ~GenomicGridState.vcf_normalizing  # type: ignore[operator]
    selection_disabled = ~selection_ready  # type: ignore[operator]
    return rx.vstack(
        rx.cond(
            GenomicGridState.vcf_normalizing,
            rx.callout(
                "Normalizing your VCF. Trait selection will unlock automatically "
                "once the genotype table is ready.",
                icon="loader",
                color_scheme="blue",
                size="1",
                width="100%",
            ),
            rx.cond(
                state.prs_genotypes_path == "",
                rx.callout(
                    "Upload a VCF above to enable trait selection. The table below "
                    "is read-only until genotypes are loaded.",
                    icon="upload",
                    color_scheme="blue",
                    size="1",
                    width="100%",
                ),
            ),
        ),
        rx.hstack(
            rx.icon("layers", size=16),
            rx.text("Select Traits", size="3", weight="bold"),
            rx.text(
                "Choose traits to compute PRS for all associated scoring models.",
                size="2",
                color="gray",
            ),
            spacing="2",
            align="center",
        ),
        rx.hstack(
            rx.button(
                rx.icon("list-checks", size=14),
                "Select Filtered",
                on_click=state.select_filtered_traits,
                variant="outline",
                size="2",
                disabled=(~state.traits_loaded) | selection_disabled,  # type: ignore[operator]
            ),
            rx.button(
                "Clear Selection",
                on_click=state.deselect_all_traits,
                variant="outline",
                color_scheme="gray",
                size="2",
                disabled=(state.selected_traits.length() == 0) | selection_disabled,  # type: ignore[operator]
            ),
            rx.spacer(),
            rx.cond(
                state.selected_traits.length() > 0,  # type: ignore[operator]
                rx.hstack(
                    rx.badge(
                        rx.text(state.selected_traits.length(), " traits"),  # type: ignore[operator]
                        color_scheme="blue",
                        size="2",
                    ),
                    rx.badge(
                        rx.text(state.selected_pgs_ids.length(), " PGS IDs"),  # type: ignore[operator]
                        color_scheme="green",
                        size="2",
                    ),
                    spacing="2",
                ),
            ),
            wrap="wrap",
            spacing="2",
            align="center",
            width="100%",
        ),
        rx.cond(
            state.traits_loaded,
            rx.vstack(
                lazyframe_grid_stats_bar(state),
                rx.box(
                    data_grid_scroll_container(
                        lazyframe_grid(
                            state,
                            height="400px",
                            density="compact",
                            column_header_height=56,
                            checkbox_selection=selection_ready,
                        ),
                    ),
                    opacity=rx.cond(selection_ready, 1.0, 0.55),
                    pointer_events=rx.cond(selection_ready, "auto", "none"),
                    position="relative",
                    width="100%",
                ),
                width="100%",
                spacing="2",
            ),
            rx.hstack(
                rx.spinner(size="3"),
                rx.text("Loading traits from PGS Catalog...", size="2", color="gray"),
                spacing="2",
                align="center",
                padding="16px",
            ),
        ),
        spacing="3",
        width="100%",
    )
