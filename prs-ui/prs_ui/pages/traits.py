"""Trait Browser page: select traits to compute PRS for all associated scores.

Instead of selecting individual PGS IDs, the user browses a datagrid of
traits (grouped from PGS Catalog scores), selects the traits of interest,
and all PGS IDs for those traits are computed together.  The output
(results table + trait summary) is identical to the Compute PRS tab.
"""

import reflex as rx

from prs_ui.components.prs_section import (
    prs_ancestry_selector,
    prs_progress_section,
    prs_results_table,
    trait_summary_table,
)
from prs_ui.grid_style import data_grid_scroll_container
from prs_ui.state import GenomicGridState, TraitBrowserState
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar

UPLOAD_ID = "trait_vcf_upload"


def _upload_area() -> rx.Component:
    """VCF file upload area with drag-and-drop."""
    return rx.upload(
        rx.vstack(
            rx.cond(
                TraitBrowserState.vcf_filename != "",
                rx.hstack(
                    rx.icon("file-check", size=18, color="green"),
                    rx.text(TraitBrowserState.vcf_filename, size="2", weight="medium"),
                    align="center",
                    spacing="2",
                ),
                rx.vstack(
                    rx.icon("upload", size=24, color="gray"),
                    rx.text(
                        "Drop a VCF file here or click to browse",
                        size="2",
                        color="gray",
                    ),
                    rx.text(
                        "Accepts .vcf and .vcf.gz files",
                        size="1",
                        color="gray",
                    ),
                    align="center",
                    spacing="1",
                ),
            ),
            align="center",
            justify="center",
            padding="24px",
        ),
        id=UPLOAD_ID,
        accept={
            "text/vcf": [".vcf"],
            "text/plain": [".vcf"],
            "application/gzip": [".vcf.gz", ".gz"],
            "application/octet-stream": [".vcf.gz", ".gz"],
        },
        max_files=1,
        on_drop=TraitBrowserState.handle_vcf_upload(rx.upload_files(upload_id=UPLOAD_ID)),  # type: ignore[arg-type]
        border="2px dashed var(--gray-6)",
        border_radius="8px",
        width="100%",
        cursor="pointer",
        _hover={"border_color": "var(--accent-9)"},
    )


def _build_selector() -> rx.Component:
    """Genome build selector with auto-detection message."""
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.text("Genome Build:", size="2", weight="medium"),
                rx.select(
                    ["GRCh37", "GRCh38"],
                    value=TraitBrowserState.genome_build,
                    on_change=TraitBrowserState.set_genome_build,
                    size="2",
                ),
                spacing="2",
                align="center",
            ),
            rx.separator(orientation="vertical", size="2"),
            prs_ancestry_selector(TraitBrowserState),
            spacing="4",
            align="center",
            wrap="wrap",
            width="100%",
        ),
        rx.cond(
            TraitBrowserState.build_detection_message != "",
            rx.cond(
                TraitBrowserState.detected_build != "",
                rx.callout(
                    TraitBrowserState.build_detection_message,
                    icon="check",
                    color_scheme="green",
                    size="1",
                ),
                rx.callout(
                    TraitBrowserState.build_detection_message,
                    icon="triangle_alert",
                    color_scheme="orange",
                    size="1",
                ),
            ),
        ),
        spacing="2",
        width="100%",
    )


def _genomic_data_section() -> rx.Component:
    """Compact VCF normalization status."""
    return rx.cond(
        GenomicGridState.genomic_loaded,
        rx.callout(
            rx.text(
                GenomicGridState.normalize_status,
                size="2",
                weight="medium",
            ),
            icon="check",
            color_scheme="green",
            size="1",
            width="100%",
        ),
        rx.cond(
            GenomicGridState.normalize_status != "",
            rx.vstack(
                rx.callout(
                    rx.hstack(
                        rx.spinner(size="2"),
                        rx.text(GenomicGridState.normalize_status, size="2"),
                        spacing="2",
                        align="center",
                    ),
                    icon="info",
                    color_scheme="blue",
                    size="1",
                    width="100%",
                ),
                rx.progress(size="2", width="100%"),
                spacing="2",
                width="100%",
                padding_y="4px",
            ),
        ),
    )


def _trait_selector() -> rx.Component:
    """Trait selection grid with Select/Clear buttons."""
    return rx.vstack(
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
                on_click=TraitBrowserState.select_filtered_traits,
                variant="outline",
                size="2",
                disabled=~TraitBrowserState.traits_loaded,  # type: ignore[operator]
            ),
            rx.button(
                "Clear Selection",
                on_click=TraitBrowserState.deselect_all_traits,
                variant="outline",
                color_scheme="gray",
                size="2",
                disabled=TraitBrowserState.selected_traits.length() == 0,  # type: ignore[operator]
            ),
            rx.spacer(),
            rx.cond(
                TraitBrowserState.selected_traits.length() > 0,  # type: ignore[operator]
                rx.hstack(
                    rx.badge(
                        rx.text(TraitBrowserState.selected_traits.length(), " traits"),  # type: ignore[operator]
                        color_scheme="blue",
                        size="2",
                    ),
                    rx.badge(
                        rx.text(TraitBrowserState.selected_pgs_ids.length(), " PGS IDs"),  # type: ignore[operator]
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
            TraitBrowserState.traits_loaded,
            rx.vstack(
                lazyframe_grid_stats_bar(TraitBrowserState),
                data_grid_scroll_container(
                    lazyframe_grid(
                        TraitBrowserState,
                        height="400px",
                        density="compact",
                        column_header_height=56,
                        checkbox_selection=True,
                    ),
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


def traits_panel() -> rx.Component:
    """Full trait browser panel: upload VCF, select traits, compute PRS, view results."""
    return rx.vstack(
        rx.hstack(
            _upload_area(),
            width="100%",
        ),
        _build_selector(),
        _genomic_data_section(),
        rx.separator(),
        _trait_selector(),
        rx.callout(
            "You are responsible for ensuring that your VCF matches the population "
            "used in the selected PRS and that the genome build is correct for the "
            "scoring files.",
            icon="triangle_alert",
            color_scheme="amber",
            size="1",
            width="100%",
        ),
        rx.hstack(
            rx.button(
                rx.icon("calculator", size=14),
                "Compute PRS for Selected Traits",
                on_click=TraitBrowserState.compute_selected_prs,
                loading=TraitBrowserState.prs_computing,
                disabled=(TraitBrowserState.selected_pgs_ids.length() == 0) | (TraitBrowserState.vcf_filename == ""),  # type: ignore[operator]
                color_scheme="green",
                size="3",
            ),
            spacing="3",
            align="center",
        ),
        prs_progress_section(TraitBrowserState),
        trait_summary_table(TraitBrowserState),
        rx.cond(
            TraitBrowserState.prs_results.length() > 0,  # type: ignore[operator]
            rx.el.details(
                rx.el.summary(
                    rx.hstack(
                        rx.icon("list", size=14),
                        rx.text("Individual PRS Results", size="2", weight="medium"),
                        rx.text(
                            TraitBrowserState.prs_results.length(),  # type: ignore[operator]
                            " scores",
                            size="1",
                            color="gray",
                        ),
                        spacing="2",
                        align="center",
                    ),
                    cursor="pointer",
                ),
                prs_results_table(TraitBrowserState),
                open=~TraitBrowserState.trait_summary_visible,  # type: ignore[operator]
            ),
        ),
        width="100%",
        spacing="4",
    )
