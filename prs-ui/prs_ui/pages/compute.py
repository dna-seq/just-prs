"""Compute PRS page: VCF upload, genome build detection, score selection via MUI DataGrid, PRS computation.

This is the standalone page for the prs-ui app. It uses the reusable
components from ``prs_ui.components.prs_section`` for score selection,
progress, and results, while keeping VCF-upload-specific UI here.
"""

import reflex as rx

from prs_ui.components.prs_section import (
    prs_progress_section,
    prs_results_table,
    prs_scores_selector,
)
from prs_ui.state import ComputeGridState, GenomicGridState
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar

UPLOAD_ID = "vcf_upload"


def _upload_area() -> rx.Component:
    """VCF file upload area with drag-and-drop."""
    return rx.upload(
        rx.vstack(
            rx.cond(
                ComputeGridState.vcf_filename != "",
                rx.hstack(
                    rx.icon("file-check", size=18, color="green"),
                    rx.text(ComputeGridState.vcf_filename, size="2", weight="medium"),
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
        accept={".vcf": ["text/plain"], ".gz": ["application/gzip"]},
        max_files=1,
        on_drop=ComputeGridState.handle_vcf_upload(rx.upload_files(upload_id=UPLOAD_ID)),  # type: ignore[arg-type]
        border="2px dashed var(--gray-6)",
        border_radius="8px",
        width="100%",
        cursor="pointer",
        _hover={"border_color": "var(--accent-9)"},
    )


def _build_selector() -> rx.Component:
    """Genome build selector with auto-detection message (standalone-specific)."""
    return rx.vstack(
        rx.hstack(
            rx.text("Genome Build:", size="2", weight="medium"),
            rx.select(
                ["GRCh37", "GRCh38"],
                value=ComputeGridState.genome_build,
                on_change=ComputeGridState.set_genome_build,
                size="2",
            ),
            spacing="2",
            align="center",
        ),
        rx.cond(
            ComputeGridState.build_detection_message != "",
            rx.cond(
                ComputeGridState.detected_build != "",
                rx.callout(
                    ComputeGridState.build_detection_message,
                    icon="check",
                    color_scheme="green",
                    size="1",
                ),
                rx.callout(
                    ComputeGridState.build_detection_message,
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
    """Collapsible section showing normalized VCF data in a DataGrid."""
    return rx.cond(
        GenomicGridState.genomic_loaded,
        rx.vstack(
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
            rx.hstack(
                rx.icon("dna", size=16),
                rx.text("Genomic Data", size="3", weight="bold"),
                rx.spacer(),
                rx.badge(
                    rx.text(GenomicGridState.genomic_row_count, " variants"),
                    color_scheme="blue",
                    size="2",
                ),
                align="center",
                spacing="2",
                width="100%",
            ),
            lazyframe_grid_stats_bar(GenomicGridState),
            lazyframe_grid(
                GenomicGridState,
                height="350px",
                density="compact",
                column_header_height=56,
            ),
            spacing="2",
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


def compute_panel() -> rx.Component:
    """Full PRS computation panel: upload, build selector, scores grid, compute button, results.

    Uses reusable PRS components for score selection and results, with
    standalone-specific VCF upload and genomic data preview.
    """
    return rx.vstack(
        rx.hstack(
            _upload_area(),
            width="100%",
        ),
        _build_selector(),
        _genomic_data_section(),
        rx.separator(),
        prs_scores_selector(ComputeGridState),
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
                "Compute PRS",
                on_click=ComputeGridState.compute_selected_prs,
                loading=ComputeGridState.prs_computing,
                disabled=(ComputeGridState.selected_pgs_ids.length() == 0) | (ComputeGridState.vcf_filename == ""),  # type: ignore[operator]
                color_scheme="green",
                size="3",
            ),
            spacing="3",
            align="center",
        ),
        prs_progress_section(ComputeGridState),
        prs_results_table(ComputeGridState),
        width="100%",
        spacing="4",
    )
