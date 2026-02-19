"""Compute PRS page: VCF upload, genome build detection, score selection via MUI DataGrid, PRS computation."""

import reflex as rx
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar

from prs_ui.state import ComputeGridState

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
    """Genome build selector with auto-detection message."""
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


def _scores_selector() -> rx.Component:
    """Score selection using MUI DataGrid with server-side virtual scrolling."""
    return rx.vstack(
        rx.hstack(
            rx.button(
                rx.icon("list-checks", size=14),
                "Select Filtered",
                on_click=ComputeGridState.select_filtered_scores,
                variant="outline",
                size="2",
                disabled=~ComputeGridState.compute_scores_loaded,  # type: ignore[operator]
            ),
            rx.button(
                "Clear Selection",
                on_click=ComputeGridState.deselect_all_scores,
                variant="outline",
                color_scheme="gray",
                size="2",
                disabled=ComputeGridState.selected_pgs_ids.length() == 0,  # type: ignore[operator]
            ),
            rx.spacer(),
            rx.cond(
                ComputeGridState.selected_pgs_ids.length() > 0,  # type: ignore[operator]
                rx.badge(
                    rx.text(ComputeGridState.selected_pgs_ids.length(), " selected"),  # type: ignore[operator]
                    color_scheme="green",
                    size="2",
                ),
            ),
            wrap="wrap",
            spacing="2",
            align="center",
            width="100%",
        ),
        rx.cond(
            ComputeGridState.compute_scores_loaded,
            rx.vstack(
                lazyframe_grid_stats_bar(ComputeGridState),
                lazyframe_grid(
                    ComputeGridState,
                    height="400px",
                    density="compact",
                    column_header_height=56,
                    checkbox_selection=True,
                    on_row_selection_model_change=ComputeGridState.handle_compute_row_selection,
                ),
                width="100%",
                spacing="2",
            ),
            rx.hstack(
                rx.spinner(size="3"),
                rx.text("Loading PGS Catalog scores...", size="2", color="gray"),
                spacing="2",
                align="center",
                padding="16px",
            ),
        ),
        spacing="3",
        width="100%",
    )


def _results_table() -> rx.Component:
    """Table displaying PRS computation results."""
    return rx.cond(
        ComputeGridState.prs_results.length() > 0,  # type: ignore[operator]
        rx.vstack(
            rx.cond(
                ComputeGridState.low_match_warning,
                rx.callout(
                    "One or more scores have a match rate below 10%. "
                    "This may indicate a genome build mismatch between "
                    "the VCF and scoring files. Check your genome build selection.",
                    icon="triangle_alert",
                    color_scheme="red",
                    size="1",
                    width="100%",
                ),
            ),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("PGS ID"),
                        rx.table.column_header_cell("Trait"),
                        rx.table.column_header_cell("PRS Score"),
                        rx.table.column_header_cell("Match Rate"),
                        rx.table.column_header_cell("Matched / Total"),
                        rx.table.column_header_cell("Effect Size"),
                        rx.table.column_header_cell("Classification"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(
                        ComputeGridState.prs_results,
                        lambda row: rx.table.row(
                            rx.table.cell(
                                rx.text(row["pgs_id"], size="2", weight="medium"),
                            ),
                            rx.table.cell(rx.text(row["trait"], size="2")),
                            rx.table.cell(
                                rx.text(row["score"], size="2", weight="bold"),
                            ),
                            rx.table.cell(
                                rx.badge(
                                    rx.text(row["match_rate"], "%"),
                                    color_scheme=row["match_color"],
                                    size="1",
                                ),
                            ),
                            rx.table.cell(
                                rx.text(
                                    row["variants_matched"], " / ", row["variants_total"],
                                    size="2",
                                ),
                            ),
                            rx.table.cell(rx.text(row["effect_size"], size="2")),
                            rx.table.cell(rx.text(row["classification"], size="2")),
                        ),
                    ),
                ),
                width="100%",
                size="2",
            ),
            rx.text(
                "Effect size and classification metrics are from the best available "
                "PGS Catalog evaluation study (largest European-ancestry cohort).",
                size="1",
                color="gray",
            ),
            spacing="3",
            width="100%",
        ),
    )


def compute_panel() -> rx.Component:
    """Full PRS computation panel: upload, build selector, scores grid, compute button, results."""
    return rx.vstack(
        rx.hstack(
            _upload_area(),
            width="100%",
        ),
        _build_selector(),
        rx.separator(),
        _scores_selector(),
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
        rx.cond(
            ComputeGridState.status_message != "",
            rx.text(ComputeGridState.status_message, size="1", color="gray"),
        ),
        _results_table(),
        width="100%",
        spacing="4",
    )
