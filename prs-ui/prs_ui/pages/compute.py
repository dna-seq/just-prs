"""Compute PRS page: VCF upload, genome build detection, score selection, PRS computation."""

from typing import Any

import reflex as rx

from prs_ui.state import AppState

UPLOAD_ID = "vcf_upload"


def _upload_area() -> rx.Component:
    """VCF file upload area with drag-and-drop."""
    return rx.upload(
        rx.vstack(
            rx.cond(
                AppState.vcf_filename != "",
                rx.hstack(
                    rx.icon("file-check", size=18, color="green"),
                    rx.text(AppState.vcf_filename, size="2", weight="medium"),
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
        on_drop=AppState.handle_vcf_upload(rx.upload_files(upload_id=UPLOAD_ID)),  # type: ignore[arg-type]
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
                value=AppState.genome_build,
                on_change=AppState.set_genome_build,
                size="2",
            ),
            spacing="2",
            align="center",
        ),
        rx.cond(
            AppState.build_detection_message != "",
            rx.cond(
                AppState.detected_build != "",
                rx.callout(
                    AppState.build_detection_message,
                    icon="check",
                    color_scheme="green",
                    size="1",
                ),
                rx.callout(
                    AppState.build_detection_message,
                    icon="triangle_alert",
                    color_scheme="orange",
                    size="1",
                ),
            ),
        ),
        spacing="2",
        width="100%",
    )


def _score_row(row: dict[str, Any]) -> rx.Component:
    """Render a single score row with a checkbox."""
    return rx.table.row(
        rx.table.cell(
            rx.checkbox(
                checked=AppState.selected_pgs_ids.contains(row["pgs_id"]),  # type: ignore[attr-defined]
                on_change=lambda _val: AppState.toggle_compute_score(row["pgs_id"]),
            ),
        ),
        rx.table.cell(rx.text(row["pgs_id"], size="2", weight="medium")),
        rx.table.cell(rx.text(row["name"], size="2")),
        rx.table.cell(
            rx.tooltip(
                rx.text(
                    rx.cond(
                        row["trait"].to(str).length() > 50,  # type: ignore[union-attr]
                        row["trait"].to(str)[:47] + "...",  # type: ignore[index]
                        row["trait"],
                    ),
                    size="2",
                ),
                content=row["trait"],
            ),
        ),
        rx.table.cell(rx.text(row["variants"], size="2")),
        rx.table.cell(rx.text(row["build"], size="2")),
    )


def _pagination_controls() -> rx.Component:
    """Pagination and page info for the scores table."""
    page_display = AppState.compute_page + 1
    total_pages = (AppState.compute_total_count + AppState.compute_page_size - 1) // AppState.compute_page_size
    return rx.hstack(
        rx.button(
            rx.icon("chevron-left", size=14),
            on_click=AppState.compute_prev_page,
            disabled=AppState.compute_page == 0,
            variant="outline",
            size="1",
        ),
        rx.text(
            f"Page ",
            page_display,
            " of ",
            total_pages,
            size="2",
            color="gray",
        ),
        rx.button(
            rx.icon("chevron-right", size=14),
            on_click=AppState.compute_next_page,
            disabled=(AppState.compute_page + 1) * AppState.compute_page_size >= AppState.compute_total_count,
            variant="outline",
            size="1",
        ),
        rx.text(
            AppState.compute_total_count,
            " scores",
            size="2",
            color="gray",
        ),
        spacing="2",
        align="center",
    )


def _scores_selector() -> rx.Component:
    """Scores selection panel with search, pagination, and a table."""
    return rx.vstack(
        rx.hstack(
            rx.button(
                rx.icon("database", size=14),
                "Load Scores",
                on_click=AppState.load_compute_scores,
                color_scheme="blue",
                size="2",
            ),
            rx.cond(
                AppState.compute_scores_loaded,
                rx.hstack(
                    rx.input(
                        value=AppState.compute_search_term,
                        on_change=AppState.set_compute_search_term,
                        placeholder="Search by ID, name, or trait...",
                        width="280px",
                        size="2",
                    ),
                    rx.button(
                        "Select All",
                        on_click=AppState.select_all_visible_scores,
                        variant="outline",
                        size="2",
                    ),
                    rx.button(
                        "Clear",
                        on_click=AppState.deselect_all_scores,
                        variant="outline",
                        color_scheme="gray",
                        size="2",
                    ),
                    spacing="2",
                    align="center",
                ),
            ),
            rx.spacer(),
            rx.cond(
                AppState.selected_pgs_ids.length() > 0,  # type: ignore[operator]
                rx.badge(
                    rx.text(AppState.selected_pgs_ids.length(), " selected"),  # type: ignore[operator]
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
            AppState.compute_scores_loaded,
            rx.vstack(
                rx.box(
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                rx.table.column_header_cell("", width="40px"),
                                rx.table.column_header_cell("PGS ID"),
                                rx.table.column_header_cell("Name"),
                                rx.table.column_header_cell("Trait"),
                                rx.table.column_header_cell("Variants"),
                                rx.table.column_header_cell("Build"),
                            ),
                        ),
                        rx.table.body(
                            rx.foreach(
                                AppState.compute_scores_rows,
                                _score_row,
                            ),
                        ),
                        width="100%",
                        size="1",
                    ),
                    width="100%",
                    border="1px solid var(--gray-5)",
                    border_radius="8px",
                ),
                _pagination_controls(),
                spacing="2",
                width="100%",
            ),
            rx.callout(
                "Click 'Load Scores' to fetch PGS Catalog scores filtered by the selected genome build.",
                icon="info",
                color_scheme="blue",
                size="1",
            ),
        ),
        spacing="3",
        width="100%",
    )


def _results_table() -> rx.Component:
    """Table displaying PRS computation results."""
    return rx.cond(
        AppState.prs_results.length() > 0,  # type: ignore[operator]
        rx.vstack(
            rx.cond(
                AppState.low_match_warning,
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
                        AppState.prs_results,
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
                on_click=AppState.compute_selected_prs,
                loading=AppState.prs_computing,
                disabled=(AppState.selected_pgs_ids.length() == 0) | (AppState.vcf_filename == ""),  # type: ignore[operator]
                color_scheme="green",
                size="3",
            ),
            spacing="3",
            align="center",
        ),
        _results_table(),
        width="100%",
        spacing="4",
    )
