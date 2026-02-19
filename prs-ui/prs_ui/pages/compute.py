"""Compute PRS page: VCF upload, genome build detection, score selection via MUI DataGrid, PRS computation."""

import reflex as rx
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar

from prs_ui.state import ComputeGridState, GenomicGridState

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


def _prs_progress_section() -> rx.Component:
    """Progress bar and status text shown during PRS computation."""
    return rx.cond(
        ComputeGridState.prs_computing,
        rx.vstack(
            rx.progress(
                value=ComputeGridState.prs_progress,
                size="3",
                width="100%",
                color_scheme="green",
            ),
            rx.text(
                ComputeGridState.status_message,
                size="2",
                weight="medium",
                color="var(--accent-11)",
            ),
            spacing="2",
            width="100%",
            padding_y="4px",
        ),
        rx.cond(
            ComputeGridState.status_message != "",
            rx.text(ComputeGridState.status_message, size="1", color="gray"),
        ),
    )


def _results_table() -> rx.Component:
    """Table displaying PRS computation results with quality assessment."""
    return rx.cond(
        ComputeGridState.prs_results.length() > 0,  # type: ignore[operator]
        rx.vstack(
            rx.callout(
                "PRS results are statistical estimates for research purposes. "
                "They should not be used as clinical diagnoses. "
                "Consult a healthcare professional for medical interpretation.",
                icon="shield_alert",
                color_scheme="amber",
                size="1",
                width="100%",
            ),
            rx.callout(
                "Raw PRS scores are NOT comparable across different PGS models. "
                "Each model uses a different scale, number of variants, and weighting. "
                "A higher score in one model does not mean higher risk than a lower "
                "score in another model. Compare scores only within the same PGS ID.",
                icon="info",
                color_scheme="blue",
                size="1",
                width="100%",
            ),
            rx.callout(
                "Percentiles marked with a star are computed from allele "
                "frequencies reported in the scoring file (theoretical population "
                "distribution under HWE). They are approximate and assume "
                "independent loci.",
                icon="info",
                color_scheme="iris",
                size="1",
                width="100%",
            ),
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
            rx.hstack(
                rx.icon("bar-chart-3", size=16),
                rx.text("PRS Results", size="3", weight="bold"),
                rx.spacer(),
                rx.button(
                    rx.icon("download", size=14),
                    "Download CSV",
                    on_click=ComputeGridState.download_prs_results_csv,
                    color_scheme="green",
                    size="2",
                ),
                align="center",
                spacing="2",
                width="100%",
            ),
            rx.box(
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("PGS ID"),
                            rx.table.column_header_cell("Trait"),
                            rx.table.column_header_cell("PRS Score"),
                            rx.table.column_header_cell("Percentile"),
                            rx.table.column_header_cell("AUROC"),
                            rx.table.column_header_cell("Quality"),
                            rx.table.column_header_cell("Population"),
                            rx.table.column_header_cell("Match Rate"),
                            rx.table.column_header_cell("Matched / Total"),
                            rx.table.column_header_cell("Effect Size"),
                        ),
                    ),
                    rx.table.body(
                        rx.foreach(
                            ComputeGridState.prs_results,
                            _result_row,
                        ),
                    ),
                    width="100%",
                    size="2",
                ),
                overflow_x="auto",
                width="100%",
            ),
            rx.foreach(
                ComputeGridState.prs_results,
                _result_interpretation_card,
            ),
            rx.text(
                "AUROC, effect size, and population are from the best available "
                "PGS Catalog evaluation study (largest sample, European-ancestry preferred). "
                "Quality combines AUROC (model accuracy) and match rate (genotype coverage). "
                "Results are most accurate when your ancestry matches the evaluation population.",
                size="1",
                color="gray",
            ),
            spacing="3",
            width="100%",
        ),
    )


def _result_row(row: dict) -> rx.Component:  # type: ignore[type-arg]
    """Single row in the PRS results table."""
    return rx.table.row(
        rx.table.cell(
            rx.text(row["pgs_id"], size="2", weight="medium"),
        ),
        rx.table.cell(rx.text(row["trait"], size="2")),
        rx.table.cell(
            rx.text(row["score"], size="2", weight="bold"),
        ),
        rx.table.cell(
            rx.cond(
                row["percentile"] != "",
                rx.hstack(
                    rx.badge(
                        rx.text(row["percentile"], "%"),
                        color_scheme="iris",
                        size="2",
                        variant="solid",
                    ),
                    rx.text("*", size="1", color="iris", weight="bold"),
                    spacing="1",
                    align="center",
                ),
                rx.text("—", size="2", color="gray"),
            ),
        ),
        rx.table.cell(
            rx.cond(
                row["auroc"] != "",
                rx.text(row["auroc"], size="2"),
                rx.text("N/A", size="2", color="gray"),
            ),
        ),
        rx.table.cell(
            rx.badge(
                row["quality_label"],
                color_scheme=row["quality_color"],
                size="1",
                variant="soft",
            ),
        ),
        rx.table.cell(
            rx.cond(
                row["ancestry"] != "",
                rx.text(row["ancestry"], size="2"),
                rx.text("N/A", size="2", color="gray"),
            ),
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
    )


def _result_interpretation_card(row: dict) -> rx.Component:  # type: ignore[type-arg]
    """Interpretation card for a single PRS result."""
    return rx.cond(
        row["summary"] != "",
        rx.card(
            rx.hstack(
                rx.badge(
                    row["pgs_id"],
                    color_scheme="gray",
                    size="1",
                    variant="outline",
                ),
                rx.badge(
                    rx.text("Quality: ", row["quality_label"]),
                    color_scheme=row["quality_color"],
                    size="1",
                    variant="soft",
                ),
                rx.cond(
                    row["percentile"] != "",
                    rx.badge(
                        rx.text("Percentile: ", row["percentile"], "%"),
                        color_scheme="iris",
                        size="1",
                        variant="solid",
                    ),
                ),
                rx.cond(
                    row["ancestry"] != "",
                    rx.badge(
                        rx.text("Pop: ", row["ancestry"]),
                        color_scheme="purple",
                        size="1",
                        variant="outline",
                    ),
                ),
                spacing="2",
                wrap="wrap",
            ),
            rx.text(row["summary"], size="2", color="var(--gray-11)"),
            width="100%",
            size="1",
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
        _genomic_data_section(),
        rx.separator(),
        _scores_selector(),
        rx.callout(
            "You are responsible for ensuring that your VCF matches the population used in the selected PRS and that the genome build is correct for the scoring files.",
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
        _prs_progress_section(),
        _results_table(),
        width="100%",
        spacing="4",
    )
