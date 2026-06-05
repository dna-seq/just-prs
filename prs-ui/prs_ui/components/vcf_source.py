"""Reusable, detachable VCF genotype-source component.

``vcf_source_section`` renders a compact VCF upload dropzone, genome-build
detection feedback, and a collapsed preview of the normalized variants.  It is
the *reference* genotype source for the PRS workbench, but it is deliberately
decoupled from the PRS consumer states: the source state pushes normalized
genotypes into consumers via ``consumer.load_genotypes(path)`` (see
``GenomicGridState`` in ``prs_ui.state``).  A host app such as just-dna-lite
can swap this for an entirely different source (e.g. a public-genome selector)
as long as that source drives the same consumer hooks.
"""

import reflex as rx
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar

from prs_ui.grid_style import data_grid_scroll_container


def vcf_source_section(
    source_state: type[rx.State],
    upload_id: str = "vcf_upload",
    show_preview: bool = True,
) -> rx.Component:
    """Compact VCF upload + build detection + collapsed normalized preview.

    Args:
        source_state: Concrete genotype-source state (owns ``vcf_filename``,
            ``detected_build``, ``build_detection_message``, ``normalize_status``,
            ``genomic_loaded``, ``genomic_row_count``, the grid vars, and the
            ``handle_vcf_upload`` / ``normalize_uploaded_vcf`` handlers).
        upload_id: DOM id for the ``rx.upload`` dropzone (must be unique per page).
        show_preview: When True, render the collapsed normalized-VCF preview grid.
    """
    return rx.vstack(
        _compact_dropzone(source_state, upload_id),
        rx.cond(
            rx.selected_files(upload_id).length() > 0,  # type: ignore[operator]
            rx.hstack(
                rx.text("Selected:", size="1", color="gray"),
                rx.foreach(
                    rx.selected_files(upload_id),
                    lambda filename: rx.badge(filename, color_scheme="blue", variant="soft"),
                ),
                spacing="2",
                align="center",
                wrap="wrap",
                width="100%",
            ),
        ),
        rx.cond(
            source_state.build_detection_message != "",
            rx.cond(
                source_state.detected_build != "",
                rx.callout(
                    source_state.build_detection_message,
                    icon="check",
                    color_scheme="green",
                    size="1",
                    width="100%",
                ),
                rx.callout(
                    source_state.build_detection_message,
                    icon="triangle_alert",
                    color_scheme="orange",
                    size="1",
                    width="100%",
                ),
            ),
        ),
        _normalization_progress(source_state),
        rx.cond(
            show_preview,
            _normalized_preview(source_state),
        ),
        spacing="2",
        width="100%",
    )


def _compact_dropzone(source_state: type[rx.State], upload_id: str) -> rx.Component:
    """Single-line VCF dropzone that stays small whether or not a file is loaded."""
    return rx.upload(
        rx.hstack(
            rx.cond(
                source_state.vcf_normalizing,
                rx.hstack(
                    rx.spinner(size="2"),
                    rx.text(source_state.normalize_status, size="2", weight="bold"),
                    rx.text("Please wait. Controls are paused until this finishes.", size="1", color="gray"),
                    align="center",
                    spacing="2",
                ),
                rx.cond(
                    source_state.vcf_filename != "",
                    rx.hstack(
                        rx.icon("file-check", size=16, color="green"),
                        rx.text(source_state.vcf_filename, size="2", weight="medium"),
                        rx.text("(click to replace)", size="1", color="gray"),
                        align="center",
                        spacing="2",
                    ),
                    rx.hstack(
                        rx.icon("upload", size=16, color="gray"),
                        rx.text(
                            "Drop a VCF file here or click to browse",
                            size="2",
                            color="gray",
                        ),
                        rx.text(".vcf / .vcf.gz", size="1", color="gray"),
                        align="center",
                        spacing="2",
                    ),
                ),
            ),
            align="center",
            justify="center",
            width="100%",
        ),
        id=upload_id,
        accept={
            "text/vcf": [".vcf"],
            "text/plain": [".vcf"],
            "application/gzip": [".vcf.gz", ".gz"],
            "application/octet-stream": [".vcf.gz", ".gz"],
        },
        max_files=1,
        on_drop=source_state.handle_vcf_upload(
            rx.upload_files(upload_id=upload_id)
        ),  # type: ignore[arg-type]
        # Override StyledUpload's default padding="5em" so the dropzone stays compact.
        padding="10px 12px",
        border=rx.cond(
            source_state.vcf_normalizing,
            "2px solid var(--accent-9)",
            "2px dashed var(--gray-6)",
        ),
        border_radius="8px",
        width="100%",
        cursor="pointer",
        background=rx.cond(source_state.vcf_normalizing, "var(--accent-2)", "transparent"),
        _hover={"border_color": "var(--accent-9)"},
    )


def _normalization_progress(source_state: type[rx.State]) -> rx.Component:
    """Visible feedback for the blocking VCF normalization step."""
    return rx.cond(
        source_state.vcf_normalizing,
        rx.vstack(
            rx.callout(
                rx.hstack(
                    rx.spinner(size="2"),
                    rx.vstack(
                        rx.text(source_state.normalize_status, size="2", weight="medium"),
                        rx.text(
                            "PRS selection and computation will be available as soon as the "
                            "normalized genotype table is ready.",
                            size="1",
                            color="gray",
                        ),
                        spacing="1",
                        align="start",
                    ),
                    spacing="2",
                    align="center",
                ),
                icon="loader",
                color_scheme="blue",
                size="1",
                width="100%",
            ),
            rx.progress(
                size="3",
                width="100%",
                color_scheme="blue",
            ),
            spacing="2",
            width="100%",
            padding_y="4px",
        ),
    )


def _normalized_preview(source_state: type[rx.State]) -> rx.Component:
    """Collapsed normalized-VCF preview grid."""
    return rx.cond(
        source_state.genomic_loaded,
        rx.el.details(
            rx.el.summary(
                rx.hstack(
                    rx.icon("dna", size=16),
                    rx.text("Normalized VCF Preview", size="2", weight="bold"),
                    rx.spacer(),
                    rx.badge(
                        rx.text(source_state.genomic_row_count, " variants"),
                        color_scheme="blue",
                        size="2",
                    ),
                    rx.text("Open table", size="1", color="gray"),
                    align="center",
                    spacing="2",
                    width="100%",
                ),
                style={"cursor": "pointer", "listStyle": "none"},
            ),
            rx.vstack(
                lazyframe_grid_stats_bar(source_state),
                data_grid_scroll_container(
                    lazyframe_grid(
                        source_state,
                        height="320px",
                        density="compact",
                        column_header_height=56,
                    ),
                ),
                spacing="2",
                width="100%",
                padding_top="8px",
            ),
            width="100%",
            style={
                "border": "1px solid var(--gray-5)",
                "borderRadius": "8px",
                "padding": "8px 12px",
                "background": "var(--gray-1)",
            },
        ),
        rx.cond(
            (source_state.normalize_status != "") & ~source_state.vcf_normalizing,  # type: ignore[operator]
            rx.callout(
                source_state.normalize_status,
                icon="info",
                color_scheme="blue",
                size="1",
                width="100%",
            ),
        ),
    )
