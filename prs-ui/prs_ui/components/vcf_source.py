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
        _sample_rows(source_state),
        # Build info lives on each sample row; only warnings need a callout.
        rx.cond(
            (source_state.build_detection_message != "")
            & (
                (source_state.detected_build == "")
                | source_state.build_detection_message.contains("⚠")  # type: ignore[union-attr]
            ),
            rx.callout(
                source_state.build_detection_message,
                icon="triangle_alert",
                color_scheme="orange",
                size="1",
                width="100%",
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


def _sample_row(source_state: type[rx.State], chip: dict) -> rx.Component:
    """One horizontal row for one loaded sample: color · label · build · ancestry · variants · remove."""
    return rx.hstack(
        rx.box(
            width="12px",
            height="12px",
            border_radius="50%",
            background=chip["color"],
            flex_shrink="0",
        ),
        rx.text(chip["label"], size="2", weight="bold"),
        rx.text(chip["filename"], size="1", color="gray"),
        rx.spacer(),
        rx.cond(
            chip["build"] != "",
            rx.badge(chip["build"], color_scheme="blue", variant="soft", size="1"),
        ),
        rx.cond(
            chip["ancestry_label"] != "",
            rx.badge(
                rx.hstack(
                    rx.icon("shield-check", size=12),
                    rx.text(chip["ancestry_label"], size="1", weight="medium"),
                    rx.cond(
                        chip["ancestry_conf"] != "",
                        rx.text(chip["ancestry_conf"], size="1", color="gray"),
                    ),
                    align="center",
                    spacing="1",
                ),
                color_scheme="green",
                variant="soft",
                size="1",
                title=(
                    "Genetic ancestry autodetected from this genome against the "
                    "1000 Genomes reference panel, with the classifier's "
                    "confidence. The detected population is preselected as the "
                    "reference population for percentiles — you can override "
                    "it below."
                ),
            ),
        ),
        rx.cond(
            chip["fine_label"] != "",
            rx.cond(
                chip["fine_url"] != "",
                rx.link(
                    rx.hstack(
                        rx.text(chip["fine_label"], size="1"),
                        rx.cond(
                            chip["fine_conf"] != "",
                            rx.text(chip["fine_conf"], size="1", color="gray"),
                        ),
                        rx.icon("external-link", size=10),
                        align="center",
                        spacing="1",
                    ),
                    href=chip["fine_url"],
                    is_external=True,
                    title=chip["fine_title"],
                    size="1",
                    color_scheme="green",
                    underline="hover",
                ),
                rx.hstack(
                    rx.text(chip["fine_label"], size="1", color="gray"),
                    rx.cond(
                        chip["fine_conf"] != "",
                        rx.text(chip["fine_conf"], size="1", color="gray"),
                    ),
                    align="center",
                    spacing="1",
                ),
            ),
        ),
        rx.cond(
            chip["variants"] != "",
            rx.text(chip["variants"], size="1", color="gray"),
        ),
        rx.icon_button(
            rx.icon("x", size=14),
            size="1",
            variant="ghost",
            color_scheme="gray",
            title="Remove this sample",
            on_click=source_state.remove_sample(chip["label"]),  # type: ignore[operator]
        ),
        align="center",
        spacing="2",
        width="100%",
        padding="6px 10px",
        border="1px solid var(--gray-4)",
        border_radius="8px",
        background="var(--gray-1)",
        title=chip["filename"],
    )


def _sample_rows(source_state: type[rx.State]) -> rx.Component:
    """One row per loaded sample, stacked vertically (CLI comparison legend)."""
    return rx.cond(
        source_state.vcf_sample_count > 0,  # type: ignore[operator]
        rx.vstack(
            rx.hstack(
                rx.cond(
                    source_state.vcf_multi_sample,
                    rx.text(
                        "Comparing ",
                        source_state.vcf_sample_count,
                        " samples",
                        size="1",
                        weight="bold",
                        color="gray",
                    ),
                    rx.text("Sample", size="1", weight="bold", color="gray"),
                ),
                rx.spacer(),
                rx.cond(
                    source_state.vcf_multi_sample,
                    rx.button(
                        "Clear all",
                        size="1",
                        variant="ghost",
                        color_scheme="gray",
                        on_click=source_state.clear_samples,
                    ),
                ),
                align="center",
                width="100%",
            ),
            rx.foreach(
                source_state.vcf_sample_chips,
                lambda chip: _sample_row(source_state, chip),
            ),
            rx.cond(
                source_state.vcf_has_fine_population,
                rx.text(
                    "Closest 1000G cohort is the nearest of the 1000 Genomes "
                    "reference cohorts (26 worldwide) — a reference point, not a "
                    "nationality. Many populations have no dedicated cohort in the "
                    "panel (e.g. Slavic / Eastern European genomes usually land on "
                    "the Northern/Western European cohort as their closest match).",
                    size="1",
                    color="gray",
                ),
            ),
            spacing="1",
            width="100%",
        ),
    )


def _compact_dropzone(source_state: type[rx.State], upload_id: str) -> rx.Component:
    """Single-line VCF dropzone that stays small whether or not a file is loaded."""
    return rx.upload(
        rx.hstack(
            rx.cond(
                source_state.vcf_normalizing | source_state.ancestry_inferring,  # type: ignore[operator]
                rx.hstack(
                    rx.spinner(size="2"),
                    rx.text(source_state.normalize_status, size="2", weight="bold"),
                    rx.text("Please wait. Controls are paused until this finishes.", size="1", color="gray"),
                    align="center",
                    spacing="2",
                ),
                rx.cond(
                    source_state.vcf_sample_count > 0,  # type: ignore[operator]
                    rx.hstack(
                        rx.icon("plus", size=16, color="var(--accent-9)"),
                        rx.text(
                            "Add another sample to compare",
                            size="2",
                            weight="medium",
                            color="var(--accent-11)",
                        ),
                        rx.text(
                            "drop a VCF here or click to browse — each sample gets its own row and color",
                            size="1",
                            color="gray",
                        ),
                        align="center",
                        spacing="2",
                    ),
                    rx.hstack(
                        rx.icon("upload", size=16, color="gray"),
                        rx.text(
                            "Drop one or more VCF files here or click to browse",
                            size="2",
                            color="gray",
                        ),
                        rx.text(".vcf / .vcf.gz — multiple files = comparison", size="1", color="gray"),
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
        max_files=8,
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
    """Visible feedback for the blocking VCF normalization + ancestry steps."""
    return rx.cond(
        source_state.vcf_normalizing | source_state.ancestry_inferring,  # type: ignore[operator]
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
