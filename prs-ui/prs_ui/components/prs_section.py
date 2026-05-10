"""Reusable PRS computation UI components for Reflex applications.

Each component function takes a ``state`` parameter -- the concrete state class
that inherits from ``PRSComputeStateMixin`` and ``LazyFrameGridMixin``.  This
allows the same components to work in different apps with different state
hierarchies.

Usage in a host app::

    from prs_ui.components import prs_section
    from prs_ui.state import PRSComputeStateMixin

    class MyPRSState(PRSComputeStateMixin, LazyFrameGridMixin, MyAppState):
        ...

    def my_page():
        return prs_section(MyPRSState)
"""

import reflex as rx
from reflex_mui_datagrid import (
    PlotlyDetailSupport,
    data_grid,
    lazyframe_grid,
    lazyframe_grid_stats_bar,
)


def prs_build_selector(state: type[rx.State]) -> rx.Component:
    """Genome build selector dropdown."""
    return rx.hstack(
        rx.text("Genome Build:", size="2", weight="medium"),
        rx.select(
            ["GRCh37", "GRCh38"],
            value=state.genome_build,
            on_change=state.set_prs_genome_build,
            size="2",
        ),
        spacing="2",
        align="center",
    )


def prs_ancestry_selector(state: type[rx.State]) -> rx.Component:
    """Ancestry superpopulation selector for 1000G-based percentile lookup."""
    return rx.hstack(
        rx.text("Ancestry (for percentile):", size="2", weight="medium"),
        rx.select(
            ["AFR", "AMR", "EAS", "EUR", "SAS"],
            value=state.selected_ancestry,
            on_change=state.set_selected_ancestry,
            size="2",
        ),
        rx.tooltip(
            rx.icon("info", size=14, color="gray"),
            content=(
                "Select your ancestry group for percentile estimation. "
                "AFR=African, AMR=American, EAS=East Asian, EUR=European, SAS=South Asian. "
                "Percentiles are computed relative to 1000 Genomes reference individuals "
                "in the selected group. Enable all populations to compute percentiles "
                "for every available 1000G superpopulation. When reference data is unavailable, "
                "a theoretical or AUROC-based approximation is used."
            ),
        ),
        rx.checkbox(
            "All available populations",
            checked=state.compute_all_populations,
            on_change=state.set_compute_all_populations,
            size="2",
        ),
        spacing="2",
        align="center",
    )


def prs_scores_selector(state: type[rx.State]) -> rx.Component:
    """Score selection using MUI DataGrid with server-side virtual scrolling."""
    return rx.vstack(
        rx.hstack(
            rx.button(
                rx.icon("list-checks", size=14),
                "Select Filtered",
                on_click=state.select_filtered_scores,
                variant="outline",
                size="2",
                disabled=~state.compute_scores_loaded,  # type: ignore[operator]
            ),
            rx.button(
                "Clear Selection",
                on_click=state.deselect_all_scores,
                variant="outline",
                color_scheme="gray",
                size="2",
                disabled=state.selected_pgs_ids.length() == 0,  # type: ignore[operator]
            ),
            rx.spacer(),
            rx.cond(
                state.selected_pgs_ids.length() > 0,  # type: ignore[operator]
                rx.badge(
                    rx.text(state.selected_pgs_ids.length(), " selected"),  # type: ignore[operator]
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
            state.compute_scores_loaded,
            rx.vstack(
                lazyframe_grid_stats_bar(state),
                lazyframe_grid(
                    state,
                    height="400px",
                    density="compact",
                    column_header_height=56,
                    checkbox_selection=True,
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


def prs_compute_button(state: type[rx.State]) -> rx.Component:
    """Compute PRS button with disclaimer callout."""
    return rx.vstack(
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
                on_click=state.compute_selected_prs,
                loading=state.prs_computing,
                disabled=state.selected_pgs_ids.length() == 0,  # type: ignore[operator]
                color_scheme="green",
                size="3",
            ),
            spacing="3",
            align="center",
        ),
        spacing="3",
        width="100%",
    )


def prs_progress_section(state: type[rx.State]) -> rx.Component:
    """Progress bar and status text shown during PRS computation."""
    return rx.cond(
        state.prs_computing,
        rx.vstack(
            rx.progress(
                value=state.prs_progress,
                size="3",
                width="100%",
                color_scheme="green",
            ),
            rx.text(
                state.status_message,
                size="2",
                weight="medium",
                color="var(--accent-11)",
            ),
            spacing="2",
            width="100%",
            padding_y="4px",
        ),
        rx.cond(
            state.status_message != "",
            rx.text(state.status_message, size="1", color="gray"),
        ),
    )



def _prs_interpretation_guide() -> rx.Component:
    """Collapsible guide explaining how to read PRS percentiles and risk levels."""
    return rx.el.details(
        rx.el.summary(
            rx.text(
                "How to interpret PRS results",
                size="2",
                weight="medium",
                style={"cursor": "pointer"},
            ),
        ),
        rx.vstack(
            rx.text(
                "The raw PRS score (e.g. 0.098) is model-specific and unitless — "
                "it cannot be read as 'protective' or 'risky' on its own, and scores "
                "from different PGS models cannot be compared to each other.",
                size="2",
            ),
            rx.text(
                "The percentile is the key number. It shows where your score falls "
                "relative to a reference population of the same ancestry. "
                "For virtually all standard PRS models, higher percentile = more "
                "genetic variants associated with increased risk for that trait.",
                size="2",
            ),
            rx.hstack(
                rx.badge("< 25th", color_scheme="blue", size="1", variant="soft"),
                rx.text("Below average predisposition", size="2"),
                spacing="2", align="center",
            ),
            rx.hstack(
                rx.badge("25th – 75th", color_scheme="gray", size="1", variant="soft"),
                rx.text("Average predisposition", size="2"),
                spacing="2", align="center",
            ),
            rx.hstack(
                rx.badge("75th – 90th", color_scheme="orange", size="1", variant="soft"),
                rx.text("Above average predisposition", size="2"),
                spacing="2", align="center",
            ),
            rx.hstack(
                rx.badge("> 90th", color_scheme="red", size="1", variant="soft"),
                rx.text("High predisposition", size="2"),
                spacing="2", align="center",
            ),
            rx.text(
                "PRS captures only inherited genetic variants — not lifestyle, "
                "environment, or treatment. Most people with a high PRS never develop "
                "the condition, and many with a low PRS do. This is a research tool, "
                "not a diagnostic test.",
                size="2",
                color="var(--gray-11)",
            ),
            spacing="2",
            padding_top="8px",
            padding_x="4px",
        ),
    )


def _prs_disclaimers(state: type[rx.State]) -> rx.Component:
    """Collapsible disclaimers and methodology notes for PRS results."""
    return rx.vstack(
        rx.callout(
            "PRS results are statistical estimates for research purposes only. "
            "They should not be used as clinical diagnoses.",
            icon="shield_alert",
            color_scheme="amber",
            size="1",
            width="100%",
        ),
        rx.cond(
            state.low_match_warning,
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
        rx.el.details(
            rx.el.summary(
                rx.text(
                    "Methodology notes",
                    size="1",
                    color="gray",
                    weight="medium",
                    style={"cursor": "pointer"},
                ),
            ),
            rx.vstack(
                rx.text(
                    "Raw PRS scores are NOT comparable across different PGS models. "
                    "Each model uses a different scale, number of variants, and weighting.",
                    size="1",
                    color="gray",
                ),
                rx.text(
                    "Percentiles use 1000 Genomes reference distributions matched to your "
                    "selected ancestry when available; otherwise a theoretical or AUROC-based "
                    "approximation is used (shown in the Pct. Method column).",
                    size="1",
                    color="gray",
                ),
                rx.text(
                    "Absolute Risk estimates translate PRS percentile into approximate "
                    "lifetime disease probability. Multiple methods may be available "
                    "(OR per SD, AUROC model, h²-liability). The Agreement column "
                    "shows how well different methods converge.",
                    size="1",
                    color="gray",
                ),
                spacing="1",
                padding_top="4px",
                padding_x="4px",
            ),
        ),
        spacing="2",
        width="100%",
    )


def prs_results_table(state: type[rx.State]) -> rx.Component:
    """Table displaying PRS computation results with quality assessment.

    Uses foldable detail panels (reflex-mui-datagrid >= 0.2.0) to show
    interpretation, reference source, and population percentiles inline
    below each row when the chevron is clicked.
    """
    return rx.cond(
        state.prs_results.length() > 0,  # type: ignore[operator]
        rx.vstack(
            _prs_disclaimers(state),
            rx.hstack(
                rx.icon("bar-chart-3", size=16),
                rx.text("PRS Results", size="3", weight="bold"),
                rx.spacer(),
                rx.button(
                    rx.icon("download", size=14),
                    "Download CSV",
                    on_click=state.download_prs_results_csv,
                    color_scheme="green",
                    size="2",
                ),
                align="center",
                spacing="2",
                width="100%",
            ),
            _prs_interpretation_guide(),
            PlotlyDetailSupport.create(),
            data_grid(
                rows=state.prs_results_rows,
                columns=state.prs_results_columns,
                column_grouping_model=state.prs_results_column_groups,
                row_id_field="id",
                pagination=False,
                hide_footer=True,
                density="compact",
                height="calc(100vh - 340px)",
                disable_row_selection_on_click=True,
                detail_columns=[
                    "result_suggestions", "population_percentiles_chart",
                    "risk_hint", "summary", "reference_source_detail",
                ],
                detail_labels={
                    "result_suggestions": "What This Means",
                    "population_percentiles_chart": "Population Percentile Chart",
                    "risk_hint": "Interpretation",
                    "summary": "Quality Summary",
                    "reference_source_detail": "Reference Data Source",
                },
                detail_renderers={
                    "result_suggestions": {"type": "badge_list"},
                    "population_percentiles_chart": {
                        "type": "bell_curve",
                        "scaleMin": 0,
                        "scaleMax": 100,
                        "bands": [
                            {"from": 0, "to": 25, "label": "below average"},
                            {"from": 25, "to": 75, "label": "usual middle range"},
                            {"from": 75, "to": 90, "label": "above average"},
                            {"from": 90, "to": 100, "label": "high tail"},
                        ],
                    },
                },
                detail_height=340,
            ),
            rx.text(
                "Click the chevron on any row for a bell curve showing your position, "
                "detailed interpretation, and quality assessment.",
                size="1",
                color="gray",
            ),
            spacing="3",
            width="100%",
        ),
    )


def trait_summary_table(state: type[rx.State]) -> rx.Component:
    """Trait-grouped summary with visualizations, built from computed PRS rows."""
    return rx.cond(
        state.prs_results.length() > 0,  # type: ignore[operator]
        rx.vstack(
            rx.separator(),
            rx.hstack(
                rx.button(
                    rx.icon("layers", size=16),
                    "Group by Trait",
                    on_click=state.build_trait_summary,
                    color_scheme="green",
                    size="3",
                ),
                rx.text(
                    "Aggregate multiple PRS models per trait to see consensus, "
                    "spread, and outliers across scoring methods. "
                    "Different models often give different percentiles — the bell curve "
                    "and interpretation below each trait explain why and what to trust.",
                    size="2",
                    color="gray",
                    style={"flex": "1"},
                ),
                spacing="3",
                align="center",
                width="100%",
            ),
            rx.cond(
                state.trait_summary_visible,
                rx.vstack(
                    PlotlyDetailSupport.create(),
                    data_grid(
                        rows=state.trait_summary_rows,
                        columns=state.trait_summary_columns,
                        row_id_field="id",
                        pagination=False,
                        hide_footer=True,
                        density="compact",
                        height="calc(100vh - 380px)",
                        disable_row_selection_on_click=True,
                        detail_columns=[
                            "key_metrics",
                            "percentile_chart",
                            "interpretation",
                            "outlier_detail",
                        ],
                        detail_labels={
                            "key_metrics": "Key Statistics",
                            "percentile_chart": "Where You Fall on the Bell Curve",
                            "interpretation": "What This Means for You",
                            "outlier_detail": "Model Agreement & Outlier Notes",
                        },
                        detail_renderers={
                            "key_metrics": {"type": "metric_list"},
                            "percentile_chart": {
                                "type": "bell_curve",
                                "scaleMin": 0,
                                "scaleMax": 100,
                                "bands": [
                                    {"from": 0, "to": 25, "label": "below average"},
                                    {"from": 25, "to": 75, "label": "usual middle range"},
                                    {"from": 75, "to": 90, "label": "above average"},
                                    {"from": 90, "to": 100, "label": "high tail"},
                                ],
                            },
                        },
                        detail_height=420,
                    ),
                    rx.text(
                        "Click the chevron on any trait row to see: a bell curve showing where "
                        "each model places you, key statistics, and a plain-language explanation "
                        "of what the results mean and why models may disagree.",
                        size="1",
                        color="gray",
                    ),
                    spacing="3",
                    width="100%",
                ),
            ),
            spacing="3",
            width="100%",
            padding_top="8px",
        ),
    )


def prs_section(state: type[rx.State]) -> rx.Component:
    """Complete PRS computation section: build selector, score grid, compute button, results.

    This is the primary reusable entry point. Pass a concrete state class that
    inherits from ``PRSComputeStateMixin`` and ``LazyFrameGridMixin``.

    Example::

        from prs_ui.components import prs_section
        prs_section(MyPRSState)
    """
    return rx.theme(
        rx.vstack(
            rx.hstack(
                prs_build_selector(state),
                rx.separator(orientation="vertical", size="2"),
                prs_ancestry_selector(state),
                spacing="4",
                align="center",
                wrap="wrap",
            ),
            prs_scores_selector(state),
            prs_compute_button(state),
            prs_progress_section(state),
            prs_results_table(state),
            trait_summary_table(state),
            width="100%",
            spacing="4",
        ),
        has_background=False,
    )
