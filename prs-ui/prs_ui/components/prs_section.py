"""Reusable PRS computation UI components for Reflex applications.

Each component function takes a ``state`` parameter -- the concrete state class
that inherits from ``PRSComputeStateMixin`` and ``LazyFrameGridMixin``.  This
allows the same components to work in different apps with different state
hierarchies.

Usage in a host app::

    from prs_ui.components import prs_section
    from prs_ui.mixin import PRSComputeStateMixin

    class MyPRSState(PRSComputeStateMixin, LazyFrameGridMixin, MyAppState):
        ...

    def my_page():
        return prs_section(MyPRSState)
"""

from typing import Any

import reflex as rx
from reflex_mui_datagrid import (
    PlotlyDetailSupport,
    data_grid,
    lazyframe_grid,
    lazyframe_grid_stats_bar,
)

from just_prs.prs import PRSEngine
from prs_ui.grid_style import data_grid_scroll_container


def _merge_bell_curve_config(
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Shallow-merge user overrides onto the default bell-curve renderer config.

    Keeps the merge non-destructive (``defaults`` is not mutated) so the same
    defaults dict can be reused across multiple component calls.
    """
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


_ACCORDION_SUMMARY_STYLE: dict[str, str] = {
    "cursor": "pointer",
    "display": "list-item",
    "listStylePosition": "inside",
    "width": "fit-content",
}


def _accordion_summary_label(
    label: str,
    size: str,
    weight: str = "medium",
    color: str | None = None,
) -> rx.Component:
    """Inline label for native details summaries so the arrow stays by the title."""
    font_weight = "500" if weight == "medium" else weight
    style: dict[str, str] = {
        "display": "inline",
        "fontSize": f"var(--font-size-{size})",
        "fontWeight": font_weight,
        "marginLeft": "4px",
    }
    if color is not None:
        style["color"] = color
    return rx.el.span(
        label,
        style=style,
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


def prs_engine_selector(state: type[rx.State]) -> rx.Component:
    """PRS computation engine selector: Polars (lazy, in-memory) or DuckDB (SQL, spills to disk)."""
    return rx.hstack(
        rx.text("Engine:", size="2", weight="medium"),
        rx.select(
            [PRSEngine.POLARS.value, PRSEngine.DUCKDB.value],
            value=state.prs_engine,
            on_change=state.set_prs_engine,
            size="2",
        ),
        rx.tooltip(
            rx.icon("info", size=14, color="gray"),
            content=(
                "Polars: fast in-memory engine using Rust (default). "
                "DuckDB: SQL engine that can spill to disk under memory pressure, "
                "better for large scoring files on low-memory machines."
            ),
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
                data_grid_scroll_container(
                    lazyframe_grid(
                        state,
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
            _accordion_summary_label("How to interpret PRS results", size="2"),
            style=_ACCORDION_SUMMARY_STYLE,
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
                _accordion_summary_label(
                    "Methodology notes",
                    size="1",
                    color="var(--gray-11)",
                ),
                style=_ACCORDION_SUMMARY_STYLE,
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


def _prs_results_header(state: type[rx.State]) -> rx.Component:
    """Header bar above PRS results: view mode toggle + download button."""
    return rx.cond(
        state.prs_results.length() > 0,  # type: ignore[operator]
        rx.vstack(
            rx.hstack(
                rx.icon("bar-chart-3", size=16),
                rx.text("PRS Results", size="3", weight="bold"),
                rx.spacer(),
                rx.segmented_control.root(
                    rx.segmented_control.item("Grouped", value="grouped"),
                    rx.segmented_control.item("Individual", value="individual"),
                    value=state.prs_view_mode,
                    on_change=state.set_prs_view_mode,
                    size="2",
                ),
                rx.button(
                    rx.icon("download", size=14),
                    "CSV",
                    on_click=state.download_prs_results_csv,
                    color_scheme="green",
                    variant="soft",
                    size="2",
                ),
                align="center",
                spacing="3",
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
    )


def prs_results_table(
    state: type[rx.State],
    bell_curve_height: int = 360,
    bell_curve_max_width: int = 1200,
    detail_height: int | str = "auto",
    bell_curve_config: dict[str, Any] | None = None,
    metric_list_config: dict[str, Any] | None = None,
) -> rx.Component:
    """Table displaying PRS computation results with quality assessment.

    Uses foldable detail panels (reflex-mui-datagrid >= 0.2.0) to show
    interpretation, reference source, and population percentiles inline
    below each row when the chevron is clicked.

    Args:
        state: Concrete state class (must mix in ``PRSComputeStateMixin``).
        bell_curve_height: Height of the per-row bell curve chart in pixels.
        bell_curve_max_width: Max width of the per-row bell curve container.
        detail_height: Detail panel height. ``"auto"`` (default) lets the
            panel grow to fit the bell curve without cropping. Pass a number
            (e.g. ``700``) to enforce a fixed height with internal scroll.
        bell_curve_config: Extra renderer config keys merged on top of the
            defaults (e.g. ``{"labelTiers": 12, "bands": [...]}``); supports
            every key accepted by the ``bell_curve`` renderer.
        metric_list_config: Extra renderer config for metric cards in the
            expanded row. Defaults keep cards compact and cap them at four per row.
    """
    bell_curve = _merge_bell_curve_config(
        {
            "type": "bell_curve",
            "scaleMin": 0,
            "scaleMax": 100,
            "height": bell_curve_height,
            "maxWidth": bell_curve_max_width,
            "sidePanelTitle": "PRS context",
            "summaryPlacement": "fullWidth",
            "labelMode": "always",
            "labelTiers": 9,
            "labelMinGapZ": 0.28,
            "bands": [
                {"from": 0, "to": 25, "label": "below average"},
                {"from": 25, "to": 75, "label": "usual middle range"},
                {"from": 75, "to": 90, "label": "above average"},
                {"from": 90, "to": 100, "label": "high tail"},
            ],
        },
        bell_curve_config,
    )
    metric_cards = _merge_bell_curve_config(
        {
            "type": "metric_list",
            "compact": True,
            "maxColumns": 4,
            "minCardWidth": 150,
            "gap": 8,
        },
        metric_list_config,
    )
    return rx.cond(
        (state.prs_results.length() > 0) & (state.prs_view_mode == "individual"),  # type: ignore[operator]
        rx.box(
            _prs_disclaimers(state),
            _prs_interpretation_guide(),
            PlotlyDetailSupport.create(),
            rx.box(
                data_grid_scroll_container(
                    data_grid(
                        rows=state.prs_results_rows,
                        columns=state.prs_results_columns,
                        column_grouping_model=state.prs_results_column_groups,
                        row_id_field="id",
                        pagination=False,
                        hide_footer=True,
                        density="compact",
                        height="100%",
                        disable_row_selection_on_click=True,
                        detail_columns=[
                            "population_percentiles_chart", "population_percentiles_summary", "risk_context",
                            "result_suggestions", "model_context",
                        ],
                        detail_labels={
                            "population_percentiles_chart": "Where You Fall on the Reference Curve",
                            "population_percentiles_summary": "Interpretation",
                            "risk_context": "Does This Change Actual Risk?",
                            "result_suggestions": "Quick Flags",
                            "model_context": "Can I Trust This Result?",
                        },
                        detail_renderers={
                            "risk_context": metric_cards,
                            "model_context": metric_cards,
                            "result_suggestions": {"type": "badge_list"},
                            "population_percentiles_chart": bell_curve,
                        },
                        detail_height=detail_height,
                    ),
                ),
                flex="1 1 0%",
                min_height="0",
                overflow="hidden",
                width="100%",
            ),
            rx.text(
                "Click the chevron on any row for a bell curve showing your position, "
                "absolute-risk context, population percentile spread, and quality flags.",
                size="1",
                color="gray",
                flex_shrink="0",
            ),
            display="flex",
            flex_direction="column",
            gap="12px",
            height="calc(100vh - 340px)",
            min_height="0",
            width="100%",
        ),
    )


def trait_summary_table(
    state: type[rx.State],
    bell_curve_height: int = 380,
    bell_curve_max_width: int = 1200,
    large_bell_curve_threshold: int = 4,
    large_bell_curve_height: int = 460,
    large_bell_curve_max_width: int = 1600,
    detail_height: int | str = "auto",
    bell_curve_config: dict[str, Any] | None = None,
    metric_list_config: dict[str, Any] | None = None,
    link_list_config: dict[str, Any] | None = None,
) -> rx.Component:
    """Trait-grouped summary with visualizations, built from computed PRS rows.

    Args:
        state: Concrete state class (must mix in ``PRSComputeStateMixin``).
        bell_curve_height: Height (px) of the per-trait bell curve when a
            trait has at most ``large_bell_curve_threshold`` PRS models.
        bell_curve_max_width: Max container width (px) for the per-trait
            bell curve at the standard size.
        large_bell_curve_threshold: Number of PRS models above which a trait
            switches to the larger, side-panel-free bell curve layout.
        large_bell_curve_height: Bell curve height (px) for traits above
            the threshold.
        large_bell_curve_max_width: Bell curve container width (px) for
            traits above the threshold.
        detail_height: Detail panel height. ``"auto"`` (default) lets the
            panel grow to fit the bell curve without cropping. Pass a number
            (e.g. ``700``) to enforce a fixed height with internal scroll.
        bell_curve_config: Extra renderer config keys merged on top of the
            defaults for the per-trait bell curve (e.g. ``{"labelTiers": 12}``).
        metric_list_config: Extra renderer config for trait metric cards.
        link_list_config: Extra renderer config for the PGS Catalog link list.
    """
    bell_curve = _merge_bell_curve_config(
        {
            "type": "bell_curve",
            "scaleMin": 0,
            "scaleMax": 100,
            "height": bell_curve_height,
            "maxWidth": bell_curve_max_width,
            "sidePanelTitle": "Interpretation",
            "labelTiers": 9,
            "labelMaxVisible": 18,
            "labelMinGapZ": 0.22,
            "bands": [
                {"from": 0, "to": 25, "label": "below average"},
                {"from": 25, "to": 75, "label": "usual middle range"},
                {"from": 75, "to": 90, "label": "above average"},
                {"from": 90, "to": 100, "label": "high tail"},
            ],
        },
        bell_curve_config,
    )
    metric_cards = _merge_bell_curve_config(
        {
            "type": "metric_list",
            "compact": True,
            "maxColumns": 4,
            "minCardWidth": 150,
            "gap": 8,
        },
        metric_list_config,
    )
    pgs_links = _merge_bell_curve_config(
        {
            "type": "link_list",
            "underline": True,
            "separator": ", ",
        },
        link_list_config,
    )
    return rx.cond(
        (state.prs_results.length() > 0) & (state.prs_view_mode == "grouped"),  # type: ignore[operator]
        rx.box(
            PlotlyDetailSupport.create(),
            rx.box(
                data_grid_scroll_container(
                    data_grid(
                        rows=state.trait_summary_rows,
                        columns=state.trait_summary_columns,
                        row_id_field="id",
                        pagination=False,
                        hide_footer=True,
                        density="compact",
                        height="100%",
                        disable_row_selection_on_click=True,
                        detail_columns=[
                            "pgs_links",
                            "percentile_chart",
                            "key_metrics",
                            "trait_quick_flags",
                            "confidence_segments",
                        ],
                        detail_labels={
                            "pgs_links": "PGS Models Included",
                            "percentile_chart": "Where You Fall on the Bell Curve",
                            "key_metrics": "Key Statistics",
                            "trait_quick_flags": "Signal Flags",
                            "confidence_segments": "All Models vs High-Quality Models",
                        },
                        detail_renderers={
                            "pgs_links": pgs_links,
                            "key_metrics": metric_cards,
                            "trait_quick_flags": {"type": "badge_list"},
                            "confidence_segments": metric_cards,
                            "percentile_chart": bell_curve,
                        },
                        detail_height=detail_height,
                    ),
                ),
                flex="1 1 0%",
                min_height="0",
                overflow="hidden",
                width="100%",
            ),
            rx.text(
                "Click the chevron on any trait row to see: a bell curve showing where "
                "each model places you, variant match rates, key statistics, and a plain-language explanation "
                "of what the results mean and why models may disagree.",
                size="1",
                color="gray",
                flex_shrink="0",
            ),
            display="flex",
            flex_direction="column",
            gap="12px",
            height="calc(100vh - 380px)",
            min_height="0",
            width="100%",
        ),
    )


def prs_section(
    state: type[rx.State],
    results_table_kwargs: dict[str, Any] | None = None,
    trait_summary_kwargs: dict[str, Any] | None = None,
) -> rx.Component:
    """Complete PRS computation section: build selector, score grid, compute button, results.

    This is the primary reusable entry point. Pass a concrete state class that
    inherits from ``PRSComputeStateMixin`` and ``LazyFrameGridMixin``.

    Args:
        state: Concrete state class with PRS computation behavior.
        results_table_kwargs: Forwarded to :func:`prs_results_table`. Use this
            to size the per-row bell curve or override its renderer config,
            e.g. ``{"bell_curve_height": 360, "bell_curve_max_width": 1200}``.
        trait_summary_kwargs: Forwarded to :func:`trait_summary_table` for
            customizing the trait-grouped bell curve dimensions and behavior.

    Example::

        from prs_ui.components import prs_section
        prs_section(
            MyPRSState,
            results_table_kwargs={"bell_curve_height": 360, "bell_curve_max_width": 1200},
            trait_summary_kwargs={"bell_curve_height": 460},
        )
    """
    return rx.theme(
        rx.vstack(
            rx.hstack(
                prs_build_selector(state),
                rx.separator(orientation="vertical", size="2"),
                prs_engine_selector(state),
                rx.separator(orientation="vertical", size="2"),
                prs_ancestry_selector(state),
                spacing="4",
                align="center",
                wrap="wrap",
            ),
            prs_scores_selector(state),
            prs_compute_button(state),
            prs_progress_section(state),
            _prs_results_header(state),
            prs_results_table(state, **(results_table_kwargs or {})),
            trait_summary_table(state, **(trait_summary_kwargs or {})),
            width="100%",
            spacing="4",
        ),
        has_background=False,
    )
