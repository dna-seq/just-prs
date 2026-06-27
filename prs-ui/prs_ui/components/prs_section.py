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

from typing import Any, Callable

import reflex as rx
from reflex_mui_datagrid import (
    PlotlyDetailSupport,
    data_grid,
    lazyframe_grid,
    lazyframe_grid_stats_bar,
)

from just_prs.prs import PRSEngine
from just_prs.reference import SUPERPOPULATIONS
from prs_ui.components.vega_chart import VegaLiteChart
from prs_ui.grid_style import data_grid_scroll_container
from prs_ui.mixin import SUPERPOPULATION_LABELS
from prs_ui.state import GenomicGridState


def _resolve_normalizing(normalizing: Any | None = None) -> Any:
    """Return the source normalizing signal for reusable PRS controls."""
    return GenomicGridState.vcf_normalizing if normalizing is None else normalizing


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
    """Genome build selector dropdown with harmonized-scores toggle."""
    return rx.hstack(
        rx.text("Genome Build:", size="2", weight="medium"),
        rx.select(
            ["GRCh37", "GRCh38"],
            value=state.genome_build,
            on_change=state.set_prs_genome_build,
            size="2",
        ),
        rx.checkbox(
            "Include harmonized scores",
            checked=state.include_harmonized,
            on_change=state.set_include_harmonized,
            size="2",
        ),
        rx.tooltip(
            rx.icon("info", size=14, color="gray"),
            content=(
                "When enabled, includes scores originally developed on a different "
                "genome build but available as harmonized (coordinate-lifted) scoring "
                "files. For example, with GRCh38 selected, this adds ~4,300 scores "
                "originally developed on GRCh37. Harmonized scores receive a quality "
                "penalty in ranking because coordinate liftover may introduce minor "
                "mapping errors."
            ),
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
    """Reference-population controls for 1000G-based percentile lookup."""
    checkbox_handlers = {
        "AFR": state.set_reference_population_AFR,
        "AMR": state.set_reference_population_AMR,
        "EAS": state.set_reference_population_EAS,
        "EUR": state.set_reference_population_EUR,
        "SAS": state.set_reference_population_SAS,
    }
    checked_vars = {
        "AFR": state.show_reference_AFR,
        "AMR": state.show_reference_AMR,
        "EAS": state.show_reference_EAS,
        "EUR": state.show_reference_EUR,
        "SAS": state.show_reference_SAS,
    }
    population_boxes = [
        rx.checkbox(
            f"{code} {SUPERPOPULATION_LABELS[code]}",
            checked=checked_vars[code],
            on_change=checkbox_handlers[code],
            size="2",
        )
        for code in SUPERPOPULATIONS
    ]
    return rx.vstack(
        rx.hstack(
            rx.text("Reference populations:", size="2", weight="medium"),
            rx.badge("PRS-native default", color_scheme="blue", variant="soft", size="2"),
            rx.tooltip(
                rx.icon("info", size=14, color="gray"),
                content=(
                    "Each result uses the PGS model's closest available evaluation ancestry "
                    "as its default reference population. Tick extra 1000 Genomes populations "
                    "to add comparison curves and columns. AFR=African, AMR=American, "
                    "EAS=East Asian, EUR=European, SAS=South Asian."
                ),
            ),
            rx.button(
                "All",
                on_click=state.select_all_reference_populations,
                size="1",
                variant="soft",
            ),
            rx.button(
                "Native only",
                on_click=state.clear_reference_populations,
                size="1",
                variant="soft",
                color_scheme="gray",
            ),
            spacing="2",
            align="center",
            wrap="wrap",
        ),
        rx.hstack(
            *population_boxes,
            rx.checkbox(
                "Refresh reference/audit cache",
                checked=state.refresh_reference_cache_before_compute,
                on_change=state.set_refresh_reference_cache_before_compute,
                size="2",
            ),
            spacing="3",
            align="center",
            wrap="wrap",
        ),
        spacing="2",
        align="start",
        width="100%",
    )


def prs_scores_selector(
    state: type[rx.State],
    normalizing: Any | None = None,
) -> rx.Component:
    """Score selection using MUI DataGrid with server-side virtual scrolling."""
    is_normalizing = _resolve_normalizing(normalizing)
    selection_ready = (state.prs_genotypes_path != "") & (is_normalizing == False)  # noqa: E712
    selection_disabled = ~selection_ready  # type: ignore[operator]
    return rx.vstack(
        rx.cond(
            is_normalizing,
            rx.callout(
                "Normalizing your VCF. Score selection will unlock automatically "
                "once the genotype table is ready.",
                icon="loader",
                color_scheme="blue",
                size="1",
                width="100%",
            ),
            rx.cond(
                state.prs_genotypes_path == "",
                rx.callout(
                    "Upload a VCF above to enable score selection. The table below "
                    "is read-only until genotypes are loaded.",
                    icon="upload",
                    color_scheme="blue",
                    size="1",
                    width="100%",
                ),
            ),
        ),
        rx.hstack(
            rx.input(
                placeholder="Search PGS ID, name, reported trait, or EFO trait...",
                value=state.prs_catalog_query,
                on_change=state.set_prs_catalog_query,
                on_key_down=lambda key: rx.cond(
                    key == "Enter",
                    state.apply_prs_catalog_query(),
                    rx.noop(),
                ),
                size="2",
                flex="1 1 280px",
                min_width="240px",
                disabled=~state.compute_scores_loaded,
            ),
            rx.button(
                rx.icon("search", size=14),
                "Search Catalog",
                on_click=state.apply_prs_catalog_query,
                variant="outline",
                size="2",
                disabled=~state.compute_scores_loaded,
            ),
            rx.button(
                "Reset Search",
                on_click=state.clear_prs_catalog_query,
                variant="outline",
                color_scheme="gray",
                size="2",
                disabled=(~state.compute_scores_loaded) | (state.prs_catalog_query == ""),
            ),
            wrap="wrap",
            spacing="2",
            align="center",
            width="100%",
        ),
        rx.hstack(
            rx.button(
                rx.icon("list-checks", size=14),
                "Select Filtered",
                on_click=state.select_filtered_scores,
                variant="outline",
                size="2",
                disabled=(~state.compute_scores_loaded) | selection_disabled,  # type: ignore[operator]
            ),
            rx.button(
                "Clear Selection",
                on_click=state.deselect_all_scores,
                variant="outline",
                color_scheme="gray",
                size="2",
                disabled=(state.selected_pgs_ids.length() == 0) | selection_disabled,  # type: ignore[operator]
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
                rx.text("Loading PGS Catalog scores...", size="2", color="gray"),
                spacing="2",
                align="center",
                padding="16px",
            ),
        ),
        spacing="3",
        width="100%",
    )


def prs_compute_button(
    state: type[rx.State],
    normalizing: Any | None = None,
) -> rx.Component:
    """Compute PRS button with disclaimer callout."""
    is_normalizing = _resolve_normalizing(normalizing)
    not_ready = (state.selected_pgs_ids.length() == 0) | (state.prs_genotypes_path == "") | is_normalizing  # type: ignore[operator]
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
                disabled=not_ready,
                color_scheme="green",
                size="3",
            ),
            rx.cond(
                not_ready,
                rx.text(
                    rx.cond(
                        is_normalizing,
                        "VCF normalization in progress...",
                        rx.cond(
                            state.prs_genotypes_path == "",
                            "Load genomic data to enable computation.",
                            "Select at least one score to enable computation.",
                        ),
                    ),
                    size="2",
                    color="gray",
                ),
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
                "One or more scores have model coverage below 10%. "
                "That means the genome file contained too little of the PRS model "
                "to interpret reliably. Check your genome build selection and VCF coverage.",
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


def _prs_results_header(
    state: type[rx.State],
    show_view_toggle: bool = True,
) -> rx.Component:
    """Header bar above PRS results: view mode toggle + download button."""
    view_toggle = (
        rx.segmented_control.root(
            rx.segmented_control.item("Grouped", value="grouped"),
            rx.segmented_control.item("Individual", value="individual"),
            value=state.prs_view_mode,
            on_change=state.set_prs_view_mode,
            size="2",
        )
        if show_view_toggle
        else rx.fragment()
    )
    return rx.cond(
        state.prs_results.length() > 0,  # type: ignore[operator]
        rx.vstack(
            rx.hstack(
                rx.icon("bar-chart-3", size=16),
                rx.text("PRS Results", size="3", weight="bold"),
                rx.spacer(),
                view_toggle,
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
    link_list_config: dict[str, Any] | None = None,
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
        link_list_config: Extra renderer config for source-study links.
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
            "minCardWidth": 130,
            "gap": 8,
        },
        metric_list_config,
    )
    source_links = _merge_bell_curve_config(
        {
            "type": "link_list",
            "underline": True,
            "separator": ", ",
        },
        link_list_config,
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
                            "result_suggestions", "model_context", "publication_links", "ai_ask",
                        ],
                        detail_labels={
                            "population_percentiles_chart": "Where You Fall on the Reference Curve",
                            "population_percentiles_summary": "Interpretation",
                            "risk_context": "Does This Change Actual Risk?",
                            "result_suggestions": "Quick Flags",
                            "model_context": "Can I Trust This Result?",
                            "publication_links": "Source Study",
                            "ai_ask": "Ask AI for Interpretation",
                        },
                        detail_renderers={
                            "risk_context": metric_cards,
                            "model_context": metric_cards,
                            "result_suggestions": {"type": "badge_list"},
                            "population_percentiles_chart": bell_curve,
                            "publication_links": source_links,
                            "ai_ask": {"type": "button_links", "size": "medium", "gap": 12},
                        },
                        detail_height=detail_height,
                    ),
                ),
                flex="1 1 0%",
                min_height="0",
                overflow="hidden",
                width="100%",
            ),
            rx.hstack(
                rx.icon("chevron-right", size=14, color="var(--accent-9)"),
                rx.text(
                    "Expand any row (chevron on the left) for a bell curve showing your "
                    "position, absolute-risk context, percentile spread, quality flags, and "
                    "source-study links, plus "
                    "one-click ",
                    rx.text.strong("Ask AI buttons (ChatGPT, Claude, and more)"),
                    " that open a ready-made prompt with PGS Catalog and source-paper links.",
                    size="1",
                    color="gray",
                ),
                spacing="1",
                align="center",
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
    bell_curve_height: int = 340,
    bell_curve_max_width: int = 1200,
    large_bell_curve_threshold: int = 4,
    large_bell_curve_height: int = 400,
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
            "minCardWidth": 112,
            "cardPadding": "5px 8px",
            "gap": 8,
        },
        metric_list_config,
    )
    trait_header_cards = {
        "type": "metric_list",
        "compact": False,
        "maxColumns": 1,
        "minCardWidth": 360,
        "cardPadding": "14px 18px",
        "gap": 8,
    }
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
                            "trait_header",
                            "pgs_links",
                            "publication_links",
                            "percentile_chart",
                            "key_metrics",
                            "trait_quick_flags",
                            "confidence_segments",
                            "ai_ask",
                        ],
                        detail_labels={
                            "trait_header": "Trait",
                            "pgs_links": "PGS Models Included",
                            "publication_links": "Source Studies",
                            "percentile_chart": "Where You Fall on the Bell Curve",
                            "key_metrics": "Key Statistics",
                            "trait_quick_flags": "Signal Flags",
                            "confidence_segments": "All Models vs High-Quality Models",
                            "ai_ask": "Ask AI for Interpretation",
                        },
                        detail_renderers={
                            "trait_header": trait_header_cards,
                            "pgs_links": pgs_links,
                            "publication_links": pgs_links,
                            "key_metrics": metric_cards,
                            "trait_quick_flags": {"type": "badge_list"},
                            "confidence_segments": metric_cards,
                            "percentile_chart": bell_curve,
                            "ai_ask": {"type": "button_links", "size": "medium", "gap": 12},
                        },
                        detail_height=detail_height,
                    ),
                ),
                flex="1 1 0%",
                min_height="0",
                overflow="hidden",
                width="100%",
            ),
            rx.hstack(
                rx.icon("chevron-right", size=14, color="var(--accent-9)"),
                rx.text(
                    "Expand any trait row (chevron on the left) to see a bell curve showing where "
                    "each model places you, model coverage, key statistics, source studies, "
                    "a plain-language explanation of why models may disagree, and one-click ",
                    rx.text.strong("Ask AI buttons (ChatGPT, Claude, and more)"),
                    " that open a ready-made prompt with PGS Catalog and source-paper links.",
                    size="1",
                    color="gray",
                ),
                spacing="1",
                align="center",
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


# ---------------------------------------------------------------------------
# Altair-based results: clickable table + always-visible chart panel
# ---------------------------------------------------------------------------


def _chart_mode_toggle(state: type[rx.State]) -> rx.Component:
    """Single / Multi-ancestry chart toggle."""
    return rx.segmented_control.root(
        rx.segmented_control.item("Single ancestry", value="single"),
        rx.segmented_control.item("All ancestries", value="multi"),
        value=state.chart_mode,
        on_change=state.set_chart_mode,
        size="1",
    )


def _result_info_panel(state: type[rx.State]) -> rx.Component:
    """Compact info panel showing metadata for the selected result."""
    info = state.selected_result_info
    return rx.cond(
        state.selected_result_id != "",
        rx.vstack(
            rx.hstack(
                rx.cond(
                    info["pgs_id"].to(str) != "",  # type: ignore[union-attr]
                    rx.badge(info["pgs_id"], color_scheme="blue", size="2"),  # type: ignore[index]
                ),
                rx.cond(
                    info["trait"].to(str) != "",  # type: ignore[union-attr]
                    rx.text(info["trait"], size="2", weight="bold", trim="both"),  # type: ignore[index]
                ),
                spacing="2",
                align="center",
                wrap="wrap",
            ),
            rx.hstack(
                rx.cond(
                    info["percentile"].to(str) != "None",  # type: ignore[union-attr]
                    rx.hstack(
                        rx.text("Percentile:", size="1", color="gray"),
                        rx.text(info["percentile"], size="1", weight="bold"),  # type: ignore[index]
                        spacing="1",
                        align="center",
                    ),
                ),
                rx.cond(
                    info["quality_label"].to(str) != "",  # type: ignore[union-attr]
                    rx.hstack(
                        rx.text("Quality:", size="1", color="gray"),
                        rx.badge(info["quality_label"], size="1", variant="soft"),  # type: ignore[index]
                        spacing="1",
                        align="center",
                    ),
                ),
                rx.cond(
                    info["match_rate"].to(str) != "None",  # type: ignore[union-attr]
                    rx.hstack(
                        rx.text("Match:", size="1", color="gray"),
                        rx.text(info["match_rate"], size="1"),  # type: ignore[index]
                        spacing="1",
                        align="center",
                    ),
                ),
                spacing="3",
                align="center",
                wrap="wrap",
            ),
            rx.cond(
                info["summary"].to(str) != "",  # type: ignore[union-attr]
                rx.text(info["summary"], size="1", color="var(--gray-11)"),  # type: ignore[index]
            ),
            spacing="2",
            width="100%",
            padding="8px 12px",
            border="1px solid var(--gray-5)",
            border_radius="var(--radius-2)",
            background="var(--gray-2)",
        ),
    )


def _trait_info_panel(state: type[rx.State]) -> rx.Component:
    """Compact info panel for a selected trait."""
    info = state.selected_result_info
    return rx.cond(
        state.selected_result_id != "",
        rx.vstack(
            rx.hstack(
                rx.text(info["trait"], size="2", weight="bold", trim="both"),  # type: ignore[index]
                rx.cond(
                    info["n_models"].to(str) != "None",  # type: ignore[union-attr]
                    rx.badge(
                        rx.text(info["n_models"], " models"),  # type: ignore[index]
                        color_scheme="blue",
                        size="1",
                    ),
                ),
                spacing="2",
                align="center",
                wrap="wrap",
            ),
            rx.hstack(
                rx.cond(
                    info["typical_percentile"].to(str) != "None",  # type: ignore[union-attr]
                    rx.hstack(
                        rx.text("Typical:", size="1", color="gray"),
                        rx.text(info["typical_percentile"], size="1", weight="bold"),  # type: ignore[index]
                        spacing="1",
                        align="center",
                    ),
                ),
                rx.cond(
                    info["reliability"].to(str) != "None",  # type: ignore[union-attr]
                    rx.hstack(
                        rx.text("Reliability:", size="1", color="gray"),
                        rx.badge(info["reliability"], size="1", variant="soft"),  # type: ignore[index]
                        spacing="1",
                        align="center",
                    ),
                ),
                rx.cond(
                    info["overall_signal"].to(str) != "None",  # type: ignore[union-attr]
                    rx.hstack(
                        rx.text("Signal:", size="1", color="gray"),
                        rx.text(info["overall_signal"], size="1"),  # type: ignore[index]
                        spacing="1",
                        align="center",
                    ),
                ),
                spacing="3",
                align="center",
                wrap="wrap",
            ),
            spacing="2",
            width="100%",
            padding="8px 12px",
            border="1px solid var(--gray-5)",
            border_radius="var(--radius-2)",
            background="var(--gray-2)",
        ),
    )


def prs_results_chart_panel(
    state: type[rx.State],
    chart_height: int = 400,
    chart_width: int | str = 560,
    show_info_panel: bool = True,
    show_chart_mode_toggle: bool = True,
    chart_actions: bool | dict = True,
) -> rx.Component:
    """Always-visible Altair chart panel for individual PRS results.

    Shows a placeholder when nothing is selected; updates when a result
    row is clicked.

    Args:
        state: Concrete state class (must mix in ``PRSComputeStateMixin``).
        chart_height: Chart container height in pixels.
        chart_width: Chart width — integer pixels or ``"container"`` for
            responsive full-width.
        show_info_panel: Show the compact metrics panel below the chart.
        show_chart_mode_toggle: Show the single/multi-ancestry toggle.
        chart_actions: Vega-Embed toolbar config.
    """
    return rx.cond(
        state.selected_result_spec != {},
        rx.vstack(
            rx.hstack(
                rx.icon("activity", size=16, color="var(--accent-9)"),
                rx.text("Distribution", size="2", weight="bold"),
                rx.spacer(),
                _chart_mode_toggle(state) if show_chart_mode_toggle else rx.fragment(),
                align="center",
                spacing="2",
                width="100%",
            ),
            rx.box(
                VegaLiteChart.create(
                    spec=state.selected_result_spec,
                    options={"actions": chart_actions, "renderer": "canvas"},
                    width="100%",
                ),
                width="100%",
            ),
            _result_info_panel(state) if show_info_panel else rx.fragment(),
            spacing="3",
            width="100%",
            padding="12px",
            border="1px solid var(--gray-5)",
            border_radius="var(--radius-3)",
            background="var(--color-background)",
        ),
    )


def trait_results_chart_panel(
    state: type[rx.State],
    chart_height: int = 400,
    show_info_panel: bool = True,
    chart_actions: bool | dict = True,
) -> rx.Component:
    """Chart panel for trait-grouped results — visible only after row click.

    Args:
        state: Concrete state class (must mix in ``PRSComputeStateMixin``).
        chart_height: Chart container height in pixels.
        show_info_panel: Show the compact metrics panel below the chart.
        chart_actions: Vega-Embed toolbar config.
    """
    return rx.vstack(
        rx.hstack(
            rx.icon("activity", size=16, color="var(--accent-9)"),
            rx.text("Trait Distribution", size="2", weight="bold"),
            rx.spacer(),
            align="center",
            spacing="2",
            width="100%",
        ),
        rx.cond(
            state.selected_result_spec != {},
            rx.cond(
                state.selected_result_html != "",
                rx.el.iframe(
                    src_doc=state.selected_result_html,
                    width="100%",
                    height=f"{max(chart_height + 420, 760)}px",
                    style={
                        "border": "0",
                        "borderRadius": "var(--radius-2)",
                        "background": "#fafafa",
                    },
                ),
                rx.box(
                    VegaLiteChart.create(
                        spec=state.selected_result_spec,
                        options={"actions": chart_actions, "renderer": "canvas"},
                        width="100%",
                    ),
                    width="100%",
                ),
            ),
            rx.callout(
                "Select PRS result above to view the trait distribution.",
                color_scheme="blue",
                size="1",
                width="100%",
            ),
        ),
        _trait_info_panel(state) if show_info_panel else rx.fragment(),
        spacing="3",
        width="100%",
        padding="12px",
        border="1px solid var(--gray-5)",
        border_radius="var(--radius-3)",
        background="var(--color-background)",
    )


_CLICKABLE_ROW_SX = {
    "& .MuiDataGrid-row": {
        "cursor": "pointer !important",
        "minHeight": "52px !important",
        "maxHeight": "52px !important",
        "borderBottom": "1px solid var(--accent-4)",
        "transition": "background-color 120ms ease, box-shadow 120ms ease",
    },
    "& .MuiDataGrid-row *, & .MuiDataGrid-cell, & .MuiDataGrid-cellContent": {
        "cursor": "pointer !important",
        "userSelect": "none",
    },
    "& .MuiDataGrid-cell": {
        "display": "flex",
        "alignItems": "center",
        "fontSize": "0.9rem",
    },
    "& .MuiDataGrid-row:hover": {
        "backgroundColor": "var(--accent-3)",
        "boxShadow": "inset 3px 0 0 var(--accent-9)",
    },
    "& .MuiDataGrid-row:hover .MuiDataGrid-cell": {
        "color": "var(--accent-12)",
    },
}


_CLICKABLE_GRID_WRAPPER_STYLE = {
    "cursor": "pointer",
    "& .MuiDataGrid-root": {"cursor": "pointer !important"},
    "& .MuiDataGrid-main": {"cursor": "pointer !important"},
    "& .MuiDataGrid-virtualScroller": {"cursor": "pointer !important"},
    "& .MuiDataGrid-virtualScrollerContent": {"cursor": "pointer !important"},
    "& .MuiDataGrid-virtualScrollerRenderZone": {"cursor": "pointer !important"},
    "& .MuiDataGrid-row": {"cursor": "pointer !important"},
    "& .MuiDataGrid-row *": {
        "cursor": "pointer !important",
        "userSelect": "none",
    },
    "& .MuiDataGrid-cell": {"cursor": "pointer !important"},
    "& .MuiDataGrid-cellContent": {"cursor": "pointer !important"},
}


def _results_action_bar(state: type[rx.State], prompt: str) -> rx.Component:
    """Instruction and explicit result deletion controls."""
    return rx.hstack(
        rx.box(
            prompt,
            flex="1",
            padding="6px 10px",
            border="1px solid var(--blue-6)",
            border_radius="var(--radius-2)",
            background="var(--blue-2)",
            color="var(--blue-11)",
            font_size="var(--font-size-1)",
        ),
        rx.button(
            rx.icon("x", size=14),
            "Remove selected",
            on_click=state.remove_selected_result,
            size="1",
            variant="soft",
            color_scheme="orange",
        ),
        rx.button(
            rx.icon("trash-2", size=14),
            "Clear all",
            on_click=state.clear_prs_results,
            size="1",
            variant="soft",
            color_scheme="red",
        ),
        spacing="2",
        align="center",
        width="100%",
    )


def prs_results_clickable_table(
    state: type[rx.State],
    table_height: str | None = None,
) -> rx.Component:
    """Compact PRS results DataGrid — click a row to chart it.

    Args:
        state: Concrete state class (must mix in ``PRSComputeStateMixin``).
        table_height: CSS height for the results table container.
    """
    return rx.cond(
        state.prs_results.length() > 0,  # type: ignore[operator]
        rx.vstack(
            _results_action_bar(
                state,
                "Select PRS result above to view its distribution.",
            ),
            rx.box(
                data_grid_scroll_container(
                    data_grid(
                        rows=state.prs_results_rows,
                        columns=state.prs_results_columns,
                        column_grouping_model=state.prs_results_column_groups,
                        row_id_field="id",
                        pagination=False,
                        hide_footer=True,
                        density="standard",
                        height="100%",
                        row_height=52,
                        column_header_height=40,
                        disable_row_selection_on_click=True,
                        on_row_click=state.select_prs_result,
                        sx=_CLICKABLE_ROW_SX,
                    ),
                ),
                height=table_height or state.prs_results_table_height,
                width="100%",
                overflow="hidden",
                style=_CLICKABLE_GRID_WRAPPER_STYLE,
            ),
            spacing="1",
            width="100%",
        ),
    )


def trait_results_clickable_table(
    state: type[rx.State],
    table_height: str | None = None,
) -> rx.Component:
    """Compact trait summary DataGrid — click a row to chart it.

    Args:
        state: Concrete state class (must mix in ``PRSComputeStateMixin``).
        table_height: CSS height for the trait table container.
    """
    return rx.cond(
        (state.prs_results.length() > 0) & (state.prs_view_mode == "grouped"),  # type: ignore[operator]
        rx.vstack(
            _results_action_bar(
                state,
                "Select PRS result above to view the trait distribution.",
            ),
            rx.box(
                data_grid_scroll_container(
                    data_grid(
                        rows=state.trait_summary_rows,
                        columns=state.trait_summary_columns,
                        row_id_field="id",
                        pagination=False,
                        hide_footer=True,
                        density="standard",
                        height="100%",
                        row_height=52,
                        column_header_height=40,
                        disable_row_selection_on_click=True,
                        on_row_click=state.select_trait_result,
                        sx=_CLICKABLE_ROW_SX,
                    ),
                ),
                height=table_height or state.trait_results_table_height,
                width="100%",
                overflow="hidden",
                style=_CLICKABLE_GRID_WRAPPER_STYLE,
            ),
            spacing="1",
            width="100%",
        ),
    )


def prs_results_with_chart(
    state: type[rx.State],
    chart_height: int = 400,
    table_height: str | None = None,
    show_info_panel: bool = True,
    show_chart_mode_toggle: bool = True,
    show_header: bool = True,
    chart_actions: bool | dict = True,
) -> rx.Component:
    """Stacked layout: compact clickable results table + Altair chart panel.

    This is the Altair-based replacement for ``prs_results_table()``
    (which uses accordion detail panels). The results table shows compact
    rows; clicking a row populates the always-visible chart panel below.

    Args:
        state: Concrete state class (must mix in ``PRSComputeStateMixin``).
        chart_height: Height of the chart panel in pixels.
        table_height: CSS height of the results table.
        show_info_panel: Show metrics panel below the chart.
        show_chart_mode_toggle: Show single/multi-ancestry toggle.
        show_header: Show the PRS Results header. Workbench callers render a
            shared header above the active result view.
        chart_actions: Vega-Embed toolbar config.
    """
    return rx.cond(
        (state.prs_results.length() > 0) & (state.prs_view_mode == "individual"),  # type: ignore[operator]
        rx.vstack(
            _prs_disclaimers(state),
            _prs_interpretation_guide(),
            _prs_results_header(state, show_view_toggle=False) if show_header else rx.fragment(),
            prs_results_clickable_table(state, table_height=table_height),
            prs_results_chart_panel(
                state,
                chart_height=chart_height,
                show_info_panel=show_info_panel,
                show_chart_mode_toggle=show_chart_mode_toggle,
                chart_actions=chart_actions,
            ),
            spacing="3",
            width="100%",
        ),
    )


def trait_results_with_chart(
    state: type[rx.State],
    chart_height: int = 400,
    table_height: str | None = None,
    show_info_panel: bool = True,
    chart_actions: bool | dict = True,
) -> rx.Component:
    """Stacked layout: compact clickable trait table + Altair chart panel.

    This is the Altair-based replacement for ``trait_summary_table()``
    (which uses accordion detail panels). The trait table shows compact
    rows; clicking a row populates the always-visible chart panel below.

    Args:
        state: Concrete state class (must mix in ``PRSComputeStateMixin``).
        chart_height: Height of the chart panel in pixels.
        table_height: CSS height of the trait table.
        show_info_panel: Show metrics panel below the chart.
        chart_actions: Vega-Embed toolbar config.
    """
    return rx.cond(
        (state.prs_results.length() > 0) & (state.prs_view_mode == "grouped"),  # type: ignore[operator]
        rx.vstack(
            trait_results_clickable_table(state, table_height=table_height),
            trait_results_chart_panel(
                state,
                chart_height=chart_height,
                show_info_panel=show_info_panel,
                chart_actions=chart_actions,
            ),
            spacing="3",
            width="100%",
        ),
    )


def prs_shared_build_bar(source_state: type[rx.State]) -> rx.Component:
    """Genome-build selector owned by a shared genotype source.

    Unlike :func:`prs_build_selector` (which is per-consumer), this binds to a
    source state's ``genome_build`` and ``set_shared_genome_build`` so a single
    control fans the build out to every consumer of that source.
    """
    return rx.hstack(
        rx.text("Genome Build:", size="2", weight="medium"),
        rx.select(
            ["GRCh37", "GRCh38"],
            value=source_state.genome_build,
            on_change=source_state.set_shared_genome_build,
            size="2",
        ),
        rx.tooltip(
            rx.icon("info", size=14, color="gray"),
            content=(
                "Genome build of your uploaded data. Auto-detected from the VCF "
                "header when possible; the selected build is applied to both the "
                "'By PRS' and 'By Trait' score selections."
            ),
        ),
        spacing="2",
        align="center",
    )


def _workbench_mode_controls(state: type[rx.State]) -> rx.Component:
    """Per-mode controls (engine, ancestry, harmonized) for a consumer state."""
    return rx.vstack(
        rx.hstack(
            prs_engine_selector(state),
            rx.separator(orientation="vertical", size="2"),
            rx.checkbox(
                "Include harmonized scores",
                checked=state.include_harmonized,
                on_change=state.set_include_harmonized,
                size="2",
            ),
            spacing="4",
            align="center",
            wrap="wrap",
            width="100%",
        ),
        prs_ancestry_selector(state),
        spacing="3",
        align="start",
        width="100%",
    )


def _workbench_compute_button(
    state: type[rx.State],
    label: str,
    normalizing: Any | None = None,
) -> rx.Component:
    """Compute button + disclaimer that reads genotype readiness from the consumer.

    Decoupled from any VCF source: readiness is inferred from the consumer's own
    ``prs_genotypes_path`` (set by the source via ``load_genotypes``), so the
    same button works regardless of where the genotypes came from.
    """
    is_normalizing = _resolve_normalizing(normalizing)
    not_ready = (state.selected_pgs_ids.length() == 0) | (state.prs_genotypes_path == "") | is_normalizing  # type: ignore[operator]
    return rx.vstack(
        rx.callout(
            "You are responsible for ensuring that your genomic data matches the "
            "population used in the selected PRS and that the genome build is "
            "correct for the scoring files.",
            icon="triangle_alert",
            color_scheme="amber",
            size="1",
            width="100%",
        ),
        rx.hstack(
            rx.button(
                rx.icon("calculator", size=14),
                label,
                on_click=state.compute_selected_prs,
                loading=state.prs_computing,
                disabled=not_ready,
                color_scheme="green",
                size="3",
            ),
            rx.cond(
                not_ready,
                rx.text(
                    rx.cond(
                        is_normalizing,
                        "VCF normalization in progress...",
                        rx.cond(
                            state.prs_genotypes_path == "",
                            rx.cond(
                                state.selected_pgs_ids.length() == 0,  # type: ignore[operator]
                                "Load genomic data and select at least one score to compute.",
                                "Load genomic data to enable computation.",
                            ),
                            "Select at least one score to enable computation.",
                        ),
                    ),
                    size="2",
                    color="gray",
                ),
            ),
            spacing="3",
            align="center",
        ),
        spacing="3",
        width="100%",
    )


def _workbench_results(
    state: type[rx.State],
    view_mode: str,
    results_table_kwargs: dict[str, Any] | None,
    trait_summary_kwargs: dict[str, Any] | None,
) -> rx.Component:
    """Progress, results header, and the active workbench result view.

    Uses the Altair-based click-to-select layout: compact clickable table
    on top + always-visible chart panel below.
    """
    individual_kwargs = {**(results_table_kwargs or {}), "show_header": False}
    result_view = (
        trait_results_with_chart(state, **(trait_summary_kwargs or {}))
        if view_mode == "grouped"
        else prs_results_with_chart(state, **individual_kwargs)
    )
    return rx.fragment(
        prs_progress_section(state),
        _prs_results_header(state, show_view_toggle=False),
        result_view,
    )


def prs_workbench_mode_panel(
    state: type[rx.State],
    selector: Callable[[], rx.Component],
    view_mode: str,
    compute_label: str = "Compute PRS",
    results_table_kwargs: dict[str, Any] | None = None,
    trait_summary_kwargs: dict[str, Any] | None = None,
    normalizing: Any | None = None,
) -> rx.Component:
    """Single By PRS or By Trait workbench panel.

    The shared genotype source and genome-build bar can live outside the panel,
    which lets host apps expose By PRS / By Trait as top-level tabs while still
    reusing the same loaded genotypes.
    """
    return rx.vstack(
        _workbench_mode_controls(state),
        selector(),
        _workbench_compute_button(state, compute_label, normalizing=normalizing),
        _workbench_results(state, view_mode, results_table_kwargs, trait_summary_kwargs),
        width="100%",
        spacing="4",
        padding="16px",
    )


def prs_workbench(
    source_section: rx.Component,
    prs_state: type[rx.State],
    trait_state: type[rx.State],
    mode_state: type[rx.State],
    trait_selector: Callable[[], rx.Component],
    build_bar: rx.Component | None = None,
    results_table_kwargs: dict[str, Any] | None = None,
    trait_summary_kwargs: dict[str, Any] | None = None,
    normalizing: Any | None = None,
) -> rx.Component:
    """Unified PRS workbench: one shared genotype source + By PRS / By Trait modes.

    A single, detachable genotype ``source_section`` feeds two consumer states
    (``prs_state`` for individual-score selection, ``trait_state`` for
    trait-group selection).  A segmented control bound to ``mode_state`` switches
    between the two; results are shown for the active mode only.

    The source is loosely coupled: it only needs to push normalized genotypes
    into the consumers via their inherited ``load_genotypes(path)`` hook (and
    optionally ``set_genome_build``).  A host app (e.g. just-dna-lite) can pass
    its own ``source_section`` (public genome, array file, ...) without changing
    the consumers or ``PRSComputeStateMixin``.

    Args:
        source_section: The genotype source UI (e.g. ``vcf_source_section(...)``).
        prs_state: Concrete "By PRS" consumer state (mixes in PRSComputeStateMixin).
        trait_state: Concrete "By Trait" consumer state.
        mode_state: State providing ``compute_mode`` / ``set_compute_mode``.
        trait_selector: Zero-arg callable returning the trait-selection grid UI.
        build_bar: Optional shared genome-build control (e.g. ``prs_shared_build_bar``).
        results_table_kwargs: Forwarded to :func:`prs_results_table`.
        trait_summary_kwargs: Forwarded to :func:`trait_summary_table`.
    """
    def _tab_trigger(label: str, icon_name: str, value: str) -> rx.Component:
        return rx.tabs.trigger(
            rx.hstack(
                rx.icon(icon_name, size=18),
                rx.text(label, size="3", weight="bold"),
                spacing="2",
                align="center",
            ),
            value=value,
            padding="10px 24px",
            cursor="pointer",
        )

    prs_content = prs_workbench_mode_panel(
        prs_state,
        lambda: prs_scores_selector(prs_state, normalizing=normalizing),
        "individual",
        "Compute PRS",
        results_table_kwargs,
        trait_summary_kwargs,
        normalizing,
    )
    trait_content = prs_workbench_mode_panel(
        trait_state,
        trait_selector,
        "grouped",
        "Compute PRS for Selected Traits",
        results_table_kwargs,
        trait_summary_kwargs,
        normalizing,
    )

    return rx.theme(
        rx.vstack(
            source_section,
            build_bar if build_bar is not None else rx.fragment(),
            rx.tabs.root(
                rx.tabs.list(
                    _tab_trigger("Select by Trait", "layers", "trait"),
                    _tab_trigger("Select by PRS", "list-checks", "prs"),
                    size="2",
                ),
                rx.tabs.content(trait_content, value="trait", width="100%"),
                rx.tabs.content(prs_content, value="prs", width="100%"),
                value=mode_state.compute_mode,
                on_change=mode_state.set_compute_mode,
                width="100%",
            ),
            width="100%",
            spacing="4",
        ),
        has_background=False,
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
