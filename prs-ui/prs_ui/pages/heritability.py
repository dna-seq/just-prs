"""Trait SNP heritability browser: LDSC estimates from trait_heritability.parquet."""

import reflex as rx
from reflex_mui_datagrid import lazyframe_grid, lazyframe_grid_stats_bar

from prs_ui.grid_style import data_grid_scroll_container
from prs_ui.state import HeritabilityGridState, SUPERPOPULATION_LABELS

_ACCORDION_SUMMARY_STYLE: dict[str, str] = {
    "cursor": "pointer",
    "display": "list-item",
    "listStylePosition": "inside",
    "width": "fit-content",
}


def _heritability_intro_callouts() -> rx.Component:
    """Citizen-science framing: what SNP h² means and why only LDSC rows are shown."""
    return rx.vstack(
        rx.callout(
            rx.vstack(
                rx.text(
                    "SNP heritability (often written h² or SNP-h²) is a population statistic: "
                    "roughly, how much of the differences between people in a large study "
                    "were tied to common genetic variants measured on a SNP chip — not how "
                    "\"genetic\" any one person is, and not a personal diagnosis.",
                    size="2",
                ),
                rx.text(
                    "Here we only show LD Score Regression results — labelled "
                    "S-LDSC (Pan-UK Biobank, mainly European ancestry summaries) or LDSC "
                    "(GWAS Atlas). Those estimates usually fall between 0 and 1 when "
                    "reported as a fraction of variance (larger means genetics explains "
                    "more trait variation in that dataset). Other estimation methods were "
                    "dropped from this view because they sometimes produced extreme "
                    "values that are hard to interpret next to LDSC.",
                    size="2",
                    padding_top="6px",
                ),
                spacing="0",
                width="100%",
            ),
            icon="info",
            color_scheme="blue",
            size="2",
            width="100%",
        ),
        rx.callout(
            "Use these numbers for curiosity and learning only. They summarize published "
            "bulk analyses and may not transfer to every ancestry or setting.",
            icon="triangle_alert",
            color_scheme="amber",
            size="1",
            width="100%",
        ),
        spacing="2",
        width="100%",
    )


def _heritability_guide_accordion() -> rx.Component:
    """Expandable legend: columns, confidence, ancestry, sources."""
    ancestry_items = [
        rx.el.li(rx.text(code + " — " + label, size="2"))
        for code, label in sorted(SUPERPOPULATION_LABELS.items())
    ]
    ancestry_items.append(
        rx.el.li(
            rx.text(
                "MID — Middle Eastern (Pan-UKBB internal ancestry label when present).",
                size="2",
            ),
        ),
    )

    return rx.el.details(
        rx.el.summary(
            rx.hstack(
                rx.icon("book-open", size=16),
                rx.text("How to read this table", size="3", weight="bold"),
                spacing="2",
                align="center",
            ),
            style=_ACCORDION_SUMMARY_STYLE,
        ),
        rx.vstack(
            rx.text("Columns", size="2", weight="bold"),
            rx.el.ul(
                rx.el.li(
                    rx.text(
                        "Trait label / EFO ID — trait name from the source file and ontology "
                        "identifier for linking to other catalog data.",
                        size="2",
                    ),
                ),
                rx.el.li(
                    rx.text(
                        "h² observed vs h² liability — for continuous traits you mostly see "
                        "observed-scale estimates; for binary disease traits researchers "
                        "often report liability-scale h² (still typically between 0 and 1 "
                        "when expressed as SNP variance contribution). When only one is "
                        "filled, the table sorts by whichever is available.",
                        size="2",
                    ),
                ),
                rx.el.li(
                    rx.text(
                        "h² z — standardized signal strength from the source pipeline "
                        "(rough rule of thumb for exploration: larger magnitude usually "
                        "means a more statistically backed estimate in that cohort).",
                        size="2",
                    ),
                ),
                rx.el.li(
                    rx.text(
                        "Confidence — heuristic bucket from z-score bands in our ingest "
                        "(high / moderate / low), not a clinical endorsement.",
                        size="2",
                    ),
                ),
                rx.el.li(
                    rx.text(
                        "Method — S-LDSC or LDSC here means LD Score Regression lineage "
                        "used to derive SNP heritability.",
                        size="2",
                    ),
                ),
                rx.el.li(
                    rx.text(
                        "Source — pan_ukbb is multi-ancestry Pan-UKBB summary metadata; "
                        "gwas_atlas is an older archival GWAS Atlas snapshot used only "
                        "where mapping exists.",
                        size="2",
                    ),
                ),
                rx.el.li(
                    rx.text(
                        "n samples — sample size reported by the source when present.",
                        size="2",
                    ),
                ),
            ),
            rx.text("Ancestry abbreviations", size="2", weight="bold", padding_top="10px"),
            rx.el.ul(*ancestry_items),
            rx.text(
                "Sorting — rows are ordered by the larger of liability or observed h² "
                "within this filtered LDSC-only subset.",
                size="2",
                color="var(--gray-11)",
                padding_top="10px",
            ),
            spacing="1",
            padding_top="8px",
            padding_x="4px",
            width="100%",
        ),
        width="100%",
        style={
            "border": "1px solid var(--gray-5)",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "background": "var(--gray-1)",
        },
    )


def heritability_panel() -> rx.Component:
    """Panel showing trait SNP heritability (LDSC family methods only)."""
    return rx.vstack(
        _heritability_intro_callouts(),
        _heritability_guide_accordion(),
        rx.hstack(
            rx.text(
                "Filtered to LD Score Regression (S-LDSC / LDSC). Reload pulls the latest "
                "trait_heritability parquet from your catalog cache.",
                size="2",
                color="gray",
                flex="1",
            ),
            rx.button(
                rx.icon("refresh-ccw", size=14),
                "Reload",
                on_click=HeritabilityGridState.load_heritability,
                loading=HeritabilityGridState.lf_grid_loading,
                variant="outline",
                size="2",
            ),
            spacing="3",
            align="center",
            width="100%",
            wrap="wrap",
        ),
        rx.cond(
            HeritabilityGridState.lf_grid_loaded,
            rx.vstack(
                lazyframe_grid_stats_bar(HeritabilityGridState),
                data_grid_scroll_container(
                    lazyframe_grid(
                        HeritabilityGridState,
                        height="calc(100vh - 520px)",
                        density="compact",
                        column_header_height=56,
                    ),
                ),
                width="100%",
                spacing="2",
            ),
            rx.cond(
                HeritabilityGridState.lf_grid_loading,
                rx.center(rx.spinner(size="3"), padding="60px"),
                rx.center(
                    rx.text(
                        "Heritability table not loaded yet.",
                        color="gray",
                        size="3",
                    ),
                    padding="60px",
                ),
            ),
        ),
        width="100%",
        spacing="3",
    )
