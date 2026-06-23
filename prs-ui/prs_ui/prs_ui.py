"""PRS UI: Reflex app for computing polygenic risk scores by PRS or trait."""

import traceback

import reflex as rx
from reflex_base.event import EventSpec
from reflex.utils import console
import reflex_mui_datagrid.datagrid as mui_datagrid
from rich.markup import escape

from prs_ui.grid_style import data_grid_scroll_css
from prs_ui.pages.compute import by_prs_panel, by_trait_panel, shared_genotype_source
from prs_ui.pages.faq import faq_panel
from prs_ui.state import AppState, ComputeGridState, TraitBrowserState


GITHUB_REPO_URL = "https://github.com/antonkulaga/just-prs"


def _patch_metric_list_card_alignment() -> None:
    """Keep detail metric cards content-height instead of flex-row stretched.

    ``reflex-mui-datagrid`` 0.3.8 renders ``metric_list`` as a wrapping flex row
    without ``alignItems``. The browser default is ``stretch``, so every card in a
    row inherits the height of the tallest card. Patch the injected wrapper until
    the upstream renderer exposes this as a config option.
    """
    stretched = 'style: { display: "flex", flexWrap: "wrap", gap: cardGap, padding: "4px 0" },'
    content_height = (
        'style: { display: "flex", flexWrap: "wrap", alignItems: "flex-start", '
        'gap: cardGap, padding: "4px 0" },'
    )
    wrapper_js = mui_datagrid._INLINE_WRAPPER_JS
    if stretched in wrapper_js and content_height not in wrapper_js:
        mui_datagrid._INLINE_WRAPPER_JS = wrapper_js.replace(stretched, content_height)


_patch_metric_list_card_alignment()


def _tab_label(icon: str, label: str, description: str) -> rx.Component:
    """Tab label with a compact explanation for non-specialist users."""
    return rx.hstack(
        rx.icon(icon, size=18),
        rx.text(label, size="3", weight="bold"),
        rx.tooltip(
            rx.icon("info", size=14, color="gray"),
            content=description,
        ),
        spacing="2",
        align="center",
    )


def _safe_backend_exception_handler(exception: Exception) -> EventSpec:
    """Log backend exceptions without letting Rich parse traceback text as markup."""
    from reflex_components_sonner.toast import toast

    error = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    console.error(f"[Reflex Backend Exception]\n {escape(error)}\n")
    return toast(
        "An error occurred.",
        level="error",
        fallback_to_alert=True,
        description=f"{type(exception).__name__}: {exception}\nSee logs for details.",
        position="top-center",
        id="backend_error",
        style={"width": "500px", "white-space": "pre-wrap"},
    )


def index() -> rx.Component:
    return rx.box(
        data_grid_scroll_css(),
        rx.link(
            "Fork me on GitHub",
            href=GITHUB_REPO_URL,
            is_external=True,
            position="absolute",
            top="18px",
            right="-44px",
            transform="rotate(45deg)",
            background="#111111",
            color="white",
            padding="6px 48px",
            font_size="12px",
            font_weight="700",
            letter_spacing="0.02em",
            text_decoration="none",
            box_shadow="0 2px 8px rgba(0, 0, 0, 0.18)",
            z_index="10",
            _hover={"background": "#000000", "text_decoration": "none"},
        ),
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.heading("just-prs", size="5"),
                    rx.text(
                        "Compute and interpret polygenic risk scores from your genome "
                        "using PGS Catalog models. Upload a VCF once, then explore by "
                        "specific PRS or by trait.",
                        size="2",
                        color="gray",
                        max_width="760px",
                    ),
                    spacing="1",
                    align="start",
                ),
                rx.spacer(),
                width="100%",
                align="start",
                padding_x="16px",
                padding_y="12px",
            ),
            rx.separator(),
            rx.cond(
                AppState.active_tab != "faq",
                rx.box(shared_genotype_source(), padding="12px 16px 0 16px"),
            ),
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger(
                        _tab_label(
                            "layers",
                            "By Trait",
                            "Start from a disease or phenotype, then compute related "
                            "PGS models together.",
                        ),
                        value="trait",
                        padding="10px 20px",
                        cursor="pointer",
                    ),
                    rx.tabs.trigger(
                        _tab_label(
                            "list-checks",
                            "By PRS",
                            "Choose specific PGS Catalog scoring models and compute "
                            "them for the uploaded genome.",
                        ),
                        value="prs",
                        padding="10px 20px",
                        cursor="pointer",
                    ),
                    rx.tabs.trigger(
                        _tab_label(
                            "circle-help",
                            "FAQ",
                            "Learn what PRS results mean, how to use the app, and what "
                            "limitations to keep in mind.",
                        ),
                        value="faq",
                        padding="10px 20px",
                        cursor="pointer",
                    ),
                    size="2",
                ),
                rx.tabs.content(
                    rx.box(by_trait_panel(), padding_top="12px"),
                    value="trait",
                ),
                rx.tabs.content(
                    rx.box(by_prs_panel(), padding_top="12px"),
                    value="prs",
                ),
                rx.tabs.content(
                    rx.box(faq_panel(), padding_top="12px"),
                    value="faq",
                ),
                value=AppState.active_tab,
                on_change=AppState.set_active_tab,  # type: ignore[arg-type]
                width="100%",
            ),
            width="100%",
            spacing="0",
        ),
        padding="16px",
        width="100%",
        min_height="100vh",
        position="relative",
        overflow_x="hidden",
    )


app = rx.App(backend_exception_handler=_safe_backend_exception_handler)
app.add_page(
    index,
    title="just-prs",
    on_load=[ComputeGridState.initialize, TraitBrowserState.initialize_traits],
)


def main() -> None:
    """CLI entrypoint: launch the Reflex dev server."""
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent.parent)
    from reflex.reflex import cli
    cli(["run"])
