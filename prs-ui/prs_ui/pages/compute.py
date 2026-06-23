"""Compute PRS pages built from reusable By PRS / By Trait workbench panels.

The app shell owns the shared VCF source so the public UI can expose "By PRS"
and "By Trait" as top-level tabs without duplicating upload controls.
"""

import reflex as rx

from prs_ui.components.prs_section import (
    prs_scores_selector,
    prs_shared_build_bar,
    prs_workbench,
    prs_workbench_mode_panel,
)
from prs_ui.components.vcf_source import vcf_source_section
from prs_ui.pages.traits import trait_selector
from prs_ui.state import AppState, ComputeGridState, GenomicGridState, TraitBrowserState


def shared_genotype_source() -> rx.Component:
    """Shared VCF source and genome-build control for the compute tabs."""
    return rx.vstack(
        vcf_source_section(GenomicGridState, upload_id="vcf_upload"),
        prs_shared_build_bar(GenomicGridState),
        width="100%",
        spacing="3",
    )


def by_prs_panel() -> rx.Component:
    """Top-level By PRS tab: select individual scoring models and compute them."""
    return prs_workbench_mode_panel(
        state=ComputeGridState,
        selector=lambda: prs_scores_selector(ComputeGridState),
        view_mode="individual",
        compute_label="Compute PRS",
    )


def by_trait_panel() -> rx.Component:
    """Top-level By Trait tab: select traits and compute associated scoring models."""
    return prs_workbench_mode_panel(
        state=TraitBrowserState,
        selector=trait_selector,
        view_mode="grouped",
        compute_label="Compute PRS for Selected Traits",
    )


def compute_panel() -> rx.Component:
    """Legacy unified workbench with nested By PRS / By Trait tabs."""
    return prs_workbench(
        source_section=vcf_source_section(GenomicGridState, upload_id="vcf_upload"),
        prs_state=ComputeGridState,
        trait_state=TraitBrowserState,
        mode_state=AppState,
        trait_selector=trait_selector,
        build_bar=prs_shared_build_bar(GenomicGridState),
    )
