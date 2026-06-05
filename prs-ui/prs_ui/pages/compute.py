"""Compute PRS page: a single workbench with a shared, detachable VCF source.

The page composes the reusable :func:`prs_ui.components.prs_workbench` with the
reference VCF genotype source (:func:`vcf_source_section`).  One uploaded VCF is
normalized once by ``GenomicGridState`` and fed into both the "By PRS"
(``ComputeGridState``) and "By Trait" (``TraitBrowserState``) consumer states.
"""

import reflex as rx

from prs_ui.components.prs_section import prs_shared_build_bar, prs_workbench
from prs_ui.components.vcf_source import vcf_source_section
from prs_ui.pages.traits import trait_selector
from prs_ui.state import AppState, ComputeGridState, GenomicGridState, TraitBrowserState


def compute_panel() -> rx.Component:
    """Unified Compute PRS workbench: shared VCF source + By PRS / By Trait modes."""
    return prs_workbench(
        source_section=vcf_source_section(GenomicGridState, upload_id="vcf_upload"),
        prs_state=ComputeGridState,
        trait_state=TraitBrowserState,
        mode_state=AppState,
        trait_selector=trait_selector,
        build_bar=prs_shared_build_bar(GenomicGridState),
    )
