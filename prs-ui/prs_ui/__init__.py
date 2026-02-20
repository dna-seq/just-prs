from importlib.metadata import metadata

from prs_ui.components.prs_section import (
    prs_ancestry_selector,
    prs_build_selector,
    prs_compute_button,
    prs_progress_section,
    prs_results_table,
    prs_scores_selector,
    prs_section,
)
from prs_ui.state import PRSComputeStateMixin

_meta = metadata("prs-ui")
__version__: str = _meta["Version"]
__package_name__: str = _meta["Name"]

__all__ = [
    "PRSComputeStateMixin",
    "prs_ancestry_selector",
    "prs_build_selector",
    "prs_compute_button",
    "prs_progress_section",
    "prs_results_table",
    "prs_scores_selector",
    "prs_section",
    "__version__",
    "__package_name__",
]
