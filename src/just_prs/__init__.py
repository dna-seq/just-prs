from importlib.metadata import metadata

from just_prs.normalize import VcfFilterConfig, normalize_vcf
from just_prs.prs_catalog import PRSCatalog
from just_prs.quality import (
    classify_model_quality,
    format_classification,
    format_effect_size,
    interpret_prs_result,
)
from just_prs.scoring import resolve_cache_dir

_meta = metadata("just-prs")
__version__: str = _meta["Version"]
__package_name__: str = _meta["Name"]

__all__ = [
    "PRSCatalog",
    "VcfFilterConfig",
    "classify_model_quality",
    "format_classification",
    "format_effect_size",
    "interpret_prs_result",
    "normalize_vcf",
    "resolve_cache_dir",
    "__version__",
    "__package_name__",
]
