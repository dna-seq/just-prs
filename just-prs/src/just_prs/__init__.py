from importlib.metadata import metadata

from just_prs.models import ReferenceDistribution
from just_prs.normalize import VcfFilterConfig, normalize_vcf
from just_prs.prs_catalog import PRSCatalog
from just_prs.quality import (
    classify_model_quality,
    format_classification,
    format_effect_size,
    interpret_prs_result,
)
from just_prs.reference import (
    BatchScoringResult,
    DEFAULT_PANEL,
    REFERENCE_PANELS,
    ReferencePanelError,
    ScoringOutcome,
    aggregate_distributions,
    ancestry_percentile,
    compute_reference_prs_batch,
    compute_reference_prs_polars,
    download_reference_panel,
    enrich_distributions,
    match_scoring_to_pvar,
    parse_psam,
    parse_pvar,
    read_pgen_genotypes,
    reference_panel_dir,
)
from just_prs.scoring import resolve_cache_dir

_meta = metadata("just-prs")
__version__: str = _meta["Version"]
__package_name__: str = _meta["Name"]
