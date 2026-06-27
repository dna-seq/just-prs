from importlib.metadata import metadata

from just_prs.ancestry import infer_ancestry
from just_prs.array_scoring import compute_array_prs
from just_prs.arrays import detect_array_format, detect_chip_generation, normalize_array
from just_prs.chip_coverage import (
    CHIPS,
    CHIPS_BY_ID,
    Chip,
    chip_typed_positions,
    compute_chip_coverage,
    parse_gsa_manifest,
)
from just_prs.enrich import enrich_prs_result
from just_prs.ld_proxy import apply_ld_proxies, build_ld_proxy_table, ld_proxy_table_path
from just_prs.models import (
    AbsoluteRisk,
    AbsoluteRiskBundle,
    AbsoluteRiskEstimate,
    AncestryCoherence,
    AncestryInference,
    ArrayPRSResult,
    ChipGeneration,
    EnrichedPRSResult,
    PercentileResult,
    PerformanceInfo,
    PRSBatchOutcome,
    PRSBatchResult,
    PRSResult,
    ReferenceDistribution,
    classify_coverage_tier,
)
from just_prs.normalize import VcfFilterConfig, normalize_vcf
from just_prs.prs import (
    GenotypeInputMode,
    PRSEngine,
    RestorationScope,
    compute_prs,
    compute_prs_batch,
    compute_prs_duckdb,
)
from just_prs.prs_catalog import PRSCatalog
from just_prs.quality import (
    classify_combined_quality,
    classify_model_quality,
    classify_synthetic_quality,
    format_classification,
    format_effect_size,
    interpret_prs_result,
    synthetic_quality_score,
    synthetic_quality_tier,
)
from just_prs.reference import (
    BatchScoringResult,
    DEFAULT_PANEL,
    REFERENCE_FASTA,
    REFERENCE_PANELS,
    ReferencePanelError,
    ScoringOutcome,
    aggregate_distributions,
    ancestry_percentile,
    compute_reference_prs_batch,
    compute_reference_prs_polars,
    distribution_quality_issues,
    download_reference_fasta,
    download_reference_panel,
    enrich_distributions,
    match_scoring_to_pvar,
    parse_psam,
    parse_pvar,
    read_pgen_genotypes,
    reference_fasta_path,
    reference_panel_dir,
    reference_distribution_audit_issues,
)
from just_prs.reference_allele import RefSource, resolve_reference_alleles
from just_prs.liftover import (
    LiftOver,
    LiftoverConfigurationError,
    LiftRecordResult,
    download_chain_file,
    get_liftover,
    lift_frame,
    liftover_preflight,
)
from just_prs.scoring import resolve_cache_dir

_meta = metadata("just-prs")
__version__: str = _meta["Version"]
__package_name__: str = _meta["Name"]
