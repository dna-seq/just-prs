"""Frozen Pydantic 2 schemas for sample-score integration outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from just_prs.sample_scores.integration.pins import StagingIndexRow
from just_prs.sample_scores.models import SampleAncestryRecord

MODEL_ANALYSIS_SCHEMA_VERSION = "1"
TRAIT_SUMMARY_SCHEMA_VERSION = "1"
INTEGRATION_MANIFEST_SCHEMA_VERSION = "1"

MODEL_ANALYSIS_PRIMARY_KEY: tuple[str, ...] = (
    "sample_id",
    "pgs_id",
    "scoring_build",
    "score_profile_id",
    "scoring_fingerprint",
)
TRAIT_SUMMARY_PRIMARY_KEY: tuple[str, ...] = (
    "sample_id",
    "trait_id",
    "score_profile_id",
    "percentile_population",
)

EXCLUSION_FAILED = "failed"
EXCLUSION_RUNTIME_INVARIANT = "runtime_invariant_failed"
EXCLUSION_FINGERPRINT = "scoring_fingerprint_inconsistent"
EXCLUSION_NOT_PUBLISHED = "not_publication_allowed"
EXCLUSION_QUARANTINED = "quarantined"
EXCLUSION_DIST_UNAVAILABLE = "reference_distribution_unavailable"
EXCLUSION_AUDIT_ERROR = "reference_audit_error"
EXCLUSION_ANCESTRY_UNKNOWN = "ancestry_unknown"
EXCLUSION_ANCESTRY_MISSING = "ancestry_missing"

FINE_COHORT_LABELS: dict[str, str] = {
    "CEU": "Northern/Western European",
    "IBS": "Iberian/Spanish",
}


class PopulationMetric(BaseModel):
    """One 1000G superpopulation calibration for a single PGS model."""

    superpopulation: str
    mean: float | None = None
    std: float | None = None
    n: int | None = None
    z_score: float | None = None
    percentile: float | None = None
    panel: str = "1000g"
    source_revision: str | None = None
    available: bool = False
    exclusion_reason: str | None = None
    caveat: str | None = None


class TraitPopulationRisk(BaseModel):
    """Trait-specific absolute risk / h² at one reference population."""

    superpopulation: str
    prevalence: float | None = None
    prevalence_source: str | None = None
    prevalence_type: str | None = None
    absolute_risk: float | None = None
    risk_ratio: float | None = None
    risk_method: str | None = None
    h2: float | None = None
    h2_source: str | None = None
    h2_scale: str | None = None
    h2_ancestry: str | None = None
    h2_confidence: str | None = None
    unavailable_reason: str | None = None


class TraitEvidence(BaseModel):
    """Normalized trait snapshot nested on a model row. Join keys only plus risk."""

    trait_id: str
    label: str | None = None
    relationship_source: str | None = None
    trait_reported: str | None = None
    prs_actionability_status: str = "not_assessed"
    condition_actionability_status: str = "not_assessed"
    context_resolution_status: str = "not_assessed"
    context_classes: list[str] = Field(default_factory=list)
    population_risks: list[TraitPopulationRisk] = Field(default_factory=list)


class PaperEvidence(BaseModel):
    """Citation identifiers only. Full paper rows stay in papers.parquet."""

    paper_id: str
    relationship_type: str | None = None
    pmid: str | None = None
    doi: str | None = None
    pgp_id: str | None = None


class ModelAnalysisRow(BaseModel):
    """One runtime row plus analysis hydration. Grain: the runtime primary key."""

    sample_id: str
    pgs_id: str
    scoring_build: str
    score_profile_id: str
    scoring_fingerprint: str
    status: str
    error: str | None = None
    score: float | None = None
    variants_matched: int | None = None
    variants_total: int | None = None
    match_rate: float | None = None
    weight_mass_coverage: float | None = None
    sample_genotype_sha256: str
    source_revision: str
    analysis_eligible: bool
    exclusion_reasons: list[str] = Field(default_factory=list)
    publication_allowed: bool = True
    selected_superpopulation: str | None = None
    selected_superpopulation_confidence: float | None = None
    closest_cohort: str | None = None
    closest_cohort_confidence: float | None = None
    closest_cohort_label: str | None = None
    catalog_name: str | None = None
    trait_reported: str | None = None
    quality_label: str | None = None
    quality_key: str | None = None
    is_harmonized: bool | None = None
    auroc: float | None = None
    or_per_sd: float | None = None
    population_metrics: list[PopulationMetric] = Field(default_factory=list)
    traits: list[TraitEvidence] = Field(default_factory=list)
    papers: list[PaperEvidence] = Field(default_factory=list)


class TraitSummaryRow(BaseModel):
    """One ancestry-selected usable-scope trait summary."""

    sample_id: str
    trait_id: str
    trait_label: str | None = None
    score_profile_id: str
    percentile_population: str
    model_scope: str = "usable"
    percentile_source: str = "selected"
    n_total: int = 0
    n_eligible: int = 0
    n_usable: int = 0
    n_percentile_available: int = 0
    n_quarantined: int = 0
    n_failed: int = 0
    n_excluded: int = 0
    median_pct: float | None = None
    mean_pct: float | None = None
    std_pct: float | None = None
    min_pct: float | None = None
    max_pct: float | None = None
    spread: float | None = None
    outliers: list[str] = Field(default_factory=list)
    most_reliable_pgs_id: str | None = None
    most_reliable_pct: float | None = None
    absolute_risk: str | None = None
    population_average: str | None = None
    risk_vs_average: str | None = None
    n_risk_models: int = 0
    heritability_text: str | None = None
    heritability_detail: str | None = None
    prs_actionability_status: str = "not_assessed"
    condition_actionability_status: str = "not_assessed"
    context_resolution_status: str = "not_assessed"
    context_classes: list[str] = Field(default_factory=list)
    paired_profile_id: str | None = None
    paired_n_usable: int | None = None
    paired_median_pct: float | None = None
    delta_median_pct: float | None = None
    caveats: list[str] = Field(default_factory=list)


class IntegrationManifest(BaseModel):
    """Final dataset manifest. Does not embed its own SHA256."""

    schema_version: str = INTEGRATION_MANIFEST_SCHEMA_VERSION
    published_at: str
    parent_revision: str
    final_revision: str | None = None
    sample_scores_repo: str
    catalog_repo: str
    percentiles_repo: str
    sample_scores_revision: str
    catalog_revision: str
    percentiles_revision: str
    scoring_set_fingerprint: str | None = None
    sample_set_fingerprint: str | None = None
    reference_universe_fingerprint: str | None = None
    n_samples: int
    n_pgs_ids: int
    n_profiles: int
    n_runtime_rows: int
    n_analysis_eligible: int
    n_trait_summaries: int
    pgs_without_distribution: list[str] = Field(default_factory=list)
    omitted_catalog_tables: list[str] = Field(default_factory=list)
    sources: list[StagingIndexRow] = Field(default_factory=list)
    outputs: dict[str, dict[str, object]] = Field(default_factory=dict)
    blocking_check_verdict: str = "passed"
    notes: list[str] = Field(default_factory=list)


__all__ = [
    "EXCLUSION_ANCESTRY_MISSING",
    "EXCLUSION_ANCESTRY_UNKNOWN",
    "EXCLUSION_AUDIT_ERROR",
    "EXCLUSION_DIST_UNAVAILABLE",
    "EXCLUSION_FAILED",
    "EXCLUSION_FINGERPRINT",
    "EXCLUSION_NOT_PUBLISHED",
    "EXCLUSION_QUARANTINED",
    "EXCLUSION_RUNTIME_INVARIANT",
    "FINE_COHORT_LABELS",
    "INTEGRATION_MANIFEST_SCHEMA_VERSION",
    "MODEL_ANALYSIS_PRIMARY_KEY",
    "MODEL_ANALYSIS_SCHEMA_VERSION",
    "TRAIT_SUMMARY_PRIMARY_KEY",
    "TRAIT_SUMMARY_SCHEMA_VERSION",
    "IntegrationManifest",
    "ModelAnalysisRow",
    "PaperEvidence",
    "PopulationMetric",
    "SampleAncestryRecord",
    "TraitEvidence",
    "TraitPopulationRisk",
    "TraitSummaryRow",
]
