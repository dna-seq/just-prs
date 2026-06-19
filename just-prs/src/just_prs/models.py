"""Pydantic v2 models for PGS Catalog data and PRS computation results."""

from dataclasses import dataclass

from pydantic import BaseModel, Field


class PublicationInfo(BaseModel):
    """Publication metadata from PGS Catalog."""

    id: str = Field(description="PGS Catalog Publication ID (PGP)")
    title: str | None = Field(default=None, description="Publication title")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    PMID: int | None = Field(default=None, description="PubMed ID")
    journal: str | None = Field(default=None, description="Journal name")
    firstauthor: str | None = Field(default=None, description="First author")
    date_publication: str | None = Field(default=None, description="Publication date")


class HarmonizedFileInfo(BaseModel):
    """URLs for harmonized scoring files by genome build."""

    positions: str | None = Field(default=None, description="URL for position-harmonized file")


class ScoreInfo(BaseModel):
    """Polygenic Score metadata from PGS Catalog REST API."""

    id: str = Field(description="PGS Catalog Score ID (e.g. PGS000001)")
    name: str | None = Field(default=None, description="Score name")
    trait_reported: str | None = Field(default=None, description="Reported trait")
    trait_additional: str | None = Field(default=None, description="Additional trait info")
    variants_number: int | None = Field(default=None, description="Number of variants in score")
    variants_genomebuild: str | None = Field(default=None, description="Original genome build")
    weight_type: str | None = Field(default=None, description="Weight type (beta, OR, HR, etc.)")
    ftp_scoring_file: str | None = Field(default=None, description="URL for original scoring file")
    ftp_harmonized_scoring_files: dict[str, HarmonizedFileInfo] | None = Field(
        default=None, description="Harmonized scoring file URLs by genome build"
    )
    publication: PublicationInfo | None = Field(default=None, description="Source publication")
    matches_publication: bool | None = Field(default=None)
    license: str | None = Field(default=None, description="License text")

    def get_download_url(self, genome_build: str = "GRCh38") -> str | None:
        """Get the harmonized scoring file download URL for a given genome build."""
        if self.ftp_harmonized_scoring_files is None:
            return self.ftp_scoring_file
        build_info = self.ftp_harmonized_scoring_files.get(genome_build)
        if build_info is not None and build_info.positions is not None:
            return build_info.positions
        return self.ftp_scoring_file


class TraitInfo(BaseModel):
    """Trait metadata from PGS Catalog REST API."""

    id: str = Field(description="EFO trait ID (e.g. EFO_0001645)")
    label: str | None = Field(default=None, description="Trait label")
    description: str | None = Field(default=None, description="Trait description")
    url: str | None = Field(default=None, description="EFO URL")
    trait_categories: list[str] = Field(default_factory=list, description="Trait categories")
    trait_synonyms: list[str] = Field(default_factory=list, description="Trait synonyms")
    associated_pgs_ids: list[str] = Field(
        default_factory=list, description="Associated PGS IDs"
    )
    child_associated_pgs_ids: list[str] = Field(
        default_factory=list, description="Child trait associated PGS IDs"
    )


class EffectSizeInfo(BaseModel):
    """A single effect size or classification accuracy metric."""

    name_short: str = Field(description="Short name (e.g. OR, HR, AUROC, C-index)")
    name_long: str | None = Field(default=None, description="Full metric name")
    estimate: float = Field(description="Point estimate")
    ci_lower: float | None = Field(default=None, description="Lower bound of confidence interval")
    ci_upper: float | None = Field(default=None, description="Upper bound of confidence interval")
    se: float | None = Field(default=None, description="Standard error")


class PerformanceInfo(BaseModel):
    """Performance metrics for a PGS score from PGS Catalog evaluation studies."""

    ppm_id: str = Field(description="Performance Metric ID (PPM)")
    effect_sizes: list[EffectSizeInfo] = Field(default_factory=list)
    class_acc: list[EffectSizeInfo] = Field(default_factory=list)
    sample_number: int | None = Field(default=None, description="Evaluation sample size")
    ancestry_broad: str | None = Field(default=None, description="Broad ancestry of evaluation cohort")
    phenotyping_reported: str | None = Field(default=None, description="Phenotype evaluated")
    covariates: str | None = Field(default=None, description="Covariates used in evaluation")


class ReferenceDistribution(BaseModel):
    """Per-superpopulation PRS distribution statistics from the 1000G reference panel."""

    pgs_id: str = Field(description="PGS Catalog Score ID")
    superpopulation: str = Field(description="1000G superpopulation code (AFR, AMR, EAS, EUR, SAS)")
    mean: float = Field(description="Mean PRS in this ancestry group")
    std: float = Field(description="Standard deviation of PRS in this ancestry group")
    n: int = Field(description="Number of reference individuals in this group")
    median: float | None = Field(default=None, description="Median PRS")
    p5: float | None = Field(default=None, description="5th percentile PRS")
    p25: float | None = Field(default=None, description="25th percentile PRS")
    p75: float | None = Field(default=None, description="75th percentile PRS")
    p95: float | None = Field(default=None, description="95th percentile PRS")


class AbsoluteRisk(BaseModel):
    """Absolute disease risk estimate derived from PRS z-score and prevalence."""

    absolute_risk: float = Field(description="Estimated absolute risk (e.g. 0.18 = 18% lifetime risk)")
    population_prevalence: float = Field(description="Population baseline prevalence (e.g. 0.11 = 11%)")
    risk_ratio: float = Field(description="Risk ratio vs population average (e.g. 1.64x)")
    method: str = Field(description="Estimation method: 'or_per_sd' or 'auc_bivariate'")
    confidence: str = Field(description="Data quality confidence: 'high', 'moderate', or 'low'")
    prevalence_source: str = Field(description="Where the prevalence data came from")
    prevalence_type: str = Field(description="Type of prevalence: 'lifetime', 'point', or 'cohort'")
    effect_size_citation: str | None = Field(
        default=None,
        description="Paper citation for the OR/AUROC used (e.g. 'Smith et al. 2023, JAMA (PMID: 12345678)')",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Warnings about estimation quality (e.g. 'cohort prevalence used, not population')",
    )


class AbsoluteRiskEstimate(BaseModel):
    """A single absolute risk estimate from one specific method and data source."""

    absolute_risk: float = Field(description="Estimated disease probability (0-1), e.g. 0.132 = 13.2%")
    population_prevalence: float = Field(description="Baseline prevalence used, e.g. 0.11 = 11%")
    risk_ratio: float = Field(description="User's risk relative to population average, e.g. 1.20x")
    method: str = Field(description="Estimation method: 'or_per_sd', 'auc_bivariate', 'h2_liability'")
    method_label: str = Field(description="Human-readable label: 'OR per SD', 'AUROC model', 'h² EUR', etc.")
    ancestry: str | None = Field(default=None, description="Ancestry of the data used (EUR, AFR, etc.)")
    h2_value: float | None = Field(default=None, description="h² value used (for h2_liability method)")
    h2_source: str | None = Field(default=None, description="Source of h² data: 'pan_ukbb', 'gwas_atlas'")
    h2_source_detail: str | None = Field(default=None, description="Detailed h² source description")
    confidence: str = Field(description="Data quality: 'high', 'moderate', or 'low'")
    prevalence_source: str = Field(description="Where the prevalence data came from")
    prevalence_type: str = Field(description="Type of prevalence: 'lifetime', 'point', or 'cohort'")
    effect_size_citation: str | None = Field(default=None, description="Paper citation for the effect size")
    caveats: list[str] = Field(default_factory=list, description="Warnings about estimation quality")


class AbsoluteRiskBundle(BaseModel):
    """All available risk estimates for a single PGS score.

    Multiple methods may produce different risk numbers for the same score.
    This bundle presents all of them for transparency, with a recommended
    best estimate and an agreement indicator.
    """

    estimates: list[AbsoluteRiskEstimate] = Field(
        default_factory=list, description="All computed risk estimates"
    )
    best_estimate: AbsoluteRiskEstimate | None = Field(
        default=None, description="System's recommended best estimate"
    )
    agreement: str = Field(
        default="unknown",
        description="How well estimates agree: 'high' (<2pp spread), 'moderate' (2-5pp), 'low' (>5pp)",
    )
    spread_pp: float | None = Field(
        default=None,
        description="Spread between min and max risk estimates in percentage points",
    )
    heritability_status: str = Field(
        default="not_checked",
        description=(
            "h² lookup status: 'used', 'no_mapped_h2', 'table_unavailable', "
            "or 'not_checked'"
        ),
    )
    heritability_detail: str = Field(
        default="",
        description="Plain-language detail about h² usage or why it was unavailable",
    )
    heritability_trait_ids: list[str] = Field(
        default_factory=list,
        description="Trait IDs considered during h² lookup after ontology expansion",
    )


class PRSResult(BaseModel):
    """Result of a polygenic risk score computation."""

    pgs_id: str = Field(description="PGS Catalog Score ID")
    score: float = Field(description="Computed polygenic risk score")
    variants_matched: int = Field(description="Number of scoring variants matched in VCF")
    variants_total: int = Field(description="Total number of variants in scoring file")
    match_rate: float = Field(description="Fraction of scoring variants matched (0-1)")
    variants_observed: int = Field(
        default=0,
        description="Scoring loci present in the genotype input with a genotype row",
    )
    variants_assumed_hom_ref: int = Field(
        default=0,
        description="Absent loci treated as homozygous-reference under variant-only VCF semantics",
    )
    variants_unscorable_absent: int = Field(
        default=0,
        description="Absent loci that could not be scored because the reference allele was unknown",
    )
    variants_no_call: int = Field(
        default=0,
        description="Scoring loci present in the genotype input but carrying a missing/no-call GT",
    )
    variants_maf_filled: int = Field(
        default=0,
        description="Absent loci filled with population MAF dosage (2 * allelefrequency_effect) instead of being unscorable",
    )
    genotype_input_mode: str = Field(
        default="plink_present_only",
        description="How absent genotype loci were interpreted during scoring",
    )
    trait_reported: str | None = Field(default=None, description="Reported trait for the score")
    performance: PerformanceInfo | None = Field(
        default=None, description="Best available performance metric from PGS Catalog"
    )
    has_allele_frequencies: bool = Field(
        default=False,
        description="Whether the scoring file contained allelefrequency_effect data",
    )
    theoretical_mean: float | None = Field(
        default=None,
        description="Theoretical population mean PRS computed from allele frequencies: sum(w_i * 2 * p_i)",
    )
    theoretical_std: float | None = Field(
        default=None,
        description="Theoretical population SD of PRS: sqrt(sum(w_i^2 * 2 * p_i * (1-p_i)))",
    )
    percentile: float | None = Field(
        default=None,
        description="Estimated population percentile (0-100) from theoretical distribution",
    )
    ancestry: str | None = Field(
        default=None,
        description="Ancestry superpopulation used for percentile (AFR, AMR, EAS, EUR, SAS)",
    )
    percentile_method: str | None = Field(
        default=None,
        description="Method used to compute percentile: 'reference_panel', 'theoretical', or 'auroc_approx'",
    )
    absolute_risk: AbsoluteRisk | None = Field(
        default=None,
        description="Absolute disease risk estimate based on PRS z-score and prevalence data",
    )


class EnrichedPRSResult(BaseModel):
    """Fully enriched PRS result: raw computation + quality, percentile, risk, reference data.

    Produced by ``enrich_prs_result()`` from a ``PRSResult`` plus catalog context.
    Every field the UI needs is present here so consumers (UI, CLI, batch scripts)
    share a single enrichment path.
    """

    pgs_id: str
    trait: str = ""
    trait_efo_id: str = Field("", description="EFO/MONDO/OBA/HP trait ontology ID(s) from PGS Catalog")
    score: float = 0.0
    variants_matched: int = 0
    variants_total: int = 0
    match_rate: float = Field(0.0, description="Match rate as percentage 0-100")
    variants_observed: int = 0
    variants_assumed_hom_ref: int = 0
    variants_unscorable_absent: int = 0
    variants_no_call: int = 0
    genotype_input_mode: str = ""
    has_allele_frequencies: bool = False
    genome_build: str = ""
    is_harmonized: bool = Field(
        False,
        description="True if the score's original build differs from the computation build (coordinates are lifted over)",
    )

    # Synthetic quality ranking (0-100)
    synthetic_quality: float = Field(0.0, description="Synthetic quality score for model ranking (0-100)")
    synthetic_quality_label: str = Field("", description="Synthetic quality label: High/Normal/Moderate/Low")
    synthetic_quality_color: str = Field("", description="Semantic color for synthetic quality label")
    quality_tier: str = Field("", description="Quality tier: T1a_auroc, T1b_beta, T2_or_hr, T3_none")
    quality_tier_metric: str = Field("", description="Primary metric for this tier, e.g. 'AUROC=0.72', 'Beta=0.24'")

    # Resolved percentile (reference > theoretical > AUROC fallback)
    percentile: float | None = None
    percentile_method: str = ""

    # Quality assessment
    match_color: str = ""
    quality_label: str = ""
    quality_color: str = ""
    summary: str = ""

    # Performance metrics from best evaluation
    effect_size: str = ""
    classification: str = ""
    auroc: float | None = None
    ancestry: str = ""
    n_individuals: int = 0

    # Risk level derived from percentile
    risk_level: str = ""
    risk_level_color: str = ""
    risk_hint: str = ""

    # Per-superpopulation reference percentiles
    selected_ancestry: str = "EUR"
    all_population_percentiles: str = ""
    pct_AFR: float | None = None
    pct_AMR: float | None = None
    pct_EAS: float | None = None
    pct_EUR: float | None = None
    pct_SAS: float | None = None

    # Reference panel status
    reference_status: str = ""
    reference_source: str = ""
    reference_source_code: str = ""
    reference_audit_status: str = ""
    reference_audit_warning_count: int = 0
    reference_audit_error_count: int = 0
    reference_audit_issues: str = ""

    # Absolute risk
    absolute_risk_text: str = ""
    absolute_risk_percent: float | None = None
    population_average_percent: float | None = None
    risk_ratio_value: float | None = None
    absolute_risk_method: str = ""
    absolute_risk_detail: str = ""

    # Heritability
    heritability: str = "N/A"
    heritability_detail: str = ""
    heritability_metrics: list[dict[str, str]] = Field(default_factory=list)

    # Multi-method risk agreement
    risk_agreement: str = ""
    risk_estimates_by_method: dict[str, str] = Field(default_factory=dict)
    risk_estimate_methods: list[str] = Field(default_factory=list)


class ArrayPRSResult(PRSResult):
    """PRS result from consumer genotyping array data with coverage and proxy metadata."""

    chip: str = Field(default="", description="Chip identifier (e.g. 'gsa_v3')")
    coverage_ratio: float = Field(
        default=0.0,
        description="Fraction of scoring variants directly typed on the chip (0-1)",
    )
    variants_proxied: int = Field(
        default=0,
        description="Scoring variants recovered via LD-proxy substitution",
    )
    effective_coverage: float = Field(
        default=0.0,
        description="(typed + proxied + maf_filled) / total — effective variant coverage after all recovery methods",
    )
    coverage_tier: str = Field(
        default="unreliable",
        description="Coverage quality tier: 'reliable' (>=90%), 'usable' (>=60%), 'low' (>=40%), 'unreliable' (<40%)",
    )
    proxy_r2_mean: float | None = Field(
        default=None,
        description="Mean LD r² of used proxy variants",
    )
    score_uncorrected: float = Field(
        default=0.0,
        description="Raw PRS score before LD-proxy weight adjustment",
    )


@dataclass
class ChipGeneration:
    """Detected consumer genotyping chip generation from raw array data."""

    chip_id: str
    platform: str
    generation_label: str
    ld_proxy_available: bool
    marker_count: int


# Coverage tier thresholds for array scoring
COVERAGE_TIERS: dict[str, float] = {
    "reliable": 0.90,
    "usable": 0.60,
    "low": 0.40,
}


def classify_coverage_tier(effective_coverage: float) -> str:
    """Classify effective coverage into a quality tier."""
    for tier, threshold in COVERAGE_TIERS.items():
        if effective_coverage >= threshold:
            return tier
    return "unreliable"


class PRSBatchOutcome(BaseModel):
    """Per-ID outcome from a batch PRS computation."""

    pgs_id: str = Field(description="PGS Catalog Score ID")
    status: str = Field(description="'ok', 'failed', or 'cache_repaired'")
    error: str | None = Field(default=None, description="Error message if status is 'failed'")
    attempts: int = Field(default=1, description="Number of attempts (2 if cache was repaired)")


class PRSBatchResult(BaseModel):
    """Result of a batch PRS computation with per-ID error tracking."""

    results: list["PRSResult"] = Field(default_factory=list, description="Successfully computed results")
    outcomes: list[PRSBatchOutcome] = Field(default_factory=list, description="Per-ID status/error tracking")
    n_total: int = Field(default=0, description="Total IDs attempted")
    n_ok: int = Field(default=0, description="Successfully computed")
    n_failed: int = Field(default=0, description="Failed IDs")
    failed_ids: list[str] = Field(default_factory=list, description="PGS IDs that failed")
