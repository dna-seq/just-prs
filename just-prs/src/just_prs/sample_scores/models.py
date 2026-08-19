"""Frozen schemas for precomputed public-genome PRS lookup and publication."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from just_prs.models import PRSResult
from just_prs.normalize import VcfFilterConfig

HASH_SCHEMA_VERSION = 1
SCORE_ALGORITHM_VERSION = "1"
NORMALIZATION_PROFILE_ID = "public-wgs-pass-v1"
DEFAULT_SAMPLE_SCORES_REPO = "just-dna-seq/prs-sample-scores"
ANCESTRY_INFERENCE_VERSION = "1000g-knn-v1"
PRIVATE_INGEST_ALIASES: dict[str, str] = {"oksana": "o-mom"}
EXPECTED_PUBLIC_SAMPLE_ANCESTRY: dict[str, tuple[str, str]] = {
    "anton": ("EUR", "CEU"),
    "livia": ("EUR", "IBS"),
    "o-mom": ("EUR", "CEU"),
    "o-dad": ("EUR", "CEU"),
    "o-son1": ("EUR", "CEU"),
    "o-son2": ("EUR", "CEU"),
    "o-daughter": ("EUR", "CEU"),
}

UNRESTORED_PROFILE_ID = "grch38-wgs-pass-unrestored-v1"
RESTORED_PROFILE_ID = "grch38-wgs-pass-restored-v1"


class PrecomputedPolicy(StrEnum):
    """Whether official-PGS compute may return a published result."""

    AUTO = "auto"
    OFF = "off"
    REQUIRE = "require"


class ComputationSource(StrEnum):
    """How a ``PRSResult`` was obtained. Not the UI Native/Harmonized ``score_source``."""

    PRECOMPUTED = "precomputed"
    COMPUTED = "computed"


class NormalizationProfile(BaseModel):
    """Frozen VCF normalization used for published public-WGS scores."""

    profile_id: str
    pass_filters: list[str]
    min_depth: int | None = None
    min_qual: float | None = None
    genotype_input_mode: str = "variant_only"
    maf_fill: bool = False

    def to_filter_config(self) -> VcfFilterConfig:
        return VcfFilterConfig(
            pass_filters=list(self.pass_filters),
            min_depth=self.min_depth,
            min_qual=self.min_qual,
        )


PUBLIC_WGS_PASS_V1 = NormalizationProfile(
    profile_id=NORMALIZATION_PROFILE_ID,
    pass_filters=["PASS", "."],
    min_depth=None,
    min_qual=None,
    genotype_input_mode="variant_only",
    maf_fill=False,
)


class ScoreProfile(BaseModel):
    """Published scoring profile. A runtime hit requires this id plus fingerprints."""

    score_profile_id: str
    genome_build: str
    normalization_profile_id: str = NORMALIZATION_PROFILE_ID
    reference_restoration: bool
    genotype_input_mode: str = "variant_only"
    maf_fill: bool = False
    score_algorithm_version: str = SCORE_ALGORITHM_VERSION


SCORE_PROFILES: dict[str, ScoreProfile] = {
    UNRESTORED_PROFILE_ID: ScoreProfile(
        score_profile_id=UNRESTORED_PROFILE_ID,
        genome_build="GRCh38",
        reference_restoration=False,
    ),
    RESTORED_PROFILE_ID: ScoreProfile(
        score_profile_id=RESTORED_PROFILE_ID,
        genome_build="GRCh38",
        reference_restoration=True,
    ),
}


def published_aliases(aliases: list[str]) -> list[str]:
    """Drop private ingest aliases that must never appear on HuggingFace."""
    private = {key.casefold() for key in PRIVATE_INGEST_ALIASES}
    return [alias for alias in aliases if alias.casefold() not in private]


class SampleAncestryRecord(BaseModel):
    """Runtime-owned 1000G ancestry call for one published public genome.

    Fine codes (CEU/IBS) are nearest 1000G cohorts, not nationality or ethnicity.
    """

    sample_id: str
    genotype_sha256_v1: str
    superpopulation: str
    confidence: float
    fine_population: str | None = None
    fine_confidence: float | None = None
    panel: str = "1000g"
    genome_build: str = "GRCh38"
    ancestry_model_revision: str | None = None
    ancestry_model_sha256: str | None = None
    inference_version: str = ANCESTRY_INFERENCE_VERSION
    n_variants_used: int = 0
    n_variants_model: int = 0
    coverage: float = 0.0
    inferred_at: str | None = None


class SampleRecord(BaseModel):
    """One registered public genome. Aliases select a candidate; hashes prove identity."""

    sample_id: str
    aliases: list[str] = Field(default_factory=list)
    display_name: str
    family_id: str | None = None
    relationship_role: str | None = None
    license: str
    publication_allowed: bool
    consent_basis: str
    source_url: str | None = None
    genome_build: str = "GRCh38"
    n_variants: int | None = None
    source_sha256: str
    genotype_sha256_v1: str
    normalization_profile_id: str = NORMALIZATION_PROFILE_ID
    hash_schema_version: int = HASH_SCHEMA_VERSION


class RuntimeResultRow(BaseModel):
    """One published sample×PGS×profile outcome. Raw score/coverage only."""

    sample_id: str
    pgs_id: str
    scoring_build: str
    score_profile_id: str
    scoring_fingerprint: str
    status: str = "ok"
    error: str | None = None
    score: float | None = None
    variants_matched: int | None = None
    variants_total: int | None = None
    match_rate: float | None = None
    variants_observed: int = 0
    variants_assumed_hom_ref: int = 0
    variants_unscorable_absent: int = 0
    variants_no_call: int = 0
    variants_maf_filled: int = 0
    variants_ref_resolved_panel: int = 0
    variants_ref_resolved_fasta: int = 0
    weight_mass_matched: float | None = None
    weight_mass_total: float | None = None
    weight_mass_coverage: float | None = None
    genotype_input_mode: str = "variant_only"
    detected_genome_build: str | None = None
    build_mismatch: bool = False
    trait_reported: str | None = None
    has_allele_frequencies: bool = False
    theoretical_mean: float | None = None
    theoretical_std: float | None = None
    score_algorithm_version: str = SCORE_ALGORITHM_VERSION
    reference_universe_fingerprint: str | None = None
    sample_genotype_sha256: str
    just_prs_version: str | None = None
    computed_at: str | None = None

    def to_prs_result(
        self,
        *,
        repo_id: str | None = None,
        revision: str | None = None,
    ) -> PRSResult:
        if self.status != "ok" or self.score is None:
            raise ValueError(
                f"Cannot hydrate PRSResult for {self.pgs_id}: status={self.status!r}"
            )
        return PRSResult(
            pgs_id=self.pgs_id,
            score=self.score,
            variants_matched=int(self.variants_matched or 0),
            variants_total=int(self.variants_total or 0),
            match_rate=float(self.match_rate or 0.0),
            variants_observed=self.variants_observed,
            variants_assumed_hom_ref=self.variants_assumed_hom_ref,
            variants_unscorable_absent=self.variants_unscorable_absent,
            variants_no_call=self.variants_no_call,
            variants_maf_filled=self.variants_maf_filled,
            variants_ref_resolved_panel=self.variants_ref_resolved_panel,
            variants_ref_resolved_fasta=self.variants_ref_resolved_fasta,
            weight_mass_matched=self.weight_mass_matched,
            weight_mass_total=self.weight_mass_total,
            weight_mass_coverage=self.weight_mass_coverage,
            genotype_input_mode=self.genotype_input_mode,
            detected_genome_build=self.detected_genome_build,
            build_mismatch=self.build_mismatch,
            trait_reported=self.trait_reported,
            has_allele_frequencies=self.has_allele_frequencies,
            theoretical_mean=self.theoretical_mean,
            theoretical_std=self.theoretical_std,
            computation_source=ComputationSource.PRECOMPUTED.value,
            precomputed_repo=repo_id,
            precomputed_revision=revision,
            score_profile_id=self.score_profile_id,
        )


def match_score_profile(
    *,
    genome_build: str,
    reference_restoration: bool,
    genotype_input_mode: str = "auto",
    maf_fill: bool = False,
) -> ScoreProfile | None:
    """Return the published profile that matches this official-PGS request, or None."""
    mode = genotype_input_mode.strip().lower()
    if mode in {"auto", "variant_only"}:
        mode = "variant_only"
    else:
        return None
    if maf_fill:
        return None
    if genome_build != "GRCh38":
        return None
    for profile in SCORE_PROFILES.values():
        if (
            profile.genome_build == genome_build
            and profile.reference_restoration is reference_restoration
            and profile.genotype_input_mode == mode
            and profile.maf_fill is maf_fill
        ):
            return profile
    return None
