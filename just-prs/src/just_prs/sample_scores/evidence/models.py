"""Frozen Pydantic schemas for sample-score evidence tables.

Join keys only: ``pgs_id``, ``trait_id``, ``paper_id``, ``guideline_id``.
These tables never store sample scores, percentiles, or coverage counters.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ActionabilityStatus(StrEnum):
    """Allowed values for the three actionability columns."""

    SUPPORTED = "supported"
    AGAINST = "against"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_ASSESSED = "not_assessed"
    NOT_APPLICABLE = "not_applicable"


class PaperRelationship(StrEnum):
    DEVELOPMENT = "development"
    EVALUATION = "evaluation"
    VALIDATION = "validation"
    GUIDELINE_CONTEXT = "guideline_context"


class AbstractStatus(StrEnum):
    INCLUDED = "included"
    LINK_ONLY = "link_only"
    UNAVAILABLE = "unavailable"


class ContextClass(StrEnum):
    """Extra-clinical usefulness. Never written into actionability statuses."""

    CURIOSITY = "curiosity"
    SPORTS = "sports"
    LIFESTYLE = "lifestyle"
    ENVIRONMENT = "environment"
    AGING = "aging"
    PHARMACOLOGY = "pharmacology"
    BEHAVIORAL = "behavioral"
    CHECKUP_HINT = "checkup_hint"
    CAREER = "career"
    APPEARANCE = "appearance"
    LIFE_OPTIMIZATION = "life_optimization"


class SearchTrack(StrEnum):
    CLINICAL = "clinical"
    EXTRA_CLINICAL = "extra_clinical"


class TraitRecord(BaseModel):
    """One canonical ontology concept. Grain: ``trait_id``."""

    trait_id: str
    label: str
    definition: str | None = None
    synonyms_json: str = "[]"
    category: str | None = None
    ontology_prefix: str | None = None
    canonical_url: str | None = None
    aliases_json: str = "[]"
    icd10_codes_json: str = "[]"
    mapping_source: str
    mapping_confidence: str = "catalog"
    mapping_status: str = "resolved"
    retrieved_at: str | None = None
    source_revision: str | None = None


class ScoreTraitLink(BaseModel):
    """Many-to-many catalog score ↔ trait. Grain: ``(pgs_id, trait_id)``."""

    pgs_id: str
    trait_id: str
    relationship_source: str
    trait_reported: str | None = None


class PaperRecord(BaseModel):
    """One paper. Grain: ``paper_id`` (PMID, then DOI, then PGP)."""

    paper_id: str
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    pgp_id: str | None = None
    title: str | None = None
    authors: str | None = None
    first_author: str | None = None
    journal: str | None = None
    date_publication: str | None = None
    citation_text: str | None = None
    pubmed_url: str | None = None
    europepmc_url: str | None = None
    doi_url: str | None = None
    is_open_access: bool | None = None
    content_license: str | None = None
    abstract_text: str | None = None
    abstract_status: str = AbstractStatus.LINK_ONLY.value
    resolution_status: str = "catalog"


class ScorePaperLink(BaseModel):
    """Many-to-many score ↔ paper. Grain: ``(pgs_id, paper_id, relationship_type)``."""

    pgs_id: str
    paper_id: str
    relationship_type: str
    pgp_id: str | None = None
    ppm_id: str | None = None


class GuidelineRecord(BaseModel):
    """One publicly obtainable recommendation. Grain: ``guideline_id``."""

    guideline_id: str
    organization: str
    jurisdiction: str | None = None
    title: str
    url: str
    version: str | None = None
    publication_date: str | None = None
    update_date: str | None = None
    withdrawal_date: str | None = None
    recommendation_direction: str
    grade: str | None = None
    action_type: str | None = None
    target_condition: str | None = None
    target_population: str | None = None
    eligibility_criteria: str | None = None
    required_context: str | None = None
    recommendation_text: str | None = None
    prs_specific: bool = False
    source_license: str | None = None
    obtainability: str
    retrieval_url: str
    retrieval_fingerprint: str
    retrieved_at: str
    is_current: bool = True


class GuidelineTraitLink(BaseModel):
    """Guideline ↔ trait. Grain: ``(guideline_id, trait_id)``."""

    guideline_id: str
    trait_id: str
    mapping_basis: str


class ActionabilityRecord(BaseModel):
    """Trait-level actionability. Grain: ``(trait_id, guideline_id)``.

    Missing evidence is ``not_assessed``, never “not actionable.”
    Every non-``not_assessed`` status requires a cited guideline.
    """

    trait_id: str
    guideline_id: str = ""
    prs_actionability_status: str = ActionabilityStatus.NOT_ASSESSED.value
    condition_actionability_status: str = ActionabilityStatus.NOT_ASSESSED.value
    context_resolution_status: str = ActionabilityStatus.NOT_ASSESSED.value
    supporting_guideline_ids_json: str = "[]"
    notes: str | None = None


class TraitContextRecord(BaseModel):
    """Extra-clinical usefulness. Grain: ``(trait_id, context_class)``."""

    trait_id: str
    context_class: str
    basis: str
    why_interesting: str
    high_percentile_means: str | None = None
    research_only: bool = False


class RecordSearchTerm(BaseModel):
    """Record / extra-clinical search term. Grain: ``(trait_id, track, term_kind, term)``."""

    trait_id: str
    track: str
    context_class: str | None = None
    term: str
    term_kind: str
    provenance: str


EVIDENCE_MODELS: dict[str, type[BaseModel]] = {
    "traits": TraitRecord,
    "score_trait_links": ScoreTraitLink,
    "papers": PaperRecord,
    "score_paper_links": ScorePaperLink,
    "guidelines": GuidelineRecord,
    "guideline_trait_links": GuidelineTraitLink,
    "actionability": ActionabilityRecord,
    "trait_contexts": TraitContextRecord,
    "record_search_terms": RecordSearchTerm,
}

EVIDENCE_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "traits": ("trait_id",),
    "score_trait_links": ("pgs_id", "trait_id"),
    "papers": ("paper_id",),
    "score_paper_links": ("pgs_id", "paper_id", "relationship_type"),
    "guidelines": ("guideline_id",),
    "guideline_trait_links": ("guideline_id", "trait_id"),
    "actionability": ("trait_id", "guideline_id"),
    "trait_contexts": ("trait_id", "context_class"),
    "record_search_terms": ("trait_id", "track", "term_kind", "term"),
}


def schema_field_docs(model: type[BaseModel]) -> list[tuple[str, str, str]]:
    """Return ``(name, type, description)`` for dataset-card generation."""
    rows: list[tuple[str, str, str]] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        description = field.description or ""
        rows.append((name, type_name, description))
    return rows
