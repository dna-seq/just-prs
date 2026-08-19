"""Catalog-level evidence tables for the public sample-score dataset.

These tables join a catalog PGS ID or a canonical trait. They do not contain
sample scores, percentiles, or coverage counters.
"""

from just_prs.sample_scores.evidence.build import (
    EVIDENCE_OWNED_FILES,
    EVIDENCE_TABLE_FILES,
    EvidenceBuildResult,
    build_sample_score_evidence,
    publish_sample_score_evidence,
)
from just_prs.sample_scores.evidence.contexts import (
    drug_response_pgs_ids,
    is_drug_response_trait,
    looks_like_drug_response,
)
from just_prs.sample_scores.evidence.checks import validate_evidence_tables
from just_prs.sample_scores.evidence.models import (
    ActionabilityRecord,
    ActionabilityStatus,
    ContextClass,
    GuidelineRecord,
    GuidelineTraitLink,
    PaperRecord,
    RecordSearchTerm,
    ScorePaperLink,
    ScoreTraitLink,
    SearchTrack,
    TraitContextRecord,
    TraitRecord,
)

__all__ = [
    "EVIDENCE_OWNED_FILES",
    "EVIDENCE_TABLE_FILES",
    "drug_response_pgs_ids",
    "is_drug_response_trait",
    "looks_like_drug_response",
    "ActionabilityRecord",
    "ActionabilityStatus",
    "ContextClass",
    "EvidenceBuildResult",
    "GuidelineRecord",
    "GuidelineTraitLink",
    "PaperRecord",
    "RecordSearchTerm",
    "ScorePaperLink",
    "ScoreTraitLink",
    "SearchTrack",
    "TraitContextRecord",
    "TraitRecord",
    "build_sample_score_evidence",
    "publish_sample_score_evidence",
    "validate_evidence_tables",
]
