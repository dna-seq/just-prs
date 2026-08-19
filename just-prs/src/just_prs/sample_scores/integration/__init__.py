"""Offline integration of repaired runtime and evidence into analysis tables."""

from just_prs.sample_scores.integration.build import build_model_analysis
from just_prs.sample_scores.integration.checks import (
    IntegrationCheckError,
    IntegrationCheckReport,
    validate_integration_outputs,
)
from just_prs.sample_scores.integration.models import (
    IntegrationManifest,
    ModelAnalysisRow,
    PopulationMetric,
    TraitEvidence,
    TraitPopulationRisk,
    TraitSummaryRow,
)
from just_prs.sample_scores.integration.publish import (
    IntegrationBuildResult,
    build_sample_score_integration,
    publish_sample_score_integration,
)
from just_prs.sample_scores.integration.staging import stage_pinned_sources
from just_prs.sample_scores.integration.summaries import build_trait_summaries

__all__ = [
    "IntegrationBuildResult",
    "IntegrationCheckError",
    "IntegrationCheckReport",
    "IntegrationManifest",
    "ModelAnalysisRow",
    "PopulationMetric",
    "TraitEvidence",
    "TraitPopulationRisk",
    "TraitSummaryRow",
    "build_model_analysis",
    "build_sample_score_integration",
    "build_trait_summaries",
    "publish_sample_score_integration",
    "stage_pinned_sources",
    "validate_integration_outputs",
]
