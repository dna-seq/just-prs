"""Derive the three actionability statuses from cited public guidelines."""

from __future__ import annotations

import json
from collections import defaultdict

from just_prs.sample_scores.evidence.models import (
    ActionabilityRecord,
    ActionabilityStatus,
    GuidelineRecord,
    GuidelineTraitLink,
    TraitRecord,
)

_DIRECTION_TO_STATUS = {
    "for": ActionabilityStatus.SUPPORTED.value,
    "against": ActionabilityStatus.AGAINST.value,
    "insufficient": ActionabilityStatus.INSUFFICIENT_EVIDENCE.value,
}


def _status_from_direction(direction: str) -> str:
    return _DIRECTION_TO_STATUS.get(direction, ActionabilityStatus.NOT_ASSESSED.value)


def derive_actionability(
    traits: list[TraitRecord],
    guidelines: list[GuidelineRecord],
    guideline_links: list[GuidelineTraitLink],
) -> list[ActionabilityRecord]:
    """One row per ``(trait_id, guideline_id)``. Missing evidence is ``not_assessed``.

    PRS-specific status is filled only when ``guideline.prs_specific`` is true.
    ClinGen and other condition-only sources never set PRS actionability.
    """
    by_id = {item.guideline_id: item for item in guidelines}
    links_by_trait: dict[str, list[GuidelineTraitLink]] = defaultdict(list)
    for link in guideline_links:
        if link.guideline_id in by_id:
            links_by_trait[link.trait_id].append(link)

    rows: list[ActionabilityRecord] = []
    for trait in traits:
        linked = links_by_trait.get(trait.trait_id, [])
        if not linked:
            rows.append(
                ActionabilityRecord(
                    trait_id=trait.trait_id,
                    guideline_id="",
                    prs_actionability_status=ActionabilityStatus.NOT_ASSESSED.value,
                    condition_actionability_status=ActionabilityStatus.NOT_ASSESSED.value,
                    context_resolution_status=ActionabilityStatus.NOT_ASSESSED.value,
                    supporting_guideline_ids_json="[]",
                    notes="No publicly recorded guideline was mapped to this trait.",
                )
            )
            continue
        supporting_ids = [link.guideline_id for link in linked]
        condition_statuses = [
            _status_from_direction(by_id[link.guideline_id].recommendation_direction)
            for link in linked
        ]
        prs_statuses = [
            _status_from_direction(by_id[link.guideline_id].recommendation_direction)
            for link in linked
            if by_id[link.guideline_id].prs_specific
        ]
        context_statuses = [
            ActionabilityStatus.SUPPORTED.value
            if (by_id[link.guideline_id].eligibility_criteria or by_id[link.guideline_id].required_context)
            else ActionabilityStatus.NOT_ASSESSED.value
            for link in linked
        ]
        rolled_condition = _roll_up(condition_statuses)
        rolled_prs = _roll_up(prs_statuses) if prs_statuses else ActionabilityStatus.NOT_ASSESSED.value
        rolled_context = _roll_up(context_statuses)
        for link in linked:
            guideline = by_id[link.guideline_id]
            rows.append(
                ActionabilityRecord(
                    trait_id=trait.trait_id,
                    guideline_id=guideline.guideline_id,
                    prs_actionability_status=(
                        _status_from_direction(guideline.recommendation_direction)
                        if guideline.prs_specific
                        else ActionabilityStatus.NOT_ASSESSED.value
                    ),
                    condition_actionability_status=_status_from_direction(
                        guideline.recommendation_direction
                    ),
                    context_resolution_status=(
                        ActionabilityStatus.SUPPORTED.value
                        if (guideline.eligibility_criteria or guideline.required_context)
                        else ActionabilityStatus.NOT_ASSESSED.value
                    ),
                    supporting_guideline_ids_json=json.dumps(supporting_ids),
                    notes=(
                        f"Trait-level rollup condition={rolled_condition} "
                        f"prs={rolled_prs} context={rolled_context}. "
                        f"Source {guideline.organization} is "
                        f"{'PRS-specific' if guideline.prs_specific else 'condition-only'}."
                    ),
                )
            )
    return rows


def _roll_up(statuses: list[str]) -> str:
    if ActionabilityStatus.AGAINST.value in statuses:
        return ActionabilityStatus.AGAINST.value
    if ActionabilityStatus.SUPPORTED.value in statuses:
        return ActionabilityStatus.SUPPORTED.value
    if ActionabilityStatus.INSUFFICIENT_EVIDENCE.value in statuses:
        return ActionabilityStatus.INSUFFICIENT_EVIDENCE.value
    if ActionabilityStatus.NOT_APPLICABLE.value in statuses:
        return ActionabilityStatus.NOT_APPLICABLE.value
    return ActionabilityStatus.NOT_ASSESSED.value
