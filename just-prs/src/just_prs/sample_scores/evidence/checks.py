"""Blocking evidence-table checks that do not need a sample×PGS matrix."""

from __future__ import annotations

from just_prs.sample_scores.evidence.models import (
    ActionabilityStatus,
    ContextClass,
    EVIDENCE_PRIMARY_KEYS,
    GuidelineRecord,
    PaperRecord,
    TraitContextRecord,
    TraitRecord,
)
from just_prs.sample_scores.evidence.papers import abstract_is_redistributable


class EvidenceCheckError(ValueError):
    """One or more blocking evidence invariants failed."""


def _unique_key_violations(rows: list[object], keys: tuple[str, ...]) -> list[str]:
    seen: set[tuple[object, ...]] = set()
    dupes: list[str] = []
    for row in rows:
        payload = row.model_dump() if hasattr(row, "model_dump") else dict(row)
        key = tuple(payload.get(name) for name in keys)
        if key in seen:
            dupes.append(repr(key))
        seen.add(key)
    return dupes


def validate_evidence_tables(
    *,
    traits: list[TraitRecord],
    score_trait_links: list[object],
    papers: list[PaperRecord],
    score_paper_links: list[object],
    guidelines: list[GuidelineRecord],
    guideline_trait_links: list[object],
    actionability: list[object],
    trait_contexts: list[TraitContextRecord],
    record_search_terms: list[object],
    drug_response_pgs_ids: set[str],
    score_trait_pgs_by_trait: dict[str, set[str]] | None = None,
) -> None:
    """Raise ``EvidenceCheckError`` if a blocking invariant fails."""
    tables = {
        "traits": traits,
        "score_trait_links": score_trait_links,
        "papers": papers,
        "score_paper_links": score_paper_links,
        "guidelines": guidelines,
        "guideline_trait_links": guideline_trait_links,
        "actionability": actionability,
        "trait_contexts": trait_contexts,
        "record_search_terms": record_search_terms,
    }
    issues: list[str] = []
    for name, rows in tables.items():
        dupes = _unique_key_violations(rows, EVIDENCE_PRIMARY_KEYS[name])
        if dupes:
            issues.append(f"{name}: duplicate keys {dupes[:5]}")

    guideline_ids = {item.guideline_id for item in guidelines}
    for row in actionability:
        payload = row.model_dump() if hasattr(row, "model_dump") else dict(row)
        statuses = [
            payload.get("prs_actionability_status"),
            payload.get("condition_actionability_status"),
            payload.get("context_resolution_status"),
        ]
        cited = payload.get("guideline_id") or ""
        if any(status != ActionabilityStatus.NOT_ASSESSED.value for status in statuses):
            if not cited or cited not in guideline_ids:
                issues.append(
                    f"actionability {payload.get('trait_id')}: non-not_assessed without cited guideline"
                )

    for paper in papers:
        if paper.abstract_text and not abstract_is_redistributable(paper.content_license):
            issues.append(f"paper {paper.paper_id}: abstract without compatible license")

    known_trait_ids = {trait.trait_id for trait in traits}
    pgs_by_trait = score_trait_pgs_by_trait or {}
    for context in trait_contexts:
        if context.trait_id not in known_trait_ids:
            issues.append(f"orphan trait_context for {context.trait_id}")
            continue
        if context.context_class != ContextClass.PHARMACOLOGY.value:
            continue
        linked = pgs_by_trait.get(context.trait_id, set())
        if not (linked & drug_response_pgs_ids):
            issues.append(f"pharmacology on {context.trait_id} without a drug-response PGS")
    for term in record_search_terms:
        payload = term.model_dump() if hasattr(term, "model_dump") else dict(term)
        trait_id = str(payload.get("trait_id") or "")
        if trait_id and trait_id not in known_trait_ids:
            issues.append(f"orphan search term for {trait_id}")

    longevity_ids = {
        trait.trait_id
        for trait in traits
        if "longevity" in trait.label.lower()
        or "lifespan" in trait.label.lower()
        or "healthspan" in (trait.definition or "").lower()
    }
    aging_by_trait = {
        row.trait_id
        for row in trait_contexts
        if row.context_class == ContextClass.AGING.value
    }
    for trait_id in longevity_ids:
        if trait_id not in aging_by_trait:
            issues.append(f"longevity trait {trait_id} missing aging context")

    extra_clinical_in_actionability = False
    for row in actionability:
        payload = row.model_dump() if hasattr(row, "model_dump") else dict(row)
        notes = str(payload.get("notes") or "")
        if "context_class=" in notes:
            extra_clinical_in_actionability = True
    if extra_clinical_in_actionability:
        issues.append("extra-clinical class written into actionability")

    if issues:
        raise EvidenceCheckError("; ".join(issues))
