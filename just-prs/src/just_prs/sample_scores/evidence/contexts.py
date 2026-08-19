"""Extra-clinical ``trait_contexts`` and ``record_search_terms``."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from just_prs.sample_scores.evidence.models import (
    ContextClass,
    GuidelineRecord,
    GuidelineTraitLink,
    RecordSearchTerm,
    ScoreTraitLink,
    SearchTrack,
    TraitContextRecord,
    TraitRecord,
)

_LONGEVITY_RE = re.compile(
    r"\b(longevity|lifespan|healthspan|aging|ageing|menarche|menopause|parental longevity)\b",
    flags=re.IGNORECASE,
)
_DRUG_RESPONSE_RE = re.compile(
    r"\b(response to|drug response|treatment response|statin response|"
    r"clopidogrel|ototoxicity|cisplatin)\b",
    flags=re.IGNORECASE,
)
_SPORTS_RE = re.compile(
    r"\b(endurance|athlete|vo2|athletic|caffeine|injury|muscle|grip strength)\b",
    flags=re.IGNORECASE,
)
_LIFESTYLE_RE = re.compile(
    r"\b(sleep|alcohol|lactose|caffeine|diet|bitter taste|chronotype|smoking)\b",
    flags=re.IGNORECASE,
)
_ENVIRONMENT_RE = re.compile(
    r"\b(uv|ultraviolet|air pollution|altitude|noise|sunburn|skin pigmentation)\b",
    flags=re.IGNORECASE,
)
_BEHAVIORAL_RE = re.compile(
    r"\b(neuroticism|risk-taking|personality|wellbeing|well-being|mood)\b",
    flags=re.IGNORECASE,
)
_CAREER_RE = re.compile(
    r"\b(educational attainment|intelligence|cognitive|years of schooling)\b",
    flags=re.IGNORECASE,
)
_APPEARANCE_RE = re.compile(
    r"\b(height|baldness|hair|eye colo|pigmentation|skin colo|body mass|bmi)\b",
    flags=re.IGNORECASE,
)
_CHECKUP_RE = re.compile(
    r"\b(myopia|refractive|ldl|hdl|cholesterol|lipid|intraocular|hearing)\b",
    flags=re.IGNORECASE,
)

_CURATED_IDS: dict[str, list[tuple[str, str, str]]] = {
    # trait_id -> [(context_class, basis, why_interesting)]
    "EFO_0004300": [
        (ContextClass.AGING.value, "curated_map", "Longevity / lifespan standing in a reference panel."),
        (ContextClass.LIFE_OPTIMIZATION.value, "curated_map", "Related healthspan levers already tracked by the person."),
    ],
    "EFO_0004339": [
        (ContextClass.CURIOSITY.value, "curated_map", "Adult height is a widely studied anthropometric trait."),
        (ContextClass.APPEARANCE.value, "curated_map", "Visible stature people already know about themselves."),
    ],
    "EFO_0004340": [
        (ContextClass.LIFESTYLE.value, "curated_map", "BMI is an everyday body-composition measurement."),
        (ContextClass.CHECKUP_HINT.value, "curated_map", "An existing weight / BMI measurement contextualizes the PRS."),
        (ContextClass.LIFE_OPTIMIZATION.value, "curated_map", "Already-tracked body-composition notes."),
    ],
    "EFO_0004337": [
        (ContextClass.CURIOSITY.value, "curated_map", "Cognitive scores are a research-interest trait."),
        (ContextClass.CAREER.value, "curated_map", "Work/school context, not a diagnosis."),
    ],
    "EFO_0004784": [
        (ContextClass.CAREER.value, "curated_map", "Educational attainment as work/school context."),
        (ContextClass.CURIOSITY.value, "curated_map", "Widely published social-science PRS."),
    ],
    "EFO_0004198": [
        (ContextClass.APPEARANCE.value, "curated_map", "Male-pattern baldness is a visible trait."),
        (ContextClass.CURIOSITY.value, "curated_map", "Common citizen-science appearance score."),
    ],
}


def _text(trait: TraitRecord) -> str:
    return f"{trait.trait_id} {trait.label} {trait.definition or ''} {trait.category or ''}"


def looks_like_drug_response(*texts: str) -> bool:
    """Single text predicate used for both assignment and validation."""
    blob = " ".join(part for part in texts if part)
    return bool(_DRUG_RESPONSE_RE.search(blob))


def is_drug_response_trait(trait: TraitRecord, linked_reported: list[str] | None = None) -> bool:
    """True when the trait itself is a published drug-response concept."""
    if trait.category == "drug_response":
        return True
    return looks_like_drug_response(_text(trait), *(linked_reported or []))


def _is_longevity(trait: TraitRecord) -> bool:
    return bool(_LONGEVITY_RE.search(_text(trait)))


def _is_disease(trait: TraitRecord) -> bool:
    return trait.category == "disease" or bool(
        re.search(r"\b(disease|cancer|diabetes|hypertension|disorder)\b", _text(trait), re.I)
    )


def assign_trait_contexts(
    traits: list[TraitRecord],
    score_links: list[ScoreTraitLink],
    *,
    drug_response_pgs_ids: set[str],
) -> list[TraitContextRecord]:
    """Assign extra-clinical classes. Pharmacology requires a drug-response PGS."""
    reported_by_trait: dict[str, list[str]] = {}
    pgs_by_trait: dict[str, set[str]] = {}
    for link in score_links:
        if link.trait_reported:
            reported_by_trait.setdefault(link.trait_id, []).append(link.trait_reported)
        pgs_by_trait.setdefault(link.trait_id, set()).add(link.pgs_id)

    rows: list[TraitContextRecord] = []
    for trait in traits:
        assigned: list[TraitContextRecord] = []
        for context_class, basis, why in _CURATED_IDS.get(trait.trait_id, []):
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=context_class,
                    basis=basis,
                    why_interesting=why,
                    high_percentile_means="higher_trait_value",
                )
            )
        text = _text(trait)
        if _is_longevity(trait) and not any(row.context_class == ContextClass.AGING.value for row in assigned):
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.AGING.value,
                    basis="curated_map",
                    why_interesting="Longevity/aging standing is a headline extra-clinical use.",
                    high_percentile_means="higher_trait_value",
                )
            )
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.LIFE_OPTIMIZATION.value,
                    basis="curated_map",
                    why_interesting="Related already-tracked healthspan levers.",
                    high_percentile_means="higher_trait_value",
                )
            )
        linked_pgs = pgs_by_trait.get(trait.trait_id, set())
        has_drug_pgs = bool(linked_pgs & drug_response_pgs_ids)
        if has_drug_pgs:
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.PHARMACOLOGY.value,
                    basis="pgs_mapped_trait_class",
                    why_interesting="Published drug- or treatment-response PGS. Useful if already on that class of drug; not a prescription.",
                    high_percentile_means="higher_trait_value",
                )
            )
        if _SPORTS_RE.search(text) and not any(row.context_class == ContextClass.SPORTS.value for row in assigned):
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.SPORTS.value,
                    basis="efo_category",
                    why_interesting=trait.definition or trait.label,
                    high_percentile_means="higher_trait_value",
                )
            )
        if _LIFESTYLE_RE.search(text) and not any(row.context_class == ContextClass.LIFESTYLE.value for row in assigned):
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.LIFESTYLE.value,
                    basis="efo_category",
                    why_interesting=trait.definition or trait.label,
                    high_percentile_means="higher_trait_value",
                )
            )
        if _ENVIRONMENT_RE.search(text):
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.ENVIRONMENT.value,
                    basis="efo_category",
                    why_interesting="Response to an external exposure, not a clinic pathway.",
                    high_percentile_means="higher_trait_value",
                )
            )
        if _BEHAVIORAL_RE.search(text):
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.BEHAVIORAL.value,
                    basis="efo_category",
                    why_interesting="Self-reflection context. Not a psychiatric diagnosis.",
                )
            )
        if _CAREER_RE.search(text) and not any(row.context_class == ContextClass.CAREER.value for row in assigned):
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.CAREER.value,
                    basis="efo_category",
                    why_interesting="Work/school context. Not a diagnosis.",
                    high_percentile_means="higher_trait_value",
                )
            )
        if _APPEARANCE_RE.search(text) and not any(row.context_class == ContextClass.APPEARANCE.value for row in assigned):
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.APPEARANCE.value,
                    basis="efo_category",
                    why_interesting="Visible trait people already use for self-image.",
                )
            )
        if _CHECKUP_RE.search(text) and not any(row.context_class == ContextClass.CHECKUP_HINT.value for row in assigned):
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.CHECKUP_HINT.value,
                    basis="efo_category",
                    why_interesting="An already-available related measurement would make the PRS less abstract.",
                )
            )
        if not assigned:
            if _is_disease(trait):
                continue
            assigned.append(
                TraitContextRecord(
                    trait_id=trait.trait_id,
                    context_class=ContextClass.CURIOSITY.value,
                    basis="efo_category",
                    why_interesting=trait.definition or trait.label,
                )
            )
        seen: set[str] = set()
        for row in assigned:
            if row.context_class in seen:
                continue
            if row.context_class == ContextClass.PHARMACOLOGY.value and not has_drug_pgs:
                continue
            seen.add(row.context_class)
            rows.append(row)
    return rows


def drug_response_pgs_ids(scores_df_rows: Iterable[dict[str, object]]) -> set[str]:
    """PGS IDs whose catalog trait looks like a published drug-response score."""
    ids: set[str] = set()
    for row in scores_df_rows:
        pgs_id = str(row.get("pgs_id") or "")
        if looks_like_drug_response(
            *(str(row.get(col) or "") for col in ("trait_reported", "trait_efo", "name"))
        ):
            ids.add(pgs_id)
    return ids


def build_record_search_terms(
    traits: list[TraitRecord],
    contexts: list[TraitContextRecord],
    guidelines: list[GuidelineRecord],
    guideline_links: list[GuidelineTraitLink],
) -> list[RecordSearchTerm]:
    """Clinical and extra-clinical search terms. No restricted vocabularies."""
    guideline_by_id = {item.guideline_id: item for item in guidelines}
    links_by_trait: dict[str, list[GuidelineRecord]] = {}
    for link in guideline_links:
        guideline = guideline_by_id.get(link.guideline_id)
        if guideline is None:
            continue
        links_by_trait.setdefault(link.trait_id, []).append(guideline)

    contexts_by_trait: dict[str, list[TraitContextRecord]] = {}
    for row in contexts:
        contexts_by_trait.setdefault(row.trait_id, []).append(row)

    terms: list[RecordSearchTerm] = []
    for trait in traits:
        synonyms = json.loads(trait.synonyms_json)
        icd10 = json.loads(trait.icd10_codes_json)
        terms.append(
            RecordSearchTerm(
                trait_id=trait.trait_id,
                track=SearchTrack.CLINICAL.value,
                term=trait.label,
                term_kind="label",
                provenance="ontology_mapping",
            )
        )
        for synonym in synonyms:
            terms.append(
                RecordSearchTerm(
                    trait_id=trait.trait_id,
                    track=SearchTrack.CLINICAL.value,
                    term=str(synonym),
                    term_kind="synonym",
                    provenance="ontology_mapping",
                )
            )
        for code in icd10:
            terms.append(
                RecordSearchTerm(
                    trait_id=trait.trait_id,
                    track=SearchTrack.CLINICAL.value,
                    term=str(code),
                    term_kind="icd10",
                    provenance="ontology_mapping",
                )
            )
        mapped_guidelines = links_by_trait.get(trait.trait_id, [])
        if mapped_guidelines:
            for resource in ("Condition", "Observation", "FamilyMemberHistory"):
                terms.append(
                    RecordSearchTerm(
                        trait_id=trait.trait_id,
                        track=SearchTrack.CLINICAL.value,
                        term=resource,
                        term_kind="fhir_resource",
                        provenance="guideline_eligibility",
                    )
                )
            for guideline in mapped_guidelines:
                if guideline.required_context:
                    terms.append(
                        RecordSearchTerm(
                            trait_id=trait.trait_id,
                            track=SearchTrack.CLINICAL.value,
                            term=guideline.required_context,
                            term_kind="data_category",
                            provenance="guideline_eligibility",
                        )
                    )
        for context in contexts_by_trait.get(trait.trait_id, []):
            extra_term = {
                ContextClass.SPORTS.value: "already-available training metrics",
                ContextClass.LIFESTYLE.value: "already-tracked sleep, diet, or caffeine notes",
                ContextClass.ENVIRONMENT.value: "exposure history",
                ContextClass.AGING.value: "chronological age and family longevity history",
                ContextClass.PHARMACOLOGY.value: "current or historical medication name",
                ContextClass.CHECKUP_HINT.value: "existing related measurement",
                ContextClass.CAREER.value: "educational history the person already knows",
                ContextClass.APPEARANCE.value: "already-known height, hair, or pigmentation",
                ContextClass.LIFE_OPTIMIZATION.value: "already-tracked sleep, training, lipids, caffeine",
                ContextClass.BEHAVIORAL.value: "self-reflection notes",
                ContextClass.CURIOSITY.value: "already-known phenotype",
            }.get(context.context_class, context.why_interesting)
            terms.append(
                RecordSearchTerm(
                    trait_id=trait.trait_id,
                    track=SearchTrack.EXTRA_CLINICAL.value,
                    context_class=context.context_class,
                    term=extra_term,
                    term_kind="data_category",
                    provenance=context.basis,
                )
            )
    unique = {
        (row.trait_id, row.track, row.term_kind, row.term): row for row in terms if row.term
    }
    return list(unique.values())
