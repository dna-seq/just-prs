"""Build ``traits`` and ``score_trait_links`` from cleaned catalog + ontology."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from eliot import start_action

from just_prs.ontology import (
    colon_trait_id,
    normalize_trait_id,
    query_ols_trait_xrefs,
    trait_iri,
)
from just_prs.sample_scores.evidence.models import ScoreTraitLink, TraitRecord

_TRAIT_ID_SPLIT = ","


def explode_catalog_trait_ids(raw: object) -> list[str]:
    """Split a catalog ``trait_efo_id`` cell into normalized ontology IDs."""
    if raw is None:
        return []
    if isinstance(raw, list):
        parts = [str(item) for item in raw]
    else:
        parts = str(raw).split(_TRAIT_ID_SPLIT)
    ids: list[str] = []
    for part in parts:
        normalized = normalize_trait_id(part.strip())
        if normalized and normalized not in ids:
            ids.append(normalized)
    return ids


def reported_trait_id(label: str) -> str:
    """Stable unresolved ID for a reported trait with no ontology mapping."""
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return f"REPORTED:{slug or 'unknown'}"


def _load_ols_record(trait_id: str, cache_dir: Path | None, allow_network: bool) -> dict[str, object]:
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{trait_id}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    if not allow_network:
        return {
            "trait_id": trait_id,
            "label": None,
            "definition": None,
            "synonyms": [],
            "aliases": [],
            "icd10_codes": [],
            "ontology_prefix": trait_id.split("_", 1)[0] if "_" in trait_id else None,
            "canonical_url": trait_iri(trait_id),
        }
    record = query_ols_trait_xrefs(trait_id)
    if cache_file is not None:
        cache_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def infer_trait_category(label: str, definition: str | None, trait_id: str) -> str:
    """Deterministic category from catalog/OLS text. Not an LLM label."""
    text = f"{label} {definition or ''} {trait_id}".lower()
    if any(token in text for token in ("response to", "drug response", "treatment response")):
        return "drug_response"
    if any(token in text for token in ("neuroticism", "risk-taking", "personality", "wellbeing")):
        return "behavioral"
    if any(
        token in text
        for token in (
            "body mass",
            "bmi",
            "height",
            "measurement",
            "level",
            "concentration",
            "cholesterol",
            "ldl",
            "hdl",
        )
    ):
        return "measurement"
    if any(
        token in text
        for token in (
            "disease",
            "cancer",
            "diabetes",
            "hypertension",
            "disorder",
            "syndrome",
            "mellitus",
        )
    ):
        return "disease"
    if trait_id.startswith("EFO_") or trait_id.startswith("MONDO_"):
        return "ontology"
    return "other"


def build_trait_tables(
    scores_df: pl.DataFrame,
    *,
    ols_cache_dir: Path | None = None,
    allow_network: bool = True,
    retrieved_at: str | None = None,
) -> tuple[list[TraitRecord], list[ScoreTraitLink]]:
    """Explode catalog trait IDs and enrich unique concepts via OLS when available."""
    stamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    with start_action(action_type="sample_scores:evidence:traits"):
        links: list[ScoreTraitLink] = []
        catalog_labels: dict[str, str] = {}
        for row in scores_df.iter_rows(named=True):
            pgs_id = str(row.get("pgs_id") or "").strip()
            if not pgs_id:
                continue
            reported = row.get("trait_reported")
            reported_text = str(reported).strip() if reported else None
            efo_ids = explode_catalog_trait_ids(row.get("trait_efo_id"))
            efo_labels = [part.strip() for part in str(row.get("trait_efo") or "").split(",") if part.strip()]
            if not efo_ids and reported_text:
                efo_ids = [reported_trait_id(reported_text)]
            for index, trait_id in enumerate(efo_ids):
                if index < len(efo_labels) and efo_labels[index]:
                    catalog_labels.setdefault(trait_id, efo_labels[index])
                elif reported_text:
                    catalog_labels.setdefault(trait_id, reported_text)
                links.append(
                    ScoreTraitLink(
                        pgs_id=pgs_id,
                        trait_id=trait_id,
                        relationship_source="pgs_catalog",
                        trait_reported=reported_text,
                    )
                )

        unique_ids = sorted({link.trait_id for link in links})
        traits: list[TraitRecord] = []
        alias_links: list[ScoreTraitLink] = []
        for trait_id in unique_ids:
            if trait_id.startswith("REPORTED:"):
                label = catalog_labels.get(trait_id, trait_id.removeprefix("REPORTED:"))
                traits.append(
                    TraitRecord(
                        trait_id=trait_id,
                        label=label.replace("_", " "),
                        mapping_source="pgs_catalog_reported",
                        mapping_confidence="unresolved",
                        mapping_status="unresolved",
                        category=infer_trait_category(label, None, trait_id),
                        retrieved_at=stamp,
                    )
                )
                continue
            ols = _load_ols_record(trait_id, ols_cache_dir, allow_network)
            label = str(ols.get("label") or catalog_labels.get(trait_id) or colon_trait_id(trait_id))
            definition = ols.get("definition")
            definition_text = str(definition) if definition else None
            synonyms = [str(item) for item in (ols.get("synonyms") or []) if item]
            aliases = [str(item) for item in (ols.get("aliases") or []) if item]
            icd10 = [str(item) for item in (ols.get("icd10_codes") or []) if item]
            resolved = bool(ols.get("label") or catalog_labels.get(trait_id))
            traits.append(
                TraitRecord(
                    trait_id=trait_id,
                    label=label,
                    definition=definition_text,
                    synonyms_json=json.dumps(synonyms, ensure_ascii=True),
                    category=infer_trait_category(label, definition_text, trait_id),
                    ontology_prefix=str(ols.get("ontology_prefix") or trait_id.split("_", 1)[0]),
                    canonical_url=str(ols.get("canonical_url") or trait_iri(trait_id) or ""),
                    aliases_json=json.dumps(aliases, ensure_ascii=True),
                    icd10_codes_json=json.dumps(icd10, ensure_ascii=True),
                    mapping_source="ols4" if ols.get("label") else "pgs_catalog",
                    mapping_confidence="ols4" if ols.get("label") else "catalog",
                    mapping_status="resolved" if resolved else "unresolved",
                    retrieved_at=stamp,
                )
            )
            catalog_pgs = {link.pgs_id for link in links if link.trait_id == trait_id}
            for alias in aliases:
                catalog_labels.setdefault(alias, label)
                for pgs_id in catalog_pgs:
                    alias_links.append(
                        ScoreTraitLink(
                            pgs_id=pgs_id,
                            trait_id=alias,
                            relationship_source="ontology_alias",
                            trait_reported=None,
                        )
                    )

        seen_alias_traits = {trait.trait_id for trait in traits}
        for link in alias_links:
            if link.trait_id in seen_alias_traits:
                continue
            seen_alias_traits.add(link.trait_id)
            traits.append(
                TraitRecord(
                    trait_id=link.trait_id,
                    label=catalog_labels.get(link.trait_id, colon_trait_id(link.trait_id)),
                    ontology_prefix=link.trait_id.split("_", 1)[0] if "_" in link.trait_id else None,
                    canonical_url=trait_iri(link.trait_id),
                    mapping_source="ols4_xref",
                    mapping_confidence="alias",
                    mapping_status="resolved",
                    retrieved_at=stamp,
                )
            )

        unique_links: dict[tuple[str, str], ScoreTraitLink] = {}
        for link in [*links, *alias_links]:
            key = (link.pgs_id, link.trait_id)
            existing = unique_links.get(key)
            if existing is None or (
                existing.relationship_source != "pgs_catalog"
                and link.relationship_source == "pgs_catalog"
            ):
                unique_links[key] = link
        return traits, list(unique_links.values())
