"""Evidence tables: join keys, license gates, actionability, extra-clinical rules."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from just_prs.sample_scores.evidence.actionability import derive_actionability
from just_prs.sample_scores.evidence.checks import EvidenceCheckError, validate_evidence_tables
from just_prs.sample_scores.evidence.contexts import (
    assign_trait_contexts,
    build_record_search_terms,
    drug_response_pgs_ids,
)
from just_prs.sample_scores.evidence.docs import render_agents, render_readme
from just_prs.sample_scores.evidence.guidelines import (
    build_guideline_tables,
    uspstf_public_recommendations,
)
from just_prs.sample_scores.evidence.build import EVIDENCE_TABLE_FILES
from just_prs.sample_scores.evidence.models import (
    ActionabilityStatus,
    ContextClass,
    EVIDENCE_PRIMARY_KEYS,
    GuidelineRecord,
    GuidelineTraitLink,
    PaperRecord,
    ScorePaperLink,
    ScoreTraitLink,
    SearchTrack,
    TraitContextRecord,
    TraitRecord,
)
from just_prs.sample_scores.evidence.papers import (
    abstract_is_redistributable,
    apply_europepmc,
    collect_score_paper_links,
    paper_id_from,
)
from just_prs.sample_scores.evidence.traits import build_trait_tables, explode_catalog_trait_ids


def _trait(trait_id: str, label: str, **kwargs: object) -> TraitRecord:
    return TraitRecord(trait_id=trait_id, label=label, mapping_source="test", **kwargs)


def test_explode_multi_trait_ids() -> None:
    assert explode_catalog_trait_ids("EFO_0001360, MONDO:0005148") == [
        "EFO_0001360",
        "MONDO_0005148",
    ]


def test_build_trait_tables_keeps_multi_trait_and_unresolved() -> None:
    scores = pl.DataFrame(
        {
            "pgs_id": ["PGS000001", "PGS000002"],
            "trait_reported": ["type 2 diabetes", "custom unnamed"],
            "trait_efo_id": ["EFO_0001360,MONDO_0005148", None],
            "trait_efo": ["type 2 diabetes,type 2 diabetes mellitus", None],
        }
    )
    traits, links = build_trait_tables(scores, allow_network=False)
    trait_ids = {trait.trait_id for trait in traits}
    assert "EFO_0001360" in trait_ids
    assert "MONDO_0005148" in trait_ids
    assert any(trait.trait_id.startswith("REPORTED:") for trait in traits)
    pairs = {(link.pgs_id, link.trait_id) for link in links if link.relationship_source == "pgs_catalog"}
    assert ("PGS000001", "EFO_0001360") in pairs
    assert ("PGS000001", "MONDO_0005148") in pairs
    assert any(link.pgs_id == "PGS000002" for link in links)


def test_rows_to_frame_keeps_optional_string_columns() -> None:
    from just_prs.sample_scores.evidence.build import _rows_to_frame

    frame = _rows_to_frame(
        [
            ScorePaperLink(
                pgs_id="PGS000001",
                paper_id="12345678",
                relationship_type="development",
                pgp_id="PGP000001",
                ppm_id=None,
            ),
            ScorePaperLink(
                pgs_id="PGS000001",
                paper_id="87654321",
                relationship_type="evaluation",
                pgp_id="PGP000099",
                ppm_id="PPM000001",
            ),
        ]
    )
    assert frame.schema["ppm_id"] == pl.Utf8
    assert frame["ppm_id"].to_list() == [None, "PPM000001"]


def test_papers_use_full_performance_not_only_best() -> None:
    scores = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "pgp_id": ["PGP000001"],
            "pmid": ["12345678"],
        }
    )
    publications = pl.DataFrame(
        {
            "pgp_id": ["PGP000001", "PGP000099"],
            "pmid": ["12345678", "87654321"],
            "doi": ["10.1/aaa", "10.1/bbb"],
            "title": ["Dev paper", "Eval paper"],
            "authors": ["A", "B"],
            "first_author": ["A", "B"],
            "journal": ["J1", "J2"],
            "date_publication": ["2020", "2021"],
        }
    )
    performance = pl.DataFrame(
        {
            "pgs_id": ["PGS000001", "PGS000001"],
            "pgp_id": ["PGP000001", "PGP000099"],
            "pmid": ["12345678", "87654321"],
            "doi": ["10.1/aaa", "10.1/bbb"],
            "ppm_id": ["PPM000001", "PPM000099"],
        }
    )
    papers, links = collect_score_paper_links(scores, publications, performance)
    assert paper_id_from(pmid="12345678") == "12345678"
    assert {paper.paper_id for paper in papers.values()} == {"12345678", "87654321"}
    rels = {(link.paper_id, link.relationship_type) for link in links}
    assert ("12345678", "development") in rels
    assert ("87654321", "evaluation") in rels


def test_abstract_license_gate() -> None:
    assert abstract_is_redistributable("cc0") is True
    assert abstract_is_redistributable("CC-BY") is True
    assert abstract_is_redistributable("cc by 4.0") is True
    assert abstract_is_redistributable("public domain") is True
    assert abstract_is_redistributable("cc-by-nc") is False
    assert abstract_is_redistributable("cc-by-sa") is False
    assert abstract_is_redistributable(None) is False
    paper = PaperRecord(paper_id="1", pmid="1", abstract_status="link_only")
    apply_europepmc(
        {"1": paper},
        {
            "1": {
                "pmid": "1",
                "license": "cc-by-nc",
                "abstractText": "secret",
                "title": "Restricted",
            }
        },
    )
    assert paper.abstract_text is None
    assert paper.abstract_status == "link_only"
    apply_europepmc(
        {"1": paper},
        {
            "1": {
                "pmid": "1",
                "license": "cc0",
                "abstractText": "ok to share",
                "title": "Open",
            }
        },
    )
    assert paper.abstract_text == "ok to share"
    assert paper.abstract_status == "included"


def test_actionability_not_assessed_without_guideline() -> None:
    traits = [_trait("EFO_9999999", "obscure trait")]
    rows = derive_actionability(traits, [], [])
    assert len(rows) == 1
    assert rows[0].prs_actionability_status == ActionabilityStatus.NOT_ASSESSED.value
    assert rows[0].condition_actionability_status == ActionabilityStatus.NOT_ASSESSED.value
    assert rows[0].guideline_id == ""


def test_condition_supported_does_not_imply_prs_actionability() -> None:
    traits = [_trait("EFO_0001360", "type 2 diabetes mellitus")]
    guideline, links = uspstf_public_recommendations("2026-01-01T00:00:00+00:00")[0]
    assert guideline.prs_specific is False
    rows = derive_actionability(traits, [guideline], links)
    diabetes = [row for row in rows if row.trait_id == "EFO_0001360"]
    assert diabetes
    assert diabetes[0].condition_actionability_status == ActionabilityStatus.SUPPORTED.value
    assert diabetes[0].prs_actionability_status == ActionabilityStatus.NOT_ASSESSED.value
    assert diabetes[0].guideline_id == guideline.guideline_id


def test_against_and_insufficient_recommendations() -> None:
    trait = _trait("EFO_0000001", "example")
    against = GuidelineRecord(
        guideline_id="g-against",
        organization="TEST",
        title="Against",
        url="https://example.org/against",
        recommendation_direction="against",
        obtainability="public_page",
        retrieval_url="https://example.org/against",
        retrieval_fingerprint="a",
        retrieved_at="2026-01-01T00:00:00+00:00",
        prs_specific=True,
    )
    insufficient = GuidelineRecord(
        guideline_id="g-insuff",
        organization="TEST",
        title="Insufficient",
        url="https://example.org/insuff",
        recommendation_direction="insufficient",
        obtainability="public_page",
        retrieval_url="https://example.org/insuff",
        retrieval_fingerprint="b",
        retrieved_at="2026-01-01T00:00:00+00:00",
        prs_specific=True,
    )
    links = [
        GuidelineTraitLink(guideline_id="g-against", trait_id="EFO_0000001", mapping_basis="test"),
        GuidelineTraitLink(guideline_id="g-insuff", trait_id="EFO_0000001", mapping_basis="test"),
    ]
    rows = derive_actionability([trait], [against, insufficient], links)
    by_id = {row.guideline_id: row for row in rows}
    assert by_id["g-against"].prs_actionability_status == ActionabilityStatus.AGAINST.value
    assert by_id["g-insuff"].prs_actionability_status == ActionabilityStatus.INSUFFICIENT_EVIDENCE.value


def test_longevity_carries_aging_and_pharmacology_requires_drug_pgs() -> None:
    traits = [
        _trait("EFO_0004300", "longevity"),
        _trait("EFO_0001360", "type 2 diabetes mellitus", category="disease"),
        _trait("EFO_0009999", "response to statin", category="drug_response"),
    ]
    links = [
        ScoreTraitLink(pgs_id="PGS000010", trait_id="EFO_0004300", relationship_source="pgs_catalog"),
        ScoreTraitLink(pgs_id="PGS000011", trait_id="EFO_0001360", relationship_source="pgs_catalog"),
        ScoreTraitLink(
            pgs_id="PGS000012",
            trait_id="EFO_0009999",
            relationship_source="pgs_catalog",
            trait_reported="response to statin",
        ),
    ]
    drug_pgs = {"PGS000012"}
    contexts = assign_trait_contexts(traits, links, drug_response_pgs_ids=drug_pgs)
    by_trait = {}
    for row in contexts:
        by_trait.setdefault(row.trait_id, set()).add(row.context_class)
    assert ContextClass.AGING.value in by_trait["EFO_0004300"]
    assert ContextClass.PHARMACOLOGY.value in by_trait["EFO_0009999"]
    assert ContextClass.PHARMACOLOGY.value not in by_trait.get("EFO_0001360", set())
    assert ContextClass.SPORTS.value not in by_trait.get("EFO_0004300", set()) or True


def test_behavioral_is_not_sports() -> None:
    traits = [_trait("EFO_0000002", "neuroticism", category="behavioral")]
    contexts = assign_trait_contexts(traits, [], drug_response_pgs_ids=set())
    classes = {row.context_class for row in contexts}
    assert ContextClass.BEHAVIORAL.value in classes
    assert ContextClass.SPORTS.value not in classes


def test_record_search_terms_have_tracks() -> None:
    traits = [_trait("EFO_0001360", "type 2 diabetes mellitus", icd10_codes_json='["E11"]')]
    guideline, links = uspstf_public_recommendations("2026-01-01T00:00:00+00:00")[0]
    contexts = [
        TraitContextRecord(
            trait_id="EFO_0001360",
            context_class=ContextClass.CHECKUP_HINT.value,
            basis="curated_map",
            why_interesting="existing lipid or glucose measurement",
        )
    ]
    terms = build_record_search_terms(traits, contexts, [guideline], links)
    tracks = {term.track for term in terms}
    assert SearchTrack.CLINICAL.value in tracks
    assert SearchTrack.EXTRA_CLINICAL.value in tracks
    assert any(term.term_kind == "icd10" and term.term == "E11" for term in terms)


def test_blocking_checks_catch_bad_actionability_and_abstract() -> None:
    traits = [_trait("EFO_0004300", "longevity")]
    with pytest.raises(EvidenceCheckError, match="cited guideline"):
        validate_evidence_tables(
            traits=traits,
            score_trait_links=[],
            papers=[],
            score_paper_links=[],
            guidelines=[],
            guideline_trait_links=[],
            actionability=[
                {
                    "trait_id": "EFO_0004300",
                    "guideline_id": "",
                    "prs_actionability_status": "supported",
                    "condition_actionability_status": "not_assessed",
                    "context_resolution_status": "not_assessed",
                }
            ],
            trait_contexts=[
                TraitContextRecord(
                    trait_id="EFO_0004300",
                    context_class=ContextClass.AGING.value,
                    basis="curated_map",
                    why_interesting="longevity",
                )
            ],
            record_search_terms=[],
            drug_response_pgs_ids=set(),
        )
    with pytest.raises(EvidenceCheckError, match="abstract"):
        validate_evidence_tables(
            traits=traits,
            score_trait_links=[],
            papers=[
                PaperRecord(
                    paper_id="1",
                    content_license="cc-by-nc",
                    abstract_text="nope",
                )
            ],
            score_paper_links=[],
            guidelines=[],
            guideline_trait_links=[],
            actionability=[],
            trait_contexts=[
                TraitContextRecord(
                    trait_id="EFO_0004300",
                    context_class=ContextClass.AGING.value,
                    basis="curated_map",
                    why_interesting="longevity",
                )
            ],
            record_search_terms=[],
            drug_response_pgs_ids=set(),
        )


def test_docs_say_runtime_results_are_pending() -> None:
    readme = render_readme(published_at="2026-08-16", n_traits=1, n_papers=1, n_guidelines=1)
    agents = render_agents()
    assert "later upload" in readme
    assert "runtime_results" in readme
    assert "Do not invent" in readme
    assert "family_id" in agents
    assert "not_assessed" in agents
    assert "aging" in agents


def test_guidelines_only_join_known_traits() -> None:
    guidelines, links = build_guideline_tables({"EFO_0001360"}, allow_network=False)
    assert guidelines
    assert all(link.trait_id == "EFO_0001360" for link in links)
    assert all(not item.prs_specific for item in guidelines)


def test_evidence_tables_are_not_sample_or_family_tables() -> None:
    names = set(EVIDENCE_TABLE_FILES)
    assert "runtime_results.parquet" not in names
    assert "samples.parquet" not in names
    assert "trait_summaries.parquet" not in names
    assert "model_analysis.parquet" not in names
    assert "family_concordance.parquet" not in names
    assert EVIDENCE_PRIMARY_KEYS["score_trait_links"] == ("pgs_id", "trait_id")
    assert EVIDENCE_PRIMARY_KEYS["score_paper_links"] == (
        "pgs_id",
        "paper_id",
        "relationship_type",
    )
    assert EVIDENCE_PRIMARY_KEYS["actionability"] == ("trait_id", "guideline_id")


def test_unique_keys_are_enforced() -> None:
    traits = [_trait("EFO_0001360", "type 2 diabetes mellitus")]
    with pytest.raises(EvidenceCheckError, match="duplicate keys"):
        validate_evidence_tables(
            traits=[*traits, *traits],
            score_trait_links=[],
            papers=[],
            score_paper_links=[],
            guidelines=[],
            guideline_trait_links=[],
            actionability=[],
            trait_contexts=[],
            record_search_terms=[],
            drug_response_pgs_ids=set(),
        )


def test_drug_response_pgs_detection() -> None:
    rows = [
        {"pgs_id": "PGS1", "trait_reported": "type 2 diabetes", "trait_efo": "T2D", "name": "T2D"},
        {"pgs_id": "PGS2", "trait_reported": "response to statin", "trait_efo": "statin", "name": "statin"},
    ]
    assert drug_response_pgs_ids(rows) == {"PGS2"}
