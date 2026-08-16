"""Trait lookup for ``prs plot trait`` / ``prs prompt`` must match the UI EFO key."""

from __future__ import annotations

import polars as pl

from just_prs.cli import _select_trait_scores
from just_prs.prs_catalog import PRSCatalog


def _scores() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pgs_id": ["PGS1", "PGS2", "PGS3", "PGS4", "PGS5"],
            "name": ["iq-a", "iq-b", "t2d-a", "cad-a", "ad-a"],
            "trait_reported": [
                "Fluid intelligence score",
                "Fluid intelligence score",
                "Type 2 Diabetes Mellitus (T2D)",
                "Coronary artery disease",
                "Alzheimer's disease",
            ],
            "trait_efo": [
                "intelligence",
                "intelligence",
                "type 2 diabetes mellitus",
                "coronary artery disease",
                "Alzheimer disease",
            ],
        }
    )


def test_select_trait_scores_matches_ui_efo_label() -> None:
    matched, kind, suggestions = _select_trait_scores(_scores(), "intelligence")
    assert suggestions == []
    assert kind == "efo"
    assert matched is not None
    assert matched["pgs_id"].to_list() == ["PGS1", "PGS2"]
    assert set(matched["trait_reported"]) == {"Fluid intelligence score"}


def test_select_trait_scores_exact_reported_name() -> None:
    matched, kind, _ = _select_trait_scores(_scores(), "Fluid intelligence score")
    assert kind == "exact"
    assert matched is not None
    assert matched["pgs_id"].to_list() == ["PGS1", "PGS2"]


def test_select_trait_scores_unique_efo_partial_does_not_need_fuzzy() -> None:
    matched, kind, _ = _select_trait_scores(_scores(), "type 2 diabetes")
    assert kind == "efo"
    assert matched is not None
    assert matched["pgs_id"].to_list() == ["PGS3"]


def test_select_trait_scores_mixed_hits_require_fuzzy() -> None:
    matched, kind, suggestions = _select_trait_scores(_scores(), "disease")
    assert matched is None
    assert kind == ""
    assert "coronary artery disease" in suggestions
    assert "Alzheimer disease" in suggestions

    matched, kind, _ = _select_trait_scores(_scores(), "disease", fuzzy=True)
    assert kind == "fuzzy"
    assert matched is not None
    assert set(matched["pgs_id"].to_list()) == {"PGS4", "PGS5"}


def test_select_trait_scores_intelligence_from_catalog() -> None:
    scores = PRSCatalog().scores(genome_build="GRCh38", include_harmonized=True).collect()
    matched, kind, suggestions = _select_trait_scores(scores, "intelligence")
    assert suggestions == []
    assert kind == "efo"
    assert matched is not None
    assert matched.height >= 5
    assert set(matched["trait_efo"].drop_nulls().to_list()) == {"intelligence"}
    assert "Fluid intelligence score" in matched["trait_reported"].to_list()
