"""prs-ui consumes the published score-direction curation (AGENTS.md "Consuming curated directions").

Real tables from ``just-dna-seq/pgs-catalog`` through ``PRSCatalog``; no mocks.
"""

from __future__ import annotations

import random

import polars as pl
import pytest

from just_prs import PRSCatalog
from just_prs.models import EnrichedPRSResult
from just_prs.score_direction import SCORE_DIRECTION_COLUMNS, curation_badge, direction_tone
from prs_ui.curation_browser import SCORE_COLUMNS, prepare_score_display, published_curation_browser
from prs_ui.mixin import (
    TRAIT_GROUP_BY_CONCEPT,
    TRAIT_GROUP_BY_ONTOLOGY,
    _CURATION_BADGE_COLORS,
    _enriched_to_row_dict,
    _group_prs_rows_by_trait,
    _percentile_tone,
    _trait_group_display_label,
    _trait_group_key,
    curated_trait_signal,
    normalize_trait_group_by,
    trait_group_label_expr,
)


@pytest.fixture(scope="module")
def catalog() -> PRSCatalog:
    return PRSCatalog()


def test_curated_tab_falls_back_to_the_published_tables(catalog: PRSCatalog) -> None:
    data = published_curation_browser(catalog)
    assert data is not None and data.origin == "published"
    published = catalog.curation_tables()
    assert set(data.tables.annotations["pgs_id"]) == set(published["annotations"]["pgs_id"])
    assert data.n_review == published["review"].height
    assert data.n_verified + data.n_unreviewed == published["annotations"].height
    shown = prepare_score_display(data.tables.annotations, "all")
    assert shown.columns == list(SCORE_COLUMNS)
    assert set(shown["pgs_id"]) == set(published["annotations"]["pgs_id"])
    assert "just-dna-seq/pgs-catalog" in data.description


def test_concept_is_the_default_grouping() -> None:
    assert normalize_trait_group_by(None) == TRAIT_GROUP_BY_CONCEPT
    assert normalize_trait_group_by("nonsense") == TRAIT_GROUP_BY_CONCEPT
    assert normalize_trait_group_by(TRAIT_GROUP_BY_ONTOLOGY) == TRAIT_GROUP_BY_ONTOLOGY


def test_selector_labels_and_result_groups_agree(catalog: PRSCatalog) -> None:
    scores = catalog.scores(genome_build="GRCh38").collect()
    labeled = scores.with_columns(trait_group_label_expr(TRAIT_GROUP_BY_CONCEPT))
    rng = random.Random(5)
    sample = labeled.filter(pl.col("concept_label").is_not_null()).sample(150, seed=5).to_dicts() + rng.sample(
        labeled.filter(pl.col("concept_label").is_null()).to_dicts(), 50
    )
    for row in sample:
        assert _trait_group_key(row, TRAIT_GROUP_BY_CONCEPT) == row["trait"].casefold()
        if row["concept_label"]:
            assert row["trait"] == row["concept_label"]
    # Every group of computed rows is titled with the label the selector showed.
    for group in _group_prs_rows_by_trait(sample, TRAIT_GROUP_BY_CONCEPT):
        label, _reported = _trait_group_display_label(group, TRAIT_GROUP_BY_CONCEPT)
        assert {row["trait"].casefold() for row in group} == {label.casefold()}


def test_enriched_rows_bridge_the_direction_columns(catalog: PRSCatalog) -> None:
    directions = catalog.score_directions()
    for status in ("resolved", "in_review", "no_direction"):
        direction = directions.filter(pl.col("curation_status") == status).row(0, named=True)
        enriched = EnrichedPRSResult(
            pgs_id=direction["pgs_id"], **{c: direction[c] for c in SCORE_DIRECTION_COLUMNS}
        )
        row = _enriched_to_row_dict(enriched)
        assert {c: row[c] for c in SCORE_DIRECTION_COLUMNS} == {c: direction[c] for c in SCORE_DIRECTION_COLUMNS}
        label, _tone = curation_badge(row)
        assert label in _CURATION_BADGE_COLORS, "every badge label has a colour in the results grid"
    uncurated = _enriched_to_row_dict(EnrichedPRSResult(pgs_id="PGS_X"))
    assert direction_tone(95.0, uncurated) == _percentile_tone(95.0)
    assert curation_badge(uncurated)[0] in _CURATION_BADGE_COLORS


def test_trait_signal_is_reworded_only_where_high_is_not_alarming() -> None:
    elevated = "Consistently elevated"
    assert curated_trait_signal(elevated, False, "desirable") == elevated
    assert curated_trait_signal(elevated, True, "undesirable") == elevated
    assert curated_trait_signal(elevated, True, "desirable") == "Consistently high (favourable)"
    assert curated_trait_signal(elevated, True, "neutral") == "Consistently high"
    assert curated_trait_signal("Mostly average", True, "desirable") == "Mostly average"
