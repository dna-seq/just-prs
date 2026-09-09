"""Regression tests for the EnrichedPRSResult -> grid-row dict bridge.

``_enriched_to_row_dict`` is the single point where the typed enrichment model is
flattened into the stringly-typed dict the results grid and CSV export consume. It
previously dropped several already-computed reliability/coverage/build fields; these
tests pin them so they cannot silently regress again.
"""

from __future__ import annotations

from just_prs.models import EnrichedPRSResult
from prs_ui.mixin import (
    _enriched_to_row_dict,
    ai_links_for_selection,
    clear_cached_trait_charts,
    get_cached_trait_chart,
    overflow_result_rows,
    preview_result_rows,
    store_cached_trait_chart,
    trait_chart_cache_key,
)


def _enriched(**overrides: object) -> EnrichedPRSResult:
    base: dict[str, object] = {
        "pgs_id": "PGS000014",
        "weight_mass_coverage": 0.12,
        "percentile_reliable": False,
        "percentile_caveat": "Only 12% of effect-weight mass matched (C_wt).",
        "z_score": 1.42,
        "reference_mean": 0.5,
        "reference_std": 0.2,
        "reference_panel_ancestry": "EUR",
        "reference_panel": "1000g",
        "detected_genome_build": "GRCh37",
        "build_mismatch": True,
    }
    base.update(overrides)
    return EnrichedPRSResult(**base)  # type: ignore[arg-type]


def test_reported_trait_is_bridged_separately_from_mapped_efo() -> None:
    row = _enriched_to_row_dict(
        _enriched(
            trait="aging rate",
            trait_reported="Facial aging, looking 'older than you are'",
            trait_efo="aging rate",
            trait_efo_id="OBA_0005494",
        )
    )

    assert row["trait_reported"] == "Facial aging, looking 'older than you are'"
    assert row["trait"] == "Facial aging, looking 'older than you are'"
    assert row["trait_efo"] == "aging rate"
    assert row["trait_efo_id"] == "OBA_0005494"


def test_new_reliability_and_build_fields_are_bridged() -> None:
    row = _enriched_to_row_dict(_enriched())

    # Coverage + reliability verdict (F9/F20).
    assert row["weight_mass_coverage"] == 0.12
    assert row["percentile_reliable"] is False
    assert row["percentile_caveat"].startswith("Only 12%")

    # True z-score and reference stats (how the percentile was derived).
    assert row["z_score"] == 1.42
    assert row["reference_mean"] == 0.5
    assert row["reference_std"] == 0.2
    assert row["reference_panel_ancestry"] == "EUR"
    assert row["reference_panel"] == "1000g"

    # VCF-build vs scoring-build mismatch.
    assert row["detected_genome_build"] == "GRCh37"
    assert row["build_mismatch"] is True


def test_reliable_defaults_round_trip() -> None:
    # A clean, reliable, build-matched score keeps the benign defaults.
    row = _enriched_to_row_dict(
        _enriched(
            weight_mass_coverage=0.92,
            percentile_reliable=True,
            percentile_caveat="",
            detected_genome_build=None,
            build_mismatch=False,
        )
    )
    assert row["weight_mass_coverage"] == 0.92
    assert row["percentile_reliable"] is True
    assert row["percentile_caveat"] == ""
    assert row["build_mismatch"] is False
    assert row["detected_genome_build"] is None


def test_ai_links_for_selection_uses_selected_then_first() -> None:
    prs_rows = [
        {"pgs_id": "PGS000001", "ai_ask": '[{"label": "Ask Claude", "url": "https://claude.ai/new?q=one", "copyText": "", "color": "#DA7756", "title": ""}]'},
        {"pgs_id": "PGS000002", "ai_ask": '[{"label": "Ask Claude", "url": "https://claude.ai/new?q=two", "copyText": "", "color": "#DA7756", "title": ""}]'},
    ]
    trait_rows = [
        {"trait": "BMI", "ai_ask": '[{"label": "Ask ChatGPT", "url": "https://chatgpt.com/?q=bmi", "copyText": "", "color": "#10A37F", "title": ""}]'},
    ]

    first = ai_links_for_selection("individual", "", prs_rows, trait_rows)
    assert first[0]["url"].endswith("one")

    selected = ai_links_for_selection("individual", "PGS000002", prs_rows, trait_rows)
    assert selected[0]["url"].endswith("two")

    trait = ai_links_for_selection("grouped", "BMI", prs_rows, trait_rows)
    assert trait[0]["label"] == "Ask ChatGPT"

    assert ai_links_for_selection("individual", "", [], []) == []


def test_trait_chart_cache_round_trips_and_clears() -> None:
    clear_cached_trait_charts()
    key = trait_chart_cache_key("intelligence", "high_moderate", "native")
    assert key == "intelligence|high_moderate|native|ontology"
    reported_key = trait_chart_cache_key(
        "intelligence", "high_moderate", "native", "reported"
    )
    assert reported_key == "intelligence|high_moderate|native|reported"
    assert get_cached_trait_chart(key) is None
    store_cached_trait_chart(key, {"spec": {"mark": "area"}, "html": "<p>ok</p>", "height": "900px"})
    cached = get_cached_trait_chart(key)
    assert cached is not None
    assert cached["html"] == "<p>ok</p>"
    cached["html"] = "mutated"
    assert get_cached_trait_chart(key)["html"] == "<p>ok</p>"
    clear_cached_trait_charts()
    assert get_cached_trait_chart(key) is None


def test_preview_keeps_first_ten_and_hides_the_rest() -> None:
    rows = [{"id": i} for i in range(15)]
    assert [row["id"] for row in preview_result_rows(rows)] == list(range(10))
    assert [row["id"] for row in overflow_result_rows(rows)] == list(range(10, 15))
    assert overflow_result_rows(rows[:8]) == []
