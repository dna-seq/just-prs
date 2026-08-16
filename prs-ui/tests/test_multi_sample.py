"""Tests for the multi-sample comparison helpers in the PRS compute mixin.

The UI mirrors the CLI's ``--vcf Label=path`` comparison: every sample gets a
stable color from the shared ``just_prs.viz.SAMPLE_COLORS`` palette, result
rows are keyed by ``(pgs_id, sample)``, and rows group per sample for the
trait-summary aggregation.  These helpers are pure functions, so the tests pin
the contract without spinning up Reflex state.
"""

from __future__ import annotations

from just_prs.viz import SAMPLE_COLORS
from prs_ui.mixin import (
    _ancestry_chip_text,
    _group_rows_by_sample,
    _merge_prs_results,
    _ordered_sample_labels,
    _result_sample,
    majority_detected_superpopulation,
    sample_color,
    sample_label_from_path,
)


def test_sample_color_matches_cli_palette_and_wraps() -> None:
    assert [sample_color(i) for i in range(len(SAMPLE_COLORS))] == list(SAMPLE_COLORS)
    # More samples than palette entries wrap around instead of failing.
    assert sample_color(len(SAMPLE_COLORS)) == SAMPLE_COLORS[0]


def test_sample_label_from_path_strips_genotype_suffixes() -> None:
    assert sample_label_from_path("/data/mom.vcf.gz") == "mom"
    assert sample_label_from_path("/data/son1.vcf") == "son1"
    assert sample_label_from_path("/x/SIMHIFQTILQ.hard-filtered.vcf.gz") == "SIMHIFQTILQ"
    assert sample_label_from_path("/cache/normalized/dad.parquet") == "dad"


def test_ancestry_chip_text_formats_and_abstains() -> None:
    full = {
        "ancestry": "EUR",
        "ancestry_confidence": 0.97,
        "fine_population": "CEU",
        "fine_confidence": 0.55,
    }
    assert _ancestry_chip_text(full) == "EUR 97% · CEU 55%"
    # Fine population without its own posterior — cohort code alone.
    no_fine_conf = {"ancestry": "EUR", "ancestry_confidence": 0.97, "fine_population": "CEU"}
    assert _ancestry_chip_text(no_fine_conf) == "EUR 97% · CEU"
    # No fine population — superpop + confidence only.
    assert _ancestry_chip_text({"ancestry": "AFR", "ancestry_confidence": 1.0}) == "AFR 100%"
    # No confidence recorded — bare superpop.
    assert _ancestry_chip_text({"ancestry": "SAS"}) == "SAS"
    # UNKNOWN / missing ancestry never renders a chip.
    assert _ancestry_chip_text({"ancestry": "UNKNOWN", "ancestry_confidence": 0.4}) == ""
    assert _ancestry_chip_text({}) == ""


def test_majority_detected_superpopulation_picks_mode_and_ignores_unknown() -> None:
    assert majority_detected_superpopulation([]) == ""
    assert majority_detected_superpopulation([{"ancestry": "UNKNOWN"}]) == ""
    assert majority_detected_superpopulation([{"ancestry": "EUR"}]) == "EUR"
    # Majority, not first-seen.
    mixed = [
        {"ancestry": "AFR"},
        {"ancestry": "EUR"},
        {"ancestry": "EUR"},
        {"ancestry": "UNKNOWN"},
    ]
    assert majority_detected_superpopulation(mixed) == "EUR"
    # Tie keeps the earliest sample's population.
    tied = [{"ancestry": "SAS"}, {"ancestry": "EAS"}]
    assert majority_detected_superpopulation(tied) == "SAS"
    # Lowercase codes still count.
    assert majority_detected_superpopulation([{"ancestry": "eur"}]) == "EUR"


def test_merge_prs_results_is_sample_aware() -> None:
    existing = [
        {"pgs_id": "PGS000001", "sample": "Mom", "score": 1.0},
        {"pgs_id": "PGS000001", "sample": "Dad", "score": 2.0},
        {"pgs_id": "PGS000002", "sample": "Mom", "score": 3.0},
    ]
    new_rows = [{"pgs_id": "PGS000001", "sample": "Mom", "score": 9.0}]

    merged = _merge_prs_results(existing, new_rows)

    # Mom's PGS000001 was replaced; Dad's row for the same PGS ID survived.
    keys = {(r["pgs_id"], r["sample"]): r["score"] for r in merged}
    assert keys == {
        ("PGS000001", "Mom"): 9.0,
        ("PGS000001", "Dad"): 2.0,
        ("PGS000002", "Mom"): 3.0,
    }
    # New rows lead the list.
    assert merged[0]["score"] == 9.0


def test_group_rows_by_sample_preserves_order() -> None:
    rows = [
        {"pgs_id": "A", "sample": "Mom"},
        {"pgs_id": "A", "sample": "Dad"},
        {"pgs_id": "B", "sample": "Mom"},
    ]
    grouped = _group_rows_by_sample(rows)
    assert list(grouped) == ["Mom", "Dad"]
    assert [r["pgs_id"] for r in grouped["Mom"]] == ["A", "B"]
    assert _ordered_sample_labels(rows) == ["Mom", "Dad"]


def test_result_sample_empty_for_single_sample_rows() -> None:
    assert _result_sample({"pgs_id": "A"}) == ""
    assert _result_sample({"pgs_id": "A", "sample": None}) == ""
    assert _result_sample({"pgs_id": "A", "sample": "Son1"}) == "Son1"
