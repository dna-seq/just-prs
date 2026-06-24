"""Regression tests for the EnrichedPRSResult -> grid-row dict bridge.

``_enriched_to_row_dict`` is the single point where the typed enrichment model is
flattened into the stringly-typed dict the results grid and CSV export consume. It
previously dropped several already-computed reliability/coverage/build fields; these
tests pin them so they cannot silently regress again.
"""

from __future__ import annotations

from just_prs.models import EnrichedPRSResult
from prs_ui.mixin import _enriched_to_row_dict


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
