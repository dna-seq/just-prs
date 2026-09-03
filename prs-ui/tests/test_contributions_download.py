"""Extract-only contribution download helpers. Never applied to PRS compute."""

from __future__ import annotations

from prs_ui.mixin import (
    contribution_top_n_choice,
    contributions_download_stem,
    extract_top_n_or_none,
    parse_contribution_top_n,
    resolve_extract_pgs_id,
)


def test_parse_contribution_top_n_all_is_zero() -> None:
    assert parse_contribution_top_n("All") == 0
    assert parse_contribution_top_n(["All"]) == 0
    assert parse_contribution_top_n(2000) == 2000
    assert extract_top_n_or_none(0) is None
    assert extract_top_n_or_none(500) == 500
    assert contribution_top_n_choice(0) == "All"
    assert contribution_top_n_choice(500) == "500"


def test_resolve_extract_pgs_id_prefers_clicked_result() -> None:
    assert resolve_extract_pgs_id("PGS000002", ["PGS000001"]) == "PGS000002"
    assert resolve_extract_pgs_id("", ["PGS000001", "PGS000003"]) == "PGS000001"
    assert resolve_extract_pgs_id("", []) == ""


def test_contributions_download_stem_includes_top_n() -> None:
    assert contributions_download_stem("PGS000001", "", 500) == (
        "pgs000001_top500_contributions"
    )
    assert contributions_download_stem("PGS000001", "Livia", 0) == (
        "pgs000001_livia_all_contributions"
    )
