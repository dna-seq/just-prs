"""Coordinate liftover tests (GRCh37 <-> GRCh38).

Uses the real UCSC chain files (auto-downloaded, ~3 MB each) and dbSNP/UCSC-
verified coordinate pairs as ground truth, per the project's real-data policy.
"""

from __future__ import annotations

import pytest

from just_prs.liftover import (
    LiftOver,
    LiftoverConfigurationError,
    chain_file_path,
    get_liftover,
    liftover_preflight,
    normalize_build,
)

# Known SNPs with curated GRCh38/GRCh37 coordinates (dbSNP + UCSC verified).
# rs429358 (APOE):  GRCh38 chr19:44908684  GRCh37 chr19:45411941
# rs7412   (APOE):  GRCh38 chr19:44908822  GRCh37 chr19:45412079
# rs1815739 (ACTN3): GRCh38 chr11:66560624 GRCh37 chr11:66328095
_GRCH38_GRCH37_PAIRS = [
    ("chr19", 44908684, "chr19", 45411941),
    ("chr19", 44908822, "chr19", 45412079),
    ("chr11", 66560624, "chr11", 66328095),
]


# --- pure config (no network) ---------------------------------------------


def test_normalize_build_aliases() -> None:
    for alias in ("GRCh38", "grch38", "hg38", "38", "b38"):
        assert normalize_build(alias) == "GRCh38"
    for alias in ("GRCh37", "hg19", "37", "b37"):
        assert normalize_build(alias) == "GRCh37"


def test_normalize_build_rejects_unknown() -> None:
    for bad in ("CHM13", "", "GRCh36", "hg18"):
        with pytest.raises(ValueError):
            normalize_build(bad)


def test_chain_file_path_unknown_pair_raises() -> None:
    with pytest.raises(LiftoverConfigurationError):
        chain_file_path("GRCh38", "GRCh38")


def test_identical_builds_rejected() -> None:
    with pytest.raises(ValueError):
        LiftOver("GRCh38", "GRCh38")


def test_preflight_same_build_not_required() -> None:
    pf = liftover_preflight("GRCh38", "GRCh38")
    assert pf["status"] == "not_required"
    assert pf["tool_will_work"] is True


def test_preflight_reports_resource_state() -> None:
    pf = liftover_preflight("GRCh38", "GRCh37")
    # pyliftover is installed in the test env, so it's at worst a chain download.
    assert pf["status"] in {"available", "requires_chain_download"}
    assert pf["tool_will_work"] is True
    assert pf["pyliftover_installed"] is True
    assert pf["chain_url"].endswith("hg38ToHg19.over.chain.gz")


# --- real liftover (auto-downloads chains) --------------------------------


def test_grch38_to_grch37_known_snps() -> None:
    lifter = get_liftover("GRCh38", "GRCh37")
    for g38_chrom, g38_pos, g37_chrom, g37_pos in _GRCH38_GRCH37_PAIRS:
        assert lifter.lift_position(g38_chrom, g38_pos) == (g37_chrom, g37_pos)


def test_grch37_to_grch38_round_trip() -> None:
    lifter = get_liftover("GRCh37", "GRCh38")
    for g38_chrom, g38_pos, g37_chrom, g37_pos in _GRCH38_GRCH37_PAIRS:
        assert lifter.lift_position(g37_chrom, g37_pos) == (g38_chrom, g38_pos)


def test_accepts_unprefixed_chrom_and_preserves_style() -> None:
    lifter = get_liftover("GRCh38", "GRCh37")
    assert lifter.lift_position("19", 44908684) == ("19", 45411941)


def test_unmapped_position_returns_none() -> None:
    lifter = get_liftover("GRCh38", "GRCh37")
    assert lifter.lift_position("chr1", 10**12) is None


def test_lift_records_splits_lifted_and_dropped() -> None:
    lifter = get_liftover("GRCh38", "GRCh37")
    records = [
        {"chrom": "chr19", "pos": 44908684, "rsid": "rs429358"},
        {"chrom": "chr1", "pos": 10**12, "rsid": "rs_unmappable"},
        {"chrom": "chrX", "pos": None, "rsid": "rs_missing_pos"},
        {"chrom": "chr19", "pos": "not-an-int", "rsid": "rs_bad_pos"},
    ]
    result = lifter.lift_records(records)
    assert len(result.lifted) == 1
    assert result.lifted[0]["pos"] == 45411941
    assert result.lifted[0]["rsid"] == "rs429358"
    reasons = sorted(r["liftover_reason"] for r in result.dropped)
    assert reasons == ["invalid_position", "missing_coordinates", "unmapped"]


def test_cached_singleton_returns_same_instance() -> None:
    assert get_liftover("GRCh38", "GRCh37") is get_liftover("GRCh38", "GRCh37")
