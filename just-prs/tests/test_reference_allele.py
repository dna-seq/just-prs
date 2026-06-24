"""Network-free tests for reference-allele resolution (just_prs.reference_allele).

Builds a tiny synthetic FASTA (+ faidx) and a synthetic pvar parquet, then
asserts the two-tier resolution: panel REF wins over FASTA, the FASTA tier
resolves single-base SNV positions only, indel/absent positions stay
``unresolved``, and a build/contig mismatch fails loudly rather than injecting a
wrong REF base.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("pysam")
pytest.importorskip("duckdb")

import pysam

from just_prs import reference_allele
from just_prs.reference import ReferencePanelError
from just_prs.reference_allele import resolve_reference_alleles

# Contig "1": 50 bases, deterministic so we can assert exact REF bases.
CONTIG1 = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC"
assert len(CONTIG1) == 50
TEST_BUILD = "TESTBUILD"


@pytest.fixture
def fasta_path(tmp_path: Path) -> Path:
    fa = tmp_path / "tiny.fa"
    fa.write_text(f">1\n{CONTIG1}\n>2\nGGGGGGGGGG\n")
    pysam.faidx(str(fa))  # writes tiny.fa.fai
    assert fa.with_name("tiny.fa.fai").exists()
    return fa


@pytest.fixture
def pvar_parquet(tmp_path: Path) -> Path:
    # Panel covers (1, 10) with an unambiguous REF, and a multiallelic (1, 20)
    # split into two rows sharing the same REF.
    df = pl.DataFrame(
        {
            "variant_idx": [0, 1, 2],
            "chrom": ["1", "1", "1"],
            "POS": [10, 20, 20],
            "ID": ["1:10:G:A", "1:20:T:C", "1:20:T:G"],
            "REF": ["G", "T", "T"],
            "ALT": ["A", "C", "G"],
        }
    )
    p = tmp_path / "panel.pvar.parquet"
    df.write_parquet(p)
    return p


@pytest.fixture(autouse=True)
def _register_test_build(monkeypatch):
    # Tell the build guard that TESTBUILD's chr1 is 50 bp (our synthetic length).
    monkeypatch.setitem(reference_allele.REFERENCE_FASTA_CHR1_LENGTH, TEST_BUILD, len(CONTIG1))


def test_panel_tier_wins_over_fasta(fasta_path: Path, pvar_parquet: Path):
    # (1,10) is in the panel (REF=G) AND in the FASTA (base at index 9 = 'C').
    # Panel must win.
    positions = pl.DataFrame({"chrom": ["1"], "pos": [10]})
    out = resolve_reference_alleles(
        positions, TEST_BUILD, panel_pvar_path=pvar_parquet, fasta_path=fasta_path
    )
    row = out.row(0, named=True)
    assert row["ref_source"] == "panel"
    assert row["ref"] == "G"
    # sanity: the FASTA base there is different, proving panel actually won
    assert CONTIG1[9] != "G"


def test_multiallelic_panel_position_resolves_single_ref(fasta_path, pvar_parquet):
    out = resolve_reference_alleles(
        pl.DataFrame({"chrom": ["1"], "pos": [20]}),
        TEST_BUILD,
        panel_pvar_path=pvar_parquet,
        fasta_path=fasta_path,
    )
    row = out.row(0, named=True)
    assert row["ref_source"] == "panel"
    assert row["ref"] == "T"  # both rows share REF=T → unambiguous


def test_fasta_tier_resolves_snv(fasta_path, pvar_parquet):
    # pos 5 (1-based) is not in the panel → FASTA tier; expect CONTIG1[4].
    out = resolve_reference_alleles(
        pl.DataFrame({"chrom": ["1"], "pos": [5], "snv_only": [True]}),
        TEST_BUILD,
        panel_pvar_path=pvar_parquet,
        fasta_path=fasta_path,
    )
    row = out.row(0, named=True)
    assert row["ref_source"] == "fasta"
    assert row["ref"] == CONTIG1[4]


def test_indel_position_stays_unresolved(fasta_path, pvar_parquet):
    # snv_only=False and not in panel → FASTA must NOT fill a single base.
    out = resolve_reference_alleles(
        pl.DataFrame({"chrom": ["1"], "pos": [7], "snv_only": [False]}),
        TEST_BUILD,
        panel_pvar_path=pvar_parquet,
        fasta_path=fasta_path,
    )
    row = out.row(0, named=True)
    assert row["ref_source"] == "unresolved"
    assert row["ref"] is None


def test_off_contig_position_unresolved(fasta_path, pvar_parquet):
    # Beyond contig length and on a missing contig → both unresolved, no crash.
    out = resolve_reference_alleles(
        pl.DataFrame({"chrom": ["1", "9"], "pos": [999, 5], "snv_only": [True, True]}),
        TEST_BUILD,
        panel_pvar_path=pvar_parquet,
        fasta_path=fasta_path,
    ).sort("chrom", "pos")
    assert set(out["ref_source"].to_list()) == {"unresolved"}


def test_conflicting_snv_flag_is_conservative(fasta_path):
    # Same position appears as SNV and non-SNV → treated as non-SNV (min) →
    # FASTA tier skips it → unresolved (no panel given).
    positions = pl.DataFrame({"chrom": ["1", "1"], "pos": [5, 5], "snv_only": [True, False]})
    out = resolve_reference_alleles(positions, TEST_BUILD, fasta_path=fasta_path)
    assert out.height == 1
    assert out.row(0, named=True)["ref_source"] == "unresolved"


def test_panel_only_no_fasta(pvar_parquet):
    out = resolve_reference_alleles(
        pl.DataFrame({"chrom": ["1", "1"], "pos": [10, 5]}),
        TEST_BUILD,
        panel_pvar_path=pvar_parquet,
        fasta_path=None,
    ).sort("pos")
    by_pos = {r["pos"]: r for r in out.iter_rows(named=True)}
    assert by_pos[10]["ref_source"] == "panel" and by_pos[10]["ref"] == "G"
    assert by_pos[5]["ref_source"] == "unresolved"  # no FASTA tier available


def test_build_mismatch_fails_loudly(fasta_path, pvar_parquet, monkeypatch):
    # Pretend TESTBUILD's chr1 should be a different length → guard must raise
    # rather than return a wrong base for the SNV that needs the FASTA tier.
    monkeypatch.setitem(reference_allele.REFERENCE_FASTA_CHR1_LENGTH, TEST_BUILD, 999_999)
    with pytest.raises(ReferencePanelError, match="chr1 length"):
        resolve_reference_alleles(
            pl.DataFrame({"chrom": ["1"], "pos": [5], "snv_only": [True]}),
            TEST_BUILD,
            panel_pvar_path=pvar_parquet,
            fasta_path=fasta_path,
        )


def test_ref_resolution_targets_unions_missing_positions(tmp_path: Path):
    """The pipeline target builder keeps positions whose reference_allele is missing
    in at least one scoring file, with a conservative snv_only flag."""
    from prs_pipeline.assets import _ref_resolution_targets

    pl.DataFrame({
        "hm_chr": ["1", "1", "1"],
        "hm_pos": [100, 200, 300],
        "effect_allele": ["G", "T", "AT"],   # 300 is an indel-shaped effect allele
        "other_allele": ["A", "C", "A"],
        "reference_allele": ["A", None, None],  # 100 present; 200 & 300 missing
    }).write_parquet(tmp_path / "PGS000001_hmPOS_GRCh38.parquet")
    pl.DataFrame({
        "hm_chr": ["1", "2"],
        "hm_pos": [200, 500],
        "effect_allele": ["T", "G"],
        "other_allele": ["C", "A"],
        "reference_allele": ["T", None],  # 200 present here, but missing in file 1
    }).write_parquet(tmp_path / "PGS000002_hmPOS_GRCh38.parquet")

    targets = _ref_resolution_targets(tmp_path, "GRCh38").sort("chrom", "pos")
    got = {(r["chrom"], r["pos"]): r["snv_only"] for r in targets.iter_rows(named=True)}
    assert set(got) == {("1", 200), ("1", 300), ("2", 500)}  # 100 excluded (ref present)
    assert got[("1", 200)] is True   # single-base across both files
    assert got[("1", 300)] is False  # indel-shaped effect allele → not SNV
    assert got[("2", 500)] is True


def test_empty_positions_returns_empty(fasta_path):
    out = resolve_reference_alleles(
        pl.DataFrame({"chrom": [], "pos": []}, schema={"chrom": pl.Utf8, "pos": pl.Int64}),
        TEST_BUILD,
        fasta_path=fasta_path,
    )
    assert out.height == 0
    assert out.columns == ["genome_build", "chrom", "pos", "ref", "ref_source"]
