"""Allele-aware genotype matching, multi-allelic ALT and haploid GT — both engines.

Regression tests for the just-dna-lite Paper 1 review (2026-10-10). The old
engines joined on ``(chrom, pos)`` only, so:

* an SNV scoring row matched a VCF indel sharing its anchor base
  (``REF=AT, ALT=A`` against effect allele ``A`` counted a dose);
* an indel row matched a *different* indel at the same position;
* a multi-allelic ALT (``G|T``) was compared as one string, giving dose 0;
* the polars engine crashed on a haploid GT (``"1"``) while DuckDB scored it.

The contract both engines must now honour:

* an SNV row is scored only from an SNV record; the dose is the number of
  called alleles equal to the effect base (multi-allelic ALTs are split, so a
  third allele still counts the REF copy);
* an indel row needs the effect allele and, when known, one other allele among
  the record's alleles;
* a position whose only records fail the check is *absent* for that row
  (``variants_allele_mismatch``), never a silent dose;
* a haploid call (``"1"``: male chrX/chrY, chrM) is one allele and counts
  0 or 1 copies. A man has one X; no tool convention changes that.

The real-data class reads the anton genome (Zenodo 18370498) and a real
PGS Catalog scoring file.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from just_prs.contributions import extract_prs_contributions
from just_prs.models import PRSResult
from just_prs.prs import compute_prs, compute_prs_duckdb


def _genotypes(rows: list[tuple[str, int, str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8, "alt": pl.Utf8, "GT": pl.Utf8},
        orient="row",
    )


def _scoring(rows: list[dict[str, object]]) -> pl.LazyFrame:
    base = {"chr_name": "1", "effect_weight": 1.0, "other_allele": None}
    frame = pl.DataFrame([{**base, **row} for row in rows])
    return frame.with_columns(
        pl.col("chr_name").cast(pl.Utf8),
        pl.col("chr_position").cast(pl.Int64),
        pl.col("effect_allele").cast(pl.Utf8),
        pl.col("other_allele").cast(pl.Utf8),
        pl.col("effect_weight").cast(pl.Float64),
    ).lazy()


def _both_engines(
    tmp_path: Path,
    genotypes: pl.DataFrame,
    scoring: pl.LazyFrame,
    mode: str = "variant_only",
) -> tuple[PRSResult, PRSResult]:
    parquet = tmp_path / "genotypes.parquet"
    genotypes.write_parquet(parquet)
    polars_result = compute_prs(
        vcf_path="__unused__",
        scoring_file=scoring,
        genotypes_lf=genotypes.lazy(),
        genotype_input_mode=mode,
    )
    duckdb_result = compute_prs_duckdb(
        vcf_path="__unused__",
        scoring_file=scoring,
        genotypes_parquet=parquet,
        genotype_input_mode=mode,
    )
    return polars_result, duckdb_result


def _assert_same(a: PRSResult, b: PRSResult) -> None:
    for field in (
        "score",
        "variants_matched",
        "variants_observed",
        "variants_no_call",
        "variants_allele_mismatch",
        "variants_unscorable_absent",
        "variants_assumed_hom_ref",
    ):
        assert getattr(a, field) == pytest.approx(getattr(b, field)), field


class TestHaploidGenotype:
    def test_haploid_call_is_one_copy_in_both_engines(self, tmp_path: Path) -> None:
        geno = _genotypes([
            ("X", 100, "A", "G", "1"),
            ("MT", 200, "C", "T", "0"),
            ("Y", 300, "G", "T,C", "2"),
        ])
        scoring = _scoring([
            {"chr_name": "X", "chr_position": 100, "effect_allele": "G", "other_allele": "A", "effect_weight": 1.0},
            {"chr_name": "MT", "chr_position": 200, "effect_allele": "C", "other_allele": "T", "effect_weight": 10.0},
            {"chr_name": "Y", "chr_position": 300, "effect_allele": "C", "other_allele": "G", "effect_weight": 100.0},
        ])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == pytest.approx(111.0)
        assert polars_result.variants_no_call == 0
        _assert_same(polars_result, duckdb_result)

    def test_haploid_reference_call_is_zero_copies(self, tmp_path: Path) -> None:
        geno = _genotypes([("X", 100, "A", "G", "0")])
        scoring = _scoring([{"chr_name": "X", "chr_position": 100, "effect_allele": "G", "other_allele": "A"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == 0.0
        assert polars_result.variants_observed == 1
        _assert_same(polars_result, duckdb_result)

    def test_haploid_scores_half_of_diploid_hom_alt(self, tmp_path: Path) -> None:
        """The caller's ploidy is what is scored: ``1`` is one copy, ``1/1`` two."""
        scoring = _scoring([{"chr_name": "X", "chr_position": 100, "effect_allele": "G", "other_allele": "A"}])
        haploid = _both_engines(tmp_path, _genotypes([("X", 100, "A", "G", "1")]), scoring)
        diploid = _both_engines(tmp_path, _genotypes([("X", 100, "A", "G", "1/1")]), scoring)
        assert [r.score for r in haploid] == [1.0, 1.0]
        assert [r.score for r in diploid] == [2.0, 2.0]

    def test_normalized_genotype_keeps_one_allele(self) -> None:
        from just_prs.normalize import genotype_expr

        frame = _genotypes([
            ("X", 1, "A", "G", "1"),
            ("X", 2, "A", "G", "0"),
            ("MT", 3, "C", "T|G", "2"),
            ("1", 4, "A", "G", "0|1"),
            ("1", 5, "A", "G", "."),
            ("1", 6, "A", "G", "./1"),
        ])
        assert frame.select(genotype_expr())["genotype"].to_list() == [
            ["G"], ["A"], ["G"], ["A", "G"], [], [],
        ]

    def test_half_call_is_still_no_call(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 100, "A", "G", "./1")])
        scoring = _scoring([{"chr_name": "1", "chr_position": 100, "effect_allele": "G", "other_allele": "A"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.variants_no_call == 1
        assert polars_result.score == 0.0
        _assert_same(polars_result, duckdb_result)

    def test_haploid_missing_is_no_call(self, tmp_path: Path) -> None:
        geno = _genotypes([("X", 100, "A", "G", ".")])
        scoring = _scoring([{"chr_name": "X", "chr_position": 100, "effect_allele": "G", "other_allele": "A"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.variants_no_call == 1
        assert polars_result.score == 0.0
        _assert_same(polars_result, duckdb_result)


class TestMultiAllelic:
    @pytest.mark.parametrize(
        ("effect", "gt", "expected"),
        [("G", "1/2", 1.0), ("T", "1/2", 1.0), ("A", "1/2", 0.0), ("T", "2/2", 2.0), ("A", "0/2", 1.0)],
    )
    def test_split_alt_gives_dose_for_matching_index(
        self, tmp_path: Path, effect: str, gt: str, expected: float
    ) -> None:
        geno = _genotypes([("1", 15274, "A", "G|T", gt)])
        scoring = _scoring([{"chr_position": 15274, "effect_allele": effect}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == pytest.approx(expected)
        _assert_same(polars_result, duckdb_result)

    def test_comma_separated_alt_is_split_too(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 15274, "A", "G,T", "1/2")])
        scoring = _scoring([{"chr_position": 15274, "effect_allele": "T", "other_allele": "A"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == pytest.approx(1.0)
        _assert_same(polars_result, duckdb_result)


class TestAlleleAwareJoin:
    def test_snv_row_does_not_match_deletion_sharing_anchor(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 1000, "AT", "A", "0/1")])
        scoring = _scoring([{"chr_position": 1000, "effect_allele": "A", "other_allele": "G"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == 0.0
        assert polars_result.variants_observed == 0
        assert polars_result.variants_allele_mismatch == 1
        assert polars_result.variants_unscorable_absent == 1
        _assert_same(polars_result, duckdb_result)

    def test_snv_row_without_other_allele_does_not_match_insertion(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 1000, "A", "AT", "1/1")])
        scoring = _scoring([{"chr_position": 1000, "effect_allele": "A"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == 0.0
        assert polars_result.variants_allele_mismatch == 1
        _assert_same(polars_result, duckdb_result)

    def test_indel_row_does_not_match_a_different_indel(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 2000, "CAAA", "C", "1/1")])
        scoring = _scoring([{"chr_position": 2000, "effect_allele": "C", "other_allele": "CAAAA"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == 0.0
        assert polars_result.variants_allele_mismatch == 1
        _assert_same(polars_result, duckdb_result)

    def test_indel_row_matches_the_same_indel(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 2000, "CA", "C", "0/1")])
        scoring = _scoring([{"chr_position": 2000, "effect_allele": "C", "other_allele": "CA"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == pytest.approx(1.0)
        assert polars_result.variants_allele_mismatch == 0
        _assert_same(polars_result, duckdb_result)

    def test_inferred_other_allele_candidates_are_used(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 2000, "CA", "C", "1/1")])
        scoring = _scoring([{"chr_position": 2000, "effect_allele": "C"}]).with_columns(
            pl.lit("CAA/CA").alias("hm_inferOtherAllele")
        )
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == pytest.approx(2.0)
        _assert_same(polars_result, duckdb_result)

    def test_third_allele_snv_still_counts_reference_copy(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 3000, "T", "G", "0/1")])
        scoring = _scoring([{"chr_position": 3000, "effect_allele": "T", "other_allele": "C"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == pytest.approx(1.0)
        _assert_same(polars_result, duckdb_result)

    def test_snv_allele_of_a_mixed_multiallelic_record_counts(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 6000, "G", "A|AA", "1/2")])
        for effect, expected in (("A", 1.0), ("G", 0.0)):
            scoring = _scoring([{"chr_position": 6000, "effect_allele": effect, "other_allele": "GA".replace(effect, "")}]).with_columns(
                pl.lit("G").alias("reference_allele")
            )
            polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
            assert polars_result.score == pytest.approx(expected), effect
            assert polars_result.variants_observed == 1
            _assert_same(polars_result, duckdb_result)

    def test_snv_and_indel_records_at_one_position_score_once(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 4000, "GT", "G", "0/1"), ("1", 4000, "G", "A", "0/1")])
        scoring = _scoring([{"chr_position": 4000, "effect_allele": "A", "other_allele": "G"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == pytest.approx(1.0)
        assert polars_result.variants_observed == 1
        assert polars_result.variants_allele_mismatch == 0
        _assert_same(polars_result, duckdb_result)

    def test_array_homozygous_encoding_still_scores(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 5000, "A", "A", "1/1")])
        scoring = _scoring([{"chr_position": 5000, "effect_allele": "A", "other_allele": "G"}])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == pytest.approx(2.0)
        _assert_same(polars_result, duckdb_result)

    def test_mismatch_falls_through_to_hom_ref_when_reference_known(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 1000, "AT", "A", "0/1")])
        scoring = _scoring([{"chr_position": 1000, "effect_allele": "A", "other_allele": "G"}]).with_columns(
            pl.lit("A").alias("reference_allele")
        )
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring)
        assert polars_result.score == pytest.approx(2.0)
        assert polars_result.variants_assumed_hom_ref == 1
        assert polars_result.variants_allele_mismatch == 1
        _assert_same(polars_result, duckdb_result)

    def test_present_only_mode_drops_mismatched_rows(self, tmp_path: Path) -> None:
        geno = _genotypes([("1", 1000, "AT", "A", "0/1"), ("1", 3000, "T", "G", "0/1")])
        scoring = _scoring([
            {"chr_position": 1000, "effect_allele": "A", "other_allele": "G"},
            {"chr_position": 3000, "effect_allele": "G", "other_allele": "T"},
        ])
        polars_result, duckdb_result = _both_engines(tmp_path, geno, scoring, mode="plink_present_only")
        assert polars_result.score == pytest.approx(1.0)
        assert polars_result.variants_observed == 1
        assert polars_result.variants_allele_mismatch == 1
        _assert_same(polars_result, duckdb_result)


def _anton_parquet() -> Path:
    from just_prs.normalize import VcfFilterConfig, normalize_vcf
    from just_prs.scoring import resolve_cache_dir

    test_dir = resolve_cache_dir() / "test-data"
    vcf = test_dir / "antonkulaga.vcf"
    if not vcf.exists():
        pytest.skip(f"anton VCF not cached at {vcf}")
    out = test_dir / "antonkulaga_pass_normalized.parquet"
    if not out.exists() or out.stat().st_mtime < vcf.stat().st_mtime:
        normalize_vcf(vcf, out, config=VcfFilterConfig(pass_filters=["PASS", "."]))
    return out


@pytest.fixture(scope="module")
def anton_parquet() -> Path:
    return _anton_parquet()


class TestRealGenome:
    """PGS000012 has no ``other_allele``; PGS000018 does. Both hit real indels."""

    @pytest.mark.parametrize("pgs_id", ["PGS000012", "PGS000018"])
    def test_no_observed_snv_row_sits_on_an_indel_record(
        self, tmp_path: Path, anton_parquet: Path, pgs_id: str
    ) -> None:
        contributions = extract_prs_contributions(
            vcf_path="__unused__",
            scoring_file=pgs_id,
            pgs_id=pgs_id,
            genotypes_parquet=anton_parquet,
            output=tmp_path / f"{pgs_id}.parquet",
        )
        rows = pl.read_parquet(contributions.output_path)
        observed = rows.filter(pl.col("status") == "observed")
        alts = pl.col("sample_alt").str.replace_all(",", "|").str.split("|")
        snv_on_indel = observed.filter(
            (pl.col("effect_allele").str.len_chars() == 1)
            & (pl.col("other_allele").is_null() | (pl.col("other_allele").str.len_chars() == 1))
            & (
                (pl.col("sample_ref").str.len_chars() != 1)
                | ~alts.list.eval(pl.element().str.len_chars() == 1).list.any()
            )
        )
        assert snv_on_indel.height == 0
        assert rows.filter(pl.col("status") == "allele_mismatch").height > 0

        polars_result = compute_prs(
            vcf_path="__unused__",
            scoring_file=pgs_id,
            pgs_id=pgs_id,
            genotypes_lf=pl.scan_parquet(anton_parquet),
        )
        duckdb_result = compute_prs_duckdb(
            vcf_path="__unused__",
            scoring_file=pgs_id,
            pgs_id=pgs_id,
            genotypes_parquet=anton_parquet,
        )
        _assert_same(polars_result, duckdb_result)
        assert contributions.score == pytest.approx(duckdb_result.score)
        assert polars_result.variants_allele_mismatch == rows.filter(
            pl.col("status") == "allele_mismatch"
        ).height

    def test_multiallelic_records_contribute(self, anton_parquet: Path) -> None:
        """A real multi-allelic record scores its matching ALT index."""
        record = (
            pl.scan_parquet(anton_parquet)
            .filter(pl.col("GT") == "1/2", pl.col("ref").str.len_chars() == 1)
            .filter(pl.col("alt").str.split("|").list.eval(pl.element().str.len_chars() == 1).list.all())
            .select("chrom", "pos", "ref", "alt", "GT")
            .head(1)
            .collect()
            .row(0, named=True)
        )
        second_alt = record["alt"].split("|")[1]
        scoring = _scoring([{
            "chr_name": record["chrom"],
            "chr_position": record["pos"],
            "effect_allele": second_alt,
            "other_allele": record["ref"],
        }])
        result = compute_prs_duckdb(
            vcf_path="__unused__", scoring_file=scoring, genotypes_parquet=anton_parquet
        )
        assert result.score == pytest.approx(1.0)
