"""Sex-chromosome karyotype: inference and copy-aware scoring in both engines.

A man has one X. DeepVariant 1.1.0 (the anton genome) writes his non-PAR X and Y
diploid: 99,903 ``1/1`` and 4,818 ``0/1`` on non-PAR X, no haploid GT. The
karyotype, inferred from the pattern of those calls, decides how many copies a
region has, so a diploid-written ``1/1`` on a man's X counts once.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from just_prs.contributions import extract_prs_contributions
from just_prs.karyotype import (
    Karyotype,
    KaryotypeSource,
    XChromosomeCoding,
    infer_karyotype,
    infer_karyotype_from_file,
)
from just_prs.models import PRSResult
from just_prs.prs import compute_prs, compute_prs_duckdb

# GRCh38 non-PAR X / Y positions and a PAR1 position.
X_NON_PAR = 50_000_000
Y_NON_PAR = 10_000_000
X_PAR1 = 1_000_000


def _sample(x_het: float, x_n: int = 1000, y_n: int = 0, x_haploid: bool = False) -> pl.DataFrame:
    """A synthetic genome: 10,000 autosomal calls at 60% het, plus X and Y calls."""
    rows: list[tuple[str, int, str, str, str]] = []
    for i in range(10_000):
        rows.append(("1", 1000 + i, "A", "G", "0/1" if i % 10 < 6 else "1/1"))
    n_het = int(round(x_het * x_n))
    for i in range(x_n):
        gt = "1" if x_haploid else ("0/1" if i < n_het else "1/1")
        rows.append(("X", 3_000_000 + i, "C", "T", gt))
    for i in range(y_n):
        rows.append(("Y", 3_000_000 + i, "G", "A", "1/1"))
    return pl.DataFrame(rows, schema={"chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8, "alt": pl.Utf8, "GT": pl.Utf8}, orient="row")


class TestInference:
    @pytest.mark.parametrize(
        ("x_het", "y_n", "haploid", "expected"),
        [
            (0.58, 0, False, Karyotype.XX),
            (0.05, 60, False, Karyotype.XY),
            (0.58, 60, False, Karyotype.XXY),
            (0.05, 0, False, Karyotype.X0),
            (0.0, 60, True, Karyotype.XY),
            (0.2, 60, False, Karyotype.UNKNOWN),
            (0.05, 5, False, Karyotype.UNKNOWN),
        ],
    )
    def test_bands(self, x_het: float, y_n: int, haploid: bool, expected: Karyotype) -> None:
        call = infer_karyotype(_sample(x_het, y_n=y_n, x_haploid=haploid).lazy())
        assert call.karyotype == expected
        assert call.source == KaryotypeSource.INFERRED

    def test_too_few_x_records_is_unknown(self) -> None:
        assert infer_karyotype(_sample(0.05, x_n=50, y_n=60).lazy()).karyotype == Karyotype.UNKNOWN

    def test_par_and_filtered_records_do_not_count(self) -> None:
        sample = _sample(0.05, y_n=60)
        # 2,000 PAR hets and 2,000 filtered non-PAR hets would make it XXY if counted.
        par = pl.DataFrame({
            "chrom": ["X"] * 2000, "pos": list(range(X_PAR1, X_PAR1 + 2000)),
            "ref": ["C"] * 2000, "alt": ["T"] * 2000, "GT": ["0/1"] * 2000,
        })
        filtered = par.with_columns(pl.col("pos") + 60_000_000)
        frame = pl.concat([
            sample.with_columns(pl.lit("PASS").alias("filter")),
            par.with_columns(pl.lit("PASS").alias("filter")),
            filtered.with_columns(pl.lit("RefCall").alias("filter")),
        ])
        assert infer_karyotype(frame.lazy()).karyotype == Karyotype.XY


def _both(
    tmp_path: Path,
    genotypes: pl.DataFrame,
    scoring: pl.LazyFrame,
    karyotype: Karyotype | str | None,
    coding: XChromosomeCoding = XChromosomeCoding.COPIES,
) -> tuple[PRSResult, PRSResult]:
    parquet = tmp_path / "g.parquet"
    genotypes.write_parquet(parquet)
    kwargs = dict(vcf_path="", scoring_file=scoring, genotype_input_mode="variant_only",
                  sample_karyotype=karyotype, x_coding=coding)
    return (
        compute_prs(genotypes_lf=genotypes.lazy(), **kwargs),
        compute_prs_duckdb(genotypes_parquet=parquet, **kwargs),
    )


def _scoring(rows: list[tuple[str, int, str, str, float, str | None]]) -> pl.LazyFrame:
    return pl.DataFrame(
        rows,
        schema={"chr_name": pl.Utf8, "chr_position": pl.Int64, "effect_allele": pl.Utf8,
                "other_allele": pl.Utf8, "effect_weight": pl.Float64, "reference_allele": pl.Utf8},
        orient="row",
    ).lazy()


# One row per case; weights are powers of ten so the score shows which rows counted.
_GENOS = pl.DataFrame(
    [
        ("X", X_NON_PAR, "C", "T", "1/1"),       # hom on non-PAR X
        ("X", X_NON_PAR + 1, "C", "T", "0/1"),   # het on non-PAR X
        ("X", X_PAR1, "C", "T", "1/1"),          # PAR: two copies in anyone
        ("Y", Y_NON_PAR, "G", "A", "1/1"),       # Y
        ("MT", 100, "A", "G", "1/1"),            # MT: one copy
        ("1", 100, "A", "G", "1/1"),             # autosome
    ],
    schema={"chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8, "alt": pl.Utf8, "GT": pl.Utf8},
    orient="row",
)
_SCORING = _scoring([
    ("X", X_NON_PAR, "T", "C", 1.0, None),
    ("X", X_NON_PAR + 1, "T", "C", 10.0, None),
    ("X", X_PAR1, "T", "C", 100.0, None),
    ("Y", Y_NON_PAR, "A", "G", 1000.0, None),
    ("MT", 100, "G", "A", 10_000.0, None),
    ("1", 100, "G", "A", 100_000.0, None),
    ("X", X_NON_PAR + 500, "C", "T", 1_000_000.0, "C"),  # absent, effect = REF: hom-ref copies
])


class TestCopyAwareScoring:
    @pytest.mark.parametrize(
        ("karyotype", "coding", "score", "conflicts"),
        [
            # XY: X hom 1, X het conflict, PAR 2, Y 1, MT 1, autosome 2, absent X hom-ref 1.
            (Karyotype.XY, XChromosomeCoding.COPIES, 1 + 0 + 200 + 1000 + 10_000 + 200_000 + 1_000_000, 1),
            # PLINK2 --xchr-model 2: a man's non-PAR X counts 0..2; Y and MT stay one copy.
            (Karyotype.XY, XChromosomeCoding.DOSAGE_COMPENSATION, 2 + 0 + 200 + 1000 + 10_000 + 200_000 + 2_000_000, 1),
            # XX: X as written, any Y call is a conflict, absent X hom-ref 2.
            (Karyotype.XX, XChromosomeCoding.COPIES, 2 + 10 + 200 + 0 + 10_000 + 200_000 + 2_000_000, 1),
            # XXY: two X, one Y.
            (Karyotype.XXY, XChromosomeCoding.COPIES, 2 + 10 + 200 + 1000 + 10_000 + 200_000 + 2_000_000, 0),
            # unknown: every GT as written, except MT, which is one copy in anyone.
            (Karyotype.UNKNOWN, XChromosomeCoding.COPIES, 2 + 10 + 200 + 2000 + 10_000 + 200_000 + 2_000_000, 0),
        ],
    )
    def test_score_follows_copies(
        self, tmp_path: Path, karyotype: Karyotype, coding: XChromosomeCoding, score: float, conflicts: int
    ) -> None:
        polars_result, duckdb_result = _both(tmp_path, _GENOS, _SCORING, karyotype, coding)
        for result in (polars_result, duckdb_result):
            assert result.score == pytest.approx(score)
            assert result.variants_ploidy_conflict == conflicts
            assert result.variants_no_call == conflicts
            assert result.karyotype == karyotype.value
            assert result.karyotype_source == "provided"
            assert result.x_coding == coding.value
        assert polars_result.variants_hemizygous == duckdb_result.variants_hemizygous
        assert polars_result.variants_matched == duckdb_result.variants_matched

    def test_reference_coding_mismatch_flag(self, tmp_path: Path) -> None:
        # The legacy 1000G curves count every one-copy row twice.
        x_only = _scoring([("X", X_NON_PAR, "T", "C", 1.0, None), ("1", 100, "G", "A", 1.0, None)])
        copies, copies_duck = _both(tmp_path, _GENOS, _SCORING, Karyotype.XY)
        compensated, compensated_duck = _both(tmp_path, _GENOS, _SCORING, Karyotype.XY, XChromosomeCoding.DOSAGE_COMPENSATION)
        compensated_x, compensated_x_duck = _both(tmp_path, _GENOS, x_only, Karyotype.XY, XChromosomeCoding.DOSAGE_COMPENSATION)
        autosomal, _ = _both(tmp_path, _GENOS, _scoring([("1", 100, "G", "A", 1.0, None)]), Karyotype.XY)
        assert copies.reference_coding_mismatch is copies_duck.reference_coding_mismatch is True
        # Y and MT count once under 0..2 too, so they still differ from the legacy curves.
        assert compensated.reference_coding_mismatch is compensated_duck.reference_coding_mismatch is True
        assert compensated_x.reference_coding_mismatch is compensated_x_duck.reference_coding_mismatch is False
        assert autosomal.reference_coding_mismatch is False
        # X hom, Y, MT and the absent X hom-ref row are one-copy rows; the het is a no-call.
        assert copies.variants_hemizygous == 4

    def test_karyotype_is_inferred_when_not_given(self, tmp_path: Path) -> None:
        male = pl.concat([_sample(0.05, y_n=60), _GENOS])
        polars_result, duckdb_result = _both(tmp_path, male, _SCORING, None)
        for result in (polars_result, duckdb_result):
            assert (result.karyotype, result.karyotype_source) == ("XY", "inferred")
        assert polars_result.score == pytest.approx(duckdb_result.score)

    def test_contributions_mark_the_conflict(self, tmp_path: Path) -> None:
        parquet = tmp_path / "g.parquet"
        _GENOS.write_parquet(parquet)
        result = extract_prs_contributions(
            vcf_path="", scoring_file=_SCORING, genotypes_parquet=parquet,
            genotype_input_mode="variant_only", sample_karyotype=Karyotype.XY,
            output=tmp_path / "c.parquet",
        )
        rows = {r["pos"]: r for r in pl.read_parquet(result.output_path).iter_rows(named=True)}
        assert rows[X_NON_PAR + 1]["status"] == "ploidy_conflict"
        assert rows[X_NON_PAR]["dosage"] == 1
        _, duckdb_result = _both(tmp_path, _GENOS, _SCORING, Karyotype.XY)
        assert result.score == pytest.approx(duckdb_result.score)


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


class TestRealGenome:
    def test_anton_is_xy(self) -> None:
        call = infer_karyotype_from_file(_anton_parquet())
        assert call.karyotype == Karyotype.XY
        assert call.x_relative_het is not None and call.x_relative_het < 0.25
        assert call.y_records > 1000

    def test_male_x_counts_half_of_the_compensated_x(self, tmp_path: Path) -> None:
        """PGS000030 has six chrX rows: the copies and 0..2 scores differ by exactly the X part."""
        parquet = _anton_parquet()
        kwargs = dict(vcf_path="", scoring_file="PGS000030", pgs_id="PGS000030", genotypes_parquet=parquet)
        copies = compute_prs_duckdb(**kwargs)
        compensated = compute_prs_duckdb(x_coding=XChromosomeCoding.DOSAGE_COMPENSATION, **kwargs)
        polars_copies = compute_prs(
            vcf_path="", scoring_file="PGS000030", pgs_id="PGS000030", genotypes_lf=pl.scan_parquet(parquet)
        )
        assert copies.karyotype == "XY"
        assert copies.variants_hemizygous > 0
        assert polars_copies.score == pytest.approx(copies.score)
        contributions = pl.read_parquet(
            extract_prs_contributions(
                vcf_path="", scoring_file="PGS000030", pgs_id="PGS000030",
                genotypes_parquet=parquet, output=tmp_path / "c.parquet",
            ).output_path
        )
        x_part = contributions.filter(pl.col("chrom") == "X")["contribution"].fill_null(0.0).sum()
        assert compensated.score - copies.score == pytest.approx(x_part)
