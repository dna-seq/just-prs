"""Reference curves that count a man's X and Y, and everyone's MT, as the copies they are.

The 1000G pgen stores a haploid call as 0/2 (pgenlib's convention), and the legacy
scorer summed it as two copies. The reference scorer now also writes a ``copies`` and a
``dosage_compensation`` score per sample, and the per-sex curves built from them are
what ``PRSCatalog.percentile_full`` reads when it knows the sample's karyotype.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from just_prs.karyotype import Karyotype, reference_curve_sex
from just_prs.prs_catalog import PRSCatalog
from just_prs.reference import (
    _REGION_MT,
    _REGION_X,
    _REGION_Y,
    _one_copy_deltas,
    aggregate_distributions,
    aggregate_sex_aware_distributions,
    parse_psam,
)

_MALE = np.array([True, False, False])
_FEMALE = np.array([False, True, False])


def _deltas(geno: list[int], region: int, weight: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One row, three samples (male, female, unknown sex), effect allele = ALT."""
    g = np.array([geno], dtype=np.int8)
    contrib = np.where(g == -9, 0, g).astype(np.float32) * np.float32(weight)
    return _one_copy_deltas(g, contrib, np.array([region], dtype=np.int8), _MALE, _FEMALE)


class TestOneCopyDeltas:
    def test_non_par_x(self) -> None:
        # A man's hom-alt reads 2: one copy, so -1; 0..2 keeps it. A woman's X is diploid.
        copies, compensated = _deltas([2, 2, 2], _REGION_X)
        assert copies.tolist() == [-1.0, 0.0, 0.0]
        assert compensated.tolist() == [0.0, 0.0, 0.0]

    def test_male_x_het_is_a_no_call_in_both_codings(self) -> None:
        copies, compensated = _deltas([1, 1, 1], _REGION_X)
        assert copies.tolist() == [-1.0, 0.0, 0.0]
        assert compensated.tolist() == [-1.0, 0.0, 0.0]

    def test_y_counts_once_and_never_for_a_woman(self) -> None:
        copies, compensated = _deltas([2, 2, 2], _REGION_Y, weight=3.0)
        # PLINK2 counts Y once under every --xchr-model, so both codings agree.
        assert copies.tolist() == compensated.tolist() == [-3.0, -6.0, 0.0]

    def test_mt_counts_once_in_anyone(self) -> None:
        copies, compensated = _deltas([2, 2, 1], _REGION_MT)
        # A het (heteroplasmy) on MT is a no-call, as in the sample engines.
        assert copies.tolist() == compensated.tolist() == [-1.0, -1.0, -1.0]

    def test_missing_and_ref_calls_change_nothing(self) -> None:
        for region in (_REGION_X, _REGION_Y, _REGION_MT):
            copies, compensated = _deltas([-9, 0, -9], region)
            assert copies.tolist() == compensated.tolist() == [0.0, 0.0, 0.0]

    def test_effect_ref_rows_halve_the_flipped_dose(self) -> None:
        # Effect = REF: the legacy contribution of a 0 call is weight * (2 - 0) = 2.
        g = np.array([[0, 2]], dtype=np.int8)
        contrib = (np.float32(2.0) - g.astype(np.float32)) * np.float32(5.0)
        copies, _ = _one_copy_deltas(
            g, contrib, np.array([_REGION_X], dtype=np.int8), np.array([True, True]), np.array([False, False])
        )
        assert copies.tolist() == [-5.0, 0.0]


def _sample_scores(sex_linked: int = 3) -> pl.DataFrame:
    rows = []
    for superpop, base in (("EUR", 0.0), ("AFR", 10.0)):
        for i in range(6):
            sex = "male" if i < 3 else "female"
            rows.append((f"{superpop}{i}", superpop, sex, base + i, base + i - (1.0 if sex == "male" else 0.0)))
    rows.append(("EURu", "EUR", "unknown", 100.0, 100.0))
    return pl.DataFrame(
        rows, schema=["iid", "superpop", "sex", "score", "score_copies"], orient="row"
    ).with_columns(
        pl.lit("PGS1").alias("pgs_id"),
        pl.col("score").alias("score_dosage_compensation"),
        pl.lit(sex_linked).alias("variants_matched_sex_linked"),
    )


class TestAggregateSexAware:
    def test_curves_by_population_sex_and_coding(self) -> None:
        scores = _sample_scores()
        curves = aggregate_sex_aware_distributions(scores)
        assert curves.height == 2 * 3 * 2
        expected = (
            scores.unpivot(
                index=["pgs_id", "superpop", "sex"],
                on=["score_copies", "score_dosage_compensation"],
                variable_name="x_coding",
                value_name="v",
            )
            .with_columns(pl.col("x_coding").str.strip_prefix("score_"))
        )
        for row in curves.iter_rows(named=True):
            sub = expected.filter(
                (pl.col("superpop") == row["superpopulation"]) & (pl.col("x_coding") == row["x_coding"])
            )
            if row["sex"] != "all":
                sub = sub.filter(pl.col("sex") == row["sex"])
            assert row["n"] == sub.height
            assert row["mean"] == pytest.approx(sub["v"].mean())
            assert row["std"] == pytest.approx(sub["v"].std())

    def test_unknown_sex_counts_toward_all_only(self) -> None:
        curves = aggregate_sex_aware_distributions(_sample_scores()).filter(pl.col("superpopulation") == "EUR")
        n = {r["sex"]: r["n"] for r in curves.filter(pl.col("x_coding") == "copies").iter_rows(named=True)}
        assert n == {"all": 7, "male": 3, "female": 3}

    def test_the_legacy_curve_is_unchanged(self) -> None:
        legacy = aggregate_distributions(_sample_scores())
        assert legacy.columns == [
            "pgs_id", "superpopulation", "mean", "std", "n", "median", "p5", "p25", "p75", "p95"
        ]

    def test_no_sex_linked_rows_means_one_curve_for_both_sexes(self) -> None:
        """Men and women score alike without X or Y rows; half-size curves would only add noise."""
        curves = aggregate_sex_aware_distributions(_sample_scores(sex_linked=0))
        assert set(curves["sex"]) == {"all"}
        assert curves.height == 2 * 2

    def test_a_null_coding_yields_no_curve(self) -> None:
        scores = _sample_scores().with_columns(pl.lit(None, dtype=pl.Float64).alias("score_copies"))
        curves = aggregate_sex_aware_distributions(scores)
        assert set(curves["x_coding"]) == {"dosage_compensation"}


class TestPanelFiles:
    def test_parse_psam_reads_sex(self, tmp_path: Path) -> None:
        psam = tmp_path / "x.psam"
        psam.write_text("#IID\tPAT\tMAT\tSEX\tSuperPop\tPopulation\nA\t0\t0\t1\tEUR\tGBR\nB\t0\t0\t2\tEUR\tGBR\nC\t0\t0\t0\tEUR\tGBR\n")
        df = parse_psam(psam)
        assert df["sex"].to_list() == ["male", "female", "unknown"]
        assert df.columns == ["iid", "superpop", "population", "sex"]

    def test_parse_psam_without_sex(self, tmp_path: Path) -> None:
        psam = tmp_path / "x.psam"
        psam.write_text("#IID\tSuperPop\tPopulation\nA\tEUR\tGBR\n")
        assert parse_psam(psam)["sex"].to_list() == ["unknown"]


@pytest.mark.parametrize(
    ("karyotype", "sex"),
    [(Karyotype.XX, "female"), (Karyotype.XY, "male"), (Karyotype.XXY, "all"),
     (Karyotype.X0, "all"), (Karyotype.UNKNOWN, "all"), (None, "all")],
)
def test_reference_curve_sex(karyotype: Karyotype | None, sex: str) -> None:
    assert reference_curve_sex(karyotype) == sex


def _offline_catalog(tmp_path: Path, *, by_sex: bool = True) -> PRSCatalog:
    """A catalog whose percentile files are local, so no lookup reaches HuggingFace."""
    catalog = PRSCatalog(cache_dir=tmp_path)
    catalog._catalog_flags_loaded = True
    catalog._catalog_flags_lf = None
    catalog._ref_dist_refresh_attempted.add("1000g")  # a lookup miss must not refetch from HF
    pct = tmp_path / "percentiles"
    pct.mkdir(parents=True)
    stats = dict(n=100, median=0.0, p5=0.0, p25=0.0, p75=0.0, p95=0.0)
    pl.DataFrame([
        dict(pgs_id="PGS1", superpopulation="EUR", mean=10.0, std=1.0, **stats),
        dict(pgs_id="PGS2", superpopulation="EUR", mean=10.0, std=1.0, **stats),
    ]).with_columns(pl.col("n").cast(pl.UInt32)).write_parquet(pct / "1000g_distributions.parquet")
    pl.DataFrame(schema={"pgs_id": pl.Utf8, "severity": pl.Utf8}).write_parquet(
        pct / "1000g_distribution_quality_issues.parquet"
    )
    if by_sex:
        means = {("all", "copies"): 9.5, ("male", "copies"): 9.0, ("female", "copies"): 10.0,
                 ("all", "dosage_compensation"): 10.0, ("male", "dosage_compensation"): 11.0,
                 ("female", "dosage_compensation"): 10.0}
        pl.DataFrame([
            dict(pgs_id="PGS1", superpopulation="EUR", sex=s, x_coding=c, mean=m, std=1.0, **stats)
            for (s, c), m in means.items()
        ]).with_columns(pl.col("n").cast(pl.UInt32)).write_parquet(pct / "1000g_distributions_by_sex.parquet")
    return catalog


class TestPercentileCurveChoice:
    @pytest.mark.parametrize(
        ("karyotype", "x_coding", "sex", "mean"),
        [
            ("XY", "copies", "male", 9.0),
            ("XX", "copies", "female", 10.0),
            ("XXY", "copies", "all", 9.5),
            ("X0", "copies", "all", 9.5),
            ("unknown", "copies", "all", 9.5),
            ("XY", "dosage_compensation", "male", 11.0),
            ("XY", None, "male", 9.0),
        ],
    )
    def test_karyotype_picks_the_curve(
        self, tmp_path: Path, karyotype: str, x_coding: str | None, sex: str, mean: float
    ) -> None:
        result = _offline_catalog(tmp_path).percentile_full(
            10.0, "PGS1", ancestry="EUR", karyotype=karyotype, x_coding=x_coding
        )
        assert result.method == "reference_panel"
        assert result.reference_mean == mean
        assert (result.reference_sex, result.reference_x_coding) == (sex, x_coding or "copies")

    def test_no_karyotype_reads_the_legacy_curve(self, tmp_path: Path) -> None:
        result = _offline_catalog(tmp_path).percentile_full(10.0, "PGS1", ancestry="EUR")
        assert (result.reference_mean, result.reference_sex, result.reference_x_coding) == (10.0, None, None)

    def test_a_score_without_sex_linked_rows_uses_the_both_sexes_curve(self, tmp_path: Path) -> None:
        catalog = _offline_catalog(tmp_path)
        path = tmp_path / "percentiles" / "1000g_distributions_by_sex.parquet"
        pl.read_parquet(path).filter(pl.col("sex") == "all").write_parquet(path)
        result = catalog.percentile_full(10.0, "PGS1", ancestry="EUR", karyotype="XY")
        assert (result.reference_mean, result.reference_sex, result.reference_x_coding) == (9.5, "all", "copies")

    def test_a_score_without_per_sex_curves_falls_back_to_legacy(self, tmp_path: Path) -> None:
        result = _offline_catalog(tmp_path).percentile_full(10.0, "PGS2", ancestry="EUR", karyotype="XY")
        assert (result.method, result.reference_mean, result.reference_sex) == ("reference_panel", 10.0, None)

    def test_quarantined_scores_are_dropped_from_the_per_sex_curves(self, tmp_path: Path) -> None:
        catalog = _offline_catalog(tmp_path)
        pl.DataFrame({"pgs_id": ["PGS1"], "severity": ["ERROR"]}).write_parquet(
            tmp_path / "percentiles" / "1000g_distribution_quality_issues.parquet"
        )
        result = catalog.percentile_full(10.0, "PGS1", ancestry="EUR", karyotype="XY")
        assert result.method != "reference_panel" or result.reference_sex is None
        by_sex = catalog.reference_distributions_by_sex()
        assert by_sex is not None and by_sex.collect().height == 0


def _panel_dir() -> Path:
    from just_prs.reference import reference_panel_dir
    from just_prs.scoring import resolve_cache_dir

    ref_dir = reference_panel_dir(resolve_cache_dir(), "1000g")
    if not any(ref_dir.glob("GRCh38*.pgen")):
        pytest.skip(f"1000G GRCh38 panel not present at {ref_dir}")
    return ref_dir


class TestRealPanel:
    def test_pgs000030_male_x_counts_once(self, tmp_path: Path) -> None:
        """PGS000030 has X rows only among its one-copy rows: women and 0..2 are unchanged,
        and a man's copies score drops by half of his X contribution, read independently here."""
        from just_prs.reference import _ResolvedRefPanel, compute_reference_prs_polars, read_pgen_genotypes
        from just_prs.prs import _normalize_scoring_columns
        from just_prs.scoring import ensure_scoring_file, parse_scoring_file, resolve_cache_dir

        ref_dir = _panel_dir()
        cache = resolve_cache_dir()
        scoring_file = ensure_scoring_file("PGS000030", cache / "scores", "GRCh38")
        panel = _ResolvedRefPanel(ref_dir, "GRCh38")
        scores = compute_reference_prs_polars("PGS000030", scoring_file, ref_dir, tmp_path, _panel=panel)
        assert scores.height == panel.psam_df.height == 3202
        assert scores["variants_matched_one_copy"][0] == scores["variants_matched_sex_linked"][0] > 0
        assert set(aggregate_sex_aware_distributions(scores)["sex"]) == {"all", "male", "female"}

        female = scores.filter(pl.col("sex") == "female")
        assert (female["score"] - female["score_copies"]).abs().max() == 0.0
        assert (scores["score"] - scores["score_dosage_compensation"]).abs().max() == 0.0

        matched = panel.match_scoring(_normalize_scoring_columns(parse_scoring_file(scoring_file)))
        x_rows = matched.filter(pl.col("chrom") == "X")
        geno = read_pgen_genotypes(
            panel.pgen_path, panel.pvar_zst_path, x_rows["variant_idx"].cast(pl.UInt32).to_numpy(),
            panel.psam_df.height, panel.pvar_variant_ct,
        ).astype(np.float64)
        dose = np.where(x_rows["effect_is_alt"].to_numpy()[:, None], geno, 2.0 - geno)
        valid = (geno != -9) & (geno != 1)
        half_x = (np.where(valid, dose, 0.0) * x_rows["effect_weight"].to_numpy()[:, None]).sum(axis=0) / 2.0
        het_x = (np.where(geno == 1, 1.0, 0.0) * x_rows["effect_weight"].to_numpy()[:, None]).sum(axis=0)
        male = (scores["sex"] == "male").to_numpy()
        drop = (scores["score"] - scores["score_copies"]).to_numpy()
        np.testing.assert_allclose(drop[male], (half_x + het_x)[male], atol=1e-4)
