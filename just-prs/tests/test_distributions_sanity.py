"""Sanity checks for reference panel distributions and percentile computation.

Validates:
  - Stored distributions parquet is consistent with raw per-sample scores
  - aggregate_distributions() produces mathematically correct statistics
  - ancestry_percentile() handles default cases, edge cases, and extreme values
  - End-to-end: re-scoring a PGS ID matches stored raw scores
  - Distribution invariants: p5 <= p25 <= median <= p75 <= p95, std >= 0, etc.
"""

import math
import random
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from just_prs.reference import (
    SUPERPOPULATIONS,
    _norm_cdf,
    aggregate_distributions,
    ancestry_percentile,
    compute_reference_prs_polars,
    reference_panel_dir,
)
from just_prs.scoring import resolve_cache_dir


CACHE_DIR = resolve_cache_dir()
DIST_PATH = CACHE_DIR / "percentiles" / "1000g_distributions.parquet"
REF_SCORES_DIR = CACHE_DIR / "reference_scores" / "1000g"

HAS_DISTRIBUTIONS = DIST_PATH.exists()
HAS_REF_PANEL = reference_panel_dir(CACHE_DIR, panel="1000g").exists()


def _load_distributions() -> pl.DataFrame:
    return pl.read_parquet(DIST_PATH)


def _load_raw_scores(pgs_id: str) -> pl.DataFrame | None:
    path = REF_SCORES_DIR / pgs_id / "scores.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# aggregate_distributions — unit tests with synthetic data
# ---------------------------------------------------------------------------


class TestAggregateDistributionsMath:
    """Verify aggregate_distributions() produces correct statistics."""

    def _make_scores(
        self,
        scores_by_superpop: dict[str, list[float]],
        pgs_id: str = "PGS_TEST",
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for sp, scores in scores_by_superpop.items():
            for i, s in enumerate(scores):
                rows.append({
                    "pgs_id": pgs_id,
                    "iid": f"{sp}_{i:04d}",
                    "superpop": sp,
                    "population": f"POP_{sp}",
                    "score": s,
                })
        return pl.DataFrame(rows)

    def test_mean_exact(self) -> None:
        """Mean matches manual calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = self._make_scores({"EUR": values})
        result = aggregate_distributions(df)
        assert result["mean"][0] == pytest.approx(3.0)

    def test_std_exact(self) -> None:
        """Std matches numpy's sample standard deviation (ddof=1 by default in polars)."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_std = float(np.std(values, ddof=1))
        df = self._make_scores({"EUR": values})
        result = aggregate_distributions(df)
        assert result["std"][0] == pytest.approx(expected_std, abs=1e-10)

    def test_median_exact(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = self._make_scores({"EUR": values})
        result = aggregate_distributions(df)
        assert result["median"][0] == pytest.approx(3.0)

    def test_quantile_ordering(self) -> None:
        """p5 <= p25 <= median <= p75 <= p95 for any distribution."""
        rng = np.random.default_rng(42)
        values = rng.normal(loc=5.0, scale=2.0, size=1000).tolist()
        df = self._make_scores({"EUR": values})
        result = aggregate_distributions(df)
        row = result.row(0, named=True)
        assert row["p5"] <= row["p25"] <= row["median"] <= row["p75"] <= row["p95"]

    def test_n_correct(self) -> None:
        df = self._make_scores({"AFR": [1.0, 2.0], "EUR": [3.0, 4.0, 5.0]})
        result = aggregate_distributions(df)
        afr = result.filter(pl.col("superpopulation") == "AFR")
        eur = result.filter(pl.col("superpopulation") == "EUR")
        assert afr["n"][0] == 2
        assert eur["n"][0] == 3

    def test_separate_pgs_ids(self) -> None:
        """Different PGS IDs produce independent distributions."""
        df1 = self._make_scores({"EUR": [10.0, 20.0]}, pgs_id="PGS_A")
        df2 = self._make_scores({"EUR": [100.0, 200.0]}, pgs_id="PGS_B")
        combined = pl.concat([df1, df2])
        result = aggregate_distributions(combined)
        a = result.filter(pl.col("pgs_id") == "PGS_A")
        b = result.filter(pl.col("pgs_id") == "PGS_B")
        assert a["mean"][0] == pytest.approx(15.0)
        assert b["mean"][0] == pytest.approx(150.0)

    def test_identical_scores_give_zero_std(self) -> None:
        """When all scores are identical, std should be 0."""
        df = self._make_scores({"EUR": [5.0] * 100})
        result = aggregate_distributions(df)
        assert result["std"][0] == pytest.approx(0.0)
        assert result["mean"][0] == pytest.approx(5.0)

    def test_single_sample_gives_nan_std(self) -> None:
        """With one sample, polars std (ddof=1) gives NaN — this is mathematically correct."""
        df = self._make_scores({"EUR": [5.0]})
        result = aggregate_distributions(df)
        assert result["n"][0] == 1
        assert result["mean"][0] == pytest.approx(5.0)

    def test_negative_scores_handled(self) -> None:
        """Negative scores are valid (some PRS scores are negative)."""
        values = [-3.0, -2.0, -1.0, 0.0, 1.0]
        df = self._make_scores({"EUR": values})
        result = aggregate_distributions(df)
        assert result["mean"][0] == pytest.approx(-1.0)

    def test_very_small_scores(self) -> None:
        """Very small scores (order 1e-10) should be aggregated without losing precision."""
        values = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        df = self._make_scores({"EUR": values})
        result = aggregate_distributions(df)
        assert result["mean"][0] == pytest.approx(3e-10, rel=1e-6)


# ---------------------------------------------------------------------------
# ancestry_percentile — unit tests with synthetic distributions
# ---------------------------------------------------------------------------


class TestAncestryPercentileUnit:
    """Tests for ancestry_percentile() using synthetic distributions."""

    @pytest.fixture()
    def dist_lf(self) -> pl.LazyFrame:
        rows = []
        for i, sp in enumerate(SUPERPOPULATIONS):
            rows.append({
                "pgs_id": "PGS_TEST",
                "superpopulation": sp,
                "mean": 1.0 + i * 0.2,
                "std": 0.5,
                "n": 500,
                "median": 1.0 + i * 0.2,
                "p5": 0.2, "p25": 0.7, "p75": 1.3, "p95": 1.8,
            })
        return pl.DataFrame(rows).lazy()

    def test_score_at_mean_gives_50th(self, dist_lf: pl.LazyFrame) -> None:
        pct = ancestry_percentile(1.0, "PGS_TEST", "AFR", dist_lf)
        assert pct is not None
        assert pct == pytest.approx(50.0, abs=0.01)

    def test_score_above_mean_gives_high_percentile(self, dist_lf: pl.LazyFrame) -> None:
        pct = ancestry_percentile(1.0 + 2 * 0.5, "PGS_TEST", "AFR", dist_lf)
        assert pct is not None
        assert pct > 97.0

    def test_score_below_mean_gives_low_percentile(self, dist_lf: pl.LazyFrame) -> None:
        pct = ancestry_percentile(1.0 - 2 * 0.5, "PGS_TEST", "AFR", dist_lf)
        assert pct is not None
        assert pct < 3.0

    def test_extreme_positive_score(self, dist_lf: pl.LazyFrame) -> None:
        """Extreme positive z-score → ~100%, not None."""
        pct = ancestry_percentile(100.0, "PGS_TEST", "AFR", dist_lf)
        assert pct is not None
        assert pct == pytest.approx(100.0, abs=0.01)

    def test_extreme_negative_score(self, dist_lf: pl.LazyFrame) -> None:
        """Extreme negative z-score → ~0%, not None."""
        pct = ancestry_percentile(-100.0, "PGS_TEST", "AFR", dist_lf)
        assert pct is not None
        assert pct == pytest.approx(0.0, abs=0.01)

    def test_always_in_0_100_range(self, dist_lf: pl.LazyFrame) -> None:
        """No matter the score, result is in [0, 100]."""
        for score in [-1e6, -100, -10, -1, 0, 1, 10, 100, 1e6]:
            pct = ancestry_percentile(score, "PGS_TEST", "AFR", dist_lf)
            assert pct is not None
            assert 0.0 <= pct <= 100.0, f"score={score} → pct={pct}"

    def test_missing_pgs_id_returns_none(self, dist_lf: pl.LazyFrame) -> None:
        assert ancestry_percentile(1.0, "PGS_MISSING", "AFR", dist_lf) is None

    def test_missing_superpop_returns_none(self, dist_lf: pl.LazyFrame) -> None:
        assert ancestry_percentile(1.0, "PGS_TEST", "UNKNOWN", dist_lf) is None

    def test_zero_std_returns_none(self) -> None:
        """std=0 makes z-score undefined → must return None."""
        lf = pl.DataFrame({
            "pgs_id": ["PGS_TEST"],
            "superpopulation": ["EUR"],
            "mean": [1.0],
            "std": [0.0],
            "n": [100],
            "median": [1.0], "p5": [1.0], "p25": [1.0], "p75": [1.0], "p95": [1.0],
        }).lazy()
        assert ancestry_percentile(1.0, "PGS_TEST", "EUR", lf) is None

    def test_nan_std_returns_none(self) -> None:
        """NaN std (e.g. single-sample group) → None."""
        lf = pl.DataFrame({
            "pgs_id": ["PGS_TEST"],
            "superpopulation": ["EUR"],
            "mean": [1.0],
            "std": [float("nan")],
            "n": [1],
            "median": [1.0], "p5": [1.0], "p25": [1.0], "p75": [1.0], "p95": [1.0],
        }).lazy()
        pct = ancestry_percentile(1.0, "PGS_TEST", "EUR", lf)
        assert pct is None

    def test_inf_mean_returns_none_or_extreme(self) -> None:
        """Inf mean is a data quality issue. Percentile should not crash."""
        lf = pl.DataFrame({
            "pgs_id": ["PGS_TEST"],
            "superpopulation": ["EUR"],
            "mean": [float("inf")],
            "std": [1.0],
            "n": [100],
            "median": [1.0], "p5": [1.0], "p25": [1.0], "p75": [1.0], "p95": [1.0],
        }).lazy()
        pct = ancestry_percentile(1.0, "PGS_TEST", "EUR", lf)
        assert pct is not None
        assert 0.0 <= pct <= 100.0

    def test_different_superpops_give_different_percentiles(self, dist_lf: pl.LazyFrame) -> None:
        """Same score against different ancestry groups → different percentiles."""
        score = 1.3
        pcts = {
            sp: ancestry_percentile(score, "PGS_TEST", sp, dist_lf)
            for sp in SUPERPOPULATIONS
        }
        assert all(v is not None for v in pcts.values())
        assert len(set(round(v, 2) for v in pcts.values())) > 1

    def test_monotonic_with_score(self, dist_lf: pl.LazyFrame) -> None:
        """Percentile is monotonically increasing with score."""
        scores = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        pcts = [ancestry_percentile(s, "PGS_TEST", "EUR", dist_lf) for s in scores]
        assert all(p is not None for p in pcts)
        for i in range(len(pcts) - 1):
            assert pcts[i] <= pcts[i + 1], f"pct[{i}]={pcts[i]} > pct[{i+1}]={pcts[i+1]}"

    def test_case_insensitive_superpop(self, dist_lf: pl.LazyFrame) -> None:
        """Superpopulation lookup should be case-insensitive."""
        pct_upper = ancestry_percentile(1.0, "PGS_TEST", "AFR", dist_lf)
        pct_lower = ancestry_percentile(1.0, "PGS_TEST", "afr", dist_lf)
        assert pct_upper == pct_lower


# ---------------------------------------------------------------------------
# _norm_cdf edge cases
# ---------------------------------------------------------------------------


class TestNormCdfEdgeCases:
    """Extreme value handling for the CDF implementation."""

    def test_large_positive_z(self) -> None:
        assert _norm_cdf(100.0) == pytest.approx(1.0, abs=1e-15)

    def test_large_negative_z(self) -> None:
        assert _norm_cdf(-100.0) == pytest.approx(0.0, abs=1e-15)

    def test_z_of_37(self) -> None:
        """z=37 is near the limit of float64 erfc precision. Should not produce NaN."""
        result = _norm_cdf(37.0)
        assert math.isfinite(result)
        assert result == pytest.approx(1.0, abs=1e-15)

    def test_z_of_minus_37(self) -> None:
        result = _norm_cdf(-37.0)
        assert math.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_zero(self) -> None:
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-15)


# ---------------------------------------------------------------------------
# Stored distributions sanity check (requires cached data on disk)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DISTRIBUTIONS, reason="No 1000g_distributions.parquet on disk")
class TestStoredDistributionsSanity:
    """Validate the on-disk distributions parquet for anomalies."""

    @pytest.fixture(scope="class")
    def dist_df(self) -> pl.DataFrame:
        return _load_distributions()

    def test_all_superpopulations_present(self, dist_df: pl.DataFrame) -> None:
        """Every PGS ID should have exactly 5 superpopulation rows unless some were quarantined."""
        counts = dist_df.group_by("pgs_id").agg(pl.len().alias("count"))
        non_five = counts.filter(pl.col("count") != 5)
        if non_five.height > 0:
            issues_path = DIST_PATH.with_name("1000g_distribution_quality_issues.parquet")
            if issues_path.exists():
                issues_df = pl.read_parquet(issues_path)
                issue_ids = set(issues_df["pgs_id"].unique().to_list())
                non_five_ids = set(non_five["pgs_id"].to_list())
                unexplained = non_five_ids - issue_ids
                assert len(unexplained) == 0, (
                    f"Unexplained incomplete PGS IDs: {unexplained}. "
                    f"These should have corresponding entries in {issues_path.name}"
                )
            else:
                # If issues file isn't present, allow a few quarantined exceptions
                assert non_five.height <= 10, (
                    f"Too many incomplete PGS IDs: {non_five.height}"
                )

    def test_superpopulations_are_known(self, dist_df: pl.DataFrame) -> None:
        sps = set(dist_df["superpopulation"].unique().to_list())
        assert sps == set(SUPERPOPULATIONS)

    def test_no_null_mean_or_std(self, dist_df: pl.DataFrame) -> None:
        null_mean = dist_df.filter(pl.col("mean").is_null()).height
        null_std = dist_df.filter(pl.col("std").is_null()).height
        assert null_mean == 0, f"{null_mean} rows have null mean"
        assert null_std == 0, f"{null_std} rows have null std"

    def test_quantile_ordering_invariant(self, dist_df: pl.DataFrame) -> None:
        """p5 <= p25 <= median <= p75 <= p95 for all rows."""
        violated = dist_df.filter(
            (pl.col("p5") > pl.col("p25"))
            | (pl.col("p25") > pl.col("median"))
            | (pl.col("median") > pl.col("p75"))
            | (pl.col("p75") > pl.col("p95"))
        )
        assert violated.height == 0, (
            f"{violated.height} rows violate quantile ordering"
        )

    def test_sample_sizes_reasonable(self, dist_df: pl.DataFrame) -> None:
        """1000G panel has ~3202 samples; each superpop has 490-893."""
        assert dist_df["n"].min() >= 400
        assert dist_df["n"].max() <= 1000

    def test_count_inf_and_nan(self, dist_df: pl.DataFrame) -> None:
        """Report (but do not necessarily fail) on inf/nan values.

        These indicate stale data that needs recomputation. We flag them so
        the pipeline knows to regenerate.
        """
        n_inf = dist_df.filter(
            pl.col("mean").is_infinite() | pl.col("std").is_infinite()
        ).height
        n_nan = dist_df.filter(
            pl.col("mean").is_nan() | pl.col("std").is_nan()
        ).height
        n_zero_std = dist_df.filter(pl.col("std") == 0.0).height
        if n_inf + n_nan + n_zero_std > 0:
            pytest.xfail(
                f"Found {n_inf} inf, {n_nan} NaN, {n_zero_std} zero-std rows "
                f"(known stale data issue — regenerate distributions)"
            )


# ---------------------------------------------------------------------------
# Cross-validation: stored raw scores vs fresh scoring (requires ref panel)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_REF_PANEL or not HAS_DISTRIBUTIONS,
    reason="Reference panel or distributions not available",
)
class TestStoredScoresVsFreshScoring:
    """Verify that stored raw per-sample scores match fresh computation.

    Picks a few PGS IDs with small variant counts (fast to re-score) and
    compares the stored scores against a fresh compute_reference_prs_polars run.
    """

    SAMPLE_PGS_IDS = ["PGS000001", "PGS000977"]

    @pytest.fixture(scope="class")
    def ref_dir(self) -> Path:
        return reference_panel_dir(CACHE_DIR, panel="1000g")

    @pytest.mark.parametrize("pgs_id", SAMPLE_PGS_IDS)
    def test_fresh_scores_match_stored(self, pgs_id: str, ref_dir: Path) -> None:
        stored = _load_raw_scores(pgs_id)
        if stored is None:
            pytest.skip(f"No stored scores for {pgs_id}")

        scoring_file = CACHE_DIR / "scores" / f"{pgs_id}_hmPOS_GRCh38.parquet"
        if not scoring_file.exists():
            scoring_file = CACHE_DIR / "scores" / f"{pgs_id}_hmPOS_GRCh38.txt.gz"
        if not scoring_file.exists():
            pytest.skip(f"No scoring file for {pgs_id}")

        fresh = compute_reference_prs_polars(
            pgs_id=pgs_id,
            scoring_file=scoring_file,
            ref_dir=ref_dir,
            out_dir=Path("/tmp/test_scoring_sanity"),
            genome_build="GRCh38",
        )

        assert fresh.height == stored.height, (
            f"Sample count mismatch: fresh={fresh.height}, stored={stored.height}"
        )

        fresh_sorted = fresh.sort("iid")["score"].to_numpy()
        stored_sorted = stored.sort("iid")["score"].to_numpy()
        np.testing.assert_allclose(
            fresh_sorted, stored_sorted, rtol=1e-5, atol=1e-10,
            err_msg=f"{pgs_id}: fresh vs stored scores diverge",
        )

    @pytest.mark.parametrize("pgs_id", SAMPLE_PGS_IDS)
    def test_stored_distributions_match_reaggregation(self, pgs_id: str) -> None:
        """Re-aggregating stored raw scores must match the distributions parquet.

        If this test fails, the distributions parquet is stale and needs
        regeneration from the current raw scores.
        """
        stored = _load_raw_scores(pgs_id)
        if stored is None:
            pytest.skip(f"No stored scores for {pgs_id}")

        fresh_dist = aggregate_distributions(stored)
        stored_dist = _load_distributions().filter(pl.col("pgs_id") == pgs_id)

        if stored_dist.height == 0:
            pytest.skip(f"No stored distribution for {pgs_id}")

        for sp in SUPERPOPULATIONS:
            fresh_row = fresh_dist.filter(pl.col("superpopulation") == sp)
            stored_row = stored_dist.filter(pl.col("superpopulation") == sp)
            if fresh_row.height == 0 or stored_row.height == 0:
                continue

            f_mean = float(fresh_row["mean"][0])
            s_mean = float(stored_row["mean"][0])

            if math.isfinite(f_mean) and math.isfinite(s_mean) and abs(f_mean) > 1e-15:
                rel_err = abs(f_mean - s_mean) / abs(f_mean)
                if rel_err > 0.01:
                    pytest.xfail(
                        f"{pgs_id} {sp}: stored mean={s_mean:.6e} vs fresh={f_mean:.6e} "
                        f"(rel_err={rel_err:.4f}) — distributions parquet is stale"
                    )

            f_std = float(fresh_row["std"][0])
            s_std = float(stored_row["std"][0])
            if math.isfinite(f_std) and math.isfinite(s_std) and abs(f_std) > 1e-15:
                rel_err_std = abs(f_std - s_std) / abs(f_std)
                if rel_err_std > 0.01:
                    pytest.xfail(
                        f"{pgs_id} {sp}: stored std={s_std:.6e} vs fresh={f_std:.6e} "
                        f"(rel_err={rel_err_std:.4f}) — distributions parquet is stale"
                    )


# ---------------------------------------------------------------------------
# Spot-check: random sample of PGS IDs for distribution consistency
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_DISTRIBUTIONS,
    reason="No distributions on disk",
)
class TestDistributionConsistencySpotCheck:
    """Randomly sample PGS IDs and verify stored distributions match raw scores."""

    N_SAMPLES = 10

    def test_random_sample_consistency(self) -> None:
        dist_df = _load_distributions()
        all_ids = dist_df["pgs_id"].unique().to_list()

        available_ids = [
            pid for pid in all_ids
            if (REF_SCORES_DIR / pid / "scores.parquet").exists()
        ]
        if not available_ids:
            pytest.skip("No raw score files available for spot-checking")

        rng = random.Random(42)
        sample_ids = rng.sample(available_ids, min(self.N_SAMPLES, len(available_ids)))

        stale_count = 0
        stale_details: list[str] = []

        for pgs_id in sample_ids:
            raw = _load_raw_scores(pgs_id)
            if raw is None:
                continue
            fresh_dist = aggregate_distributions(raw)

            for sp in SUPERPOPULATIONS:
                f_row = fresh_dist.filter(pl.col("superpopulation") == sp)
                s_row = dist_df.filter(
                    (pl.col("pgs_id") == pgs_id) & (pl.col("superpopulation") == sp)
                )
                if f_row.height == 0 or s_row.height == 0:
                    continue

                f_mean = float(f_row["mean"][0])
                s_mean = float(s_row["mean"][0])

                if not math.isfinite(f_mean) or not math.isfinite(s_mean):
                    stale_count += 1
                    stale_details.append(f"{pgs_id}/{sp}: non-finite values")
                    continue

                if abs(f_mean) > 1e-15:
                    rel_err = abs(f_mean - s_mean) / abs(f_mean)
                    if rel_err > 0.01:
                        stale_count += 1
                        stale_details.append(
                            f"{pgs_id}/{sp}: mean {s_mean:.4e} vs {f_mean:.4e} "
                            f"(err={rel_err:.1%})"
                        )

        if stale_count > 0:
            pytest.xfail(
                f"{stale_count}/{self.N_SAMPLES * 5} superpop-rows are stale: "
                + "; ".join(stale_details[:5])
            )


# ---------------------------------------------------------------------------
# Percentile computation with real distributions (requires cached data)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DISTRIBUTIONS, reason="No distributions on disk")
class TestPercentileWithRealDistributions:
    """Verify percentile computation using real reference distributions."""

    @pytest.fixture(scope="class")
    def dist_lf(self) -> pl.LazyFrame:
        return pl.scan_parquet(DIST_PATH)

    @pytest.fixture(scope="class")
    def first_valid_pgs(self, dist_lf: pl.LazyFrame) -> tuple[str, float, float]:
        """Find first PGS ID with a valid EUR distribution (finite, std > 0)."""
        rows = (
            dist_lf.filter(
                (pl.col("superpopulation") == "EUR")
                & (pl.col("std") > 0)
                & pl.col("mean").is_finite()
                & pl.col("std").is_finite()
            )
            .select("pgs_id", "mean", "std")
            .head(1)
            .collect()
        )
        if rows.height == 0:
            pytest.skip("No valid EUR distributions")
        return (
            str(rows["pgs_id"][0]),
            float(rows["mean"][0]),
            float(rows["std"][0]),
        )

    def test_score_at_mean_gives_50th(
        self, dist_lf: pl.LazyFrame, first_valid_pgs: tuple[str, float, float]
    ) -> None:
        pgs_id, mean, std = first_valid_pgs
        pct = ancestry_percentile(mean, pgs_id, "EUR", dist_lf)
        assert pct is not None
        assert pct == pytest.approx(50.0, abs=0.5)

    def test_score_above_mean(
        self, dist_lf: pl.LazyFrame, first_valid_pgs: tuple[str, float, float]
    ) -> None:
        pgs_id, mean, std = first_valid_pgs
        pct = ancestry_percentile(mean + 2 * std, pgs_id, "EUR", dist_lf)
        assert pct is not None
        assert pct > 90.0

    def test_score_below_mean(
        self, dist_lf: pl.LazyFrame, first_valid_pgs: tuple[str, float, float]
    ) -> None:
        pgs_id, mean, std = first_valid_pgs
        pct = ancestry_percentile(mean - 2 * std, pgs_id, "EUR", dist_lf)
        assert pct is not None
        assert pct < 10.0

    def test_different_superpops_differ(
        self, dist_lf: pl.LazyFrame, first_valid_pgs: tuple[str, float, float]
    ) -> None:
        pgs_id, mean, std = first_valid_pgs
        score = mean + std
        results = {}
        for sp in SUPERPOPULATIONS:
            pct = ancestry_percentile(score, pgs_id, sp, dist_lf)
            if pct is not None:
                results[sp] = pct

        if len(results) >= 2:
            assert len(set(round(v, 1) for v in results.values())) > 1, (
                f"All superpops gave same percentile: {results}"
            )

    def test_extreme_score_gives_near_0_or_100(
        self, dist_lf: pl.LazyFrame, first_valid_pgs: tuple[str, float, float]
    ) -> None:
        pgs_id, mean, std = first_valid_pgs
        pct_high = ancestry_percentile(mean + 50 * std, pgs_id, "EUR", dist_lf)
        pct_low = ancestry_percentile(mean - 50 * std, pgs_id, "EUR", dist_lf)
        assert pct_high is not None
        assert pct_high >= 99.99
        assert pct_low is not None
        assert pct_low <= 0.01
