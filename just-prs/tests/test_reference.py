"""Integration tests for ancestry-aware PRS percentile estimation.

Tests the reference module pure functions (ancestry_percentile, aggregate_distributions)
and the PRSCatalog.percentile() 3-tier fallback using real data from HuggingFace
(just-dna-seq/prs-percentiles) when available, falling back to synthetic distributions
for unit-level assertions.
"""

import math
from pathlib import Path

import polars as pl
import pytest

from just_prs.models import ReferenceDistribution
from just_prs.prs_catalog import PRSCatalog
from just_prs.reference import (
    SUPERPOPULATIONS,
    aggregate_distributions,
    ancestry_percentile,
)
from just_prs.scoring import resolve_cache_dir

TEST_CACHE_DIR = resolve_cache_dir() / "test-data"

# PGS IDs known to have allele frequencies (for tier-2 percentile testing)
PGS_IDS_WITH_FREQ = ["PGS000004", "PGS000005"]
# PGS IDs without allele frequencies (to test tier-1/tier-3 fallback)
PGS_IDS_WITHOUT_FREQ = ["PGS000001", "PGS000002"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_distributions_lf() -> pl.LazyFrame:
    """Build a synthetic reference distributions LazyFrame for unit tests.

    Values are chosen so EUR mean=1.5, std=0.5, which gives predictable percentiles.
    """
    rows = []
    for pgs_id in ["PGS000001", "PGS000002", "PGS000004"]:
        for i, superpop in enumerate(SUPERPOPULATIONS):
            rows.append({
                "pgs_id": pgs_id,
                "superpopulation": superpop,
                "mean": 1.0 + i * 0.1,
                "std": 0.5,
                "n": 500,
                "median": 1.0 + i * 0.1,
                "p5": 0.2 + i * 0.1,
                "p25": 0.7 + i * 0.1,
                "p75": 1.3 + i * 0.1,
                "p95": 1.8 + i * 0.1,
            })
    return pl.DataFrame(rows).lazy()


@pytest.fixture(scope="session")
def catalog() -> PRSCatalog:
    """PRSCatalog instance using the test cache dir."""
    return PRSCatalog(cache_dir=TEST_CACHE_DIR)


# ---------------------------------------------------------------------------
# ancestry_percentile unit tests (synthetic data)
# ---------------------------------------------------------------------------

class TestAncestryPercentile:
    """Tests for ancestry_percentile() using synthetic distributions."""

    def test_returns_50th_at_mean(self, synthetic_distributions_lf: pl.LazyFrame) -> None:
        """Score equal to mean → 50th percentile."""
        pct = ancestry_percentile(1.0, "PGS000001", "AFR", synthetic_distributions_lf)
        assert pct is not None
        assert abs(pct - 50.0) < 0.1

    def test_returns_high_percentile_above_mean(self, synthetic_distributions_lf: pl.LazyFrame) -> None:
        """Score 2 SD above EUR mean → above 95th percentile."""
        eur_mean = 1.4  # EUR is index 3
        pct = ancestry_percentile(eur_mean + 2 * 0.5, "PGS000001", "EUR", synthetic_distributions_lf)
        assert pct is not None
        assert pct > 95.0, f"Expected > 95th percentile, got {pct}"

    def test_returns_low_percentile_below_mean(self, synthetic_distributions_lf: pl.LazyFrame) -> None:
        """Score 2 SD below EUR mean → below 5th percentile."""
        eur_mean = 1.4
        pct = ancestry_percentile(eur_mean - 2 * 0.5, "PGS000001", "EUR", synthetic_distributions_lf)
        assert pct is not None
        assert pct < 5.0, f"Expected < 5th percentile, got {pct}"

    def test_returns_none_for_missing_pgs_id(self, synthetic_distributions_lf: pl.LazyFrame) -> None:
        """Unknown PGS ID → None."""
        pct = ancestry_percentile(1.0, "PGS999999", "EUR", synthetic_distributions_lf)
        assert pct is None

    def test_returns_none_for_missing_superpop(self, synthetic_distributions_lf: pl.LazyFrame) -> None:
        """Invalid superpopulation code → None."""
        pct = ancestry_percentile(1.0, "PGS000001", "XYZ", synthetic_distributions_lf)
        assert pct is None

    def test_all_superpopulations_give_different_percentiles(
        self, synthetic_distributions_lf: pl.LazyFrame
    ) -> None:
        """Same score against different ancestry groups should yield different percentiles."""
        score = 1.3
        percentiles = {
            sp: ancestry_percentile(score, "PGS000001", sp, synthetic_distributions_lf)
            for sp in SUPERPOPULATIONS
        }
        # All should be non-None
        assert all(v is not None for v in percentiles.values()), percentiles
        # At least 2 different values (means differ by 0.1 per group)
        assert len(set(round(v, 2) for v in percentiles.values())) > 1, percentiles

    def test_percentile_in_valid_range(self, synthetic_distributions_lf: pl.LazyFrame) -> None:
        """Percentile should always be in [0, 100]."""
        for score in [-5.0, 0.0, 1.0, 2.0, 10.0]:
            pct = ancestry_percentile(score, "PGS000001", "EUR", synthetic_distributions_lf)
            assert pct is not None
            assert 0.0 <= pct <= 100.0, f"score={score} → pct={pct} out of range"


# ---------------------------------------------------------------------------
# aggregate_distributions unit tests
# ---------------------------------------------------------------------------

class TestAggregateDistributions:
    """Tests for aggregate_distributions() using synthetic per-individual data."""

    def test_produces_correct_columns(self) -> None:
        """Output has all expected columns."""
        scores_df = pl.DataFrame({
            "pgs_id": ["PGS000001"] * 10,
            "iid": [f"NA{i:05d}" for i in range(10)],
            "superpop": ["EUR"] * 5 + ["AFR"] * 5,
            "population": ["GBR"] * 5 + ["YRI"] * 5,
            "score": [1.0, 1.1, 1.2, 0.9, 1.05, 0.8, 0.9, 0.85, 0.95, 0.7],
        })
        result = aggregate_distributions(scores_df)
        expected_cols = {"pgs_id", "superpopulation", "mean", "std", "n", "median", "p5", "p25", "p75", "p95"}
        assert expected_cols.issubset(set(result.columns)), result.columns

    def test_correct_n_per_group(self) -> None:
        """Each group should have the correct count."""
        scores_df = pl.DataFrame({
            "pgs_id": ["PGS000001"] * 10,
            "iid": [f"NA{i:05d}" for i in range(10)],
            "superpop": ["EUR"] * 6 + ["AFR"] * 4,
            "population": ["GBR"] * 6 + ["YRI"] * 4,
            "score": list(range(10)),
        })
        result = aggregate_distributions(scores_df)
        eur_row = result.filter(pl.col("superpopulation") == "EUR")
        afr_row = result.filter(pl.col("superpopulation") == "AFR")
        assert eur_row["n"][0] == 6
        assert afr_row["n"][0] == 4

    def test_mean_is_correct(self) -> None:
        """Computed mean should match manual calculation."""
        eur_scores = [1.0, 2.0, 3.0]
        scores_df = pl.DataFrame({
            "pgs_id": ["PGS000001"] * 3,
            "iid": ["NA00001", "NA00002", "NA00003"],
            "superpop": ["EUR"] * 3,
            "population": ["GBR"] * 3,
            "score": eur_scores,
        })
        result = aggregate_distributions(scores_df)
        assert abs(result["mean"][0] - 2.0) < 1e-10

    def test_multiple_pgs_ids_are_separate(self) -> None:
        """Distributions for different PGS IDs are computed independently."""
        scores_df = pl.DataFrame({
            "pgs_id": ["PGS000001"] * 5 + ["PGS000002"] * 5,
            "iid": [f"NA{i:05d}" for i in range(10)],
            "superpop": ["EUR"] * 10,
            "population": ["GBR"] * 10,
            "score": [1.0] * 5 + [2.0] * 5,
        })
        result = aggregate_distributions(scores_df)
        assert result.filter(pl.col("pgs_id") == "PGS000001")["mean"][0] == pytest.approx(1.0)
        assert result.filter(pl.col("pgs_id") == "PGS000002")["mean"][0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# ReferenceDistribution model tests
# ---------------------------------------------------------------------------

class TestReferenceDistributionModel:
    """Tests for the ReferenceDistribution Pydantic model."""

    def test_valid_model(self) -> None:
        rd = ReferenceDistribution(
            pgs_id="PGS000001",
            superpopulation="EUR",
            mean=1.5,
            std=0.5,
            n=503,
        )
        assert rd.pgs_id == "PGS000001"
        assert rd.superpopulation == "EUR"
        assert rd.n == 503

    def test_optional_percentile_fields_default_none(self) -> None:
        rd = ReferenceDistribution(
            pgs_id="PGS000001",
            superpopulation="AFR",
            mean=0.8,
            std=0.4,
            n=661,
        )
        assert rd.median is None
        assert rd.p5 is None
        assert rd.p95 is None


# ---------------------------------------------------------------------------
# PRSCatalog.percentile() 3-tier fallback tests (uses HF data if available)
# ---------------------------------------------------------------------------

class TestPRSCatalogPercentile:
    """Tests for the 3-tier percentile fallback in PRSCatalog.percentile()."""

    def test_returns_tuple(self, catalog: PRSCatalog) -> None:
        """percentile() always returns a (value, method) tuple."""
        result = catalog.percentile(1.5, "PGS000001", ancestry="EUR")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_method_label_is_known_value(self, catalog: PRSCatalog) -> None:
        """Method label must be one of the documented values."""
        _, method = catalog.percentile(1.5, "PGS000001", ancestry="EUR")
        assert method in {"reference_panel", "theoretical", "auroc_approx", "unavailable"}

    def test_explicit_std_uses_theoretical(self, catalog: PRSCatalog) -> None:
        """When std is provided explicitly, method must be 'theoretical'."""
        pct, method = catalog.percentile(1.5, "PGS000001", ancestry="EUR", mean=1.0, std=0.5)
        assert method == "theoretical"
        assert pct is not None
        # 1 SD above mean → ~84th percentile (rational approx has ~1% error)
        assert 82.0 < pct < 86.0, f"Expected ~84th percentile, got {pct}"

    def test_explicit_std_computes_correct_percentile(self, catalog: PRSCatalog) -> None:
        """Explicit mean/std gives Phi((score-mean)/std)*100."""
        score, mean, std = 2.0, 1.0, 1.0
        pct, _ = catalog.percentile(score, "PGS000001", ancestry="EUR", mean=mean, std=std)
        z = (score - mean) / std
        expected = round(0.5 * math.erfc(-z / math.sqrt(2)) * 100.0, 2)
        assert pct == pytest.approx(expected)

    def test_different_ancestries_may_differ(self, catalog: PRSCatalog) -> None:
        """Percentile for the same score and PGS ID may differ across ancestries
        when reference panel data is available."""
        score = 1.5
        pgs_id = "PGS000001"
        results = {
            sp: catalog.percentile(score, pgs_id, ancestry=sp)
            for sp in SUPERPOPULATIONS
        }
        methods = {sp: m for sp, (_, m) in results.items()}
        # If all used reference_panel, percentiles should differ (different distributions)
        ref_panel_used = all(m == "reference_panel" for m in methods.values())
        if ref_panel_used:
            percentiles = {sp: p for sp, (p, _) in results.items() if p is not None}
            assert len(set(round(p, 1) for p in percentiles.values())) > 1, (
                "Reference panel percentiles should differ across ancestries"
            )

    def test_unknown_pgs_id_does_not_crash(self, catalog: PRSCatalog) -> None:
        """A PGS ID not in any tier still returns a (None, 'unavailable') tuple."""
        pct, method = catalog.percentile(0.0, "PGS999999", ancestry="EUR")
        assert pct is None
        assert method == "unavailable"
