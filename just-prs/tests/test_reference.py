"""Integration tests for ancestry-aware PRS percentile estimation.

Tests the reference module pure functions (ancestry_percentile, aggregate_distributions)
and the PRSCatalog.percentile() 3-tier fallback using real data from HuggingFace
(just-dna-seq/prs-percentiles) when available, falling back to synthetic distributions
for unit-level assertions.
"""

import math
import tempfile
from pathlib import Path

import polars as pl
import pytest

from just_prs.models import ReferenceDistribution
from just_prs.prs_catalog import PRSCatalog
from just_prs.reference import (
    SUPERPOPULATIONS,
    _find_reference_panel_file,
    _reference_panel_complete,
    aggregate_distributions,
    ancestry_percentile,
    distribution_quality_issues,
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
# Reference panel file resolution tests
# ---------------------------------------------------------------------------

class TestReferencePanelResolution:
    """Tests for local reference panel file discovery."""

    def test_prefers_grch38_files_over_grch37(self, tmp_path: Path) -> None:
        (tmp_path / "GRCh37_1000G_ALL.pvar.zst").touch()
        expected = tmp_path / "GRCh38_1000G_ALL.pvar.zst"
        expected.touch()

        result = _find_reference_panel_file(tmp_path, "GRCh38", ".pvar.zst")

        assert result == expected

    def test_hg38_token_matches_build_file(self, tmp_path: Path) -> None:
        expected = tmp_path / "panel_hg38.pgen"
        expected.touch()

        result = _find_reference_panel_file(tmp_path, "GRCh38", ".pgen")

        assert result == expected

    def test_incomplete_panel_is_not_valid(self, tmp_path: Path) -> None:
        (tmp_path / "GRCh37_1000G_ALL.pgen").touch()
        (tmp_path / "GRCh37_1000G_ALL.pvar.zst").touch()
        (tmp_path / "GRCh37_1000G_ALL.psam").touch()
        (tmp_path / "GRCh38_1000G_ALL.pgen").touch()

        assert not _reference_panel_complete(tmp_path)


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


class TestDistributionQualityIssues:
    """Tests for distribution-level anomaly reporting."""

    def test_reports_each_nonfinite_and_zero_std_issue(self) -> None:
        distributions = pl.DataFrame({
            "pgs_id": ["PGS_BAD", "PGS_FLAT"],
            "superpopulation": ["EUR", "AFR"],
            "mean": [float("inf"), 0.0],
            "std": [float("nan"), 0.0],
            "n": [633, 893],
            "median": [1.0, 0.0],
            "p5": [0.5, 0.0],
            "p25": [0.8, 0.0],
            "p75": [1.2, 0.0],
            "p95": [1.5, 0.0],
        })

        issues = distribution_quality_issues(distributions)

        assert issues.height == 3
        assert set(issues["issue"].to_list()) == {
            "mean_nonfinite",
            "std_nonfinite",
            "std_zero",
        }
        assert issues.filter(pl.col("severity") == "ERROR").height == 2
        assert issues.filter(pl.col("severity") == "WARN").height == 1

    def test_returns_empty_schema_when_clean(self) -> None:
        distributions = pl.DataFrame({
            "pgs_id": ["PGS_OK"],
            "superpopulation": ["EUR"],
            "mean": [1.0],
            "std": [0.5],
            "n": [633],
            "median": [1.0],
            "p5": [0.2],
            "p25": [0.7],
            "p75": [1.3],
            "p95": [1.8],
        })

        issues = distribution_quality_issues(distributions)

        assert issues.height == 0
        assert {"pgs_id", "superpopulation", "severity", "issue"}.issubset(set(issues.columns))

    def test_reports_finite_but_robustly_absurd_distribution(self) -> None:
        distributions = pl.DataFrame({
            "pgs_id": ["PGS_CORRUPT"],
            "superpopulation": ["EUR"],
            "mean": [1.0e30],
            "std": [1.0e31],
            "n": [633],
            "median": [1.0],
            "p5": [0.5],
            "p25": [0.8],
            "p75": [1.2],
            "p95": [1.5],
        })

        issues = distribution_quality_issues(distributions)

        assert issues.height == 1
        assert issues["issue"][0] == "robust_outlier_suspected"
        assert issues["severity"][0] == "ERROR"


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
        pgs_id = "PGS000001"
        ref_lf = catalog.reference_distributions()
        dist_row = ref_lf.filter(
            (pl.col("pgs_id") == pgs_id) & (pl.col("superpopulation") == "EUR")
        ).select("mean", "std").collect()
        if dist_row.height == 0:
            pytest.skip("No reference distribution for PGS000001")
        score = float(dist_row["mean"][0]) + float(dist_row["std"][0])

        results = {
            sp: catalog.percentile(score, pgs_id, ancestry=sp)
            for sp in SUPERPOPULATIONS
        }
        methods = {sp: m for sp, (_, m) in results.items()}
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

    def test_reference_distributions_filters_untrustworthy_rows(self) -> None:
        """Bad reference rows must not be exposed as trustworthy percentiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            percentiles_dir = cache_dir / "percentiles"
            percentiles_dir.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({
                "pgs_id": ["PGS_BAD", "PGS_BAD", "PGS_OK"],
                "superpopulation": ["EUR", "AFR", "EUR"],
                "mean": [float("inf"), 0.0, 1.0],
                "std": [float("nan"), 0.0, 0.5],
                "n": [633, 893, 633],
                "median": [1.0, 0.0, 1.0],
                "p5": [0.5, 0.0, 0.2],
                "p25": [0.8, 0.0, 0.7],
                "p75": [1.2, 0.0, 1.3],
                "p95": [1.5, 0.0, 1.8],
            }).write_parquet(percentiles_dir / "1000g_distributions.parquet")

            catalog = PRSCatalog(cache_dir=cache_dir)
            filtered = catalog.reference_distributions().collect()

            assert filtered["pgs_id"].to_list() == ["PGS_OK"]

    def test_percentile_refreshes_distributions_on_miss(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When local distributions are stale, percentile() refreshes from HF once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            percentiles_dir = cache_dir / "percentiles"
            percentiles_dir.mkdir(parents=True, exist_ok=True)

            # Start with stale local data that does not include the target PGS.
            stale = pl.DataFrame(
                {
                    "pgs_id": ["PGS_OLD"],
                    "superpopulation": ["EUR"],
                    "mean": [0.0],
                    "std": [1.0],
                    "n": [100],
                    "median": [0.0],
                    "p5": [-1.0],
                    "p25": [-0.5],
                    "p75": [0.5],
                    "p95": [1.0],
                }
            )
            stale.write_parquet(percentiles_dir / "1000g_distributions.parquet")

            fresh = pl.DataFrame(
                {
                    "pgs_id": ["PGS_NEW"],
                    "superpopulation": ["EUR"],
                    "mean": [1.0],
                    "std": [0.5],
                    "n": [500],
                    "median": [1.0],
                    "p5": [0.2],
                    "p25": [0.7],
                    "p75": [1.3],
                    "p95": [1.8],
                }
            )

            pull_calls = {"count": 0}

            def fake_pull_reference_distributions(
                local_dir: Path,
                repo_id: str = "just-dna-seq/prs-percentiles",
                token: str | None = None,
                panel: str = "1000g",
            ) -> Path | None:
                del repo_id, token
                pull_calls["count"] += 1
                target = local_dir / f"{panel}_distributions.parquet"
                fresh.write_parquet(target)
                return target

            monkeypatch.setattr(
                "just_prs.prs_catalog.pull_reference_distributions",
                fake_pull_reference_distributions,
            )

            catalog = PRSCatalog(cache_dir=cache_dir)

            pct, method = catalog.percentile(1.0, "PGS_NEW", ancestry="EUR", panel="1000g")
            assert method == "reference_panel"
            assert pct == pytest.approx(50.0, abs=0.1)
            assert pull_calls["count"] == 1

            # Guarded refresh: repeated calls should not re-pull.
            pct2, method2 = catalog.percentile(1.0, "PGS_NEW", ancestry="EUR", panel="1000g")
            assert method2 == "reference_panel"
            assert pct2 == pytest.approx(50.0, abs=0.1)
            assert pull_calls["count"] == 1
