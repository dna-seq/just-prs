"""Raw-score units cannot be inferred from discrimination (AUROC) alone."""

from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest

from just_prs.models import AbsoluteRiskBundle
from just_prs.prs import _norm_cdf
from just_prs.prs_catalog import PRSCatalog


@pytest.fixture
def catalog(tmp_path: Path) -> PRSCatalog:
    """Seed real cached frames; exercise lookups/math without network or genomes."""
    cat = PRSCatalog(cache_dir=tmp_path)
    cat._scores_lf = pl.LazyFrame({"pgs_id": ["PGS_SYNTHETIC"]})
    cat._best_perf_lf = pl.LazyFrame({
        "pgs_id": ["PGS_SYNTHETIC"], "auroc_estimate": [0.7],
    })
    cat._ref_dist_cache["1000g"] = pl.LazyFrame(schema={
        "pgs_id": pl.String, "superpopulation": pl.String,
        "mean": pl.Float64, "std": pl.Float64,
    })
    # Model an already-completed refresh that found no reference for this score.
    cat._ref_dist_refresh_attempted.add("1000g")
    return cat


@pytest.mark.parametrize("score", [1.0, 5.0, 10.0])
def test_auroc_cannot_calibrate_a_fully_covered_raw_score(
    catalog: PRSCatalog, score: float,
) -> None:
    result = catalog.percentile_full(
        score, "PGS_SYNTHETIC", weight_mass_coverage=1.0, user_match_rate=1.0,
    )
    assert result.method == "auroc_approx"
    assert result.percentile is None
    assert result.z_score is None
    assert result.reference_mean is None
    assert result.reference_std is None
    assert result.ancestry is None and result.panel is None
    assert result.reliable is False
    assert "AUROC" in result.caveat
    assert catalog.percentile(score, "PGS_SYNTHETIC") == (None, "auroc_approx")


def test_unavailable_percentile_is_not_reliable(catalog: PRSCatalog) -> None:
    result = catalog.percentile_full(5.0, "PGS_UNKNOWN", weight_mass_coverage=1.0)
    assert result.method == "unavailable"
    assert result.percentile is None
    assert result.z_score is None
    assert result.reliable is False
    assert result.caveat


def test_explicit_calibration_is_invariant_under_positive_affine_transform(
    catalog: PRSCatalog,
) -> None:
    original = catalog.percentile_full(3.0, "PGS_SYNTHETIC", mean=1.0, std=2.0)
    transformed = catalog.percentile_full(
        3.0 * 10.0 + 100.0, "PGS_SYNTHETIC", mean=1.0 * 10.0 + 100.0, std=20.0,
    )
    assert original.reliable and transformed.reliable
    assert original.z_score == transformed.z_score == 1.0
    assert original.percentile == transformed.percentile == round(_norm_cdf(1.0) * 100, 2)
    assert original.method == transformed.method == "theoretical"


def _add_reference(catalog: PRSCatalog) -> None:
    catalog._ref_dist_cache["1000g"] = pl.LazyFrame({
        "pgs_id": ["PGS_SYNTHETIC"], "superpopulation": ["EUR"],
        "mean": [1.0], "std": [2.0],
    })


def test_reference_calibration_and_provenance_are_preserved(catalog: PRSCatalog) -> None:
    _add_reference(catalog)
    result = catalog.percentile_full(3.0, "PGS_SYNTHETIC", weight_mass_coverage=1.0)
    assert result.method == "reference_panel"
    assert result.z_score == 1.0
    assert result.percentile == round(_norm_cdf(1.0) * 100, 2)
    assert (result.reference_mean, result.reference_std) == (1.0, 2.0)
    assert (result.ancestry, result.panel) == ("EUR", "1000g")
    assert result.reliable and not result.caveat


@pytest.mark.parametrize("has_reference", [False, True])
def test_absolute_risk_does_not_consume_unreliable_calibration(
    catalog: PRSCatalog, monkeypatch: pytest.MonkeyPatch, has_reference: bool,
) -> None:
    if has_reference:
        _add_reference(catalog)
    # Only spy on the downstream consumer; percentile lookup/math execute normally.
    downstream = Mock(side_effect=AssertionError("unreliable calibration was consumed"))
    monkeypatch.setattr(catalog, "absolute_risk_bundle", downstream)
    result = catalog.absolute_risk_from_score(
        "PGS_SYNTHETIC", 5.0, weight_mass_coverage=0.05 if has_reference else 1.0,
    )
    assert result == AbsoluteRiskBundle()
    downstream.assert_not_called()


def test_absolute_risk_still_receives_reliable_z_score(
    catalog: PRSCatalog, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_reference(catalog)
    expected = AbsoluteRiskBundle()
    downstream = Mock(return_value=expected)
    monkeypatch.setattr(catalog, "absolute_risk_bundle", downstream)
    result = catalog.absolute_risk_from_score("PGS_SYNTHETIC", 3.0, weight_mass_coverage=1.0)
    assert result is expected
    downstream.assert_called_once_with("PGS_SYNTHETIC", 1.0, sex=None, selected_ancestry="EUR")
