import math

import polars as pl
import pytest

from just_prs.prs_catalog import PRSCatalog
from just_prs.quality import classify_model_quality, synthetic_quality_score
from just_prs.scoring import resolve_cache_dir


def _quality_comparison_df() -> pl.DataFrame:
    """Build the reproducible quality comparison table for filtered GRCh38 PGS."""
    catalog = PRSCatalog(cache_dir=resolve_cache_dir())
    scores = catalog.scores(genome_build="GRCh38").select(
        "pgs_id",
        "trait_reported",
        "genome_build",
        "n_variants",
    )
    best_performance = catalog.best_performance().select(
        "pgs_id",
        "auroc_estimate",
        "cindex_estimate",
        "or_estimate",
        "hr_estimate",
        "beta_estimate",
        "n_individuals",
        "ancestry_broad",
    )

    rows: list[dict] = []
    joined = scores.join(best_performance, on="pgs_id", how="left").collect()
    for row in joined.iter_rows(named=True):
        auroc = row.get("auroc_estimate")
        cindex = row.get("cindex_estimate")
        or_est = row.get("or_estimate")
        hr_est = row.get("hr_estimate")
        beta_est = row.get("beta_estimate")
        n_individuals = row.get("n_individuals")
        quality_score = synthetic_quality_score(
            auroc=auroc,
            cindex=cindex,
            or_estimate=or_est,
            hr_estimate=hr_est,
            beta_estimate=beta_est,
            n_individuals=n_individuals,
        )
        coarse_label, _ = classify_model_quality(coverage=1.0, auroc=auroc)
        if auroc is not None:
            tier = "T1a-AUROC"
        elif cindex is not None:
            tier = "T1a-Cindex"
        elif beta_est is not None:
            tier = "T1b-beta"
        elif or_est is not None or hr_est is not None:
            tier = "T2-OR/HR"
        else:
            tier = "T3-none"
        rows.append({
            **row,
            "synthetic_quality_score": quality_score,
            "coarse_quality_label": coarse_label,
            "tier": tier,
        })

    return pl.DataFrame(rows).sort("synthetic_quality_score", descending=True)


# ---------------------------------------------------------------------------
# Tier 1a — AUROC / C-index
# ---------------------------------------------------------------------------

def test_tier1a_auroc_small_and_large_cohort() -> None:
    # Power-stretch: disc = 0.5 + 0.5*(0.4^0.78) ≈ 0.745
    # log10(1000)/5.5 ≈ 0.545, log10(1e6)/5.5 clamped to 1.0
    assert synthetic_quality_score(auroc=0.7, n_individuals=1_000) == pytest.approx(40.6)
    assert synthetic_quality_score(auroc=0.7, n_individuals=1_000_000) == pytest.approx(74.5)


def test_tier1a_cindex_fallback() -> None:
    # Power-stretch on cindex=0.66: disc = 0.5 + 0.5*(0.32^0.78) ≈ 0.706
    # cohort = log10(1e5)/5.5 ≈ 0.909
    assert synthetic_quality_score(cindex=0.66, n_individuals=100_000) == pytest.approx(64.1)


def test_tier1a_auroc_preferred_over_cindex() -> None:
    auroc_only = synthetic_quality_score(auroc=0.75, n_individuals=50_000)
    both = synthetic_quality_score(auroc=0.75, cindex=0.60, n_individuals=50_000)
    assert auroc_only == both


# ---------------------------------------------------------------------------
# Tier 1b — beta (continuous traits), 0.95 penalty
# ---------------------------------------------------------------------------

def test_tier1b_beta_with_penalty() -> None:
    # disc = 0.5 + 0.5*tanh(ln(1.24)) ≈ 0.606, penalty=0.95
    # cohort = log10(1e5)/5.5 ≈ 0.909
    score = synthetic_quality_score(beta_estimate=0.24, n_individuals=100_000)
    assert score == pytest.approx(52.3)


def test_tier1b_beta_saturates_for_extreme() -> None:
    # tanh(ln(1+152.57)) saturates near 1.0 → disc ≈ 1.0
    # tanh(ln(1.5)) ≈ 0.385 → disc ≈ 0.693
    # Extreme beta >> moderate beta (no clamping at 0.5)
    big = synthetic_quality_score(beta_estimate=152.57, n_individuals=100_000)
    moderate = synthetic_quality_score(beta_estimate=0.5, n_individuals=100_000)
    assert big > moderate
    assert big == pytest.approx(86.4)
    assert moderate == pytest.approx(59.8)


def test_tier1b_beta_lower_than_equivalent_auroc() -> None:
    # Beta has 0.95 penalty vs AUROC's 1.0, plus different discrimination mapping
    beta_score = synthetic_quality_score(beta_estimate=0.3, n_individuals=50_000)
    auroc_score = synthetic_quality_score(auroc=0.8, n_individuals=50_000)
    assert beta_score < auroc_score


# ---------------------------------------------------------------------------
# Tier 2 — OR/HR → approximate AUROC, 0.90 penalty
# ---------------------------------------------------------------------------

def test_tier2_or_applies_penalty() -> None:
    # OR=1.5, disc = Φ(ln(1.5)/2) ≈ 0.613, penalty=0.90
    # cohort = log10(1000)/5.5 ≈ 0.545
    assert synthetic_quality_score(or_estimate=1.5, n_individuals=1_000) == pytest.approx(30.1)


def test_tier2_or_lower_than_same_auroc_direct() -> None:
    approx_auroc = 0.5 * (1 + math.erf(math.log(1.5) / (math.sqrt(2) * math.sqrt(2))))
    direct = synthetic_quality_score(auroc=approx_auroc, n_individuals=100_000)
    via_or = synthetic_quality_score(or_estimate=1.5, n_individuals=100_000)
    assert via_or < direct


# ---------------------------------------------------------------------------
# Tier 3 — no metric at all, 0.51 floor, 0.6 penalty
# ---------------------------------------------------------------------------

def test_tier3_floor_value() -> None:
    # No metric → discrimination=0.51, penalty=0.6, cohort_factor=0.5 (no n)
    assert synthetic_quality_score() == pytest.approx(15.3)
    assert synthetic_quality_score(n_individuals=None) == pytest.approx(15.3)


def test_tier3_lower_than_tier2_at_same_cohort() -> None:
    tier3 = synthetic_quality_score(n_individuals=500_000)
    tier2 = synthetic_quality_score(or_estimate=1.1, n_individuals=500_000)
    assert tier3 < tier2


# ---------------------------------------------------------------------------
# Coarse classifier must not change
# ---------------------------------------------------------------------------

def test_synthetic_quality_score_does_not_change_coarse_classifier() -> None:
    assert classify_model_quality(coverage=0.5, auroc=0.7) == ("High", "green")
    assert classify_model_quality(coverage=0.5, auroc=0.59) == ("Moderate", "yellow")
    assert classify_model_quality(coverage=0.09, auroc=0.9) == ("Very Low", "red")


def test_filtered_grch38_numeric_quality_concords_with_coarse_low_grades() -> None:
    comparison = _quality_comparison_df()

    assert comparison.height >= 600

    # With the four-tier formula no score returns 0 (Tier 3 floor = 0.51),
    # so the median is now a real positive number
    median_score = comparison["synthetic_quality_score"].median()
    mean_score = comparison["synthetic_quality_score"].mean()
    assert median_score is not None and median_score > 0
    assert mean_score is not None

    # Tier coverage — all tiers must be represented in GRCh38 set
    tiers_present = set(comparison["tier"].to_list())
    assert "T1a-AUROC" in tiers_present
    assert "T1b-beta" in tiers_present
    assert "T3-none" in tiers_present

    binned = comparison.with_columns(
        pl.when(pl.col("synthetic_quality_score") <= median_score)
        .then(pl.lit("low"))
        .when(pl.col("synthetic_quality_score") > mean_score)
        .then(pl.lit("high"))
        .otherwise(pl.lit("middle"))
        .alias("numeric_quality_bin")
    )

    assert binned.filter(pl.col("numeric_quality_bin") == "high").height > 0
    assert binned.filter(pl.col("numeric_quality_bin") == "low").height > 0

    # The core concordance invariant: high numeric score must never carry a
    # coarse "Low" or "Very Low" label (coarse classifier uses only AUROC+match_rate)
    high_numeric_low_grade = binned.filter(
        (pl.col("numeric_quality_bin") == "high")
        & pl.col("coarse_quality_label").is_in(["Low", "Very Low"])
    )
    assert high_numeric_low_grade.is_empty(), high_numeric_low_grade.select(
        "pgs_id", "trait_reported", "synthetic_quality_score",
        "coarse_quality_label", "auroc_estimate", "cindex_estimate",
        "or_estimate", "hr_estimate", "beta_estimate",
        "n_individuals", "tier",
    ).to_dicts()


# ---------------------------------------------------------------------------
# Harmonized score penalty
# ---------------------------------------------------------------------------

def test_harmonized_penalty_reduces_score() -> None:
    """Harmonized scores should receive a multiplicative penalty."""
    native = synthetic_quality_score(auroc=0.7, n_individuals=1_000_000)
    harmonized = synthetic_quality_score(auroc=0.7, n_individuals=1_000_000, is_harmonized=True)
    assert harmonized < native
    assert harmonized == pytest.approx(native * 0.90, abs=0.15)


def test_harmonized_penalty_applies_to_all_tiers() -> None:
    """Penalty applies across all discrimination tiers."""
    for kwargs in [
        {"auroc": 0.8, "n_individuals": 50_000},
        {"beta_estimate": 0.3, "n_individuals": 50_000},
        {"or_estimate": 1.5, "n_individuals": 1_000},
        {},  # Tier 3: no metric
    ]:
        native = synthetic_quality_score(**kwargs)
        harmonized = synthetic_quality_score(**kwargs, is_harmonized=True)
        assert harmonized < native, f"Failed for {kwargs}"


def test_harmonized_false_is_identity() -> None:
    """is_harmonized=False must not change the score at all."""
    score_default = synthetic_quality_score(auroc=0.7, n_individuals=100_000)
    score_explicit = synthetic_quality_score(auroc=0.7, n_individuals=100_000, is_harmonized=False)
    assert score_default == score_explicit


# ---------------------------------------------------------------------------
# Catalog harmonized filtering
# ---------------------------------------------------------------------------

def test_catalog_include_harmonized_grch38() -> None:
    """With include_harmonized=True, GRCh38 should return many more scores."""
    catalog = PRSCatalog(cache_dir=resolve_cache_dir())
    native_df = catalog.scores(genome_build="GRCh38", include_harmonized=False).collect()
    harmonized_df = catalog.scores(genome_build="GRCh38", include_harmonized=True).collect()
    assert harmonized_df.height > native_df.height
    assert "is_harmonized" in native_df.columns
    assert "is_harmonized" in harmonized_df.columns
    assert all(~native_df["is_harmonized"])
    assert harmonized_df.filter(pl.col("is_harmonized")).height > 0


def test_catalog_include_harmonized_has_is_harmonized_column() -> None:
    """is_harmonized column should be present regardless of include_harmonized setting."""
    catalog = PRSCatalog(cache_dir=resolve_cache_dir())
    lf_with = catalog.scores(genome_build="GRCh38", include_harmonized=True)
    lf_without = catalog.scores(genome_build="GRCh38", include_harmonized=False)
    lf_no_build = catalog.scores()
    assert "is_harmonized" in lf_with.collect_schema().names()
    assert "is_harmonized" in lf_without.collect_schema().names()
    assert "is_harmonized" in lf_no_build.collect_schema().names()
