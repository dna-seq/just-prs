"""Canonical trait-summary aggregation: scope, match-rate scale, ancestry order."""

from just_prs.enrich import _compute_z_score, apply_absolute_risk_to_row
from just_prs.trait_summary import (
    NO_RISK_DATA_USABLE,
    is_high_or_moderate_model,
    is_high_quality_model,
    is_usable_model,
    normalize_match_rate,
    prefer_prevalence_row,
    summarize_heritability,
    summarize_trait_rows,
)
from just_prs.viz import build_prs_ai_prompt


def _row(
    pgs_id: str,
    *,
    match_rate: float,
    percentile: float = 60.0,
    quality_label: str = "Moderate",
    synthetic_quality: float = 50.0,
    absolute_risk: float | str | None = None,
    population_prevalence: float | None = None,
    risk_ratio: float | None = None,
    selected_ancestry: str = "EUR",
) -> dict:
    row: dict = {
        "pgs_id": pgs_id,
        "percentile": percentile,
        "match_rate": match_rate,
        "quality_label": quality_label,
        "synthetic_quality": synthetic_quality,
        "selected_ancestry": selected_ancestry,
    }
    if absolute_risk is not None:
        row["absolute_risk"] = absolute_risk
    if population_prevalence is not None:
        row["population_prevalence"] = population_prevalence
    if risk_ratio is not None:
        row["risk_ratio"] = risk_ratio
    return row


def test_normalize_match_rate_accepts_fraction_and_percent() -> None:
    assert normalize_match_rate(0.30) == 30.0
    assert normalize_match_rate(0.70) == 70.0
    assert normalize_match_rate(30) == 30.0
    assert normalize_match_rate(70) == 70.0
    assert normalize_match_rate("70%") == 70.0


def test_usable_scope_requires_fifty_percent_coverage() -> None:
    low = _row("PGS_LOW", match_rate=0.30)
    high = _row("PGS_HIGH", match_rate=0.70)
    assert is_usable_model(low) is False
    assert is_usable_model(high) is True
    assert is_usable_model(_row("PGS_PCT", match_rate=30)) is False
    assert is_usable_model(_row("PGS_PCT_OK", match_rate=70)) is True


def test_usable_scope_excludes_only_absolute_risk_model_below_50() -> None:
    """The only model with absolute risk has match <50% — usable must not use it."""
    rows = [
        _row("PGS_OK", match_rate=0.80, percentile=55.0, synthetic_quality=70.0),
        _row(
            "PGS_RISK",
            match_rate=0.30,
            percentile=99.0,
            quality_label="Low",
            synthetic_quality=10.0,
            absolute_risk=0.25,
            population_prevalence=0.08,
            risk_ratio=3.1,
        ),
    ]

    usable = summarize_trait_rows(rows, model_scope="usable", selected_ancestry="EUR")
    assert [row["pgs_id"] for row in usable.usable_rows] == ["PGS_OK"]
    assert usable.best_risk_row is None
    assert usable.absolute_risk == NO_RISK_DATA_USABLE
    assert "based on 1 usable model" in usable.scope_label

    all_scope = summarize_trait_rows(rows, model_scope="all", selected_ancestry="EUR")
    assert all_scope.best_risk_row is not None
    assert all_scope.best_risk_row["pgs_id"] == "PGS_RISK"
    assert all_scope.absolute_risk != NO_RISK_DATA_USABLE


def test_high_quality_ignores_published_synthetic_label() -> None:
    """A famous published model is not High when this sample's coverage is poor.

    The high_moderate scope shares the same coverage gate: a 42.8%-coverage
    "Moderate" row must not slip into the high + moderate dashboard scope and
    drag the median (the 0th-percentile / 42.8%-coverage regression).
    """
    row = _row("PGS003724", match_rate=0.428, quality_label="Moderate")
    row["synthetic_quality_label"] = "High"
    row["synthetic_quality"] = 90.0
    assert is_high_quality_model(row) is False
    assert is_high_or_moderate_model(row) is False
    usable = _row("PGS_OK", match_rate=0.80, quality_label="Moderate")
    assert is_high_or_moderate_model(usable) is True


def test_high_quality_requires_usable_coverage() -> None:
    """Even a High stamp is rejected below 50% match (PGS003724 / 42.8%)."""
    stamped_high = _row(
        "PGS003724",
        match_rate=0.428,
        percentile=0.0,
        quality_label="High",
        synthetic_quality=90.0,
    )
    well_covered = _row(
        "PGS_OK",
        match_rate=0.95,
        percentile=83.0,
        quality_label="Moderate",
        synthetic_quality=50.0,
    )
    assert is_high_quality_model(stamped_high) is False
    stats = summarize_trait_rows(
        [stamped_high, well_covered],
        model_scope="all",
        selected_ancestry="EUR",
    )
    assert stats.n_high_quality == 0
    assert stats.high_quality_median is None
    assert stats.best_row is not None
    assert stats.best_row["pgs_id"] == "PGS_OK"


def test_high_quality_scope_with_no_high_models_is_na() -> None:
    """No High models → no selected median, no borrowed All-models number."""
    rows = [
        _row("PGS_MOD", match_rate=0.80, percentile=60.0, quality_label="Moderate"),
        _row("PGS_LOW", match_rate=0.85, percentile=10.0, quality_label="Low"),
    ]
    high_only = summarize_trait_rows(rows, model_scope="high_quality", selected_ancestry="EUR")
    assert high_only.n_scoped == 0
    assert high_only.n_high_quality == 0
    assert high_only.median_pct is None
    assert high_only.high_quality_median is None
    assert high_only.best_row is None
    assert high_only.best_risk_row is None
    assert high_only.absolute_risk.startswith("N/A")
    assert "0 high-quality" in high_only.scope_label
    assert high_only.overall_signal == "No models in scope"
    assert high_only.consistency == "No models in scope"

    high_mod = summarize_trait_rows(rows, model_scope="high_moderate", selected_ancestry="EUR")
    assert high_mod.n_scoped == 1
    assert high_mod.median_pct == 60.0
    assert high_mod.high_quality_median is None

    prompt = build_prs_ai_prompt(
        "trait_results",
        user_results=rows,
        trait="intelligence",
        model_scope="high_quality",
    )
    assert "Scoped median percentile: N/A" in prompt
    assert "Percentile range: N/A (no models in scope)" in prompt


def test_median_recomputes_when_quality_scope_changes() -> None:
    """Typical / Median is the median of the Quality-dropdown selection."""
    rows = [
        _row("PGS_HIGH", match_rate=0.90, percentile=90.0, quality_label="High"),
        _row("PGS_MOD", match_rate=0.80, percentile=60.0, quality_label="Moderate"),
        _row("PGS_LOW", match_rate=0.85, percentile=10.0, quality_label="Low"),
    ]
    all_models = summarize_trait_rows(rows, model_scope="all", selected_ancestry="EUR")
    high_mod = summarize_trait_rows(rows, model_scope="high_moderate", selected_ancestry="EUR")
    high_only = summarize_trait_rows(rows, model_scope="high_quality", selected_ancestry="EUR")
    assert all_models.median_pct == 60.0
    assert high_mod.median_pct == 75.0
    assert high_only.median_pct == 90.0
    assert all_models.n_scoped == 3
    assert high_mod.n_scoped == 2
    assert high_only.n_scoped == 1


def test_high_moderate_scope_keeps_high_and_moderate_only() -> None:
    rows = [
        _row("PGS_HIGH", match_rate=0.90, quality_label="High", synthetic_quality=80.0),
        _row("PGS_MOD", match_rate=0.80, quality_label="Moderate", synthetic_quality=50.0),
        _row("PGS_LOW", match_rate=0.85, quality_label="Low", synthetic_quality=20.0),
    ]
    assert is_high_or_moderate_model(rows[0]) is True
    assert is_high_or_moderate_model(rows[1]) is True
    assert is_high_or_moderate_model(rows[2]) is False
    stats = summarize_trait_rows(rows, model_scope="high_moderate", selected_ancestry="EUR")
    assert [row["pgs_id"] for row in stats.scoped_rows] == ["PGS_HIGH", "PGS_MOD"]
    assert "high + moderate" in stats.scope_label


def test_heritability_orders_selected_ancestry_before_combined_and_afr() -> None:
    rows = [
        {
            "pgs_id": "PGS000001",
            "match_rate": 0.80,
            "percentile": 60.0,
            "heritability_metrics": [
                {"population": "African", "h2": "0.210", "source": "Pan-UKBB"},
                {"population": "Admixed American", "h2": "0.180", "source": "Pan-UKBB"},
                {"population": "East Asian", "h2": "0.190", "source": "Pan-UKBB"},
                {"population": "South Asian", "h2": "0.200", "source": "Pan-UKBB"},
                {"population": "Combined population", "h2": "0.400", "source": "Pan-UKBB"},
                {"population": "European", "h2": "0.550", "source": "Pan-UKBB"},
            ],
        }
    ]

    text, _detail, metrics = summarize_heritability(rows, selected_ancestry="EUR")
    populations = [str(metric["population"]) for metric in metrics]
    assert populations[0] == "European"
    assert populations[1] == "Combined population"
    assert populations[2] != "African" or populations[0] == "European"
    assert "African" not in text.split(";")[0]
    assert text.startswith("European h²=0.550 (Pan-UKBB)")
    assert "Combined population h²=0.400" in text
    # Truncation is after the sort, so EUR survives even with 6 populations.
    assert "European" in text
    assert "+2 more" in text

    stats = summarize_trait_rows(rows, model_scope="usable", selected_ancestry="EUR")
    assert stats.heritability_metrics[0]["population"] == "European"
    assert stats.heritability_metrics[1]["population"] == "Combined population"

    selected_text, _selected_detail, selected_metrics = summarize_heritability(
        rows, selected_ancestry="EUR", restrict_to_selected=True,
    )
    assert [str(metric["population"]) for metric in selected_metrics] == ["European"]
    assert selected_text == "European h²=0.550 (Pan-UKBB)"
    assert "Combined" not in selected_text

    selected_stats = summarize_trait_rows(
        rows,
        model_scope="usable",
        selected_ancestry="EUR",
        percentile_source="selected",
    )
    assert [str(metric["population"]) for metric in selected_stats.heritability_metrics] == [
        "European",
    ]
    assert "Combined" not in selected_stats.heritability_text


def test_heritability_falls_back_to_combined_when_selected_missing() -> None:
    rows = [
        {
            "pgs_id": "PGS000001",
            "match_rate": 0.80,
            "percentile": 60.0,
            "heritability_metrics": [
                {"population": "African", "h2": "0.210", "source": "Pan-UKBB"},
                {"population": "Combined population", "h2": "0.400", "source": "Pan-UKBB"},
            ],
        }
    ]
    text, _detail, metrics = summarize_heritability(
        rows, selected_ancestry="EUR", restrict_to_selected=True,
    )
    assert [str(metric["population"]) for metric in metrics] == ["Combined population"]
    assert text == "Combined population h²=0.400 (Pan-UKBB)"


def test_risk_vs_average_follows_quality_scope_not_stale_ratio() -> None:
    """Risk vs Average must change when the Quality dropdown changes the best model."""
    rows = [
        _row(
            "PGS_HIGH",
            match_rate=0.95,
            percentile=90.0,
            quality_label="High",
            synthetic_quality=80.0,
            absolute_risk=0.24,
            population_prevalence=0.10,
            risk_ratio=2.40,
        ),
        _row(
            "PGS_MOD",
            match_rate=0.90,
            percentile=55.0,
            quality_label="Moderate",
            synthetic_quality=50.0,
            absolute_risk=0.10,
            population_prevalence=0.10,
            risk_ratio=1.00,
        ),
        _row(
            "PGS_LOW",
            match_rate=0.88,
            percentile=20.0,
            quality_label="Low",
            synthetic_quality=20.0,
            absolute_risk=0.06,
            population_prevalence=0.10,
            risk_ratio=0.60,
        ),
    ]
    high_only = summarize_trait_rows(rows, model_scope="high_quality", selected_ancestry="EUR")
    high_mod = summarize_trait_rows(rows, model_scope="high_moderate", selected_ancestry="EUR")
    all_models = summarize_trait_rows(rows, model_scope="all", selected_ancestry="EUR")
    # Headline risk is the MEDIAN ratio of the scoped rows, so it must track the
    # Quality dropdown together with the median-percentile card.
    assert high_only.risk_vs_average == "2.40x"
    assert high_only.n_risk_models == 1
    assert high_only.best_risk_row is not None
    assert high_only.best_risk_row["pgs_id"] == "PGS_HIGH"
    assert high_mod.risk_vs_average == "1.70x"  # median of {2.40, 1.00}
    assert high_mod.n_risk_models == 2
    assert all_models.risk_vs_average == "1.00x"  # median of {2.40, 1.00, 0.60}
    assert all_models.n_risk_models == 3
    # The representative row sits at the median ratio, not at the quality-best model.
    assert all_models.best_risk_row is not None
    assert all_models.best_risk_row["pgs_id"] == "PGS_MOD"
    assert high_only.median_pct == 90.0
    assert high_mod.median_pct == 72.5
    assert all_models.median_pct == 55.0


def test_apply_absolute_risk_clears_stale_one_times_ratio() -> None:
    """A frozen 1.00x snapshot must not survive a dashboard recompute with no estimate."""
    stale = _row(
        "PGS_STALE",
        match_rate=0.90,
        percentile=83.0,
        quality_label="High",
        synthetic_quality=80.0,
        absolute_risk=0.10,
        population_prevalence=0.10,
        risk_ratio=1.0,
    )
    stale["risk_ratio_value"] = 1.0
    stale["absolute_risk_percent"] = 10.0
    stale["absolute_risk_text"] = "10.0% (pop. avg: 10.0%)"
    cleared = apply_absolute_risk_to_row(
        stale,
        {
            "absolute_risk_text": "",
            "absolute_risk_percent": None,
            "population_average_percent": None,
            "risk_ratio_value": None,
            "heritability": "No mapped h²",
            "heritability_detail": "",
            "heritability_metrics": [],
        },
    )
    assert cleared["risk_ratio"] is None
    assert cleared["risk_ratio_value"] is None
    assert cleared["absolute_risk_percent"] is None
    refreshed_stats = summarize_trait_rows(
        [cleared], model_scope="high_quality", selected_ancestry="EUR",
    )
    assert refreshed_stats.risk_vs_average == "N/A"
    assert refreshed_stats.absolute_risk.startswith("N/A")


def test_compute_z_score_does_not_collapse_extremes_to_zero() -> None:
    assert _compute_z_score(None) is None
    assert _compute_z_score(50.0) is not None
    assert abs(_compute_z_score(50.0) or 99.0) < 0.01
    assert (_compute_z_score(0.0) or 0.0) < -2.5
    assert (_compute_z_score(100.0) or 0.0) > 2.5
    assert (_compute_z_score(83.0) or 0.0) > 0.9


def test_trait_prompt_match_rate_scale_only_marks_70_percent_usable() -> None:
    """Regression: percent-scale match_rate 30/70 must not treat 30 as usable."""
    prompt = build_prs_ai_prompt(
        "trait_results",
        user_results=[
            _row(
                "PGS_LOW",
                match_rate=30,
                percentile=90.0,
                quality_label="Low",
                synthetic_quality=10.0,
            ),
            _row(
                "PGS_OK",
                match_rate=70,
                percentile=40.0,
                quality_label="High",
                synthetic_quality=80.0,
            ),
        ],
        trait="type 2 diabetes mellitus",
        ancestry="EUR",
    )

    assert "1 usable (>=50% marker coverage)" in prompt
    assert "most reliable of 1 usable model" in prompt.lower()
    assert "PGS_OK" in prompt
    # Quality-ranked best is the High/70% model, not the 90th-percentile low-match one.
    assert "Most reliable of 1 usable model: PGS_OK" in prompt


def test_headline_risk_tracks_median_not_quality_best_model() -> None:
    """Regression (screenshot bug): median 83rd but Risk vs Average 1.00x.

    The quality-best model sat at the 46th percentile with a 1.00x ratio while
    the scoped median percentile was 83rd — the headline risk must follow the
    group median, never pin to the single quality-best model.
    """
    rows = [
        _row("PGS_A", match_rate=0.95, percentile=46.3, quality_label="High",
             synthetic_quality=90.0, absolute_risk=0.64, population_prevalence=0.64,
             risk_ratio=1.00),
        _row("PGS_B", match_rate=0.90, percentile=94.0, quality_label="High",
             synthetic_quality=80.0, absolute_risk=0.80, population_prevalence=0.64,
             risk_ratio=1.25),
        _row("PGS_C", match_rate=0.90, percentile=88.0, quality_label="Moderate",
             synthetic_quality=60.0, absolute_risk=0.78, population_prevalence=0.64,
             risk_ratio=1.22),
        _row("PGS_D", match_rate=0.85, percentile=78.0, quality_label="Moderate",
             synthetic_quality=55.0, absolute_risk=0.74, population_prevalence=0.64,
             risk_ratio=1.16),
    ]
    stats = summarize_trait_rows(rows, model_scope="all", selected_ancestry="EUR")
    assert stats.median_pct == 83.0
    assert stats.n_risk_models == 4
    # Median ratio of {1.00, 1.16, 1.22, 1.25} = 1.19 — never the best model's 1.00x.
    assert stats.risk_vs_average == "1.19x"
    assert stats.best_risk_row is not None
    assert stats.best_risk_row["pgs_id"] in {"PGS_C", "PGS_D"}


def test_risk_basis_label_wording() -> None:
    from just_prs.trait_summary import risk_basis_label

    assert risk_basis_label(0) == "no models with risk data"
    assert risk_basis_label(1) == "only model with risk data"
    assert risk_basis_label(4) == "median of 4 models with risk data"


def test_prefer_prevalence_row_matches_selected_ancestry() -> None:
    rows = [
        {"ancestry": "African", "prevalence": 0.12},
        {"ancestry": "European", "prevalence": 0.08},
    ]
    chosen = prefer_prevalence_row(rows, "EUR")
    assert chosen is not None
    assert chosen["prevalence"] == 0.08


def test_trait_prompt_fraction_match_rates_agree_with_percent_scale() -> None:
    prompt = build_prs_ai_prompt(
        "trait_results",
        user_results=[
            _row("PGS_LOW", match_rate=0.30, percentile=90.0, synthetic_quality=10.0),
            _row(
                "PGS_OK",
                match_rate=0.70,
                percentile=40.0,
                quality_label="High",
                synthetic_quality=80.0,
            ),
        ],
        trait="type 2 diabetes mellitus",
        ancestry="EUR",
    )

    assert "1 usable (>=50% marker coverage)" in prompt
    assert "Most reliable of 1 usable model: PGS_OK" in prompt
