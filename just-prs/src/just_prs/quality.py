"""PRS result quality assessment and formatting helpers.

Pure functions for classifying model quality, interpreting PRS results,
and formatting performance metrics. No Reflex or UI dependency -- these
are shared between the core library and any UI that displays PRS results.
"""

from typing import Any


_DEFAULT_HARMONIZED_PENALTY: float = 0.90


def _harmonized_penalty() -> float:
    """Read harmonized score quality penalty from env, default 0.85 (15% reduction)."""
    import os
    raw = os.environ.get("PRS_HARMONIZED_PENALTY")
    if raw is None:
        return _DEFAULT_HARMONIZED_PENALTY
    try:
        val = float(raw)
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return _DEFAULT_HARMONIZED_PENALTY


def synthetic_quality_score(
    *,
    auroc: float | None = None,
    cindex: float | None = None,
    or_estimate: float | None = None,
    hr_estimate: float | None = None,
    beta_estimate: float | None = None,
    n_individuals: int | float | None = None,
    match_rate: float | None = None,
    is_harmonized: bool = False,
) -> float:
    """Compute a transparent numeric PRS quality score for ranking (0–100).

    Four-tier discrimination resolution, chosen in priority order:

    Tier 1a — AUROC or C-index directly reported: no penalty.
        Discrimination is power-stretched so that AUROC 0.70 (good for PRS)
        maps to ~0.74 instead of the raw 0.70. Below AUROC 0.60, a
        steeper exponent (0.88 vs 0.78) further penalizes weak models
        that offer little practical discrimination.
    Tier 1b — Beta only (continuous/quantitative traits, e.g. BMI, height):
        discrimination ≈ 0.5 + 0.5·tanh(ln(1+|β|)), with 0.95 penalty.
        Log-based mapping gives diminishing returns for extreme betas (unit-
        dependent artifacts like β=152 for lab-value traits). Mild penalty
        reflects empirically lower cross-genome stability vs AUROC-validated
        models.
    Tier 2  — OR or HR, no direct discrimination metric:
        discrimination ≈ Φ(ln(OR_or_HR) / √2) (normal CDF approximation),
        0.90 penalty to reflect that AUROC could have been measured directly.
    Tier 3  — No performance metric at all: discrimination floored at 0.51
        (biased coin — published models are assumed marginally better than
        random), 0.6 penalty.

    Cohort factor is log10-scaled on n_individuals (350 k→1.0, unknown→0.5).
    Denominator 5.5 (vs 6.0) gives more credit to large GWAS cohorts.
    match_rate is optional (absent in metadata-only comparisons); when given
    it linearly reduces the score for poor genotype coverage.

    Harmonized penalty: when ``is_harmonized`` is True, the score is reduced
    by the ``PRS_HARMONIZED_PENALTY`` env factor (default 0.90, i.e. 10%
    reduction) because coordinate liftover may introduce minor mapping errors.
    """
    import math

    _clamp = lambda v, lo, hi: max(lo, min(hi, v))  # noqa: E731

    # --- Tier 1a: direct discrimination (power-stretch, no penalty) ---
    if auroc is not None or cindex is not None:
        raw = float(auroc if auroc is not None else cindex)  # type: ignore[arg-type]
        if raw <= 0.5:
            discrimination = 0.5
        else:
            exp = 0.78 if raw >= 0.60 else 0.88
            discrimination = 0.5 + 0.5 * ((raw - 0.5) / 0.5) ** exp
        penalty = 1.0

    # --- Tier 1b: beta, log-mapped (0.95 penalty) ---
    elif beta_estimate is not None:
        discrimination = 0.5 + 0.5 * math.tanh(math.log1p(abs(float(beta_estimate))))
        penalty = 0.95

    # --- Tier 2: OR / HR → approximate AUROC ---
    elif or_estimate is not None or hr_estimate is not None:
        ratio = float(or_estimate if or_estimate is not None else hr_estimate)  # type: ignore[arg-type]
        if ratio <= 0:
            discrimination = 0.51
            penalty = 0.6
        else:
            discrimination = _clamp(0.5 * (1 + math.erf(math.log(ratio) / (math.sqrt(2) * math.sqrt(2)))), 0.5, 1.0)
            penalty = 0.90

    # --- Tier 3: nothing at all ---
    else:
        discrimination = 0.51
        penalty = 0.6

    if n_individuals is None or n_individuals <= 0:
        cohort_factor = 0.5
    else:
        cohort_factor = _clamp(math.log10(float(n_individuals)) / 5.5, 0.0, 1.0)

    match_factor = 1.0 if match_rate is None else _clamp(float(match_rate), 0.0, 1.0)
    harmonized_factor = _harmonized_penalty() if is_harmonized else 1.0
    return round(100.0 * discrimination * cohort_factor * match_factor * penalty * harmonized_factor, 1)


#: AUROC penalty applied to harmonized (cross-build, e.g. GRCh37→GRCh38 lifted)
#: scores when classifying quality.  Coordinate liftover drops/mismaps a fraction
#: of variants by design, so a lifted score is inherently less trustworthy than a
#: natively-built one.  The High/Moderate AUROC bands here are 0.1 wide
#: (High ≥0.7, Moderate ≥0.6), so 0.05 demotes a model by ~half a tier — and a
#: harmonized model with no discrimination metric is dropped one tier outright.
_HARMONIZED_AUROC_PENALTY = 0.05

#: Coverage thresholds for quality, expressed on **weight-mass coverage (C_wt)** —
#: the fraction of the score's effect-weight mass actually carried by observed
#: genotypes.  C_wt is used instead of the count match_rate because WGS
#: reference-restoration fills absent loci as hom-ref, inflating count match_rate
#: to ~100% for every model (destroying its discriminative power) while leaving
#: C_wt honest (restored hom-ref carries zero dosage, so zero matched weight).
#: For chips — where coverage genuinely varies — C_wt still tracks real coverage,
#: so it stays a valid, discriminative signal there.  0.20 mirrors
#: ``prs_catalog.MIN_RELIABLE_WEIGHT_MASS_COVERAGE`` (below it the percentile is
#: already flagged unreliable); 0.50 = "well covered" (half the weight mass).
_MIN_RELIABLE_COVERAGE = 0.20
_WELL_COVERED_COVERAGE = 0.50


def classify_model_quality(
    coverage: float,
    auroc: float | None,
    is_harmonized: bool = False,
) -> tuple[str, str]:
    """Classify overall model quality from weight-mass coverage and AUROC.

    ``coverage`` is the score's **weight-mass coverage (C_wt)** when available
    (callers fall back to the count match_rate only when C_wt is unknown).  See
    the module constants for why C_wt rather than count match_rate.

    Returns (label, color_name) where color_name is a semantic token
    (e.g. "green", "red") that UIs can map to their own palette.

    ``is_harmonized`` demotes cross-build lifted scores (less reliable by
    design): AUROC is penalized by ``_HARMONIZED_AUROC_PENALTY`` before banding,
    and a harmonized score with no discrimination metric drops Moderate→Low.
    """
    if coverage < _MIN_RELIABLE_COVERAGE:
        return "Very Low", "red"
    if auroc is not None:
        eff_auroc = auroc - _HARMONIZED_AUROC_PENALTY if is_harmonized else auroc
        if coverage >= _WELL_COVERED_COVERAGE and eff_auroc >= 0.7:
            return "High", "green"
        if eff_auroc >= 0.6:
            return "Moderate", "yellow"
    # Reliable coverage but no strong discrimination metric: a native score gets
    # the benefit of the doubt (Moderate); a cross-build lifted one drops to Low.
    return ("Low", "orange") if is_harmonized else ("Moderate", "yellow")


#: Quality tiers that count as "top of the scale".  When every model in a cohort
#: lands here the absolute heatmap is a single colour and useless, which triggers
#: the relative re-scaling below.
_TOP_QUALITY_KEYS = frozenset({"high", "moderate"})

#: Low-quality band width for the relative re-scale, in score units (0–1 scale).
#: The worst model and everything within this margin above it are marked "low";
#: 0.05 (5 points of a 0–100 synthetic score) gives a visible bottom band.
_RELATIVE_LOW_MARGIN = 0.05


def relative_quality_keys(
    scores: list[float | None],
    *,
    margin: float = _RELATIVE_LOW_MARGIN,
) -> list[str | None] | None:
    """Stretch a cohort's quality across the palette relative to its own range.

    Quality colouring is comparative: if every model classifies into the top
    tiers (e.g. a WGS sample where C_wt ≈ 1.0 for all), the heatmap is uniformly
    green and hides the real ranking.  This re-anchors the low-quality threshold
    at **worst score + ``margin``** (so the weakest model for this user, plus any
    within ``margin`` of it, read as low) and **stretch-transforms** the rest of
    the cohort across ``moderate``/``high`` by min–max normalising to the
    *surviving* range ``[worst + margin, best]`` — so the moderate/high threshold
    moves with the low anchor and all three tiers stay proportionally spread.

    ``scores`` are on a 0–1 scale (e.g. ``synthetic_quality / 100``).  Returns a
    parallel list of tier keys, or ``None`` when there is nothing to spread
    (fewer than two scored models, or a range no wider than ``margin``).
    """
    present = [s for s in scores if s is not None]
    if len(present) < 2:
        return None
    worst, best = min(present), max(present)
    span = best - worst
    if span <= margin:
        return None
    # The low band is the bottom `margin` above the worst model.  The moderate/
    # high threshold is then re-anchored to the *surviving* range [low_cut, best]
    # — not the full [worst, best] — so moving the low anchor moves the medium
    # threshold with it, keeping all three tiers proportionally spread.
    low_cut = worst + margin
    hi_span = best - low_cut  # > 0 because span > margin
    keys: list[str | None] = []
    for s in scores:
        if s is None:
            keys.append(None)
            continue
        if s <= low_cut:
            keys.append("low")
        else:
            rel = (s - low_cut) / hi_span  # 0..1 across the non-low range
            keys.append("high" if rel >= 0.5 else "moderate")
    return keys


def rescale_quality_if_degenerate(
    tiers: list[str | None],
    scores: list[float | None],
    *,
    margin: float = _RELATIVE_LOW_MARGIN,
) -> list[str | None]:
    """Return relative tier keys when the absolute scale is degenerate, else ``tiers``.

    "Degenerate" = every (present) model sits in :data:`_TOP_QUALITY_KEYS`, so
    the absolute heatmap can't discriminate.  When some model is already below
    Moderate the absolute scale carries information and is left untouched.
    """
    present_tiers = [t for t in tiers if t]
    if not present_tiers or any(t not in _TOP_QUALITY_KEYS for t in present_tiers):
        return list(tiers)
    relative = relative_quality_keys(scores, margin=margin)
    return relative if relative is not None else list(tiers)


#: Normalize any quality *label* (from ``classify_model_quality`` /
#: ``classify_synthetic_quality`` / ``classify_combined_quality``) to one of the
#: four lowercase tier keys used for chart colors and table dots.  "Normal" is a
#: synthetic-score band that sits between high and moderate; it maps to the
#: moderate color so the four-color palette stays stable.
_QUALITY_LABEL_TO_KEY: dict[str, str] = {
    "high": "high",
    "normal": "moderate",
    "moderate": "moderate",
    "low": "low",
    "very low": "very_low",
    "very_low": "very_low",
}


def resolve_quality_key(
    label: str | None = None,
    n_var: int | None = None,
    auroc: float | None = None,
) -> str:
    """Resolve a model's quality tier key: ``high`` / ``moderate`` / ``low`` / ``very_low``.

    This is the single source of truth shared by the trait chart and the HTML
    report table, so the two can never disagree (the table previously read a
    ``quality`` key that was never populated and silently defaulted every model
    to ``"low"`` while the chart used a separate variant-count heuristic).

    Prefers the genotype-aware quality *label* already attached to a scored
    result (``classify_model_quality``).  Only when no label exists — e.g. a
    reference-panel model shown for context that was never scored against a
    genome — does it fall back to a metadata-only estimate from variant count
    and AUROC.
    """
    if label:
        key = _QUALITY_LABEL_TO_KEY.get(label.strip().lower())
        if key:
            return key
    if auroc is not None and auroc >= 0.7:
        return "high"
    if n_var is not None and n_var >= 100_000:
        return "high"
    if n_var is not None and n_var >= 10_000:
        return "moderate"
    if n_var is not None and n_var >= 100:
        return "low"
    return "very_low"


_PERCENTILE_METHOD_DESC: dict[str, str] = {
    "reference_panel": "from a reference-panel population distribution",
    "theoretical": "theoretical, from allele frequencies in the scoring file",
    "auroc_approx": "an AUROC-based approximation",
}


def interpret_prs_result(
    percentile: float | None,
    match_rate: float,
    auroc: float | None,
    percentile_method: str | None = None,
    reliable: bool = True,
    caveat: str = "",
) -> dict[str, str]:
    """Produce human-readable interpretation of a single PRS result.

    Returns a dict with keys: quality_label, quality_color, summary.

    ``percentile_method`` (``'reference_panel'`` / ``'theoretical'`` / ``'auroc_approx'``)
    lets the summary describe *how* the percentile was derived, instead of always
    attributing it to scoring-file allele frequencies. When ``reliable`` is False the
    ``caveat`` is appended. The percentile-unavailable message is generic and no longer
    claims "no allele frequencies" when a reference-panel percentile could exist.
    """
    quality_label, quality_color = classify_model_quality(match_rate, auroc)

    parts: list[str] = []

    if percentile is not None:
        method_desc = _PERCENTILE_METHOD_DESC.get(percentile_method or "", "estimated")
        parts.append(f"Estimated percentile: {percentile:.1f}% ({method_desc}).")
        if not reliable and caveat:
            parts.append(caveat)

    if auroc is not None:
        if auroc >= 0.7:
            parts.append(f"Good predictive model (AUROC={auroc:.3f}).")
        elif auroc >= 0.6:
            parts.append(f"Moderate predictive model (AUROC={auroc:.3f}).")
        else:
            parts.append(f"Weak predictive model (AUROC={auroc:.3f}).")
    else:
        parts.append("No AUROC available to assess model accuracy.")

    if match_rate < 0.1:
        parts.append(f"Only {match_rate * 100:.0f}% of scoring variants matched \u2014 results may be unreliable.")
    elif match_rate < 0.5:
        parts.append(f"{match_rate * 100:.0f}% of scoring variants matched \u2014 interpret with caution.")
    else:
        parts.append(f"{match_rate * 100:.0f}% of scoring variants matched.")

    if percentile is None:
        parts.append(
            "No population percentile is available for this score \u2014 compare your "
            "score to a matched reference cohort for meaningful interpretation."
        )

    return {
        "quality_label": quality_label,
        "quality_color": quality_color,
        "summary": " ".join(parts),
    }


def classify_synthetic_quality(score: float) -> tuple[str, str]:
    """Classify PGS model quality from its synthetic_quality_score (0-100).

    Boundaries derived from the empirical distribution of 604 GRCh38 scores:

        >=70  High        (top 9%  — strong AUROC/C-index or large-cohort beta)
        >=50  Normal      (next 27% — median and above, solid discrimination)
        >=30  Moderate    (next 32% — weak discrimination or small cohort)
        < 30  Low         (bottom 32% — no performance data or very weak)

    The 30-point boundary sits in the natural gap between T3 (no metrics,
    ceiling ~30.6) and models with any real performance data.  The 50-point
    boundary is the dataset median.  The 70-point boundary captures the
    top performers (p90 = 69.8).

    Returns (label, color_name).
    """
    if score >= 70:
        return "High", "green"
    if score >= 50:
        return "Normal", "yellow"
    if score >= 30:
        return "Moderate", "orange"
    return "Low", "red"


def classify_combined_quality(score: float) -> tuple[str, str]:
    """Classify PGS quality from its combined_quality_score (0-100).

    The combined score blends synthetic model quality (0.40) with practical
    signals: genotype match rate (0.25), cross-genome percentile stability
    (0.15), and absolute risk concordance (0.20).

    Boundaries derived from the empirical distribution of 604 scores across
    10 genomes:

        >=70  High        (top 15% — strong model validated in practice)
        >=55  Normal      (next 35% — above median, reliable for demo)
        >=40  Moderate    (next 35% — usable but some practical weakness)
        < 40  Low         (bottom 15% — poor match, unstable, or no metrics)

    The 40-point boundary separates the T3-dominated tail from scored models.
    The 55-point boundary is the dataset median.  The 70-point boundary
    aligns with the demo trait cutoff (see docs/demo-trait-ranking.md).

    Returns (label, color_name).
    """
    if score >= 70:
        return "High", "green"
    if score >= 55:
        return "Normal", "yellow"
    if score >= 40:
        return "Moderate", "orange"
    return "Low", "red"


def synthetic_quality_tier(
    *,
    auroc: float | None = None,
    cindex: float | None = None,
    or_estimate: float | None = None,
    hr_estimate: float | None = None,
    beta_estimate: float | None = None,
) -> tuple[str, str]:
    """Identify the quality tier and primary metric used for a model.

    Returns (tier, metric_summary) where tier is one of:
        "T1a_auroc"  — AUROC or C-index reported
        "T1b_beta"   — Beta only (continuous trait)
        "T2_or_hr"   — OR or HR, no discrimination metric
        "T3_none"    — No performance data

    metric_summary is a short human-readable string like "AUROC=0.72",
    "Beta=0.24", "OR=2.10", or "No metric".
    """
    if auroc is not None or cindex is not None:
        raw = auroc if auroc is not None else cindex
        label = "AUROC" if auroc is not None else "C-index"
        return "T1a_auroc", f"{label}={raw:.3f}"
    if beta_estimate is not None:
        return "T1b_beta", f"Beta={abs(float(beta_estimate)):.2f}"
    if or_estimate is not None or hr_estimate is not None:
        ratio = or_estimate if or_estimate is not None else hr_estimate
        label = "OR" if or_estimate is not None else "HR"
        return "T2_or_hr", f"{label}={float(ratio):.2f}"  # type: ignore[arg-type]
    return "T3_none", "No metric"


def format_effect_size(perf_row: dict[str, Any]) -> str | None:
    """Format the best available effect size metric from a cleaned performance row.

    Checks OR, HR, then Beta in order. Returns ``None`` (not an empty string) when
    no effect-size estimate exists, so callers can distinguish "unavailable" from
    an empty value (F11).
    """
    for prefix, label in [("or", "OR"), ("hr", "HR"), ("beta", "Beta")]:
        est = perf_row.get(f"{prefix}_estimate")
        if est is not None:
            ci_lo = perf_row.get(f"{prefix}_ci_lower")
            ci_hi = perf_row.get(f"{prefix}_ci_upper")
            se = perf_row.get(f"{prefix}_se")
            result = f"{label}={est:.2f}"
            if ci_lo is not None and ci_hi is not None:
                result += f" [{ci_lo:.2f}-{ci_hi:.2f}]"
            elif se is not None:
                result += f" (SE={se:.2f})"
            return result
    return None


def format_classification(perf_row: dict[str, Any]) -> str | None:
    """Format the best available classification metric from a cleaned performance row.

    Checks AUROC then C-index. Returns ``None`` (not an empty string) when no
    classification metric exists (F11).
    """
    for prefix, label in [("auroc", "AUROC"), ("cindex", "C-index")]:
        est = perf_row.get(f"{prefix}_estimate")
        if est is not None:
            ci_lo = perf_row.get(f"{prefix}_ci_lower")
            ci_hi = perf_row.get(f"{prefix}_ci_upper")
            result = f"{label}={est:.3f}"
            if ci_lo is not None and ci_hi is not None:
                result += f" [{ci_lo:.3f}-{ci_hi:.3f}]"
            return result
    return None
