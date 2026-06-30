"""Absolute risk estimation from PRS z-scores and population prevalence.

Three methods:
  1. OR-per-SD: Uses odds ratio per standard deviation of PRS.
  2. AUC-bivariate: Uses AUROC and Cohen's d via the bivariate normal model.
  3. h²-liability: Uses SNP heritability on the liability scale via the
     liability threshold model. Works for any binary trait with known h² and
     prevalence, even when no per-score OR or AUROC is available.

The multi-estimate architecture computes ALL available methods and returns
them as an AbsoluteRiskBundle so the user can see where methods agree or
diverge.
"""

import math

from just_prs.models import AbsoluteRisk, AbsoluteRiskBundle, AbsoluteRiskEstimate

#: Absolute risk is asymptotic — a logistic/liability model can approach but
#: never reach certainty, so a PRS estimate of a literal 100% is always an
#: artifact of clamping.  Cap just below 1.0 so the strongest predisposition
#: reads "99.9%" (very high, near-certain) rather than the impossible "100.0%".
_MAX_ABSOLUTE_RISK = 0.999


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function).

    Rational approximation from Abramowitz & Stegun, accurate to ~4.5e-4.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def risk_ratio_vs_population(user_risk: float, prevalence: float) -> float:
    """Compute the risk ratio of the user's risk vs population average.

    Args:
        user_risk: User's absolute risk (0-1).
        prevalence: Population prevalence (0-1).

    Returns:
        Risk ratio (e.g. 1.64 means 64% higher risk than average).
    """
    if prevalence <= 0:
        return float("inf")
    return user_risk / prevalence


def estimate_absolute_risk_or(
    z_score: float,
    or_per_sd: float,
    prevalence: float,
    prevalence_source: str = "",
    prevalence_type: str = "lifetime",
    confidence: str = "moderate",
    effect_size_citation: str | None = None,
    caveats: list[str] | None = None,
) -> AbsoluteRisk:
    """Estimate absolute risk using odds ratio per standard deviation of PRS.

    The model:
        baseline_odds = prevalence / (1 - prevalence)
        user_odds = baseline_odds * OR^z_score
        user_risk = user_odds / (1 + user_odds)

    Args:
        z_score: PRS z-score (how many SDs above/below the mean).
        or_per_sd: Odds ratio per standard deviation of PRS.
        prevalence: Population prevalence (0-1).
        prevalence_source: Description of prevalence data source.
        prevalence_type: Type of prevalence ('lifetime', 'point', 'cohort').
        confidence: Data quality level ('high', 'moderate', 'low').
        effect_size_citation: Citation for the OR value.
        caveats: Additional warning messages.

    Returns:
        AbsoluteRisk model with estimated risk.
    """
    all_caveats = list(caveats or [])

    if prevalence <= 0 or prevalence >= 1:
        all_caveats.append(f"Prevalence {prevalence} is outside valid range (0, 1)")
        prevalence = max(0.001, min(0.999, prevalence))

    if or_per_sd <= 0:
        all_caveats.append(f"OR per SD {or_per_sd} is invalid; using 1.0")
        or_per_sd = 1.0

    baseline_odds = prevalence / (1.0 - prevalence)
    user_odds = baseline_odds * (or_per_sd ** z_score)
    user_risk = user_odds / (1.0 + user_odds)
    user_risk = max(0.0, min(_MAX_ABSOLUTE_RISK, user_risk))

    return AbsoluteRisk(
        absolute_risk=round(user_risk, 6),
        population_prevalence=prevalence,
        risk_ratio=round(risk_ratio_vs_population(user_risk, prevalence), 4),
        method="or_per_sd",
        confidence=confidence,
        prevalence_source=prevalence_source,
        prevalence_type=prevalence_type,
        effect_size_citation=effect_size_citation,
        caveats=all_caveats,
    )


def estimate_absolute_risk_auc(
    z_score: float,
    auc: float,
    prevalence: float,
    prevalence_source: str = "",
    prevalence_type: str = "lifetime",
    confidence: str = "moderate",
    effect_size_citation: str | None = None,
    caveats: list[str] | None = None,
) -> AbsoluteRisk:
    """Estimate absolute risk using AUROC via the bivariate normal model.

    Derives Cohen's d from AUC: d = sqrt(2) * Phi^{-1}(AUC)
    Then uses the bivariate normal model to compute the probability that
    a person at the given z-score is a case.

    The model assumes PRS is normally distributed in both cases and controls,
    with equal variance and means separated by d. The case mean is shifted
    by +d relative to the control mean.

    For a person at z-score z in the combined population:
        P(case | z) = prevalence * phi_case(z) / phi_combined(z)

    where phi_case is N(d*prevalence, 1) and phi_combined is the mixture.

    Args:
        z_score: PRS z-score.
        auc: AUROC from a case-control study.
        prevalence: Population prevalence (0-1).
        prevalence_source: Description of prevalence data source.
        prevalence_type: Type of prevalence.
        confidence: Data quality level.
        effect_size_citation: Citation for the AUROC value.
        caveats: Additional warning messages.

    Returns:
        AbsoluteRisk model with estimated risk.
    """
    all_caveats = list(caveats or [])

    if prevalence <= 0 or prevalence >= 1:
        all_caveats.append(f"Prevalence {prevalence} is outside valid range (0, 1)")
        prevalence = max(0.001, min(0.999, prevalence))

    if auc <= 0.5 or auc >= 1.0:
        all_caveats.append(f"AUROC {auc} is outside useful range (0.5, 1.0)")
        return AbsoluteRisk(
            absolute_risk=round(prevalence, 6),
            population_prevalence=prevalence,
            risk_ratio=1.0,
            method="auc_bivariate",
            confidence="low",
            prevalence_source=prevalence_source,
            prevalence_type=prevalence_type,
            effect_size_citation=effect_size_citation,
            caveats=all_caveats,
        )

    d = math.sqrt(2.0) * _norm_ppf(auc)

    if d <= 0:
        all_caveats.append("Cohen's d derived from AUROC is non-positive")
        return AbsoluteRisk(
            absolute_risk=round(prevalence, 6),
            population_prevalence=prevalence,
            risk_ratio=1.0,
            method="auc_bivariate",
            confidence="low",
            prevalence_source=prevalence_source,
            prevalence_type=prevalence_type,
            effect_size_citation=effect_size_citation,
            caveats=all_caveats,
        )

    K = prevalence
    mean_case = d * (1.0 - K)
    mean_control = -d * K

    def _phi(x: float, mu: float, sigma: float = 1.0) -> float:
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))

    phi_case = _phi(z_score, mean_case)
    phi_control = _phi(z_score, mean_control)
    phi_combined = K * phi_case + (1.0 - K) * phi_control

    if phi_combined <= 0:
        user_risk = prevalence
    else:
        user_risk = K * phi_case / phi_combined

    user_risk = max(0.0, min(_MAX_ABSOLUTE_RISK, user_risk))

    return AbsoluteRisk(
        absolute_risk=round(user_risk, 6),
        population_prevalence=prevalence,
        risk_ratio=round(risk_ratio_vs_population(user_risk, prevalence), 4),
        method="auc_bivariate",
        confidence=confidence,
        prevalence_source=prevalence_source,
        prevalence_type=prevalence_type,
        effect_size_citation=effect_size_citation,
        caveats=all_caveats,
    )


def estimate_absolute_risk(
    z_score: float,
    prevalence: float,
    or_estimate: float | None = None,
    auroc_estimate: float | None = None,
    prevalence_source: str = "",
    prevalence_type: str = "lifetime",
    confidence: str = "moderate",
    effect_size_citation: str | None = None,
    caveats: list[str] | None = None,
) -> AbsoluteRisk | None:
    """Facade: estimate absolute risk using the best available method.

    Prefers OR-per-SD (more direct) when available, falls back to
    AUC-bivariate.

    Args:
        z_score: PRS z-score.
        prevalence: Population prevalence (0-1).
        or_estimate: OR per SD, if available.
        auroc_estimate: AUROC, if available.
        prevalence_source: Where the prevalence came from.
        prevalence_type: Type of prevalence.
        confidence: Data quality level.
        effect_size_citation: Citation for the effect size.
        caveats: Additional warnings.

    Returns:
        AbsoluteRisk or None if neither OR nor AUROC is available.
    """
    if prevalence <= 0 or prevalence >= 1.0:
        if prevalence == 1.0:
            return None
        if prevalence <= 0:
            return None

    if or_estimate is not None and or_estimate > 0:
        return estimate_absolute_risk_or(
            z_score=z_score,
            or_per_sd=or_estimate,
            prevalence=prevalence,
            prevalence_source=prevalence_source,
            prevalence_type=prevalence_type,
            confidence=confidence,
            effect_size_citation=effect_size_citation,
            caveats=caveats,
        )

    if auroc_estimate is not None and 0.5 < auroc_estimate < 1.0:
        return estimate_absolute_risk_auc(
            z_score=z_score,
            auc=auroc_estimate,
            prevalence=prevalence,
            prevalence_source=prevalence_source,
            prevalence_type=prevalence_type,
            confidence=confidence,
            effect_size_citation=effect_size_citation,
            caveats=caveats,
        )

    return None


def estimate_absolute_risk_h2(
    z_score: float,
    h2_liability: float,
    prevalence: float,
    n_variants_score: int | None = None,
    n_variants_gwas: int | None = None,
    prevalence_source: str = "",
    prevalence_type: str = "lifetime",
    confidence: str = "moderate",
    h2_source: str = "",
    h2_source_detail: str | None = None,
    ancestry: str | None = None,
    caveats: list[str] | None = None,
) -> AbsoluteRiskEstimate:
    """Estimate absolute risk using SNP heritability via the liability threshold model.

    The model assumes disease liability follows a standard normal distribution
    with a threshold t = Phi^{-1}(1 - K). The PRS captures a fraction r² of
    the total genetic variance h². A person at z-score z has shifted liability:

        P(disease | z) = Phi((z * sqrt(r²) - t) / sqrt(1 - r²))

    When r² is unknown, we use h2_liability as an upper bound (assumes PRS
    captures all SNP heritability, which overestimates risk).

    Args:
        z_score: PRS z-score (SDs from population mean).
        h2_liability: SNP heritability on the liability scale (0-1).
        prevalence: Population prevalence (0-1).
        n_variants_score: Number of variants in the PRS (for r² estimation).
        n_variants_gwas: Number of genome-wide significant variants (for r² estimation).
        prevalence_source: Where the prevalence data came from.
        prevalence_type: Type of prevalence.
        confidence: Data quality level.
        h2_source: Source of h² data (e.g. 'pan_ukbb', 'gwas_atlas').
        ancestry: Ancestry of the h² estimate.
        caveats: Additional warnings.

    Returns:
        AbsoluteRiskEstimate with the liability threshold estimate.
    """
    all_caveats = list(caveats or [])

    if prevalence <= 0 or prevalence >= 1:
        all_caveats.append(f"Prevalence {prevalence} is outside valid range (0, 1)")
        prevalence = max(0.001, min(0.999, prevalence))

    if h2_liability <= 0 or h2_liability >= 1:
        all_caveats.append(f"h² liability {h2_liability} is outside valid range (0, 1)")
        h2_liability = max(0.001, min(0.999, h2_liability))

    r_squared = h2_liability
    if r_squared > 0.95:
        r_squared = 0.95
    all_caveats.append(
        "Assumes PRS captures all SNP heritability (upper bound on risk shift). "
        "True PRS r² is typically much less than h²_SNP."
    )

    t = _norm_ppf(1.0 - prevalence)
    shifted = z_score * math.sqrt(r_squared)
    denominator = math.sqrt(1.0 - r_squared)
    if denominator <= 0:
        denominator = 0.01

    user_risk = _norm_cdf((shifted - t) / denominator)
    user_risk = max(0.0, min(_MAX_ABSOLUTE_RISK, user_risk))

    ancestry_label = f" {ancestry}" if ancestry else ""
    method_label = f"h²{ancestry_label} ({h2_source})" if h2_source else f"h² liability{ancestry_label}"

    return AbsoluteRiskEstimate(
        absolute_risk=round(user_risk, 6),
        population_prevalence=prevalence,
        risk_ratio=round(risk_ratio_vs_population(user_risk, prevalence), 4),
        method="h2_liability",
        method_label=method_label,
        ancestry=ancestry,
        h2_value=h2_liability,
        h2_source=h2_source,
        h2_source_detail=h2_source_detail,
        confidence=confidence,
        prevalence_source=prevalence_source,
        prevalence_type=prevalence_type,
        effect_size_citation=None,
        caveats=all_caveats,
    )


def _absolute_risk_to_estimate(
    risk: AbsoluteRisk,
    method_label: str,
    ancestry: str | None = None,
) -> AbsoluteRiskEstimate:
    """Convert a legacy AbsoluteRisk to the new AbsoluteRiskEstimate model."""
    return AbsoluteRiskEstimate(
        absolute_risk=risk.absolute_risk,
        population_prevalence=risk.population_prevalence,
        risk_ratio=risk.risk_ratio,
        method=risk.method,
        method_label=method_label,
        ancestry=ancestry,
        h2_value=None,
        h2_source=None,
        confidence=risk.confidence,
        prevalence_source=risk.prevalence_source,
        prevalence_type=risk.prevalence_type,
        effect_size_citation=risk.effect_size_citation,
        caveats=risk.caveats,
    )


def _compute_agreement(estimates: list[AbsoluteRiskEstimate]) -> tuple[str, float | None]:
    """Compute agreement level across multiple risk estimates.

    Returns (agreement_label, spread_in_percentage_points).
    """
    if len(estimates) < 2:
        return "single_estimate", None

    risks = [e.absolute_risk for e in estimates]
    spread = (max(risks) - min(risks)) * 100.0

    if spread < 2.0:
        return "high", round(spread, 1)
    if spread < 5.0:
        return "moderate", round(spread, 1)
    return "low", round(spread, 1)


def estimate_all_absolute_risks(
    z_score: float,
    prevalence: float,
    or_estimate: float | None = None,
    auroc_estimate: float | None = None,
    h2_estimates: list[dict] | None = None,
    prevalence_source: str = "",
    prevalence_type: str = "lifetime",
    confidence: str = "moderate",
    effect_size_citation: str | None = None,
    caveats: list[str] | None = None,
) -> AbsoluteRiskBundle:
    """Compute ALL available absolute risk estimates and return as a bundle.

    Runs every method for which input data is available:
    - OR-per-SD (if or_estimate provided)
    - AUC-bivariate (if auroc_estimate provided)
    - h²-liability (for each entry in h2_estimates)

    Args:
        z_score: PRS z-score.
        prevalence: Population prevalence (0-1).
        or_estimate: OR per SD, if available.
        auroc_estimate: AUROC, if available.
        h2_estimates: List of dicts with keys: h2_liability, ancestry, source,
            confidence, source_detail. One entry per ancestry/source combination.
        prevalence_source: Where the prevalence came from.
        prevalence_type: Type of prevalence.
        confidence: Data quality level for OR/AUROC methods.
        effect_size_citation: Citation for the OR/AUROC.
        caveats: Additional warnings for OR/AUROC methods.

    Returns:
        AbsoluteRiskBundle with all estimates, a best pick, and agreement info.
    """
    if prevalence <= 0 or prevalence >= 1.0:
        return AbsoluteRiskBundle()

    estimates: list[AbsoluteRiskEstimate] = []

    if or_estimate is not None and or_estimate > 0:
        or_risk = estimate_absolute_risk_or(
            z_score=z_score,
            or_per_sd=or_estimate,
            prevalence=prevalence,
            prevalence_source=prevalence_source,
            prevalence_type=prevalence_type,
            confidence=confidence,
            effect_size_citation=effect_size_citation,
            caveats=caveats,
        )
        estimates.append(_absolute_risk_to_estimate(or_risk, "OR per SD"))

    if auroc_estimate is not None and 0.5 < auroc_estimate < 1.0:
        auc_risk = estimate_absolute_risk_auc(
            z_score=z_score,
            auc=auroc_estimate,
            prevalence=prevalence,
            prevalence_source=prevalence_source,
            prevalence_type=prevalence_type,
            confidence=confidence,
            effect_size_citation=effect_size_citation,
            caveats=caveats,
        )
        estimates.append(_absolute_risk_to_estimate(auc_risk, "AUROC model"))

    if h2_estimates:
        for h2_entry in h2_estimates:
            h2_val = h2_entry.get("h2_liability")
            if h2_val is None or h2_val <= 0:
                continue
            h2_est = estimate_absolute_risk_h2(
                z_score=z_score,
                h2_liability=float(h2_val),
                prevalence=prevalence,
                prevalence_source=prevalence_source,
                prevalence_type=prevalence_type,
                confidence=h2_entry.get("confidence", "moderate"),
                h2_source=h2_entry.get("source", ""),
                h2_source_detail=h2_entry.get("source_detail"),
                ancestry=h2_entry.get("ancestry"),
                caveats=list(caveats or []),
            )
            estimates.append(h2_est)

    if not estimates:
        return AbsoluteRiskBundle()

    best: AbsoluteRiskEstimate | None = None
    for est in estimates:
        if est.method == "or_per_sd":
            best = est
            break
    if best is None:
        for est in estimates:
            if est.method == "auc_bivariate":
                best = est
                break
    if best is None:
        best = estimates[0]

    agreement, spread = _compute_agreement(estimates)

    return AbsoluteRiskBundle(
        estimates=estimates,
        best_estimate=best,
        agreement=agreement,
        spread_pp=spread,
    )
