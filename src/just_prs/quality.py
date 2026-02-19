"""PRS result quality assessment and formatting helpers.

Pure functions for classifying model quality, interpreting PRS results,
and formatting performance metrics. No Reflex or UI dependency -- these
are shared between the core library and any UI that displays PRS results.
"""

from typing import Any


def classify_model_quality(
    match_rate: float,
    auroc: float | None,
) -> tuple[str, str]:
    """Classify overall model quality from match rate and AUROC.

    Returns (label, color_name) where color_name is a semantic token
    (e.g. "green", "red") that UIs can map to their own palette.
    """
    if match_rate < 0.1:
        return "Very Low", "red"
    if auroc is not None:
        if match_rate >= 0.5 and auroc >= 0.7:
            return "High", "green"
        if match_rate >= 0.5 and auroc >= 0.6:
            return "Moderate", "yellow"
    if match_rate >= 0.5:
        return "Moderate", "yellow"
    return "Low", "orange"


def interpret_prs_result(
    percentile: float | None,
    match_rate: float,
    auroc: float | None,
) -> dict[str, str]:
    """Produce human-readable interpretation of a single PRS result.

    Returns a dict with keys: quality_label, quality_color, summary.

    When a theoretical percentile is available (computed from allele frequencies
    in the scoring file), it is included in the summary as an approximate
    population position.
    """
    quality_label, quality_color = classify_model_quality(match_rate, auroc)

    parts: list[str] = []

    if percentile is not None:
        parts.append(
            f"Estimated percentile: {percentile:.1f}% "
            f"(theoretical, from allele frequencies in the scoring file)."
        )

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
            "No allele frequencies in scoring file \u2014 percentile not available. "
            "Compare your score to a matched reference cohort for meaningful interpretation."
        )

    return {
        "quality_label": quality_label,
        "quality_color": quality_color,
        "summary": " ".join(parts),
    }


def format_effect_size(perf_row: dict[str, Any]) -> str:
    """Format the best available effect size metric from a cleaned performance row.

    Checks OR, HR, then Beta in order. Returns empty string if none available.
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
    return ""


def format_classification(perf_row: dict[str, Any]) -> str:
    """Format the best available classification metric from a cleaned performance row.

    Checks AUROC then C-index. Returns empty string if none available.
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
    return ""
