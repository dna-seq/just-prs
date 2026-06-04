"""Pure enrichment of a PRSResult with catalog context: percentiles, quality, risk, heritability.

No Reflex dependency.  The single entry point ``enrich_prs_result()`` accepts
all dependencies as parameters and returns an ``EnrichedPRSResult`` — shared
between the web UI, CLI, and batch scripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from just_prs.absolute_risk import _norm_ppf
from just_prs.models import EnrichedPRSResult, PRSResult
from just_prs.quality import (
    classify_model_quality,
    classify_synthetic_quality,
    format_classification,
    format_effect_size,
    interpret_prs_result,
    synthetic_quality_score,
    synthetic_quality_tier,
)
from just_prs.reference import SUPERPOPULATIONS

if TYPE_CHECKING:
    from just_prs.prs_catalog import PRSCatalog

SUPERPOPULATION_LABELS: dict[str, str] = {
    "AFR": "African",
    "AMR": "Admixed American",
    "EAS": "East Asian",
    "EUR": "European",
    "SAS": "South Asian",
}


def enrich_prs_result(
    result: PRSResult,
    catalog: PRSCatalog,
    best_perf_df: pl.DataFrame,
    genome_build: str = "GRCh38",
    selected_ancestry: str = "EUR",
    compute_all_populations: bool = True,
    is_harmonized: bool = False,
) -> EnrichedPRSResult:
    """Enrich a raw PRS computation result with quality, percentile, risk and reference data.

    This replicates the full enrichment logic that the UI performs inline,
    producing a single structured result suitable for display, CSV export,
    or parquet storage.

    Args:
        result: Raw PRS computation result from ``compute_prs()``.
        catalog: ``PRSCatalog`` instance (used for percentile, absolute risk,
            reference status lookups).
        best_perf_df: Pre-collected ``catalog.best_performance().collect()``
            DataFrame — passed in so callers load it once per batch.
        genome_build: Genome build used for computation.
        selected_ancestry: Default ancestry for percentile lookup.
        compute_all_populations: If True, fetch percentiles for all 1000G
            superpopulations.

    Returns:
        Fully enriched result with all fields populated.
    """
    match_pct = round(result.match_rate * 100, 1)
    match_color = "red" if match_pct < 10 else ("orange" if match_pct < 50 else "green")

    # --- EFO ID from catalog ---
    score_info = catalog.score_info_row(result.pgs_id)
    trait_efo_id = str(score_info.get("trait_efo_id") or "") if score_info else ""

    # --- Percentile resolution: result's own → catalog fallback ---
    pct_value = result.percentile
    pct_method = result.percentile_method or (
        "theoretical" if result.has_allele_frequencies else ""
    )
    if pct_value is None:
        pct_value, pct_method = catalog.percentile(
            result.score, result.pgs_id, ancestry=selected_ancestry
        )

    # --- Per-population percentiles ---
    all_pop_values: dict[str, float] = {}
    if compute_all_populations:
        all_pop_values = _all_population_percentiles(catalog, result.score, result.pgs_id)
    all_pop_text = (
        ", ".join(f"{sp}: {pct:.1f}%" for sp, pct in sorted(all_pop_values.items()))
        if all_pop_values
        else ""
    )

    # --- Performance metrics ---
    auroc_val: float | None = None
    effect_size_str = ""
    classification_str = ""
    ancestry_str = ""
    n_individuals: int | None = None

    perf_rows = best_perf_df.filter(pl.col("pgs_id") == result.pgs_id)
    perf_dict: dict = {}
    if perf_rows.height > 0:
        perf_dict = perf_rows.row(0, named=True)
        effect_size_str = format_effect_size(perf_dict)
        classification_str = format_classification(perf_dict)
        auroc_val = perf_dict.get("auroc_estimate")
        ancestry_str = perf_dict.get("ancestry_broad") or ""
        n_individuals = perf_dict.get("n_individuals")

    # --- Synthetic quality score and tier ---
    # Metadata-only score (no match penalty) for model quality ranking.
    # Match rate is genome-specific, not a model quality property — it's
    # already shown separately in match_rate / match_color.
    sq_auroc = perf_dict.get("auroc_estimate") if perf_dict else None
    sq_cindex = perf_dict.get("cindex_estimate") if perf_dict else None
    sq_or = perf_dict.get("or_estimate") if perf_dict else None
    sq_hr = perf_dict.get("hr_estimate") if perf_dict else None
    sq_beta = perf_dict.get("beta_estimate") if perf_dict else None
    sq_n = perf_dict.get("n_individuals") if perf_dict else None

    sq_score = synthetic_quality_score(
        auroc=sq_auroc, cindex=sq_cindex,
        or_estimate=sq_or, hr_estimate=sq_hr,
        beta_estimate=sq_beta, n_individuals=sq_n,
        is_harmonized=is_harmonized,
    )
    sq_label, sq_color = classify_synthetic_quality(sq_score)
    q_tier, q_tier_metric = synthetic_quality_tier(
        auroc=sq_auroc, cindex=sq_cindex,
        or_estimate=sq_or, hr_estimate=sq_hr,
        beta_estimate=sq_beta,
    )

    # --- Quality ---
    interp = interpret_prs_result(pct_value, result.match_rate, auroc_val)
    quality_label, quality_color = classify_model_quality(result.match_rate, auroc_val)

    # --- Reference status (after percentile lookups which may trigger HF refresh) ---
    ref_status = catalog.reference_data_status(result.pgs_id, panel="1000g")
    ref_superpops: list[str] = list(ref_status["available_superpopulations"])
    ref_has_data = bool(ref_status["has_reference_data"])
    ref_source_label = str(ref_status["source_label"])
    ref_source_code = str(ref_status["source_code"])

    if all_pop_values:
        ref_has_data = True
        ref_superpops = sorted(set(ref_superpops) | set(all_pop_values.keys()))

    ref_status_text = (
        f"precomputed ({', '.join(ref_superpops)})" if ref_has_data else "not precomputed"
    )

    # --- Risk level ---
    risk_level, risk_level_color = _risk_level_from_percentile(pct_value)

    # --- Risk hint ---
    risk_hint = _build_risk_hint(
        result, pct_value, risk_level, ancestry_str, selected_ancestry,
        all_pop_text, ref_has_data, ref_source_label, compute_all_populations,
    )

    # --- Absolute risk & heritability ---
    z_score = _compute_z_score(pct_value)
    abs_risk = _enrich_absolute_risk(catalog, result.pgs_id, z_score)

    return EnrichedPRSResult(
        pgs_id=result.pgs_id,
        trait=result.trait_reported or "",
        trait_efo_id=trait_efo_id,
        score=round(result.score, 6),
        variants_matched=result.variants_matched,
        variants_total=result.variants_total,
        match_rate=match_pct,
        has_allele_frequencies=result.has_allele_frequencies,
        genome_build=genome_build,
        is_harmonized=is_harmonized,
        synthetic_quality=sq_score,
        synthetic_quality_label=sq_label,
        synthetic_quality_color=sq_color,
        quality_tier=q_tier,
        quality_tier_metric=q_tier_metric,
        percentile=pct_value,
        percentile_method=pct_method or "",
        match_color=match_color,
        quality_label=quality_label,
        quality_color=quality_color,
        summary=interp["summary"],
        effect_size=effect_size_str,
        classification=classification_str,
        auroc=auroc_val,
        ancestry=ancestry_str,
        n_individuals=n_individuals if n_individuals is not None else 0,
        risk_level=risk_level,
        risk_level_color=risk_level_color,
        risk_hint=risk_hint,
        selected_ancestry=selected_ancestry,
        all_population_percentiles=all_pop_text,
        pct_AFR=all_pop_values.get("AFR"),
        pct_AMR=all_pop_values.get("AMR"),
        pct_EAS=all_pop_values.get("EAS"),
        pct_EUR=all_pop_values.get("EUR"),
        pct_SAS=all_pop_values.get("SAS"),
        reference_status=ref_status_text,
        reference_source=ref_source_label,
        reference_source_code=ref_source_code,
        absolute_risk_text=abs_risk["absolute_risk_text"],
        absolute_risk_percent=abs_risk["absolute_risk_percent"],
        population_average_percent=abs_risk["population_average_percent"],
        risk_ratio_value=abs_risk["risk_ratio_value"],
        absolute_risk_method=abs_risk["absolute_risk_method"],
        absolute_risk_detail=abs_risk["absolute_risk_detail"],
        heritability=abs_risk["heritability"],
        heritability_detail=abs_risk["heritability_detail"],
        heritability_metrics=abs_risk["heritability_metrics"],
        risk_agreement=abs_risk["risk_agreement"],
        risk_estimates_by_method=abs_risk["risk_estimates_by_method"],
        risk_estimate_methods=abs_risk["risk_estimate_methods"],
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _all_population_percentiles(
    catalog: PRSCatalog, score: float, pgs_id: str
) -> dict[str, float]:
    values: dict[str, float] = {}
    for sp in SUPERPOPULATIONS:
        pct, method = catalog.percentile(score, pgs_id, ancestry=sp)
        if pct is not None and method == "reference_panel":
            values[sp] = round(pct, 1)
    return values


def _compute_z_score(pct_value: float | None) -> float | None:
    if pct_value is None:
        return None
    try:
        return _norm_ppf(pct_value / 100.0) if 0 < pct_value < 100 else 0.0
    except ValueError:
        return 0.0


def _risk_level_from_percentile(pct_value: float | None) -> tuple[str, str]:
    if pct_value is None:
        return "", "gray"
    if pct_value >= 90:
        return "High predisposition", "red"
    if pct_value >= 75:
        return "Above average predisposition", "orange"
    if pct_value >= 25:
        return "Average predisposition", "gray"
    return "Below average predisposition", "blue"


def _build_risk_hint(
    result: PRSResult,
    pct_value: float | None,
    risk_level: str,
    ancestry_str: str,
    selected_ancestry: str,
    all_pop_text: str,
    ref_has_data: bool,
    ref_source_label: str,
    compute_all_populations: bool,
) -> str:
    trait_name = result.trait_reported or result.pgs_id
    pop_label = ancestry_str or selected_ancestry or "the reference population"

    if pct_value is not None:
        pct_int = int(pct_value)
        sfx = "th"
        if pct_int % 100 not in (11, 12, 13):
            sfx = {1: "st", 2: "nd", 3: "rd"}.get(pct_int % 10, "th")
        hint = (
            f"Your PRS for {trait_name} is at the {pct_int}{sfx} percentile — "
            f"{risk_level.lower()} compared to the {pop_label} reference population. "
            "For standard PRS models, higher percentile = more genetic variants "
            "associated with increased risk."
        )
        if all_pop_text:
            hint += f" Available 1000G population percentiles: {all_pop_text}."
    else:
        hint = (
            f"No reference percentile is available for {trait_name}. "
            "The raw score is model-specific and cannot be read as protective or risky "
            "without a population reference. Try selecting a different ancestry or "
            "checking whether a reference panel exists for this score."
        )
        if compute_all_populations:
            hint += (
                " All-population lookup is enabled, but no 1000G reference "
                "distribution is currently available for this PGS ID."
            )

    if ref_has_data:
        hint += (
            f" Reference distributions source: {ref_source_label}. "
            "These are precomputed from reference panel scoring and are not "
            "provided directly by the PGS Catalog API."
        )
    else:
        hint += (
            f" Reference distributions source status: {ref_source_label}. "
            "Percentile falls back to theoretical/AUROC approximation when available."
        )
    return hint


def _enrich_absolute_risk(
    catalog: PRSCatalog, pgs_id: str, z_score: float | None
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "absolute_risk_text": "",
        "absolute_risk_percent": None,
        "population_average_percent": None,
        "risk_ratio_value": None,
        "absolute_risk_method": "",
        "absolute_risk_detail": "",
        "heritability": "N/A",
        "heritability_detail": "Absolute risk was not computed, so h²-liability was not checked.",
        "heritability_metrics": [],
        "risk_agreement": "",
        "risk_estimates_by_method": {},
        "risk_estimate_methods": [],
    }
    if z_score is None:
        return out

    bundle = catalog.absolute_risk_bundle(pgs_id, z_score)

    # --- Heritability ---
    h2_estimates = [est for est in bundle.estimates if est.h2_value is not None]
    heritability_metrics: list[dict[str, str]] = []
    if h2_estimates:
        h_parts: list[str] = []
        h_detail_parts = [
            "h² means population-level heritability: the fraction of trait variation "
            "statistically associated with genetic differences in a studied population, "
            "not an individual causal percentage."
        ]
        for est in h2_estimates:
            ancestry_label = f"{est.ancestry} " if est.ancestry else ""
            population_label = (
                SUPERPOPULATION_LABELS.get(est.ancestry, est.ancestry)
                if est.ancestry
                else "Combined population"
            )
            source_label = est.h2_source or est.method_label
            h_parts.append(f"{ancestry_label}h²={est.h2_value:.3f} ({source_label})")
            heritability_metrics.append({
                "population": population_label,
                "h2": f"{est.h2_value:.3f}",
                "source": source_label,
                "risk": f"{est.absolute_risk * 100:.1f}%",
                "ratio": f"{est.risk_ratio:.2f}x",
                "confidence": est.confidence,
            })
            detail = (
                f"{est.method_label}: h²={est.h2_value:.3f}, "
                f"risk={est.absolute_risk * 100:.1f}%, "
                f"ratio={est.risk_ratio:.2f}x, confidence={est.confidence}"
            )
            if est.h2_source_detail:
                detail += f", source detail={est.h2_source_detail}"
            h_detail_parts.append(detail)
        out["heritability"] = "; ".join(h_parts)
        out["heritability_detail"] = " | ".join(h_detail_parts)
    else:
        out["heritability"] = "No mapped h²"
        out["heritability_detail"] = (
            bundle.heritability_detail
            or "No mapped h²-liability estimate is available for this trait."
        )
    out["heritability_metrics"] = heritability_metrics

    # --- Best absolute risk estimate ---
    if bundle.best_estimate is not None:
        best = bundle.best_estimate
        user_pct = best.absolute_risk * 100
        pop_pct = best.population_prevalence * 100
        out["absolute_risk_percent"] = user_pct
        out["population_average_percent"] = pop_pct
        out["risk_ratio_value"] = best.risk_ratio
        out["absolute_risk_method"] = best.method_label
        out["absolute_risk_text"] = f"{user_pct:.1f}% (pop. avg: {pop_pct:.1f}%)"

        detail_parts = [
            f"Best estimate: {user_pct:.1f}% via {best.method_label}",
            f"Population average: {pop_pct:.1f}%",
            f"Risk ratio: {best.risk_ratio:.2f}x",
            f"Prevalence source: {best.prevalence_source}",
            f"Confidence: {best.confidence}",
        ]
        if best.effect_size_citation:
            detail_parts.append(f"Citation: {best.effect_size_citation}")
        if len(bundle.estimates) > 1:
            detail_parts.append(
                f"Agreement: {bundle.agreement} "
                f"(spread: {bundle.spread_pp:.1f}pp across {len(bundle.estimates)} methods)"
            )
            for est in bundle.estimates:
                est_pct = est.absolute_risk * 100
                md = (
                    f"  {est.method_label}: {est_pct:.1f}% "
                    f"(ratio: {est.risk_ratio:.2f}x, conf: {est.confidence})"
                )
                if est.h2_value is not None:
                    md += f", h²={est.h2_value:.3f}, source={est.h2_source or 'heritability table'}"
                detail_parts.append(md)
        if best.caveats:
            detail_parts.append(f"Caveats: {'; '.join(best.caveats)}")
        out["absolute_risk_detail"] = " | ".join(detail_parts)

        agreement_label = bundle.agreement
        out["risk_agreement"] = "single" if agreement_label == "single_estimate" else agreement_label

        estimates_by_method: dict[str, str] = {}
        methods: list[str] = []
        for est in bundle.estimates:
            est_pct = est.absolute_risk * 100
            risk_text = f"{est_pct:.1f}%"
            if est.h2_value is not None:
                risk_text += f" (h²={est.h2_value:.3f})"
            estimates_by_method[est.method_label] = risk_text
            if est.method_label not in methods:
                methods.append(est.method_label)
        out["risk_estimates_by_method"] = estimates_by_method
        out["risk_estimate_methods"] = methods

    elif bundle.estimates:
        abs_risk_legacy = catalog.absolute_risk(pgs_id, z_score)
        if abs_risk_legacy is not None:
            user_pct = abs_risk_legacy.absolute_risk * 100
            pop_pct = abs_risk_legacy.population_prevalence * 100
            out["absolute_risk_percent"] = user_pct
            out["population_average_percent"] = pop_pct
            out["risk_ratio_value"] = abs_risk_legacy.risk_ratio
            out["absolute_risk_method"] = abs_risk_legacy.method
            out["absolute_risk_text"] = f"{user_pct:.1f}% (pop. avg: {pop_pct:.1f}%)"

    return out
