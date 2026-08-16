"""Canonical trait-level aggregation for dashboards, CLI plots, and AI prompts.

``summarize_trait_rows`` is the single source of truth for usable/all/high-quality
scope, best-model pick, absolute-risk pick, percentile-panel resolution, and
ancestry-ordered heritability.  The UI mixin, ``just_prs.viz`` prompt builders,
HTML reports, and just-prs-mcp must consume this module instead of re-implementing
the same rules.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

ModelScope = Literal["usable", "all", "high_quality", "high_moderate"]
PercentileSource = Literal["native", "selected"]

USABLE_MATCH_RATE_PERCENT = 50.0
NO_RISK_DATA_USABLE = "N/A (no usable model with risk data)"
NO_MAPPED_H2 = "No mapped h²"
COMBINED_POPULATION_LABEL = "Combined population"

MODEL_SCOPES: tuple[str, ...] = ("usable", "all", "high_quality", "high_moderate")
PERCENTILE_SOURCES: tuple[str, ...] = ("native", "selected")

SUPERPOPULATION_LABELS: dict[str, str] = {
    "AFR": "African",
    "AMR": "Admixed American",
    "EAS": "East Asian",
    "EUR": "European",
    "SAS": "South Asian",
}

_LABEL_TO_CODE: dict[str, str] = {
    label.lower(): code for code, label in SUPERPOPULATION_LABELS.items()
}
_LABEL_TO_CODE.update({
    "american": "AMR",
    "admixed american": "AMR",
    "combined": "",
    "combined population": "",
    "all": "",
    "meta": "",
})

_HIGH_QUALITY_LABELS = frozenset({"high"})
_MODERATE_QUALITY_LABELS = frozenset({"moderate", "medium", "med"})
_QUALITY_SCORE_FALLBACK: dict[str, float] = {
    "high": 80.0,
    "normal": 60.0,
    "moderate": 50.0,
    "low": 25.0,
    "very_low": 10.0,
    "very low": 10.0,
}


def normalize_match_rate(value: Any) -> float | None:
    """Return match rate on the 0–100 percent scale.

    Accepts a fraction (``0.7``), a percent (``70``), or a display string
    (``"70%"`` / ``"0.70"``).  Values in ``(0, 1]`` are treated as fractions.
    """
    number = _finite_number(value)
    if number is None and isinstance(value, str):
        number = _finite_number(value.strip().rstrip("%"))
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number


def parse_percentile(value: Any) -> float | None:
    """Parse a percentile from a number or display string such as ``"72.3%"``."""
    number = _finite_number(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    token = text.split("%", maxsplit=1)[0].split()[-1]
    return _finite_number(token)


def ancestry_code(label: str | None) -> str:
    """Map a population label or superpopulation code to ``AFR``/``EUR``/… or ``""``."""
    text = (label or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in SUPERPOPULATION_LABELS:
        return upper
    return _LABEL_TO_CODE.get(text.lower(), "")


def matches_ancestry(label: str | None, selected_ancestry: str | None) -> bool:
    """True when *label* refers to the same superpopulation as *selected_ancestry*."""
    wanted = ancestry_code(selected_ancestry)
    got = ancestry_code(label)
    return bool(wanted) and wanted == got


def is_combined_population(label: str | None) -> bool:
    """True for blank / 'combined' heritability rows (not a single superpopulation)."""
    text = (label or "").strip()
    if not text:
        return True
    return ancestry_code(text) == "" and text.lower() in {
        "combined",
        "combined population",
        "all",
        "meta",
        "population",
    }


def heritability_keep_codes(
    selected_ancestry: str | None,
    *,
    restrict_to_selected: bool = False,
    sample_ancestries: Iterable[str] | None = None,
) -> list[str]:
    """Superpopulation codes the h² card may list.

    A dashboard Population override keeps that one code. Otherwise keep the
    distinct ancestries detected in *sample_ancestries* (selected first). With
    no sample calls, keep the selected (majority) ancestry only — never the
    full Pan-UKBB catalog.
    """
    selected = ancestry_code(selected_ancestry)
    if restrict_to_selected:
        return [selected] if selected else []
    codes: list[str] = []
    seen: set[str] = set()
    for raw in sample_ancestries or []:
        code = ancestry_code(raw)
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    if not codes:
        return [selected] if selected else []
    if selected and selected in seen:
        return [selected, *[code for code in codes if code != selected]]
    return codes


def _filter_heritability_metrics(
    metrics: list[dict[str, Any]],
    keep_codes: list[str],
) -> list[dict[str, Any]]:
    """Keep metrics for *keep_codes*; Combined is fallback only, never alongside."""
    if not keep_codes:
        return [
            metric
            for metric in metrics
            if is_combined_population(
                str(metric.get("population") or metric.get("ancestry") or ""),
            )
        ]
    matched = [
        metric
        for metric in metrics
        if any(
            matches_ancestry(
                str(metric.get("population") or metric.get("ancestry") or ""),
                code,
            )
            for code in keep_codes
        )
    ]
    if matched:
        return matched
    return [
        metric
        for metric in metrics
        if is_combined_population(
            str(metric.get("population") or metric.get("ancestry") or ""),
        )
    ]


def ancestry_sort_rank(label: str | None, selected_ancestry: str | None) -> int:
    """0 = selected ancestry, 1 = combined population, 2 = everything else."""
    if matches_ancestry(label, selected_ancestry):
        return 0
    if is_combined_population(label):
        return 1
    return 2


def sort_by_selected_ancestry(
    items: list[dict[str, Any]],
    selected_ancestry: str | None,
    label_key: str = "population",
) -> list[dict[str, Any]]:
    """Stable-ish sort: selected ancestry, then Combined population, then the rest."""

    def _key(item: dict[str, Any]) -> tuple[int, str]:
        label = str(
            item.get(label_key)
            or item.get("population")
            or item.get("ancestry")
            or ""
        )
        return (ancestry_sort_rank(label, selected_ancestry), label)

    return sorted(items, key=_key)


def prefer_prevalence_row(
    rows: list[dict[str, Any]],
    selected_ancestry: str | None,
) -> dict[str, Any] | None:
    """Return the prevalence row matching *selected_ancestry*, else the first row."""
    if not rows:
        return None
    if selected_ancestry:
        for row in rows:
            if matches_ancestry(row.get("ancestry"), selected_ancestry):
                return row
    return rows[0]


def is_usable_model(row: dict[str, Any]) -> bool:
    """True when the row's match rate is at least 50% marker coverage."""
    match_rate = normalize_match_rate(row.get("match_rate"))
    return match_rate is not None and match_rate >= USABLE_MATCH_RATE_PERCENT


def is_high_quality_model(row: dict[str, Any]) -> bool:
    """True when this sample's coverage is usable and the genotype-aware label is High.

    High is undefined below 50% match: a published-famous score with 43% coverage
    is not High for this genome, even if ``quality_label`` was stamped High.
    """
    return is_usable_model(row) and _row_has_quality_label(row, _HIGH_QUALITY_LABELS)


def is_high_or_moderate_model(row: dict[str, Any]) -> bool:
    """True when this sample's coverage is usable and the label is High or Moderate.

    Same coverage gate as :func:`is_high_quality_model`: a model that matched
    <50% of its markers for this genome is not a trustworthy High/Moderate
    model for this genome, whatever its stamped label says — without the gate a
    42.8%-coverage model at the 0th percentile lands in the "high + moderate"
    dashboard scope and drags the median.  The scopes are a strict hierarchy:
    all ⊇ usable ⊇ high_moderate ⊇ high_quality.
    """
    return is_usable_model(row) and _row_has_quality_label(
        row, _HIGH_QUALITY_LABELS | _MODERATE_QUALITY_LABELS
    )


def _row_has_quality_label(row: dict[str, Any], labels: frozenset[str]) -> bool:
    """Use the genotype-aware label only — never published synthetic quality."""
    for key in ("quality_label", "quality"):
        raw = row.get(key)
        if raw is None:
            continue
        label = str(raw).strip().lower().replace(" ", "_")
        if label in labels:
            return True
    return False


def filter_rows_by_scope(
    rows: list[dict[str, Any]],
    model_scope: str = "usable",
) -> list[dict[str, Any]]:
    """Return the subset of *rows* that belong to *model_scope*."""
    scope = _normalize_scope(model_scope)
    if scope == "all":
        return list(rows)
    if scope == "high_quality":
        return [row for row in rows if is_high_quality_model(row)]
    if scope == "high_moderate":
        return [row for row in rows if is_high_or_moderate_model(row)]
    return [row for row in rows if is_usable_model(row)]


def resolve_row_percentile(
    row: dict[str, Any],
    percentile_source: str = "native",
    selected_ancestry: str = "EUR",
) -> tuple[float | None, str]:
    """Return ``(percentile, panel_code)`` for one model.

    ``native`` (default) uses the stored percentile and the model's native
    superpopulation.  ``selected`` prefers ``pct_{selected_ancestry}`` and
    falls back to the native percentile when that panel is missing.
    """
    native_panel = _native_panel(row, selected_ancestry)
    native_pct = parse_percentile(row.get("percentile"))
    source = _normalize_percentile_source(percentile_source)
    if source == "selected":
        code = ancestry_code(selected_ancestry) or selected_ancestry.upper()
        selected_pct = parse_percentile(row.get(f"pct_{code}"))
        if selected_pct is not None:
            return selected_pct, code
    return native_pct, native_panel


def format_percentile_with_panel(pct: float | None, panel: str = "") -> str:
    """Format a percentile with its 1000G panel code, e.g. ``72.3 (EUR)``."""
    if pct is None:
        return "N/A"
    code = ancestry_code(panel) or (panel or "").strip().upper()
    if code:
        return f"{pct:.1f} ({code})"
    return f"{pct:.1f}"


def row_has_absolute_risk(row: dict[str, Any]) -> bool:
    """True when the row carries a real absolute-risk estimate (not N/A / empty)."""
    if _finite_number(row.get("absolute_risk_percent")) is not None:
        return True
    if _finite_number(row.get("risk_ratio_value")) is not None:
        return True
    if _finite_number(row.get("risk_ratio")) is not None:
        return True
    text = str(row.get("absolute_risk") or row.get("absolute_risk_text") or "").strip()
    if not text or text.upper() == "N/A" or text.startswith("N/A ("):
        return False
    return parse_percentile(text) is not None or _finite_number(text) is not None


def pick_best_model(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Quality-ranked representative: usable coverage first, then synthetic quality."""
    if not rows:
        return None
    return max(rows, key=_best_model_key)


def pick_absolute_risk_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Quality-ranked in-scope row that actually has an absolute-risk estimate."""
    with_risk = [row for row in rows if row_has_absolute_risk(row)]
    return pick_best_model(with_risk)


def pick_median_risk_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """In-scope row whose risk ratio sits at the median of the scoped estimates.

    The trait headline risk must be consistent with the trait median percentile
    card, so the representative model is the one closest to the median risk —
    not the single quality-best model, whose own percentile can sit far from
    the scoped median (the "median 83rd but risk 1.00x" confusion). Rows
    without a derivable ratio fall back to the quality-best row with risk.
    """
    with_risk = [row for row in rows if row_has_absolute_risk(row)]
    if not with_risk:
        return None
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in with_risk:
        _user_pct, _pop_pct, ratio, _text = _row_risk_numbers(row)
        if ratio is not None:
            ranked.append((ratio, row))
    if not ranked:
        return pick_best_model(with_risk)
    target = _median([ratio for ratio, _row in ranked])
    if target is None:
        return pick_best_model(with_risk)
    best_diff = min(abs(ratio - target) for ratio, _row in ranked)
    candidates = [row for ratio, row in ranked if abs(ratio - target) <= best_diff + 1e-12]
    return pick_best_model(candidates)


def no_risk_data_label(model_scope: str = "usable") -> str:
    """Citizen-facing N/A when the active scope has no absolute-risk estimate."""
    scope = _normalize_scope(model_scope)
    if scope == "usable":
        return NO_RISK_DATA_USABLE
    if scope == "high_quality":
        return "N/A (no high-quality model with risk data)"
    if scope == "high_moderate":
        return "N/A (no high or moderate model with risk data)"
    return "N/A (no model with risk data)"


def scope_label(
    n_scoped: int,
    model_scope: str = "usable",
) -> str:
    """Card-header phrase, e.g. ``based on 3 usable models, match ≥50%``."""
    scope = _normalize_scope(model_scope)
    noun = "model" if n_scoped == 1 else "models"
    if scope == "usable":
        return f"based on {n_scoped} usable {noun}, match ≥50%"
    if scope == "high_quality":
        return f"based on {n_scoped} high-quality {noun}"
    if scope == "high_moderate":
        return f"based on {n_scoped} high + moderate {noun}"
    return f"based on {n_scoped} {noun} (all)"


def best_of_label(
    n_scoped: int,
    model_scope: str = "usable",
) -> str:
    """Prompt/card phrase, e.g. ``most reliable of 3 usable models``.

    Deliberately says "most reliable", never "best": the ranking is by model
    *quality* (coverage + evaluation metrics), and for direction-dependent
    traits (intelligence, longevity, ...) "best" reads as "best outcome",
    which the quality-ranked model's percentile is not.
    """
    scope = _normalize_scope(model_scope)
    noun = "model" if n_scoped == 1 else "models"
    if scope == "usable":
        return f"most reliable of {n_scoped} usable {noun}"
    if scope == "high_quality":
        return f"most reliable of {n_scoped} high-quality {noun}"
    if scope == "high_moderate":
        return f"most reliable of {n_scoped} high + moderate {noun}"
    return f"most reliable of {n_scoped} {noun}"


def risk_basis_label(n_risk: int) -> str:
    """Subtext for the trait risk cards, e.g. ``median of 4 models with risk data``."""
    if n_risk <= 0:
        return "no models with risk data"
    if n_risk == 1:
        return "only model with risk data"
    return f"median of {n_risk} models with risk data"


def summarize_heritability(
    rows: list[dict[str, Any]],
    selected_ancestry: str = "EUR",
    restrict_to_selected: bool = False,
    sample_ancestries: Iterable[str] | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """De-duplicate h² metrics and keep ancestries that matter for these samples.

    ``restrict_to_selected`` (dashboard Population is a specific superpop) keeps
    only that ancestry. Otherwise keep ancestries detected in
    *sample_ancestries* (selected first). Combined is a fallback when none of
    those have a mapped estimate — it is not listed alongside them. With no
    sample calls, keep the selected ancestry only so the card does not dump
    every Pan-UKBB row.
    """
    metric_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    detail_parts: list[str] = []

    for row in rows:
        metrics = row.get("heritability_metrics", [])
        if isinstance(metrics, list):
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                key = (
                    str(metric.get("population") or metric.get("ancestry") or ""),
                    str(metric.get("h2") or ""),
                    str(metric.get("source") or ""),
                )
                if key in metric_by_key or not key[1]:
                    continue
                metric_by_key[key] = metric

        h_detail = str(row.get("heritability_detail") or "").strip()
        if h_detail and h_detail not in detail_parts:
            detail_parts.append(h_detail)

    metrics = sort_by_selected_ancestry(
        list(metric_by_key.values()),
        selected_ancestry,
        label_key="population",
    )
    keep_codes = heritability_keep_codes(
        selected_ancestry,
        restrict_to_selected=restrict_to_selected,
        sample_ancestries=sample_ancestries,
    )
    if keep_codes:
        metrics = _filter_heritability_metrics(metrics, keep_codes)
    if metrics:
        parts = [
            f"{metric.get('population', 'Population')} h²={metric.get('h2', 'N/A')}"
            + (f" ({metric.get('source')})" if metric.get("source") else "")
            for metric in metrics[:4]
        ]
        if len(metrics) > 4:
            parts.append(f"+{len(metrics) - 4} more")
        return "; ".join(parts), " | ".join(detail_parts), metrics
    return NO_MAPPED_H2, "No mapped population-level heritability estimate.", []


def format_heritability_risk_prompt(
    metrics: list[dict[str, Any]],
) -> str:
    """Compact h²-liability risk line for AI prompts (metrics already sorted)."""
    parts: list[str] = []
    for metric in metrics[:4]:
        population = str(metric.get("population") or "Population").strip() or "Population"
        h2 = str(metric.get("h2") or "N/A").strip() or "N/A"
        source = str(metric.get("source") or "").strip()
        risk = str(metric.get("risk") or "").strip()
        ratio = str(metric.get("ratio") or "").strip()
        confidence = str(metric.get("confidence") or "").strip()
        label = f"{population} h²={h2}" + (f" ({source})" if source else "")
        risk_bits = [
            bit
            for bit in (
                f"risk {risk}" if risk else "",
                f"{ratio} vs average" if ratio else "",
                confidence,
            )
            if bit
        ]
        parts.append(label + (f": {', '.join(risk_bits)}" if risk_bits else ""))
    if len(metrics) > 4:
        parts.append(f"+{len(metrics) - 4} more")
    return "; ".join(parts)


@dataclass
class TraitSummaryStats:
    """Canonical numbers for one trait group under one model scope."""

    model_scope: str
    selected_ancestry: str
    percentile_source: str
    scope_label: str
    n_total: int
    n_scoped: int
    n_usable: int
    n_high_quality: int
    scoped_rows: list[dict[str, Any]] = field(default_factory=list)
    usable_rows: list[dict[str, Any]] = field(default_factory=list)
    high_quality_rows: list[dict[str, Any]] = field(default_factory=list)
    pct_by_id: dict[str, float] = field(default_factory=dict)
    panel_by_id: dict[str, str] = field(default_factory=dict)
    median_pct: float | None = None
    mean_pct: float | None = None
    std_pct: float | None = None
    min_pct: float | None = None
    max_pct: float | None = None
    spread: float | None = None
    outliers: list[str] = field(default_factory=list)
    outlier_detail: str = ""
    overall_signal: str = "Mostly average"
    consistency: str = "Only one model"
    reliability: str = "No percentile"
    best_row: dict[str, Any] | None = None
    best_risk_row: dict[str, Any] | None = None
    n_risk_models: int = 0
    worst_row: dict[str, Any] | None = None
    best_model_pctl: float | None = None
    best_model_panel: str = ""
    typical_panel: str = ""
    absolute_risk: str = NO_RISK_DATA_USABLE
    population_average: str = "N/A"
    risk_vs_average: str = "N/A"
    risk_agreement: str = ""
    user_risk_pct: float | None = None
    pop_avg_pct: float | None = None
    heritability_text: str = NO_MAPPED_H2
    heritability_detail: str = ""
    heritability_metrics: list[dict[str, Any]] = field(default_factory=list)
    high_quality_median: float | None = None


def summarize_trait_rows(
    rows: list[dict[str, Any]],
    model_scope: str = "usable",
    selected_ancestry: str = "EUR",
    percentile_source: str = "native",
    sample_ancestries: Iterable[str] | None = None,
) -> TraitSummaryStats:
    """Aggregate one trait group's models under a single explicit scope.

    All headline statistics (median, best percentile, absolute risk, risk vs
    average, signal, spread, outliers) are computed from the scoped subset
    only.  Absolute risk is never borrowed from an out-of-scope low-match model.

    The headline absolute risk / risk-vs-average is the **median** across the
    scoped rows with risk estimates (each already refreshed at its dashboard
    percentile), so it moves together with the median-percentile card.
    ``best_risk_row`` is the representative row closest to that median ratio.
    Heritability lists ancestries detected in *sample_ancestries* (or the
    selected population when none were detected). Combined is fallback-only.
    """
    scope = _normalize_scope(model_scope)
    source = _normalize_percentile_source(percentile_source)
    ancestry = ancestry_code(selected_ancestry) or (selected_ancestry or "EUR").upper()

    usable_rows = [row for row in rows if is_usable_model(row)]
    high_quality_rows = [row for row in rows if is_high_quality_model(row)]
    scoped_rows = filter_rows_by_scope(rows, scope)

    pct_by_id: dict[str, float] = {}
    panel_by_id: dict[str, str] = {}
    for row in scoped_rows:
        pgs_id = str(row.get("pgs_id") or "")
        pct, panel = resolve_row_percentile(
            row, percentile_source=source, selected_ancestry=ancestry,
        )
        if pgs_id and pct is not None:
            pct_by_id[pgs_id] = pct
            panel_by_id[pgs_id] = panel

    pct_values = list(pct_by_id.values())
    median_pct = _median(pct_values)
    mean_pct = _mean(pct_values)
    std_pct = _std(pct_values)
    min_pct = min(pct_values) if pct_values else None
    max_pct = max(pct_values) if pct_values else None
    spread = (max_pct - min_pct) if max_pct is not None and min_pct is not None else None
    outliers, outlier_detail = detect_trait_outliers(pct_by_id)
    signal = trait_overall_signal(
        median_pct=median_pct,
        max_pct=max_pct,
        spread=spread,
        outlier_count=len(outliers),
        n_models=len(scoped_rows),
    )
    consistency = _consistency_label(
        n_models=len(scoped_rows),
        outliers=outliers,
        spread=spread,
        std_pct=std_pct,
    )
    reliability = _reliability_label(
        scope=scope,
        n_scoped=len(scoped_rows),
        n_total=len(rows),
        n_with_percentile=len(pct_by_id),
    )

    best_row = pick_best_model(scoped_rows)
    worst_row = min(scoped_rows, key=_best_model_key) if scoped_rows else None
    risk_rows = [row for row in scoped_rows if row_has_absolute_risk(row)]
    best_risk_row = pick_median_risk_row(scoped_rows)
    best_model_pctl: float | None = None
    best_model_panel = ""
    if best_row is not None:
        pgs_id = str(best_row.get("pgs_id") or "")
        best_model_pctl = pct_by_id.get(pgs_id)
        best_model_panel = panel_by_id.get(pgs_id, "")
        if best_model_pctl is None:
            best_model_pctl, best_model_panel = resolve_row_percentile(
                best_row, percentile_source=source, selected_ancestry=ancestry,
            )

    typical_panel = _typical_panel(panel_by_id, source, ancestry)
    absolute_risk, population_average, risk_vs_average, user_risk, pop_avg, risk_agreement = (
        _aggregate_risk_fields(risk_rows, best_risk_row, scope)
    )

    h_text, h_detail, h_metrics = summarize_heritability(
        scoped_rows or rows,
        selected_ancestry=ancestry,
        restrict_to_selected=(source == "selected"),
        sample_ancestries=sample_ancestries,
    )

    high_quality_pcts = [
        pct
        for row in high_quality_rows
        if (pct := resolve_row_percentile(row, source, ancestry)[0]) is not None
    ]

    return TraitSummaryStats(
        model_scope=scope,
        selected_ancestry=ancestry,
        percentile_source=source,
        scope_label=scope_label(len(scoped_rows), scope),
        n_total=len(rows),
        n_scoped=len(scoped_rows),
        n_usable=len(usable_rows),
        n_high_quality=len(high_quality_rows),
        scoped_rows=scoped_rows,
        usable_rows=usable_rows,
        high_quality_rows=high_quality_rows,
        pct_by_id=pct_by_id,
        panel_by_id=panel_by_id,
        median_pct=median_pct,
        mean_pct=mean_pct,
        std_pct=std_pct,
        min_pct=min_pct,
        max_pct=max_pct,
        spread=spread,
        outliers=outliers,
        outlier_detail=outlier_detail,
        overall_signal=signal,
        consistency=consistency,
        reliability=reliability,
        best_row=best_row,
        best_risk_row=best_risk_row,
        n_risk_models=len(risk_rows),
        worst_row=worst_row,
        best_model_pctl=best_model_pctl,
        best_model_panel=best_model_panel,
        typical_panel=typical_panel,
        absolute_risk=absolute_risk,
        population_average=population_average,
        risk_vs_average=risk_vs_average,
        risk_agreement=risk_agreement,
        user_risk_pct=user_risk,
        pop_avg_pct=pop_avg,
        heritability_text=h_text,
        heritability_detail=h_detail,
        heritability_metrics=h_metrics,
        high_quality_median=_median(high_quality_pcts),
    )


def detect_trait_outliers(values_by_id: dict[str, float]) -> tuple[list[str], str]:
    """Detect trait-level percentile outliers with small-sample safeguards."""
    values = list(values_by_id.values())
    if len(values) <= 1:
        return [], "Only one PRS model; no spread estimate."

    min_value = min(values)
    max_value = max(values)
    spread = max_value - min_value
    if len(values) < 4:
        if spread >= 35:
            low_id = min(values_by_id, key=values_by_id.get)  # type: ignore[arg-type]
            high_id = max(values_by_id, key=values_by_id.get)  # type: ignore[arg-type]
            return [], (
                f"Wide spread across {len(values)} models; lowest {low_id}={min_value:.1f}, "
                f"highest {high_id}={max_value:.1f}. Treat this as disagreement, not a proven outlier."
            )
        return [], "Models are close enough that no outlier is suggested."

    median_value = _median(values)
    if median_value is None:
        return [], "No percentile values available for outlier detection."
    deviations = [abs(value - median_value) for value in values]
    mad = _median(deviations)
    if mad is None or mad == 0:
        if spread >= 35:
            low_id = min(values_by_id, key=values_by_id.get)  # type: ignore[arg-type]
            high_id = max(values_by_id, key=values_by_id.get)  # type: ignore[arg-type]
            return [low_id, high_id], (
                "Most models cluster together, but the percentile range is wide. "
                f"Review {low_id} and {high_id} in the PRS-level table."
            )
        return [], "Models cluster tightly; no percentile outlier detected."

    outliers = [
        pgs_id
        for pgs_id, value in values_by_id.items()
        if abs(0.6745 * (value - median_value) / mad) > 2.5
    ]
    if outliers:
        return outliers, (
            "Possible outlier PRS model(s) detected using a robust percentile spread rule. "
            "Review them in the PRS-level table before trusting the trait summary."
        )
    if spread >= 35:
        return [], "No single outlier, but the models disagree widely."
    return [], "No percentile outlier detected."


def trait_overall_signal(
    median_pct: float | None,
    max_pct: float | None,
    spread: float | None,
    outlier_count: int,
    n_models: int,
) -> str:
    """Citizen-facing summary label for a grouped trait."""
    if n_models == 0:
        return "No models in scope"
    if n_models == 1:
        return "Only one model"
    if outlier_count > 0:
        return "Possible outlier"
    if spread is not None and spread >= 35:
        return "Mixed"
    if median_pct is not None and median_pct >= 75:
        return "Consistently elevated"
    if max_pct is not None and max_pct >= 75:
        return "Elevated in some models"
    return "Mostly average"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _normalize_scope(model_scope: str) -> str:
    scope = (model_scope or "usable").strip().lower()
    return scope if scope in MODEL_SCOPES else "usable"


def _normalize_percentile_source(percentile_source: str) -> str:
    source = (percentile_source or "native").strip().lower()
    return source if source in PERCENTILE_SOURCES else "native"


def _native_panel(row: dict[str, Any], fallback: str) -> str:
    for key in ("reference_panel_ancestry", "selected_ancestry", "ancestry"):
        code = ancestry_code(str(row.get(key) or ""))
        if code:
            return code
    return ancestry_code(fallback) or (fallback or "EUR").upper()


def _typical_panel(
    panel_by_id: dict[str, str],
    percentile_source: str,
    selected_ancestry: str,
) -> str:
    if percentile_source == "selected":
        return selected_ancestry
    codes = [code for code in panel_by_id.values() if code]
    if not codes:
        return ""
    if len(set(codes)) == 1:
        return codes[0]
    return "native"


def _quality_score(row: dict[str, Any]) -> float:
    raw = row.get("synthetic_quality")
    number = _finite_number(raw)
    if number is not None:
        return number
    for key in ("quality_label", "synthetic_quality_label", "quality"):
        label = str(row.get(key) or "").strip().lower().replace(" ", "_")
        if label in _QUALITY_SCORE_FALLBACK:
            return _QUALITY_SCORE_FALLBACK[label]
    return 0.0


def _best_model_key(row: dict[str, Any]) -> tuple[float, float, float]:
    usable = 1.0 if is_usable_model(row) else 0.0
    return (usable, _quality_score(row), normalize_match_rate(row.get("match_rate")) or 0.0)


def _consistency_label(
    n_models: int,
    outliers: list[str],
    spread: float | None,
    std_pct: float | None,
) -> str:
    if n_models == 0:
        return "No models in scope"
    if n_models == 1:
        return "Only one model"
    if outliers:
        return "Possible outlier"
    if spread is not None and spread >= 35:
        return "Wide spread"
    if std_pct is not None and std_pct <= 10:
        return "Consistent"
    return "Some variation"


def _reliability_label(
    scope: str,
    n_scoped: int,
    n_total: int,
    n_with_percentile: int,
) -> str:
    if n_scoped == 0:
        return "⚠ UNRELIABLE"
    if scope == "usable" and n_scoped < n_total / 2:
        return "Partial match"
    if n_with_percentile == 0:
        return "No percentile"
    return "Reliable"


def _row_risk_numbers(
    risk_row: dict[str, Any],
) -> tuple[float | None, float | None, float | None, str]:
    """Parse one row's risk fields into ``(user_pct, pop_pct, ratio, risk_text)``."""
    risk_raw = risk_row.get("absolute_risk_text")
    if risk_raw is None:
        risk_raw = risk_row.get("absolute_risk")
    risk_text = str(risk_raw or "").strip()
    user_pct = _finite_number(risk_row.get("absolute_risk_percent"))
    pop_pct = _finite_number(risk_row.get("population_average_percent"))
    raw_is_numeric = (
        user_pct is None
        and _finite_number(risk_raw) is not None
        and (not isinstance(risk_raw, str) or "%" not in risk_raw)
    )
    if raw_is_numeric:
        user_pct = _as_percent(risk_raw)
        risk_text = ""
    if user_pct is None:
        user_pct = parse_percentile(risk_text)
    if pop_pct is None and "pop. avg:" in risk_text:
        pop_pct = parse_percentile(risk_text.split("pop. avg:", maxsplit=1)[1])
    if pop_pct is None and "pop. avg." in risk_text:
        pop_pct = parse_percentile(risk_text.split("pop. avg.", maxsplit=1)[1])
    if pop_pct is None:
        pop_pct = _as_percent(risk_row.get("population_prevalence"))
    if user_pct is None:
        user_pct = _as_percent(risk_row.get("absolute_risk"))

    ratio = _finite_number(risk_row.get("risk_ratio_value"))
    if ratio is None:
        ratio = _finite_number(risk_row.get("risk_ratio"))
    if ratio is None and user_pct is not None and pop_pct not in (None, 0):
        ratio = user_pct / pop_pct
    return user_pct, pop_pct, ratio, risk_text


def _absolute_risk_fields(
    risk_row: dict[str, Any] | None,
    scope: str,
) -> tuple[str, str, str, float | None, float | None, str]:
    if risk_row is None:
        return no_risk_data_label(scope), "N/A", "N/A", None, None, ""

    user_pct, pop_pct, ratio, risk_text = _row_risk_numbers(risk_row)

    if not risk_text or risk_text.upper() == "N/A":
        if user_pct is not None and pop_pct is not None:
            risk_text = f"{user_pct:.1f}% (pop. avg: {pop_pct:.1f}%)"
        elif user_pct is not None:
            risk_text = f"{user_pct:.1f}%"
        else:
            risk_text = no_risk_data_label(scope)

    risk_vs = f"{ratio:.2f}x" if ratio is not None else "N/A"
    pop_text = f"{pop_pct:.1f}%" if pop_pct is not None else "N/A"
    agreement = str(risk_row.get("risk_agreement") or "")
    return risk_text, pop_text, risk_vs, user_pct, pop_pct, agreement


def _aggregate_risk_fields(
    risk_rows: list[dict[str, Any]],
    representative: dict[str, Any] | None,
    scope: str,
) -> tuple[str, str, str, float | None, float | None, str]:
    """Median risk across the scoped rows (each refreshed at its dashboard percentile).

    A single risk row keeps the exact per-row semantics; with several rows the
    headline user %, population %, and ratio are each the scoped median, so the
    risk cards move together with the median-percentile card instead of pinning
    to whichever single model ranks quality-best.
    """
    if not risk_rows:
        return no_risk_data_label(scope), "N/A", "N/A", None, None, ""
    if len(risk_rows) == 1:
        return _absolute_risk_fields(risk_rows[0], scope)

    user_values: list[float] = []
    pop_values: list[float] = []
    ratio_values: list[float] = []
    for row in risk_rows:
        user_pct, pop_pct, ratio, _text = _row_risk_numbers(row)
        if user_pct is not None:
            user_values.append(user_pct)
        if pop_pct is not None:
            pop_values.append(pop_pct)
        if ratio is not None:
            ratio_values.append(ratio)

    med_user = _median(user_values)
    med_pop = _median(pop_values)
    med_ratio = _median(ratio_values)
    if med_ratio is None and med_user is not None and med_pop not in (None, 0):
        med_ratio = med_user / med_pop

    if med_user is not None and med_pop is not None:
        risk_text = f"{med_user:.1f}% (pop. avg: {med_pop:.1f}%)"
    elif med_user is not None:
        risk_text = f"{med_user:.1f}%"
    else:
        risk_text = no_risk_data_label(scope)

    risk_vs = f"{med_ratio:.2f}x" if med_ratio is not None else "N/A"
    pop_text = f"{med_pop:.1f}%" if med_pop is not None else "N/A"
    agreement = str((representative or {}).get("risk_agreement") or "")
    return risk_text, pop_text, risk_vs, med_user, med_pop, agreement


def _as_percent(value: Any) -> float | None:
    number = _finite_number(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number
