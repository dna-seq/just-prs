"""Reusable PRS computation mixin and helper functions.

This module contains ``PRSComputeStateMixin`` and all supporting constants
and helper functions.  It deliberately contains **no concrete rx.State
subclasses** so that consumer apps can ``from prs_ui.mixin import
PRSComputeStateMixin`` without registering demo-app states.
"""

import csv
import io
import json
import math
import os
import urllib.parse
from pathlib import Path
from typing import Any

import polars as pl
import reflex as rx
from reflex_mui_datagrid.lazyframe_grid import (
    LazyFrameGridMixin,
    _get_cache,
    apply_filter_model,
)

from just_prs.enrich import enrich_prs_result
from just_prs.ftp import METADATA_FILES
from just_prs.prs import PRSEngine, compute_prs, compute_prs_duckdb
from just_prs.prs_catalog import PRSCatalog
from just_prs.reference import SUPERPOPULATIONS
from just_prs.scoring import resolve_cache_dir

SHEET_NAMES: list[str] = list(METADATA_FILES.keys())

SHEET_LABELS: dict[str, str] = {
    "scores": "Scores",
    "publications": "Publications",
    "efo_traits": "EFO Traits",
    "score_development_samples": "Dev Samples",
    "performance_metrics": "Performance",
    "evaluation_sample_sets": "Eval Samples",
    "cohorts": "Cohorts",
}

SUPERPOPULATION_LABELS: dict[str, str] = {
    "AFR": "African",
    "AMR": "American",
    "EAS": "East Asian",
    "EUR": "European",
    "SAS": "South Asian",
}


def _resolve_cache_dir() -> Path:
    raw = os.environ.get("PRS_CACHE_DIR", "")
    if raw:
        return Path(raw)
    return resolve_cache_dir()


def _resolve_preloaded_vcf_path() -> Path | None:
    """Return the optional local VCF path used for fast UI testing."""
    if os.environ.get("PRS_UI_PRESELECT_ENABLED") != "1":
        return None
    raw = os.environ.get("PRS_UI_PRESELECT_VCF", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _resolve_preselect_query() -> str:
    """Return the optional startup score-selection query."""
    if os.environ.get("PRS_UI_PRESELECT_ENABLED") != "1":
        return ""
    return os.environ.get("PRS_UI_PRESELECT_QUERY", "").strip()


def _compute_score_column_overrides() -> dict[str, dict[str, Any]]:
    """Column overrides for the compute score selection grid."""
    return {
        "pgs_id": {
            "width": 130,
            "cellRendererType": "url",
            "cellRendererConfig": {
                "baseUrl": "https://www.pgscatalog.org/score/",
                "suffixUrl": "/",
                "color": "#1565c0",
            },
        },
        "trait_reported": {"minWidth": 150, "flex": 2, "headerName": "Trait"},
        "quality_label": {
            "width": 100,
            "headerName": "Grade",
            "cellRendererType": "badge",
            "cellRendererConfig": {
                "colorMap": {
                    "High": "#2e7d32",
                    "Normal": "#1565c0",
                    "Moderate": "#f57f17",
                    "Low": "#c62828",
                },
                "bgColorMap": {
                    "High": "#e8f5e9",
                    "Normal": "#e3f2fd",
                    "Moderate": "#fff3e0",
                    "Low": "#ffebee",
                },
            },
        },
        "weight_type": {"width": 100, "headerName": "Weight"},
        "n_variants": {"width": 100, "headerName": "N var"},
        "trait_efo": {"minWidth": 130, "flex": 1, "headerName": "Trait EFO"},
        "citation": {"minWidth": 120, "flex": 1, "headerName": "Citation"},
        "pmid": {
            "width": 110,
            "headerName": "PubMed",
            "cellRendererType": "url",
            "cellRendererConfig": {
                "baseUrl": "https://pubmed.ncbi.nlm.nih.gov/",
                "suffixUrl": "/",
                "color": "#1565c0",
            },
        },
        "release_date": {"width": 110},
        "trait_efo_id": {
            "width": 130,
            "cellRendererType": "url",
            "cellRendererConfig": {
                "baseUrl": "http://www.ebi.ac.uk/efo/",
                "color": "#1565c0",
            },
        },
        "quality_score": {"width": 80, "headerName": "Score"},
        "score_source": {
            "width": 110,
            "headerName": "Source",
            "cellRendererType": "badge",
            "cellRendererConfig": {
                "colorMap": {
                    "Harmonized": "#e65100",
                    "Native": "#2e7d32",
                },
                "bgColorMap": {
                    "Harmonized": "#fff3e0",
                    "Native": "#e8f5e9",
                },
            },
        },
        "genome_build": {"width": 100, "headerName": "Original Build"},
        "pgp_id": {
            "width": 120,
            "cellRendererType": "url",
            "cellRendererConfig": {
                "baseUrl": "https://www.pgscatalog.org/publication/",
                "suffixUrl": "/",
                "color": "#1565c0",
            },
        },
        # hidden columns
        "is_harmonized": {"hide": True},
        "name": {"hide": True},
        "ftp_link": {"hide": True},
        "ftp_link_ebi": {"hide": True},
        "scoring_parquet_filename": {"hide": True},
        "scoring_parquet_path": {"hide": True},
    }


_catalog = PRSCatalog(cache_dir=_resolve_cache_dir())


def _enrich_scores_for_grid(lf: pl.LazyFrame, catalog: PRSCatalog) -> pl.LazyFrame:
    """Rename pre-computed quality column, add source badge, and join publication citations."""
    lf = lf.rename({"synthetic_score": "quality_score"}).drop("combined_quality_score")

    if "is_harmonized" in lf.collect_schema().names():
        lf = lf.with_columns(
            pl.when(pl.col("is_harmonized"))
            .then(pl.lit("Harmonized"))
            .otherwise(pl.lit("Native"))
            .alias("score_source"),
        )

    pubs = catalog.publications()
    pubs_cols = pubs.select(
        "pgp_id",
        pl.concat_str([
            pl.col("first_author"),
            pl.lit(" ("),
            pl.col("date_publication").cast(pl.Utf8),
            pl.lit(")"),
        ]).alias("citation"),
    )
    lf = lf.join(pubs_cols, on="pgp_id", how="left")
    return lf


def _enriched_to_row_dict(enriched: Any) -> dict[str, Any]:
    """Convert an ``EnrichedPRSResult`` to the dict format ``_build_prs_results_grid`` expects.

    Bridges the typed enrichment model to the stringly-typed dict the grid
    builder and CSV export consume.
    """
    row: dict[str, Any] = {
        "pgs_id": enriched.pgs_id,
        "trait": enriched.trait,
        "trait_efo_id": enriched.trait_efo_id,
        "score": enriched.score,
        "percentile": f"{enriched.percentile:.1f}" if enriched.percentile is not None else "",
        "percentile_method": enriched.percentile_method,
        "has_allele_frequencies": enriched.has_allele_frequencies,
        "match_rate": enriched.match_rate,
        "match_color": enriched.match_color,
        "variants_matched": enriched.variants_matched,
        "variants_total": enriched.variants_total,
        "effect_size": enriched.effect_size,
        "classification": enriched.classification,
        "auroc": f"{enriched.auroc:.3f}" if enriched.auroc is not None else "",
        "quality_label": enriched.quality_label,
        "quality_color": enriched.quality_color,
        "summary": enriched.summary,
        "ancestry": enriched.ancestry,
        "selected_ancestry": enriched.selected_ancestry,
        "n_individuals": enriched.n_individuals,
        "synthetic_quality": enriched.synthetic_quality,
        "synthetic_quality_label": enriched.synthetic_quality_label,
        "synthetic_quality_color": enriched.synthetic_quality_color,
        "quality_tier": enriched.quality_tier,
        "quality_tier_metric": enriched.quality_tier_metric,
        "is_harmonized": enriched.is_harmonized,
        "risk_level": enriched.risk_level,
        "risk_level_color": enriched.risk_level_color,
        "risk_hint": enriched.risk_hint,
        "all_population_percentiles": enriched.all_population_percentiles,
        "reference_status": enriched.reference_status,
        "reference_source": enriched.reference_source,
        "reference_source_code": enriched.reference_source_code,
        "absolute_risk": enriched.absolute_risk_text,
        "absolute_risk_percent": enriched.absolute_risk_percent,
        "population_average_percent": enriched.population_average_percent,
        "risk_ratio_value": enriched.risk_ratio_value,
        "absolute_risk_method": enriched.absolute_risk_method,
        "absolute_risk_detail": enriched.absolute_risk_detail,
        "heritability": enriched.heritability,
        "heritability_detail": enriched.heritability_detail,
        "heritability_metrics": enriched.heritability_metrics,
        "risk_agreement": enriched.risk_agreement,
        "risk_estimates_by_method": enriched.risk_estimates_by_method,
        "risk_estimate_methods": enriched.risk_estimate_methods,
    }
    for sp in ("AFR", "AMR", "EAS", "EUR", "SAS"):
        val = getattr(enriched, f"pct_{sp}", None)
        row[f"pct_{sp}"] = f"{val:.1f}" if val is not None else ""
    return row


def _parse_percent_text(value: Any) -> float | None:
    """Extract the first percentage-like number from a UI text field."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    token = text.split("%", maxsplit=1)[0].split()[-1]
    try:
        return float(token)
    except ValueError:
        return None


def _median(values: list[float]) -> float | None:
    """Return the median of a non-empty numeric list."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean of a non-empty numeric list."""
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    """Return sample standard deviation when at least two values exist."""
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _percentile_tone(percentile: float | None) -> str:
    """Semantic tone for citizen-facing percentile interpretation."""
    if percentile is None:
        return "neutral"
    if percentile >= 90:
        return "danger"
    if percentile >= 75:
        return "warning"
    if percentile < 25:
        return "good"
    return "neutral"


def _percentile_summary(percentile: float | None) -> str:
    """Plain-language explanation for what a percentile band means."""
    if percentile is None:
        return (
            "No percentile is available, so the raw PRS cannot be interpreted "
            "against a population reference."
        )
    if percentile >= 90:
        return (
            "This score is in the high tail of the reference population. "
            "It suggests elevated inherited predisposition, not a diagnosis."
        )
    if percentile >= 75:
        return (
            "This score is above the usual middle range of the reference population. "
            "For most PRS models, that means more inherited predisposition than average, "
            "but it is not a diagnosis and should be read together with model quality and match rate."
        )
    if percentile < 25:
        return (
            "This score is below the usual middle range. For most PRS models, "
            "that means lower inherited predisposition than average."
        )
    return "This score is within the broad middle range of the reference population."


def _percentile_badge_label(percentile: float | None) -> str:
    """Short label for compact detail-panel badges."""
    if percentile is None:
        return "No percentile available"
    if percentile >= 90:
        return "High tail (90th+)"
    if percentile >= 75:
        return "Above usual range"
    if percentile < 25:
        return "Below usual range"
    return "Usual middle range"


def _quality_tone(label: str) -> str:
    """Semantic tone for PRS model quality labels."""
    normalized = label.strip().lower()
    if normalized == "high":
        return "good"
    if normalized == "moderate":
        return "warning"
    if normalized in {"low", "very low"}:
        return "danger"
    return "neutral"


def _synthetic_quality_tone(label: str) -> str:
    """Semantic tone derived from classify_synthetic_quality labels."""
    normalized = label.strip().lower()
    if normalized == "high":
        return "good"
    if normalized == "normal":
        return "info"
    if normalized == "moderate":
        return "warning"
    if normalized == "low":
        return "danger"
    return "neutral"


def _trait_signal_tone(signal: str) -> str:
    """Semantic tone for the trait-level overall_signal label."""
    return {
        "Consistently elevated": "danger",
        "Elevated in some models": "warning",
        "Mixed": "warning",
        "Possible outlier": "warning",
        "Only one model": "neutral",
        "Mostly average": "good",
    }.get(signal, "neutral")


def _trait_consistency_tone(consistency: str) -> str:
    """Semantic tone for the trait-level inter-model consistency label."""
    return {
        "Consistent": "good",
        "Some variation": "warning",
        "Wide spread": "danger",
        "Possible outlier": "warning",
        "Only one model": "neutral",
    }.get(consistency, "neutral")


_QUALITY_SHAPE_MAP: dict[str, str] = {
    "high": "star",
    "normal": "pentagon",
    "moderate": "square",
    "low": "triangle-down",
}

_BELL_GREEN = "#2e7d32"
_BELL_GREY = "#9e9e9e"
_BELL_RED = "#c62828"


def _quality_marker_shape(quality_label: str) -> str:
    """Map synthetic quality label to a Plotly marker base shape."""
    return _QUALITY_SHAPE_MAP.get(quality_label.strip().lower(), "circle")


def _bell_curve_marker(
    base_shape: str,
    pct: float,
    is_outlier: bool,
    match_rate: float = 100.0,
) -> tuple[str, str]:
    """Determine bell curve marker symbol and color.

    Returns (plotly_symbol, hex_color):
        - Outlier: ``{shape}-open``, red
        - In average range (25-75th pctl) with >=50% match: ``{shape}`` (filled), green
        - Extreme or low match: ``{shape}-dot``, grey
    """
    if is_outlier:
        return f"{base_shape}-open", _BELL_RED
    if match_rate < 50.0:
        return f"{base_shape}-dot", _BELL_GREY
    if 25 <= pct <= 75:
        return base_shape, _BELL_GREEN
    return f"{base_shape}-dot", _BELL_GREY


def _match_rate_tone(match_rate: float) -> str:
    """Return display tone for per-model variant match rate."""
    if match_rate >= 50.0:
        return "good"
    if match_rate >= 25.0:
        return "warning"
    return "danger"


def _absolute_risk_tone(user_pct: float | None, risk_ratio: float | None) -> str:
    """Semantic tone for absolute-risk context, not just PRS percentile."""
    if user_pct is None:
        return "neutral"
    if user_pct >= 10.0 or (risk_ratio is not None and risk_ratio >= 3.0 and user_pct >= 5.0):
        return "danger"
    if user_pct >= 5.0 or (risk_ratio is not None and risk_ratio >= 2.0):
        return "warning"
    if user_pct < 2.0 and risk_ratio is not None and risk_ratio >= 1.5:
        return "info"
    if user_pct < 2.0:
        return "good"
    return "info"


def _absolute_risk_takeaway(
    trait: str,
    percentile: float | None,
    user_pct: float | None,
    pop_pct: float | None,
    risk_ratio: float | None = None,
) -> str:
    """One-sentence citizen-facing summary for the expanded result panel."""
    if user_pct is None:
        return _percentile_summary(percentile)

    trait_label = trait or "this trait"
    if pop_pct is None:
        return (
            f"Your estimated absolute risk for {trait_label} is {user_pct:.1f}%. "
            "Read this together with PRS percentile, model quality, and match rate."
        )

    if user_pct < 2.0:
        ratio_part = ""
        if risk_ratio is not None and risk_ratio >= 1.5:
            ratio_part = (
                f" That is {risk_ratio:.1f}x the population average — "
                f"relatively elevated, but still a low probability in absolute terms."
            )
        elif risk_ratio is not None and risk_ratio > 1.0:
            ratio_part = (
                f" That is {risk_ratio:.1f}x the population average."
            )
        return (
            f"Your estimated absolute risk for {trait_label} is {user_pct:.1f}% "
            f"(population average {pop_pct:.1f}%).{ratio_part}"
        )
    return (
        f"Your estimated absolute risk for {trait_label} is {user_pct:.1f}% "
        f"compared with a population average of {pop_pct:.1f}%."
    )


def _absolute_risk_label(
    user_pct: float | None,
    percentile: float | None,
    risk_ratio: float | None = None,
) -> str:
    """Short label for the visual risk context card."""
    if user_pct is None:
        return _percentile_badge_label(percentile)
    if user_pct < 2.0:
        if risk_ratio is not None and risk_ratio >= 1.5:
            return f"Low absolute risk ({risk_ratio:.1f}x average)"
        return "Low absolute risk"
    if user_pct < 5.0:
        return "Modest absolute risk"
    if user_pct < 10.0:
        return "Elevated absolute risk"
    return "High absolute risk"


def _percentile_method_tone(method_label: str) -> str:
    """Semantic tone for percentile source labels in the chart side panel."""
    if method_label == "1000G ref":
        return "good"
    if method_label == "theoretical":
        return "info"
    if method_label == "AUROC est.":
        return "warning"
    return "neutral"


def _format_percentile_spread(values: list[float]) -> str:
    """Format the range of available population percentiles."""
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.1f} pct"
    return f"{min(values):.1f}-{max(values):.1f} pct"


_SYNTHETIC_QUALITY_LINK_COLORS: dict[str, str] = {
    "High": "#2e7d32",
    "Normal": "#1565c0",
    "Moderate": "#e65100",
    "Low": "#c62828",
}


def _pgs_link_items(
    pgs_ids: list[str],
    quality_labels: dict[str, str] | None = None,
) -> str:
    """Return detail-panel link items for PGS Catalog score pages.

    When *quality_labels* is provided (pgs_id -> synthetic quality label),
    each link is colored according to the model's quality tier.
    """
    items: list[dict[str, str]] = []
    for pgs_id in pgs_ids:
        item: dict[str, str] = {
            "label": pgs_id,
            "url": f"https://www.pgscatalog.org/score/{pgs_id}/",
        }
        if quality_labels:
            label = quality_labels.get(pgs_id, "")
            color = _SYNTHETIC_QUALITY_LINK_COLORS.get(label, "")
            if color:
                item["color"] = color
        items.append(item)
    return json.dumps(items)


def _quality_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count model quality labels in a stable display order."""
    quality_order = ["High", "Moderate", "Low", "Very Low"]
    counts = {quality: 0 for quality in quality_order}
    for row in rows:
        label = str(row.get("quality_label", "")).strip()
        if label in counts:
            counts[label] += 1
    return counts


def _format_quality_distribution(counts: dict[str, int]) -> str:
    """Format quality counts without ambiguous one-letter abbreviations."""
    if sum(counts.values()) == 0:
        return "Quality labels unavailable"
    return ", ".join(
        f"{quality}: {counts.get(quality, 0)}"
        for quality in ["High", "Moderate", "Low", "Very Low"]
    )


def _quality_distribution_tone(counts: dict[str, int]) -> str:
    """Return the display tone for an aggregate model-quality distribution."""
    if counts.get("Very Low", 0) > 0 or counts.get("Low", 0) > 0:
        return "danger"
    if counts.get("Moderate", 0) > 0:
        return "warning"
    if counts.get("High", 0) > 0:
        return "good"
    return "neutral"


def _trait_segment_card(
    label: str,
    rows: list[dict[str, Any]],
    pct_by_id: dict[str, float],
    tone: str,
) -> dict[str, Any]:
    """Build one summary card for a trait-level model segment."""
    if not rows:
        return {
            "label": label,
            "value": "N/A",
            "tone": "neutral",
            "subtext": "No models in this segment.",
        }
    values = list(pct_by_id.values())
    median_pct = _median(values)
    if median_pct is None:
        value = "N/A"
        range_text = "no percentile values"
    else:
        value = f"Median {median_pct:.1f}"
        range_text = f"range {min(values):.1f}-{max(values):.1f}"
    quality_text = _format_quality_distribution(_quality_distribution(rows))
    return {
        "label": label,
        "value": value,
        "tone": tone,
        "subtext": f"{len(rows)} model(s); {range_text}; {quality_text}",
    }


def _normalize_genotypes_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Return a copy of *lf* with polars-bio interval columns mapped to VCF convention.

    ``start`` -> ``pos`` (if ``pos`` is absent) and ``end`` is dropped.
    The original LazyFrame is never mutated.
    """
    cols = lf.collect_schema().names()
    if "pos" not in cols and "start" in cols:
        lf = lf.rename({"start": "pos"})
        cols = lf.collect_schema().names()
    if "end" in cols:
        lf = lf.drop("end")
    return lf


_AI_ASSISTANTS: list[dict[str, str]] = [
    {
        "name": "ChatGPT", "base_url": "https://chatgpt.com/?q=",
        "color": "#10A37F", "iconUrl": "/openai.svg",
        "limit_env": "PRS_AI_CHATGPT_MAX_CHARS", "default_limit": "3000",
    },
    {
        "name": "Claude", "base_url": "https://claude.ai/new?q=",
        "color": "#DA7756", "iconUrl": "/anthropic.svg",
        "limit_env": "PRS_AI_CLAUDE_MAX_CHARS", "default_limit": "6000",
    },
    {
        "name": "Perplexity", "base_url": "https://www.perplexity.ai/search?q=",
        "color": "#21808D", "iconUrl": "/perplexity.svg",
        "limit_env": "PRS_AI_PERPLEXITY_MAX_CHARS", "default_limit": "3000",
    },
    {
        "name": "Grok", "base_url": "https://grok.com/?q=",
        "color": "#1D1D1F", "iconUrl": "/grok.svg",
        "limit_env": "PRS_AI_GROK_MAX_CHARS", "default_limit": "3000",
    },
]

_METHODOLOGY_CONTEXT = (
    "Methodology: percentiles were computed by scoring the 1000 Genomes Project "
    "phase 3 reference panel (2,504 individuals, 5 superpopulations: AFR, AMR, "
    "EAS, EUR, SAS) on GRCh38 harmonized scoring files from the PGS Catalog. "
    "Each individual's PRS was computed as the sum of effect_weight * dosage "
    "for matched variants, then percentiles were derived per superpopulation. "
    "The user's VCF was scored with the same engine and placed on this distribution."
)

_QUALITY_METHODOLOGY = (
    "Quality scoring: each model gets a synthetic quality score (0-100) based on "
    "four tiers: T1a (AUROC/C-index reported, no penalty), T1b (Beta only, 0.95 "
    "penalty), T2 (OR/HR only, 0.90 penalty), T3 (no metric, 0.6 penalty). "
    "The score also factors cohort size (log-scaled), variant match rate, and a "
    "harmonized-score penalty if coordinates were lifted over. "
    "Quality labels: High (>=70), Normal (>=50), Moderate (>=30), Low (<30)."
)

_FORMAT_SINGLE_COMPACT = (
    "Reply in under 150 words. Start with ONE bold sentence verdict, "
    "then 2-3 bullet points (percentile meaning, confidence, context). "
    "Do NOT assume health — this may be a behavioral/physical/cognitive trait. "
    "PRS is genetic predisposition, not a measurement. Be honest about limitations."
)

_FORMAT_ADDENDUM = (
    "After the main section, you MAY add a clearly separated section "
    "(use a horizontal rule ---) with additional commentary: caveats, "
    "ancestry considerations, trait-specific biology, or links to further reading. "
    "This optional section has no word limit but should earn its length — "
    "only include it if you have genuinely useful additional context."
)

_FORMAT_SINGLE_FULL = (
    "Structure your response EXACTLY as follows (keep this part under 250 words):\n"
    "1. **Verdict** — one bold sentence (e.g. 'Your genetic score for [trait] is "
    "moderately elevated (74th percentile) with moderate confidence.')\n"
    "2. **Key numbers** — 2-4 bullet points: percentile meaning, match quality, "
    "model confidence in plain language.\n"
    "3. **Context** — 1-2 sentences: what this trait IS (health, behavioral, "
    "physical, cognitive — do NOT assume health), how much genetics vs environment "
    "matters, why PRS is one factor among many.\n"
    "4. **What to do** — 1-2 sentences: only if actionable (screening for health "
    "traits). For non-health traits, say no action is needed and why.\n"
    "Citizen scientist audience — clarity and honesty over length.\n\n"
    + _FORMAT_ADDENDUM
)

_FORMAT_TRAIT_COMPACT = (
    "Reply in under 150 words. Start with ONE bold sentence verdict about the "
    "combined result, then 3 bullet points: model agreement, percentile meaning, "
    "confidence level. Do NOT assume health — this may be a non-health trait. "
    "PRS is genetic predisposition, not a measurement. Be honest about limitations."
)

_FORMAT_TRAIT_FULL = (
    "Structure your response EXACTLY as follows (keep this part under 300 words):\n"
    "1. **Verdict** — one bold sentence (e.g. 'Five models consistently place your "
    "[trait] score in the top 10% with moderate confidence.')\n"
    "2. **Model agreement** — 2-3 bullet points: do models agree, percentile "
    "spread, which model is best and why.\n"
    "3. **What the percentile means** — 1-2 sentences in plain language. This is "
    "a genetic predisposition score, not a measurement of the trait itself.\n"
    "4. **Confidence** — 1-2 sentences: combine match rate, quality tier, and "
    "number of high-quality models into an honest statement.\n"
    "5. **Context & actions** — 1-2 sentences: what this trait IS (health, "
    "behavioral, physical, cognitive — do NOT assume health), and whether any "
    "action makes sense. For non-health traits, say no action is needed.\n"
    "Citizen scientist audience — clarity and honesty over length.\n\n"
    + _FORMAT_ADDENDUM
)


def _get_char_limit(assistant: dict[str, str]) -> int:
    return int(os.environ.get(assistant["limit_env"], assistant["default_limit"]))


def _build_score_ai_prompt(row: dict[str, Any], limit: int) -> str:
    pgs_id = str(row.get("pgs_id") or "")
    trait = str(row.get("trait") or "unknown trait")
    efo = str(row.get("trait_efo_id") or "")
    pct = row.get("percentile", "")
    score = row.get("score", "")
    match_rate = row.get("match_rate", "")
    quality = str(row.get("quality_label") or "N/A")
    sq = row.get("synthetic_quality", "")
    auroc = row.get("auroc", "")
    abs_risk = str(row.get("absolute_risk_text") or row.get("absolute_risk") or "")
    effect = str(row.get("effect_size") or "")
    classification = str(row.get("classification") or "")
    ancestry = str(row.get("ancestry") or "")
    heritability = str(row.get("heritability") or "")
    variants_matched = row.get("variants_matched", "")
    variants_total = row.get("variants_total", "")
    method = str(row.get("percentile_method") or "")
    is_harmonized = row.get("is_harmonized", False)
    risk_agreement = str(row.get("risk_agreement") or "")

    lines = [
        f"Interpret this Polygenic Risk Score (PRS) result for a citizen scientist.",
        "",
        f"Trait: {trait}" + (f" (EFO: {efo})" if efo else ""),
        f"PGS ID: {pgs_id}  https://www.pgscatalog.org/score/{pgs_id}/",
    ]
    if pct:
        lines.append(f"Percentile: {pct} (method: {method})")
    if score:
        lines.append(f"Raw PRS value: {score}")
    if variants_matched and variants_total:
        lines.append(f"Variants matched: {variants_matched}/{variants_total} ({match_rate}%)")
    elif match_rate:
        lines.append(f"Variant match rate: {match_rate}%")
    lines.append(f"Quality tier: {quality}" + (f" (synthetic score: {sq}/100)" if sq else ""))
    if auroc and str(auroc) != "N/A":
        lines.append(f"AUROC: {auroc}")
    if effect:
        lines.append(f"Effect size: {effect}")
    if classification:
        lines.append(f"Classification metric: {classification}")
    if abs_risk and abs_risk != "N/A":
        lines.append(f"Absolute risk: {abs_risk}")
    if risk_agreement:
        lines.append(f"Risk method agreement: {risk_agreement}")
    if ancestry:
        lines.append(f"Evaluation ancestry: {ancestry}")
    if heritability and heritability != "N/A":
        lines.append(f"Heritability (h²): {heritability}")
    if is_harmonized:
        orig = str(row.get("original_genome_build", "") or "")
        lines.append(f"Note: harmonized score (lifted from {orig} to GRCh38)")
    lines.append("")
    if limit >= 3000:
        lines.append(_QUALITY_METHODOLOGY)
        lines.append("")
        lines.append(_METHODOLOGY_CONTEXT)
        lines.append("")
        lines.append(_FORMAT_SINGLE_FULL)
    elif limit >= 1500:
        lines.append(_FORMAT_SINGLE_FULL)
    else:
        lines.append(_FORMAT_SINGLE_COMPACT)
    prompt = "\n".join(lines)
    if len(prompt) > limit:
        prompt = prompt[:limit - 3] + "..."
    return prompt


def _build_trait_ai_prompt(row: dict[str, Any], limit: int) -> str:
    trait = str(row.get("trait") or "unknown trait")
    efo = str(row.get("trait_efo_id") or "")
    n_models = row.get("n_models", 0)
    n_usable = row.get("usable_models", 0)
    typical = row.get("typical_percentile", "N/A")
    consistency = str(row.get("consistency") or "")
    signal = str(row.get("overall_signal") or "")
    best_id = str(row.get("best_pgs_id") or "")
    best_pctl = row.get("best_model_pctl", "N/A")
    best_risk = str(row.get("best_absolute_risk") or "N/A")
    risk_vs_avg = str(row.get("risk_vs_average") or "N/A")
    best_quality = str(row.get("best_quality") or "N/A")
    pct_range = str(row.get("percentile_range") or "N/A")
    pct_std = row.get("percentile_std", "N/A")
    reliability = str(row.get("reliability") or "")
    best_match = row.get("best_match_rate", "")
    high_q = row.get("high_confidence_models", 0)
    high_q_median = row.get("high_confidence_median", "N/A")
    outlier_ids = str(row.get("outlier_ids") or "")
    pgs_ids = str(row.get("pgs_ids") or "")

    lines = [
        f"Interpret these combined Polygenic Risk Score (PRS) results "
        f"for \"{trait}\"" + (f" (EFO: {efo})" if efo else "") + ".",
        "",
        f"Models computed: {n_models} total, {n_usable} usable (>=50% variant match)",
        f"PGS IDs: {pgs_ids}",
    ]
    if best_id:
        lines.append(
            f"Best model: {best_id} (pctl: {best_pctl}, quality: {best_quality})"
            f"  https://www.pgscatalog.org/score/{best_id}/"
        )
    lines.append(f"Median percentile (usable): {typical}")
    lines.append(f"Percentile range: {pct_range} (SD: {pct_std})")
    lines.append(f"Consistency: {consistency}")
    lines.append(f"Overall signal: {signal}")
    if reliability:
        lines.append(f"Reliability: {reliability}")
    if best_match:
        lines.append(f"Best match rate: {best_match:.1f}%")
    lines.append(f"High-quality models: {high_q}" + (f" (median pctl: {high_q_median})" if high_q else ""))
    if outlier_ids:
        lines.append(f"Outlier models: {outlier_ids}")
    if best_risk and best_risk != "N/A":
        lines.append(f"Absolute risk (best model): {best_risk}")
    if risk_vs_avg and risk_vs_avg != "N/A":
        lines.append(f"Risk vs population average: {risk_vs_avg}")
    lines.append("")
    if limit >= 3000:
        lines.append(_QUALITY_METHODOLOGY)
        lines.append("")
        lines.append(_METHODOLOGY_CONTEXT)
        lines.append("")
        lines.append(_FORMAT_TRAIT_FULL)
    elif limit >= 1500:
        lines.append(_FORMAT_TRAIT_FULL)
    else:
        lines.append(_FORMAT_TRAIT_COMPACT)
    prompt = "\n".join(lines)
    if len(prompt) > limit:
        prompt = prompt[:limit - 3] + "..."
    return prompt


def _build_ai_links(prompt_fn: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for assistant in _AI_ASSISTANTS:
        limit = _get_char_limit(assistant)
        prompt = prompt_fn(row, limit)
        encoded = urllib.parse.quote(prompt, safe="")
        url = assistant["base_url"] + encoded
        link: dict[str, Any] = {
            "label": f"Ask {assistant['name']}",
            "url": url,
            "color": assistant["color"],
        }
        if assistant.get("iconUrl"):
            link["iconUrl"] = assistant["iconUrl"]
        links.append(link)
    return links


class PRSComputeStateMixin(rx.State, mixin=True):
    """Reusable mixin for PRS score selection, computation, and result display.

    Designed for inheritance: any Reflex state class that also inherits
    ``LazyFrameGridMixin`` can mix this in to get the full PRS workflow.

    Input contract -- the host app must provide genotype data via one of
    (in order of preference):
    1. **LazyFrame (recommended)** -- call ``set_prs_genotypes_lf(lf)`` with a
       ``pl.scan_parquet()`` LazyFrame.  Memory-efficient and avoids redundant
       I/O when computing multiple scores.
    2. **Parquet path (fallback)** -- set ``prs_genotypes_path`` to a string.
       The mixin calls ``pl.scan_parquet()`` internally if no LazyFrame is set.

    The mixin reads ``genome_build``, ``cache_dir``, and ``status_message``
    from whatever parent state provides them (e.g. AppState).
    """

    selected_pgs_ids: list[str] = []
    prs_results: list[dict] = []
    prs_results_rows: list[dict] = []
    prs_results_columns: list[dict] = []
    prs_results_column_groups: list[dict] = []
    trait_summary_rows: list[dict] = []
    trait_summary_columns: list[dict] = []
    trait_summary_visible: bool = False
    prs_view_mode: str = "grouped"
    prs_computing: bool = False
    prs_progress: int = 0
    low_match_warning: bool = False
    compute_scores_loaded: bool = False
    prs_engine: str = PRSEngine.DUCKDB.value
    prs_genotypes_path: str = ""
    selected_ancestry: str = "EUR"
    compute_all_populations: bool = False
    show_all_risk_estimates: bool = True
    include_harmonized: bool = True

    _scores_initialized: bool = False
    _compute_scores_lf: pl.LazyFrame | None = None
    _prs_genotypes_lf: pl.LazyFrame | None = None

    def set_prs_view_mode(self, mode: str | list[str]) -> None:
        """Switch between 'individual' and 'grouped' result views."""
        value = mode if isinstance(mode, str) else (mode[0] if mode else "grouped")
        self.prs_view_mode = value
        if value == "grouped" and self.prs_results and not self.trait_summary_rows:
            self.build_trait_summary()

    def set_prs_engine(self, value: str) -> None:
        """Switch PRS computation engine: 'polars' (default) or 'duckdb'."""
        try:
            self.prs_engine = PRSEngine(value).value
        except ValueError:
            self.prs_engine = PRSEngine.POLARS.value

    def set_selected_ancestry(self, value: str) -> None:
        """Set the ancestry superpopulation for percentile lookup."""
        self.selected_ancestry = value

    def set_compute_all_populations(self, value: bool) -> None:
        """Enable/disable percentile lookup for all available superpopulations."""
        self.compute_all_populations = bool(value)

    def set_show_all_risk_estimates(self, value: bool) -> None:
        """Toggle multi-method absolute risk estimate display."""
        self.show_all_risk_estimates = bool(value)
        if self.prs_results:
            self._build_prs_results_grid()

    def set_include_harmonized(self, value: bool) -> Any:
        """Enable/disable including harmonized (cross-build) scores."""
        self.include_harmonized = bool(value)
        if self.compute_scores_loaded:
            yield from self.load_compute_scores()

    def set_prs_genotypes_lf(self, lf: pl.LazyFrame) -> None:
        """Provide a pre-loaded genotypes LazyFrame for PRS computation.

        The LazyFrame is stored as-is (not mutated) so that callers who share
        it with other components (e.g. a genomic data grid) are not affected.
        Column normalization (``start`` → ``pos``, dropping ``end``) is applied
        lazily inside ``_get_genotypes_lf()`` on a copy.
        """
        self._prs_genotypes_lf = lf

    def load_genotypes(self, path: str) -> None:
        """Loose-coupling hook: feed normalized genotypes from any source.

        This is the contract a genotype **source** uses to push data into this
        consumer state.  The reference source is the VCF upload (``prs-ui``),
        but a host app (e.g. ``just-dna-lite``) can call this from any source
        -- a public genome, a consumer-array file, or a pre-normalized parquet
        -- without the mixin knowing anything about where the data came from.

        ``path`` is a normalized genotypes parquet (same schema as
        ``normalize_vcf()`` output).  Passing an empty string clears the source.
        Any previously computed results are reset so stale output is never shown
        against a new genome.
        """
        self.prs_genotypes_path = path
        if path and Path(path).exists():
            self._prs_genotypes_lf = pl.scan_parquet(path)
        else:
            self._prs_genotypes_lf = None
        self.prs_results = []
        self.trait_summary_rows = []
        self.trait_summary_visible = False
        self.low_match_warning = False

    def _get_genotypes_lf(self) -> pl.LazyFrame | None:
        """Resolve genotypes: explicit LazyFrame first, then parquet path.

        Returns a normalized copy with ``pos`` column (renamed from ``start``
        if needed) and without ``end``.  The original LazyFrame stored by
        ``set_prs_genotypes_lf()`` is never modified.
        """
        if self._prs_genotypes_lf is not None:
            return _normalize_genotypes_lf(self._prs_genotypes_lf)
        if self.prs_genotypes_path and Path(self.prs_genotypes_path).exists():
            return _normalize_genotypes_lf(pl.scan_parquet(self.prs_genotypes_path))
        return None

    def _reference_percentiles_all_populations(
        self,
        prs_score: float,
        pgs_id: str,
    ) -> dict[str, float]:
        """Return available 1000G reference percentiles for all superpopulations."""
        values: dict[str, float] = {}
        for superpop in SUPERPOPULATIONS:
            pct, method = _catalog.percentile(prs_score, pgs_id, ancestry=superpop)
            if pct is not None and method == "reference_panel":
                values[superpop] = round(pct, 1)
        return values

    def _quality_rank(self, label: str) -> int:
        """Map quality labels to sortable ranks for representative-row selection."""
        return {
            "High": 4,
            "Moderate": 3,
            "Low": 2,
            "Very Low": 1,
        }.get(label, 0)

    def _trait_outliers(self, values_by_id: dict[str, float]) -> tuple[list[str], str]:
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

    def _trait_overall_signal(
        self,
        median_pct: float | None,
        max_pct: float | None,
        spread: float | None,
        outlier_count: int,
        n_models: int,
    ) -> str:
        """Citizen-facing summary label for a grouped trait."""
        if n_models <= 1:
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

    def _build_trait_summary_columns(self) -> list[dict]:
        """Build column definitions for the trait-level summary grid."""
        from reflex_mui_datagrid.models import ColumnDef

        _SIGNAL_COLORS: dict[str, str] = {
            "Consistently elevated": "#c62828",
            "Elevated in some models": "#e65100",
            "Mixed": "#f57f17",
            "Possible outlier": "#6a1b9a",
            "Only one model": "#757575",
            "Mostly average": "#2e7d32",
        }
        _SIGNAL_BG: dict[str, str] = {
            "Consistently elevated": "#ffebee",
            "Elevated in some models": "#fff3e0",
            "Mixed": "#fffde7",
            "Possible outlier": "#f3e5f5",
            "Only one model": "#f5f5f5",
            "Mostly average": "#e8f5e9",
        }
        _CONSISTENCY_COLORS: dict[str, str] = {
            "Consistent": "#2e7d32",
            "Some variation": "#f57f17",
            "Wide spread": "#c62828",
            "Possible outlier": "#6a1b9a",
            "Only one model": "#757575",
        }
        _CONSISTENCY_BG: dict[str, str] = {
            "Consistent": "#e8f5e9",
            "Some variation": "#fff3e0",
            "Wide spread": "#ffebee",
            "Possible outlier": "#f3e5f5",
            "Only one model": "#f5f5f5",
        }
        _QUALITY_COLORS: dict[str, str] = {
            "High": "#2e7d32",
            "Normal": "#1565c0",
            "Moderate": "#f57f17",
            "Low": "#c62828",
            "Very Low": "#c62828",
            "N/A": "#757575",
        }
        _QUALITY_BG: dict[str, str] = {
            "High": "#e8f5e9",
            "Normal": "#e3f2fd",
            "Moderate": "#fff3e0",
            "Low": "#ffebee",
            "Very Low": "#ffebee",
            "N/A": "#f5f5f5",
        }

        _RELIABILITY_COLORS: dict[str, str] = {
            "Reliable": "#2e7d32",
            "Partial match": "#f57f17",
            "⚠ UNRELIABLE": "#ffffff",
            "No percentile": "#757575",
        }
        _RELIABILITY_BG: dict[str, str] = {
            "Reliable": "#e8f5e9",
            "Partial match": "#fff3e0",
            "⚠ UNRELIABLE": "#c62828",
            "No percentile": "#f5f5f5",
        }

        columns = [
            ColumnDef(field="trait", header_name="Trait", min_width=180, flex=1),
            ColumnDef(field="trait_efo_id", header_name="EFO ID", min_width=140),
            ColumnDef(
                field="reliability", header_name="Data Reliability", min_width=160,
                cell_renderer_type="badge",
                cell_renderer_config={
                    "colorMap": _RELIABILITY_COLORS,
                    "bgColorMap": _RELIABILITY_BG,
                },
            ),
            ColumnDef(
                field="overall_signal", header_name="Signal", min_width=170,
                cell_renderer_type="badge",
                cell_renderer_config={
                    "colorMap": _SIGNAL_COLORS,
                    "bgColorMap": _SIGNAL_BG,
                },
            ),
            ColumnDef(
                field="best_model_pctl", header_name="Your Percentile (best model)", type="number", min_width=200,
                description="Percentile from the highest-quality model with usable match rate (≥50%)",
                cell_renderer_type="progress_bar",
                cell_renderer_config={
                    "color": "#5b5bd6", "trackColor": "#e0e0e0", "showValue": True,
                },
            ),
            ColumnDef(
                field="typical_percentile", header_name="Median Pctl (usable models)", type="number", min_width=200,
                description="Median percentile across models with match rate ≥50% only",
                cell_renderer_type="progress_bar",
                cell_renderer_config={
                    "color": "#7c7cc9", "trackColor": "#e0e0e0", "showValue": True,
                },
            ),
            ColumnDef(
                field="highest_percentile", header_name="Highest Pctl.", type="number", min_width=140,
                cell_renderer_type="progress_bar",
                cell_renderer_config={
                    "color": "#c62828", "trackColor": "#ffcdd2", "showValue": True,
                },
            ),
            ColumnDef(field="best_absolute_risk", header_name="Abs. Risk (best model)", min_width=170),
            ColumnDef(field="risk_vs_average", header_name="vs Average", min_width=110),
            ColumnDef(field="n_models", header_name="Models", type="number", min_width=90),
            ColumnDef(field="usable_models", header_name="Usable Models", type="number", min_width=130,
                      description="Models with variant match rate ≥50%"),
            ColumnDef(field="high_confidence_models", header_name="High Quality Models", type="number", min_width=150),
            ColumnDef(field="high_confidence_median", header_name="High Quality Median", min_width=160),
            ColumnDef(field="percentile_range", header_name="Pctl. Range", min_width=110),
            ColumnDef(
                field="consistency", header_name="Consistency", min_width=140,
                cell_renderer_type="badge",
                cell_renderer_config={
                    "colorMap": _CONSISTENCY_COLORS,
                    "bgColorMap": _CONSISTENCY_BG,
                },
            ),
            ColumnDef(
                field="best_quality", header_name="Best Quality", min_width=120,
                cell_renderer_type="badge",
                cell_renderer_config={
                    "colorMap": _QUALITY_COLORS,
                    "bgColorMap": _QUALITY_BG,
                },
            ),
            ColumnDef(field="best_metric", header_name="Best Metric", min_width=150),
            ColumnDef(
                field="best_match_rate", header_name="Best Match", type="number", min_width=120,
                cell_renderer_type="progress_bar",
                cell_renderer_config={
                    "color": "#43a047", "trackColor": "#e8e8e8", "showValue": True,
                },
            ),
            ColumnDef(field="outlier_ids", header_name="Outlier IDs", min_width=140),
            ColumnDef(field="pgs_ids", header_name="PGS IDs", min_width=220),
        ]
        return [column.dict() for column in columns]

    def _trait_interpretation(
        self,
        trait: str,
        median_pct: float | None,
        mean_pct: float | None,
        max_pct: float | None,
        min_pct: float | None,
        spread: float | None,
        std_pct: float | None,
        n_models: int,
        overall_signal: str,
        consistency: str,
        best_risk: str,
        outliers: list[str],
        pct_by_id: dict[str, float],
    ) -> str:
        """Build citizen-friendly interpretation text for a trait summary.

        Written for a citizen scientist who may not understand why different PRS
        models give different numbers for the same trait.
        """
        parts: list[str] = []

        # --- What the percentile means ---
        if median_pct is not None:
            band = _percentile_summary(median_pct)
            parts.append(
                f"YOUR RESULT: Across {n_models} independent PRS scoring model(s) for "
                f"{trait}, your median percentile is {median_pct:.1f} out of 100. "
                f"{band}"
            )
        else:
            parts.append(
                f"{n_models} PRS model(s) were evaluated for {trait}, "
                "but no percentile could be computed (reference data may be missing)."
            )

        # --- Risk estimate ---
        if best_risk and best_risk != "N/A":
            parts.append(
                f"ABSOLUTE RISK ESTIMATE: The best-quality model estimates your "
                f"approximate lifetime risk as {best_risk}. "
                "This is a statistical estimate based on population data, not a personal diagnosis."
            )

        # --- Explain discrepancies between models ---
        if n_models > 1:
            parts.append(
                f"WHY DO {n_models} MODELS GIVE DIFFERENT NUMBERS? "
                "Each PRS model was built by a different research team using different "
                "genetic variants, sample sizes, and statistical methods. "
                "It is normal for models to disagree — this does not mean one is 'wrong'. "
                "The median percentile is the most robust single summary."
            )

            if spread is not None and spread >= 35:
                parts.append(
                    f"WIDE DISAGREEMENT: The models span a range of {spread:.0f} percentile points "
                    f"(from {min_pct:.1f} to {max_pct:.1f}). "
                    "This large spread means the genetic signal for this trait is captured "
                    "differently by each model. Possible reasons: (1) some models use fewer "
                    "variants and have lower coverage of your genotype; (2) models trained on "
                    "different ancestries transfer imperfectly; (3) the trait itself may be "
                    "genetically complex with many small contributions. "
                    "When models disagree this much, focus on the median and treat the "
                    "spread as a measure of uncertainty."
                )
            elif consistency == "Consistent":
                parts.append(
                    "GOOD AGREEMENT: The models agree closely with each other "
                    f"(std. dev. {std_pct:.1f} points). "
                    "When multiple independent models converge, confidence in the "
                    "result is higher."
                )
            elif consistency == "Some variation":
                parts.append(
                    "MODERATE AGREEMENT: The models show some variation but no extreme "
                    "disagreement. This is typical — the median percentile is still a "
                    "reasonable summary."
                )

            if outliers:
                parts.append(
                    f"OUTLIER MODELS: {', '.join(outliers)} deviate noticeably from the "
                    "other models. Common causes: lower match rate (fewer of your variants "
                    "overlap with the model), different training ancestry, or a model that "
                    "captures a different genetic sub-signal. Check their match rate and "
                    "quality label in the PRS results table above."
                )

        # --- How to read the chart ---
        parts.append(
            "HOW TO READ THE CHART: The bell curve shows where a 'typical' person falls "
            "(center of the curve). Each colored dot is one PRS model's percentile for you. "
            "The orange line marks the median across all models. "
            "Dots clustered together = models agree; dots spread out = models disagree. "
            "Dots in the right tail (above ~75th percentile) suggest above-average "
            "genetic predisposition for this trait."
        )

        # --- Standard caveat ---
        parts.append(
            "IMPORTANT: A PRS captures only inherited genetic variants. It does not "
            "account for lifestyle, environment, diet, medications, or family-specific "
            "factors. Most people with an elevated PRS never develop the condition, "
            "and many people with a low PRS do. This is a research tool for exploring "
            "your genetic data — not a medical test or diagnosis."
        )
        return "\n\n".join(parts)

    def build_trait_summary(
        self,
        large_chart_threshold: int = 4,
        large_chart_height: int = 520,
        large_chart_max_width: int = 1800,
    ) -> None:
        """Group computed PRS rows by EFO ID and build a citizen-facing summary.

        Models within each group are ranked by synthetic_quality_score.
        """
        if not self.prs_results:
            self.trait_summary_rows = []
            self.trait_summary_columns = self._build_trait_summary_columns()
            self.trait_summary_visible = False
            self.status_message = "Compute PRS results before building a trait summary."  # type: ignore[attr-defined]
            return

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in self.prs_results:
            efo_id = str(row.get("trait_efo_id") or "").strip()
            if not efo_id:
                efo_id = str(row.get("trait") or "Unlabeled trait").strip().casefold()
            grouped.setdefault(efo_id, []).append(row)

        summary_rows: list[dict[str, Any]] = []
        for index, rows in enumerate(grouped.values()):
            rows.sort(key=lambda r: float(r.get("synthetic_quality") or 0), reverse=True)

            trait = str(rows[0].get("trait") or "Unlabeled trait")
            efo_id = str(rows[0].get("trait_efo_id") or "")
            pgs_ids = [str(row.get("pgs_id", "")) for row in rows if row.get("pgs_id")]
            pct_by_id: dict[str, float] = {}
            for row in rows:
                pgs_id = str(row.get("pgs_id", ""))
                pct = _parse_percent_text(row.get("percentile"))
                if pgs_id and pct is not None:
                    pct_by_id[pgs_id] = pct

            usable_rows = [
                row for row in rows
                if float(row.get("match_rate") or 0.0) >= 50.0
            ]
            usable_pct_by_id: dict[str, float] = {}
            for row in usable_rows:
                pgs_id = str(row.get("pgs_id", ""))
                pct = _parse_percent_text(row.get("percentile"))
                if pgs_id and pct is not None:
                    usable_pct_by_id[pgs_id] = pct

            best_usable_row = max(
                usable_rows,
                key=lambda row: (
                    float(row.get("synthetic_quality") or 0),
                    float(row.get("match_rate") or 0.0),
                ),
            ) if usable_rows else None
            best_model_pctl: float | None = None
            if best_usable_row:
                best_model_pctl = _parse_percent_text(best_usable_row.get("percentile"))

            n_usable = len(usable_rows)
            if n_usable == 0:
                reliability = "⚠ UNRELIABLE"
            elif n_usable < len(rows) / 2:
                reliability = "Partial match"
            elif not usable_pct_by_id:
                reliability = "No percentile"
            else:
                reliability = "Reliable"

            high_confidence_rows = [
                row
                for row in rows
                if str(row.get("quality_label", "")).strip() == "High"
            ]
            high_confidence_pct_by_id: dict[str, float] = {}
            for row in high_confidence_rows:
                pgs_id = str(row.get("pgs_id", ""))
                pct = _parse_percent_text(row.get("percentile"))
                if pgs_id and pct is not None:
                    high_confidence_pct_by_id[pgs_id] = pct
            high_confidence_values = list(high_confidence_pct_by_id.values())
            high_confidence_median = _median(high_confidence_values)
            confidence_segments = [
                _trait_segment_card(
                    f"Usable models (match ≥50%): {n_usable} of {len(rows)}",
                    usable_rows,
                    usable_pct_by_id,
                    "good" if n_usable > 0 else "danger",
                ),
                _trait_segment_card(
                    "All PRS models (including low match)",
                    rows,
                    pct_by_id,
                    "info",
                ),
                _trait_segment_card(
                    "High-quality PRS models only",
                    high_confidence_rows,
                    high_confidence_pct_by_id,
                    "good",
                ),
            ]

            use_pct = usable_pct_by_id if usable_pct_by_id else pct_by_id
            pct_values = list(use_pct.values())
            median_pct = _median(pct_values)
            mean_pct = _mean(pct_values)
            std_pct = _std(pct_values)
            all_pct_values = list(pct_by_id.values())
            min_pct = min(all_pct_values) if all_pct_values else None
            max_pct = max(all_pct_values) if all_pct_values else None
            spread = (max_pct - min_pct) if max_pct is not None and min_pct is not None else None
            outliers, outlier_detail = self._trait_outliers(pct_by_id)
            overall_signal = self._trait_overall_signal(
                median_pct=median_pct,
                max_pct=max_pct,
                spread=spread,
                outlier_count=len(outliers),
                n_models=len(rows),
            )

            best_row = max(
                rows,
                key=lambda row: (
                    1 if row.get("absolute_risk") else 0,
                    float(row.get("synthetic_quality") or 0),
                    float(row.get("match_rate") or 0.0),
                ),
            )
            worst_row = min(
                rows,
                key=lambda row: (
                    float(row.get("synthetic_quality") or 0),
                    float(row.get("match_rate") or 0.0),
                ),
            )
            best_risk = str(best_row.get("absolute_risk") or "N/A")
            best_pgs_id = str(best_row.get("pgs_id", ""))
            best_auroc_str = str(best_row.get("auroc", ""))
            best_auroc: float | None = None
            if best_auroc_str:
                try:
                    best_auroc = float(best_auroc_str)
                except (TypeError, ValueError):
                    pass
            best_quality = str(best_row.get("synthetic_quality_label", "")) or "N/A"
            best_user_pct = _parse_percent_text(best_row.get("absolute_risk"))
            pop_avg_pct = None
            abs_text = str(best_row.get("absolute_risk") or "")
            if "pop. avg:" in abs_text:
                pop_avg_pct = _parse_percent_text(abs_text.split("pop. avg:", maxsplit=1)[1])
            risk_vs_average = (
                f"{best_user_pct / pop_avg_pct:.2f}x"
                if best_user_pct is not None and pop_avg_pct not in (None, 0)
                else "N/A"
            )

            if len(rows) <= 1:
                consistency = "Only one model"
            elif outliers:
                consistency = "Possible outlier"
            elif spread is not None and spread >= 35:
                consistency = "Wide spread"
            elif std_pct is not None and std_pct <= 10:
                consistency = "Consistent"
            else:
                consistency = "Some variation"

            # Percentile spread chart: each PRS model as a data point
            # Shape = quality tier (star/pentagon/square/triangle-down)
            # Fill = range status: filled+green = in range with good match,
            # dot+grey = extreme or low match, open+red = outlier.
            sq_label_by_id: dict[str, str] = {
                str(row.get("pgs_id", "")): str(row.get("synthetic_quality_label") or "")
                for row in rows
            }
            row_by_id: dict[str, dict[str, Any]] = {
                str(row.get("pgs_id", "")): row
                for row in rows
                if row.get("pgs_id")
            }
            model_items: list[dict[str, Any]] = []
            model_outlier_labels: list[str] = []
            for pgs_id, pct in sorted(pct_by_id.items(), key=lambda item: item[1], reverse=True):
                is_outlier = pgs_id in outliers
                quality_label = sq_label_by_id.get(pgs_id, "")
                base_shape = _quality_marker_shape(quality_label)
                match_rate_val = float(row_by_id.get(pgs_id, {}).get("match_rate") or 100.0)
                symbol, marker_color = _bell_curve_marker(base_shape, pct, is_outlier, match_rate_val)
                model_items.append({
                    "label": pgs_id,
                    "value": pct,
                    "symbol": symbol,
                    "markerColor": marker_color,
                })
                if is_outlier:
                    model_outlier_labels.append(pgs_id)

            match_rate_items: list[dict[str, Any]] = []
            for row in sorted(rows, key=lambda item: float(item.get("match_rate") or 0.0), reverse=True):
                pgs_id = str(row.get("pgs_id", ""))
                if not pgs_id:
                    continue
                match_rate_pct = float(row.get("match_rate") or 0.0)  # already 0-100 scale
                sq_label = str(row.get("synthetic_quality_label") or "N/A")
                tier_metric = str(row.get("quality_tier_metric") or "No metric")
                match_rate_items.append({
                    "label": f"{pgs_id} ({sq_label})",
                    "value": f"{match_rate_pct:.1f}%",
                    "tone": _match_rate_tone(match_rate_pct),
                    "subtext": tier_metric,
                })

            percentile_chart: dict[str, Any] = {
                "score": median_pct,
                "scoreLabel": f"Median: {median_pct:.1f}th" if median_pct is not None else "No data",
                "items": model_items,
                "outliers": model_outlier_labels,
                "match_rate_items": match_rate_items,
                "summary": (
                    f"{len(pct_by_id)} models plotted. "
                    + (f"Range: {min_pct:.1f}–{max_pct:.1f}. " if min_pct is not None and max_pct is not None else "")
                    + (f"Outliers marked: {', '.join(model_outlier_labels)}. " if model_outlier_labels else "No outliers detected. ")
                    + "Shape = model quality (\u2605 High, \u2B1F Normal, \u25A0 Moderate, \u25BC Low). "
                    + "Filled green = in typical range (25\u201375th pctl) with good match, "
                    + "dot grey = extreme or low match, open red = outlier."
                ),
            }
            if len(rows) > large_chart_threshold:
                percentile_chart["rendererConfig"] = {
                    "height": large_chart_height,
                    "maxWidth": large_chart_max_width,
                    "showSidePanel": False,
                    "summaryPlacement": "none",
                    "labelTiers": 12,
                    "labelMaxVisible": 24,
                    "labelMinGapZ": 0.18,
                }

            # Key metrics as structured data for metric_list renderer
            key_metrics: list[dict[str, Any]] = []
            if reliability == "⚠ UNRELIABLE":
                key_metrics.append({
                    "label": "⚠ DATA UNRELIABLE",
                    "value": f"0 of {len(rows)} models usable",
                    "tone": "danger",
                    "subtext": (
                        "All models have <50% variant match rate. "
                        "Percentiles are meaningless — the score uses too few variants "
                        "compared to the reference panel. Check genome build and VCF coverage."
                    ),
                })
            elif reliability == "Partial match":
                key_metrics.append({
                    "label": "⚠ Partial data",
                    "value": f"{n_usable} of {len(rows)} models usable",
                    "tone": "warning",
                    "subtext": (
                        "Some models have <50% variant match. "
                        "Percentiles below are from usable models only."
                    ),
                })
            if best_model_pctl is not None:
                best_usable_id = str(best_usable_row.get("pgs_id", "")) if best_usable_row else ""
                key_metrics.append({
                    "label": f"Your Percentile ({best_usable_id})",
                    "value": f"{best_model_pctl:.1f}",
                    "tone": _percentile_tone(best_model_pctl),
                    "subtext": "from highest-quality model with ≥50% match rate",
                })
            elif reliability != "⚠ UNRELIABLE":
                key_metrics.append({
                    "label": "Your Percentile",
                    "value": "N/A",
                    "tone": "neutral",
                    "subtext": "No usable model with reference data available",
                })
            if median_pct is not None:
                key_metrics.append({
                    "label": "Median Percentile",
                    "value": f"{median_pct:.1f}",
                    "tone": _percentile_tone(median_pct),
                    "subtext": f"across {len(pct_values)} usable model(s)" if usable_pct_by_id else "across all models (none usable)",
                })
            if mean_pct is not None:
                key_metrics.append({
                    "label": "Mean Percentile",
                    "value": f"{mean_pct:.1f}",
                    "tone": _percentile_tone(mean_pct),
                    "subtext": "arithmetic average of usable models",
                })
            if std_pct is not None:
                key_metrics.append({
                    "label": "Model Spread (SD)",
                    "value": f"{std_pct:.1f} pts",
                    "tone": "warning" if std_pct > 15 else "neutral",
                    "subtext": "standard deviation of model percentiles",
                })
            if best_risk and best_risk != "N/A":
                key_metrics.append({
                    "label": "Absolute Risk",
                    "value": best_risk.split("(")[0].strip(),
                    "tone": "warning" if best_user_pct is not None and pop_avg_pct is not None and best_user_pct > pop_avg_pct else "neutral",
                    "subtext": f"pop. avg: {pop_avg_pct:.1f}%" if pop_avg_pct is not None else "best model",
                })
            key_metrics.append({
                "label": "Models",
                "value": f"{n_usable} usable / {len(rows)} total",
                "tone": "danger" if n_usable == 0 else "warning" if n_usable < len(rows) else "neutral",
                "subtext": f"{len(pct_by_id)} with percentiles",
            })

            best_sq = float(best_row.get("synthetic_quality") or 0)
            best_sq_label = str(best_row.get("synthetic_quality_label") or "")
            best_tier_metric = str(best_row.get("quality_tier_metric") or "No metric")
            key_metrics.append({
                "label": f"Best Model ({best_pgs_id})",
                "value": f"{best_tier_metric} — score {best_sq:.0f}",
                "tone": _synthetic_quality_tone(best_sq_label),
                "subtext": f"Rank: {best_sq_label}; match {float(best_row.get('match_rate') or 0):.1f}%",
            })

            if len(rows) > 1:
                worst_pgs_id = str(worst_row.get("pgs_id", ""))
                worst_sq = float(worst_row.get("synthetic_quality") or 0)
                worst_sq_label = str(worst_row.get("synthetic_quality_label") or "")
                worst_tier_metric = str(worst_row.get("quality_tier_metric") or "No metric")
                key_metrics.append({
                    "label": f"Worst Model ({worst_pgs_id})",
                    "value": f"{worst_tier_metric} — score {worst_sq:.0f}",
                    "tone": _synthetic_quality_tone(worst_sq_label),
                    "subtext": f"Rank: {worst_sq_label}; match {float(worst_row.get('match_rate') or 0):.1f}%",
                })

            if risk_vs_average != "N/A":
                key_metrics.append({
                    "label": "Risk vs Average",
                    "value": risk_vs_average,
                    "tone": "warning" if best_user_pct is not None and pop_avg_pct is not None and best_user_pct > pop_avg_pct else "neutral",
                    "subtext": "your risk compared to population average",
                })
            quality_dist = _quality_distribution(rows)
            quality_text = _format_quality_distribution(quality_dist)
            if sum(quality_dist.values()) > 0:
                key_metrics.append({
                    "label": "Quality Breakdown",
                    "value": quality_text,
                    "tone": _quality_distribution_tone(quality_dist),
                    "subtext": "Counts by PRS model quality label",
                })

            quick_flags: list[dict[str, Any]] = [
                {"label": overall_signal, "tone": _trait_signal_tone(overall_signal)},
                {
                    "label": f"Consistency: {consistency}",
                    "tone": _trait_consistency_tone(consistency),
                },
                {
                    "label": _percentile_badge_label(median_pct),
                    "tone": _percentile_tone(median_pct),
                },
                {
                    "label": (
                        f"{len(outliers)} outlier model(s)"
                        if outliers
                        else "No outliers detected"
                    ),
                    "tone": "warning" if outliers else "good",
                },
                {
                    "label": (
                        f"{len(high_confidence_rows)} of {len(rows)} high quality"
                        if len(high_confidence_rows) > 0
                        else f"No high-quality models in {len(rows)}"
                    ),
                    "tone": "good" if len(high_confidence_rows) >= 1 else "warning",
                },
                {
                    "label": (
                        f"Spread {spread:.0f} pctl pts" if spread is not None else "Spread N/A"
                    ),
                    "tone": (
                        "danger"
                        if spread is not None and spread >= 35
                        else "warning"
                        if spread is not None and spread >= 15
                        else "good"
                        if spread is not None
                        else "neutral"
                    ),
                },
            ]
            if reliability == "⚠ UNRELIABLE":
                quick_flags.insert(0, {
                    "label": f"⚠ UNRELIABLE: 0 of {len(rows)} models have ≥50% variant match",
                    "tone": "danger",
                })
            elif reliability == "Partial match":
                quick_flags.insert(0, {
                    "label": f"⚠ Only {n_usable} of {len(rows)} models have ≥50% match — percentiles may be inaccurate",
                    "tone": "warning",
                })

            if best_risk and best_risk != "N/A":
                quick_flags.append({
                    "label": f"Best-model risk {best_risk.split('(')[0].strip()}",
                    "tone": (
                        "warning"
                        if best_user_pct is not None
                        and pop_avg_pct is not None
                        and best_user_pct > pop_avg_pct
                        else "neutral"
                    ),
                })

            summary_rows.append({
                "id": index,
                "trait": trait,
                "trait_efo_id": efo_id,
                "reliability": reliability,
                "n_models": len(rows),
                "usable_models": n_usable,
                "overall_signal": overall_signal,
                "best_absolute_risk": best_risk,
                "population_average": f"{pop_avg_pct:.1f}%" if pop_avg_pct is not None else "N/A",
                "risk_vs_average": risk_vs_average,
                "best_model_pctl": round(best_model_pctl, 1) if best_model_pctl is not None else "N/A",
                "highest_percentile": round(max_pct, 1) if max_pct is not None else "N/A",
                "typical_percentile": round(median_pct, 1) if median_pct is not None else "N/A",
                "high_confidence_models": len(high_confidence_rows),
                "high_confidence_median": (
                    round(high_confidence_median, 1)
                    if high_confidence_median is not None
                    else "N/A"
                ),
                "percentile_range": (
                    f"{min_pct:.1f}–{max_pct:.1f}" if min_pct is not None and max_pct is not None else "N/A"
                ),
                "percentile_std": round(std_pct, 1) if std_pct is not None else "N/A",
                "outlier_count": len(outliers),
                "outlier_ids": ", ".join(outliers) if outliers else "",
                "consistency": consistency,
                "best_quality": best_quality,
                "best_metric": str(best_row.get("quality_tier_metric") or "N/A"),
                "best_pgs_id": best_pgs_id,
                "best_match_rate": max(float(row.get("match_rate") or 0.0) for row in rows),
                "pgs_ids": ", ".join(pgs_ids),
                "pgs_links": _pgs_link_items(pgs_ids, quality_labels={
                    str(row.get("pgs_id", "")): str(row.get("synthetic_quality_label") or "")
                    for row in rows
                }),
                "key_metrics": key_metrics,
                "confidence_segments": confidence_segments,
                "percentile_chart": percentile_chart,
                "trait_quick_flags": quick_flags,
                "outlier_summary": outlier_detail,
            })
            summary_rows[-1]["ai_ask"] = json.dumps(
                _build_ai_links(_build_trait_ai_prompt, summary_rows[-1])
            )

        self.trait_summary_rows = summary_rows
        self.trait_summary_columns = self._build_trait_summary_columns()
        self.trait_summary_visible = True
        self.status_message = f"Built trait summary for {len(summary_rows)} trait(s)."  # type: ignore[attr-defined]

    def _build_prs_results_grid(self) -> None:
        """Convert prs_results into DataGrid rows + column defs."""
        from reflex_mui_datagrid.models import ColumnDef

        _POP_COLORS: dict[str, tuple[str, str]] = {
            "AFR": ("#f57f17", "#fff9c4"),
            "AMR": ("#d81b60", "#f8bbd0"),
            "EAS": ("#388e3c", "#c8e6c9"),
            "EUR": ("#1976d2", "#bbdefb"),
            "SAS": ("#8e24aa", "#e1bee7"),
        }
        _MATCH_RELIABILITY_COLORS: dict[str, str] = {
            "Usable": "#2e7d32",
            "⚠ Low match": "#f57f17",
            "⚠ UNRELIABLE": "#ffffff",
        }
        _MATCH_RELIABILITY_BG: dict[str, str] = {
            "Usable": "#e8f5e9",
            "⚠ Low match": "#fff3e0",
            "⚠ UNRELIABLE": "#c62828",
        }
        cols: list[ColumnDef] = [
            ColumnDef(
                field="pgs_id", header_name="PGS ID", min_width=120,
                cell_renderer_type="url",
                cell_renderer_config={
                    "baseUrl": "https://www.pgscatalog.org/score/",
                    "suffixUrl": "/",
                    "color": "#1565c0",
                },
            ),
            ColumnDef(field="trait", header_name="Trait", min_width=150, flex=1),
            ColumnDef(
                field="match_reliability", header_name="Reliability", min_width=150,
                cell_renderer_type="badge",
                cell_renderer_config={
                    "colorMap": _MATCH_RELIABILITY_COLORS,
                    "bgColorMap": _MATCH_RELIABILITY_BG,
                },
            ),
            ColumnDef(
                field="build_source", header_name="Build",
                min_width=110,
                cell_renderer_type="badge",
                cell_renderer_config={
                    "colorMap": {
                        "Native": "#2e7d32",
                        "⚠ Harmonized": "#e65100",
                    },
                    "bgColorMap": {
                        "Native": "#e8f5e9",
                        "⚠ Harmonized": "#fff3e0",
                    },
                },
            ),
            ColumnDef(field="score", header_name="PRS Score", type="number", min_width=110),
            ColumnDef(
                field="percentile_num", header_name="Percentile", type="number",
                min_width=140,
                cell_renderer_type="progress_bar",
                cell_renderer_config={
                    "color": "#5b5bd6", "trackColor": "#e0e0e0", "showValue": True,
                },
            ),
        ]

        if self.compute_all_populations:
            for sp in SUPERPOPULATIONS:
                fg, bg = _POP_COLORS[sp]
                cols.append(ColumnDef(
                    field=f"pct_{sp}_num",
                    header_name=SUPERPOPULATION_LABELS[sp],
                    description=f"{sp} — 1000 Genomes superpopulation",
                    type="number",
                    min_width=130,
                    cell_renderer_type="progress_bar",
                    cell_renderer_config={
                        "color": fg, "trackColor": bg, "showValue": True,
                    },
                ))
            self.prs_results_column_groups = [{
                "groupId": "pop_percentiles",
                "headerName": "Percentiles by Population (1000G)",
                "children": [{"field": f"pct_{sp}_num"} for sp in SUPERPOPULATIONS],
            }]
        else:
            self.prs_results_column_groups = []

        cols.extend([
            ColumnDef(
                field="percentile_method", header_name="Pct. Method",
                min_width=110,
                cell_renderer_type="badge",
                cell_renderer_config={
                    "colorMap": {
                        "1000G ref": "#2e7d32",
                        "theoretical": "#1565c0",
                        "AUROC est.": "#e65100",
                        "unavailable": "#757575",
                    },
                    "bgColorMap": {
                        "1000G ref": "#e8f5e9",
                        "theoretical": "#e3f2fd",
                        "AUROC est.": "#fff3e0",
                        "unavailable": "#f5f5f5",
                    },
                },
            ),
            ColumnDef(field="auroc", header_name="AUROC", type="number", min_width=80),
            ColumnDef(
                field="quality_label", header_name="Quality",
                min_width=100,
                cell_renderer_type="badge",
                cell_renderer_config={
                    "colorMap": {
                        "High": "#2e7d32", "Moderate": "#f57f17",
                        "Low": "#c62828", "Very Low": "#c62828",
                    },
                    "bgColorMap": {
                        "High": "#e8f5e9", "Moderate": "#fff3e0",
                        "Low": "#ffebee", "Very Low": "#ffebee",
                    },
                },
            ),
            ColumnDef(field="absolute_risk", header_name="Absolute Risk (best)", min_width=180),
            ColumnDef(
                field="match_rate", header_name="Match Rate", type="number",
                min_width=130,
                cell_renderer_type="progress_bar",
                cell_renderer_config={
                    "color": "#43a047", "trackColor": "#e8e8e8", "showValue": True,
                },
            ),
        ])

        if self.show_all_risk_estimates:
            risk_method_fields: list[str] = []
            all_methods: list[str] = []
            for r in self.prs_results:
                for m in r.get("risk_estimate_methods", []):
                    if m not in all_methods:
                        all_methods.append(m)

            for method_label in all_methods:
                field = f"risk_{method_label.replace(' ', '_').replace('²', '2').replace('(', '').replace(')', '')}"
                risk_method_fields.append(field)
                cols.append(ColumnDef(
                    field=field,
                    header_name=method_label,
                    min_width=130,
                    type="string",
                ))

            if risk_method_fields:
                existing_groups = list(self.prs_results_column_groups)
                existing_groups.append({
                    "groupId": "risk_estimates",
                    "headerName": "Absolute Risk by Method",
                    "children": [{"field": f} for f in risk_method_fields],
                })
                self.prs_results_column_groups = existing_groups

        rows: list[dict[str, Any]] = []
        for i, r in enumerate(self.prs_results):
            pct_str = r.get("percentile", "")
            pct_num: float | str = "N/A"
            if pct_str:
                try:
                    pct_num = float(pct_str)
                except (TypeError, ValueError):
                    pass

            method_raw = r.get("percentile_method", "")
            method_label = {
                "reference_panel": "1000G ref",
                "theoretical": "theoretical",
                "auroc_approx": "AUROC est.",
                "": "unavailable",
            }.get(method_raw, method_raw)

            auroc_raw = r.get("auroc", "")
            auroc_num: float | str = "N/A"
            if auroc_raw:
                try:
                    auroc_num = float(auroc_raw)
                except (TypeError, ValueError):
                    pass

            # Keep source labels compact; detailed methodology lives in the collapsible guide.
            ref_source = r.get("reference_source", "")
            ref_source_detail = ref_source

            # Build effect size + classification detail for the CSV and compact metric cards.
            effect_size_val = r.get("effect_size", "")
            classification_val = r.get("classification", "")
            effect_size_detail = " | ".join(
                p for p in [effect_size_val, classification_val] if p
            ) or "N/A"

            percentile_items: list[dict[str, Any]] = []
            percentile_outliers: list[str] = []
            if self.compute_all_populations:
                for sp in SUPERPOPULATIONS:
                    pct_for_pop = _parse_percent_text(r.get(f"pct_{sp}"))
                    if pct_for_pop is not None:
                        percentile_items.append({
                            "label": SUPERPOPULATION_LABELS.get(sp, sp),
                            "value": pct_for_pop,
                            "tone": _percentile_tone(pct_for_pop),
                        })
                        if pct_for_pop >= 90 or pct_for_pop < 10:
                            percentile_outliers.append(SUPERPOPULATION_LABELS.get(sp, sp))
            if not percentile_items and isinstance(pct_num, float):
                selected_pop = str(r.get("selected_ancestry") or self.selected_ancestry)
                percentile_items.append({
                    "label": SUPERPOPULATION_LABELS.get(selected_pop, selected_pop),
                    "value": pct_num,
                    "tone": _percentile_tone(pct_num),
                })
                if pct_num >= 90 or pct_num < 10:
                    percentile_outliers.append(SUPERPOPULATION_LABELS.get(selected_pop, selected_pop))
            absolute_risk_percent = r.get("absolute_risk_percent")
            absolute_risk_value = (
                float(absolute_risk_percent)
                if isinstance(absolute_risk_percent, int | float)
                else None
            )
            population_average_percent = r.get("population_average_percent")
            population_average_value = (
                float(population_average_percent)
                if isinstance(population_average_percent, int | float)
                else None
            )
            risk_ratio_raw = r.get("risk_ratio_value")
            risk_ratio = (
                float(risk_ratio_raw)
                if isinstance(risk_ratio_raw, int | float)
                else None
            )
            match_rate = float(r.get("match_rate") or 0.0)
            population_values = [
                float(item["value"])
                for item in percentile_items
                if isinstance(item.get("value"), int | float)
            ]
            percentile_value = pct_num if isinstance(pct_num, float) else None
            chart_takeaway = _absolute_risk_takeaway(
                trait=str(r.get("trait") or ""),
                percentile=percentile_value,
                user_pct=absolute_risk_value,
                pop_pct=population_average_value,
                risk_ratio=risk_ratio,
            )
            row_summary = str(r.get("summary") or "").strip()
            if row_summary and row_summary not in chart_takeaway:
                chart_takeaway = f"{chart_takeaway} {row_summary}"
            percentile_side_items: list[dict[str, Any]] = [
                {
                    "label": "Your percentile",
                    "value": (
                        f"{percentile_value:.1f}th"
                        if percentile_value is not None
                        else "Not available"
                    ),
                    "tone": _percentile_tone(percentile_value),
                    "subtext": _percentile_badge_label(percentile_value),
                },
                {
                    "label": "Absolute risk",
                    "value": (
                        f"{absolute_risk_value:.1f}%"
                        if absolute_risk_value is not None
                        else "N/A"
                    ),
                    "tone": _absolute_risk_tone(absolute_risk_value, risk_ratio),
                    "subtext": _absolute_risk_label(
                        absolute_risk_value,
                        percentile_value,
                        risk_ratio,
                    ),
                },
                {
                    "label": "Scoring method",
                    "value": method_label,
                    "tone": _percentile_method_tone(method_label),
                    "subtext": f"Match rate {match_rate:.1f}%",
                },
                {
                    "label": "Model quality",
                    "value": r.get("quality_label", "N/A") or "N/A",
                    "tone": _quality_tone(str(r.get("quality_label") or "")),
                    "subtext": (
                        f"AUROC {auroc_num}"
                        if auroc_num != "N/A"
                        else "AUROC unavailable"
                    ),
                },
            ]
            if r.get("is_harmonized"):
                orig_build = str(r.get("original_genome_build", "") or "")
                percentile_side_items.append({
                    "label": "Build source",
                    "value": "Harmonized",
                    "tone": "warning",
                    "subtext": f"Lifted from {orig_build}" if orig_build else "Coordinate liftover applied",
                })
            if len(population_values) > 1:
                percentile_side_items.append({
                    "label": "Population spread",
                    "value": _format_percentile_spread(population_values),
                    "tone": "info",
                    "subtext": "Across available 1000G populations",
                })
            percentile_chart = {
                "score": percentile_value,
                "scoreLabel": (
                    f"You: {pct_num:.1f} of 100 ({r.get('selected_ancestry') or self.selected_ancestry})"
                    if percentile_value is not None
                    else "No percentile"
                ),
                "items": percentile_items,
                "outliers": percentile_outliers,
                "summary": "",
                "interpretation": chart_takeaway,
                "sideItems": percentile_side_items,
            }
            risk_context_items: list[dict[str, Any]] = [
                {
                    "label": "Estimated absolute risk",
                    "value": (
                        f"{absolute_risk_value:.1f}%"
                        if absolute_risk_value is not None
                        else "N/A"
                    ),
                    "tone": _absolute_risk_tone(absolute_risk_value, risk_ratio),
                    "subtext": _absolute_risk_label(
                        absolute_risk_value,
                        pct_num if isinstance(pct_num, float) else None,
                        risk_ratio,
                    ),
                },
                {
                    "label": "Population average",
                    "value": (
                        f"{population_average_value:.1f}%"
                        if population_average_value is not None
                        else "N/A"
                    ),
                    "tone": "neutral",
                    "subtext": "Baseline risk",
                },
                {
                    "label": "Relative to average",
                    "value": f"{risk_ratio:.2f}x" if risk_ratio is not None else "N/A",
                    "tone": _absolute_risk_tone(absolute_risk_value, risk_ratio),
                    "subtext": "Ratio can be high even when absolute risk is low",
                },
                {
                    "label": "Population percentile spread",
                    "value": _format_percentile_spread(population_values),
                    "tone": "info" if len(population_values) > 1 else "neutral",
                    "subtext": (
                        "Across available 1000G populations"
                        if len(population_values) > 1
                        else "Enable all populations to compare"
                    ),
                },
            ]
            heritability_metrics = r.get("heritability_metrics", [])
            if isinstance(heritability_metrics, list) and heritability_metrics:
                risk_context_items.append({
                    "label": "What h² means",
                    "value": "Population-level",
                    "tone": "neutral",
                    "subtext": "Fraction of trait variation statistically associated with genetics, not an individual causal percentage.",
                })
                for metric in heritability_metrics:
                    if not isinstance(metric, dict):
                        continue
                    population = str(metric.get("population") or "Population")
                    source = str(metric.get("source") or "heritability table")
                    confidence = str(metric.get("confidence") or "unknown")
                    risk_context_items.append({
                        "label": f"h² {population}",
                        "value": f"{metric.get('h2', 'N/A')} ({source})",
                        "tone": "neutral",
                        "subtext": (
                            f"Risk {metric.get('risk', 'N/A')}; "
                            f"{metric.get('ratio', 'N/A')} vs population average; "
                            f"{confidence} confidence."
                        ),
                    })
            else:
                risk_context_items.append({
                    "label": "Heritability (h²)",
                    "value": str(r.get("heritability") or "N/A"),
                    "tone": "neutral",
                    "subtext": "No mapped population-level estimate.",
                })
            if match_rate < 10:
                match_tone = "danger"
            elif match_rate < 50:
                match_tone = "warning"
            else:
                match_tone = "good"
            model_context_items: list[dict[str, Any]] = [
                {
                    "label": "Model quality",
                    "value": r.get("quality_label", "N/A") or "N/A",
                    "tone": _quality_tone(str(r.get("quality_label") or "")),
                    "subtext": (
                        f"AUROC {auroc_num}: how well this PRS separated affected vs unaffected people in evaluation; "
                        "below 0.60 is weak discrimination."
                        if auroc_num != "N/A"
                        else "AUROC unavailable; model discrimination was not reported."
                    ),
                },
                {
                    "label": "Variant match",
                    "value": f"{match_rate:.1f}%",
                    "tone": match_tone,
                    "subtext": (
                        f"Matched variants: {r.get('variants_matched', 0)} / {r.get('variants_total', 0)}. "
                        "Low match means the score used only part of the model; check genome build and VCF coverage."
                    ),
                },
                {
                    "label": "Evaluation population",
                    "value": r.get("ancestry", "N/A") or "N/A",
                    "tone": "neutral",
                    "subtext": "Population used in the PGS Catalog evaluation metadata.",
                },
                {
                    "label": "Risk-method agreement",
                    "value": r.get("risk_agreement", "N/A") or "N/A",
                    "tone": {
                        "high": "good",
                        "moderate": "warning",
                        "low": "danger",
                        "single": "neutral",
                    }.get(str(r.get("risk_agreement") or ""), "neutral"),
                    "subtext": (
                        "Agreement between absolute-risk calculation methods, not model accuracy. "
                        "It can be high even when AUROC is low."
                    ),
                },
                {
                    "label": "Effect size",
                    "value": effect_size_val or "N/A",
                    "tone": "neutral",
                    "subtext": (
                        f"{classification_val}. Effect size estimates risk per PRS unit or SD in the evaluation study."
                        if classification_val
                        else "PGS Catalog effect size; separate from AUROC discrimination."
                    ),
                },
            ]
            if match_rate < 10:
                match_reliability = "⚠ UNRELIABLE"
            elif match_rate < 50:
                match_reliability = "⚠ Low match"
            else:
                match_reliability = "Usable"

            suggestion_badges = [{
                "label": _percentile_badge_label(pct_num if isinstance(pct_num, float) else None),
                "tone": _percentile_tone(pct_num if isinstance(pct_num, float) else None),
            }]
            if match_rate < 10:
                suggestion_badges.insert(0, {
                    "label": "⚠ UNRELIABLE — very low match, percentile is meaningless",
                    "tone": "danger",
                })
            elif match_rate < 50:
                suggestion_badges.insert(0, {
                    "label": "⚠ Low match — percentile may be inaccurate",
                    "tone": "warning",
                })
            else:
                suggestion_badges.append({
                    "label": "Match rate usable",
                    "tone": "good",
                })

            build_source = "⚠ Harmonized" if r.get("is_harmonized") else "Native"

            row: dict[str, Any] = {
                "id": i,
                "pgs_id": r.get("pgs_id", ""),
                "build_source": build_source,
                "match_reliability": match_reliability,
                "trait": r.get("trait", ""),
                "score": r.get("score", 0),
                "percentile_num": pct_num,
                "percentile_method": method_label,
                "auroc": auroc_num,
                "quality_label": r.get("quality_label", ""),
                "ancestry": r.get("ancestry", ""),
                "reference_status": r.get("reference_status", ""),
                "match_rate": r.get("match_rate", 0),
                "variants_text": f"{r.get('variants_matched', 0)} / {r.get('variants_total', 0)}",
                "effect_size": r.get("effect_size", ""),
                "risk_level": r.get("risk_level", ""),
                "risk_hint": r.get("risk_hint", ""),
                "summary": r.get("summary", ""),
                "all_population_percentiles": r.get("all_population_percentiles", ""),
                "reference_source_detail": ref_source_detail,
                "effect_size_detail": effect_size_detail,
                "absolute_risk": r.get("absolute_risk", ""),
                "absolute_risk_detail": r.get("absolute_risk_detail", ""),
                "heritability": r.get("heritability", ""),
                "heritability_detail": r.get("heritability_detail", ""),
                "risk_agreement": r.get("risk_agreement", ""),
                "risk_context": risk_context_items,
                "model_context": model_context_items,
                "population_percentiles_chart": percentile_chart,
                "population_percentiles_summary": chart_takeaway,
                "result_suggestions": suggestion_badges,
                "ai_ask": json.dumps(_build_ai_links(_build_score_ai_prompt, r)),
            }

            if self.show_all_risk_estimates:
                for method_label_key, risk_text in r.get("risk_estimates_by_method", {}).items():
                    field = f"risk_{method_label_key.replace(' ', '_').replace('²', '2').replace('(', '').replace(')', '')}"
                    row[field] = risk_text

            if self.compute_all_populations:
                for sp in SUPERPOPULATIONS:
                    val_str = r.get(f"pct_{sp}", "")
                    val_num: float | str = "N/A"
                    if val_str:
                        try:
                            val_num = float(val_str)
                        except (TypeError, ValueError):
                            pass
                    row[f"pct_{sp}_num"] = val_num

            rows.append(row)

        self.prs_results_rows = rows
        self.prs_results_columns = [c.dict() for c in cols]

    def initialize_prs(self) -> Any:
        """Auto-load cleaned scores on first access."""
        if self._scores_initialized:
            return
        self._scores_initialized = True
        yield from self.load_compute_scores()

    def set_prs_genome_build(self, value: str) -> Any:
        """Set genome build and reload compute scores if already loaded."""
        self.genome_build = value  # type: ignore[attr-defined]
        if self.compute_scores_loaded:
            yield from self.load_compute_scores()

    def load_compute_scores(self) -> Any:
        """Load cleaned scores into the compute grid, filtered by genome build."""
        self.status_message = "Loading scores for selection..."  # type: ignore[attr-defined]
        yield
        lf = _catalog.scores(
            genome_build=self.genome_build,  # type: ignore[attr-defined]
            include_harmonized=self.include_harmonized,
        )
        lf = _enrich_scores_for_grid(lf, _catalog)
        self._compute_scores_lf = lf
        self.compute_scores_loaded = True
        self.selected_pgs_ids = []
        yield from self.set_lazyframe(  # type: ignore[attr-defined]
            lf,
            chunk_size=500,
            eager_value_options_row_limit=0,
            column_overrides=_compute_score_column_overrides(),
        )
        total = lf.select(pl.len()).collect().item()
        if "is_harmonized" in lf.collect_schema().names():
            native_count = lf.filter(~pl.col("is_harmonized")).select(pl.len()).collect().item()
            harmonized_count = total - native_count
            if harmonized_count > 0:
                self.status_message = (  # type: ignore[attr-defined]
                    f"Loaded {total} scores for {self.genome_build}"  # type: ignore[attr-defined]
                    f" ({native_count} native, {harmonized_count} harmonized)"
                )
            else:
                self.status_message = f"Loaded {total} scores for {self.genome_build}"  # type: ignore[attr-defined]
        else:
            self.status_message = f"Loaded {total} scores for {self.genome_build}"  # type: ignore[attr-defined]

    def handle_lf_grid_row_selection(self, model: dict) -> None:
        """Merge grid checkbox changes into the durable ``selected_pgs_ids``.

        Overrides ``LazyFrameGridMixin.handle_lf_grid_row_selection`` so that
        ``lazyframe_grid()`` automatically calls this without needing an
        explicit ``on_row_selection_model_change`` kwarg.

        ``selected_pgs_ids`` is the authoritative selection store. The MUI
        selection model, by contrast, is scoped to the rows currently loaded
        under the active filter/sort and is keyed by the positional
        ``__row_id__`` that is renumbered every time the filter or sort
        changes. Treating that model as the source of truth wipes selections
        made under a previous filter (the user's complaint). Instead, each
        event only updates the pgs_ids within the **currently loaded scope**
        and leaves out-of-scope selections (e.g. from a different trait
        filter) intact.

        Handles MUI DataGrid v8 selection model:
        - {type: "include", ids: [...]} -- only listed rows are selected
        - {type: "exclude", ids: [...]} -- all rows EXCEPT listed are selected
        - {type: "exclude", ids: []} -- all rows selected (header checkbox)
        """
        selection_type: str = model.get("type", "include")
        raw_ids: list = model.get("ids", [])
        selected_row_ids: set[int] = {int(i) for i in raw_ids}

        # Header "select all" (exclude with no explicit ids) -> add the full
        # filtered set to the existing selection.
        if selection_type == "exclude" and not selected_row_ids:
            self.select_filtered_scores()
            return

        # Map the event onto the rows currently loaded in the grid. Only these
        # pgs_ids are "in scope" of this selection event; everything else stays
        # as-is so cross-filter selections survive.
        loaded_scope: set[str] = set()
        within_scope: list[str] = []
        for row in self.lf_grid_rows:  # type: ignore[attr-defined]
            pgs_id_raw = row.get("pgs_id")
            if not pgs_id_raw:
                continue
            pgs_id = str(pgs_id_raw)
            loaded_scope.add(pgs_id)
            row_id = row.get("__row_id__")
            in_set = (int(row_id) in selected_row_ids) if row_id is not None else False
            if (selection_type == "include" and in_set) or (
                selection_type == "exclude" and not in_set
            ):
                within_scope.append(pgs_id)

        preserved = [pid for pid in self.selected_pgs_ids if pid not in loaded_scope]
        merged = list(dict.fromkeys([*preserved, *within_scope]))
        self.selected_pgs_ids = merged
        self._sync_loaded_grid_selection(merged)
        self.status_message = f"Selected {len(merged)} scores"  # type: ignore[attr-defined]

    def handle_lf_grid_filter(self, filter_model: dict) -> Any:
        """Apply a server-side filter, then restore selection on the new view.

        The upstream handler reloads the grid rows (renumbering ``__row_id__``)
        but never touches the selection. Without re-syncing, MUI prunes the now
        non-existent selected row ids and the user's selection appears to
        vanish when the filter changes. ``selected_pgs_ids`` is durable, so we
        just re-project it onto the freshly loaded rows.
        """
        yield from LazyFrameGridMixin.handle_lf_grid_filter(self, filter_model)
        self._sync_loaded_grid_selection(self.selected_pgs_ids)

    def handle_lf_grid_sort(self, sort_model: list) -> Any:
        """Apply a server-side sort, then restore selection on the new view."""
        yield from LazyFrameGridMixin.handle_lf_grid_sort(self, sort_model)
        self._sync_loaded_grid_selection(self.selected_pgs_ids)

    def clear_lf_grid_filters(self) -> Any:
        """Clear all grid filters, then restore selection on the new view."""
        yield from LazyFrameGridMixin.clear_lf_grid_filters(self)
        self._sync_loaded_grid_selection(self.selected_pgs_ids)

    def handle_lf_grid_scroll_end(self, params: dict) -> Any:
        """Load the next scroll chunk, then mark any selected rows as checked."""
        yield from LazyFrameGridMixin.handle_lf_grid_scroll_end(self, params)
        self._sync_loaded_grid_selection(self.selected_pgs_ids)

    def select_filtered_scores(self) -> None:
        """Add PGS IDs matching the current grid filter to the selection.

        Additive (unions with the existing selection) so that selecting one
        trait's scores, changing the filter, and selecting another trait's
        scores accumulates instead of replacing the previous selection.
        """
        if self._compute_scores_lf is None:
            return
        lf = self._compute_scores_lf
        if self._lf_grid_filter and self._lf_grid_filter.get("items"):  # type: ignore[attr-defined]
            cache = _get_cache(self._lf_grid_cache_id) if self._lf_grid_cache_id else None  # type: ignore[attr-defined]
            schema = cache.schema if cache else None
            lf = apply_filter_model(lf, self._lf_grid_filter, schema)  # type: ignore[attr-defined]
        ids = lf.select("pgs_id").collect()["pgs_id"].to_list()
        merged = list(dict.fromkeys([*self.selected_pgs_ids, *(str(i) for i in ids)]))
        added = len(merged) - len(self.selected_pgs_ids)
        self.selected_pgs_ids = merged
        self._sync_loaded_grid_selection(merged)
        self.status_message = (  # type: ignore[attr-defined]
            f"Added {added} score(s) to selection ({len(merged)} total)"
        )

    def deselect_all_scores(self) -> None:
        """Clear all selected PGS IDs."""
        self.selected_pgs_ids = []
        self.lf_grid_row_selection_model = {"type": "include", "ids": []}  # type: ignore[assignment]
        self.status_message = ""  # type: ignore[attr-defined]

    def select_scores_by_query(self, query: str) -> None:
        """Select all PGS IDs matching a text query in cleaned PGS metadata."""
        term = query.strip()
        if not term:
            self.deselect_all_scores()
            return
        ids = (
            _catalog.search(term, genome_build=self.genome_build, include_harmonized=self.include_harmonized)  # type: ignore[attr-defined]
            .select("pgs_id")
            .unique()
            .sort("pgs_id")
            .collect()["pgs_id"]
            .to_list()
        )
        self.selected_pgs_ids = [str(pgs_id) for pgs_id in ids]
        self._sync_loaded_grid_selection(self.selected_pgs_ids)
        self.status_message = (
            f"Preselected {len(self.selected_pgs_ids)} score(s) matching '{term}' "
            f"for {self.genome_build}"  # type: ignore[attr-defined]
        )

    def filter_and_select_scores_by_query(self, query: str) -> Any:
        """Filter the score grid to a query and select every matching PGS ID."""
        term = query.strip()
        if not term:
            self.deselect_all_scores()
            return

        lf = _catalog.search(term, genome_build=self.genome_build, include_harmonized=self.include_harmonized)  # type: ignore[attr-defined]
        lf = _enrich_scores_for_grid(lf, _catalog)
        self._compute_scores_lf = lf
        self.compute_scores_loaded = True
        yield from self.set_lazyframe(  # type: ignore[attr-defined]
            lf,
            chunk_size=500,
            eager_value_options_row_limit=0,
            column_overrides=_compute_score_column_overrides(),
        )

        ids = (
            lf.select("pgs_id")
            .unique()
            .sort("pgs_id")
            .collect()["pgs_id"]
            .to_list()
        )
        self.selected_pgs_ids = [str(pgs_id) for pgs_id in ids]
        self._sync_loaded_grid_selection(self.selected_pgs_ids)
        self.status_message = (
            f"Filtered and preselected {len(self.selected_pgs_ids)} score(s) "
            f"matching '{term}' for {self.genome_build}"  # type: ignore[attr-defined]
        )

    def _sync_loaded_grid_selection(self, pgs_ids: list[str]) -> None:
        """Mark currently loaded grid rows as selected when their PGS IDs match."""
        selected = set(pgs_ids)
        loaded_row_ids = [
            int(row["__row_id__"])
            for row in self.lf_grid_rows  # type: ignore[attr-defined]
            if row.get("pgs_id") in selected and row.get("__row_id__") is not None
        ]
        self.lf_grid_row_selection_model = {"type": "include", "ids": loaded_row_ids}  # type: ignore[assignment]

    def compute_selected_prs(self) -> Any:
        """Compute PRS for all selected PGS IDs using available genotype data.

        Uses PRSCatalog for metadata lookup (no REST API calls) and for
        performance metrics from pre-downloaded bulk metadata.  Enrichment
        (quality, percentile, absolute risk, heritability) is delegated to
        ``enrich_prs_result()`` so it is shared with CLI and batch scripts.
        """
        if not self.selected_pgs_ids:
            self.status_message = "No PGS scores selected. Load and select scores above."  # type: ignore[attr-defined]
            return

        pre_genotypes = self._get_genotypes_lf()

        total = len(self.selected_pgs_ids)
        self.prs_computing = True
        self.prs_progress = 0
        self.prs_results = []
        self.trait_summary_rows = []
        self.trait_summary_visible = False
        self.low_match_warning = False
        self.status_message = f"Computing PRS for {total} score(s)..."  # type: ignore[attr-defined]
        yield

        cache = Path(self.cache_dir) / "scores"  # type: ignore[attr-defined]
        results: list[dict] = []
        any_low_match = False

        best_perf_df = _catalog.best_performance().collect()

        harmonized_lookup: dict[str, bool] = {}
        original_build_lookup: dict[str, str] = {}
        if self._compute_scores_lf is not None:
            try:
                harm_df = (
                    self._compute_scores_lf
                    .filter(pl.col("pgs_id").is_in(self.selected_pgs_ids))
                    .select("pgs_id", "is_harmonized", "genome_build")
                    .collect()
                )
                for row in harm_df.iter_rows(named=True):
                    harmonized_lookup[row["pgs_id"]] = bool(row["is_harmonized"])
                    original_build_lookup[row["pgs_id"]] = str(row["genome_build"])
            except Exception:
                pass

        for i, pgs_id in enumerate(self.selected_pgs_ids, start=1):
            self.prs_progress = round(i / total * 100)
            self.status_message = f"Computing {i}/{total}: {pgs_id}..."  # type: ignore[attr-defined]
            yield

            info = _catalog.score_info_row(pgs_id)
            trait = info["trait_reported"] if info else None

            vcf_path = self.prs_genotypes_path or ""
            if self.prs_engine == PRSEngine.DUCKDB.value:
                result = compute_prs_duckdb(
                    vcf_path=vcf_path,
                    scoring_file=pgs_id,
                    genome_build=self.genome_build,  # type: ignore[attr-defined]
                    cache_dir=cache,
                    pgs_id=pgs_id,
                    trait_reported=trait,
                    genotypes_parquet=vcf_path if vcf_path else None,
                    genotypes_lf=pre_genotypes,
                )
            else:
                result = compute_prs(
                    vcf_path=vcf_path,
                    scoring_file=pgs_id,
                    genome_build=self.genome_build,  # type: ignore[attr-defined]
                    cache_dir=cache,
                    pgs_id=pgs_id,
                    trait_reported=trait,
                    genotypes_lf=pre_genotypes,
                )

            enriched = enrich_prs_result(
                result,
                _catalog,
                best_perf_df,
                genome_build=self.genome_build,  # type: ignore[attr-defined]
                selected_ancestry=self.selected_ancestry,
                compute_all_populations=self.compute_all_populations,
                is_harmonized=harmonized_lookup.get(pgs_id, False),
            )

            row = _enriched_to_row_dict(enriched)
            row["original_genome_build"] = original_build_lookup.get(pgs_id, "")

            if result.match_rate < 0.1:
                any_low_match = True

            results.append(row)

        self.prs_results = results
        self._build_prs_results_grid()
        self.low_match_warning = any_low_match
        self.prs_computing = False
        self.prs_progress = 100
        self.status_message = f"Computed {total} PRS score(s)"  # type: ignore[attr-defined]
        if self.prs_view_mode == "grouped" and results:
            self.build_trait_summary()

    def download_prs_results_csv(self) -> Any:
        """Build a CSV from prs_results and trigger a browser download."""
        if not self.prs_results:
            return
        columns = [
            "pgs_id", "trait", "trait_efo_id", "is_harmonized", "score", "percentile", "absolute_risk",
            "synthetic_quality", "synthetic_quality_label", "quality_tier", "quality_tier_metric",
            "heritability", "heritability_detail",
            "auroc", "quality_label",
            "match_rate", "variants_matched", "variants_total",
            "effect_size", "classification", "ancestry",
            "n_individuals", "summary",
            "all_population_percentiles",
            "pct_AFR", "pct_AMR", "pct_EAS", "pct_EUR", "pct_SAS",
            "reference_status",
            "reference_source",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in self.prs_results:
            writer.writerow(row)
        return rx.download(data=buf.getvalue(), filename="prs_results.csv")

