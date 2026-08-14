"""Altair-based PRS visualizations: bell curves, percentile markers, multi-population comparisons.

Altair is a core dependency. For PNG/SVG export, install the ``viz`` extra:
``pip install just-prs[viz]`` (adds ``vl-convert-python``).
HTML and JSON export work out of the box.
"""

from __future__ import annotations

import logging
import math
import json
from pathlib import Path
from typing import Any, Literal

import altair as alt
import polars as pl

from just_prs.quality import rescale_quality_if_degenerate, resolve_quality_key
from just_prs.trait_summary import (
    NO_MAPPED_H2,
    best_of_label,
    format_heritability_risk_prompt,
    format_percentile_with_panel,
    is_high_quality_model,
    is_usable_model,
    parse_percentile,
    risk_basis_label,
    summarize_heritability,
    summarize_trait_rows,
)

logger = logging.getLogger(__name__)


SUPERPOP_COLORS: dict[str, str] = {
    "AFR": "#E69F00",
    "AMR": "#56B4E9",
    "EAS": "#009E73",
    "EUR": "#0072B2",
    "SAS": "#CC79A7",
}

SUPERPOP_LABELS: dict[str, str] = {
    "AFR": "African",
    "AMR": "American",
    "EAS": "East Asian",
    "EUR": "European",
    "SAS": "South Asian",
}

QUALITY_COLORS: dict[str, str] = {
    "high": "#2E7D32",
    "moderate": "#1565C0",
    "low": "#E65100",
    "very_low": "#B71C1C",
}

RISK_BANDS: list[dict] = [
    {"from": 0, "to": 10, "label": "very low", "color": "#d4edda"},
    {"from": 10, "to": 25, "label": "below average", "color": "#e8f5e9"},
    {"from": 25, "to": 75, "label": "average", "color": "#f5f5f5"},
    {"from": 75, "to": 90, "label": "above average", "color": "#fff3e0"},
    {"from": 90, "to": 100, "label": "high", "color": "#fce4ec"},
]


def _norm_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(min(max(float(p), 1e-9), 1.0 - 1e-9))


def _trait_visibility_params() -> tuple[list[alt.Parameter], str]:
    """Checkbox bindings that only show or hide chart marks."""
    show_high = alt.param(name="showHigh", value=True, bind=alt.binding_checkbox(name="High"))
    show_moderate = alt.param(
        name="showModerate", value=True, bind=alt.binding_checkbox(name="Moderate"),
    )
    show_low = alt.param(name="showLow", value=True, bind=alt.binding_checkbox(name="Low"))
    show_very_low = alt.param(
        name="showVeryLow", value=True, bind=alt.binding_checkbox(name="Very low"),
    )
    quality_filter = (
        "(datum.quality == 'high' && showHigh) || "
        "(datum.quality == 'moderate' && showModerate) || "
        "(datum.quality == 'low' && showLow) || "
        "(datum.quality == 'very_low' && showVeryLow)"
    )
    return [show_high, show_moderate, show_low, show_very_low], quality_filter


def _finite_float(value: object) -> float | None:
    """Return a finite float for numeric values, otherwise None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _parse_float(value: object) -> float | None:
    """Parse a finite float from numeric values or simple display strings."""
    number = _finite_float(value)
    if number is not None:
        return number
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _format_percent_like(value: object) -> str:
    """Format fractions or 0-100 percentages, preserving preformatted strings."""
    if isinstance(value, str):
        return value.strip() or "—"
    number = _finite_float(value)
    if number is None:
        return "—"
    percent = number if abs(number) > 1.0 else number * 100.0
    return f"{percent:.1f}%"


#: Explains what the "Quality" column/colours mean — they are relative on three
#: axes at once, so a "low" is not necessarily a bad model in absolute terms.
_QUALITY_NOTE_HTML = (
    '<div class="quality-note">'
    '<div class="qn-title">How to read “Quality”</div>'
    "Quality is <b>relative on three axes at once</b>, not an absolute grade:"
    "<ul>"
    "<li><b>The model itself</b> — its published predictive power (AUROC / effect "
    "size) and the size of the study it was trained on.</li>"
    "<li><b>How well it fits your sample</b> — the share of the model’s effect-weight "
    "your genotypes actually cover (C<sub>wt</sub>); cross-build (e.g. GRCh37→GRCh38) "
    "lifted models are demoted because liftover is lossy by design.</li>"
    "<li><b>How it ranks within this trait’s models</b> — when every model scores "
    "high for you the colour scale is stretched across this set, so the strongest "
    "stay green and the weakest turn orange.</li>"
    "</ul>"
    "So a <b>“low” here can still be a sound model</b> — it may simply trail the "
    "others for this trait or cover less of your genotype."
    "</div>"
)


# Absolute risk is asymptotic: it can approach but never reach 100% or 0%, so the
# extremes are rendered as bounds (">99.9%", "<0.01%") instead of the impossible
# "100.0%" / "0.0%".  Mirrors the source clamp in absolute_risk.py.
_ABS_RISK_DISPLAY_MAX = 0.999
_ABS_RISK_DISPLAY_MIN = 0.0001


def _format_absolute_risk(value: object) -> str:
    """Format an absolute-risk value, showing the extremes as bounds.

    Numeric values (fractions or 0–100 percents) clamp to ``>99.9%`` / ``<0.01%``
    at the asymptotic extremes.  Preformatted strings (e.g.
    ``"10.0% (pop. avg: 8.3%)"``) are passed through unchanged.
    """
    number = _parse_float(value)
    if number is None:
        return _format_percent_like(value)
    frac = number / 100.0 if number > 1.0 else number
    if frac >= _ABS_RISK_DISPLAY_MAX:
        return ">99.9%"
    if frac <= _ABS_RISK_DISPLAY_MIN:
        return "<0.01%"
    return f"{frac * 100.0:.1f}%"


def _bell_curve_data(
    mean: float, std: float, n_points: int = 200, z_range: float = 4.0
) -> list[dict]:
    rows = []
    lo, hi = mean - z_range * std, mean + z_range * std
    step = (hi - lo) / (n_points - 1)
    for i in range(n_points):
        x = lo + i * step
        y = _norm_pdf(x, mean, std)
        pct = _norm_cdf((x - mean) / std) * 100
        rows.append({"score": x, "density": y, "percentile": pct})
    return rows


def _std_bell_curve_data(n_points: int = 200, z_range: float = 4.0) -> list[dict]:
    rows = []
    lo, hi = -z_range, z_range
    step = (hi - lo) / (n_points - 1)
    for i in range(n_points):
        z = lo + i * step
        y = _norm_pdf(z, 0.0, 1.0)
        pct = _norm_cdf(z) * 100
        rows.append({"z_score": z, "density": y, "percentile": pct})
    return rows


def plot_prs_bell_curve(
    pgs_id: str,
    distributions_df: pl.DataFrame,
    user_score: float | None = None,
    ancestry: str = "EUR",
    title: str | None = None,
    width: int = 560,
    height: int = 220,
) -> alt.LayerChart:
    """Single PGS bell curve for one ancestry with optional user score marker."""
    row = distributions_df.filter(
        (pl.col("pgs_id") == pgs_id) & (pl.col("superpopulation") == ancestry)
    )
    if row.height == 0:
        raise ValueError(f"No distribution found for {pgs_id} / {ancestry}")

    r = row.row(0, named=True)
    mean, std = r["mean"], r["std"]
    trait = r.get("trait_reported") or r.get("name") or pgs_id

    curve_data = _bell_curve_data(mean, std)
    source = alt.Data(values=curve_data)

    area = (
        alt.Chart(source)
        .mark_area(opacity=0.35, color=SUPERPOP_COLORS.get(ancestry, "#0072B2"))
        .encode(
            x=alt.X("score:Q", title="PRS Score"),
            y=alt.Y("density:Q", title="Density"),
        )
    )

    line = (
        alt.Chart(source)
        .mark_line(color=SUPERPOP_COLORS.get(ancestry, "#0072B2"), strokeWidth=2)
        .encode(x="score:Q", y="density:Q")
    )

    layers: list = [area, line]

    pct_marks = []
    for p_val, p_label in [(5, "5th"), (25, "25th"), (50, "50th"), (75, "75th"), (95, "95th")]:
        p_key = f"p{p_val}" if p_val != 50 else "median"
        p_score = r.get(p_key)
        if p_score is not None:
            pct_marks.append({"score": p_score, "label": p_label, "density": _norm_pdf(p_score, mean, std)})

    if pct_marks:
        pct_source = alt.Data(values=pct_marks)
        ticks = (
            alt.Chart(pct_source)
            .mark_rule(strokeDash=[3, 3], opacity=0.4, color="gray")
            .encode(x="score:Q")
        )
        tick_labels = (
            alt.Chart(pct_source)
            .mark_text(dy=12, fontSize=9, color="gray")
            .encode(x="score:Q", text="label:N")
        )
        layers.extend([ticks, tick_labels])

    if user_score is not None:
        z_user = (user_score - mean) / std if std > 0 else 0.0
        if abs(z_user) <= 10:
            pct_user = _norm_cdf(z_user) * 100
            user_data = alt.Data(values=[{
                "score": user_score,
                "density": _norm_pdf(user_score, mean, std),
                "label": f"You: {pct_user:.1f}th",
            }])
            user_rule = (
                alt.Chart(user_data)
                .mark_rule(color="#D32F2F", strokeWidth=2.5)
                .encode(x="score:Q")
            )
            user_point = (
                alt.Chart(user_data)
                .mark_point(color="#D32F2F", size=60, filled=True)
                .encode(x="score:Q", y="density:Q")
            )
            user_label = (
                alt.Chart(user_data)
                .mark_text(dy=-12, fontSize=11, fontWeight="bold", color="#D32F2F")
                .encode(x="score:Q", y="density:Q", text="label:N")
            )
            layers.extend([user_rule, user_point, user_label])
        else:
            logger.warning(
                "User score for %s is %.0f SDs from reference mean "
                "(likely coverage mismatch) — skipping marker",
                pgs_id, abs(z_user),
            )

    chart_title = title or f"{pgs_id}: {trait} ({SUPERPOP_LABELS.get(ancestry, ancestry)})"

    return (
        alt.layer(*layers)
        .properties(width=width, height=height, title=chart_title)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )


def plot_prs_multi_ancestry(
    pgs_id: str,
    distributions_df: pl.DataFrame,
    user_score: float | None = None,
    ancestries: list[str] | None = None,
    title: str | None = None,
    width: int = 560,
    height: int = 220,
) -> alt.LayerChart:
    """Overlaid bell curves for multiple ancestries on a single chart."""
    sub = distributions_df.filter(pl.col("pgs_id") == pgs_id)
    if sub.height == 0:
        raise ValueError(f"No distributions found for {pgs_id}")

    available = sub["superpopulation"].unique().sort().to_list()
    pops = ancestries or available
    trait = sub.row(0, named=True).get("trait_reported") or sub.row(0, named=True).get("name") or pgs_id

    all_rows: list[dict] = []
    for pop in pops:
        pop_row = sub.filter(pl.col("superpopulation") == pop)
        if pop_row.height == 0:
            continue
        r = pop_row.row(0, named=True)
        m, s = r["mean"], r["std"]
        label = SUPERPOP_LABELS.get(pop, pop)
        for pt in _bell_curve_data(m, s):
            pt["population"] = label
            all_rows.append(pt)

    source = alt.Data(values=all_rows)

    pop_labels = [SUPERPOP_LABELS.get(p, p) for p in pops if p in available]
    pop_colors = [SUPERPOP_COLORS.get(p, "#999") for p in pops if p in available]
    color_scale = alt.Scale(domain=pop_labels, range=pop_colors)

    area = (
        alt.Chart(source)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("score:Q", title="PRS Score"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("population:N", scale=color_scale, title="Population"),
        )
    )

    line = (
        alt.Chart(source)
        .mark_line(strokeWidth=2)
        .encode(
            x="score:Q",
            y="density:Q",
            color=alt.Color("population:N", scale=color_scale, title="Population"),
        )
    )

    layers: list = [area, line]

    if user_score is not None:
        user_data = alt.Data(values=[{"score": user_score, "label": "Your score"}])
        user_rule = (
            alt.Chart(user_data)
            .mark_rule(color="#D32F2F", strokeWidth=2.5, strokeDash=[6, 3])
            .encode(x="score:Q")
        )
        user_label = (
            alt.Chart(user_data)
            .mark_text(dy=-8, fontSize=11, fontWeight="bold", color="#D32F2F", align="left", dx=5)
            .encode(x="score:Q", y=alt.value(0), text="label:N")
        )
        layers.extend([user_rule, user_label])

    chart_title = title or f"{pgs_id}: {trait} — All Populations"

    return (
        alt.layer(*layers)
        .properties(width=width, height=height, title=chart_title)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )


def plot_trait_scores(
    trait: str,
    distributions_df: pl.DataFrame,
    quality_df: pl.DataFrame | None = None,
    user_results: list[dict] | None = None,
    ancestry: str = "EUR",
    ancestries: list[str] | None = None,
    default_visible_ancestries: list[str] | None = None,
    max_scores: int = 25,
    title: str | None = None,
    width: int = 600,
    height: int = 250,
    show_table: bool = False,
    table_height: int | None = None,
    model_scope: str = "usable",
    percentile_source: str = "native",
) -> alt.LayerChart | alt.VConcatChart:
    """Trait-grouped visualization: reference bell curve + per-model user percentile scatter.

    Shows a standard-normal reference bell curve (the shared z-score space) with
    each model's user score plotted on it as a dot colored by model quality tier.
    Without user results, shows model quality comparison as a jittered dot strip.

    When ``ancestries`` is provided (e.g. ``["EUR", "AFR", "EAS", "AMR", "SAS"]``),
    overlays one color-coded bell curve per population instead of the single gray
    N(0,1) reference.  User dots are z-normalized against ``ancestry`` (the primary).
    Quality-tier checkboxes only change which dots are visible. Dashboard
    card numbers are computed separately from the quality-threshold and
    population dropdowns.

    Trait matching is case-insensitive and substring-based: ``"BMI"`` matches
    ``"Body mass index (BMI)"`` as well as ``"Body mass index"``.

    Args:
        trait: Trait name or substring (e.g. "BMI", "Body mass index").
        distributions_df: DataFrame with pgs_id, superpopulation, mean, std, etc.
        quality_df: Optional quality DataFrame with pgs_id, variants_total, variants_matched, match_rate.
        user_results: Optional list of per-model dicts with keys: pgs_id, score (and/or percentile, z_score).
        ancestry: Primary superpopulation code (used for z-normalization and dot placement).
        ancestries: List of superpopulation codes to overlay as separate bell curves.
            When None, shows a single gray N(0,1) reference.
        default_visible_ancestries: When set with multi-pop, only these populations
            are visible initially; others can be toggled on via the interactive legend.
            When None, all provided ancestries are visible by default.
        max_scores: Maximum number of PGS models to show (top by variant count).
        title: Chart title override.
        width: Chart width in pixels.
        height: Bell curve chart height in pixels.
        show_table: If True, append a model summary table below the bell curve.
        table_height: Height of the summary table in pixels (default: auto-sized from row count).
        model_scope: Quality-dropdown scope for the in-chart median line, so the
            "Median: Nth" annotation matches the dashboard cards exactly.
        percentile_source: Percentile resolution ("native"/"selected") for the median line.

    Returns:
        An Altair LayerChart (bell curve only) or VConcatChart (bell curve + table).
    """
    anc_mask = pl.col("superpopulation") == ancestry

    sub = distributions_df.filter(
        (pl.col("trait_reported") == trait) & anc_mask
    )
    if sub.height == 0:
        sub = distributions_df.filter(
            pl.col("trait_reported").str.to_lowercase().str.contains(trait.lower())
            & anc_mask
        )
    if sub.height == 0:
        raise ValueError(f"No distributions found for trait '{trait}' / {ancestry}")

    user_pgs_ids: set[str] = set()
    if user_results:
        user_pgs_ids = {r["pgs_id"] for r in user_results}
        user_in_dists = distributions_df.filter(
            pl.col("pgs_id").is_in(list(user_pgs_ids)) & anc_mask
        )
        if user_in_dists.height > 0:
            sub = pl.concat([sub, user_in_dists]).unique(subset=["pgs_id"])

    if quality_df is not None:
        q_cols = ["pgs_id"]
        for c in ("variants_total", "variants_matched", "match_rate"):
            if c in quality_df.columns:
                q_cols.append(c)
        sub = sub.join(quality_df.select(q_cols), on="pgs_id", how="left")
    elif "n_variants" in sub.columns:
        sub = sub.with_columns(pl.col("n_variants").alias("variants_total"))

    if "variants_total" not in sub.columns:
        sub = sub.with_columns(pl.lit(None).cast(pl.Int64).alias("variants_total"))

    if len(user_pgs_ids) > 0 and sub.height > max_scores:
        user_rows = sub.filter(pl.col("pgs_id").is_in(list(user_pgs_ids)))
        other_rows = (
            sub.filter(~pl.col("pgs_id").is_in(list(user_pgs_ids)))
            .with_columns(pl.col("variants_total").fill_null(0).alias("_sort_variants"))
            .sort("_sort_variants", descending=True)
            .head(max(0, max_scores - user_rows.height))
            .drop("_sort_variants")
        )
        sub = pl.concat([user_rows, other_rows]).unique(subset=["pgs_id"])
    else:
        sub = sub.with_columns(
            pl.col("variants_total").fill_null(0).alias("_sort_variants"),
        ).sort("_sort_variants", descending=True).head(max_scores)
        if "_sort_variants" in sub.columns:
            sub = sub.drop("_sort_variants")

    user_lookup: dict[str, dict] = {}
    if user_results:
        for ur in user_results:
            user_lookup[ur["pgs_id"]] = ur

    has_name_col = "name" in sub.columns

    model_meta: list[dict] = []
    for row_dict in sub.iter_rows(named=True):
        pgs_id = row_dict["pgs_id"]
        score_name = row_dict.get("name", "") if has_name_col else ""
        mean, std = row_dict["mean"], row_dict["std"]
        n_var = row_dict.get("variants_total") or row_dict.get("n_variants") or 0
        auroc = row_dict.get("auroc_estimate")
        n_var_label = f"{n_var:,}" if n_var else "?"

        z_user: float | None = None
        pct_user: float | None = None
        match_rate: float | None = None
        coverage: float | None = None
        user_quality_label: str | None = None
        user_synthetic_quality: float | None = None
        z_selected: float | None = None
        pct_selected: float | None = None
        mark_is_usable = False
        mark_is_high_quality = False
        if pgs_id in user_lookup:
            ur = user_lookup[pgs_id]
            user_quality_label = ur.get("quality_label") or None
            user_synthetic_quality = _parse_float(ur.get("synthetic_quality"))
            z_user = _parse_float(ur.get("z_score"))
            pct_user = _parse_float(ur.get("percentile"))
            match_rate = _parse_float(ur.get("match_rate"))
            coverage = _parse_float(ur.get("weight_mass_coverage"))
            if coverage is None:
                coverage = match_rate
            score_value = _parse_float(ur.get("score"))
            if z_user is None and score_value is not None and std > 0:
                z_candidate = (score_value - mean) / std
                if abs(z_candidate) <= 10:
                    z_user = z_candidate
                    pct_user = _norm_cdf(z_user) * 100
            if z_user is None and pct_user is not None:
                z_user = _norm_ppf(pct_user / 100.0)
            pct_selected = parse_percentile(ur.get(f"pct_{ancestry}"))
            if pct_selected is not None:
                z_selected = _norm_ppf(pct_selected / 100.0)
            mark_is_usable = is_usable_model(ur)
            mark_is_high_quality = is_high_quality_model(ur)
        if match_rate is None:
            match_rate = row_dict.get("match_rate")

        risk_ratio: float | None = None
        absolute_risk: float | None = None
        population_prevalence: float | None = None
        trait_reported: str = row_dict.get("trait_reported", "")
        if pgs_id in user_lookup:
            ur = user_lookup[pgs_id]
            risk_ratio = ur.get("risk_ratio")
            absolute_risk = ur.get("absolute_risk")
            population_prevalence = ur.get("population_prevalence")
            if not trait_reported:
                trait_reported = ur.get("trait_reported", "")

        reliable = True
        if pgs_id in user_lookup:
            reliable = user_lookup[pgs_id].get("reliable", True)

        # Prefer the genotype-aware quality label carried on the scored result;
        # fall back to variant/AUROC only for reference-context models that the
        # user never scored.  All classification lives in just_prs.quality.
        tier = resolve_quality_key(
            label=user_quality_label,
            n_var=int(n_var) if n_var else None,
            auroc=_parse_float(auroc),
            coverage=coverage,
        )

        model_meta.append({
            "pgs_id": pgs_id,
            "score_name": score_name or "",
            "trait_reported": trait_reported,
            "quality": tier,
            "synthetic_quality": user_synthetic_quality,
            "n_variants": n_var,
            "n_variants_label": n_var_label,
            "auroc": auroc,
            "auroc_label": f"AUROC={auroc:.2f}" if auroc is not None else "",
            "z_score": z_user,
            "z_selected": z_selected,
            "percentile": pct_user,
            "percentile_selected": pct_selected,
            "is_usable": mark_is_usable,
            "is_high_quality": mark_is_high_quality,
            "match_rate": match_rate,
            "coverage": coverage,
            "risk_ratio": risk_ratio,
            "absolute_risk": absolute_risk,
            "population_prevalence": population_prevalence,
            "mean": mean,
            "std": std,
            "reliable": reliable,
        })

    q_domain = list(QUALITY_COLORS.keys())
    q_range = list(QUALITY_COLORS.values())

    multi_pop = ancestries is not None and len(ancestries) > 1

    # Build background layers (reference curves + percentile lines)
    bg_layers: list = []

    if multi_pop:
        pop_curves: list[dict] = []
        pgs_ids_in_sub = sub["pgs_id"].to_list()

        ref_rows = sub.select("pgs_id", "mean", "std").to_dicts()
        ref_stats = {r["pgs_id"]: (r["mean"], r["std"]) for r in ref_rows if r["std"] and r["std"] > 0}

        for pop in ancestries:
            pop_sub = distributions_df.filter(
                pl.col("pgs_id").is_in(pgs_ids_in_sub)
                & (pl.col("superpopulation") == pop)
            )
            if pop_sub.height == 0:
                continue

            pop_rows = pop_sub.select("pgs_id", "mean", "std").to_dicts()
            shifts = []
            scales = []
            for pr in pop_rows:
                pid = pr["pgs_id"]
                if pid not in ref_stats or not pr["std"] or pr["std"] <= 0:
                    continue
                ref_mean, ref_std = ref_stats[pid]
                shifts.append((pr["mean"] - ref_mean) / ref_std)
                scales.append(pr["std"] / ref_std)

            if not shifts:
                continue

            avg_shift = sum(shifts) / len(shifts)
            avg_scale = sum(scales) / len(scales)

            pop_label = SUPERPOP_LABELS.get(pop, pop)
            for pt in _std_bell_curve_data(n_points=150):
                z_ref = pt["z_score"]
                z_pop = avg_shift + z_ref * avg_scale
                density = _norm_pdf(z_pop, avg_shift, avg_scale)
                pop_curves.append({
                    "z_score": z_pop,
                    "density": density,
                    "population": pop_label,
                })

        if pop_curves:
            pop_source = alt.Data(values=pop_curves)
            available_pops = {r["population"] for r in pop_curves}
            pop_domain = [SUPERPOP_LABELS.get(p, p) for p in ancestries if SUPERPOP_LABELS.get(p, p) in available_pops]
            pop_colors = [SUPERPOP_COLORS.get(p, "#666") for p in ancestries if SUPERPOP_LABELS.get(p, p) in available_pops]

            if default_visible_ancestries is not None:
                visible_labels = [
                    SUPERPOP_LABELS.get(a, a)
                    for a in default_visible_ancestries
                    if SUPERPOP_LABELS.get(a, a) in available_pops
                ]
                pop_select = alt.selection_point(
                    fields=["population"],
                    bind="legend",
                    toggle="true",
                    value=[{"population": lab} for lab in visible_labels] if visible_labels else [],
                )
            else:
                pop_select = alt.selection_point(
                    fields=["population"],
                    bind="legend",
                    toggle="true",
                )

            pop_color = alt.Color(
                "population:N",
                scale=alt.Scale(domain=pop_domain, range=pop_colors),
                title="Population",
            )

            bg_layers.append(
                alt.Chart(pop_source)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x=alt.X("z_score:Q", title="Z-Score (each model independently normalized)"),
                    y=alt.Y("density:Q", title="Density"),
                    color=pop_color,
                    opacity=alt.condition(pop_select, alt.value(0.85), alt.value(0)),
                )
                .add_params(pop_select)
            )
    else:
        std_curve = _std_bell_curve_data(n_points=150)
        curve_source = alt.Data(values=std_curve)

        bg_layers.append(
            alt.Chart(curve_source)
            .mark_area(opacity=0.12, color="#455A64")
            .encode(
                x=alt.X("z_score:Q", title="Z-Score (each model independently normalized)"),
                y=alt.Y("density:Q", title="Density"),
            )
        )
        bg_layers.append(
            alt.Chart(curve_source)
            .mark_line(color="#455A64", strokeWidth=1.5, opacity=0.5)
            .encode(x="z_score:Q", y="density:Q")
        )

    ref_zs = [(-1.645, "5th"), (-0.674, "25th"), (0, "50th"), (0.674, "75th"), (1.645, "95th")]
    ref_data = [{"z_score": z, "label": lab} for z, lab in ref_zs]
    bg_layers.append(
        alt.Chart(alt.Data(values=ref_data))
        .mark_rule(strokeDash=[4, 4], opacity=0.3, color="#666")
        .encode(x="z_score:Q")
    )
    bg_layers.append(
        alt.Chart(alt.Data(values=ref_data))
        .mark_text(dy=16, fontSize=14, color="#555", fontWeight="bold")
        .encode(x="z_score:Q", text="label:N")
    )

    # Build foreground layers (user dots + labels + median)
    fg_layers: list = []

    user_marks = [m for m in model_meta if m["z_score"] is not None]
    has_user = len(user_marks) > 0
    quality_params, quality_filter = _trait_visibility_params()
    model_select = alt.selection_point(name="modelSelect", fields=["pgs_id"], toggle="true")
    z_display_expr = "datum.z_native_display"
    density_expr = (
        f"exp(-0.5 * datum.z_display * datum.z_display) / {math.sqrt(2 * math.pi)} "
        "+ datum.jitter"
    )
    visibility_params = [*quality_params, model_select]

    # When every scored model lands in the top tiers the dot heatmap is uniformly
    # green; re-colour the user's models relative to their own cohort (ranked on
    # synthetic quality) so the gradient is visible.  No-op otherwise.  Keep the
    # report table (trait_report_html) in sync — both call the same helper.
    if has_user:
        _rescaled = rescale_quality_if_degenerate(
            [m["quality"] for m in user_marks],
            [
                (m["synthetic_quality"] / 100.0)
                if m.get("synthetic_quality") is not None
                else None
                for m in user_marks
            ],
            coverages=[m.get("coverage") for m in user_marks],
        )
        for _m, _k in zip(user_marks, _rescaled):
            if _k is not None:
                _m["quality"] = _k

    if has_user:
        import random
        rng = random.Random(42)
        for um in user_marks:
            z_c = max(-3.4, min(3.4, um["z_score"]))
            um["z_native_display"] = z_c
            um["z_display"] = z_c
            um["jitter"] = rng.uniform(-0.02, 0.04)
            um["density"] = _norm_pdf(z_c) + um["jitter"]

        for um in user_marks:
            um["short_id"] = um["pgs_id"].replace("PGS00", "").replace("PGS0", "").replace("PGS", "")
            um["reliability"] = "reliable" if um.get("reliable", True) else "low coverage"

        reliable_marks = [m for m in user_marks if m.get("reliable", True)]
        unreliable_marks = [m for m in user_marks if not m.get("reliable", True)]

        sorted_marks = sorted(user_marks, key=lambda m: m["z_native_display"])
        for i, um in enumerate(sorted_marks):
            um["label_dy"] = -14 if i % 2 == 0 else 18

        common_tooltip = [
            alt.Tooltip("pgs_id:N", title="PGS ID"),
            alt.Tooltip("score_name:N", title="Name"),
            alt.Tooltip("quality:N"),
            alt.Tooltip("n_variants_label:N", title="Variants"),
            alt.Tooltip("auroc_label:N", title="AUROC"),
            alt.Tooltip("percentile:Q", title="Percentile", format=".1f"),
            alt.Tooltip("z_score:Q", title="Z-score", format=".2f"),
            alt.Tooltip("reliability:N", title="Reliability"),
        ]
        quality_color = alt.Color(
            "quality:N",
            scale=alt.Scale(domain=q_domain, range=q_range),
            title="Model quality",
            legend=None,
        )

        def _scoped_chart(values: list[dict]) -> alt.Chart:
            return (
                alt.Chart(alt.Data(values=values))
                .transform_filter(quality_filter)
                .transform_calculate(z_display=z_display_expr, density=density_expr)
            )

        if reliable_marks:
            reliable_base = _scoped_chart(reliable_marks)
            fg_layers.append(
                reliable_base.mark_rule(strokeDash=[3, 3], strokeWidth=1).encode(
                    x="z_display:Q",
                    color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), legend=None),
                    opacity=alt.condition(model_select, alt.value(0.35), alt.value(0.08)),
                )
            )
            fg_layers.append(
                reliable_base.mark_point(size=55, filled=True, strokeWidth=1, stroke="white").encode(
                    x="z_display:Q",
                    y="density:Q",
                    color=quality_color,
                    tooltip=common_tooltip,
                    opacity=alt.condition(model_select, alt.value(0.9), alt.value(0.15)),
                    size=alt.condition(model_select, alt.value(70), alt.value(30)),
                )
            )
            for um in reliable_marks:
                fg_layers.append(
                    _scoped_chart([um]).mark_text(fontSize=10, dy=um["label_dy"], fontWeight="bold").encode(
                        x="z_display:Q",
                        y="density:Q",
                        text="short_id:N",
                        color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), legend=None),
                        opacity=alt.condition(model_select, alt.value(0.75), alt.value(0.0)),
                    )
                )

        if unreliable_marks:
            unreliable_base = _scoped_chart(unreliable_marks)
            fg_layers.append(
                unreliable_base.mark_rule(strokeDash=[2, 4], strokeWidth=0.5).encode(
                    x="z_display:Q",
                    color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), legend=None),
                    opacity=alt.condition(model_select, alt.value(0.2), alt.value(0.05)),
                )
            )
            fg_layers.append(
                unreliable_base.mark_point(size=40, filled=False, strokeWidth=1.5).encode(
                    x="z_display:Q",
                    y="density:Q",
                    color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), legend=None),
                    tooltip=common_tooltip,
                    opacity=alt.condition(model_select, alt.value(0.5), alt.value(0.1)),
                    size=alt.condition(model_select, alt.value(50), alt.value(25)),
                )
            )
            for um in unreliable_marks:
                fg_layers.append(
                    _scoped_chart([um]).mark_text(fontSize=9, dy=um["label_dy"], fontStyle="italic").encode(
                        x="z_display:Q",
                        y="density:Q",
                        text="short_id:N",
                        color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), legend=None),
                        opacity=alt.condition(model_select, alt.value(0.4), alt.value(0.0)),
                    )
                )

        # The median annotation is a dashboard number: it must equal the scoped
        # median-percentile card (Quality/Population dropdowns), never a private
        # recomputation over all dots.  The old ``all_z[len(all_z)//2]`` picked
        # the upper-middle element (94th for a 6-model group whose true median
        # was 83rd) and ignored the dropdowns entirely.
        median_stats = summarize_trait_rows(
            list(user_results or []),
            model_scope=model_scope,
            selected_ancestry=ancestry,
            percentile_source=percentile_source,
        )
        mz: float | None = None
        median_label = ""
        if median_stats.median_pct is not None:
            mz = _norm_ppf(median_stats.median_pct / 100.0)
            median_label = f"Median: {median_stats.median_pct:.0f}th"
        else:
            all_z = sorted(m["z_score"] for m in user_marks if m.get("z_score") is not None)
            if all_z:
                mid = len(all_z) // 2
                mz = all_z[mid] if len(all_z) % 2 else (all_z[mid - 1] + all_z[mid]) / 2.0
                median_label = f"Median: {_norm_cdf(mz) * 100:.0f}th"
        if mz is not None:
            fg_layers.append(
                alt.Chart(alt.Data(values=[{"z_score": mz}]))
                .mark_rule(color="#D32F2F", strokeWidth=2)
                .encode(x="z_score:Q")
            )
            fg_layers.append(
                alt.Chart(alt.Data(values=[{
                    "z_score": mz,
                    "density": 0.56,
                    "label": median_label,
                }]))
                .mark_text(fontSize=14, fontWeight="bold", color="#D32F2F", align="left", dx=5)
                .encode(x="z_score:Q", y="density:Q", text="label:N")
            )

    n_total = len(model_meta)
    n_high = sum(1 for m in model_meta if m["quality"] == "high")
    n_mod = sum(1 for m in model_meta if m["quality"] == "moderate")
    chart_title = title or f"{trait} — {n_total} models ({n_high} high, {n_mod} moderate)"

    bg_chart = alt.layer(*bg_layers)
    fg_chart = alt.layer(*fg_layers) if fg_layers else None

    if fg_chart is not None and multi_pop:
        bell = (
            alt.layer(bg_chart, fg_chart)
            .resolve_scale(color="independent")
            .properties(width=width, height=height, title=chart_title)
        )
    elif fg_chart is not None:
        bell = (
            alt.layer(*bg_layers, *fg_layers)
            .properties(width=width, height=height, title=chart_title)
        )
    else:
        bell = (
            alt.layer(*bg_layers)
            .properties(width=width, height=height, title=chart_title)
        )

    if has_user:
        bell = bell.add_params(*visibility_params)

    if not show_table:
        return bell.configure_axis(
            grid=False, labelFontSize=14, titleFontSize=15,
        ).configure_view(strokeWidth=0).configure_legend(
            titleFontSize=16, labelFontSize=15, symbolSize=120, symbolStrokeWidth=2,
        )

    # --- Model summary table (HTML-style using Altair text marks) ---
    table_rows = sorted(
        [m for m in model_meta if m["z_score"] is not None],
        key=lambda m: m.get("percentile") or 0,
        reverse=True,
    )
    if not table_rows:
        table_rows = sorted(model_meta, key=lambda m: m.get("n_variants") or 0, reverse=True)

    has_risk_data = any(tr.get("risk_ratio") is not None for tr in table_rows)
    distinct_traits = {tr.get("trait_reported", "") for tr in table_rows} - {""}
    has_multi_traits = len(distinct_traits) > 1

    max_trait_len = 30
    for idx, tr in enumerate(table_rows):
        tr["pctl_label"] = f'{tr["percentile"]:.1f}%' if tr.get("percentile") is not None else "—"
        mr = tr.get("match_rate")
        tr["match_label"] = _format_percent_like(mr)
        tr["short_id"] = tr.get("short_id") or tr["pgs_id"]
        tr["row_idx"] = idx
        rr = tr.get("risk_ratio")
        rr_value = _finite_float(rr)
        tr["risk_label"] = f"{rr_value:.2f}x" if rr_value is not None else "—"
        ar = tr.get("absolute_risk")
        prev = tr.get("population_prevalence")
        if ar is not None:
            ar_str = _format_percent_like(ar)
            if prev is not None:
                ar_str += f" (avg {_format_percent_like(prev)})"
            tr["abs_risk_label"] = ar_str
        else:
            tr["abs_risk_label"] = "—"
        raw_trait = tr.get("trait_reported", "")
        tr["trait_label"] = (raw_trait[:max_trait_len - 1] + "…") if len(raw_trait) > max_trait_len else raw_trait

    row_height = 24
    n_rows = len(table_rows)
    t_height = table_height or max(60, n_rows * row_height + 30)
    table_source = alt.Data(values=table_rows)

    label_font = max(11, min(14, width // 60))
    y_enc = alt.Y(
        "row_idx:O",
        title=None,
        axis=alt.Axis(
            labels=False,
            ticks=False,
            domain=False,
        ),
        sort=list(range(n_rows)),
    )

    if has_multi_traits and has_risk_data:
        col_positions = {
            "id": int(width * 0.01),
            "trait": int(width * 0.10),
            "pctl": int(width * 0.27),
            "risk": int(width * 0.36),
            "abs_risk": int(width * 0.49),
            "variants": int(width * 0.65),
            "match": int(width * 0.76),
            "quality": int(width * 0.86),
        }
    elif has_multi_traits:
        col_positions = {
            "id": int(width * 0.01),
            "trait": int(width * 0.10),
            "pctl": int(width * 0.30),
            "variants": int(width * 0.44),
            "match": int(width * 0.57),
            "quality": int(width * 0.68),
        }
    elif has_risk_data:
        col_positions = {
            "id": int(width * 0.01),
            "pctl": int(width * 0.15),
            "risk": int(width * 0.25),
            "abs_risk": int(width * 0.40),
            "variants": int(width * 0.58),
            "match": int(width * 0.71),
            "quality": int(width * 0.82),
        }
    else:
        col_positions = {
            "id": int(width * 0.02),
            "pctl": int(width * 0.22),
            "variants": int(width * 0.36),
            "match": int(width * 0.50),
            "quality": int(width * 0.62),
        }

    base = alt.Chart(table_source).encode(y=y_enc)

    col_id = base.mark_text(align="left", fontSize=label_font, color="#333").encode(
        x=alt.value(col_positions["id"]),
        text="pgs_id:N",
    )
    col_pctl = base.mark_text(align="center", fontSize=label_font).encode(
        x=alt.value(col_positions["pctl"]),
        text="pctl_label:N",
        color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), legend=None),
    )
    col_variants = base.mark_text(align="center", fontSize=label_font, color="#555").encode(
        x=alt.value(col_positions["variants"]),
        text="n_variants_label:N",
    )
    col_match = base.mark_text(align="center", fontSize=label_font, color="#555").encode(
        x=alt.value(col_positions["match"]),
        text="match_label:N",
    )
    col_quality = base.mark_point(size=40, filled=True).encode(
        x=alt.value(col_positions["quality"]),
        color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), legend=None),
    )

    table_layers = [col_id, col_pctl, col_variants, col_match, col_quality]

    header_vals: list[dict] = [
        {"x": col_positions["id"], "label": "PGS ID", "align": "left"},
    ]

    if has_multi_traits:
        col_trait = base.mark_text(align="left", fontSize=label_font, color="#555").encode(
            x=alt.value(col_positions["trait"]),
            text="trait_label:N",
        )
        table_layers.insert(1, col_trait)
        header_vals.append({"x": col_positions["trait"], "label": "Trait", "align": "left"})

    header_vals.append({"x": col_positions["pctl"], "label": "Percentile", "align": "center"})

    if has_risk_data:
        col_risk = base.mark_text(align="center", fontSize=label_font, fontWeight="bold").encode(
            x=alt.value(col_positions["risk"]),
            text="risk_label:N",
            color=alt.condition(
                alt.datum.risk_ratio >= 1.0,
                alt.value("#C62828"),
                alt.value("#2E7D32"),
            ),
        )
        col_abs_risk = base.mark_text(align="center", fontSize=label_font, color="#555").encode(
            x=alt.value(col_positions["abs_risk"]),
            text="abs_risk_label:N",
        )
        risk_insert = len(table_layers) - 3
        table_layers.insert(risk_insert, col_risk)
        table_layers.insert(risk_insert + 1, col_abs_risk)
        header_vals.append({"x": col_positions["risk"], "label": "Risk×", "align": "center"})
        header_vals.append({"x": col_positions["abs_risk"], "label": "Abs Risk", "align": "center"})

    header_vals.extend([
        {"x": col_positions["variants"], "label": "Variants", "align": "center"},
        {"x": col_positions["match"], "label": "Match%", "align": "center"},
        {"x": col_positions["quality"], "label": "Quality", "align": "center"},
    ])

    header = (
        alt.Chart(alt.Data(values=header_vals))
        .mark_text(fontSize=label_font, fontWeight="bold", color="#333")
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, width])),
            y=alt.value(-10),
            text="label:N",
        )
    )
    table_layers.append(header)

    table = (
        alt.layer(*table_layers)
        .properties(width=width, height=t_height)
    )

    return (
        alt.vconcat(bell, table, spacing=10)
        .configure_axis(grid=False, labelFontSize=14, titleFontSize=15)
        .configure_view(strokeWidth=0)
        .configure_legend(titleFontSize=16, labelFontSize=15, symbolSize=120, symbolStrokeWidth=2)
    )


def plot_prs_percentile_strip(
    results: list[dict],
    title: str = "PRS Percentiles",
    width: int = 560,
    height: int | None = None,
) -> alt.Chart:
    """Horizontal strip chart showing percentile positions for multiple PGS scores."""
    for r in results:
        r.setdefault("trait", r["pgs_id"])
        r.setdefault("label", f'{r["pgs_id"]}: {r.get("trait", "")}')
        r.setdefault("quality_label", "")

    if height is None:
        height = max(100, len(results) * 30 + 40)

    bands_data = [
        {"x": b["from"], "x2": b["to"], "band_label": b["label"], "fill": b["color"]}
        for b in RISK_BANDS
    ]

    bands = (
        alt.Chart(alt.Data(values=bands_data))
        .mark_rect(opacity=0.6)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 100]), title="Percentile"),
            x2="x2:Q",
            color=alt.Color(
                "fill:N",
                scale=alt.Scale(
                    domain=[b["color"] for b in RISK_BANDS],
                    range=[b["color"] for b in RISK_BANDS],
                ),
                legend=None,
            ),
        )
    )

    source = alt.Data(values=results)

    points = (
        alt.Chart(source)
        .mark_point(size=100, filled=True)
        .encode(
            x=alt.X("percentile:Q", scale=alt.Scale(domain=[0, 100]), title="Percentile"),
            y=alt.Y("label:N", title=None, sort=None),
            color=alt.value("#D32F2F"),
            tooltip=[
                alt.Tooltip("pgs_id:N", title="PGS ID"),
                alt.Tooltip("trait:N", title="Trait"),
                alt.Tooltip("percentile:Q", title="Percentile", format=".1f"),
                alt.Tooltip("quality_label:N", title="Quality"),
            ],
        )
    )

    labels = (
        alt.Chart(source)
        .mark_text(dx=10, fontSize=10, align="left")
        .encode(
            x="percentile:Q",
            y=alt.Y("label:N", sort=None),
            text=alt.Text("percentile:Q", format=".1f"),
        )
    )

    return (
        alt.layer(bands, points, labels)
        .properties(width=width, height=height, title=title)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )


def save_chart(
    chart: alt.Chart | alt.LayerChart | alt.VConcatChart,
    path: Path,
    scale_factor: float = 2.0,
) -> Path:
    """Save an Altair chart to SVG, PNG, HTML, or JSON based on file extension.

    HTML and JSON work with altair alone.  SVG and PNG require
    ``vl-convert-python`` (install via ``pip install just-prs[viz]``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix in (".svg", ".png") and not _has_vl_convert():
        raise ImportError(
            f"Exporting to {suffix} requires vl-convert-python. "
            "Install with: pip install just-prs[viz]\n"
            "Alternatively, export as .html or .json (no extra dependency needed)."
        )

    if suffix == ".html":
        chart.save(str(path), format="html")
    elif suffix == ".svg":
        chart.save(str(path), format="svg")
    elif suffix == ".png":
        chart.save(str(path), format="png", scale_factor=scale_factor)
    elif suffix == ".json":
        chart.save(str(path), format="json")
    else:
        chart.save(str(path), format="html")

    return path


def _has_vl_convert() -> bool:
    try:
        import vl_convert  # noqa: F401
        return True
    except ImportError:
        return False


_AI_ASSISTANTS = [
    {"name": "Claude", "url": "https://claude.ai/new?q=", "color": "#DA7756", "limit": 6000},
    {"name": "ChatGPT", "url": "https://chatgpt.com/?q=", "color": "#10A37F", "limit": 4000},
    {"name": "Perplexity", "url": "https://www.perplexity.ai/search?q=", "color": "#21808D", "limit": 4000},
]

_REFERENCE_AUDIT_ISSUE_LABELS: dict[str, str] = {
    "quality_report_missing": "Reference quality report missing",
    "quality_match_metadata_missing": (
        "Reference match metadata missing; old reference run cannot prove variant coverage"
    ),
    "quality_low_match_rate": "Low reference-panel match rate",
    "quality_sample_count_mismatch": "Reference sample count mismatch",
    "quality_score_mean_mismatch": "Reference mean is stale vs raw scores",
    "quality_score_std_mismatch": "Reference standard deviation is stale vs raw scores",
    "missing_distribution_columns": "Reference distribution schema is incomplete",
    "missing_quality_columns": "Reference quality report schema is incomplete",
}

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
    "The score also factors cohort size (log-scaled), model coverage, and a "
    "harmonized-score penalty if coordinates were lifted over. "
    "Quality labels: High (>=70), Normal (>=50), Moderate (>=30), Low (<30)."
)

_FORMAT_ADDENDUM = (
    "After the main section, you MAY add a clearly separated section "
    "(use a horizontal rule ---) with additional commentary: caveats, "
    "ancestry considerations, trait-specific biology, or links to further reading. "
    "This optional section has no word limit but should earn its length — "
    "only include it if you have genuinely useful additional context."
)

_FORMAT_SINGLE_COMPACT = (
    "Reply in under 150 words. Start with ONE bold sentence verdict, "
    "then 2-3 bullet points: risk in real terms, h²/heredity if provided, "
    "confidence/actionability. Do NOT assume health — this may be a "
    "behavioral/physical/cognitive trait. PRS is genetic predisposition, "
    "not a measurement. Be honest about limitations."
)

_FORMAT_SINGLE_FULL = (
    "Structure your response EXACTLY as follows (keep this part under 220 words):\n"
    "1. **Verdict** — one bold sentence (e.g. 'Your genetic score for [trait] is "
    "moderately elevated (74th percentile) with moderate confidence.')\n"
    "2. **Risk in real terms** — 2-3 bullets with percentile, absolute risk vs "
    "population average when provided, and h²/heredity context when provided.\n"
    "3. **Confidence** — 1-2 sentences: coverage, quality tier, audit/build warnings. "
    "Mention model agreement only as supporting evidence, not as the main story.\n"
    "4. **What to do** — 1-2 sentences: only if actionable (screening for health "
    "traits). For non-health traits, say no action is needed and why.\n"
    "Citizen scientist audience — clarity and honesty over length.\n\n"
    + _FORMAT_ADDENDUM
)

_FORMAT_TRAIT_COMPACT = (
    "Reply in under 150 words. Start with ONE bold sentence verdict about the "
    "combined result, then 3 bullet points: risk in real terms, heritability/h² "
    "if provided, and confidence/actionability. Do NOT make model disagreement "
    "the main answer. Do NOT assume health — this may be a non-health trait. "
    "PRS is genetic predisposition, not a measurement. Be honest about limitations."
)

_FORMAT_TRAIT_FULL = (
    "Structure your response EXACTLY as follows (keep this part under 240 words):\n"
    "1. **Verdict** — one bold sentence (e.g. 'Five models consistently place your "
    "[trait] score in the top 10% with moderate confidence.')\n"
    "2. **Risk in real terms** — 2-4 bullets with the best usable percentile, "
    "absolute risk vs population average when provided, and h²/heredity context "
    "when provided. Do not bury these behind agreement commentary.\n"
    "3. **Confidence** — 1-2 sentences: use model coverage, quality, h² source, "
    "risk-method agreement, and ancestry. Mention model spread/outliers only as "
    "supporting evidence.\n"
    "4. **Context & actions** — 1-2 sentences: what this trait IS (health, "
    "behavioral, physical, cognitive — do NOT assume health), and whether any "
    "action makes sense. For non-health traits, say no action is needed.\n"
    "Citizen scientist audience — clarity and honesty over length.\n\n"
    + _FORMAT_ADDENDUM
)

_TRAIT_TYPE_GUIDANCE = (
    "Trait-type handling: first decide whether this is a disease/medical outcome "
    "or a non-disease trait. For disease traits, discuss absolute risk and risk "
    "elevation vs population average; when h²-liability estimates are provided, "
    "use them as risk estimates too and say whether the h²-based risk agrees with "
    "the other methods. If prevalence or absolute-risk inputs are missing, do not "
    "invent them. For non-disease traits (for example longevity, intelligence, "
    "height, behavior, or preferences), do NOT use disease language such as "
    "'lifetime risk', 'screening', or 'diagnosis'. Interpret the percentile as "
    "genetic predisposition/tendency relative to the reference population, explain "
    "h² as population-level heredity context, and use trait-appropriate reasoning: "
    "for sport/performance/body/behavior traits, explain likely direction, practical "
    "relevance, trainability/environment, and limitations without pretending the PRS "
    "determines ability. Do not invent a medical action plan. If there is no clear "
    "action, say that plainly."
)


def _quality_tier(label: str | None) -> int:
    """Map quality label to a numeric tier for sorting (higher = better)."""
    return {"High": 4, "Normal": 3, "Moderate": 2, "Low": 1}.get(label or "", 0)


def _clean_metadata_text(value: Any) -> str:
    """Return a display-safe string for optional metadata values."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def _shorten_text(text: str, limit: int = 90) -> str:
    """Trim long labels for compact AI prompts."""
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _reference_audit_issue_text(issue_codes: str | list[str]) -> str:
    """Return citizen-readable text for reference audit issue codes."""
    if isinstance(issue_codes, str):
        raw_codes = [code.strip() for code in issue_codes.split(",") if code.strip()]
    else:
        raw_codes = [str(code).strip() for code in issue_codes if str(code).strip()]
    labels = [_REFERENCE_AUDIT_ISSUE_LABELS.get(code, code.replace("_", " ")) for code in raw_codes]
    return "; ".join(dict.fromkeys(labels))


def _publication_display_label(row: dict[str, Any]) -> str:
    """Build a compact source-study label from joined publication metadata."""
    citation = _clean_metadata_text(row.get("citation"))
    if citation and citation != "()":
        return citation
    title = _clean_metadata_text(row.get("publication_title"))
    if title:
        return _shorten_text(title)
    pgp_id = _clean_metadata_text(row.get("pgp_id"))
    if pgp_id:
        return pgp_id
    pmid = _clean_metadata_text(row.get("publication_pmid") or row.get("pmid"))
    if pmid:
        return f"PMID {pmid}"
    return "Publication metadata unavailable"


def _publication_link_dicts(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return source-publication link objects for AI prompts."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        label = _publication_display_label(row)
        pgp_id = _clean_metadata_text(row.get("pgp_id"))
        pmid = _clean_metadata_text(row.get("publication_pmid") or row.get("pmid"))
        doi = _clean_metadata_text(row.get("publication_doi") or row.get("doi"))

        if pgp_id:
            key = f"pgp:{pgp_id}"
            if key not in seen:
                items.append({
                    "label": label,
                    "url": f"https://www.pgscatalog.org/publication/{pgp_id}/",
                })
                seen.add(key)
            continue

        if pmid:
            key = f"pmid:{pmid}"
            if key not in seen:
                items.append({
                    "label": f"{label} (PubMed)",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
                seen.add(key)
            continue

        if doi:
            key = f"doi:{doi}"
            if key not in seen:
                items.append({
                    "label": f"{label} (DOI)",
                    "url": f"https://doi.org/{doi}",
                })
                seen.add(key)

    return items


def _source_link_prompt_lines(
    rows: list[dict[str, Any]],
    pgs_ids: list[str],
    publication_links_json: str = "",
) -> list[str]:
    """Build source-link prompt lines for external LLM assistants."""
    lines = ["Source links to use if your environment can open web pages:"]
    for pgs_id in pgs_ids[:8]:
        clean_id = _clean_metadata_text(pgs_id)
        if clean_id:
            lines.append(f"- PGS Catalog score {clean_id}: https://www.pgscatalog.org/score/{clean_id}/")

    pub_links = _publication_link_dicts(rows) if rows else []
    if not pub_links and publication_links_json:
        try:
            parsed = json.loads(publication_links_json)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            pub_links = [
                item
                for item in parsed
                if isinstance(item, dict)
                and _clean_metadata_text(item.get("label"))
                and _clean_metadata_text(item.get("url"))
            ]

    for item in pub_links[:8]:
        label = _clean_metadata_text(item.get("label"))
        url = _clean_metadata_text(item.get("url"))
        if label and url and "Publication metadata unavailable" not in label:
            lines.append(f"- Source publication {label}: {url}")

    lines.append(
        "Please try to use the PGS Catalog page and source publication links for extra context. "
        "If you cannot access a link, say that clearly instead of inventing details."
    )
    return lines


def _heritability_prompt_summary(
    user_results: list[dict],
    selected_ancestry: str = "EUR",
    restrict_to_selected: bool = False,
) -> str:
    """Return a compact, de-duplicated h2 summary for AI prompts."""
    text, _, _ = summarize_heritability(
        user_results,
        selected_ancestry=selected_ancestry,
        restrict_to_selected=restrict_to_selected,
    )
    if text in {NO_MAPPED_H2, ""}:
        return ""
    return text


def _heritability_risk_prompt_summary(
    user_results: list[dict[str, Any]],
    selected_ancestry: str = "EUR",
    restrict_to_selected: bool = False,
) -> str:
    """Return h2-liability risk estimates for prompts when available."""
    _, _, metrics = summarize_heritability(
        user_results,
        selected_ancestry=selected_ancestry,
        restrict_to_selected=restrict_to_selected,
    )
    risk_metrics = [
        metric
        for metric in metrics
        if _clean_metadata_text(metric.get("risk")) or _clean_metadata_text(metric.get("ratio"))
    ]
    return format_heritability_risk_prompt(risk_metrics)


def build_prs_ai_prompt(
    kind: Literal["score", "trait_summary", "trait_results"],
    *,
    row: dict[str, Any] | None = None,
    user_results: list[dict[str, Any]] | None = None,
    trait: str = "",
    ancestry: str = "EUR",
    limit: int = 6000,
    sample_name: str | None = None,
    model_scope: str = "usable",
    percentile_source: str = "native",
) -> str:
    """Build the LLM prompt used by both CLI reports and the web UI."""
    lines: list[str]

    if kind == "trait_results":
        return _build_trait_prompt(
            trait,
            user_results or [],
            ancestry=ancestry,
            limit=limit,
            sample_name=sample_name,
            model_scope=model_scope,
            percentile_source=percentile_source,
        )

    prompt_row = row or {}
    if kind == "score":
        pgs_id = str(prompt_row.get("pgs_id") or "")
        row_trait = str(prompt_row.get("trait") or "unknown trait")
        efo = str(prompt_row.get("trait_efo_id") or "")
        pct = prompt_row.get("percentile", "")
        score = prompt_row.get("score", "")
        match_rate = prompt_row.get("match_rate", "")
        quality = str(prompt_row.get("quality_label") or "N/A")
        sq = prompt_row.get("synthetic_quality", "")
        auroc = prompt_row.get("auroc", "")
        abs_risk = str(prompt_row.get("absolute_risk_text") or prompt_row.get("absolute_risk") or "")
        effect = str(prompt_row.get("effect_size") or "")
        classification = str(prompt_row.get("classification") or "")
        row_ancestry = str(prompt_row.get("ancestry") or "")
        genome_file = str(prompt_row.get("genome_file") or prompt_row.get("vcf_name") or "")
        heritability = str(prompt_row.get("heritability") or "")
        variants_matched = prompt_row.get("variants_matched", "")
        variants_total = prompt_row.get("variants_total", "")
        variants_observed = int(prompt_row.get("variants_observed") or 0)
        variants_assumed_hom_ref = int(prompt_row.get("variants_assumed_hom_ref") or 0)
        variants_unscorable_absent = int(prompt_row.get("variants_unscorable_absent") or 0)
        variants_no_call = int(prompt_row.get("variants_no_call") or 0)
        method = str(prompt_row.get("percentile_method") or "")
        reference_audit_status = str(prompt_row.get("reference_audit_status") or "")
        reference_audit_issues = _reference_audit_issue_text(str(prompt_row.get("reference_audit_issues") or ""))
        is_harmonized = prompt_row.get("is_harmonized", False)
        risk_agreement = str(prompt_row.get("risk_agreement") or "")
        publication_label = _publication_display_label(prompt_row)

        lines = [
            "Interpret this Polygenic Risk Score (PRS) result for a citizen scientist.",
            "",
            f"Trait: {row_trait}" + (f" (EFO: {efo})" if efo else ""),
            f"PGS ID: {pgs_id}  https://www.pgscatalog.org/score/{pgs_id}/",
        ]
        if genome_file:
            lines.append(f"Genome/VCF input: {genome_file}")
        lines.extend(_source_link_prompt_lines([prompt_row], [pgs_id]))
        lines.append("")
        if pct:
            lines.append(f"Percentile: {pct} (method: {method})")
        if reference_audit_status in {"warning", "error"}:
            lines.append(
                f"Reference percentile audit status: {reference_audit_status}. "
                f"Issues: {reference_audit_issues or 'see audit sidecar'}."
            )
        if score:
            lines.append(f"Raw PRS value: {score}")
        if variants_matched and variants_total:
            lines.append(
                f"Model coverage: {variants_matched}/{variants_total} markers found "
                f"in this genome file ({match_rate}%). Explain that low coverage "
                "means the score used only part of the PRS model."
            )
            if variants_unscorable_absent or variants_assumed_hom_ref or variants_no_call:
                lines.append(
                    "Coverage breakdown: "
                    f"{variants_observed} observed, "
                    f"{variants_assumed_hom_ref} safely inferred as homozygous-reference, "
                    f"{variants_unscorable_absent} unavailable because the reference allele was unknown, "
                    f"{variants_no_call} present but no-called."
                )
        elif match_rate:
            lines.append(f"Model coverage: {match_rate}%")
        if publication_label != "Publication metadata unavailable":
            lines.append(f"Source publication: {publication_label}")
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
        if row_ancestry:
            lines.append(f"Evaluation ancestry: {row_ancestry}")
        if heritability and heritability != "N/A":
            lines.append(f"Heritability (h²): {heritability}")
        h2_risk_summary = _heritability_risk_prompt_summary(
            [prompt_row],
            selected_ancestry=str(prompt_row.get("selected_ancestry") or ancestry),
        )
        if h2_risk_summary:
            lines.append(f"h²-liability risk estimates: {h2_risk_summary}")
        if is_harmonized:
            orig = str(prompt_row.get("original_genome_build", "") or "")
            lines.append(
                f"Note: harmonized score — coordinates lifted over from {orig or 'another build'} "
                "to GRCh38. Liftover is never a complete, 100%-reliable remap, so some variants are "
                "dropped or mismapped and coverage/percentile carry extra uncertainty."
            )
        if prompt_row.get("build_mismatch"):
            detected = str(prompt_row.get("detected_genome_build", "") or "a different build")
            lines.append(
                f"Warning: the uploaded VCF was detected as {detected} but scored on GRCh38 — "
                "match rate and percentile may be unreliable due to a genome-build mismatch."
            )
        lines.append("")
        lines.append(_TRAIT_TYPE_GUIDANCE)
        lines.append("")
        if limit >= 3000:
            lines.extend([_QUALITY_METHODOLOGY, "", _METHODOLOGY_CONTEXT, "", _FORMAT_SINGLE_FULL])
        elif limit >= 1500:
            lines.append(_FORMAT_SINGLE_FULL)
        else:
            lines.append(_FORMAT_SINGLE_COMPACT)
    elif kind == "trait_summary":
        row_trait = str(prompt_row.get("trait") or "unknown trait")
        efo = str(prompt_row.get("trait_efo_id") or "")
        pgs_ids = str(prompt_row.get("pgs_ids") or "")
        pgs_id_list = [pgs_id.strip() for pgs_id in pgs_ids.split(",") if pgs_id.strip()]
        best_id = str(prompt_row.get("best_pgs_id") or "")
        best_pctl = prompt_row.get("best_model_pctl", "N/A")
        best_quality = str(prompt_row.get("best_quality") or "N/A")
        best_risk = str(prompt_row.get("best_absolute_risk") or "N/A")
        risk_vs_avg = str(prompt_row.get("risk_vs_average") or "N/A")
        population_average = str(prompt_row.get("population_average") or "N/A")
        risk_agreement = str(prompt_row.get("risk_agreement") or "")
        heritability = str(prompt_row.get("heritability") or "")
        heritability_detail = str(prompt_row.get("heritability_detail") or "")
        genome_file = str(prompt_row.get("genome_file") or prompt_row.get("vcf_name") or "")

        lines = [
            f"Interpret these combined Polygenic Risk Score (PRS) results "
            f"for \"{row_trait}\"" + (f" (EFO: {efo})" if efo else "") + ".",
            "",
            f"Models computed: {prompt_row.get('n_models', 0)} total, "
            f"{prompt_row.get('usable_models', 0)} usable (>=50% model coverage)"
            + (f" ({scope_phrase})" if (scope_phrase := str(prompt_row.get("scope_label") or "")) else ""),
            f"PGS IDs: {pgs_ids}",
        ]
        if genome_file:
            lines.append(f"Genome/VCF input: {genome_file}")
        lines.extend(_source_link_prompt_lines([], pgs_id_list, str(prompt_row.get("publication_links") or "")))
        lines.append("")
        scope_label = str(prompt_row.get("scope_label") or "")
        n_scoped = int(prompt_row.get("n_scoped") or prompt_row.get("usable_models") or 0)
        model_scope = str(prompt_row.get("model_scope") or "usable")
        of_scope = best_of_label(n_scoped, model_scope)
        if best_id:
            scope_note = f" ({scope_label})" if scope_label else ""
            lines.append(
                f"{of_scope.capitalize()}{scope_note}: {best_id} (pctl: {best_pctl}, quality: {best_quality})"
                f"  https://www.pgscatalog.org/score/{best_id}/"
            )
        lines.append(f"Median percentile: {prompt_row.get('typical_percentile', 'N/A')}")
        lines.append(
            f"Percentile range: {prompt_row.get('percentile_range', 'N/A')} "
            f"(SD: {prompt_row.get('percentile_std', 'N/A')})"
        )
        lines.append(f"Consistency: {str(prompt_row.get('consistency') or '')}")
        lines.append(f"Overall signal: {str(prompt_row.get('overall_signal') or '')}")
        if reliability := str(prompt_row.get("reliability") or ""):
            lines.append(f"Reliability: {reliability}")
        if best_match := prompt_row.get("best_match_rate", ""):
            lines.append(f"Best model coverage: {best_match:.1f}%")
        high_q = prompt_row.get("high_confidence_models", 0)
        high_q_median = prompt_row.get("high_confidence_median", "N/A")
        lines.append(f"High-quality models: {high_q}" + (f" (median pctl: {high_q_median})" if high_q else ""))
        if outlier_ids := str(prompt_row.get("outlier_ids") or ""):
            lines.append(f"Outlier models: {outlier_ids}")
        if best_risk and best_risk != "N/A":
            lines.append(f"Absolute risk ({of_scope}): {best_risk}")
        if population_average and population_average != "N/A":
            lines.append(f"Population average risk: {population_average}")
        if risk_vs_avg and risk_vs_avg != "N/A":
            lines.append(f"Risk vs population average: {risk_vs_avg}")
        if risk_agreement:
            lines.append(f"Risk method agreement: {risk_agreement}")
        if heritability and heritability != "N/A":
            lines.append(f"Heritability (h²): {heritability}")
        if heritability_detail and heritability_detail != "N/A":
            lines.append(f"Heritability detail: {heritability_detail}")
        h2_risk_summary = _heritability_risk_prompt_summary(
            [prompt_row],
            selected_ancestry=str(prompt_row.get("selected_ancestry") or ancestry),
        )
        if h2_risk_summary:
            lines.append(f"h²-liability risk estimates: {h2_risk_summary}")
        lines.append(
            "Priority: give a quick bottom-line risk interpretation first. Treat "
            "model agreement/spread as confidence evidence, not as the main answer."
        )
        lines.append("")
        lines.append(_TRAIT_TYPE_GUIDANCE)
        lines.append("")
        if limit >= 3000:
            lines.extend([_QUALITY_METHODOLOGY, "", _METHODOLOGY_CONTEXT, "", _FORMAT_TRAIT_FULL])
        elif limit >= 1500:
            lines.append(_FORMAT_TRAIT_FULL)
        else:
            lines.append(_FORMAT_TRAIT_COMPACT)
    else:
        raise ValueError(f"Unknown PRS AI prompt kind: {kind}")

    prompt = "\n".join(lines)
    if len(prompt) > limit:
        return prompt[:limit - 3] + "..."
    return prompt


def _build_trait_prompt(
    trait: str,
    user_results: list[dict],
    ancestry: str = "EUR",
    limit: int = 6000,
    sample_name: str | None = None,
    model_scope: str = "usable",
    percentile_source: str = "native",
) -> str:
    """Build a rich AI prompt summarizing a trait's PRS results.

    Includes per-model details, quality-stratified medians, source links,
    and instructions for looking up PGS Catalog pages.
    """
    scored = [
        {**r, "percentile": percentile}
        for r in user_results
        if (percentile := _parse_float(r.get("percentile"))) is not None
    ]
    if not scored:
        return ""

    stats = summarize_trait_rows(
        scored,
        model_scope=model_scope,
        selected_ancestry=ancestry,
        percentile_source=percentile_source,
    )
    pctls = sorted(stats.pct_by_id.values())
    median_pctl = stats.median_pct
    mean_pctl = stats.mean_pct
    best = stats.best_row
    pgs_ids = [r["pgs_id"] for r in scored]

    high_q = stats.high_quality_rows
    high_mod = [r for r in scored if _quality_tier(r.get("quality_label")) >= 3]
    usable = stats.usable_rows

    pctl_sd = stats.std_pct if stats.std_pct is not None else 0.0

    lines = [
        f'Interpret these combined Polygenic Risk Score (PRS) results for "{trait}".',
        "",
        "== SUMMARY ==",
        f"Models scored: {len(scored)} total"
        + (f", {len(usable)} usable (>=50% marker coverage)" if len(usable) != len(scored) else ""),
        f"Scope: {stats.scope_label}",
    ]
    if sample_name:
        lines.append(f"Genome/VCF input: {sample_name}")
    if high_q:
        hq_median = stats.high_quality_median
        if hq_median is None:
            hq_pctls = [r["percentile"] for r in high_q]
            hq_median = sorted(hq_pctls)[len(hq_pctls) // 2]
        lines.append(f"High-quality models: {len(high_q)} (median percentile: {hq_median:.1f})")
    if high_mod and len(high_mod) != len(high_q):
        hm_pctls = [r["percentile"] for r in high_mod]
        lines.append(f"High+Normal quality: {len(high_mod)} (median percentile: {sorted(hm_pctls)[len(hm_pctls)//2]:.1f})")
    if median_pctl is not None:
        lines.append(
            f"Scoped median percentile: {format_percentile_with_panel(median_pctl, stats.typical_panel)}"
            + (f", mean: {mean_pctl:.1f}" if mean_pctl is not None else "")
        )
    else:
        lines.append(f"Scoped median percentile: N/A ({stats.scope_label})")
    if stats.min_pct is not None and stats.max_pct is not None:
        lines.append(f"Percentile range: {stats.min_pct:.1f} - {stats.max_pct:.1f} (SD: {pctl_sd:.1f})")
    elif pctls:
        lines.append(f"Percentile range: {min(pctls):.1f} - {max(pctls):.1f} (SD: {pctl_sd:.1f})")
    else:
        lines.append("Percentile range: N/A (no models in scope)")
    lines.append(f"Consistency: {stats.consistency}")
    lines.append(f"Overall signal: {stats.overall_signal}")
    of_scope = best_of_label(stats.n_scoped, stats.model_scope)
    if stats.best_row is not None:
        best_pctl = stats.best_model_pctl if stats.best_model_pctl is not None else _parse_float(best.get("percentile"))
        lines.append(
            f"{of_scope.capitalize()}: {best.get('pgs_id')} "
            f"(percentile: {format_percentile_with_panel(best_pctl, stats.best_model_panel)}"
            + (f", quality: {best.get('quality_label', 'N/A')}" if best.get("quality_label") else "")
            + ")"
        )

    if stats.risk_vs_average != "N/A":
        lines.append(f"Risk vs population average: {stats.risk_vs_average}")
    if stats.absolute_risk and not stats.absolute_risk.startswith("N/A"):
        ar_str = stats.absolute_risk
        method = (stats.best_risk_row or {}).get("risk_method", "")
        if method:
            ar_str += f" [{method}]"
        lines.append(f"Absolute risk ({risk_basis_label(stats.n_risk_models)}): {ar_str}")
    elif stats.absolute_risk.startswith("N/A"):
        lines.append(f"Absolute risk: {stats.absolute_risk}")
    restrict_h2 = percentile_source == "selected"
    h2_summary = _heritability_prompt_summary(
        stats.scoped_rows or scored,
        selected_ancestry=ancestry,
        restrict_to_selected=restrict_h2,
    )
    if h2_summary:
        lines.append(f"Heritability (h²): {h2_summary}")
    h2_risk_summary = _heritability_risk_prompt_summary(
        stats.scoped_rows or scored,
        selected_ancestry=ancestry,
        restrict_to_selected=restrict_h2,
    )
    if h2_risk_summary:
        lines.append(f"h²-liability risk estimates: {h2_risk_summary}")

    lines.append("")
    lines.append("== PER-MODEL DETAILS ==")
    sorted_scored = sorted(scored, key=lambda r: (
        -_quality_tier(r.get("quality_label")), -r["percentile"]
    ))
    if limit < 5000:
        display_models = [r for r in sorted_scored if _quality_tier(r.get("quality_label")) >= 3]
        if not display_models:
            display_models = sorted_scored
        skipped_count = len(sorted_scored) - len(display_models)
        skipped_label = "lower-quality models omitted" if skipped_count else ""
    else:
        display_models = sorted_scored
        skipped_count = 0
        skipped_label = ""
    model_limit = 30 if limit >= 5000 else (15 if limit >= 3000 else 8)
    for r in display_models[:model_limit]:
        parts = [f"{r['pgs_id']}: pctl={r['percentile']:.1f}"]
        score = _parse_float(r.get("score"))
        if score is not None:
            parts.append(f"score={score:.6f}")
        mr = _parse_float(r.get("match_rate"))
        if mr is not None:
            mr_pct = mr * 100 if mr <= 1.0 else mr
            parts.append(f"match={mr_pct:.0f}%")
            vt = r.get("variants_total")
            vm = r.get("variants_matched")
            if vt and vm:
                parts.append(f"({vm}/{vt} variants)")
        if r.get("quality_label"):
            parts.append(f"quality={r['quality_label']}")
        auroc = _parse_float(r.get("auroc"))
        or_estimate = _parse_float(r.get("or_estimate"))
        if auroc is not None:
            parts.append(f"AUROC={auroc:.3f}")
        elif or_estimate is not None:
            parts.append(f"OR={or_estimate:.2f}")
        if r.get("is_harmonized"):
            parts.append("(harmonized)")
        lines.append("  " + ", ".join(parts))
    remaining = len(display_models) - model_limit
    if remaining > 0 or skipped_count > 0:
        extra = []
        if remaining > 0:
            extra.append(f"{remaining} more")
        if skipped_count > 0:
            extra.append(f"{skipped_count} {skipped_label}")
        lines.append(f"  ... ({', '.join(extra)})")

    lines.append("")
    lines.append("== SOURCE LINKS ==")
    lines.append("Look up PGS Catalog pages for detail on each model (publication, metrics, variants):")
    link_limit = 10 if limit >= 5000 else (5 if limit >= 3000 else 3)
    for pid in pgs_ids[:link_limit]:
        lines.append(f"  https://www.pgscatalog.org/score/{pid}/")
    if len(pgs_ids) > link_limit:
        lines.append(f"  (+ {len(pgs_ids) - link_limit} more at https://www.pgscatalog.org/)")

    lines.append("")
    if limit >= 3000:
        lines.append("== METHODOLOGY ==")
        lines.append(
            "Percentiles computed by scoring the 1000 Genomes Project "
            "phase 3 reference panel (2,504 individuals, 5 superpopulations: "
            f"AFR, AMR, EAS, EUR, SAS) on GRCh38 harmonized scoring files. Ancestry: {ancestry}."
        )
        lines.append(
            "Quality scoring: each model gets a synthetic quality score (0-100) based on "
            "AUROC/C-index (no penalty), Beta-only (0.95), OR/HR-only (0.90), "
            "no-metric (0.6). Labels: High (>=70), Normal (>=50), Moderate (>=30), Low (<30)."
        )
        lines.append("")
        lines.append(_TRAIT_TYPE_GUIDANCE)
        lines.append("")
    lines.append("== RESPONSE FORMAT ==")
    if limit >= 3000:
        lines.extend([
            "Structure your response EXACTLY as follows (under 240 words for the main part):",
            "1. **Verdict** — one bold sentence summarizing the overall genetic predisposition.",
            "2. **Risk in real terms** — 2-4 bullets with percentile, absolute risk vs population average when provided, and h²/heredity context when provided.",
            "3. **Confidence** — combine quality tiers, model coverage, h² source, risk-method agreement, and ancestry. Mention model spread only as supporting evidence.",
            "4. **Context & actions** — what this trait IS (health/behavioral/physical), and whether action makes sense.",
            "Citizen scientist audience — clarity and honesty over length.",
            "Do not spend the main answer grouping where models agree/disagree unless it changes the risk interpretation.",
            "",
            "After the main section, you MAY add a clearly separated section (---) with:",
            "- Additional caveats, ancestry considerations, or trait-specific biology",
            "- Links to PGS Catalog pages for the top models if you can access them",
        ])
    else:
        lines.extend([
            "Reply in under 150 words. Start with ONE bold verdict sentence, "
            "then 3 bullet points: risk in real terms, h²/heredity if provided, confidence/actionability. "
            "Do not make model disagreement the main answer. PRS is genetic predisposition, not diagnosis.",
        ])

    prompt = "\n".join(lines)
    if len(prompt) > limit:
        prompt = prompt[:limit - 3] + "..."
    return prompt


def trait_report_html(
    chart: alt.Chart | alt.LayerChart | alt.VConcatChart,
    trait: str,
    user_results: list[dict],
    ancestry: str = "EUR",
    sample_name: str | None = None,
    model_scope: str = "usable",
    percentile_source: str = "native",
) -> str:
    """Render an Altair trait chart as a rich HTML report string.

    Generates the same standalone report used by the CLI:
    - The Vega-Lite chart
    - Key Statistics panel (percentile, risk vs average, absolute risk)
    - "Ask AI" buttons linking to Claude, ChatGPT, Perplexity with a pre-built prompt
    """
    import urllib.parse

    scored = [
        {**r, "percentile": percentile}
        for r in user_results
        if (percentile := _parse_float(r.get("percentile"))) is not None
    ]
    # Stamp each model with the same canonical quality tier the chart uses so the
    # table, the quality-stratified median rows, and the headline cohort-median
    # cards all agree with the chart's High/Moderate/Low grouping.  Classification
    # lives in just_prs.quality.resolve_quality_key — never reinvent it here.
    # ``scored`` holds copies (``{**r, ...}``), so this never mutates caller dicts.
    for r in scored:
        if not r.get("quality"):
            n_var_val = _parse_float(r.get("variants_total"))
            if n_var_val is None:
                n_var_val = _parse_float(r.get("n_variants"))
            coverage = _parse_float(r.get("weight_mass_coverage"))
            if coverage is None:
                coverage = _parse_float(r.get("match_rate"))
            r["quality"] = resolve_quality_key(
                label=r.get("quality_label"),
                n_var=int(n_var_val) if n_var_val is not None else None,
                auroc=_parse_float(r.get("auroc")),
                coverage=coverage,
            )

    # If the absolute scale bunches every model into the top tiers (e.g. a WGS
    # sample where C_wt ≈ 1.0 for all), re-colour relative to this cohort so the
    # quality heatmap stays informative instead of uniformly green.  Ranks on the
    # model-intrinsic synthetic quality (which keeps spread even when coverage is
    # uniform).  No-op when the absolute scale already discriminates.
    def _sq_norm(row: dict) -> float | None:
        sq = _parse_float(row.get("synthetic_quality"))
        return sq / 100.0 if sq is not None else None

    def _row_coverage(row: dict) -> float | None:
        coverage = _parse_float(row.get("weight_mass_coverage"))
        if coverage is None:
            coverage = _parse_float(row.get("match_rate"))
        return coverage

    rescaled = rescale_quality_if_degenerate(
        [r.get("quality") for r in scored],
        [_sq_norm(r) for r in scored],
        coverages=[_row_coverage(r) for r in scored],
    )
    for r, key in zip(scored, rescaled):
        if key is not None:
            r["quality"] = key
    spec_json = chart.to_json()

    stats_html = ""
    ai_html = ""

    if scored:
        stats = summarize_trait_rows(
            scored,
            model_scope=model_scope,
            selected_ancestry=ancestry,
            percentile_source=percentile_source,
        )
        pctls = sorted(stats.pct_by_id.values()) or sorted([r["percentile"] for r in scored])
        median_pctl = stats.median_pct
        of_scope = best_of_label(stats.n_scoped, stats.model_scope)

        cards = []
        median_display = (
            format_percentile_with_panel(median_pctl, stats.typical_panel)
            if median_pctl is not None
            else "N/A"
        )
        best_display = (
            format_percentile_with_panel(stats.best_model_pctl, stats.best_model_panel)
            if stats.best_model_pctl is not None
            else "N/A"
        )
        # Median first: it is the robust headline number.  The quality-best
        # model's percentile is secondary context and must never be called
        # "best" — for direction-dependent traits that reads as "best outcome".
        cards.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">Median Percentile</div>'
            f'<div class="stat-value">{median_display}</div>'
            f'<div class="stat-sub">{stats.scope_label}</div></div>'
        )
        best_pgs_id = str((stats.best_row or {}).get("pgs_id") or "")
        best_sub = best_pgs_id or best_of_label(stats.n_scoped, stats.model_scope)
        if best_pgs_id:
            best_sub = f"{best_pgs_id} · {best_of_label(stats.n_scoped, stats.model_scope)}"
        cards.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">Most Reliable Model</div>'
            f'<div class="stat-value">{best_display}</div>'
            f'<div class="stat-sub">{best_sub}</div></div>'
        )
        cards.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">Models</div>'
            f'<div class="stat-value">{stats.n_scoped}</div>'
            f'<div class="stat-sub">{stats.scope_label}</div></div>'
        )
        restrict_h2 = percentile_source == "selected"
        h2_summary = stats.heritability_text if stats.heritability_text != NO_MAPPED_H2 else ""
        if not h2_summary:
            h2_summary = _heritability_prompt_summary(
                scored,
                selected_ancestry=ancestry,
                restrict_to_selected=restrict_h2,
            )
        if h2_summary:
            cards.append(
                f'<div class="stat-card">'
                f'<div class="stat-label">Heritability (h²)</div>'
                f'<div class="stat-value h2-value">{_shorten_text(h2_summary, 48)}</div>'
                f'<div class="stat-sub">population-level heredity context</div></div>'
            )

        risk_basis = risk_basis_label(stats.n_risk_models)
        if stats.risk_vs_average != "N/A":
            rr = _parse_float(str(stats.risk_vs_average).rstrip("x"))
            if rr is not None and rr >= 1.05:
                color, direction = "#C62828", "higher than population"
            elif rr is not None and rr <= 0.95:
                color, direction = "#2E7D32", "lower than population"
            else:
                color, direction = "#455A64", "similar to population"
            cards.append(
                f'<div class="stat-card">'
                f'<div class="stat-label">Risk vs Average</div>'
                f'<div class="stat-value" style="color:{color}">{stats.risk_vs_average}</div>'
                f'<div class="stat-sub">{direction} · {risk_basis}</div></div>'
            )

        cards.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">Absolute Risk</div>'
            f'<div class="stat-value">{stats.absolute_risk}</div>'
            f'<div class="stat-sub">{risk_basis}</div></div>'
        )

        if stats.std_pct is not None:
            cards.append(
                f'<div class="stat-card">'
                f'<div class="stat-label">Model Spread</div>'
                f'<div class="stat-value">{stats.std_pct:.1f}</div>'
                f'<div class="stat-sub">percentile SD · {stats.scope_label}</div></div>'
            )

        stats_html = '<div class="stats-grid">' + "".join(cards) + "</div>"

        btns = []
        for ai in _AI_ASSISTANTS:
            char_limit = ai.get("limit", 6000)
            prompt = build_prs_ai_prompt(
                "trait_results",
                user_results=user_results,
                trait=trait,
                ancestry=ancestry,
                limit=char_limit,
                sample_name=sample_name,
                model_scope=model_scope,
                percentile_source=percentile_source,
            )
            if prompt:
                encoded = urllib.parse.quote(prompt, safe="")
                btns.append(
                    f'<a class="ai-btn" style="background:{ai["color"]}" '
                    f'href="{ai["url"]}{encoded}" target="_blank" rel="noopener">'
                    f'Ask {ai["name"]}</a>'
                )
        if btns:
            ai_html = '<div class="ai-buttons"><span class="ai-label">Interpret with AI:</span>' + "".join(btns) + "</div>"

    table_html = ""
    if scored:
        q_colors = {"high": "#2E7D32", "moderate": "#1565C0", "low": "#E65100", "very_low": "#C62828"}
        has_risk = any(r.get("risk_ratio") is not None for r in scored)
        has_h2 = any(
            _heritability_prompt_summary(
                [r],
                selected_ancestry=ancestry,
                restrict_to_selected=restrict_h2,
            )
            for r in scored
        )
        distinct_traits = {r.get("trait_reported", "") for r in scored} - {""}
        has_multi = len(distinct_traits) > 1

        _TIER_ORDER = {"high": 0, "moderate": 1, "low": 2, "very_low": 3}
        sorted_results = sorted(
            scored,
            key=lambda r: (_TIER_ORDER.get(r.get("quality", "low"), 3), -(r.get("percentile") or 0)),
        )

        # One median, the Quality-dropdown selection — not three always-on
        # comparison rows that ignore the dropdown.
        _SCOPE_MEDIAN_COLOR = {
            "high_quality": "#2E7D32",
            "high_moderate": "#1565C0",
            "usable": "#1565C0",
            "all": "#D32F2F",
        }
        median_rows: list[tuple[str, str, float | None, int]] = [(
            "Median (selected)",
            _SCOPE_MEDIAN_COLOR.get(stats.model_scope, "#1565C0"),
            stats.median_pct,
            stats.n_scoped,
        )]

        rows_html = []
        n_data_cols = 5 + (1 if has_multi else 0) + (2 if has_risk else 0) + (1 if has_h2 else 0)
        for label, color, med_pctl, count in median_rows:
            cells = [f'<td colspan="2" style="font-weight:700;color:{color};border-bottom:2px solid {color}20">{label}</td>']
            med_s = f"{med_pctl:.1f}%" if med_pctl is not None else "N/A"
            cells.append(f'<td style="font-weight:700;color:{color};font-size:1.1em;border-bottom:2px solid {color}20">{med_s}</td>')
            remaining = n_data_cols - 3
            for _ in range(remaining - 1):
                cells.append(f'<td style="border-bottom:2px solid {color}20"></td>')
            cells.append(f'<td style="color:{color};font-weight:600;border-bottom:2px solid {color}20">{count} models</td>')
            rows_html.append("<tr>" + "".join(cells) + "</tr>")

        for r in sorted_results:
            pctl = r.get("percentile")
            pctl_s = f"{pctl:.1f}%" if pctl is not None else "—"
            mr = r.get("match_rate")
            match_s = _format_percent_like(mr)
            q = r.get("quality", "low")
            q_col = q_colors.get(q, "#999")
            vt = r.get("variants_total")
            vm = r.get("variants_matched")
            if vt and vm:
                n_var = f"{vm:,}/{vt:,}"
            elif vt:
                n_var = f"{vt:,}"
            else:
                n_var = "—"
            sname = r.get("score_name", "")
            id_display = f'{r["pgs_id"]} ({sname})' if sname else r["pgs_id"]
            cells = [f'<td class="id-cell">{id_display}</td>']
            if has_multi:
                trait_raw = r.get("trait_reported", "")
                trait_short = (trait_raw[:29] + "…") if len(trait_raw) > 30 else trait_raw
                cells.append(f'<td class="trait-cell">{trait_short}</td>')
            cells.append(f'<td style="color:{q_col};font-weight:600">{pctl_s}</td>')
            if has_risk:
                rr = r.get("risk_ratio")
                rr_value = _parse_float(rr)
                if rr_value is not None:
                    rc = "#C62828" if rr_value >= 1.0 else "#2E7D32"
                    cells.append(f'<td style="color:{rc};font-weight:700">{rr_value:.2f}x</td>')
                else:
                    cells.append("<td>—</td>")
                ar = r.get("absolute_risk")
                prev = r.get("population_prevalence")
                if ar is not None:
                    ar_s = _format_absolute_risk(ar)
                    if prev is not None:
                        ar_s += f" (avg {_format_percent_like(prev)})"
                    cells.append(f"<td>{ar_s}</td>")
                else:
                    cells.append("<td>—</td>")
            if has_h2:
                h2_text = _heritability_prompt_summary(
                    [r],
                    selected_ancestry=ancestry,
                    restrict_to_selected=restrict_h2,
                )
                cells.append(f'<td class="h2-cell">{_shorten_text(h2_text, 80) if h2_text else "—"}</td>')
            cells.append(f"<td>{n_var}</td>")
            cells.append(f"<td>{match_s}</td>")
            cells.append(f'<td><span class="q-dot" style="background:{q_col}"></span> {q}</td>')
            rows_html.append("<tr>" + "".join(cells) + "</tr>")

        hdr_cells = ["<th>PGS ID</th>"]
        if has_multi:
            hdr_cells.append("<th>Trait</th>")
        hdr_cells.append("<th>Percentile</th>")
        if has_risk:
            hdr_cells.extend(["<th>Risk×</th>", "<th>Abs Risk</th>"])
        if has_h2:
            hdr_cells.append("<th>h²</th>")
        hdr_cells.extend(["<th>Variants</th>", "<th>Match%</th>", "<th>Quality</th>"])

        table_html = (
            '<table class="model-table"><thead><tr>'
            + "".join(hdr_cells)
            + "</tr></thead><tbody>"
            + "".join(rows_html)
            + "</tbody></table>"
        )

    quality_note_html = _QUALITY_NOTE_HTML if scored else ""

    dashboard_stamp = (
        f"<!-- prs-dashboard {model_scope}|{percentile_source}|{ancestry}|"
        f"{stats.median_pct if scored else 'na'}|{stats.risk_vs_average if scored else 'na'}|"
        f"{stats.n_scoped if scored else 0} -->"
    )
    html = f"""<!DOCTYPE html>
{dashboard_stamp}
<html lang="en">
<head>
<meta charset="utf-8">
<title>PRS Report: {trait}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #fafafa; color: #333; }}
h1 {{ font-size: 1.4em; margin-bottom: 4px; }}
.subtitle {{ color: #666; margin-bottom: 16px; font-size: 0.95em; }}
.stats-grid {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }}
.stat-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; min-width: 140px; flex: 1; }}
.stat-label {{ font-size: 0.78em; color: #888; text-transform: uppercase; letter-spacing: 0.03em; }}
.stat-value {{ font-size: 1.6em; font-weight: 700; margin: 2px 0; }}
.h2-value {{ font-size: 1.05em; line-height: 1.25; }}
.stat-sub {{ font-size: 0.82em; color: #999; }}
.ai-buttons {{ margin: 18px 0 0; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.ai-label {{ font-size: 0.9em; color: #666; }}
.ai-btn {{ display: inline-block; padding: 7px 16px; border-radius: 6px; color: #fff; text-decoration: none; font-size: 0.88em; font-weight: 600; }}
.ai-btn:hover {{ opacity: 0.85; }}
#vis {{ margin-top: 12px; }}
.vega-embed {{ width: 100%; }}
.vega-embed canvas, .vega-embed svg {{ max-width: 100%; }}
.model-table {{ width: 100%; border-collapse: collapse; margin-top: 18px; font-size: 0.9em; }}
.model-table th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid #ddd; color: #555; font-weight: 600; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.03em; }}
.model-table td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
.model-table tbody tr:hover {{ background: #f5f5f5; }}
.id-cell {{ font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.92em; }}
.trait-cell {{ max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #555; }}
.h2-cell {{ max-width: 260px; white-space: normal; color: #444; font-size: 0.88em; }}
.q-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
.quality-note {{ margin-top: 16px; padding: 12px 16px; background: #f0f4f8; border: 1px solid #d6e0ea; border-left: 4px solid #1565c0; border-radius: 8px; font-size: 0.85em; color: #44525e; line-height: 1.5; }}
.quality-note .qn-title {{ font-weight: 700; color: #1565c0; margin-bottom: 6px; }}
.quality-note ul {{ margin: 6px 0 0; padding-left: 18px; }}
.quality-note li {{ margin: 3px 0; }}
.quality-note b {{ color: #2d3a44; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/vega@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>
</head>
<body>
<h1>PRS Report: {trait}</h1>
<div class="subtitle">Ancestry: {SUPERPOP_LABELS.get(ancestry, ancestry)} ({ancestry})</div>
{stats_html}
<div id="vis"></div>
{table_html}
{quality_note_html}
{ai_html}
<script>
// Report its true content height to the embedding parent so the host iframe can
// size itself to the content instead of guessing a fixed height (the root cause
// of the "jumping height" — a fixed iframe height clips a 65-model table or
// leaves a tall empty gap).  Harmless when opened standalone (parent === self).
function _prsPostHeight() {{
  var h = Math.max(
    document.body.scrollHeight, document.documentElement.scrollHeight,
    document.body.offsetHeight, document.documentElement.offsetHeight
  );
  parent.postMessage({{type: 'prs-report-height', height: h}}, '*');
}}
vegaEmbed('#vis', {spec_json}, {{actions: true, width: Math.max(800, window.innerWidth - 80)}})
  .then(_prsPostHeight)
  .catch(console.error);
window.addEventListener('load', _prsPostHeight);
window.addEventListener('resize', _prsPostHeight);
if (window.ResizeObserver) {{ new ResizeObserver(_prsPostHeight).observe(document.body); }}
</script>
</body>
</html>"""

    return html


def save_trait_report(
    chart: alt.Chart | alt.LayerChart | alt.VConcatChart,
    path: Path,
    trait: str,
    user_results: list[dict],
    ancestry: str = "EUR",
    sample_name: str | None = None,
) -> Path:
    """Save an Altair trait chart as a rich HTML report with Key Statistics and AI buttons.

    For .html output, generates a self-contained page with the same report
    returned by ``trait_report_html()``. For non-HTML formats, falls back to
    ``save_chart()``.
    """
    path = Path(path)
    if path.suffix.lower() != ".html":
        return save_chart(chart, path)

    path.parent.mkdir(parents=True, exist_ok=True)
    html = trait_report_html(chart, trait, user_results, ancestry, sample_name=sample_name)
    path.write_text(html, encoding="utf-8")
    return path
