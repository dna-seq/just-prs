"""Altair-based PRS visualizations: bell curves, percentile markers, multi-population comparisons.

Altair is a core dependency. For PNG/SVG export, install the ``viz`` extra:
``pip install just-prs[viz]`` (adds ``vl-convert-python``).
HTML and JSON export work out of the box.
"""

from __future__ import annotations

import math
from pathlib import Path

import altair as alt
import polars as pl


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
        pct_user = _norm_cdf((user_score - mean) / std) * 100
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
    max_scores: int = 25,
    title: str | None = None,
    width: int = 600,
    height: int = 250,
    show_table: bool = False,
    table_height: int | None = None,
) -> alt.LayerChart | alt.VConcatChart:
    """Trait-grouped visualization: reference bell curve + per-model user percentile scatter.

    Shows a standard-normal reference bell curve (the shared z-score space) with
    each model's user score plotted on it as a dot colored by model quality tier.
    Without user results, shows model quality comparison as a jittered dot strip.

    When ``ancestries`` is provided (e.g. ``["EUR", "AFR", "EAS", "AMR", "SAS"]``),
    overlays one color-coded bell curve per population instead of the single gray
    N(0,1) reference.  User dots are z-normalized against ``ancestry`` (the primary).

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
        max_scores: Maximum number of PGS models to show (top by variant count).
        title: Chart title override.
        width: Chart width in pixels.
        height: Bell curve chart height in pixels.
        show_table: If True, append a model summary table below the bell curve.
        table_height: Height of the summary table in pixels (default: auto-sized from row count).

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

    def _quality_tier(n_var: int | None, auroc: float | None) -> str:
        if auroc is not None and auroc >= 0.7:
            return "high"
        if n_var is not None and n_var >= 100_000:
            return "high"
        if n_var is not None and n_var >= 10_000:
            return "moderate"
        if n_var is not None and n_var >= 100:
            return "low"
        return "very_low"

    model_meta: list[dict] = []
    for row_dict in sub.iter_rows(named=True):
        pgs_id = row_dict["pgs_id"]
        mean, std = row_dict["mean"], row_dict["std"]
        n_var = row_dict.get("variants_total") or row_dict.get("n_variants") or 0
        auroc = row_dict.get("auroc_estimate")
        tier = _quality_tier(n_var, auroc)
        n_var_label = f"{n_var:,}" if n_var else "?"

        z_user: float | None = None
        pct_user: float | None = None
        match_rate: float | None = None
        if pgs_id in user_lookup:
            ur = user_lookup[pgs_id]
            z_user = ur.get("z_score")
            pct_user = ur.get("percentile")
            match_rate = ur.get("match_rate")
            if z_user is None and "score" in ur and std > 0:
                z_user = (ur["score"] - mean) / std
                pct_user = _norm_cdf(z_user) * 100
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

        model_meta.append({
            "pgs_id": pgs_id,
            "trait_reported": trait_reported,
            "quality": tier,
            "n_variants": n_var,
            "n_variants_label": n_var_label,
            "auroc": auroc,
            "auroc_label": f"AUROC={auroc:.2f}" if auroc is not None else "",
            "z_score": z_user,
            "percentile": pct_user,
            "match_rate": match_rate,
            "risk_ratio": risk_ratio,
            "absolute_risk": absolute_risk,
            "population_prevalence": population_prevalence,
            "mean": mean,
            "std": std,
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

            bg_layers.append(
                alt.Chart(pop_source)
                .mark_area(opacity=0.08)
                .encode(
                    x=alt.X("z_score:Q", title="Z-Score (each model independently normalized)"),
                    y=alt.Y("density:Q", title="Density"),
                    color=alt.Color(
                        "population:N",
                        scale=alt.Scale(domain=pop_domain, range=pop_colors),
                        title="Population",
                    ),
                )
            )
            bg_layers.append(
                alt.Chart(pop_source)
                .mark_line(strokeWidth=1.8, opacity=0.6)
                .encode(
                    x="z_score:Q",
                    y="density:Q",
                    color=alt.Color("population:N", scale=alt.Scale(domain=pop_domain, range=pop_colors), legend=None),
                )
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
        .mark_text(dy=14, fontSize=9, color="#666")
        .encode(x="z_score:Q", text="label:N")
    )

    # Build foreground layers (user dots + labels + median)
    fg_layers: list = []

    user_marks = [m for m in model_meta if m["z_score"] is not None]
    has_user = len(user_marks) > 0

    if has_user:
        import random
        rng = random.Random(42)
        for um in user_marks:
            z_c = max(-3.4, min(3.4, um["z_score"]))
            um["z_display"] = z_c
            base_density = _norm_pdf(z_c)
            um["density"] = base_density + rng.uniform(-0.02, 0.04)

        for um in user_marks:
            um["short_id"] = um["pgs_id"].replace("PGS00", "").replace("PGS0", "").replace("PGS", "")

        marks_source = alt.Data(values=user_marks)

        fg_layers.append(
            alt.Chart(marks_source)
            .mark_rule(strokeDash=[3, 3], opacity=0.35, strokeWidth=1)
            .encode(
                x="z_display:Q",
                color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), legend=None),
            )
        )

        fg_layers.append(
            alt.Chart(marks_source)
            .mark_point(size=50, filled=True, strokeWidth=1, stroke="white", opacity=0.8)
            .encode(
                x="z_display:Q",
                y="density:Q",
                color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), title="Model quality"),
                tooltip=[
                    alt.Tooltip("pgs_id:N", title="PGS ID"),
                    alt.Tooltip("quality:N"),
                    alt.Tooltip("n_variants_label:N", title="Variants"),
                    alt.Tooltip("auroc_label:N", title="AUROC"),
                    alt.Tooltip("percentile:Q", title="Percentile", format=".1f"),
                    alt.Tooltip("z_score:Q", title="Z-score", format=".2f"),
                ],
            )
        )

        fg_layers.append(
            alt.Chart(marks_source)
            .mark_text(fontSize=8, dy=-10, opacity=0.7)
            .encode(
                x="z_display:Q",
                y="density:Q",
                text="short_id:N",
                color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), legend=None),
            )
        )

        z_vals = [m["z_score"] for m in user_marks]
        median_z = sorted(z_vals)[len(z_vals) // 2]
        median_pct = _norm_cdf(median_z) * 100
        fg_layers.append(
            alt.Chart(alt.Data(values=[{
                "z_score": median_z,
                "label": f"Median: {median_pct:.0f}th pctl",
            }]))
            .mark_rule(color="#D32F2F", strokeWidth=2.5)
            .encode(x="z_score:Q")
        )
        fg_layers.append(
            alt.Chart(alt.Data(values=[{
                "z_score": median_z,
                "density": _norm_pdf(median_z) + 0.02,
                "label": f"Median: {median_pct:.0f}th",
            }]))
            .mark_text(fontSize=11, fontWeight="bold", color="#D32F2F", dy=-10)
            .encode(x="z_score:Q", y="density:Q", text="label:N")
        )
    else:
        import random
        rng = random.Random(42)
        for i, mm in enumerate(model_meta):
            mm["jitter_y"] = 0.005 + rng.uniform(0, 0.06)
            mm["x_pos"] = (i / max(len(model_meta) - 1, 1)) * 6 - 3

        fg_layers.append(
            alt.Chart(alt.Data(values=model_meta))
            .mark_point(size=40, filled=True, opacity=0.7)
            .encode(
                x=alt.X("x_pos:Q"),
                y=alt.Y("jitter_y:Q"),
                color=alt.Color("quality:N", scale=alt.Scale(domain=q_domain, range=q_range), title="Model quality"),
                tooltip=[
                    alt.Tooltip("pgs_id:N", title="PGS ID"),
                    alt.Tooltip("quality:N"),
                    alt.Tooltip("n_variants_label:N", title="Variants"),
                    alt.Tooltip("auroc_label:N", title="AUROC"),
                ],
            )
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

    if not show_table:
        return bell.configure_axis(grid=False).configure_view(strokeWidth=0)

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
        tr["match_label"] = f'{mr:.0%}' if mr is not None else "—"
        tr["short_id"] = tr.get("short_id") or tr["pgs_id"]
        tr["row_idx"] = idx
        rr = tr.get("risk_ratio")
        tr["risk_label"] = f"{rr:.2f}x" if rr is not None else "—"
        ar = tr.get("absolute_risk")
        prev = tr.get("population_prevalence")
        if ar is not None:
            ar_str = f"{ar:.1%}"
            if prev is not None:
                ar_str += f" (avg {prev:.1%})"
            tr["abs_risk_label"] = ar_str
        else:
            tr["abs_risk_label"] = "—"
        raw_trait = tr.get("trait_reported", "")
        tr["trait_label"] = (raw_trait[:max_trait_len - 1] + "…") if len(raw_trait) > max_trait_len else raw_trait

    row_height = 24
    n_rows = len(table_rows)
    t_height = table_height or max(60, n_rows * row_height + 30)
    table_source = alt.Data(values=table_rows)

    label_font = max(9, min(12, width // 70))
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
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
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
    {"name": "Claude", "url": "https://claude.ai/new?q=", "color": "#DA7756"},
    {"name": "ChatGPT", "url": "https://chatgpt.com/?q=", "color": "#10A37F"},
    {"name": "Perplexity", "url": "https://www.perplexity.ai/search?q=", "color": "#21808D"},
]


def _build_trait_prompt(
    trait: str,
    user_results: list[dict],
    ancestry: str = "EUR",
) -> str:
    """Build an AI prompt summarizing a trait's PRS results."""
    scored = [r for r in user_results if r.get("percentile") is not None]
    if not scored:
        return ""

    pctls = sorted([r["percentile"] for r in scored])
    median_pctl = pctls[len(pctls) // 2]
    best = max(scored, key=lambda r: r["percentile"])
    pgs_ids = [r["pgs_id"] for r in scored]

    lines = [
        f'Interpret these combined Polygenic Risk Score (PRS) results for "{trait}".',
        "",
        f"Models computed: {len(scored)} scored",
        f"PGS IDs: {', '.join(pgs_ids)}",
    ]
    for pid in pgs_ids[:5]:
        lines.append(f"  https://www.pgscatalog.org/score/{pid}/")
    lines.append("")
    lines.append(f"Best model: {best['pgs_id']} (percentile: {best['percentile']:.1f})")
    lines.append(f"Median percentile: {median_pctl:.1f}")

    pct_range = f"{min(pctls):.1f} - {max(pctls):.1f}"
    lines.append(f"Percentile range: {pct_range}")

    with_risk = [r for r in scored if "risk_ratio" in r]
    if with_risk:
        br = max(with_risk, key=lambda r: r["percentile"])
        lines.append(f"Risk vs population average: {br['risk_ratio']:.2f}x")
    with_abs = [r for r in scored if "absolute_risk" in r]
    if with_abs:
        ba = max(with_abs, key=lambda r: r["percentile"])
        ar_str = f"{ba['absolute_risk']:.1%}"
        prev = ba.get("population_prevalence")
        if prev:
            ar_str += f" (pop. avg. {prev:.1%})"
        lines.append(f"Absolute risk (best model): {ar_str}")

    lines.append("")
    lines.append(
        "Methodology: percentiles computed by scoring the 1000 Genomes Project "
        "phase 3 reference panel (2,504 individuals, 5 superpopulations) on "
        f"GRCh38 harmonized scoring files. Ancestry: {ancestry}."
    )
    lines.append("")
    lines.append(
        "Structure your response as: 1) One bold verdict sentence, "
        "2) Model agreement (2-3 bullets), 3) What the percentile means, "
        "4) Confidence assessment, 5) Context and actions if relevant. "
        "Citizen scientist audience. PRS is genetic predisposition, not diagnosis."
    )
    return "\n".join(lines)


def save_trait_report(
    chart: alt.Chart | alt.LayerChart | alt.VConcatChart,
    path: Path,
    trait: str,
    user_results: list[dict],
    ancestry: str = "EUR",
) -> Path:
    """Save an Altair trait chart as a rich HTML report with Key Statistics and AI buttons.

    For .html output, generates a self-contained page with:
    - The Vega-Lite chart
    - Key Statistics panel (percentile, risk vs average, absolute risk)
    - "Ask AI" buttons linking to Claude, ChatGPT, Perplexity with a pre-built prompt

    For non-HTML formats, falls back to save_chart().
    """
    path = Path(path)
    if path.suffix.lower() != ".html":
        return save_chart(chart, path)

    import json as _json
    import urllib.parse

    path.parent.mkdir(parents=True, exist_ok=True)

    scored = [r for r in user_results if r.get("percentile") is not None]
    spec_json = chart.to_json()

    stats_html = ""
    ai_html = ""

    if scored:
        pctls = sorted([r["percentile"] for r in scored])
        median_pctl = pctls[len(pctls) // 2]
        mean_pctl = sum(pctls) / len(pctls)
        best = max(scored, key=lambda r: r["percentile"])

        cards = []
        cards.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">Your Percentile (best model)</div>'
            f'<div class="stat-value">{best["percentile"]:.1f}</div>'
            f'<div class="stat-sub">{best["pgs_id"]}</div></div>'
        )
        cards.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">Median Percentile</div>'
            f'<div class="stat-value">{median_pctl:.1f}</div>'
            f'<div class="stat-sub">across {len(scored)} models</div></div>'
        )
        cards.append(
            f'<div class="stat-card">'
            f'<div class="stat-label">Models</div>'
            f'<div class="stat-value">{len(scored)}</div>'
            f'<div class="stat-sub">scored</div></div>'
        )

        with_risk = [r for r in scored if "risk_ratio" in r]
        if with_risk:
            br = max(with_risk, key=lambda r: r["percentile"])
            rr = br["risk_ratio"]
            color = "#C62828" if rr >= 1.0 else "#2E7D32"
            direction = "higher" if rr >= 1.0 else "lower"
            cards.append(
                f'<div class="stat-card">'
                f'<div class="stat-label">Risk vs Average</div>'
                f'<div class="stat-value" style="color:{color}">{rr:.2f}x</div>'
                f'<div class="stat-sub">{direction} than population</div></div>'
            )

        with_abs = [r for r in scored if "absolute_risk" in r]
        if with_abs:
            ba = max(with_abs, key=lambda r: r["percentile"])
            ar = ba["absolute_risk"]
            prev = ba.get("population_prevalence")
            method = ba.get("risk_method", "")
            sub_parts = []
            if prev:
                sub_parts.append(f"pop. avg. {prev:.1%}")
            if method:
                sub_parts.append(method)
            sub_text = " | ".join(sub_parts) if sub_parts else ""
            cards.append(
                f'<div class="stat-card">'
                f'<div class="stat-label">Absolute Risk</div>'
                f'<div class="stat-value">{ar:.1%}</div>'
                f'<div class="stat-sub">{sub_text}</div></div>'
            )

        if len(pctls) > 1:
            import statistics
            spread = statistics.pstdev(pctls)
            cards.append(
                f'<div class="stat-card">'
                f'<div class="stat-label">Model Spread</div>'
                f'<div class="stat-value">{spread:.1f}</div>'
                f'<div class="stat-sub">percentile SD</div></div>'
            )

        stats_html = '<div class="stats-grid">' + "".join(cards) + "</div>"

        prompt = _build_trait_prompt(trait, user_results, ancestry)
        if prompt:
            encoded = urllib.parse.quote(prompt, safe="")
            btns = []
            for ai in _AI_ASSISTANTS:
                btns.append(
                    f'<a class="ai-btn" style="background:{ai["color"]}" '
                    f'href="{ai["url"]}{encoded}" target="_blank" rel="noopener">'
                    f'Ask {ai["name"]}</a>'
                )
            ai_html = '<div class="ai-buttons"><span class="ai-label">Interpret with AI:</span>' + "".join(btns) + "</div>"

    table_html = ""
    if scored:
        q_colors = {"high": "#2E7D32", "moderate": "#1565C0", "low": "#E65100", "very_low": "#C62828"}
        has_risk = any(r.get("risk_ratio") is not None for r in scored)
        distinct_traits = {r.get("trait_reported", "") for r in scored} - {""}
        has_multi = len(distinct_traits) > 1

        sorted_results = sorted(scored, key=lambda r: r.get("percentile") or 0, reverse=True)
        rows_html = []
        for r in sorted_results:
            pctl = r.get("percentile")
            pctl_s = f"{pctl:.1f}%" if pctl is not None else "—"
            mr = r.get("match_rate")
            match_s = f"{mr:.0%}" if mr is not None else "—"
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
            cells = [f'<td class="id-cell">{r["pgs_id"]}</td>']
            if has_multi:
                trait_raw = r.get("trait_reported", "")
                trait_short = (trait_raw[:29] + "…") if len(trait_raw) > 30 else trait_raw
                cells.append(f'<td class="trait-cell">{trait_short}</td>')
            cells.append(f'<td style="color:{q_col};font-weight:600">{pctl_s}</td>')
            if has_risk:
                rr = r.get("risk_ratio")
                if rr is not None:
                    rc = "#C62828" if rr >= 1.0 else "#2E7D32"
                    cells.append(f'<td style="color:{rc};font-weight:700">{rr:.2f}x</td>')
                else:
                    cells.append("<td>—</td>")
                ar = r.get("absolute_risk")
                prev = r.get("population_prevalence")
                if ar is not None:
                    ar_s = f"{ar:.1%}"
                    if prev is not None:
                        ar_s += f" (avg {prev:.1%})"
                    cells.append(f"<td>{ar_s}</td>")
                else:
                    cells.append("<td>—</td>")
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
        hdr_cells.extend(["<th>Variants</th>", "<th>Match%</th>", "<th>Quality</th>"])

        table_html = (
            '<table class="model-table"><thead><tr>'
            + "".join(hdr_cells)
            + "</tr></thead><tbody>"
            + "".join(rows_html)
            + "</tbody></table>"
        )

    html = f"""<!DOCTYPE html>
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
.q-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head>
<body>
<h1>PRS Report: {trait}</h1>
<div class="subtitle">Ancestry: {SUPERPOP_LABELS.get(ancestry, ancestry)} ({ancestry})</div>
{stats_html}
<div id="vis"></div>
{table_html}
{ai_html}
<script>
vegaEmbed('#vis', {spec_json}, {{actions: true, width: Math.max(800, window.innerWidth - 80)}}).catch(console.error);
</script>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path
