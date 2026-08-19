"""Ancestry-selected trait summaries with independent restored/unrestored pairing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from just_prs.sample_scores.integration.models import TraitSummaryRow
from just_prs.sample_scores.integration.pins import SUPERPOPULATIONS
from just_prs.sample_scores.models import RESTORED_PROFILE_ID, UNRESTORED_PROFILE_ID
from just_prs.trait_summary import summarize_trait_rows

ProgressLog = Callable[[str], None]

_RESTORATION_CAVEAT = (
    "The restored profile fills additional absent WGS loci as homozygous "
    "reference from a pinned reference-allele universe. It is not genotype "
    "imputation and does not infer alternate alleles from LD."
)


def _selected_metrics(analysis: pl.DataFrame) -> pl.DataFrame:
    exploded = (
        analysis.drop("source_revision")
        .explode("population_metrics", empty_as_null=True)
        .unnest("population_metrics")
    )
    selected = exploded.filter(pl.col("superpopulation") == pl.col("selected_superpopulation"))
    return selected.select(
        "sample_id",
        "pgs_id",
        "scoring_build",
        "score_profile_id",
        "scoring_fingerprint",
        "analysis_eligible",
        "exclusion_reasons",
        "status",
        "match_rate",
        "weight_mass_coverage",
        "quality_label",
        "selected_superpopulation",
        "or_per_sd",
        "auroc",
        pl.col("percentile").alias("selected_percentile"),
        pl.col("z_score").alias("selected_z"),
        "traits",
    )


def _trait_model_row(row: dict[str, object], trait: dict[str, object], pop: str) -> dict[str, object]:
    risks = trait.get("population_risks") or []
    selected_risk = next((item for item in risks if item.get("superpopulation") == pop), None) or {}
    abs_risk = selected_risk.get("absolute_risk")
    prevalence = selected_risk.get("prevalence")
    h2 = selected_risk.get("h2")
    metrics = []
    if h2 is not None:
        metrics.append({
            "population": selected_risk.get("h2_ancestry") or pop,
            "ancestry": selected_risk.get("h2_ancestry") or pop,
            "h2": h2,
            "source": selected_risk.get("h2_source"),
            "scale": selected_risk.get("h2_scale"),
            "confidence": selected_risk.get("h2_confidence"),
        })
    return {
        "pgs_id": row["pgs_id"],
        "match_rate": row.get("match_rate"),
        "quality_label": row.get("quality_label"),
        "percentile": row.get("selected_percentile"),
        f"pct_{pop}": row.get("selected_percentile"),
        "absolute_risk": abs_risk,
        "absolute_risk_percent": (float(abs_risk) * 100.0) if abs_risk is not None else None,
        "population_prevalence": prevalence,
        "population_average_percent": (float(prevalence) * 100.0) if prevalence is not None else None,
        "risk_ratio": selected_risk.get("risk_ratio"),
        "risk_ratio_value": selected_risk.get("risk_ratio"),
        "heritability_metrics": metrics,
    }


def build_trait_summaries(
    analysis: pl.DataFrame,
    *,
    log: ProgressLog | None = None,
    progress_every: int = 10,
) -> pl.DataFrame:
    """One usable/selected summary per sample×trait×profile×detected superpopulation."""
    slim = _selected_metrics(analysis)
    if slim.height == 0:
        return pl.DataFrame(schema=TraitSummaryRow.model_fields)

    exploded = slim.explode("traits", empty_as_null=True).unnest("traits")
    exploded = exploded.filter(pl.col("trait_id").is_not_null())
    summaries: list[dict[str, object]] = []
    grouped = exploded.group_by(
        "sample_id", "trait_id", "score_profile_id", "selected_superpopulation",
        maintain_order=True,
    )
    total = exploded.select("sample_id", "trait_id", "score_profile_id").n_unique()
    for i, (key, block) in enumerate(grouped, start=1):
        sample_id, trait_id, profile, pop = (str(part) for part in key)
        if pop not in SUPERPOPULATIONS:
            raise ValueError(f"selected population {pop!r} is not a 1000G superpopulation")
        if pop in {"CEU", "IBS"}:
            raise ValueError("closest 1000G cohort cannot be a percentile population")
        all_rows = block.to_dicts()
        eligible_rows = [row for row in all_rows if row.get("analysis_eligible")]
        model_rows = [_trait_model_row(row, row, pop) for row in eligible_rows]
        stats = summarize_trait_rows(
            model_rows,
            model_scope="usable",
            selected_ancestry=pop,
            percentile_source="selected",
            sample_ancestries=[pop],
        )
        n_quarantined = sum(
            1 for row in all_rows if "quarantined" in (row.get("exclusion_reasons") or [])
        )
        n_failed = sum(1 for row in all_rows if row.get("status") == "failed")
        first = all_rows[0]
        summaries.append({
            "sample_id": sample_id,
            "trait_id": trait_id,
            "trait_label": first.get("label"),
            "score_profile_id": profile,
            "percentile_population": pop,
            "model_scope": "usable",
            "percentile_source": "selected",
            "n_total": len(all_rows),
            "n_eligible": len(eligible_rows),
            "n_usable": stats.n_usable,
            "n_percentile_available": len(stats.pct_by_id),
            "n_quarantined": n_quarantined,
            "n_failed": n_failed,
            "n_excluded": len(all_rows) - len(eligible_rows),
            "median_pct": stats.median_pct,
            "mean_pct": stats.mean_pct,
            "std_pct": stats.std_pct,
            "min_pct": stats.min_pct,
            "max_pct": stats.max_pct,
            "spread": stats.spread,
            "outliers": stats.outliers,
            "most_reliable_pgs_id": (stats.best_row or {}).get("pgs_id") if stats.best_row else None,
            "most_reliable_pct": stats.best_model_pctl,
            "absolute_risk": stats.absolute_risk,
            "population_average": stats.population_average,
            "risk_vs_average": stats.risk_vs_average,
            "n_risk_models": stats.n_risk_models,
            "heritability_text": stats.heritability_text,
            "heritability_detail": stats.heritability_detail,
            "prs_actionability_status": first.get("prs_actionability_status") or "not_assessed",
            "condition_actionability_status": first.get("condition_actionability_status") or "not_assessed",
            "context_resolution_status": first.get("context_resolution_status") or "not_assessed",
            "context_classes": first.get("context_classes") or [],
            "paired_profile_id": None,
            "paired_n_usable": None,
            "paired_median_pct": None,
            "delta_median_pct": None,
            "caveats": [_RESTORATION_CAVEAT] if profile == RESTORED_PROFILE_ID else [],
        })
        if log and progress_every and (i % progress_every == 0 or i == total):
            log(f"Trait summaries: {i}/{total} groups ({100.0 * i / total:.1f}%)")

    frame = pl.DataFrame(summaries)
    return _pair_profiles(frame)


def _pair_profiles(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.height == 0:
        return frame
    key = ["sample_id", "trait_id", "percentile_population"]
    unrestored = frame.filter(pl.col("score_profile_id") == UNRESTORED_PROFILE_ID).select(
        *key,
        pl.col("n_usable").alias("pair_n_usable"),
        pl.col("median_pct").alias("pair_median"),
        pl.lit(UNRESTORED_PROFILE_ID).alias("pair_id"),
    )
    restored = frame.filter(pl.col("score_profile_id") == RESTORED_PROFILE_ID).select(
        *key,
        pl.col("n_usable").alias("pair_n_usable"),
        pl.col("median_pct").alias("pair_median"),
        pl.lit(RESTORED_PROFILE_ID).alias("pair_id"),
    )
    with_unrestored = frame.join(restored, on=key, how="left")
    unrestored_side = with_unrestored.filter(pl.col("score_profile_id") == UNRESTORED_PROFILE_ID).with_columns(
        pl.col("pair_id").alias("paired_profile_id"),
        pl.col("pair_n_usable").alias("paired_n_usable"),
        pl.col("pair_median").alias("paired_median_pct"),
        (pl.col("median_pct") - pl.col("pair_median")).alias("delta_median_pct"),
    ).drop(["pair_id", "pair_n_usable", "pair_median"])
    with_restored = frame.join(unrestored, on=key, how="left")
    restored_side = with_restored.filter(pl.col("score_profile_id") == RESTORED_PROFILE_ID).with_columns(
        pl.col("pair_id").alias("paired_profile_id"),
        pl.col("pair_n_usable").alias("paired_n_usable"),
        pl.col("pair_median").alias("paired_median_pct"),
        (pl.col("median_pct") - pl.col("pair_median")).alias("delta_median_pct"),
    ).drop(["pair_id", "pair_n_usable", "pair_median"])
    return pl.concat([unrestored_side, restored_side], how="diagonal_relaxed").sort(
        ["sample_id", "trait_id", "score_profile_id"]
    )
