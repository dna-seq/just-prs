"""Build ``model_analysis.parquet`` from pinned local snapshots only."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from just_prs.absolute_risk import estimate_absolute_risk
from just_prs.canary_audit import excluded_pgs_ids
from just_prs.ontology import normalize_trait_id
from just_prs.quality import classify_model_quality, resolve_quality_key
from just_prs.sample_scores.completeness import ok_row_invariant_issues
from just_prs.sample_scores.integration.models import (
    EXCLUSION_ANCESTRY_MISSING,
    EXCLUSION_ANCESTRY_UNKNOWN,
    EXCLUSION_AUDIT_ERROR,
    EXCLUSION_DIST_UNAVAILABLE,
    EXCLUSION_FAILED,
    EXCLUSION_FINGERPRINT,
    EXCLUSION_NOT_PUBLISHED,
    EXCLUSION_QUARANTINED,
    EXCLUSION_RUNTIME_INVARIANT,
    FINE_COHORT_LABELS,
    MODEL_ANALYSIS_PRIMARY_KEY,
)
from just_prs.sample_scores.integration.pins import (
    PERCENTILES_REVISION,
    SAMPLE_SCORES_REVISION,
    SUPERPOPULATIONS,
)
from just_prs.sample_scores.integration.staging import sources_dir, staged_path
from just_prs.sample_scores.models import RESTORED_PROFILE_ID, UNRESTORED_PROFILE_ID
from just_prs.scoring import parquet_cache_is_readable

ProgressLog = Callable[[str], None]


def _read(name: str, cache_dir: Path | None) -> pl.DataFrame:
    path = staged_path(name, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"staged source missing: {name}")
    if path.suffix == ".parquet":
        if not parquet_cache_is_readable(path):
            raise FileNotFoundError(f"unreadable staged parquet: {name}")
        return pl.read_parquet(path)
    raise ValueError(f"not a parquet: {name}")


def _norm_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.vectorize(math.erf)(-values / math.sqrt(2.0)))


def _percentile_from_score(score: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = (score - mean) / std
    pct = np.round(_norm_cdf(z) * 100.0, 2)
    return z, pct


def _normalize_ids(series: pl.Series) -> list[str | None]:
    return [normalize_trait_id(value) for value in series.to_list()]


def _ontology_lookup(table: pl.DataFrame, *, value_cols: list[str]) -> pl.DataFrame:
    """Stack canonical/mapped/raw IDs so trait joins never use labels."""
    frames: list[pl.DataFrame] = []
    for column in ("canonical_efo_id", "mapped_from_id", "efo_id"):
        if column not in table.columns:
            continue
        frames.append(
            table.select(
                pl.col(column).alias("join_id"),
                *[pl.col(name) for name in value_cols if name in table.columns],
            )
        )
    if not frames:
        return pl.DataFrame(schema={"join_id": pl.Utf8})
    stacked = pl.concat(frames, how="diagonal_relaxed").filter(
        pl.col("join_id").is_not_null() & (pl.col("join_id") != "")
    )
    stacked = stacked.with_columns(
        pl.Series("join_id", _normalize_ids(stacked["join_id"]))
    ).filter(pl.col("join_id").is_not_null())
    return stacked.unique(subset=["join_id"], keep="first")


def pgs_without_trustworthy_distribution(
    runtime: pl.DataFrame,
    distributions: pl.DataFrame,
    error_pgs: set[str],
) -> list[str]:
    runtime_ids = set(runtime["pgs_id"].unique().to_list())
    dist_ids = set(distributions["pgs_id"].unique().to_list()) if distributions.height else set()
    missing = sorted((runtime_ids - dist_ids) | (runtime_ids & error_pgs))
    return missing


def build_model_analysis(
    cache_dir: Path | None = None,
    *,
    log: ProgressLog | None = None,
    progress_every: int = 10,
) -> pl.DataFrame:
    """Hydrate every runtime row. Ineligible rows stay for audit."""
    dest_root = sources_dir(cache_dir)
    runtime = _read("runtime_results.parquet", cache_dir)
    samples = _read("samples.parquet", cache_dir)
    ancestry = _read("sample_ancestry.parquet", cache_dir)
    scores = _read("scores.parquet", cache_dir)
    best = _read("best_performance.parquet", cache_dir)
    flags = _read("catalog_scoring_flags.parquet", cache_dir)
    distributions = _read("1000g_distributions.parquet", cache_dir)
    issues = _read("1000g_distribution_quality_issues.parquet", cache_dir)
    links = _read("score_trait_links.parquet", cache_dir)
    traits = _read("traits.parquet", cache_dir)
    actionability = _read("actionability.parquet", cache_dir)
    contexts = _read("trait_contexts.parquet", cache_dir)
    papers = _read("papers.parquet", cache_dir)
    paper_links = _read("score_paper_links.parquet", cache_dir)
    prevalence = _read("trait_prevalence.parquet", cache_dir)
    heritability = _read("trait_heritability.parquet", cache_dir)

    n_before = runtime.height
    if log:
        log(f"Integration build: loaded {n_before} runtime rows")

    excluded = set(excluded_pgs_ids(flags))
    error_pgs = set()
    if issues.height and "severity" in issues.columns:
        error_pgs = set(
            issues.filter(pl.col("severity") == "ERROR")["pgs_id"].unique().to_list()
        )
    warn_pgs = set()
    if issues.height and "severity" in issues.columns:
        warn_pgs = set(
            issues.filter(pl.col("severity") == "WARN")["pgs_id"].unique().to_list()
        )

    unrestored_fp = (
        runtime.filter(pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
        .select("pgs_id", pl.col("scoring_fingerprint").alias("unrestored_fp"))
        .unique()
    )
    restored_fp = (
        runtime.filter(pl.col("score_profile_id") == RESTORED_PROFILE_ID)
        .select("pgs_id", pl.col("scoring_fingerprint").alias("restored_fp"))
        .unique()
    )
    fp_join = unrestored_fp.join(restored_fp, on="pgs_id", how="full", coalesce=True).with_columns(
        (pl.col("unrestored_fp") != pl.col("restored_fp")).fill_null(True).alias("fp_inconsistent")
    )

    invariant_pgs: set[tuple[str, str, str]] = set()
    ok_rows = runtime.filter(pl.col("status") == "ok")
    for row in ok_rows.iter_rows(named=True):
        if ok_row_invariant_issues(row):
            invariant_pgs.add((str(row["sample_id"]), str(row["pgs_id"]), str(row["score_profile_id"])))

    sample_pub = samples.select("sample_id", "publication_allowed")
    ancestry_slim = ancestry.select(
        "sample_id",
        pl.col("superpopulation").alias("selected_superpopulation"),
        pl.col("confidence").alias("selected_superpopulation_confidence"),
        pl.col("fine_population").alias("closest_cohort"),
        pl.col("fine_confidence").alias("closest_cohort_confidence"),
        pl.col("genotype_sha256_v1").alias("ancestry_genotype_sha256"),
    )
    catalog = scores.select(
        "pgs_id",
        pl.col("name").alias("catalog_name"),
        pl.col("trait_reported"),
        pl.col("genome_build").alias("catalog_genome_build"),
        pl.col("n_variants").alias("catalog_n_variants"),
    )
    perf = best.select(
        "pgs_id",
        pl.col("or_estimate").alias("or_per_sd"),
        pl.col("auroc_estimate").alias("auroc"),
    ).unique(subset=["pgs_id"], keep="first")

    dist_ok = distributions.filter(
        pl.col("mean").is_not_null()
        & pl.col("std").is_not_null()
        & pl.col("mean").is_finite()
        & pl.col("std").is_finite()
        & (pl.col("std") > 0)
    )
    wide_parts: list[pl.DataFrame] = [dist_ok.select("pgs_id").unique()]
    for pop in SUPERPOPULATIONS:
        part = dist_ok.filter(pl.col("superpopulation") == pop).select(
            "pgs_id",
            pl.col("mean").alias(f"{pop}_mean"),
            pl.col("std").alias(f"{pop}_std"),
            pl.col("n").alias(f"{pop}_n"),
        )
        wide_parts[0] = wide_parts[0].join(part, on="pgs_id", how="left")
    dist_wide = wide_parts[0]

    frame = (
        runtime.join(sample_pub, on="sample_id", how="left")
        .join(ancestry_slim, on="sample_id", how="left")
        .join(catalog, on="pgs_id", how="left")
        .join(perf, on="pgs_id", how="left")
        .join(fp_join.select("pgs_id", "fp_inconsistent"), on="pgs_id", how="left")
        .join(dist_wide, on="pgs_id", how="left")
    )
    if frame.height != n_before:
        raise RuntimeError(f"join duplicated runtime keys: {n_before} -> {frame.height}")

    scores_np = frame["score"].to_numpy()
    eligible_mask_parts: list[pl.Expr] = []
    pop_z: dict[str, np.ndarray] = {}
    pop_pct: dict[str, np.ndarray] = {}
    for pop in SUPERPOPULATIONS:
        mean = frame[f"{pop}_mean"].to_numpy()
        std = frame[f"{pop}_std"].to_numpy()
        z = np.full(frame.height, np.nan)
        pct = np.full(frame.height, np.nan)
        valid = (
            np.isfinite(scores_np.astype(float, copy=False))
            & np.isfinite(mean.astype(float, copy=False))
            & np.isfinite(std.astype(float, copy=False))
            & (std.astype(float, copy=False) > 0)
        )
        if valid.any():
            z_v, pct_v = _percentile_from_score(
                scores_np[valid].astype(float),
                mean[valid].astype(float),
                std[valid].astype(float),
            )
            z[valid] = z_v
            pct[valid] = pct_v
        pop_z[pop] = z
        pop_pct[pop] = pct
        frame = frame.with_columns(
            pl.Series(f"{pop}_z", z),
            pl.Series(f"{pop}_pct", pct),
        )

    invariant_keys = pl.DataFrame({
        "sample_id": [item[0] for item in invariant_pgs],
        "pgs_id": [item[1] for item in invariant_pgs],
        "score_profile_id": [item[2] for item in invariant_pgs],
        "invariant_failed": [True] * len(invariant_pgs),
    }) if invariant_pgs else pl.DataFrame({
        "sample_id": pl.Series("sample_id", [], dtype=pl.Utf8),
        "pgs_id": pl.Series("pgs_id", [], dtype=pl.Utf8),
        "score_profile_id": pl.Series("score_profile_id", [], dtype=pl.Utf8),
        "invariant_failed": pl.Series("invariant_failed", [], dtype=pl.Boolean),
    })
    frame = frame.join(invariant_keys, on=["sample_id", "pgs_id", "score_profile_id"], how="left")
    if frame.height != n_before:
        raise RuntimeError("invariant join duplicated runtime keys")

    selected = frame["selected_superpopulation"].to_list()
    selected_z = np.full(frame.height, np.nan)
    selected_pct = np.full(frame.height, np.nan)
    selected_available = np.zeros(frame.height, dtype=bool)
    for i, pop in enumerate(selected):
        if pop in pop_z:
            selected_z[i] = pop_z[pop][i]
            selected_pct[i] = pop_pct[pop][i]
            selected_available[i] = np.isfinite(selected_z[i])
    frame = frame.with_columns(
        pl.Series("selected_z", selected_z),
        pl.Series("selected_pct", selected_pct),
        pl.Series("selected_dist_ok", selected_available),
    )

    reasons: list[list[str]] = []
    eligible: list[bool] = []
    for row in frame.iter_rows(named=True):
        row_reasons: list[str] = []
        if str(row.get("status") or "") != "ok":
            row_reasons.append(EXCLUSION_FAILED)
        if row.get("invariant_failed"):
            row_reasons.append(EXCLUSION_RUNTIME_INVARIANT)
        if row.get("fp_inconsistent"):
            row_reasons.append(EXCLUSION_FINGERPRINT)
        if row.get("publication_allowed") is False:
            row_reasons.append(EXCLUSION_NOT_PUBLISHED)
        if str(row["pgs_id"]) in excluded:
            row_reasons.append(EXCLUSION_QUARANTINED)
        if row.get("selected_superpopulation") is None:
            row_reasons.append(EXCLUSION_ANCESTRY_MISSING)
        elif str(row.get("selected_superpopulation")) == "UNKNOWN":
            row_reasons.append(EXCLUSION_ANCESTRY_UNKNOWN)
        if str(row["pgs_id"]) in error_pgs:
            row_reasons.append(EXCLUSION_AUDIT_ERROR)
        if not row.get("selected_dist_ok"):
            row_reasons.append(EXCLUSION_DIST_UNAVAILABLE)
        reasons.append(row_reasons)
        eligible.append(len(row_reasons) == 0)

    frame = frame.with_columns(
        pl.Series("analysis_eligible", eligible),
        pl.Series("exclusion_reasons", reasons),
        pl.lit(SAMPLE_SCORES_REVISION).alias("source_revision"),
    )

    is_harmonized = (
        pl.col("catalog_genome_build").is_not_null()
        & (pl.col("catalog_genome_build") != pl.col("scoring_build"))
    )
    quality_labels: list[str | None] = []
    quality_keys: list[str | None] = []
    for row, is_ok in zip(frame.iter_rows(named=True), eligible, strict=True):
        if not is_ok:
            quality_labels.append(None)
            quality_keys.append(None)
            continue
        cov = row.get("weight_mass_coverage")
        if cov is None:
            cov = row.get("match_rate")
        auroc = row.get("auroc")
        harm = bool(
            row.get("catalog_genome_build")
            and row.get("catalog_genome_build") != row.get("scoring_build")
        )
        label, _color = classify_model_quality(
            float(cov) if cov is not None else 0.0,
            float(auroc) if auroc is not None else None,
            is_harmonized=harm,
        )
        quality_labels.append(label)
        quality_keys.append(resolve_quality_key(label=label, coverage=float(cov) if cov is not None else None, auroc=float(auroc) if auroc is not None else None))
    frame = frame.with_columns(
        pl.Series("quality_label", quality_labels),
        pl.Series("quality_key", quality_keys),
        is_harmonized.alias("is_harmonized"),
        pl.col("closest_cohort").replace_strict(FINE_COHORT_LABELS, default=None).alias("closest_cohort_label"),
    )

    metric_structs: list[pl.Expr] = []
    for pop in SUPERPOPULATIONS:
        available = pl.col("analysis_eligible") & pl.col(f"{pop}_z").is_finite()
        reason = (
            pl.when(~pl.col("analysis_eligible"))
            .then(pl.lit("analysis_ineligible"))
            .when(pl.col(f"{pop}_mean").is_null())
            .then(pl.lit("distribution_missing"))
            .when(~pl.col(f"{pop}_z").is_finite())
            .then(pl.lit("non_finite_statistics"))
            .otherwise(None)
        )
        caveat = pl.when(pl.col("pgs_id").is_in(sorted(warn_pgs))).then(
            pl.lit("reference distribution has WARN-severity audit issues")
        ).otherwise(None)
        metric_structs.append(
            pl.struct(
                pl.lit(pop).alias("superpopulation"),
                pl.when(available).then(pl.col(f"{pop}_mean")).otherwise(None).alias("mean"),
                pl.when(available).then(pl.col(f"{pop}_std")).otherwise(None).alias("std"),
                pl.when(available).then(pl.col(f"{pop}_n")).otherwise(None).alias("n"),
                pl.when(available).then(pl.col(f"{pop}_z")).otherwise(None).alias("z_score"),
                pl.when(available).then(pl.col(f"{pop}_pct")).otherwise(None).alias("percentile"),
                pl.lit("1000g").alias("panel"),
                pl.lit(PERCENTILES_REVISION).alias("source_revision"),
                available.alias("available"),
                reason.alias("exclusion_reason"),
                caveat.alias("caveat"),
            )
        )
    frame = frame.with_columns(pl.concat_list(metric_structs).alias("population_metrics"))

    if log:
        n_ok = int(frame.filter(pl.col("analysis_eligible")).height)
        log(f"Integration build: {n_ok}/{frame.height} analysis_eligible rows")

    papers_nested = _nest_papers(paper_links, papers)
    traits_nested = _nest_traits(
        frame,
        links,
        traits,
        actionability,
        contexts,
        prevalence,
        heritability,
        log=log,
        progress_every=progress_every,
    )
    out = (
        frame.join(papers_nested, on="pgs_id", how="left")
        .join(traits_nested, on=list(MODEL_ANALYSIS_PRIMARY_KEY), how="left")
    )
    if out.height != n_before:
        raise RuntimeError(f"evidence join duplicated runtime keys: {n_before} -> {out.height}")
    out = out.with_columns(
        pl.col("papers").fill_null([]),
        pl.col("traits").fill_null([]),
    )
    keep = [
        *MODEL_ANALYSIS_PRIMARY_KEY,
        "status",
        "error",
        "score",
        "variants_matched",
        "variants_total",
        "match_rate",
        "weight_mass_coverage",
        "sample_genotype_sha256",
        "source_revision",
        "analysis_eligible",
        "exclusion_reasons",
        "publication_allowed",
        "selected_superpopulation",
        "selected_superpopulation_confidence",
        "closest_cohort",
        "closest_cohort_confidence",
        "closest_cohort_label",
        "catalog_name",
        "trait_reported",
        "quality_label",
        "quality_key",
        "is_harmonized",
        "auroc",
        "or_per_sd",
        "population_metrics",
        "traits",
        "papers",
    ]
    result = out.select([col for col in keep if col in out.columns])
    if dest_root.exists() and log:
        missing = pgs_without_trustworthy_distribution(runtime, distributions, error_pgs)
        log(f"PGS without trustworthy selected-pop distribution: {len(missing)}")
    return result


def _nest_papers(paper_links: pl.DataFrame, papers: pl.DataFrame) -> pl.DataFrame:
    slim = papers.select("paper_id", "pmid", "doi", "pgp_id")
    joined = paper_links.join(slim, on="paper_id", how="left")
    return joined.group_by("pgs_id").agg(
        pl.struct(
            "paper_id",
            "relationship_type",
            "pmid",
            "doi",
            "pgp_id",
        ).alias("papers")
    )


def _first_actionability(actionability: pl.DataFrame) -> pl.DataFrame:
    if actionability.height == 0:
        return pl.DataFrame(schema={
            "trait_id": pl.Utf8,
            "prs_actionability_status": pl.Utf8,
            "condition_actionability_status": pl.Utf8,
            "context_resolution_status": pl.Utf8,
        })
    return (
        actionability.sort("guideline_id")
        .unique(subset=["trait_id"], keep="first")
        .select(
            "trait_id",
            "prs_actionability_status",
            "condition_actionability_status",
            "context_resolution_status",
        )
    )


def _nest_traits(
    frame: pl.DataFrame,
    links: pl.DataFrame,
    traits: pl.DataFrame,
    actionability: pl.DataFrame,
    contexts: pl.DataFrame,
    prevalence: pl.DataFrame,
    heritability: pl.DataFrame,
    *,
    log: ProgressLog | None,
    progress_every: int,
) -> pl.DataFrame:
    action = _first_actionability(actionability)
    context_lists = (
        contexts.group_by("trait_id").agg(pl.col("context_class").unique().alias("context_classes"))
        if contexts.height
        else pl.DataFrame({"trait_id": [], "context_classes": []})
    )
    trait_meta = (
        links.join(traits.select("trait_id", "label"), on="trait_id", how="left")
        .join(action, on="trait_id", how="left")
        .join(context_lists, on="trait_id", how="left")
    )
    keys = frame.select(
        *MODEL_ANALYSIS_PRIMARY_KEY,
        "analysis_eligible",
        "or_per_sd",
        "auroc",
        *[f"{pop}_z" for pop in SUPERPOPULATIONS],
        *[f"{pop}_pct" for pop in SUPERPOPULATIONS],
    )
    exploded = keys.join(trait_meta, on="pgs_id", how="left")
    exploded = exploded.with_columns(
        pl.Series("trait_id_norm", _normalize_ids(exploded["trait_id"]))
        if exploded.height and "trait_id" in exploded.columns
        else pl.lit(None).alias("trait_id_norm")
    )
    prev_lookup = _ontology_lookup(
        prevalence,
        value_cols=["prevalence", "prevalence_type", "source", "confidence", "ancestry"],
    )
    if prev_lookup.height and "prevalence" in prev_lookup.columns:
        prev_lookup = prev_lookup.rename({
            "prevalence": "trait_prevalence",
            "source": "prevalence_source",
        })
        keep_prev = [
            col for col in ("join_id", "trait_prevalence", "prevalence_source", "prevalence_type")
            if col in prev_lookup.columns
        ]
        prev_lookup = prev_lookup.select(keep_prev)
    h2_lookup = _ontology_lookup(
        heritability,
        value_cols=["h2_liability", "h2_observed", "source", "ancestry", "confidence", "method"],
    )
    if h2_lookup.height:
        rename_h2 = {
            name: alias
            for name, alias in (
                ("source", "h2_source"),
                ("ancestry", "h2_ancestry"),
                ("confidence", "h2_confidence"),
                ("method", "h2_method"),
            )
            if name in h2_lookup.columns
        }
        if rename_h2:
            h2_lookup = h2_lookup.rename(rename_h2)
    before_onto = exploded.height
    if "join_id" in prev_lookup.columns and prev_lookup.height:
        exploded = exploded.join(prev_lookup, left_on="trait_id_norm", right_on="join_id", how="left")
    else:
        exploded = exploded.with_columns(
            pl.lit(None).cast(pl.Float64).alias("trait_prevalence"),
            pl.lit(None).alias("prevalence_source"),
            pl.lit(None).alias("prevalence_type"),
        )
    if "join_id" in h2_lookup.columns and h2_lookup.height:
        exploded = exploded.join(h2_lookup, left_on="trait_id_norm", right_on="join_id", how="left")
    if exploded.height != before_onto:
        raise RuntimeError("ontology join duplicated trait rows")

    risk_rows = _compute_trait_population_risks(exploded, log=log, progress_every=progress_every)
    trait_rows = risk_rows.rename({"population_risk": "population_risks"})
    return trait_rows.group_by(list(MODEL_ANALYSIS_PRIMARY_KEY)).agg(
        pl.struct(
            "trait_id",
            "label",
            "relationship_source",
            "trait_reported",
            "prs_actionability_status",
            "condition_actionability_status",
            "context_resolution_status",
            "context_classes",
            "population_risks",
        ).alias("traits")
    )


def _estimate_risk_table(keys: pl.DataFrame, *, log: ProgressLog | None, progress_every: int) -> pl.DataFrame:
    """Call ``estimate_absolute_risk`` once per unique (z, prevalence, OR, AUROC)."""
    if keys.height == 0:
        return keys.with_columns(
            pl.lit(None).cast(pl.Float64).alias("absolute_risk"),
            pl.lit(None).cast(pl.Float64).alias("risk_ratio"),
            pl.lit(None).cast(pl.Utf8).alias("risk_method"),
            pl.lit(None).cast(pl.Utf8).alias("unavailable_reason"),
        )
    abs_risk: list[float | None] = []
    ratios: list[float | None] = []
    methods: list[str | None] = []
    reasons: list[str | None] = []
    total = keys.height
    for i, row in enumerate(keys.iter_rows(named=True), start=1):
        eligible = bool(row.get("analysis_eligible"))
        z = row.get("z_score")
        prev = row.get("trait_prevalence")
        or_est = row.get("or_per_sd")
        auroc = row.get("auroc")
        if not eligible or z is None or not math.isfinite(float(z)):
            abs_risk.append(None)
            ratios.append(None)
            methods.append(None)
            reasons.append("ineligible_or_no_z")
        elif prev is None:
            abs_risk.append(None)
            ratios.append(None)
            methods.append(None)
            reasons.append("prevalence_unavailable")
        else:
            estimate = estimate_absolute_risk(
                z_score=float(z),
                prevalence=float(prev),
                or_estimate=float(or_est) if or_est is not None else None,
                auroc_estimate=float(auroc) if auroc is not None else None,
                prevalence_source=str(row.get("prevalence_source") or ""),
                prevalence_type=str(row.get("prevalence_type") or "lifetime"),
            )
            if estimate is None:
                abs_risk.append(None)
                ratios.append(None)
                methods.append(None)
                reasons.append("effect_size_unavailable")
            else:
                abs_risk.append(estimate.absolute_risk)
                ratios.append(estimate.risk_ratio)
                methods.append(estimate.method)
                reasons.append(None)
        if log and progress_every and i % max(progress_every * 200, 1) == 0:
            log(f"Integration trait-risk: {i}/{total} unique z/prevalence keys")
    return keys.with_columns(
        pl.Series("absolute_risk", abs_risk),
        pl.Series("risk_ratio", ratios),
        pl.Series("risk_method", methods),
        pl.Series("unavailable_reason", reasons),
    )


def _compute_trait_population_risks(
    exploded: pl.DataFrame,
    *,
    log: ProgressLog | None,
    progress_every: int,
) -> pl.DataFrame:
    if exploded.height == 0 or "trait_id" not in exploded.columns:
        return exploded.with_columns(pl.lit(None).alias("population_risk"))
    for name, dtype in (
        ("trait_prevalence", pl.Float64),
        ("prevalence_source", pl.Utf8),
        ("prevalence_type", pl.Utf8),
        ("h2_liability", pl.Float64),
        ("h2_observed", pl.Float64),
        ("h2_source", pl.Utf8),
        ("h2_ancestry", pl.Utf8),
        ("h2_confidence", pl.Utf8),
        ("or_per_sd", pl.Float64),
        ("auroc", pl.Float64),
        ("label", pl.Utf8),
        ("relationship_source", pl.Utf8),
        ("trait_reported", pl.Utf8),
        ("prs_actionability_status", pl.Utf8),
        ("condition_actionability_status", pl.Utf8),
        ("context_resolution_status", pl.Utf8),
    ):
        if name not in exploded.columns:
            exploded = exploded.with_columns(pl.lit(None).cast(dtype).alias(name))
    z_cols = [f"{pop}_z" for pop in SUPERPOPULATIONS]
    index_cols = [col for col in exploded.columns if col not in z_cols]
    long = exploded.unpivot(
        index=index_cols,
        on=z_cols,
        variable_name="z_name",
        value_name="z_score",
    ).with_columns(
        pl.col("z_name").str.strip_suffix("_z").alias("superpopulation"),
    ).with_columns(
        pl.col("superpopulation").replace_strict(
            {pop: idx for idx, pop in enumerate(SUPERPOPULATIONS)},
            default=99,
        ).alias("_pop_ord"),
        pl.concat_str(
            [
                pl.col("analysis_eligible").cast(pl.Utf8).fill_null("none"),
                pl.col("z_score").round(10).cast(pl.Utf8).fill_null("na"),
                pl.col("trait_prevalence").round(10).cast(pl.Utf8).fill_null("na"),
                pl.col("or_per_sd").round(10).cast(pl.Utf8).fill_null("na"),
                pl.col("auroc").round(10).cast(pl.Utf8).fill_null("na"),
                pl.col("prevalence_source").fill_null(""),
                pl.col("prevalence_type").fill_null(""),
            ],
            separator="|",
        ).alias("_risk_key"),
    )
    risk_keys = long.select(
        "_risk_key",
        "analysis_eligible",
        "z_score",
        "trait_prevalence",
        "or_per_sd",
        "auroc",
        "prevalence_source",
        "prevalence_type",
    ).unique(subset=["_risk_key"], keep="first")
    estimated = _estimate_risk_table(risk_keys, log=log, progress_every=progress_every)
    long = long.join(
        estimated.select(
            "_risk_key",
            "absolute_risk",
            "risk_ratio",
            "risk_method",
            "unavailable_reason",
        ),
        on="_risk_key",
        how="left",
    )
    if long.height != exploded.height * len(SUPERPOPULATIONS):
        raise RuntimeError("risk unpivot/join changed row cardinality")
    long = long.with_columns(
        pl.when(pl.col("h2_liability").is_not_null())
        .then(pl.col("h2_liability"))
        .otherwise(pl.col("h2_observed"))
        .alias("h2"),
        pl.when(pl.col("h2_liability").is_not_null())
        .then(pl.lit("liability"))
        .when(pl.col("h2_observed").is_not_null())
        .then(pl.lit("observed"))
        .otherwise(None)
        .alias("h2_scale"),
    ).with_columns(
        pl.struct(
            "superpopulation",
            pl.col("trait_prevalence").alias("prevalence"),
            "prevalence_source",
            "prevalence_type",
            "absolute_risk",
            "risk_ratio",
            "risk_method",
            "h2",
            "h2_source",
            "h2_scale",
            "h2_ancestry",
            "h2_confidence",
            "unavailable_reason",
        ).alias("population_risk_item")
    ).sort([*MODEL_ANALYSIS_PRIMARY_KEY, "trait_id", "_pop_ord"])
    group_cols = [*MODEL_ANALYSIS_PRIMARY_KEY, "trait_id"]
    return long.group_by(group_cols, maintain_order=True).agg(
        pl.col("label").first(),
        pl.col("relationship_source").first(),
        pl.col("trait_reported").first(),
        pl.col("prs_actionability_status").first(),
        pl.col("condition_actionability_status").first(),
        pl.col("context_resolution_status").first(),
        pl.col("context_classes").first(),
        pl.col("population_risk_item").alias("population_risk"),
    )
