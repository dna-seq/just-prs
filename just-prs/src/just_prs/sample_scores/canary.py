"""Derive public canary rows from unrestored runtime scores.

Public genomes are scored once under ``grch38-wgs-pass-unrestored-v1``
(``public-wgs-pass-v1`` normalization). Canary flags reuse those raw scores
and attach percentiles in this module — never inside a scoring worker, and
never by converting thin canary rows into ``RuntimeResultRow``.

Unknown/private ``--vcf`` labels keep a separate non-publishing scorer.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import polars as pl

from just_prs.canary_audit import (
    CANARY_RESULT_SCHEMA,
    CanarySample,
    CanaryScoreProgress,
    canary_scores_path,
)
from just_prs.sample_scores.checkpoints import write_parquet_atomic
from just_prs.sample_scores.models import UNRESTORED_PROFILE_ID
from just_prs.sample_scores.publish import is_publication_allowed, public_sample_spec
from just_prs.sample_scores.store import load_runtime_results


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (matches ``just_prs.prs._norm_cdf``)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_cdf_expr(z: pl.Expr) -> pl.Expr:
    return z.map_elements(_norm_cdf, return_dtype=pl.Float64)


def attach_percentiles(
    rows: pl.DataFrame,
    distributions: pl.DataFrame,
    *,
    ancestry: str = "EUR",
) -> pl.DataFrame:
    """Join 1000G mean/std once and compute percentile + z. No per-row catalog calls."""
    if rows.height == 0:
        return pl.DataFrame(schema=CANARY_RESULT_SCHEMA)
    stats = (
        distributions
        .filter(pl.col("superpopulation") == ancestry.upper())
        .select("pgs_id", "mean", "std")
        .unique(subset=["pgs_id"])
    )
    joined = rows.join(stats, on="pgs_id", how="left")
    z = (pl.col("score") - pl.col("mean")) / pl.col("std")
    annotated = joined.with_columns(
        pl.when(pl.col("std").is_not_null() & (pl.col("std") > 0) & pl.col("score").is_not_null())
        .then(z)
        .otherwise(None)
        .alias("z_score"),
    ).with_columns(
        pl.when(pl.col("z_score").is_not_null())
        .then((_norm_cdf_expr(pl.col("z_score")) * 100.0).round(2))
        .otherwise(None)
        .alias("percentile"),
    )
    return annotated.select(list(CANARY_RESULT_SCHEMA))


def canary_rows_from_unrestored_runtime(
    cache_dir: Path,
    *,
    sample_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Thin canary rows from unrestored public runtime scores (no percentiles yet)."""
    runtime = load_runtime_results(cache_dir)
    if runtime.is_empty():
        return pl.DataFrame(schema=CANARY_RESULT_SCHEMA)
    filtered = runtime.filter(
        (pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
        & (pl.col("status") == "ok")
        & pl.col("score").is_not_null()
    )
    if sample_ids is not None:
        filtered = filtered.filter(pl.col("sample_id").is_in(sample_ids))
    if filtered.is_empty():
        return pl.DataFrame(schema=CANARY_RESULT_SCHEMA)
    return filtered.select(
        pl.col("pgs_id"),
        pl.col("sample_id"),
        pl.lit(None).cast(pl.Float64).alias("percentile"),
        pl.lit(None).cast(pl.Float64).alias("z_score"),
        pl.col("match_rate"),
        pl.col("score"),
    )


def public_canary_labels(samples: list[CanarySample]) -> list[str]:
    """Canonical public sample_ids present in this canary run."""
    labels: list[str] = []
    for sample in samples:
        spec = public_sample_spec(sample.label)
        if spec is not None and spec.publication_allowed:
            labels.append(spec.sample_id)
    return labels


def private_canary_samples(samples: list[CanarySample]) -> list[CanarySample]:
    return [sample for sample in samples if not is_publication_allowed(sample.label)]


def score_private_canary_catalog(
    samples: list[CanarySample],
    cache_dir: Path,
    *,
    pgs_ids: list[str] | None = None,
    limit: int | None = None,
    skip_existing: bool = True,
    progress_every: int = 10,
    log: Callable[[str], None] | None = None,
) -> tuple[pl.DataFrame, CanaryScoreProgress]:
    """Low-level scorer for unknown/private canary inputs only.

    Does not publish, does not write ``RuntimeResultRow``, and does not load
    1000G distributions. Percentiles are attached later by the canary asset.
    """
    from just_prs.canary_audit import score_canary_catalog

    private = private_canary_samples(samples)
    if not private:
        return pl.DataFrame(schema=CANARY_RESULT_SCHEMA), CanaryScoreProgress()
    rows, progress = score_canary_catalog(
        private,
        cache_dir,
        pgs_ids=pgs_ids,
        limit=limit,
        skip_existing=skip_existing,
        progress_every=progress_every,
        log=log,
    )
    # score_canary_catalog already attaches percentiles. Keep raw score/match
    # and let the caller re-attach against the current distributions so a
    # stale canary cache cannot freeze z/percentile.
    raw = rows.select("pgs_id", "sample_id", "match_rate", "score")
    return raw, progress


def compact_canary_scores(
    public_rows: pl.DataFrame,
    private_rows: pl.DataFrame,
    distributions: pl.DataFrame,
    cache_dir: Path,
    *,
    ancestry: str = "EUR",
) -> pl.DataFrame:
    """Percentile-annotate, unique by (pgs_id, sample_id), atomically replace."""
    combined = pl.concat(
        [public_rows, private_rows],
        how="diagonal_relaxed",
    )
    if combined.height:
        combined = combined.unique(subset=["pgs_id", "sample_id"], keep="last")
    annotated = attach_percentiles(combined, distributions, ancestry=ancestry)
    dest = canary_scores_path(cache_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(annotated, dest)
    return annotated
