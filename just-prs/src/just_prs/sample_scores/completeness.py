"""Blocking correctness checks for compacted public-sample runtime results."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from just_prs.sample_scores.checkpoints import RUNTIME_KEY, discover_valid_parts
from just_prs.sample_scores.models import (
    RESTORED_PROFILE_ID,
    SCORE_ALGORITHM_VERSION,
    SCORE_PROFILES,
    UNRESTORED_PROFILE_ID,
)
from just_prs.sample_scores.publish import PUBLIC_SAMPLE_SPECS, is_publication_allowed
from just_prs.sample_scores.store import load_runtime_results, load_samples, runtime_results_path
from just_prs.scoring import parquet_cache_is_readable


class CompletenessError(ValueError):
    """Runtime matrix is incomplete or unsafe to publish."""


class CompletenessReport(BaseModel):
    passed: bool
    n_rows: int = 0
    n_ok: int = 0
    n_failed: int = 0
    n_expected: int = 0
    n_samples: int = 0
    n_pgs_ids: int = 0
    issues: list[str] = Field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise CompletenessError("; ".join(self.issues) or "runtime completeness failed")


_PATH_COLUMNS = ("parquet_path", "vcf_path", "local_path", "source_path")


def _finite_or_null(name: str) -> pl.Expr:
    return pl.col(name).is_null() | pl.col(name).is_finite()


def validate_runtime_results(
    cache_dir: Path,
    *,
    expected_sample_ids: list[str] | None = None,
    expected_pgs_ids: list[str] | None = None,
    expected_checkpoint_keys: set[str] | None = None,
) -> CompletenessReport:
    """Assert the compacted runtime table is a complete, publishable matrix."""
    issues: list[str] = []
    path = runtime_results_path(cache_dir)
    if not parquet_cache_is_readable(path):
        return CompletenessReport(passed=False, issues=["runtime_results.parquet missing or unreadable"])

    frame = load_runtime_results(cache_dir)
    samples = [sample for sample in load_samples(cache_dir) if sample.publication_allowed]
    sample_ids = expected_sample_ids or [sample.sample_id for sample in samples]
    if not sample_ids:
        issues.append("no publication-allowed samples")

    unknown = [sid for sid in sample_ids if not is_publication_allowed(sid)]
    if unknown:
        issues.append(f"unpublished sample ids in expected set: {unknown}")

    extra_samples = sorted(set(frame["sample_id"].to_list()) - set(PUBLIC_SAMPLE_SPECS) - set(sample_ids)) if frame.height else []
    if extra_samples:
        issues.append(f"unknown/private sample_id in runtime outputs: {extra_samples}")

    if frame.height:
        for col in _PATH_COLUMNS:
            if col in frame.columns and frame.filter(pl.col(col).is_not_null()).height:
                issues.append(f"local path column {col} leaked into runtime results")
        # Original private filenames must not appear in published text fields.
        for col in ("error", "trait_reported"):
            if col not in frame.columns:
                continue
            leaked = frame.filter(
                pl.col(col).cast(pl.Utf8).str.contains(r"(?i)ksuhaster|oksana\.vcf|/home/")
            )
            if leaked.height:
                issues.append(f"local path or private filename in {col}")

        dupes = frame.group_by(list(RUNTIME_KEY)).len().filter(pl.col("len") > 1)
        if dupes.height:
            issues.append(f"{dupes.height} duplicate runtime primary keys")

        bad_status = frame.filter(~pl.col("status").is_in(["ok", "failed"]))
        if bad_status.height:
            issues.append(f"{bad_status.height} rows with status not ok/failed")

        ok = frame.filter(pl.col("status") == "ok")
        if ok.height:
            nonfinite = ok.filter(
                ~_finite_or_null("score")
                | ~_finite_or_null("match_rate")
                | ~_finite_or_null("weight_mass_coverage")
            )
            if nonfinite.height:
                issues.append(f"{nonfinite.height} ok rows with non-finite score/counters")
            null_score = ok.filter(pl.col("score").is_null())
            if null_score.height:
                issues.append(f"{null_score.height} ok rows missing score")

        algo = frame.filter(pl.col("score_algorithm_version") != SCORE_ALGORITHM_VERSION)
        if algo.height:
            issues.append("score_algorithm_version mismatch")

        profiles = set(frame["score_profile_id"].unique().to_list())
        unexpected = profiles - set(SCORE_PROFILES)
        if unexpected:
            issues.append(f"unknown score_profile_id values: {sorted(unexpected)}")

    pgs_ids = expected_pgs_ids
    if pgs_ids is None and frame.height:
        pgs_ids = sorted(frame["pgs_id"].unique().to_list())
    pgs_ids = pgs_ids or []
    n_expected = len(sample_ids) * len(pgs_ids) * len(SCORE_PROFILES)
    if pgs_ids and sample_ids:
        expected_pairs = (
            pl.DataFrame({"sample_id": sample_ids})
            .join(pl.DataFrame({"pgs_id": pgs_ids}), how="cross")
            .join(
                pl.DataFrame({
                    "score_profile_id": [UNRESTORED_PROFILE_ID, RESTORED_PROFILE_ID],
                }),
                how="cross",
            )
        )
        if frame.height:
            present = frame.select("sample_id", "pgs_id", "score_profile_id").unique()
            missing = expected_pairs.join(
                present, on=["sample_id", "pgs_id", "score_profile_id"], how="anti"
            )
            if missing.height:
                examples = missing.head(5).to_dicts()
                issues.append(
                    f"{missing.height} missing sample×PGS×profile outcomes; examples={examples}"
                )
        elif n_expected:
            issues.append(f"runtime table empty; expected {n_expected} outcomes")

    if expected_checkpoint_keys is not None:
        discovery = discover_valid_parts(
            cache_dir, validate_parquet=True, expected_keys=expected_checkpoint_keys
        )
        found = {meta.checkpoint_key for meta in discovery.valid}
        missing_keys = expected_checkpoint_keys - found
        if missing_keys:
            issues.append(
                f"{len(missing_keys)} checkpoint keys missing (examples {sorted(missing_keys)[:3]})"
            )
        part_rows = sum(meta.n_rows for meta in discovery.valid)
        if frame.height != part_rows:
            issues.append(
                f"compaction parity: runtime rows {frame.height} != part rows {part_rows}"
            )

    n_ok = int(frame.filter(pl.col("status") == "ok").height) if frame.height else 0
    n_failed = int(frame.filter(pl.col("status") == "failed").height) if frame.height else 0
    return CompletenessReport(
        passed=not issues,
        n_rows=frame.height,
        n_ok=n_ok,
        n_failed=n_failed,
        n_expected=n_expected,
        n_samples=len(sample_ids),
        n_pgs_ids=len(pgs_ids),
        issues=issues,
    )


def assert_finite_coverage(row: dict[str, object]) -> None:
    """Helper for unit tests: coverage counters are internally consistent."""
    total = int(row.get("variants_total") or 0)
    matched = int(row.get("variants_matched") or 0)
    if matched > total:
        raise CompletenessError("variants_matched exceeds variants_total")
    score = row.get("score")
    if score is not None and not math.isfinite(float(score)):
        raise CompletenessError("score is not finite")
