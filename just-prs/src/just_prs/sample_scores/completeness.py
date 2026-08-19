"""Blocking correctness checks for compacted public-sample runtime results."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from just_prs.sample_scores.checkpoints import RUNTIME_KEY, discover_valid_parts
from just_prs.sample_scores.models import (
    EXPECTED_PUBLIC_SAMPLE_ANCESTRY,
    PRIVATE_INGEST_ALIASES,
    RESTORED_PROFILE_ID,
    SCORE_ALGORITHM_VERSION,
    SCORE_PROFILES,
    UNRESTORED_PROFILE_ID,
)
from just_prs.sample_scores.publish import PUBLIC_SAMPLE_SPECS, is_publication_allowed
from just_prs.sample_scores.store import (
    load_runtime_results,
    load_sample_ancestry,
    load_samples,
    runtime_results_path,
    sample_ancestry_path,
)
from just_prs.scoring import parquet_cache_is_readable

MATCH_RATE_TOLERANCE = 1e-6
WEIGHT_COVERAGE_TOLERANCE = 1e-6
_NONNEG_COUNTERS = (
    "variants_matched",
    "variants_total",
    "variants_observed",
    "variants_assumed_hom_ref",
    "variants_unscorable_absent",
    "variants_no_call",
    "variants_maf_filled",
    "variants_ref_resolved_panel",
    "variants_ref_resolved_fasta",
)
_FINITE_FIELDS = (
    "score",
    "match_rate",
    "weight_mass_matched",
    "weight_mass_total",
    "weight_mass_coverage",
    "theoretical_mean",
    "theoretical_std",
)


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


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ok_row_invariant_issues(row: dict[str, object]) -> list[str]:
    """Blocking numeric invariants for a published ``ok`` runtime row.

    ``MATCH_RATE_TOLERANCE`` and ``WEIGHT_COVERAGE_TOLERANCE`` are the documented
    floating-point allowances (1e-6) for ratio vs stored coverage fields.
    """
    issues: list[str] = []
    if str(row.get("status") or "") != "ok":
        return issues
    for name in _FINITE_FIELDS:
        value = _as_float(row.get(name))
        if value is not None and not math.isfinite(value):
            issues.append(f"{name} is not finite")
    if row.get("score") is None:
        issues.append("ok row missing score")
    for name in _NONNEG_COUNTERS:
        value = _as_int(row.get(name))
        if value is not None and value < 0:
            issues.append(f"{name} is negative")
    total = _as_int(row.get("variants_total")) or 0
    matched = _as_int(row.get("variants_matched")) or 0
    if matched > total:
        issues.append("variants_matched exceeds variants_total")
    match_rate = _as_float(row.get("match_rate"))
    if match_rate is not None:
        if match_rate < 0.0 or match_rate > 1.0:
            issues.append("match_rate outside [0,1]")
        if total > 0:
            expected = matched / total
            if abs(match_rate - expected) > MATCH_RATE_TOLERANCE:
                issues.append("match_rate inconsistent with variants_matched/variants_total")
    assumed = _as_int(row.get("variants_assumed_hom_ref")) or 0
    resolved = (
        (_as_int(row.get("variants_ref_resolved_panel")) or 0)
        + (_as_int(row.get("variants_ref_resolved_fasta")) or 0)
    )
    if resolved > assumed:
        issues.append("reference-resolved counts exceed variants_assumed_hom_ref")
    weight_total = _as_float(row.get("weight_mass_total"))
    weight_matched = _as_float(row.get("weight_mass_matched"))
    weight_coverage = _as_float(row.get("weight_mass_coverage"))
    if weight_total is not None and weight_total < 0:
        issues.append("weight_mass_total is negative")
    if weight_matched is not None and weight_matched < 0:
        issues.append("weight_mass_matched is negative")
    if (
        weight_total
        and weight_matched is not None
        and weight_coverage is not None
        and abs(weight_coverage - (weight_matched / weight_total)) > WEIGHT_COVERAGE_TOLERANCE
    ):
        issues.append("weight_mass_coverage inconsistent with matched/total")
    if weight_coverage is not None and (weight_coverage < 0.0 or weight_coverage > 1.0 + WEIGHT_COVERAGE_TOLERANCE):
        issues.append("weight_mass_coverage outside [0,1]")
    return issues


def validate_runtime_results(
    cache_dir: Path,
    *,
    expected_sample_ids: list[str] | None = None,
    expected_pgs_ids: list[str] | None = None,
    expected_checkpoint_keys: set[str] | None = None,
    current_scoring_fingerprints: dict[str, str] | None = None,
    require_ancestry: bool = False,
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
    for sample in samples:
        leaked = [alias for alias in sample.aliases if alias.casefold() in PRIVATE_INGEST_ALIASES]
        if leaked:
            issues.append(f"private ingest alias published for {sample.sample_id}: {leaked}")

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
            n_invalid = 0
            for row in ok.iter_rows(named=True):
                row_issues = ok_row_invariant_issues(row)
                if row_issues:
                    n_invalid += 1
            if n_invalid:
                issues.append(f"{n_invalid} ok rows fail numeric invariants")
            unrestored = {
                str(row["pgs_id"]): str(row["scoring_fingerprint"])
                for row in ok.filter(pl.col("score_profile_id") == UNRESTORED_PROFILE_ID).iter_rows(named=True)
            }
            restored = {
                str(row["pgs_id"]): str(row["scoring_fingerprint"])
                for row in ok.filter(pl.col("score_profile_id") == RESTORED_PROFILE_ID).iter_rows(named=True)
            }
            drifted = sorted(
                pgs_id
                for pgs_id, digest in unrestored.items()
                if pgs_id in restored and digest != restored[pgs_id]
            )
            if drifted:
                issues.append(
                    f"{len(drifted)} PGS IDs have unrestored/restored fingerprint mismatch "
                    f"(examples {drifted[:5]})"
                )
            if current_scoring_fingerprints:
                stale = sorted({
                    str(row["pgs_id"])
                    for row in ok.iter_rows(named=True)
                    if str(row["pgs_id"]) in current_scoring_fingerprints
                    and str(row["scoring_fingerprint"])
                    != current_scoring_fingerprints[str(row["pgs_id"])]
                })
                if stale:
                    issues.append(
                        f"{len(stale)} PGS IDs drifted from the current scoring snapshot "
                        f"(examples {stale[:5]})"
                    )

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
    if expected_pgs_ids is not None and frame.height:
        extra_pgs = sorted(set(frame["pgs_id"].to_list()) - set(expected_pgs_ids))
        if extra_pgs:
            issues.append(
                f"{len(extra_pgs)} withdrawn/unplanned PGS IDs in runtime results "
                f"(examples {extra_pgs[:5]})"
            )
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
        if expected_pgs_ids is None and frame.height != part_rows:
            issues.append(
                f"compaction parity: runtime rows {frame.height} != part rows {part_rows}"
            )

    ancestry_path = sample_ancestry_path(cache_dir)
    ancestry_rows = load_sample_ancestry(cache_dir)
    if require_ancestry and not parquet_cache_is_readable(ancestry_path):
        issues.append("sample_ancestry.parquet missing or unreadable")
    if ancestry_rows or require_ancestry:
        by_sample = {row.sample_id: row for row in ancestry_rows}
        if len(by_sample) != len(ancestry_rows):
            issues.append("sample_ancestry.parquet has duplicate sample_id rows")
        for sample in samples:
            row = by_sample.get(sample.sample_id)
            if row is None:
                issues.append(f"sample_ancestry missing {sample.sample_id}")
                continue
            if row.genotype_sha256_v1 != sample.genotype_sha256_v1:
                issues.append(f"sample_ancestry genotype hash mismatch for {sample.sample_id}")
            if row.superpopulation == "UNKNOWN":
                issues.append(f"sample_ancestry UNKNOWN for {sample.sample_id}")
            expected = EXPECTED_PUBLIC_SAMPLE_ANCESTRY.get(sample.sample_id)
            if expected is not None:
                want_super, want_fine = expected
                if row.superpopulation != want_super or row.confidence < 1.0:
                    issues.append(
                        f"{sample.sample_id} ancestry gate failed: "
                        f"{row.superpopulation}@{row.confidence}"
                    )
                if row.fine_population != want_fine:
                    issues.append(
                        f"{sample.sample_id} fine-population gate failed: "
                        f"{row.fine_population} (expected {want_fine})"
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
    issues = ok_row_invariant_issues(row)
    if issues:
        raise CompletenessError("; ".join(issues))
