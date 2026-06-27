"""Dagster asset checks for PRS pipeline data quality.

These checks run after asset materialization and validate invariants that
the pipeline code itself cannot guarantee (e.g., data consistency between
raw scores and aggregated distributions, absence of inf/nan, statistical
plausibility of distributions).

Asset checks surface in the Dagster UI alongside each asset and can be
configured as ``blocking=True`` to prevent downstream assets from executing
when data quality is compromised.
"""

import math
import os

import polars as pl
from dagster import AssetCheckResult, AssetCheckSeverity, asset_check

from just_prs.reference import reference_distribution_audit_issues
from prs_pipeline.resources import CacheDirResource


# ---------------------------------------------------------------------------
# reference_scores — distribution quality checks
# ---------------------------------------------------------------------------


@asset_check(
    asset="reference_scores",
    description=(
        "Verify that every PGS ID in the distributions has exactly 5 rows "
        "(one per superpopulation: AFR, AMR, EAS, EUR, SAS)."
    ),
)
def check_distributions_superpop_completeness(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
    subdir = "test" if test_spec else None
    dist_path = _distributions_path(cache_dir_resource, panel, subdir)

    if not dist_path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": "Distributions parquet not found", "path": str(dist_path)},
        )

    df = pl.read_parquet(dist_path)
    counts = df.group_by("pgs_id").agg(pl.len().alias("count"))
    non_five = counts.filter(pl.col("count") != 5)

    return AssetCheckResult(
        passed=non_five.height == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "n_pgs_ids": int(df["pgs_id"].n_unique()),
            "n_with_incomplete_superpops": non_five.height,
            "examples": non_five.head(5).to_dicts() if non_five.height > 0 else [],
        },
    )


@asset_check(
    asset="reference_scores",
    description=(
        "Detect inf, NaN, or zero-std values in the distributions. "
        "These indicate corrupted scoring results or edge-case PGS IDs "
        "where an entire superpopulation has identical scores."
    ),
)
def check_distributions_no_inf_nan(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
    subdir = "test" if test_spec else None
    dist_path = _distributions_path(cache_dir_resource, panel, subdir)

    if not dist_path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": "Distributions parquet not found"},
        )

    df = pl.read_parquet(dist_path)
    quality_path = dist_path.with_name(f"{panel}_quality.parquet")
    quality_df = pl.read_parquet(quality_path) if quality_path.exists() else None
    issue_df = reference_distribution_audit_issues(df, quality_df)
    issue_path = dist_path.with_name(f"{panel}_distribution_quality_issues.parquet")
    issue_df.write_parquet(issue_path)

    n_inf = df.filter(
        pl.col("mean").is_infinite() | pl.col("std").is_infinite()
    ).height
    n_nan = df.filter(pl.col("mean").is_nan() | pl.col("std").is_nan()).height
    n_zero_std = issue_df.filter(pl.col("issue") == "std_zero").height

    n_problematic = n_inf + n_nan + n_zero_std
    passed = n_problematic == 0
    severity = AssetCheckSeverity.WARN if n_zero_std > 0 and n_inf + n_nan == 0 else AssetCheckSeverity.ERROR

    return AssetCheckResult(
        passed=passed,
        severity=severity if not passed else AssetCheckSeverity.WARN,
        metadata={
            "n_total_rows": df.height,
            "n_inf": n_inf,
            "n_nan": n_nan,
            "n_zero_std": n_zero_std,
            "n_problematic_total": n_problematic,
            "issue_report_path": str(issue_path),
            "quality_report_path": str(quality_path) if quality_path.exists() else "",
            "issue_examples": issue_df.head(20).to_dicts() if issue_df.height > 0 else [],
        },
    )


@asset_check(
    asset="reference_scores",
    description=(
        "Verify the quantile ordering invariant: p5 <= p25 <= median <= p75 <= p95 "
        "for all rows. Violations indicate a bug in aggregate_distributions()."
    ),
)
def check_distributions_quantile_ordering(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
    subdir = "test" if test_spec else None
    dist_path = _distributions_path(cache_dir_resource, panel, subdir)

    if not dist_path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": "Distributions parquet not found"},
        )

    df = pl.read_parquet(dist_path)
    violated = df.filter(
        (pl.col("p5") > pl.col("p25"))
        | (pl.col("p25") > pl.col("median"))
        | (pl.col("median") > pl.col("p75"))
        | (pl.col("p75") > pl.col("p95"))
    )

    return AssetCheckResult(
        passed=violated.height == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "n_total_rows": df.height,
            "n_violations": violated.height,
            "examples": violated.select("pgs_id", "superpopulation", "p5", "p25", "median", "p75", "p95").head(5).to_dicts() if violated.height > 0 else [],
        },
    )


@asset_check(
    asset="reference_scores",
    description=(
        "Spot-check that distributions match re-aggregation from raw per-sample "
        "scores. Catches stale distributions (e.g., distributions generated from "
        "a different scoring engine version). Samples up to 20 PGS IDs."
    ),
)
def check_distributions_vs_raw_scores(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    import random

    from just_prs.reference import aggregate_distributions

    panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
    subdir = "test" if test_spec else None
    cache_dir = cache_dir_resource.get_path()
    dist_path = _distributions_path(cache_dir_resource, panel, subdir)
    ref_scores_dir = cache_dir / "reference_scores" / panel

    if not dist_path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": "Distributions parquet not found"},
        )

    dist_df = pl.read_parquet(dist_path)
    all_ids = dist_df["pgs_id"].unique().to_list()
    available_ids = [
        pid for pid in all_ids
        if (ref_scores_dir / pid / "scores.parquet").exists()
    ]

    if not available_ids:
        return AssetCheckResult(
            passed=True,
            severity=AssetCheckSeverity.WARN,
            metadata={"warning": "No raw score files found for spot-checking"},
        )

    rng = random.Random(42)
    sample_ids = rng.sample(available_ids, min(20, len(available_ids)))

    stale_count = 0
    stale_examples: list[dict[str, object]] = []

    for pgs_id in sample_ids:
        raw = pl.read_parquet(ref_scores_dir / pgs_id / "scores.parquet")
        fresh_dist = aggregate_distributions(raw)

        for sp in ("AFR", "AMR", "EAS", "EUR", "SAS"):
            f_row = fresh_dist.filter(pl.col("superpopulation") == sp)
            s_row = dist_df.filter(
                (pl.col("pgs_id") == pgs_id) & (pl.col("superpopulation") == sp)
            )
            if f_row.height == 0 or s_row.height == 0:
                continue

            f_mean = float(f_row["mean"][0])
            s_mean = float(s_row["mean"][0])

            if not math.isfinite(f_mean) or not math.isfinite(s_mean):
                stale_count += 1
                stale_examples.append({"pgs_id": pgs_id, "sp": sp, "issue": "non-finite"})
                continue

            if abs(f_mean) > 1e-15:
                rel_err = abs(f_mean - s_mean) / abs(f_mean)
                if rel_err > 0.01:
                    stale_count += 1
                    if len(stale_examples) < 5:
                        stale_examples.append({
                            "pgs_id": pgs_id,
                            "sp": sp,
                            "stored_mean": round(s_mean, 8),
                            "fresh_mean": round(f_mean, 8),
                            "rel_err": round(rel_err, 4),
                        })

    return AssetCheckResult(
        passed=stale_count == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "n_sampled": len(sample_ids),
            "n_stale": stale_count,
            "stale_examples": stale_examples,
        },
    )


@asset_check(
    asset="reference_scores",
    description=(
        "Verify sample sizes are within expected 1000G panel ranges "
        "(490-893 per superpopulation). Unexpected sizes indicate a "
        "panel mismatch or corrupted psam."
    ),
)
def check_distributions_sample_sizes(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
    subdir = "test" if test_spec else None
    dist_path = _distributions_path(cache_dir_resource, panel, subdir)

    if not dist_path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": "Distributions parquet not found"},
        )

    df = pl.read_parquet(dist_path)
    min_n = int(df["n"].min())
    max_n = int(df["n"].max())

    passed = min_n >= 400 and max_n <= 1000

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "min_n": min_n,
            "max_n": max_n,
            "expected_min": 400,
            "expected_max": 1000,
        },
    )


# ---------------------------------------------------------------------------
# hf_prs_percentiles — enriched distributions checks
# ---------------------------------------------------------------------------


@asset_check(
    asset="hf_prs_percentiles",
    description=(
        "Verify the enriched distributions parquet has all expected metadata "
        "columns joined from the cleaned PGS Catalog data."
    ),
)
def check_enriched_has_metadata_columns(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
    test_spec = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
    subdir = "test" if test_spec else None
    dist_path = _distributions_path(cache_dir_resource, panel, subdir)

    if not dist_path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": "Distributions parquet not found"},
        )

    df = pl.read_parquet(dist_path, n_rows=0)
    cols = set(df.columns)

    required_stats = {"pgs_id", "superpopulation", "mean", "std", "n", "median", "p5", "p25", "p75", "p95"}
    expected_metadata = {"name", "trait_reported", "genome_build"}

    missing_stats = required_stats - cols
    missing_metadata = expected_metadata - cols

    passed = len(missing_stats) == 0 and len(missing_metadata) == 0

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR if missing_stats else AssetCheckSeverity.WARN,
        metadata={
            "columns_found": sorted(cols),
            "missing_stats_columns": sorted(missing_stats),
            "missing_metadata_columns": sorted(missing_metadata),
        },
    )


# ---------------------------------------------------------------------------
# cleaned_pgs_metadata — metadata quality checks
# ---------------------------------------------------------------------------


@asset_check(
    asset="cleaned_pgs_metadata",
    description=(
        "Verify cleaned metadata has at least one row per PGS ID in scores.parquet, "
        "and that genome_build values are normalized (only GRCh37, GRCh38, GRCh36, NR)."
    ),
)
def check_cleaned_metadata_quality(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    cache_dir = cache_dir_resource.get_path()
    scores_path = cache_dir / "metadata" / "scores.parquet"
    best_path = cache_dir / "metadata" / "best_performance.parquet"

    if not scores_path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": "scores.parquet not found"},
        )

    scores = pl.read_parquet(scores_path)
    n_scores = scores.height
    n_unique_ids = scores["pgs_id"].n_unique()

    builds = set(scores["genome_build"].unique().to_list()) if "genome_build" in scores.columns else set()
    valid_builds = {"GRCh37", "GRCh38", "GRCh36", "NR", None}
    invalid_builds = builds - valid_builds

    has_best_perf = best_path.exists()
    n_best_perf = 0
    if has_best_perf:
        best = pl.read_parquet(best_path)
        n_best_perf = best.height

    passed = n_scores > 0 and len(invalid_builds) == 0

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "n_scores": n_scores,
            "n_unique_pgs_ids": n_unique_ids,
            "genome_builds": sorted(str(b) for b in builds),
            "invalid_builds": sorted(str(b) for b in invalid_builds),
            "has_best_performance": has_best_perf,
            "n_best_performance": n_best_perf,
        },
    )


# ---------------------------------------------------------------------------
# chip_coverage — consumer-chip coverage quality check
# ---------------------------------------------------------------------------


@asset_check(
    asset="chip_coverage",
    description=(
        "Verify chip coverage ratios are in [0, 1], n_typed <= n_total, every chip "
        "covers the same set of PGS IDs, and the table is non-empty."
    ),
)
def check_chip_coverage_valid(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    cache_dir = cache_dir_resource.get_path()
    path = cache_dir / "percentiles" / "chip_coverage.parquet"
    if not path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": "chip_coverage.parquet not found"},
        )

    df = pl.read_parquet(path)
    n_bad_ratio = df.filter(
        (pl.col("coverage_ratio") < 0.0) | (pl.col("coverage_ratio") > 1.0)
    ).height
    n_bad_counts = df.filter(pl.col("n_typed") > pl.col("n_total")).height

    # Every chip must cover the same set of PGS IDs.
    ids_per_chip = df.group_by("chip").agg(pl.col("pgs_id").n_unique().alias("n_ids"))
    n_id_sets = ids_per_chip["n_ids"].n_unique()
    chips_consistent = n_id_sets <= 1

    passed = (
        df.height > 0
        and n_bad_ratio == 0
        and n_bad_counts == 0
        and chips_consistent
    )
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "n_rows": df.height,
            "n_chips": df["chip"].n_unique() if df.height > 0 else 0,
            "n_scores": df["pgs_id"].n_unique() if df.height > 0 else 0,
            "n_bad_ratio": n_bad_ratio,
            "n_typed_gt_total": n_bad_counts,
            "chips_cover_same_ids": chips_consistent,
            "median_coverage_ratio": round(float(df["coverage_ratio"].median()), 4) if df.height > 0 else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _distributions_path(
    cache_dir_resource: CacheDirResource,
    panel: str,
    subdir: str | None,
) -> "Path":
    from pathlib import Path
    cache_dir = cache_dir_resource.get_path()
    percentiles_dir = cache_dir / "percentiles"
    if subdir:
        percentiles_dir = percentiles_dir / subdir
    return percentiles_dir / f"{panel}_distributions.parquet"


# ---------------------------------------------------------------------------
# Collect all checks for registration
# ---------------------------------------------------------------------------


@asset_check(asset="ld_proxy_table", description="Validate LD-proxy table schema and r² range.")
def check_ld_proxy_table_valid(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    from just_prs.ftp import list_all_pgs_ids
    from just_prs.ld_proxy import LD_PROXY_TABLE_COLUMNS, ld_proxy_pgs_path

    cache_dir = cache_dir_resource.get_path()
    panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")

    combos = [("gsa_v3", "GRCh38")]
    total_rows = 0
    issues: list[str] = []

    for chip, build in combos:
        pgs_ids_spec = os.environ.get("PRS_LD_PGS_IDS", "").strip()
        limit_targets = os.environ.get("PRS_LD_LIMIT_TARGETS", "").strip()
        if pgs_ids_spec:
            pgs_ids = [pid.strip().upper() for pid in pgs_ids_spec.split(",") if pid.strip()]
        elif limit_targets:
            pgs_ids = list_all_pgs_ids()[:int(limit_targets)]
        elif os.environ.get("PRS_LD_FULL_CATALOG", "").strip().lower() in {"1", "true", "yes"}:
            pgs_ids = list_all_pgs_ids()
        else:
            issues.append("No LD proxy scope set for validation.")
            pgs_ids = []

        for pgs_id in pgs_ids:
            path = ld_proxy_pgs_path(cache_dir, pgs_id, chip, build, panel)
            if not path.exists():
                issues.append(f"Missing: {pgs_id} ({path.name})")
                continue

            df = pl.read_parquet(path)
            missing_cols = set(LD_PROXY_TABLE_COLUMNS) - set(df.columns)
            if missing_cols:
                issues.append(f"{pgs_id}: missing columns {missing_cols}")
                continue
            if df.height == 0:
                issues.append(f"{pgs_id}: empty LD proxy table")
                continue

            n_bad_r2 = df.filter(
                (pl.col("r_squared") < 0.0) | (pl.col("r_squared") > 1.0)
            ).height
            if n_bad_r2 > 0:
                issues.append(f"{pgs_id}: {n_bad_r2} rows with r² outside [0,1]")

            n_bad_signed = df.filter(
                (pl.col("r_signed") < -1.0) | (pl.col("r_signed") > 1.0)
            ).height
            if n_bad_signed > 0:
                issues.append(f"{pgs_id}: {n_bad_signed} rows with r_signed outside [-1,1]")

            total_rows += df.height

    passed = len(issues) == 0 and total_rows > 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "total_rows": total_rows,
            "n_tables_expected": len(combos) * len(pgs_ids) if "pgs_ids" in locals() else 0,
            "issues": str(issues) if issues else "none",
        },
    )


@asset_check(
    asset="reference_allele_universe",
    description=(
        "Validate the reference-allele universe: resolved REF bases are clean "
        "(single A/C/G/T for the FASTA tier; non-empty for panel), ref_source is "
        "one of {panel, fasta, unresolved}, unresolved rows carry a null ref, and "
        "the table is non-empty."
    ),
)
def check_reference_allele_universe_valid(
    cache_dir_resource: CacheDirResource,
) -> AssetCheckResult:
    cache_dir = cache_dir_resource.get_path()
    path = cache_dir / "percentiles" / "reference_allele_universe.parquet"
    if not path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": f"missing {path}"},
        )

    df = pl.read_parquet(path)
    issues: list[str] = []

    valid_sources = {"panel", "fasta", "unresolved"}
    bad_sources = set(df["ref_source"].unique().to_list()) - valid_sources
    if bad_sources:
        issues.append(f"unexpected ref_source values: {bad_sources}")

    # Resolved rows must have a non-null REF; unresolved must be null.
    n_resolved_null = df.filter(
        (pl.col("ref_source") != "unresolved") & pl.col("ref").is_null()
    ).height
    if n_resolved_null:
        issues.append(f"{n_resolved_null} resolved rows with null ref")
    n_unresolved_nonnull = df.filter(
        (pl.col("ref_source") == "unresolved") & pl.col("ref").is_not_null()
    ).height
    if n_unresolved_nonnull:
        issues.append(f"{n_unresolved_nonnull} unresolved rows with non-null ref")

    # FASTA tier must be a single A/C/G/T base.
    n_bad_fasta = df.filter(
        (pl.col("ref_source") == "fasta")
        & ~pl.col("ref").is_in(["A", "C", "G", "T"])
    ).height
    if n_bad_fasta:
        issues.append(f"{n_bad_fasta} fasta rows with non-ACGT/multi-base ref")

    # Unique positions (no duplicate (chrom,pos)).
    n_dupes = df.height - df.select("chrom", "pos").n_unique()
    if n_dupes:
        issues.append(f"{n_dupes} duplicate (chrom,pos) rows")

    passed = len(issues) == 0 and df.height > 0
    n_unresolved = df.filter(pl.col("ref_source") == "unresolved").height
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "n_positions": df.height,
            "n_panel": df.filter(pl.col("ref_source") == "panel").height,
            "n_fasta": df.filter(pl.col("ref_source") == "fasta").height,
            "n_unresolved": n_unresolved,
            "issues": str(issues) if issues else "none",
        },
    )


ALL_ASSET_CHECKS = [
    check_distributions_superpop_completeness,
    check_distributions_no_inf_nan,
    check_distributions_quantile_ordering,
    check_distributions_vs_raw_scores,
    check_distributions_sample_sizes,
    check_enriched_has_metadata_columns,
    check_cleaned_metadata_quality,
    check_chip_coverage_valid,
    check_ld_proxy_table_valid,
    check_reference_allele_universe_valid,
]
