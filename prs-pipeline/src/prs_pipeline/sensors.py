"""Smart sensors for the PRS pipeline.

Implements the 7 Robustness Guarantees from AGENTS.md:

1. **startup_sensor** (30s) — handles the initial materialization check.
   It submits the target job only when key assets are unmaterialized.

2. **completeness_sensor** (5min) — compares on-disk scored PGS IDs
   against the EBI catalog.  If scored < catalog, submits ``score_and_push``
   to fill the gap (with ``skip_existing``).

3. **failure_retry_sensor** (15min) — reads the quality parquet, collects
   PGS IDs with ``status == "failed"``, and submits a targeted retry via
   ``PRS_PIPELINE_TEST_IDS``.  Stops after N consecutive retries with the
   same failure set (configurable via ``PRS_PIPELINE_MAX_FAILURE_RETRIES``).

4. **upstream_freshness_sensor** (6h) — compares a live HTTP fingerprint
   of the EBI scoring manifest against the fingerprint stored in the last
   ``ebi_scoring_files_fingerprint`` materialization.  Submits
   ``full_pipeline`` when upstream has new content.
"""

import hashlib
import os
from pathlib import Path

import dagster as dg

from just_prs.scoring import resolve_cache_dir

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_JOB_CHECK_KEYS: dict[str, list[dg.AssetKey]] = {
    "full_pipeline": [
        dg.AssetKey("ebi_reference_panel_fingerprint"),
        dg.AssetKey("ebi_scoring_files_fingerprint"),
        dg.AssetKey("scoring_files"),
        dg.AssetKey("reference_scores"),
        dg.AssetKey("cleaned_pgs_metadata"),
        dg.AssetKey("hf_pgs_catalog"),
        dg.AssetKey("hf_pgs_catalog_risk_metadata"),
        dg.AssetKey("hf_prs_percentiles"),
    ],
    "catalog_pipeline": [
        dg.AssetKey("ebi_scoring_files_fingerprint"),
        dg.AssetKey("scoring_files"),
        dg.AssetKey("scoring_files_parquet"),
        dg.AssetKey("cleaned_pgs_metadata"),
        dg.AssetKey("hf_pgs_catalog"),
        dg.AssetKey("hf_pgs_catalog_risk_metadata"),
    ],
    "reference_percentile_audit_job": [
        dg.AssetKey("reference_percentile_audit"),
    ],
    "ld_proxy_pipeline": [
        dg.AssetKey("ld_proxy_table"),
        dg.AssetKey("hf_ld_proxy_table"),
    ],
}

_PIPELINE_JOB_NAMES = (
    "full_pipeline",
    "catalog_pipeline",
    "score_and_push",
    "reference_percentile_audit_job",
    "ld_proxy_pipeline",
)


def _has_any_active_pipeline_run(instance: dg.DagsterInstance) -> dg.DagsterRun | None:
    """Return an active pipeline run so sensors do not overlap heavy jobs."""
    active = instance.get_runs(
        filters=dg.RunsFilter(
            statuses=[
                dg.DagsterRunStatus.STARTED,
                dg.DagsterRunStatus.NOT_STARTED,
                dg.DagsterRunStatus.QUEUED,
            ],
        ),
        limit=20,
    )
    return next((run for run in active if run.job_name in _PIPELINE_JOB_NAMES), None)


def _last_unsuccessful_run_id(instance: dg.DagsterInstance, job_name: str) -> str | None:
    """Return the run id of the most recent *terminal-but-not-successful* run.

    Covers both ``FAILURE`` and ``CANCELED`` so an OOM-killed run (orphan-cleaned
    to ``CANCELED``) does not permanently block resubmission. Returns ``None``
    when the latest run succeeded or no run exists. The returned id is used to
    build a fresh, deterministic retry run_key (one retry per failed run — not a
    timestamp), so Dagster's deduplication still prevents tick-by-tick resubmits.
    """
    runs = instance.get_runs(
        filters=dg.RunsFilter(job_name=job_name),
        limit=1,
    )
    if not runs:
        return None
    run = runs[0]
    if run.status in (dg.DagsterRunStatus.FAILURE, dg.DagsterRunStatus.CANCELED):
        return run.run_id
    return None


def _resolve_cache() -> Path:
    """Resolve the cache directory from env or platformdirs."""
    env = os.environ.get("PRS_CACHE_DIR", "").strip()
    if env:
        return Path(env)
    return resolve_cache_dir()


# ---------------------------------------------------------------------------
# 1. Startup sensor
# ---------------------------------------------------------------------------

def _make_startup_sensor(
    full_pipeline_job: object,
    catalog_pipeline_job: object,
    reference_percentile_audit_job: object,
    ld_proxy_pipeline_job: object,
) -> dg.SensorDefinition:
    """Startup sensor: initial materialization check."""

    @dg.sensor(
        jobs=[full_pipeline_job, catalog_pipeline_job, reference_percentile_audit_job, ld_proxy_pipeline_job],
        default_status=dg.DefaultSensorStatus.RUNNING,
        minimum_interval_seconds=30,
        name="startup_sensor",
        description=(
            "Submits a pipeline job on startup only when key assets are "
            "unmaterialized."
        ),
    )
    def startup_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult | dg.SkipReason:
        target_job = os.environ.get("PRS_PIPELINE_STARTUP_JOB", "full_pipeline")

        active = _has_any_active_pipeline_run(context.instance)
        if active:
            return dg.SkipReason(
                f"{active.job_name} already in progress (run {active.run_id[:8]})."
            )

        request_id = os.environ.get("PRS_PIPELINE_STARTUP_REQUEST_ID", "").strip()
        no_cache = os.environ.get("PRS_PIPELINE_NO_CACHE", "").strip().lower() in {"1", "true", "yes"}
        test_ids = os.environ.get("PRS_PIPELINE_TEST_IDS", "").strip()
        if target_job == "reference_percentile_audit_job" or request_id or no_cache or test_ids:
            explicit_id = request_id or ("no_cache" if no_cache else "test" if test_ids else "startup")
            run_key = f"{target_job}_{explicit_id}"
            context.log.info(f"Submitting explicit {target_job} run with run_key={run_key}.")
            return dg.SensorResult(
                run_requests=[dg.RunRequest(run_key=run_key, job_name=target_job)],
            )

        check_keys = _JOB_CHECK_KEYS.get(target_job, _JOB_CHECK_KEYS["full_pipeline"])
        missing = [
            k.to_user_string()
            for k in check_keys
            if context.instance.get_latest_materialization_event(k) is None
        ]
        if not missing:
            return dg.SkipReason(f"All {target_job} assets already materialized.")

        last_bad_run_id = _last_unsuccessful_run_id(context.instance, target_job)
        if last_bad_run_id:
            run_key = f"pipeline_startup_retry_{target_job}_{last_bad_run_id[:12]}"
            context.log.info(
                f"Last {target_job} run {last_bad_run_id[:8]} did not succeed "
                f"(failed/canceled) — retrying with fresh run_key={run_key}."
            )
        else:
            run_key = f"pipeline_startup_{target_job}"

        context.log.info(f"Unmaterialized assets {missing} — submitting {target_job}.")
        return dg.SensorResult(
            run_requests=[dg.RunRequest(run_key=run_key, job_name=target_job)],
        )

    return startup_sensor


# ---------------------------------------------------------------------------
# 2. Completeness sensor
# ---------------------------------------------------------------------------

def _make_completeness_sensor(
    score_and_push_job: object,
) -> dg.SensorDefinition:
    """Completeness sensor: compare on-disk scored IDs vs EBI catalog."""

    @dg.sensor(
        jobs=[score_and_push_job],
        default_status=dg.DefaultSensorStatus.RUNNING,
        minimum_interval_seconds=300,
        name="completeness_sensor",
        description=(
            "Compares on-disk scored PGS IDs against the EBI catalog. "
            "Submits score_and_push when the gap exceeds the threshold. "
            "Implements Robustness Guarantee #2 (interrupted-run recovery)."
        ),
    )
    def completeness_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult | dg.SkipReason:
        active = _has_any_active_pipeline_run(context.instance)
        if active:
            return dg.SkipReason(
                f"{active.job_name} already in progress (run {active.run_id[:8]})."
            )

        cache_dir = _resolve_cache()
        panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
        scores_dir = cache_dir / "reference_scores" / panel

        scored_ids: set[str] = set()
        corrupt_count = 0
        if scores_dir.exists():
            import polars as pl

            for p in scores_dir.iterdir():
                if not (p.is_dir() and (p / "scores.parquet").exists()):
                    continue
                parquet_path = p / "scores.parquet"
                try:
                    lf = pl.scan_parquet(parquet_path)
                    lf.collect_schema()
                    # collect_schema() reads only footer metadata; data-page
                    # corruption is only caught when row groups are read.
                    # Read one row to probe the first row group cheaply.
                    lf.head(1).collect()
                    scored_ids.add(p.name)
                except Exception:
                    corrupt_count += 1
                    context.log.warning(
                        f"Corrupt parquet detected for {p.name} at {parquet_path} — "
                        f"deleting and marking as missing."
                    )
                    parquet_path.unlink(missing_ok=True)

        try:
            from just_prs.ftp import list_all_pgs_ids
            all_ids = list_all_pgs_ids()
        except Exception as exc:
            context.log.warning(f"Failed to fetch PGS ID list: {exc}")
            return dg.SkipReason(f"Could not fetch EBI catalog: {exc}")

        catalog_total = len(all_ids)
        missing_ids = set(all_ids) - scored_ids
        n_missing = len(missing_ids)
        coverage = len(scored_ids) / catalog_total if catalog_total > 0 else 0.0

        context.log.info(
            f"Completeness: {len(scored_ids)}/{catalog_total} scored "
            f"(coverage={coverage:.1%}, missing={n_missing}"
            + (f", corrupt_deleted={corrupt_count}" if corrupt_count else "")
            + ")."
        )

        if n_missing == 0:
            return dg.SkipReason(
                f"All {catalog_total} PGS IDs scored on disk (coverage=100%)."
            )

        run_key = f"completeness_gap_{n_missing}"
        context.log.info(
            f"Gap of {n_missing} PGS IDs detected — submitting score_and_push."
        )
        return dg.SensorResult(
            run_requests=[dg.RunRequest(run_key=run_key, job_name="score_and_push")],
        )

    return completeness_sensor


# ---------------------------------------------------------------------------
# 3. Failure retry sensor
# ---------------------------------------------------------------------------

_failure_retry_counts: dict[str, int] = {}
_ld_failure_retry_counts: dict[str, int] = {}


def _make_failure_retry_sensor(
    score_and_push_job: object,
) -> dg.SensorDefinition:
    """Failure retry sensor: read quality parquet, retry failed IDs."""

    @dg.sensor(
        jobs=[score_and_push_job],
        default_status=dg.DefaultSensorStatus.RUNNING,
        minimum_interval_seconds=900,
        name="failure_retry_sensor",
        description=(
            "Reads the quality parquet, collects PGS IDs with status='failed', "
            "and submits a targeted retry via PRS_PIPELINE_TEST_IDS. "
            "Stops after N consecutive retries with the same failure set. "
            "Implements Robustness Guarantee #3 (automatic failure retry)."
        ),
    )
    def failure_retry_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult | dg.SkipReason:
        active = _has_any_active_pipeline_run(context.instance)
        if active:
            return dg.SkipReason(
                f"{active.job_name} already in progress (run {active.run_id[:8]})."
            )

        cache_dir = _resolve_cache()
        panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
        quality_path = cache_dir / "percentiles" / f"{panel}_quality.parquet"

        if not quality_path.exists():
            return dg.SkipReason(f"No quality parquet at {quality_path}.")

        import polars as pl

        quality_df = pl.read_parquet(quality_path)
        failed = quality_df.filter(pl.col("status") == "failed")
        if failed.height == 0:
            return dg.SkipReason("No failed PGS IDs in quality report.")

        failed_ids = sorted(failed["pgs_id"].to_list())
        failure_hash = hashlib.sha256(",".join(failed_ids).encode()).hexdigest()[:12]

        max_retries_str = os.environ.get("PRS_PIPELINE_MAX_FAILURE_RETRIES", "3").strip()
        max_retries = int(max_retries_str) if max_retries_str else 3

        retry_count = _failure_retry_counts.get(failure_hash, 0)
        if retry_count >= max_retries:
            context.log.warning(
                f"Exhausted {max_retries} retries for failure set {failure_hash} "
                f"({len(failed_ids)} IDs: {failed_ids[:5]}...). "
                f"These are likely permanent failures."
            )
            return dg.SkipReason(
                f"Max retries ({max_retries}) exhausted for {len(failed_ids)} permanently failed IDs."
            )

        _failure_retry_counts[failure_hash] = retry_count + 1

        os.environ["PRS_PIPELINE_TEST_IDS"] = ",".join(failed_ids)

        run_key = f"failure_retry_{failure_hash}_attempt{retry_count + 1}"
        context.log.info(
            f"Retrying {len(failed_ids)} failed PGS IDs "
            f"(attempt {retry_count + 1}/{max_retries}, hash={failure_hash}): "
            f"{failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}."
        )
        return dg.SensorResult(
            run_requests=[dg.RunRequest(run_key=run_key, job_name="score_and_push")],
        )

    return failure_retry_sensor


# ---------------------------------------------------------------------------
# 3b. LD-proxy completeness / retry sensors
# ---------------------------------------------------------------------------

def _ld_proxy_automation_enabled() -> bool:
    """Return True when LD-proxy full-catalog automation is explicitly enabled."""
    return (
        os.environ.get("PRS_LD_ENABLE_SENSORS", "").strip().lower() in {"1", "true", "yes"}
        or os.environ.get("PRS_LD_FULL_CATALOG", "").strip().lower() in {"1", "true", "yes"}
    )


def _make_ld_proxy_completeness_sensor(
    ld_proxy_pipeline_job: object,
) -> dg.SensorDefinition:
    """LD completeness sensor: compare per-PGS proxy files against catalog."""

    @dg.sensor(
        jobs=[ld_proxy_pipeline_job],
        default_status=dg.DefaultSensorStatus.RUNNING,
        minimum_interval_seconds=900,
        name="ld_proxy_completeness_sensor",
        description=(
            "Compares per-PGS LD-proxy outputs against the PGS catalog and "
            "submits ld_proxy_pipeline to fill gaps when explicitly enabled."
        ),
    )
    def ld_proxy_completeness_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult | dg.SkipReason:
        if not _ld_proxy_automation_enabled():
            return dg.SkipReason("LD-proxy completeness automation is disabled.")

        active = _has_any_active_pipeline_run(context.instance)
        if active:
            return dg.SkipReason(
                f"{active.job_name} already in progress (run {active.run_id[:8]})."
            )

        cache_dir = _resolve_cache()
        panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
        chip = os.environ.get("PRS_LD_CHIP", "gsa_v3")
        build = os.environ.get("PRS_LD_BUILD", "GRCh38")
        out_dir = cache_dir / "percentiles" / "ld_proxy" / panel / chip / build

        proxied_ids: set[str] = set()
        corrupt_count = 0
        if out_dir.exists():
            import polars as pl

            for parquet_path in out_dir.glob("PGS*.parquet"):
                try:
                    lf = pl.scan_parquet(parquet_path)
                    lf.collect_schema()
                    lf.head(1).collect()
                    proxied_ids.add(parquet_path.stem)
                except Exception:
                    corrupt_count += 1
                    context.log.warning(
                        f"Corrupt LD proxy parquet detected at {parquet_path} — deleting."
                    )
                    parquet_path.unlink(missing_ok=True)

        try:
            from just_prs.ftp import list_all_pgs_ids
            all_ids = list_all_pgs_ids()
        except Exception as exc:
            context.log.warning(f"Failed to fetch PGS ID list: {exc}")
            return dg.SkipReason(f"Could not fetch EBI catalog: {exc}")

        missing_ids = sorted(set(all_ids) - proxied_ids)
        if not missing_ids:
            return dg.SkipReason(f"All {len(all_ids)} PGS IDs have LD-proxy files.")

        os.environ["PRS_LD_FULL_CATALOG"] = "1"
        os.environ.pop("PRS_LD_PGS_IDS", None)
        run_key = f"ld_proxy_completeness_gap_{len(missing_ids)}"
        context.log.info(
            f"LD proxy gap: {len(proxied_ids)}/{len(all_ids)} files present, "
            f"missing={len(missing_ids)}"
            + (f", corrupt_deleted={corrupt_count}" if corrupt_count else "")
            + " — submitting ld_proxy_pipeline."
        )
        return dg.SensorResult(
            run_requests=[dg.RunRequest(run_key=run_key, job_name="ld_proxy_pipeline")],
        )

    return ld_proxy_completeness_sensor


def _make_ld_proxy_failure_retry_sensor(
    ld_proxy_pipeline_job: object,
) -> dg.SensorDefinition:
    """LD failure retry sensor: retry failed per-PGS LD-proxy outcomes."""

    @dg.sensor(
        jobs=[ld_proxy_pipeline_job],
        default_status=dg.DefaultSensorStatus.RUNNING,
        minimum_interval_seconds=1800,
        name="ld_proxy_failure_retry_sensor",
        description=(
            "Reads per-PGS LD-proxy quality parquet and retries failed PGS IDs "
            "when explicitly enabled."
        ),
    )
    def ld_proxy_failure_retry_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult | dg.SkipReason:
        if not _ld_proxy_automation_enabled():
            return dg.SkipReason("LD-proxy failure retry automation is disabled.")

        active = _has_any_active_pipeline_run(context.instance)
        if active:
            return dg.SkipReason(
                f"{active.job_name} already in progress (run {active.run_id[:8]})."
            )

        cache_dir = _resolve_cache()
        panel = os.environ.get("PRS_PIPELINE_PANEL", "1000g")
        chip = os.environ.get("PRS_LD_CHIP", "gsa_v3")
        build = os.environ.get("PRS_LD_BUILD", "GRCh38")
        quality_path = cache_dir / "percentiles" / "ld_proxy" / panel / chip / build / "_quality.parquet"
        if not quality_path.exists():
            return dg.SkipReason(f"No LD proxy quality parquet at {quality_path}.")

        import polars as pl

        quality_df = pl.read_parquet(quality_path)
        failed = quality_df.filter(pl.col("status") == "failed")
        if failed.height == 0:
            return dg.SkipReason("No failed PGS IDs in LD proxy quality report.")

        failed_ids = sorted(failed["pgs_id"].to_list())
        failure_hash = hashlib.sha256(",".join(failed_ids).encode()).hexdigest()[:12]
        max_retries_str = os.environ.get("PRS_PIPELINE_MAX_FAILURE_RETRIES", "3").strip()
        max_retries = int(max_retries_str) if max_retries_str else 3
        retry_count = _ld_failure_retry_counts.get(failure_hash, 0)
        if retry_count >= max_retries:
            return dg.SkipReason(
                f"Max retries ({max_retries}) exhausted for {len(failed_ids)} LD proxy failures."
            )

        _ld_failure_retry_counts[failure_hash] = retry_count + 1
        os.environ["PRS_LD_PGS_IDS"] = ",".join(failed_ids)
        os.environ.pop("PRS_LD_FULL_CATALOG", None)
        run_key = f"ld_proxy_failure_retry_{failure_hash}_attempt{retry_count + 1}"
        context.log.info(
            f"Retrying {len(failed_ids)} failed LD proxy PGS IDs "
            f"(attempt {retry_count + 1}/{max_retries}, hash={failure_hash})."
        )
        return dg.SensorResult(
            run_requests=[dg.RunRequest(run_key=run_key, job_name="ld_proxy_pipeline")],
        )

    return ld_proxy_failure_retry_sensor


# ---------------------------------------------------------------------------
# 4. Upstream freshness sensor
# ---------------------------------------------------------------------------

def _make_upstream_freshness_sensor(
    full_pipeline_job: object,
) -> dg.SensorDefinition:
    """Upstream freshness sensor: detect new content at EBI."""

    @dg.sensor(
        jobs=[full_pipeline_job],
        default_status=dg.DefaultSensorStatus.RUNNING,
        minimum_interval_seconds=21600,
        name="upstream_freshness_sensor",
        description=(
            "Compares a live HTTP fingerprint of the EBI scoring manifest "
            "against the stored fingerprint. Submits full_pipeline when "
            "upstream has new content. "
            "Implements Robustness Guarantee #1 (input-change invalidation)."
        ),
    )
    def upstream_freshness_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult | dg.SkipReason:
        active = _has_any_active_pipeline_run(context.instance)
        if active:
            return dg.SkipReason(
                f"{active.job_name} already in progress (run {active.run_id[:8]})."
            )

        fingerprint_key = dg.AssetKey("ebi_scoring_files_fingerprint")
        last_event = context.instance.get_latest_materialization_event(fingerprint_key)
        if last_event is None:
            return dg.SkipReason("No previous fingerprint materialization to compare against.")

        stored_fingerprint: str | None = None
        mat = last_event.asset_materialization
        if mat and mat.metadata:
            fp_entry = mat.metadata.get("fingerprint_sha256")
            if fp_entry:
                stored_fingerprint = str(fp_entry.value) if hasattr(fp_entry, "value") else str(fp_entry)

        if not stored_fingerprint:
            return dg.SkipReason("No fingerprint_sha256 in stored materialization metadata.")

        try:
            from just_prs.ftp import PGS_SCORES_LIST_URL
            from prs_pipeline.fingerprint import fingerprint_http_resource
            live_fingerprint, _ = fingerprint_http_resource(PGS_SCORES_LIST_URL, include_body_hash=True)
        except Exception as exc:
            context.log.warning(f"Failed to fetch live fingerprint: {exc}")
            return dg.SkipReason(f"Could not fetch live fingerprint: {exc}")

        if live_fingerprint == stored_fingerprint:
            context.log.info("Upstream fingerprint unchanged — no new content.")
            return dg.SkipReason("Upstream EBI catalog fingerprint unchanged.")

        run_key = f"upstream_change_{live_fingerprint[:16]}"
        context.log.info(
            f"Upstream fingerprint changed: {stored_fingerprint[:16]}... -> {live_fingerprint[:16]}... "
            f"— submitting full_pipeline."
        )
        return dg.SensorResult(
            run_requests=[dg.RunRequest(run_key=run_key, job_name="full_pipeline")],
        )

    return upstream_freshness_sensor


# ---------------------------------------------------------------------------
# Public factory: creates all 4 sensors
# ---------------------------------------------------------------------------

def make_startup_sensor(
    full_pipeline_job: object,
    catalog_pipeline_job: object,
    reference_percentile_audit_job: object | None = None,
    ld_proxy_pipeline_job: object | None = None,
) -> dg.SensorDefinition:
    """Create the startup sensor (backward-compatible entry point).

    Use ``make_all_sensors()`` for the full set of smart sensors.
    """
    return _make_startup_sensor(
        full_pipeline_job,
        catalog_pipeline_job,
        reference_percentile_audit_job or full_pipeline_job,
        ld_proxy_pipeline_job or full_pipeline_job,
    )


def make_all_sensors(
    full_pipeline_job: object,
    catalog_pipeline_job: object,
    score_and_push_job: object,
    reference_percentile_audit_job: object,
    ld_proxy_pipeline_job: object | None = None,
) -> list[dg.SensorDefinition]:
    """Create all 4 smart pipeline sensors.

    Returns:
        List of sensor definitions to register in Definitions(sensors=[...]).
    """
    return [
        _make_startup_sensor(
            full_pipeline_job,
            catalog_pipeline_job,
            reference_percentile_audit_job,
            ld_proxy_pipeline_job or full_pipeline_job,
        ),
        _make_completeness_sensor(score_and_push_job),
        _make_failure_retry_sensor(score_and_push_job),
        _make_ld_proxy_completeness_sensor(ld_proxy_pipeline_job or full_pipeline_job),
        _make_ld_proxy_failure_retry_sensor(ld_proxy_pipeline_job or full_pipeline_job),
        _make_upstream_freshness_sensor(full_pipeline_job),
    ]
