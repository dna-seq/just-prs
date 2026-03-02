"""Sensors for the PRS pipeline.

The run-once sensor checks whether key pipeline assets have been
materialized.  If any are missing AND no full_pipeline run is already
in progress, it submits the ``full_pipeline`` job.
Once all assets are materialized it skips on subsequent ticks.
"""

import os

import dagster as dg


@dg.sensor(
    job_name="full_pipeline",
    default_status=dg.DefaultSensorStatus.RUNNING,
    minimum_interval_seconds=30,
    description=(
        "Submits full_pipeline when reference_scores, cleaned_pgs_metadata, "
        "or hf_prs_percentiles are not yet materialized, and no run is active."
    ),
)
def run_pipeline_on_startup(context: dg.SensorEvaluationContext) -> dg.SensorResult | dg.SkipReason:
    """Submit the full pipeline job if key assets are unmaterialized and no run is in flight."""
    check_keys = [
        dg.AssetKey("ebi_reference_panel_fingerprint"),
        dg.AssetKey("ebi_scoring_files_fingerprint"),
        dg.AssetKey("reference_scores"),
        dg.AssetKey("cleaned_pgs_metadata"),
        dg.AssetKey("hf_prs_percentiles"),
    ]

    force_run = os.environ.get("PRS_PIPELINE_FORCE_RUN_ON_STARTUP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    startup_run_key = os.environ.get("PRS_PIPELINE_STARTUP_RUN_KEY", "pipeline_startup")

    all_materialized = all(
        context.instance.get_latest_materialization_event(key) is not None
        for key in check_keys
    )
    if all_materialized and not force_run:
        return dg.SkipReason("All pipeline assets already materialized.")

    active_statuses = [
        dg.DagsterRunStatus.STARTED,
        dg.DagsterRunStatus.NOT_STARTED,
        dg.DagsterRunStatus.QUEUED,
    ]
    active_runs = context.instance.get_runs(
        filters=dg.RunsFilter(job_name="full_pipeline", statuses=active_statuses)
    )
    if active_runs:
        return dg.SkipReason(f"full_pipeline already in progress (run {active_runs[0].run_id[:8]}).")

    if force_run:
        context.log.info("Forced startup run requested; submitting full_pipeline.")
    else:
        missing = [
            k.to_user_string()
            for k in check_keys
            if context.instance.get_latest_materialization_event(k) is None
        ]
        context.log.info(f"Unmaterialized assets {missing} — submitting full_pipeline.")

    return dg.SensorResult(run_requests=[dg.RunRequest(run_key=startup_run_key)])
