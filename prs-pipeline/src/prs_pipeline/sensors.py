"""Sensors that orchestrate the full reference panel pipeline as a single-trigger flow.

Flow:
  1. User triggers `download_reference_data` job (manually or via CLI).
  2. `score_all_partitions_sensor` detects that `download_reference_data` succeeded,
     reads all registered dynamic partition keys, and yields one RunRequest per PGS ID
     to the `per_pgs_scores_job`.
  3. `aggregate_when_done_sensor` polls the partition status of `per_pgs_scores`.
     When all registered partitions have been materialized and no scoring runs are
     still in progress, it triggers `aggregate_and_push`.

Sensors are constructed via `build_pipeline_sensors()` which accepts job objects
to avoid circular imports between definitions.py and sensors.py.
"""

import json

from dagster import (
    AssetKey,
    DagsterInstance,
    DagsterRunStatus,
    RunRequest,
    RunsFilter,
    RunStatusSensorContext,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    run_status_sensor,
    sensor,
)


def build_pipeline_sensors(
    per_pgs_scores_job: object,
    aggregate_and_push_job: object,
) -> list:
    """Build the two pipeline sensors, given the actual job objects.

    Returns a list of sensor definitions ready to pass to ``Definitions(sensors=...)``.
    """

    @run_status_sensor(
        run_status=DagsterRunStatus.SUCCESS,
        name="score_all_partitions_sensor",
        description=(
            "After download_reference_data succeeds, launch per_pgs_scores_job "
            "for every PGS ID partition that hasn't been scored yet."
        ),
        request_job=per_pgs_scores_job,
        monitor_all_code_locations=True,
    )
    def score_all_partitions_sensor(context: RunStatusSensorContext) -> SensorResult:
        """Trigger per_pgs_scores for all unmaterialized PGS ID partitions."""
        dagster_run = context.dagster_run
        if dagster_run.job_name != "download_reference_data":
            return SensorResult(run_requests=[], skip_reason="Not the download_reference_data job.")

        from prs_pipeline.assets import PGS_IDS_PARTITIONS

        partition_keys = context.instance.get_dynamic_partitions(PGS_IDS_PARTITIONS.name)
        if not partition_keys:
            return SensorResult(
                run_requests=[],
                skip_reason="No PGS ID partitions registered yet.",
            )

        already_materialized = _get_materialized_partitions(
            context.instance, AssetKey("per_pgs_scores")
        )

        run_id_prefix = dagster_run.run_id[:8]
        run_requests: list[RunRequest] = []
        for pgs_id in partition_keys:
            if pgs_id in already_materialized:
                continue
            run_requests.append(
                RunRequest(
                    run_key=f"per_pgs_scores_{run_id_prefix}_{pgs_id}",
                    partition_key=pgs_id,
                )
            )

        context.log.info(
            f"Launching {len(run_requests)} partition runs "
            f"({len(already_materialized)} already materialized, "
            f"{len(partition_keys)} total)."
        )
        return SensorResult(run_requests=run_requests)

    @sensor(
        name="aggregate_when_done_sensor",
        description=(
            "Polls per_pgs_scores partition status. When all registered partitions "
            "are materialized and no scoring runs are in progress, triggers "
            "aggregate_and_push to collect results and upload to HuggingFace."
        ),
        job=aggregate_and_push_job,
        minimum_interval_seconds=60,
    )
    def aggregate_when_done_sensor(context: SensorEvaluationContext) -> SensorResult | SkipReason:
        """Trigger aggregation once all per_pgs_scores partitions are materialized."""
        from prs_pipeline.assets import PGS_IDS_PARTITIONS

        cursor: dict[str, int] = json.loads(context.cursor) if context.cursor else {}
        last_triggered_count = cursor.get("last_triggered_count", 0)

        partition_keys = context.instance.get_dynamic_partitions(PGS_IDS_PARTITIONS.name)
        if not partition_keys:
            return SkipReason("No PGS ID partitions registered yet.")

        materialized = _get_materialized_partitions(
            context.instance, AssetKey("per_pgs_scores")
        )
        n_total = len(partition_keys)
        n_done = len(materialized)

        if n_done == 0:
            return SkipReason("No partitions materialized yet.")

        if n_done <= last_triggered_count:
            return SkipReason(
                f"Already triggered aggregation for {last_triggered_count} partitions. "
                f"Currently {n_done}/{n_total} materialized."
            )

        scoring_in_progress = context.instance.get_run_records(
            filters=RunsFilter(
                job_name="per_pgs_scores_job",
                statuses=[
                    DagsterRunStatus.STARTED,
                    DagsterRunStatus.NOT_STARTED,
                    DagsterRunStatus.QUEUED,
                ],
            ),
            limit=1,
        )

        if scoring_in_progress:
            return SkipReason(
                f"Scoring still in progress: {n_done}/{n_total} partitions done."
            )

        context.log.info(
            f"All scoring runs finished: {n_done}/{n_total} partitions materialized. "
            f"Triggering aggregation."
        )
        context.update_cursor(json.dumps({"last_triggered_count": n_done}))
        return SensorResult(
            run_requests=[RunRequest(run_key=f"aggregate_batch_{n_done}")]
        )

    return [score_all_partitions_sensor, aggregate_when_done_sensor]


def _get_materialized_partitions(instance: DagsterInstance, asset_key: AssetKey) -> set[str]:
    """Return the set of partition keys that have been materialized for an asset."""
    return set(instance.get_materialized_partitions(asset_key))
