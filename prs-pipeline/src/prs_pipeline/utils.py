"""Dagster hooks and helpers for the PRS pipeline."""

from dagster import DagsterEventType, HookContext, success_hook


@success_hook
def resource_summary_hook(context: HookContext) -> None:
    """Aggregate resource metrics across all assets in a run.

    Logs total duration, peak memory (bottleneck identification), and the
    top memory consumers.  Attach to jobs via ``hooks={resource_summary_hook}``.
    """
    run_id = context.run_id
    instance = context.instance

    log_entries = instance.all_logs(run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION)

    total_duration = 0.0
    max_peak_memory = 0.0
    total_cpu = 0.0
    asset_count = 0
    asset_metrics: list[dict] = []

    for entry in log_entries:
        mat = entry.asset_materialization
        if mat is None:
            continue

        metadata = mat.metadata or {}

        has_metrics = False
        info: dict = {"asset": mat.asset_key.to_user_string()}

        for key, md_val in metadata.items():
            if key.endswith("_duration_sec"):
                val = md_val.value if hasattr(md_val, "value") else float(md_val)
                total_duration += val
                info["duration_sec"] = val
                has_metrics = True
            elif key.endswith("_peak_memory_mb"):
                val = md_val.value if hasattr(md_val, "value") else float(md_val)
                max_peak_memory = max(max_peak_memory, val)
                info["peak_memory_mb"] = val
                has_metrics = True
            elif key.endswith("_cpu_percent"):
                val = md_val.value if hasattr(md_val, "value") else float(md_val)
                total_cpu += val
                info["cpu_percent"] = val

        if has_metrics:
            asset_metrics.append(info)
            asset_count += 1

    if asset_count == 0:
        return

    avg_cpu = total_cpu / asset_count
    sorted_by_memory = sorted(asset_metrics, key=lambda x: x.get("peak_memory_mb", 0), reverse=True)
    top_memory = sorted_by_memory[:3]

    context.log.info(
        f"RUN RESOURCE SUMMARY\n"
        f"  Total Duration: {total_duration:.1f}s ({total_duration / 60:.1f} min)\n"
        f"  Max Peak Memory: {max_peak_memory:.1f} MB\n"
        f"  Average CPU: {avg_cpu:.1f}%\n"
        f"  Assets with metrics: {asset_count}\n"
        f"  Top memory consumers:\n"
        + "\n".join(f"    - {a['asset']}: {a.get('peak_memory_mb', 0):.1f} MB" for a in top_memory)
    )
