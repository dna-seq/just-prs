"""Resource tracking for Dagster assets.

Provides a context manager that records CPU, memory, and wall-clock time
for compute-heavy assets. Metrics are logged to the Dagster UI via
``context.add_output_metadata`` and to the Dagster logger.

Peak memory is sampled from the **process tree** (this process plus
descendants) on a timer. Start/end RSS alone cannot explain an OOM and
cannot emit after SIGKILL — the parent must sample the worker tree.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from typing import Any, Optional

import psutil
from pydantic import BaseModel

from just_prs.memory import ProcessTreeSampler, sample_interval_sec


class ResourceReport(BaseModel):
    """Snapshot of resource consumption for a single asset execution."""

    name: str
    duration_sec: float
    cpu_percent: float
    peak_memory_mb: float
    memory_delta_mb: float
    start_mem_bytes: int
    end_mem_bytes: int
    start_tree_rss_bytes: int = 0
    end_tree_rss_bytes: int = 0
    peak_tree_rss_bytes: int = 0
    available_bytes: int = 0
    n_processes: int = 1


@contextmanager
def resource_tracker(name: str = "resource_usage", context: Optional[Any] = None):
    """Track execution time, CPU and peak process-tree memory for a block.

    Args:
        name: Human-readable label (usually the asset name).
        context: Optional ``AssetExecutionContext``.  When provided, metrics
            are attached as Dagster output metadata so they appear in the UI.

    Yields a mutable dict.  After the block finishes the dict contains a
    ``"report"`` key with a :class:`ResourceReport`. Callers may append
    checkpoint reports onto ``data["checkpoints"]``.
    """
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss
    sampler = ProcessTreeSampler(interval_sec=sample_interval_sec())
    start_tree = sampler.snapshot()
    sampler.start()

    process.cpu_percent(interval=None)

    data: dict[str, Any] = {
        "name": name,
        "start_time": start_time,
        "start_mem": start_mem,
        "sampler": sampler,
        "checkpoints": [],
    }
    try:
        yield data
    finally:
        end_tree = sampler.stop()
        end_time = time.perf_counter()
        end_mem = process.memory_info().rss
        cpu_usage = process.cpu_percent(interval=None)
        peak_tree = max(start_tree.peak_rss_bytes, end_tree.peak_rss_bytes, sampler.peak_rss_bytes)

        report = ResourceReport(
            name=name,
            duration_sec=round(end_time - start_time, 2),
            cpu_percent=round(cpu_usage, 1),
            peak_memory_mb=round(peak_tree / (1024 * 1024), 2),
            memory_delta_mb=round((end_tree.rss_bytes - start_tree.rss_bytes) / (1024 * 1024), 2),
            start_mem_bytes=start_mem,
            end_mem_bytes=end_mem,
            start_tree_rss_bytes=start_tree.rss_bytes,
            end_tree_rss_bytes=end_tree.rss_bytes,
            peak_tree_rss_bytes=peak_tree,
            available_bytes=end_tree.available_bytes,
            n_processes=end_tree.n_processes,
        )
        data["report"] = report

        from dagster import get_dagster_logger
        logger = get_dagster_logger()
        logger.info(
            f"Resource Report [{name}]: "
            f"Duration: {report.duration_sec:.1f}s, "
            f"CPU: {report.cpu_percent:.1f}%, "
            f"Peak tree RAM: {report.peak_memory_mb:.1f} MB, "
            f"Delta tree RAM: {report.memory_delta_mb:+.1f} MB, "
            f"Available: {end_tree.available_mb:.1f} MB"
        )

        if context is not None:
            from dagster import MetadataValue

            clean_key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "resource_usage"
            context.add_output_metadata({
                f"{clean_key}_duration_sec": MetadataValue.float(report.duration_sec),
                f"{clean_key}_cpu_percent": MetadataValue.float(report.cpu_percent),
                f"{clean_key}_peak_memory_mb": MetadataValue.float(report.peak_memory_mb),
                f"{clean_key}_memory_delta_mb": MetadataValue.float(report.memory_delta_mb),
                f"{clean_key}_peak_tree_memory_mb": MetadataValue.float(report.peak_memory_mb),
                f"{clean_key}_available_memory_mb": MetadataValue.float(end_tree.available_mb),
                f"{clean_key}_n_processes": MetadataValue.int(report.n_processes),
            })
