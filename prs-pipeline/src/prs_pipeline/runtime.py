"""Resource tracking for Dagster assets.

Provides a context manager that records CPU, memory, and wall-clock time
for compute-heavy assets. Metrics are logged to the Dagster UI via
``context.add_output_metadata`` and to the Dagster logger.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from typing import Any, Optional

import psutil
from pydantic import BaseModel


class ResourceReport(BaseModel):
    """Snapshot of resource consumption for a single asset execution."""

    name: str
    duration_sec: float
    cpu_percent: float
    peak_memory_mb: float
    memory_delta_mb: float
    start_mem_bytes: int
    end_mem_bytes: int


@contextmanager
def resource_tracker(name: str = "resource_usage", context: Optional[Any] = None):
    """Track execution time, CPU and peak memory for a block of code.

    Args:
        name: Human-readable label (usually the asset name).
        context: Optional ``AssetExecutionContext``.  When provided, metrics
            are attached as Dagster output metadata so they appear in the UI.

    Yields a mutable dict.  After the block finishes the dict contains a
    ``"report"`` key with a :class:`ResourceReport`.
    """
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss

    process.cpu_percent(interval=None)

    data: dict[str, Any] = {"name": name, "start_time": start_time, "start_mem": start_mem}
    yield data

    end_time = time.perf_counter()
    end_mem = process.memory_info().rss
    cpu_usage = process.cpu_percent(interval=None)

    report = ResourceReport(
        name=name,
        duration_sec=round(end_time - start_time, 2),
        cpu_percent=round(cpu_usage, 1),
        peak_memory_mb=round(max(start_mem, end_mem) / (1024 * 1024), 2),
        memory_delta_mb=round((end_mem - start_mem) / (1024 * 1024), 2),
        start_mem_bytes=start_mem,
        end_mem_bytes=end_mem,
    )
    data["report"] = report

    from dagster import get_dagster_logger
    logger = get_dagster_logger()
    logger.info(
        f"Resource Report [{name}]: "
        f"Duration: {report.duration_sec:.1f}s, "
        f"CPU: {report.cpu_percent:.1f}%, "
        f"Peak RAM: {report.peak_memory_mb:.1f} MB, "
        f"Delta RAM: {report.memory_delta_mb:+.1f} MB"
    )

    if context is not None:
        from dagster import MetadataValue

        clean_key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "resource_usage"
        context.add_output_metadata({
            f"{clean_key}_duration_sec": MetadataValue.float(report.duration_sec),
            f"{clean_key}_cpu_percent": MetadataValue.float(report.cpu_percent),
            f"{clean_key}_peak_memory_mb": MetadataValue.float(report.peak_memory_mb),
            f"{clean_key}_memory_delta_mb": MetadataValue.float(report.memory_delta_mb),
        })
