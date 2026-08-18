"""Process-tree RSS sampling for memory-safe pipeline workers.

``resource_tracker`` historically recorded only start/end RSS of the current
process. That cannot capture peak usage, child-worker trees, or a SIGKILL.
This module samples the full process tree on a timer so a parent can emit
per-checkpoint metrics before a worker is recycled or killed.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psutil
from pydantic import BaseModel

_DEFAULT_SAMPLE_INTERVAL_SEC = 1.0


class ProcessTreeSnapshot(BaseModel):
    """One sample of a process tree plus host available memory."""

    pid: int
    rss_bytes: int
    peak_rss_bytes: int
    available_bytes: int
    total_bytes: int
    n_processes: int
    sampled_at: float
    missing_children: int = 0

    @property
    def rss_mb(self) -> float:
        return round(self.rss_bytes / (1024 * 1024), 2)

    @property
    def peak_rss_mb(self) -> float:
        return round(self.peak_rss_bytes / (1024 * 1024), 2)

    @property
    def available_mb(self) -> float:
        return round(self.available_bytes / (1024 * 1024), 2)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def sample_interval_sec() -> float:
    """Seconds between RSS samples. Override with ``PRS_MEMORY_SAMPLE_INTERVAL``."""
    return max(_env_float("PRS_MEMORY_SAMPLE_INTERVAL", _DEFAULT_SAMPLE_INTERVAL_SEC), 0.1)


def process_tree_rss_bytes(pid: int | None = None) -> tuple[int, int, int]:
    """Return ``(rss_bytes, n_processes, missing_children)`` for ``pid`` and descendants.

    A vanished child is counted as missing rather than treated as zero RSS so a
    parent can tell sampling noise from a real drop.
    """
    root_pid = pid if pid is not None else os.getpid()
    try:
        root = psutil.Process(root_pid)
    except psutil.Error:
        return 0, 0, 1
    rss = 0
    n_processes = 0
    missing = 0
    try:
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        procs = [root]
        missing += 1
    for proc in procs:
        try:
            rss += int(proc.memory_info().rss)
            n_processes += 1
        except psutil.Error:
            missing += 1
    return rss, n_processes, missing


def process_tree_snapshot(
    pid: int | None = None,
    *,
    peak_rss_bytes: int = 0,
) -> ProcessTreeSnapshot:
    """Point-in-time process-tree RSS plus host available/total memory."""
    root_pid = pid if pid is not None else os.getpid()
    rss, n_processes, missing = process_tree_rss_bytes(root_pid)
    host = psutil.virtual_memory()
    return ProcessTreeSnapshot(
        pid=root_pid,
        rss_bytes=rss,
        peak_rss_bytes=max(peak_rss_bytes, rss),
        available_bytes=int(host.available),
        total_bytes=int(host.total),
        n_processes=n_processes,
        sampled_at=time.time(),
        missing_children=missing,
    )


class ProcessTreeSampler:
    """Background sampler that tracks peak process-tree RSS for one PID."""

    def __init__(
        self,
        pid: int | None = None,
        interval_sec: float | None = None,
    ) -> None:
        self.pid = pid if pid is not None else os.getpid()
        self.interval_sec = interval_sec if interval_sec is not None else sample_interval_sec()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest = process_tree_snapshot(self.pid)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._latest = process_tree_snapshot(self.pid)
        self._thread = threading.Thread(
            target=self._run,
            name=f"rss-sampler-{self.pid}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            snap = process_tree_snapshot(self.pid, peak_rss_bytes=self.peak_rss_bytes)
            with self._lock:
                self._latest = snap

    @property
    def peak_rss_bytes(self) -> int:
        with self._lock:
            return self._latest.peak_rss_bytes

    def snapshot(self) -> ProcessTreeSnapshot:
        snap = process_tree_snapshot(self.pid, peak_rss_bytes=self.peak_rss_bytes)
        with self._lock:
            self._latest = snap
        return snap

    def stop(self) -> ProcessTreeSnapshot:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec * 2, 1.0))
            self._thread = None
        return self.snapshot()


@contextmanager
def sample_process_tree(
    pid: int | None = None,
    interval_sec: float | None = None,
) -> Iterator[ProcessTreeSampler]:
    """Sample process-tree RSS until the block exits."""
    sampler = ProcessTreeSampler(pid=pid, interval_sec=interval_sec)
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()


def memory_safety_floor_bytes() -> int:
    """Host RAM that must stay free. ``PRS_MEMORY_SAFETY_PERCENT`` / ``_MIN_MB``."""
    total = psutil.virtual_memory().total
    pct_raw = os.environ.get("PRS_MEMORY_SAFETY_PERCENT", "").strip()
    pct = int(pct_raw) if pct_raw else 10
    min_raw = os.environ.get("PRS_MEMORY_SAFETY_MIN_MB", "").strip()
    min_mb = int(min_raw) if min_raw else 512
    return max(int(total * pct / 100), min_mb * 1024 * 1024)


def check_memory_pressure(label: str) -> None:
    """Raise ``MemoryError`` if available RAM is below the safety floor.

    Same guard the 1000G reference scorer uses before each genotype chunk:
    fail this item instead of letting the OOM killer take the machine.
    """
    floor_bytes = memory_safety_floor_bytes()
    available = psutil.virtual_memory().available
    if available < floor_bytes:
        raise MemoryError(
            f"Available RAM ({available / (1024 * 1024):.0f} MB) dropped below "
            f"safety floor ({floor_bytes / (1024 * 1024):.0f} MB) while scoring "
            f"{label}. Aborting to avoid OOM-killing other processes."
        )


MIN_SCORING_JOIN_CHUNK = 250_000
MAX_SCORING_JOIN_CHUNK = 2_000_000
DEFAULT_LARGE_SCORE_VARIANTS = 1_000_000

# Resident cost of one arrow-registered scoring row, and the share of available RAM
# the chunk may claim. Measured across catalog scoring parquets (position, alleles,
# weight, frequency columns).
_SCORING_CHUNK_ROW_BYTES = 128
_SCORING_CHUNK_MEMORY_PERCENT = 5


def scoring_join_chunk_size(variants_remaining: int) -> int:
    """Scoring-file rows registered with DuckDB per join chunk.

    The chunk bounds only what crosses into DuckDB per iteration: the
    reference-universe join runs in SQL against the universe parquet, so a larger
    chunk buys fewer universe scans at the cost of resident bytes. Measured on a
    6.9M-variant score with restoration on: 250K rows = 6.8 s / 2.1 GB,
    1M = 3.7 s / 2.5 GB, 2M = 2.1 s / 3.2 GB (the pre-DuckDB polars join was
    3.3 s / 7.6 GB). The default is therefore auto-sized from available RAM between
    ``MIN_SCORING_JOIN_CHUNK`` and ``MAX_SCORING_JOIN_CHUNK`` rather than pinned to a
    constant; ``PRS_SCORING_JOIN_CHUNK_SIZE`` overrides it exactly.
    """
    remaining = max(int(variants_remaining), 0)
    if remaining == 0:
        return 0
    raw = os.environ.get("PRS_SCORING_JOIN_CHUNK_SIZE", "").strip()
    if raw:
        return min(max(int(raw), 1), remaining)
    usable = max(psutil.virtual_memory().available - memory_safety_floor_bytes(), 0)
    affordable = int(usable * _SCORING_CHUNK_MEMORY_PERCENT / 100 / _SCORING_CHUNK_ROW_BYTES)
    chunk = min(max(affordable, MIN_SCORING_JOIN_CHUNK), MAX_SCORING_JOIN_CHUNK)
    return min(chunk, remaining)


def large_score_variant_threshold() -> int:
    """Variant count that forces a singleton public-sample checkpoint."""
    raw = os.environ.get("PRS_SAMPLE_SCORE_LARGE_VARIANT_THRESHOLD", "").strip()
    return max(int(raw) if raw else DEFAULT_LARGE_SCORE_VARIANTS, 1)


def sample_score_memory_budget_bytes() -> int:
    """Worker RSS budget. ``PRS_SAMPLE_SCORE_MEMORY_LIMIT_GB`` or percent of RAM."""
    explicit = os.environ.get("PRS_SAMPLE_SCORE_MEMORY_LIMIT_GB", "").strip()
    if explicit:
        return int(float(explicit) * 1024 * 1024 * 1024)
    total = psutil.virtual_memory().total
    pct_raw = os.environ.get("PRS_SAMPLE_SCORE_MEMORY_PERCENT", "").strip()
    pct = int(pct_raw) if pct_raw else 65
    return int(total * pct / 100)


def duckdb_limit_for_resident(resident_rss_bytes: int) -> str:
    """DuckDB cap after subtracting resident RSS and the safety floor.

    Scoring workers already hold a reference universe. DuckDB must not
    independently claim a percentage of total RAM on top of that.
    """
    host = psutil.virtual_memory()
    safety = memory_safety_floor_bytes()
    budget = sample_score_memory_budget_bytes()
    remaining_budget = max(budget - resident_rss_bytes, 0)
    remaining_host = max(int(host.available) - safety, 0)
    usable = min(remaining_budget, remaining_host)
    # Leave half of the leftover for genotype/scoring frames outside DuckDB.
    duckdb_bytes = max(usable // 2, 1024 * 1024 * 1024)
    return f"{duckdb_bytes / (1024 ** 3):.1f}GB"


def recycle_reason(
    snapshot: ProcessTreeSnapshot,
    *,
    budget_bytes: int | None = None,
) -> str | None:
    """Return why a worker should exit, or None if it may continue."""
    budget = budget_bytes if budget_bytes is not None else sample_score_memory_budget_bytes()
    if snapshot.rss_bytes >= budget:
        return "memory_budget"
    if snapshot.available_bytes <= memory_safety_floor_bytes():
        return "safety_floor"
    return None


def snapshot_metrics(snapshot: ProcessTreeSnapshot, **extra: Any) -> dict[str, Any]:
    """Flat metrics dict for Dagster metadata / checkpoint reports."""
    payload: dict[str, Any] = {
        "pid": snapshot.pid,
        "rss_mb": snapshot.rss_mb,
        "peak_rss_mb": snapshot.peak_rss_mb,
        "available_mb": snapshot.available_mb,
        "n_processes": snapshot.n_processes,
        "sampled_at": snapshot.sampled_at,
    }
    payload.update(extra)
    return payload
