"""Process-tree RSS sampling is peak-aware, not start/end only."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

from just_prs.memory import (
    MAX_SCORING_JOIN_CHUNK,
    MIN_SCORING_JOIN_CHUNK,
    ProcessTreeSampler,
    check_memory_pressure,
    process_tree_rss_bytes,
    process_tree_snapshot,
    recycle_reason,
    sample_process_tree,
    scoring_join_chunk_size,
)


def test_process_tree_rss_includes_current_process() -> None:
    rss, n_procs, _missing = process_tree_rss_bytes()
    assert rss > 0
    assert n_procs >= 1


def test_sampler_tracks_peak_not_just_endpoints() -> None:
    sampler = ProcessTreeSampler(interval_sec=0.1)
    start = sampler.snapshot()
    sampler.start()
    blob = bytearray(8 * 1024 * 1024)
    time.sleep(0.25)
    mid = sampler.snapshot()
    del blob
    end = sampler.stop()
    assert mid.peak_rss_bytes >= start.rss_bytes
    assert end.peak_rss_bytes >= start.rss_bytes
    assert end.available_bytes > 0


def test_sample_process_tree_context_and_recycle(monkeypatch) -> None:
    with sample_process_tree(interval_sec=0.1) as sampler:
        snap = sampler.snapshot()
        assert snap.pid == os.getpid()
        assert recycle_reason(snap, budget_bytes=snap.rss_bytes + 10**12) is None
        assert recycle_reason(snap, budget_bytes=0) == "memory_budget"
        monkeypatch.setattr(
            "just_prs.memory.memory_safety_floor_bytes",
            lambda: snap.available_bytes + 1,
        )
        assert recycle_reason(snap, budget_bytes=10**18) == "safety_floor"


def test_check_memory_pressure_raises_below_floor(monkeypatch) -> None:
    check_memory_pressure("ok")
    monkeypatch.setattr("just_prs.memory.memory_safety_floor_bytes", lambda: 10**18)
    with pytest.raises(MemoryError, match="safety floor"):
        check_memory_pressure("PGS005178")


def test_scoring_join_chunk_size_honors_env(monkeypatch) -> None:
    monkeypatch.delenv("PRS_SCORING_JOIN_CHUNK_SIZE", raising=False)
    assert MIN_SCORING_JOIN_CHUNK <= scoring_join_chunk_size(9_500_000) <= MAX_SCORING_JOIN_CHUNK
    # Never hand back more rows than remain, and never a chunk for no work.
    assert scoring_join_chunk_size(100) == 100
    assert scoring_join_chunk_size(0) == 0
    monkeypatch.setenv("PRS_SCORING_JOIN_CHUNK_SIZE", "10")
    assert scoring_join_chunk_size(77) == 10
    assert scoring_join_chunk_size(5) == 5


def test_scoring_join_chunk_size_scales_with_available_memory(monkeypatch) -> None:
    """Auto-sizing tracks free RAM and stays inside the measured safe band."""
    monkeypatch.delenv("PRS_SCORING_JOIN_CHUNK_SIZE", raising=False)
    monkeypatch.setattr("just_prs.memory.memory_safety_floor_bytes", lambda: 0)

    def with_available(available_bytes: int) -> int:
        monkeypatch.setattr(
            "just_prs.memory.psutil.virtual_memory",
            lambda: SimpleNamespace(available=available_bytes, total=available_bytes),
        )
        return scoring_join_chunk_size(9_500_000)

    starved = with_available(64 * 1024 * 1024)
    roomy = with_available(256 * 1024 * 1024 * 1024)
    assert starved == MIN_SCORING_JOIN_CHUNK
    assert roomy == MAX_SCORING_JOIN_CHUNK
    assert starved < roomy
