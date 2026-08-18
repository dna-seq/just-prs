"""Background 1000G sensors must not steal an explicit CLI job."""

from __future__ import annotations

from prs_pipeline.sensors import (
    _background_score_and_push_blocked,
    _requested_startup_job,
)


def test_sample_scores_blocks_score_and_push(monkeypatch) -> None:
    monkeypatch.setenv("PRS_PIPELINE_STARTUP_JOB", "public_sample_scores_job")
    assert _requested_startup_job() == "public_sample_scores_job"
    assert _background_score_and_push_blocked() == "public_sample_scores_job"


def test_full_pipeline_allows_score_and_push(monkeypatch) -> None:
    monkeypatch.setenv("PRS_PIPELINE_STARTUP_JOB", "full_pipeline")
    assert _background_score_and_push_blocked() is None


def test_canary_alias_maps_to_public_sample_scores(monkeypatch) -> None:
    monkeypatch.setenv("PRS_PIPELINE_STARTUP_JOB", "canary_collapse_audit_job")
    assert _requested_startup_job() == "public_sample_scores_job"
    assert _background_score_and_push_blocked() == "public_sample_scores_job"
