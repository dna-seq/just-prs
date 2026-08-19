"""Atomic sample-score HF allowlists and parent-revision guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from just_prs.hf import (
    EVIDENCE_SAMPLE_SCORES_FILES,
    RUNTIME_SAMPLE_SCORES_FILES,
    SampleScorePublishError,
    assert_sample_score_parent_revision,
    sample_score_commit_operations,
    sample_score_integration_commit_operations,
)
from just_prs.sample_scores.evidence.build import EVIDENCE_OWNED_FILES


def test_commit_operations_are_exactly_owned_paths(tmp_path: Path) -> None:
    for name in (*RUNTIME_SAMPLE_SCORES_FILES, *EVIDENCE_SAMPLE_SCORES_FILES):
        (tmp_path / name).write_text("ok\n", encoding="utf-8")
    (tmp_path / "identity_cache.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("no\n", encoding="utf-8")
    parts = tmp_path / "parts" / "runtime"
    parts.mkdir(parents=True)
    (parts / "chunk.parquet").write_text("no\n", encoding="utf-8")

    runtime_ops = sample_score_commit_operations(tmp_path, RUNTIME_SAMPLE_SCORES_FILES)
    evidence_ops = sample_score_commit_operations(tmp_path, EVIDENCE_SAMPLE_SCORES_FILES)
    runtime_names = {Path(str(op.path_in_repo)).name for op in runtime_ops}
    evidence_names = {Path(str(op.path_in_repo)).name for op in evidence_ops}
    assert runtime_names == set(RUNTIME_SAMPLE_SCORES_FILES)
    assert evidence_names == set(EVIDENCE_SAMPLE_SCORES_FILES)
    assert runtime_names.isdisjoint(evidence_names)
    assert "identity_cache.json" not in runtime_names
    assert "README.md" not in evidence_names
    assert set(EVIDENCE_OWNED_FILES) == set(EVIDENCE_SAMPLE_SCORES_FILES)


def test_commit_operations_reject_local_paths_and_cache_artifacts(tmp_path: Path) -> None:
    with pytest.raises(SampleScorePublishError, match="forbidden"):
        sample_score_commit_operations(tmp_path, ("identity_cache.json",))
    with pytest.raises(SampleScorePublishError, match="non-flat"):
        sample_score_commit_operations(tmp_path, ("parts/chunk.parquet",))
    with pytest.raises(SampleScorePublishError, match="forbidden"):
        sample_score_commit_operations(tmp_path, ("README.md",))
    with pytest.raises(SampleScorePublishError, match="forbidden"):
        sample_score_commit_operations(tmp_path, ("ANALYSIS.md",))


def test_integration_commit_operations_are_exactly_owned_paths(tmp_path: Path) -> None:
    for name in (
        "model_analysis.parquet",
        "trait_summaries.parquet",
        "manifest.json",
        "README.md",
        "AGENTS.md",
        "ANALYSIS.md",
    ):
        (tmp_path / name).write_text("ok\n", encoding="utf-8")
    ops = sample_score_integration_commit_operations(tmp_path)
    assert [str(op.path_in_repo) for op in ops] == [
        "data/model_analysis.parquet",
        "data/trait_summaries.parquet",
        "data/manifest.json",
        "README.md",
        "AGENTS.md",
        "ANALYSIS.md",
    ]


def test_parent_revision_guard_rejects_changed_head() -> None:
    assert_sample_score_parent_revision("abc", "abc")
    assert_sample_score_parent_revision("abc", None)
    with pytest.raises(SampleScorePublishError, match="parent revision changed"):
        assert_sample_score_parent_revision("new-head", "old-parent")
