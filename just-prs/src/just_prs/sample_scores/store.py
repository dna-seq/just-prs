"""Local parquet IO for the public sample-score dataset."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from eliot import start_action

from just_prs.sample_scores.models import (
    DEFAULT_SAMPLE_SCORES_REPO,
    RuntimeResultRow,
    SampleAncestryRecord,
    SampleRecord,
)
from just_prs.scoring import parquet_cache_is_readable, resolve_cache_dir

SAMPLES_FILENAME = "samples.parquet"
RUNTIME_RESULTS_FILENAME = "runtime_results.parquet"
MANIFEST_FILENAME = "manifest.json"
EVIDENCE_MANIFEST_FILENAME = "evidence_manifest.json"
RUNTIME_MANIFEST_FILENAME = "runtime_manifest.json"
SAMPLE_ANCESTRY_FILENAME = "sample_ancestry.parquet"


def sample_scores_dir(cache_dir: Path | None = None) -> Path:
    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    return root / "sample_scores"


def samples_path(cache_dir: Path | None = None) -> Path:
    return sample_scores_dir(cache_dir) / SAMPLES_FILENAME


def runtime_results_path(cache_dir: Path | None = None) -> Path:
    return sample_scores_dir(cache_dir) / RUNTIME_RESULTS_FILENAME


def sample_ancestry_path(cache_dir: Path | None = None) -> Path:
    return sample_scores_dir(cache_dir) / SAMPLE_ANCESTRY_FILENAME


def load_samples(cache_dir: Path | None = None) -> list[SampleRecord]:
    path = samples_path(cache_dir)
    if not parquet_cache_is_readable(path):
        return []
    frame = pl.read_parquet(path)
    return [SampleRecord.model_validate(row) for row in frame.iter_rows(named=True)]


def write_samples(samples: list[SampleRecord], cache_dir: Path | None = None) -> Path:
    path = samples_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([sample.model_dump() for sample in samples]).write_parquet(path)
    return path


def load_runtime_results(cache_dir: Path | None = None) -> pl.DataFrame:
    path = runtime_results_path(cache_dir)
    if not parquet_cache_is_readable(path):
        return pl.DataFrame()
    return pl.read_parquet(path)


def load_sample_ancestry(cache_dir: Path | None = None) -> list[SampleAncestryRecord]:
    path = sample_ancestry_path(cache_dir)
    if not parquet_cache_is_readable(path):
        return []
    frame = pl.read_parquet(path)
    return [SampleAncestryRecord.model_validate(row) for row in frame.iter_rows(named=True)]


def write_sample_ancestry(
    rows: list[SampleAncestryRecord],
    cache_dir: Path | None = None,
) -> Path:
    path = sample_ancestry_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row.model_dump() for row in rows]).write_parquet(path)
    return path


def write_runtime_results(
    rows: list[RuntimeResultRow] | pl.DataFrame,
    cache_dir: Path | None = None,
) -> Path:
    path = runtime_results_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(
        [row.model_dump() for row in rows]
    )
    frame.write_parquet(path)
    return path


def lookup_runtime_row(
    *,
    sample_id: str,
    pgs_id: str,
    scoring_build: str,
    score_profile_id: str,
    scoring_fingerprint: str | None = None,
    cache_dir: Path | None = None,
) -> RuntimeResultRow | None:
    """Return the published runtime row when keys (and optional fingerprint) match."""
    frame = load_runtime_results(cache_dir)
    if frame.is_empty():
        return None
    filtered = frame.filter(
        (pl.col("sample_id") == sample_id)
        & (pl.col("pgs_id") == pgs_id)
        & (pl.col("scoring_build") == scoring_build)
        & (pl.col("score_profile_id") == score_profile_id)
        & (pl.col("status") == "ok")
    )
    if scoring_fingerprint is not None and "scoring_fingerprint" in filtered.columns:
        filtered = filtered.filter(pl.col("scoring_fingerprint") == scoring_fingerprint)
    if filtered.is_empty():
        return None
    return RuntimeResultRow.model_validate(filtered.row(0, named=True))


def ensure_local_dataset(
    cache_dir: Path | None = None,
    *,
    repo_id: str = DEFAULT_SAMPLE_SCORES_REPO,
    pull: bool = True,
) -> Path:
    """Return the local sample-scores directory, pulling from HF when asked and missing."""
    local = sample_scores_dir(cache_dir)
    if parquet_cache_is_readable(samples_path(cache_dir)):
        return local
    if not pull:
        return local
    with start_action(action_type="sample_scores:ensure_local_dataset", repo_id=repo_id):
        from just_prs.hf import pull_sample_scores

        pull_sample_scores(local, repo_id=repo_id)
    return local
