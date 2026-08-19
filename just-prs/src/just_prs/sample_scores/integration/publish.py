"""Orchestrate staging, analysis, docs, and the allowlisted integration commit."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from eliot import start_action

from just_prs.hf import (
    DEFAULT_HF_SAMPLE_SCORES_REPO,
    pull_sample_scores,
    push_sample_score_integration,
)
from just_prs.sample_scores.integration.build import (
    build_model_analysis,
    pgs_without_trustworthy_distribution,
)
from just_prs.sample_scores.integration.checks import (
    IntegrationCheckReport,
    validate_integration_outputs,
)
from just_prs.sample_scores.integration.docs import write_final_docs
from just_prs.sample_scores.integration.pins import (
    SAMPLE_SCORES_REPO,
    SAMPLE_SCORES_REVISION,
    PinnedFile,
)
from just_prs.sample_scores.integration.staging import (
    stage_pinned_sources,
    staged_path,
    work_cache_dir,
)
from just_prs.sample_scores.integration.summaries import build_trait_summaries
from just_prs.sample_scores.store import sample_scores_dir
from just_prs.scoring import parquet_cache_is_readable, resolve_cache_dir

ProgressLog = Callable[[str], None]


@dataclass
class IntegrationBuildResult:
    output_dir: Path
    n_analysis_rows: int
    n_eligible: int
    n_summaries: int
    check: IntegrationCheckReport
    published_revision: str | None = None


def _runtime_manifest_payload(cache_dir: Path | None) -> dict[str, object]:
    path = staged_path("runtime_manifest.json", cache_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_sample_score_integration(
    cache_dir: Path | None = None,
    *,
    allow_network: bool = True,
    push: bool = False,
    log: ProgressLog | None = None,
    progress_every: int = 10,
    local_files: dict[tuple[str, str], Path] | None = None,
    pins: tuple[PinnedFile, ...] | None = None,
    token: str | None = None,
    parent_commit: str = SAMPLE_SCORES_REVISION,
    repo_id: str = SAMPLE_SCORES_REPO,
    recompute_summary_checks: bool = True,
) -> IntegrationBuildResult:
    """Stage pins, build analysis/summaries/docs, optionally publish."""
    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    output_dir = sample_scores_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_every = max(int(progress_every), 1)
    with start_action(action_type="sample_scores:integration_build"):
        if log:
            log("Integration: staging pinned sources")
        sources = stage_pinned_sources(
            root,
            allow_network=allow_network,
            local_files=local_files,
            pins=pins,
            token=token,
            log=log,
        )
        if log:
            log("Integration: building model_analysis")
        analysis = build_model_analysis(root, log=log, progress_every=progress_every)
        if log:
            log(f"Integration: model_analysis {analysis.height} rows")
        summaries = build_trait_summaries(
            analysis, log=log, progress_every=progress_every
        )
        if log:
            log(f"Integration: trait_summaries {summaries.height} rows")
        runtime = pl.read_parquet(staged_path("runtime_results.parquet", root))
        distributions = pl.read_parquet(staged_path("1000g_distributions.parquet", root))
        issues = pl.read_parquet(staged_path("1000g_distribution_quality_issues.parquet", root))
        error_pgs = set(issues.filter(pl.col("severity") == "ERROR")["pgs_id"].to_list()) if issues.height else set()
        missing = pgs_without_trustworthy_distribution(runtime, distributions, error_pgs)
        manifest_payload = _runtime_manifest_payload(root)
        write_final_docs(
            output_dir,
            analysis=analysis,
            summaries=summaries,
            sources=sources,
            scoring_set_fingerprint=str(manifest_payload.get("scoring_set_fingerprint") or "") or None,
            sample_set_fingerprint=str(manifest_payload.get("sample_set_fingerprint") or "") or None,
            reference_universe_fingerprint=str(manifest_payload.get("reference_universe_fingerprint") or "") or None,
            n_pgs_ids=int(runtime["pgs_id"].n_unique()),
            pgs_without_distribution=missing,
            omitted_catalog_tables=["score_development_ancestry.parquet"],
            parent_revision=parent_commit,
        )
        check = validate_integration_outputs(
            cache_dir=root,
            analysis=analysis,
            summaries=summaries,
            runtime=runtime,
            recompute_summaries=recompute_summary_checks,
        )
        check.raise_if_failed()
        revision = None
        if push:
            if log:
                log("Integration: publishing allowlisted integration commit")
            revision = push_sample_score_integration(
                output_dir,
                repo_id=repo_id,
                token=token,
                parent_commit=parent_commit,
            )
        return IntegrationBuildResult(
            output_dir=output_dir,
            n_analysis_rows=analysis.height,
            n_eligible=int(analysis.filter(pl.col("analysis_eligible")).height),
            n_summaries=summaries.height,
            check=check,
            published_revision=revision,
        )


def publish_sample_score_integration(
    local_dir: Path,
    *,
    repo_id: str = DEFAULT_HF_SAMPLE_SCORES_REPO,
    token: str | None = None,
    parent_commit: str = SAMPLE_SCORES_REVISION,
) -> str | None:
    return push_sample_score_integration(
        local_dir,
        repo_id=repo_id,
        token=token,
        parent_commit=parent_commit,
    )


def clean_room_verify(
    revision: str,
    dest: Path,
    *,
    repo_id: str = SAMPLE_SCORES_REPO,
    token: str | None = None,
) -> Path:
    """Download one exact revision into an empty directory."""
    dest.mkdir(parents=True, exist_ok=True)
    pull_sample_scores(dest, repo_id=repo_id, token=token, revision=revision)
    if not parquet_cache_is_readable(dest / "model_analysis.parquet"):
        raise FileNotFoundError("clean-room download missing model_analysis.parquet")
    return dest
