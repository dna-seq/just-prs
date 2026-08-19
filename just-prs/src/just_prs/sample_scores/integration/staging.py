"""Exact-revision staging for the sample-score integration job."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from eliot import start_action

from just_prs.hf import _hf_download_with_retry, _resolve_token
from just_prs.sample_scores.integration.pins import (
    PINNED_FILES,
    SAMPLE_SCORES_REPO,
    SAMPLE_SCORES_REVISION,
    PinnedFile,
    StagingIndexRow,
)
from just_prs.scoring import parquet_cache_is_readable, resolve_cache_dir

ProgressLog = Callable[[str], None]


class StagingError(ValueError):
    """A pinned source file is missing, unreadable, or hash-mismatched."""


def integration_root(cache_dir: Path | None = None) -> Path:
    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    return root / "sample_scores" / "integration"


def sources_dir(cache_dir: Path | None = None, revision: str = SAMPLE_SCORES_REVISION) -> Path:
    return integration_root(cache_dir) / "sources" / revision


def work_cache_dir(cache_dir: Path | None = None) -> Path:
    """Isolated cache-shaped tree used by runtime/evidence validators."""
    return integration_root(cache_dir) / "work"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    if path.suffix == ".parquet" and parquet_cache_is_readable(path):
        return int(pl.scan_parquet(path).select(pl.len()).collect().item())
    return None


def _copy_verified(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _candidate_local_paths(pin: PinnedFile, cache_dir: Path) -> list[Path]:
    name = pin.local_name
    return [
        cache_dir / "sample_scores" / name,
        cache_dir / "metadata" / name,
        cache_dir / "percentiles" / name,
        sources_dir(cache_dir, pin.revision) / name,
    ]


def _download_pinned(pin: PinnedFile, dest: Path, token: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = dest.parent / f".dl-{pin.local_name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    downloaded = Path(
        _hf_download_with_retry(
            repo_id=pin.repo_id,
            filename=pin.repo_path,
            repo_type="dataset",
            local_dir=tmp_dir,
            token=token,
            revision=pin.revision,
        )
    )
    if not downloaded.exists():
        raise StagingError(f"download missing for {pin.repo_id}/{pin.repo_path}@{pin.revision}")
    _copy_verified(downloaded, dest)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _evidence_hashes_from_manifest(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hashes = payload.get("file_hashes") or payload.get("sha256") or {}
    if not isinstance(hashes, dict):
        return {}
    return {str(name): str(digest) for name, digest in hashes.items()}


def resolve_pins(pins: tuple[PinnedFile, ...] | None = None) -> list[PinnedFile]:
    """Fill empty evidence parquet hashes from the staged evidence manifest later."""
    return list(pins or PINNED_FILES)


def stage_pinned_sources(
    cache_dir: Path | None = None,
    *,
    allow_network: bool = True,
    local_files: Mapping[tuple[str, str], Path] | None = None,
    pins: tuple[PinnedFile, ...] | None = None,
    token: str | None = None,
    log: ProgressLog | None = None,
) -> list[StagingIndexRow]:
    """Download or copy every pin into an isolated staging directory.

    After this returns, callers must not refresh mutable HF HEAD. A missing or
    hash-mismatched file is a hard failure.
    """
    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    dest_root = sources_dir(root)
    dest_root.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    resolved_token = _resolve_token(token)
    local_files = local_files or {}
    pin_list = resolve_pins(pins)
    index: list[StagingIndexRow] = []

    with start_action(action_type="sample_scores:integration_stage"):
        for pin in pin_list:
            dest = dest_root / pin.local_name
            override = local_files.get((pin.repo_id, pin.repo_path))
            if override is not None:
                if not override.exists():
                    raise StagingError(f"local override missing: {override}")
                _copy_verified(override, dest)
            elif dest.exists() and (not pin.sha256 or sha256_file(dest) == pin.sha256):
                pass
            else:
                copied = False
                for candidate in _candidate_local_paths(pin, root):
                    if not candidate.exists() or candidate.resolve() == dest.resolve():
                        continue
                    digest = sha256_file(candidate)
                    if pin.sha256 and digest != pin.sha256:
                        continue
                    if not pin.sha256 and candidate.suffix == ".parquet" and not parquet_cache_is_readable(candidate):
                        continue
                    _copy_verified(candidate, dest)
                    copied = True
                    break
                if not copied:
                    if not allow_network:
                        raise StagingError(
                            f"offline staging miss for {pin.repo_id}/{pin.repo_path} "
                            f"@{pin.revision}"
                        )
                    _download_pinned(pin, dest, resolved_token)

            if not dest.exists():
                raise StagingError(f"staged file missing: {pin.local_name}")
            digest = sha256_file(dest)
            if pin.sha256 and digest != pin.sha256:
                raise StagingError(
                    f"hash mismatch for {pin.repo_path}: expected {pin.sha256}, got {digest}"
                )
            rows = _row_count(dest)
            if pin.expected_rows is not None and rows is not None and rows != pin.expected_rows:
                raise StagingError(
                    f"row count mismatch for {pin.repo_path}: expected {pin.expected_rows}, got {rows}"
                )
            if dest.suffix == ".parquet" and not parquet_cache_is_readable(dest):
                raise StagingError(f"unreadable parquet: {pin.local_name}")
            index.append(
                StagingIndexRow(
                    repo_id=pin.repo_id,
                    revision=pin.revision,
                    repo_path=pin.repo_path,
                    sha256=digest,
                    bytes=dest.stat().st_size,
                    rows=rows,
                    retrieved_at=retrieved_at,
                    local_name=pin.local_name,
                )
            )
            if log is not None:
                log(f"Staged {pin.repo_path} ({digest[:12]}…, rows={rows})")

        _apply_evidence_manifest_hashes(dest_root, index)
        _write_index(dest_root, index)
        materialize_work_tree(root, dest_root)
    return index


def _apply_evidence_manifest_hashes(dest_root: Path, index: list[StagingIndexRow]) -> None:
    manifest_path = dest_root / "evidence_manifest.json"
    if not manifest_path.exists():
        return
    hashes = _evidence_hashes_from_manifest(manifest_path)
    for row in index:
        if row.repo_id != SAMPLE_SCORES_REPO:
            continue
        expected = hashes.get(row.local_name) or hashes.get(f"data/{row.local_name}")
        if expected and row.sha256 != expected:
            raise StagingError(
                f"evidence manifest hash mismatch for {row.local_name}: "
                f"expected {expected}, got {row.sha256}"
            )


def _write_index(dest_root: Path, index: list[StagingIndexRow]) -> Path:
    path = dest_root / "staging_index.parquet"
    pl.DataFrame([row.model_dump() for row in index]).write_parquet(path)
    return path


def materialize_work_tree(cache_dir: Path, dest_root: Path) -> Path:
    """Copy staged files into an isolated cache-shaped tree (never the mutable cache)."""
    work = work_cache_dir(cache_dir)
    sample_dir = work / "sample_scores"
    metadata_dir = work / "metadata"
    percentiles_dir = work / "percentiles"
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    percentiles_dir.mkdir(parents=True, exist_ok=True)
    sample_names = {
        "samples.parquet",
        "runtime_results.parquet",
        "runtime_manifest.json",
        "sample_ancestry.parquet",
        "evidence_manifest.json",
        "traits.parquet",
        "score_trait_links.parquet",
        "papers.parquet",
        "score_paper_links.parquet",
        "guidelines.parquet",
        "guideline_trait_links.parquet",
        "actionability.parquet",
        "trait_contexts.parquet",
        "record_search_terms.parquet",
    }
    metadata_names = {
        "scores.parquet",
        "best_performance.parquet",
        "performance.parquet",
        "pgs_quality_scores.parquet",
        "publications.parquet",
        "trait_prevalence.parquet",
        "trait_heritability.parquet",
        "catalog_scoring_flags.parquet",
    }
    percentile_names = {
        "1000g_distributions.parquet",
        "1000g_quality.parquet",
        "1000g_distribution_quality_issues.parquet",
        "1000g_distribution_audit_summary.json",
    }
    for name in sample_names:
        src = dest_root / name
        if src.exists():
            _copy_verified(src, sample_dir / name)
    for name in metadata_names:
        src = dest_root / name
        if src.exists():
            _copy_verified(src, metadata_dir / name)
    for name in percentile_names:
        src = dest_root / name
        if src.exists():
            _copy_verified(src, percentiles_dir / name)
    return work


def load_staging_index(cache_dir: Path | None = None) -> list[StagingIndexRow]:
    path = sources_dir(cache_dir) / "staging_index.parquet"
    if not parquet_cache_is_readable(path):
        return []
    return [StagingIndexRow.model_validate(row) for row in pl.read_parquet(path).iter_rows(named=True)]


def staged_path(name: str, cache_dir: Path | None = None) -> Path:
    return sources_dir(cache_dir) / name


def file_sha256(path: Path) -> str:
    """Public alias used by docs/manifest hashing."""
    return sha256_file(path)
