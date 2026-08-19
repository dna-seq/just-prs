"""PGS-major public-sample scoring with bounded worker lifetime.

Two serial phases (unrestored, then restored). Each checkpoint prepares one
PGS set once and scores every publication-allowed sample. Scores with
≥ ``PRS_SAMPLE_SCORE_LARGE_VARIANT_THRESHOLD`` variants (default 1M) are
planned as singleton checkpoints so a 9.5M-variant file is never batched
with nine neighbors. DuckDB joins those files in
``PRS_SCORING_JOIN_CHUNK_SIZE`` chunks (same idea as reference-panel
genotype chunking). Workers persist an atomic part after every checkpoint
and exit on memory budget, safety floor, or after a large score whose
process-tree RSS is actually high. The parent starts a fresh worker at the first missing
checkpoint. A native worker death (SIGSEGV) isolates that checkpoint to
one PGS, records failed rows, and continues.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from just_prs.canary_audit import CanarySample
from just_prs.memory import (
    ProcessTreeSampler,
    check_memory_pressure,
    duckdb_limit_for_resident,
    large_score_recycle_rss_bytes,
    large_score_variant_threshold,
    process_tree_snapshot,
    recycle_reason,
    sample_interval_sec,
    sample_score_memory_budget_bytes,
    snapshot_metrics,
)
from just_prs.sample_scores.checkpoints import (
    CheckpointMeta,
    completed_pgs_for_profile,
    discover_valid_parts,
    make_checkpoint_key,
    reopen_failed_parts,
    reopen_invalid_parts,
    sample_set_fingerprint,
    scoring_set_fingerprint,
    write_runtime_part,
)
from just_prs.sample_scores.models import (
    RESTORED_PROFILE_ID,
    SCORE_ALGORITHM_VERSION,
    SCORE_PROFILES,
    UNRESTORED_PROFILE_ID,
    RuntimeResultRow,
    SampleRecord,
    ScoreProfile,
)
from just_prs.sample_scores.publish import (
    SampleScoreProgress,
    build_sample_record,
    failed_runtime_row,
    is_publication_allowed,
    merge_sample_records,
    normalized_public_parquet,
    public_sample_spec,
    runtime_row_from_result,
)
from just_prs.prs import PreparedGenotypeTables
from just_prs.scoring import ensure_scoring_file, resolve_cache_dir, scoring_parquet_path

DEFAULT_CHECKPOINT_SIZE = 10


class PreparedSample(BaseModel):
    """Publication-allowed sample ready for DuckDB scoring from parquet."""

    record: SampleRecord
    parquet_path: str
    vcf_path: str


class CheckpointWork(BaseModel):
    """One deterministic PGS-range checkpoint for a single profile."""

    meta: CheckpointMeta
    scoring_fingerprints: dict[str, str]


class WorkerWorkOrder(BaseModel):
    """JSON payload a scoring subprocess reads."""

    cache_dir: str
    scores_cache: str
    profile_id: str
    genome_build: str
    reference_restoration: bool
    universe_path: str | None = None
    universe_fingerprint: str | None = None
    samples: list[PreparedSample]
    checkpoints: list[CheckpointWork]
    traits: dict[str, str] = Field(default_factory=dict)
    just_prs_version: str
    computed_at: str
    pgs_total: int = 0
    checkpoints_total: int = 0
    pgs_already_done: int = 0
    checkpoints_already_done: int = 0
    progress_every: int = DEFAULT_CHECKPOINT_SIZE


class CheckpointReport(BaseModel):
    checkpoint_key: str
    n_rows: int
    n_ok: int
    n_failed: int
    duration_sec: float
    rss_mb: float
    peak_rss_mb: float
    available_mb: float
    recycle_reason: str | None = None


class WorkerReport(BaseModel):
    profile_id: str
    n_checkpoints: int = 0
    n_ok: int = 0
    n_failed: int = 0
    recycle_reason: str | None = None
    peak_rss_mb: float = 0.0
    checkpoints: list[CheckpointReport] = Field(default_factory=list)
    error: str | None = None


@dataclass
class PhaseProgress:
    n_ok: int = 0
    n_failed: int = 0
    n_cached: int = 0
    n_checkpoints: int = 0
    n_workers: int = 0
    recycle_reasons: list[str] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    checkpoint_reports: list[CheckpointReport] = field(default_factory=list)


def checkpoint_size() -> int:
    raw = os.environ.get("PRS_SAMPLE_SCORE_CHECKPOINT_SIZE", "").strip()
    return max(int(raw) if raw else DEFAULT_CHECKPOINT_SIZE, 1)


def scoring_variant_count(
    pgs_id: str,
    scores_cache: Path,
    genome_build: str,
    cache: dict[str, int | None] | None = None,
) -> int | None:
    """Parquet row count from metadata. None if the scoring file is missing."""
    if cache is not None and pgs_id in cache:
        return cache[pgs_id]
    path = scoring_parquet_path(pgs_id, scores_cache, genome_build)
    n: int | None
    if not path.exists():
        n = None
    else:
        import polars as pl

        n = int(pl.scan_parquet(path).select(pl.len()).collect().item())
    if cache is not None:
        cache[pgs_id] = n
    return n


def is_large_scoring_file(
    pgs_id: str,
    scores_cache: Path,
    genome_build: str,
    cache: dict[str, int | None] | None = None,
) -> bool:
    """True when the cached scoring parquet has ≥ the large-score threshold."""
    n = scoring_variant_count(pgs_id, scores_cache, genome_build, cache)
    return n is not None and n >= large_score_variant_threshold()


def pack_checkpoint_id_batches(
    pgs_ids: list[str],
    *,
    batch_size: int,
    large_ids: set[str],
) -> list[list[str]]:
    """Keep small IDs packed; emit each large score as its own batch.

    Order is preserved so resume stays deterministic. A 10-wide batch of
    9.5M-variant scores is what SIGSEGV'd the unrestored worker.
    """
    packed: list[list[str]] = []
    current: list[str] = []
    for pgs_id in pgs_ids:
        if pgs_id in large_ids:
            if current:
                packed.append(current)
                current = []
            packed.append([pgs_id])
            continue
        current.append(pgs_id)
        if len(current) >= batch_size:
            packed.append(current)
            current = []
    if current:
        packed.append(current)
    return packed


def worker_concurrency() -> int:
    raw = os.environ.get("PRS_SAMPLE_SCORE_WORKERS", "").strip()
    return max(int(raw) if raw else 1, 1)


def retry_failed_enabled() -> bool:
    """Re-score failed checkpoint rows. ``PRS_SAMPLE_SCORE_RETRY_FAILED``."""
    return os.environ.get("PRS_SAMPLE_SCORE_RETRY_FAILED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def repair_invalid_enabled() -> bool:
    """Quarantine invalid/stale ok parts. ``PRS_SAMPLE_SCORE_REPAIR_INVALID``."""
    return os.environ.get("PRS_SAMPLE_SCORE_REPAIR_INVALID", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def checkpoint_work_for_ids(work: CheckpointWork, pgs_ids: list[str]) -> CheckpointWork:
    """Shrink a planned checkpoint to the IDs that are still pending."""
    if pgs_ids == work.meta.pgs_ids:
        return work
    fps = {
        pgs_id: work.scoring_fingerprints[pgs_id]
        for pgs_id in pgs_ids
        if pgs_id in work.scoring_fingerprints
    }
    deferred = work.meta.scoring_set_fingerprint == "deferred"
    if deferred:
        key = (
            f"deferred-{pgs_ids[0]}"
            if len(pgs_ids) == 1
            else f"deferred-{pgs_ids[0]}_{pgs_ids[-1]}_n{len(pgs_ids)}"
        )
        scoring_fp = "deferred"
    else:
        scoring_fp = scoring_set_fingerprint(fps) if fps else work.meta.scoring_set_fingerprint
        key = make_checkpoint_key(
            score_profile_id=work.meta.score_profile_id,
            score_algorithm_version=work.meta.score_algorithm_version,
            genome_build=work.meta.genome_build,
            pgs_ids=pgs_ids,
            scoring_set_fp=scoring_fp,
            sample_set_fp=work.meta.sample_set_genotype_fingerprint,
            reference_universe_fp=work.meta.reference_universe_fingerprint,
        )
    meta = work.meta.model_copy(
        update={
            "pgs_ids": pgs_ids,
            "checkpoint_key": key,
            "scoring_set_fingerprint": scoring_fp,
            "n_rows": 0,
            "n_ok": 0,
            "n_failed": 0,
        }
    )
    return CheckpointWork(meta=meta, scoring_fingerprints=fps)


def format_sample_score_progress(
    *,
    profile_id: str,
    pgs_done: int,
    pgs_total: int,
    checkpoint_done: int,
    checkpoints_total: int,
    pgs_ids: list[str],
    n_ok: int,
    n_failed: int,
    n_samples: int,
    duration_sec: float,
    peak_rss_mb: float,
    available_mb: float,
    recycle_reason: str | None = None,
) -> str:
    """One line of catalog scoring progress for Dagster / CLI logs."""
    pct = (100.0 * pgs_done / pgs_total) if pgs_total else 100.0
    first = pgs_ids[0] if pgs_ids else "?"
    last = pgs_ids[-1] if pgs_ids else "?"
    span = first if first == last else f"{first}..{last}"
    extra = f", recycle={recycle_reason}" if recycle_reason else ""
    return (
        f"Sample scores {profile_id}: "
        f"{pgs_done}/{pgs_total} PGS ({pct:.1f}%), "
        f"checkpoint {checkpoint_done}/{checkpoints_total} {span}, "
        f"+{n_ok} ok / +{n_failed} failed this batch "
        f"({n_samples} samples), {duration_sec:.1f}s, "
        f"peak={peak_rss_mb:.0f} MB, avail={available_mb:.0f} MB{extra}"
    )


def _should_log_progress(pgs_done: int, pgs_batch: int, pgs_total: int, every: int) -> bool:
    if pgs_done >= pgs_total or every <= 1:
        return True
    previous = pgs_done - pgs_batch
    return (pgs_done // every) > (previous // every)


def worker_exit_recycle_reason(returncode: int | None) -> str | None:
    """Map a worker exit to a recycle reason. None means a clean exit."""
    if returncode is None or returncode == 0:
        return None
    if returncode < 0:
        sig = -returncode
        try:
            name = signal.Signals(sig).name.lower()
        except ValueError:
            name = f"signal_{sig}"
        return f"worker_{name}"
    return f"worker_exit_{returncode}"


def split_checkpoint_work(work: CheckpointWork) -> list[CheckpointWork]:
    """One CheckpointWork per PGS so a native crash can be isolated."""
    if len(work.meta.pgs_ids) <= 1:
        return [work]
    split: list[CheckpointWork] = []
    for pgs_id in work.meta.pgs_ids:
        fps = {pgs_id: work.scoring_fingerprints[pgs_id]} if pgs_id in work.scoring_fingerprints else {}
        meta = work.meta.model_copy(
            update={
                "pgs_ids": [pgs_id],
                "checkpoint_key": f"deferred-{pgs_id}",
                "scoring_set_fingerprint": "deferred",
                "n_rows": 0,
            }
        )
        split.append(CheckpointWork(meta=meta, scoring_fingerprints=fps))
    return split


def isolation_hint_dir(cache_dir: Path, profile_id: str) -> Path:
    return cache_dir / "sample_scores" / "isolate" / profile_id


def write_isolation_hint(
    cache_dir: Path,
    profile_id: str,
    pgs_ids: list[str],
    error: str,
) -> Path:
    """Remember a multi-PGS batch that killed the worker so resume skips it."""
    dest = isolation_hint_dir(cache_dir, profile_id)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{pgs_ids[0]}_{pgs_ids[-1]}_n{len(pgs_ids)}.json"
    payload = (
        json.dumps(
            {
                "pgs_ids": list(pgs_ids),
                "error": error,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    # Written right after a worker death, so the parent may itself be killed
    # mid-write. Stage + rename keeps the visible file either absent or complete.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return path


def apply_isolation_hints(
    planned: list[CheckpointWork],
    cache_dir: Path,
    profile_id: str,
) -> int:
    """Replace previously crashing batches with one-PGS work before retrying."""
    directory = isolation_hint_dir(cache_dir, profile_id)
    if not directory.exists():
        return 0
    hinted: set[tuple[str, ...]] = set()
    for path in directory.glob("*.json"):
        # A hint is a resume aid, so an unreadable one must never be fatal: a
        # truncated file from a killed parent would otherwise abort every later
        # resume. Discard it and let the batch be retried wide and re-isolated.
        raw = path.read_text(encoding="utf-8").strip()
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            path.unlink(missing_ok=True)
            continue
        ids = payload.get("pgs_ids") or []
        if len(ids) > 1:
            hinted.add(tuple(ids))
    if not hinted:
        return 0
    replaced: list[CheckpointWork] = []
    n_split = 0
    for item in planned:
        key = tuple(item.meta.pgs_ids)
        if key in hinted:
            replaced.extend(split_checkpoint_work(item))
            n_split += 1
        else:
            replaced.append(item)
    planned[:] = replaced
    return n_split


def reconcile_checkpoint_keys(
    planned: dict[str, list[CheckpointWork]],
    cache_dir: Path,
) -> set[str]:
    """Planned batch keys, plus isolated parts that replaced a crashed batch.

    A planned key is dropped only when every PGS in that batch already has a
    part on disk. Partial isolation keeps the planned key so compact still
    fails until the rest of the batch is written.
    """
    keys = {
        item.meta.checkpoint_key
        for works in planned.values()
        for item in works
    }
    discovery = discover_valid_parts(cache_dir, list(planned))
    disk = {meta.checkpoint_key for meta in discovery.valid}
    for profile_id, works in planned.items():
        done = discovery.completed_pgs_ids.get(profile_id, set())
        for work in works:
            if set(work.meta.pgs_ids) <= done and work.meta.checkpoint_key not in disk:
                keys.discard(work.meta.checkpoint_key)
    keys.update(disk)
    return keys


def record_worker_crash_failures(
    work: CheckpointWork,
    prepared: list[PreparedSample],
    *,
    cache_dir: Path,
    scores_cache: Path,
    profile: ScoreProfile,
    universe_fp: str | None,
    just_prs_version: str,
    computed_at: str,
    error: str,
) -> int:
    """Persist failed rows for a checkpoint the worker could not finish."""
    filled = materialize_checkpoint_fingerprints(
        work,
        scores_cache,
        profile,
        {},
        universe_fp=universe_fp,
    )
    rows = [
        failed_runtime_row(
            sample=sample.record,
            pgs_id=pgs_id,
            profile=profile,
            scoring_fingerprint_value=filled.scoring_fingerprints[pgs_id],
            reference_universe_fp=universe_fp if profile.reference_restoration else None,
            just_prs_version=just_prs_version,
            computed_at=computed_at,
            error=error,
        )
        for pgs_id in filled.meta.pgs_ids
        for sample in prepared
    ]
    meta = filled.meta.model_copy(update={"n_ok": 0, "n_failed": len(rows), "n_rows": len(rows)})
    write_runtime_part(rows, cache_dir, meta)
    return len(rows)


def inprocess_workers() -> bool:
    return os.environ.get("PRS_SAMPLE_SCORE_INPROCESS", "").strip().lower() in {
        "1", "true", "yes",
    }


def _catalog_pgs_ids(
    catalog: object,
    genome_build: str,
    requested: list[str] | None,
    limit: int | None,
) -> list[str]:
    lf = catalog.scores(  # type: ignore[attr-defined]
        genome_build=genome_build,
        include_harmonized=True,
        include_excluded=True,
    )
    ids = sorted(lf.select("pgs_id").unique().collect()["pgs_id"].to_list())
    if requested:
        wanted = [item.strip().upper() for item in requested if item.strip()]
        in_catalog = set(ids)
        ids = [pgs_id for pgs_id in wanted if pgs_id in in_catalog]
        ids.extend(pgs_id for pgs_id in wanted if pgs_id not in in_catalog)
    if limit is not None:
        ids = ids[:limit]
    return ids


def _scoring_fingerprint_for(
    pgs_id: str,
    genome_build: str,
    scores_cache: Path,
    cache: dict[str, str],
) -> str:
    from just_prs.sample_scores.fingerprints import scoring_file_fingerprint

    if pgs_id in cache:
        return cache[pgs_id]
    from just_prs.scoring import parquet_cache_is_readable, scoring_parquet_path

    path = scoring_parquet_path(pgs_id, scores_cache, genome_build)
    if not parquet_cache_is_readable(path):
        digest = hashlib.sha256(
            f"missing-scoring:{pgs_id}:{genome_build}".encode()
        ).hexdigest()
        cache[pgs_id] = digest
        return digest
    digest = scoring_file_fingerprint(path)
    cache[pgs_id] = digest
    return digest


def current_scoring_fingerprints(
    cache_dir: Path,
    pgs_ids: list[str] | None = None,
    scores_cache: Path | None = None,
) -> dict[str, str]:
    """Hash the current scoring parquets for a PGS-ID set (or already-scored IDs)."""
    scores = scores_cache if scores_cache is not None else cache_dir / "scores"
    ids = pgs_ids
    if ids is None:
        ids = sorted(
            completed_pgs_for_profile(cache_dir, UNRESTORED_PROFILE_ID)
            | completed_pgs_for_profile(cache_dir, RESTORED_PROFILE_ID)
        )
    cache: dict[str, str] = {}
    return {
        pgs_id: _scoring_fingerprint_for(pgs_id, "GRCh38", scores, cache)
        for pgs_id in ids
    }


def prepare_public_samples(
    samples: list[CanarySample],
    cache_dir: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[list[PreparedSample], list[str]]:
    """Normalize + hash publication-allowed genomes. Skip unknown/private."""
    del log
    prepared: list[PreparedSample] = []
    skipped: list[str] = []
    for sample in samples:
        spec = public_sample_spec(sample.label)
        if spec is None or not spec.publication_allowed:
            skipped.append(sample.label)
            continue
        record = build_sample_record(sample, cache_dir, spec=spec)
        if record is None:
            skipped.append(sample.label)
            continue
        parquet = normalized_public_parquet(cache_dir, record.sample_id, sample.vcf_path)
        prepared.append(
            PreparedSample(
                record=record,
                parquet_path=str(parquet),
                vcf_path=str(sample.vcf_path),
            )
        )
    if prepared:
        merge_sample_records([item.record for item in prepared], cache_dir)
    return prepared, skipped


def plan_checkpoints(
    pgs_ids: list[str],
    prepared: list[PreparedSample],
    profile: ScoreProfile,
    scores_cache: Path,
    fingerprint_cache: dict[str, str],
    *,
    universe_fp: str | None,
    batch_size: int | None = None,
    log: Callable[[str], None] | None = None,
    defer_fingerprints: bool = False,
) -> list[CheckpointWork]:
    """Split the PGS list into deterministic checkpoint work items.

    Scoring passes ``defer_fingerprints=True`` so the parent does not hash
    the whole catalog before the first worker starts. Completeness planning
    computes the real keys.
    """
    emit = log or (lambda _msg: None)
    size = batch_size if batch_size is not None else checkpoint_size()
    sample_fp = sample_set_fingerprint(
        {item.record.sample_id: item.record.genotype_sha256_v1 for item in prepared}
    )
    counts: dict[str, int | None] = {}
    large_ids = {
        pgs_id
        for pgs_id in pgs_ids
        if is_large_scoring_file(pgs_id, scores_cache, profile.genome_build, counts)
    }
    batches = pack_checkpoint_id_batches(pgs_ids, batch_size=size, large_ids=large_ids)
    if large_ids:
        emit(
            f"Sample scores {profile.score_profile_id}: "
            f"planned {len(large_ids)} large score(s) as singleton checkpoints "
            f"(≥{large_score_variant_threshold()} variants)"
        )
    work: list[CheckpointWork] = []
    n_ids = len(pgs_ids)
    done = 0
    for batch_i, batch in enumerate(batches):
        if defer_fingerprints:
            fps: dict[str, str] = {}
            scoring_fp = "deferred"
        else:
            if batch_i == 0 or batch_i % 20 == 0:
                emit(
                    f"Sample scores {profile.score_profile_id}: "
                    f"fingerprinting scoring files {done + 1}-{done + len(batch)}/{n_ids}"
                )
            fps = {
                pgs_id: _scoring_fingerprint_for(
                    pgs_id, profile.genome_build, scores_cache, fingerprint_cache
                )
                for pgs_id in batch
            }
            scoring_fp = scoring_set_fingerprint(fps)
        key = make_checkpoint_key(
            score_profile_id=profile.score_profile_id,
            score_algorithm_version=SCORE_ALGORITHM_VERSION,
            genome_build=profile.genome_build,
            pgs_ids=batch,
            scoring_set_fp=scoring_fp,
            sample_set_fp=sample_fp,
            reference_universe_fp=universe_fp if profile.reference_restoration else None,
        )
        meta = CheckpointMeta(
            checkpoint_key=key,
            score_profile_id=profile.score_profile_id,
            score_algorithm_version=SCORE_ALGORITHM_VERSION,
            genome_build=profile.genome_build,
            pgs_ids=batch,
            scoring_set_fingerprint=scoring_fp,
            sample_set_genotype_fingerprint=sample_fp,
            reference_universe_fingerprint=(
                universe_fp if profile.reference_restoration else None
            ),
            n_rows=len(batch) * len(prepared),
        )
        work.append(CheckpointWork(meta=meta, scoring_fingerprints=fps))
        done += len(batch)
    return work


def materialize_checkpoint_fingerprints(
    work: CheckpointWork,
    scores_cache: Path,
    profile: ScoreProfile,
    fingerprint_cache: dict[str, str],
    *,
    universe_fp: str | None,
) -> CheckpointWork:
    """Fill file-byte scoring fingerprints and the real checkpoint key."""
    if work.scoring_fingerprints and work.meta.scoring_set_fingerprint != "deferred":
        return work
    fps = {
        pgs_id: _scoring_fingerprint_for(
            pgs_id, profile.genome_build, scores_cache, fingerprint_cache
        )
        for pgs_id in work.meta.pgs_ids
    }
    scoring_fp = scoring_set_fingerprint(fps)
    key = make_checkpoint_key(
        score_profile_id=work.meta.score_profile_id,
        score_algorithm_version=work.meta.score_algorithm_version,
        genome_build=work.meta.genome_build,
        pgs_ids=work.meta.pgs_ids,
        scoring_set_fp=scoring_fp,
        sample_set_fp=work.meta.sample_set_genotype_fingerprint,
        reference_universe_fp=universe_fp if profile.reference_restoration else None,
    )
    meta = work.meta.model_copy(
        update={"checkpoint_key": key, "scoring_set_fingerprint": scoring_fp}
    )
    return work.model_copy(update={"meta": meta, "scoring_fingerprints": fps})


def score_checkpoint(
    work: CheckpointWork,
    prepared: list[PreparedSample],
    *,
    scores_cache: Path,
    profile: ScoreProfile,
    universe: object | None,
    universe_fp: str | None,
    traits: dict[str, str],
    just_prs_version: str,
    computed_at: str,
    duckdb_limit: str,
    genotype_tables: PreparedGenotypeTables | None = None,
) -> list[RuntimeResultRow]:
    """Prepare each PGS once, then score every public sample."""
    import polars as pl
    from just_prs.prs import compute_prs_duckdb

    work = materialize_checkpoint_fingerprints(
        work,
        scores_cache,
        profile,
        {},
        universe_fp=universe_fp,
    )
    import gc

    rows: list[RuntimeResultRow] = []
    for pgs_id in work.meta.pgs_ids:
        digest = work.scoring_fingerprints[pgs_id]
        trait = traits.get(pgs_id)
        large = is_large_scoring_file(pgs_id, scores_cache, profile.genome_build)
        try:
            scoring_path = ensure_scoring_file(pgs_id, scores_cache, profile.genome_build)
            scoring_lf = pl.scan_parquet(scoring_path)
        except Exception as exc:
            for sample in prepared:
                rows.append(
                    failed_runtime_row(
                        sample=sample.record,
                        pgs_id=pgs_id,
                        profile=profile,
                        scoring_fingerprint_value=digest,
                        reference_universe_fp=universe_fp if profile.reference_restoration else None,
                        just_prs_version=just_prs_version,
                        computed_at=computed_at,
                        error=str(exc),
                    )
                )
            continue
        for sample in prepared:
            if large:
                check_memory_pressure(pgs_id)
            try:
                result = compute_prs_duckdb(
                    vcf_path=sample.vcf_path,
                    scoring_file=scoring_lf,
                    genome_build=profile.genome_build,
                    cache_dir=scores_cache,
                    pgs_id=pgs_id,
                    trait_reported=trait,
                    genotypes_parquet=sample.parquet_path,
                    genotype_tables=genotype_tables,
                    genotype_table_key=sample.record.sample_id,
                    memory_limit=duckdb_limit,
                    genotype_input_mode=profile.genotype_input_mode,
                    maf_fill=profile.maf_fill,
                    reference_restoration=profile.reference_restoration,
                    reference_universe=universe if profile.reference_restoration else None,
                    sample_build=sample.record.genome_build,
                )
                rows.append(
                    runtime_row_from_result(
                        result,
                        sample=sample.record,
                        profile=profile,
                        scoring_fingerprint_value=digest,
                        reference_universe_fp=universe_fp if profile.reference_restoration else None,
                        just_prs_version=just_prs_version,
                        computed_at=computed_at,
                    )
                )
            except Exception as exc:
                rows.append(
                    failed_runtime_row(
                        sample=sample.record,
                        pgs_id=pgs_id,
                        profile=profile,
                        scoring_fingerprint_value=digest,
                        reference_universe_fp=universe_fp if profile.reference_restoration else None,
                        just_prs_version=just_prs_version,
                        computed_at=computed_at,
                        error=str(exc),
                    )
                )
        if large:
            gc.collect()
            check_memory_pressure(pgs_id)
    return rows


def run_worker(
    order: WorkerWorkOrder,
    log: Callable[[str], None] | None = None,
) -> WorkerReport:
    """Score checkpoints until the memory budget says to recycle."""
    from just_prs.prs import prepare_genotype_tables, prepare_reference_universe

    emit = log or (lambda _msg: None)
    cache_dir = Path(order.cache_dir)
    scores_cache = Path(order.scores_cache)
    profile = SCORE_PROFILES[order.profile_id]
    universe = None
    if order.reference_restoration and order.universe_path:
        universe = prepare_reference_universe(
            Path(order.universe_path),
            genome_build=order.genome_build,
        )
    resident = process_tree_snapshot()
    duckdb_limit = duckdb_limit_for_resident(resident.rss_bytes)
    budget = sample_score_memory_budget_bytes()
    report = WorkerReport(profile_id=order.profile_id)
    fingerprint_cache: dict[str, str] = {}
    pgs_done = order.pgs_already_done
    checkpoint_done = order.checkpoints_already_done
    pgs_total = order.pgs_total or sum(len(item.meta.pgs_ids) for item in order.checkpoints) + pgs_done
    checkpoints_total = order.checkpoints_total or (len(order.checkpoints) + checkpoint_done)
    every = max(order.progress_every, 1)
    n_samples = len(order.samples)
    emit(
        f"Sample scores {order.profile_id}: worker starting at "
        f"{pgs_done}/{pgs_total} PGS, {len(order.checkpoints)} checkpoints in this process"
    )
    sampler = ProcessTreeSampler(interval_sec=sample_interval_sec())
    sampler.start()
    genotype_tables = None
    existing = [
        sample for sample in order.samples if Path(sample.parquet_path).is_file()
    ]
    if existing and len(existing) == len(order.samples):
        genotype_tables = prepare_genotype_tables(
            {sample.record.sample_id: sample.parquet_path for sample in existing},
            memory_limit=duckdb_limit,
            genotype_input_mode=profile.genotype_input_mode,
        )
        emit(
            f"Sample scores {order.profile_id}: materialized "
            f"{len(existing)} genotype tables ({duckdb_limit})"
        )
    try:
        for work in order.checkpoints:
            work = materialize_checkpoint_fingerprints(
                work,
                scores_cache,
                profile,
                fingerprint_cache,
                universe_fp=order.universe_fingerprint,
            )
            started = time.perf_counter()
            rows = score_checkpoint(
                work,
                order.samples,
                scores_cache=scores_cache,
                profile=profile,
                universe=universe,
                universe_fp=order.universe_fingerprint,
                traits=order.traits,
                just_prs_version=order.just_prs_version,
                computed_at=order.computed_at,
                duckdb_limit=duckdb_limit,
                genotype_tables=genotype_tables,
            )
            n_ok = sum(1 for row in rows if row.status == "ok")
            n_failed = len(rows) - n_ok
            meta = work.meta.model_copy(
                update={"n_ok": n_ok, "n_failed": n_failed, "n_rows": len(rows)}
            )
            write_runtime_part(rows, cache_dir, meta)
            snap = sampler.snapshot()
            reason = recycle_reason(snap, budget_bytes=budget)
            if reason is None and any(
                is_large_scoring_file(pgs_id, scores_cache, profile.genome_build)
                for pgs_id in work.meta.pgs_ids
            ):
                # Singleton checkpoints already isolate a huge file. Only
                # kill the process when it has actually grown fat — recycling
                # after every ≥1M-variant score spawned ~1200 interpreters
                # overnight and froze the host at ~1.7 GB worker RSS.
                fat_after_large = large_score_recycle_rss_bytes()
                if snap.rss_bytes >= fat_after_large:
                    reason = "large_score"
            ckpt = CheckpointReport(
                checkpoint_key=meta.checkpoint_key,
                n_rows=len(rows),
                n_ok=n_ok,
                n_failed=n_failed,
                duration_sec=round(time.perf_counter() - started, 3),
                rss_mb=snap.rss_mb,
                peak_rss_mb=snap.peak_rss_mb,
                available_mb=snap.available_mb,
                recycle_reason=reason,
            )
            report.checkpoints.append(ckpt)
            report.n_checkpoints += 1
            report.n_ok += n_ok
            report.n_failed += n_failed
            report.peak_rss_mb = max(report.peak_rss_mb, snap.peak_rss_mb)
            pgs_done += len(work.meta.pgs_ids)
            checkpoint_done += 1
            if _should_log_progress(pgs_done, len(work.meta.pgs_ids), pgs_total, every) or reason:
                emit(
                    format_sample_score_progress(
                        profile_id=order.profile_id,
                        pgs_done=pgs_done,
                        pgs_total=pgs_total,
                        checkpoint_done=checkpoint_done,
                        checkpoints_total=checkpoints_total,
                        pgs_ids=work.meta.pgs_ids,
                        n_ok=n_ok,
                        n_failed=n_failed,
                        n_samples=n_samples,
                        duration_sec=ckpt.duration_sec,
                        peak_rss_mb=snap.peak_rss_mb,
                        available_mb=snap.available_mb,
                        recycle_reason=reason,
                    )
                )
            if reason:
                report.recycle_reason = reason
                break
    except Exception as exc:
        report.error = str(exc)
        raise
    finally:
        if genotype_tables is not None:
            genotype_tables.close()
        final = sampler.stop()
        report.peak_rss_mb = max(report.peak_rss_mb, final.peak_rss_mb)
    return report


def _write_work_order(order: WorkerWorkOrder, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(order.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _run_worker_subprocess(
    order: WorkerWorkOrder,
    work_path: Path,
    report_path: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> WorkerReport:
    emit = log or (lambda _msg: None)
    _write_work_order(order, work_path)
    cmd = [
        sys.executable,
        "-m",
        "just_prs.sample_scores.worker",
        str(work_path),
        str(report_path),
    ]
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    returncode: int | None = None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    sampler = ProcessTreeSampler(pid=proc.pid, interval_sec=sample_interval_sec())
    sampler.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip()
            if text:
                emit(text)
        returncode = proc.wait()
    finally:
        parent_snap = sampler.stop()
        if returncode is None:
            returncode = proc.poll()
    duration = time.perf_counter() - started
    crash = worker_exit_recycle_reason(returncode)
    if report_path.exists():
        report = WorkerReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    else:
        report = WorkerReport(
            profile_id=order.profile_id,
            error=f"worker exited {returncode} without a report",
        )
    report.peak_rss_mb = max(report.peak_rss_mb, parent_snap.peak_rss_mb)
    if crash and not report.recycle_reason:
        report.recycle_reason = crash
        if not report.error:
            report.error = f"worker exited {returncode}"
    emit(
        f"Sample-score worker pid={proc.pid} profile={order.profile_id}: "
        f"{report.n_checkpoints} checkpoints, peak={report.peak_rss_mb:.1f} MB, "
        f"available={parent_snap.available_mb:.1f} MB, "
        f"recycle={report.recycle_reason or 'done'}, {duration:.1f}s"
    )
    return report


def run_phase(
    profile: ScoreProfile,
    planned: list[CheckpointWork],
    prepared: list[PreparedSample],
    *,
    cache_dir: Path,
    scores_cache: Path,
    universe_path: Path | None,
    universe_fp: str | None,
    traits: dict[str, str],
    just_prs_version: str,
    computed_at: str,
    log: Callable[[str], None] | None = None,
    progress_every: int = DEFAULT_CHECKPOINT_SIZE,
    retry_failed: bool = False,
    repair_invalid: bool = False,
    current_scoring_fingerprints: dict[str, str] | None = None,
) -> PhaseProgress:
    """Resume missing checkpoints with recycled workers.

    ``retry_failed`` reopens parts that recorded ``failed`` rows, then
    continues into IDs that still have no part. Successful cached scores stay.
    """
    emit = log or (lambda _msg: None)
    if retry_failed:
        reopened = reopen_failed_parts(cache_dir, profile.score_profile_id)
        if reopened.pgs_ids:
            emit(
                f"Sample scores {profile.score_profile_id}: retrying "
                f"{len(reopened.pgs_ids)} failed PGS "
                f"(quarantined {reopened.n_parts_quarantined} part(s), "
                f"rewrote {reopened.n_parts_rewritten}); "
                f"successful cache and missing IDs continue"
            )
    if repair_invalid:
        from just_prs.sample_scores.completeness import ok_row_invariant_issues

        audited = reopen_invalid_parts(
            cache_dir,
            profile.score_profile_id,
            current_scoring_fingerprints=current_scoring_fingerprints,
            row_issues=ok_row_invariant_issues,
        )
        emit(
            f"Sample scores {profile.score_profile_id}: repair-invalid "
            f"audited {len(audited.pgs_ids)} PGS, "
            f"quarantined {audited.n_parts_quarantined} part(s) "
            f"({audited.n_invalid_ok_rows} invalid ok rows, "
            f"{audited.n_stale_fingerprint_rows} stale fingerprints); "
            f"re-scoring {audited.pgs_ids[0] + '..' + audited.pgs_ids[-1] if audited.pgs_ids else 'none'}"
        )
    n_isolated = apply_isolation_hints(planned, cache_dir, profile.score_profile_id)
    sample_ids = ", ".join(item.record.sample_id for item in prepared)
    pgs_total = sum(len(item.meta.pgs_ids) for item in planned)
    checkpoints_total = len(planned)
    if n_isolated:
        emit(
            f"Sample scores {profile.score_profile_id}: isolated {n_isolated} "
            f"previously crashing batch(es) into one-PGS checkpoints"
        )

    def _pending() -> list[CheckpointWork]:
        done = completed_pgs_for_profile(cache_dir, profile.score_profile_id)
        pending: list[CheckpointWork] = []
        for item in planned:
            leftover = [pgs_id for pgs_id in item.meta.pgs_ids if pgs_id not in done]
            if leftover:
                pending.append(checkpoint_work_for_ids(item, leftover))
        return pending

    remaining = _pending()
    cached_pgs = pgs_total - sum(len(item.meta.pgs_ids) for item in remaining)
    progress = PhaseProgress(n_cached=cached_pgs * len(prepared))
    emit(
        f"Sample scores {profile.score_profile_id}: {sample_ids} "
        f"({len(prepared)} samples); {len(remaining)}/{checkpoints_total} "
        f"checkpoints to run, {cached_pgs} PGS cached"
    )
    while remaining:
        head_ids = tuple(remaining[0].meta.pgs_ids)
        done_pgs = pgs_total - sum(len(item.meta.pgs_ids) for item in remaining)
        order = WorkerWorkOrder(
            cache_dir=str(cache_dir),
            scores_cache=str(scores_cache),
            profile_id=profile.score_profile_id,
            genome_build=profile.genome_build,
            reference_restoration=profile.reference_restoration,
            universe_path=str(universe_path) if universe_path else None,
            universe_fingerprint=universe_fp if profile.reference_restoration else None,
            samples=prepared,
            checkpoints=remaining,
            traits=traits,
            just_prs_version=just_prs_version,
            computed_at=computed_at,
            pgs_total=pgs_total,
            checkpoints_total=checkpoints_total,
            pgs_already_done=done_pgs,
            checkpoints_already_done=checkpoints_total - len(remaining),
            progress_every=progress_every,
        )
        work_dir = cache_dir / "sample_scores" / "worker"
        work_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        work_path = work_dir / f"{profile.score_profile_id}_{stamp}.json"
        report_path = work_dir / f"{profile.score_profile_id}_{stamp}.report.json"
        if inprocess_workers():
            report = run_worker(order, log=emit)
            report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        else:
            report = _run_worker_subprocess(order, work_path, report_path, log=emit)
        progress.n_workers += 1
        progress.n_ok += report.n_ok
        progress.n_failed += report.n_failed
        progress.n_checkpoints += report.n_checkpoints
        progress.peak_rss_mb = max(progress.peak_rss_mb, report.peak_rss_mb)
        progress.checkpoint_reports.extend(report.checkpoints)
        if report.recycle_reason:
            progress.recycle_reasons.append(report.recycle_reason)
        remaining = _pending()
        crashed = bool(report.recycle_reason and report.recycle_reason.startswith("worker_"))
        if crashed and remaining and tuple(remaining[0].meta.pgs_ids) == head_ids:
            head = remaining[0]
            crash_error = report.error or report.recycle_reason or "worker crashed"
            if len(head.meta.pgs_ids) > 1:
                singles = split_checkpoint_work(head)
                replaced: list[CheckpointWork] = []
                seen = False
                for item in planned:
                    if not seen and tuple(item.meta.pgs_ids) == head_ids:
                        replaced.extend(singles)
                        seen = True
                    else:
                        replaced.append(item)
                planned[:] = replaced
                checkpoints_total = len(planned)
                write_isolation_hint(
                    cache_dir,
                    profile.score_profile_id,
                    list(head_ids),
                    crash_error,
                )
                emit(
                    f"Sample scores {profile.score_profile_id}: worker crashed on "
                    f"{head.meta.pgs_ids[0]}..{head.meta.pgs_ids[-1]} ({crash_error}); "
                    f"retrying those {len(singles)} PGS one at a time"
                )
                remaining = _pending()
                continue
            n_failed = record_worker_crash_failures(
                head,
                prepared,
                cache_dir=cache_dir,
                scores_cache=scores_cache,
                profile=profile,
                universe_fp=universe_fp,
                just_prs_version=just_prs_version,
                computed_at=computed_at,
                error=crash_error,
            )
            progress.n_failed += n_failed
            emit(
                f"Sample scores {profile.score_profile_id}: recorded failure for "
                f"{head.meta.pgs_ids[0]} after {crash_error}; continuing"
            )
            remaining = _pending()
            continue
        if not remaining:
            break
        if tuple(remaining[0].meta.pgs_ids) == head_ids and not crashed:
            raise RuntimeError(
                f"Worker produced no checkpoints but {len(remaining)} remain"
            )
    return progress


def score_public_samples_pgs_major(
    samples: list[CanarySample],
    cache_dir: Path | None = None,
    *,
    pgs_ids: list[str] | None = None,
    limit: int | None = None,
    skip_existing: bool = True,
    retry_failed: bool = False,
    repair_invalid: bool = False,
    progress_every: int = DEFAULT_CHECKPOINT_SIZE,
    log: Callable[[str], None] | None = None,
) -> SampleScoreProgress:
    """Score publication-allowed samples under both WGS profiles, PGS-major."""
    from just_prs import __version__
    from just_prs.prs_catalog import PRSCatalog

    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    emit = log or (lambda _msg: None)
    retry_failed = retry_failed or retry_failed_enabled()
    repair_invalid = repair_invalid or repair_invalid_enabled()
    catalog = PRSCatalog(cache_dir=root)
    scores_cache = root / "scores"
    fingerprint_cache: dict[str, str] = {}
    computed_at = datetime.now(timezone.utc).isoformat()
    prepared, skipped = prepare_public_samples(samples, root, log=emit)
    progress = SampleScoreProgress(
        n_skipped_private=len(skipped),
        skipped_labels=skipped,
        published_sample_ids=[item.record.sample_id for item in prepared],
    )
    if skipped:
        emit(
            "Sample scores: skipped unpublished --vcf labels: " + ", ".join(skipped)
        )
    if not prepared:
        return progress

    ids = _catalog_pgs_ids(catalog, "GRCh38", pgs_ids, limit)
    emit(
        "Sample scores: "
        + ", ".join(item.record.sample_id for item in prepared)
        + f" ({len(prepared)} public samples), {len(ids)} PGS IDs"
    )
    trait_rows = (
        catalog.scores(genome_build="GRCh38", include_harmonized=True, include_excluded=True)
        .select("pgs_id", "trait_reported")
        .unique(subset=["pgs_id"])
        .collect()
    )
    traits = {
        str(row["pgs_id"]): str(row["trait_reported"])
        for row in trait_rows.iter_rows(named=True)
        if row.get("trait_reported")
    }
    universe_path = catalog._reference_universe_path("GRCh38")
    universe_fp: str | None = None
    if universe_path is not None and Path(universe_path).exists():
        from just_prs.sample_scores.fingerprints import reference_universe_fingerprint

        universe_fp = reference_universe_fingerprint(Path(universe_path))
    elif any(profile.reference_restoration for profile in SCORE_PROFILES.values()):
        handle = catalog.prepare_reference_universe("GRCh38", reference_restoration=True)
        if handle is not None:
            from just_prs.sample_scores.fingerprints import reference_universe_fingerprint

            universe_fp = reference_universe_fingerprint(
                handle.source_path or handle.frame
            )
            universe_path = catalog._reference_universe_path("GRCh38")

    current_fps: dict[str, str] | None = None
    if repair_invalid and skip_existing:
        existing_ids = sorted(
            completed_pgs_for_profile(root, UNRESTORED_PROFILE_ID)
            | completed_pgs_for_profile(root, RESTORED_PROFILE_ID)
        )
        current_fps = {
            pgs_id: _scoring_fingerprint_for(pgs_id, "GRCh38", scores_cache, fingerprint_cache)
            for pgs_id in existing_ids
        }
        emit(
            f"Sample scores: repair-invalid hashing {len(current_fps)} current "
            "scoring-file fingerprints"
        )

    # Unrestored first so canary can reuse those rows without the universe resident.
    for profile_id in (UNRESTORED_PROFILE_ID, RESTORED_PROFILE_ID):
        profile = SCORE_PROFILES[profile_id]
        planned = plan_checkpoints(
            ids,
            prepared,
            profile,
            scores_cache,
            fingerprint_cache,
            universe_fp=universe_fp,
            log=emit,
            defer_fingerprints=True,
        )
        if not skip_existing:
            for meta_path in (root / "sample_scores" / "parts" / "runtime" / profile_id).glob("*.parquet"):
                meta_path.unlink(missing_ok=True)
                sidecar = Path(str(meta_path) + ".meta.json")
                sidecar.unlink(missing_ok=True)
            planned_work = planned
        else:
            planned_work = planned
        phase = run_phase(
            profile,
            planned_work,
            prepared,
            cache_dir=root,
            scores_cache=scores_cache,
            universe_path=Path(universe_path) if universe_path else None,
            universe_fp=universe_fp,
            traits=traits,
            just_prs_version=__version__,
            computed_at=computed_at,
            progress_every=progress_every,
            retry_failed=retry_failed and skip_existing,
            repair_invalid=repair_invalid and skip_existing,
            current_scoring_fingerprints=current_fps,
            log=emit,
        )
        progress.n_total += len(ids) * len(prepared)
        progress.n_ok += phase.n_ok
        progress.n_failed += phase.n_failed
        progress.n_cached += phase.n_cached
        progress.failed_ids.extend(
            ckpt.checkpoint_key for ckpt in phase.checkpoint_reports if ckpt.n_failed
        )
        progress.peak_rss_mb = max(progress.peak_rss_mb, phase.peak_rss_mb)
        progress.recycle_reasons.extend(phase.recycle_reasons)
        emit(
            f"Sample scores {profile_id}: {phase.n_ok} ok, {phase.n_failed} failed, "
            f"{phase.n_cached} cached rows, {phase.n_workers} workers, "
            f"peak={phase.peak_rss_mb:.1f} MB, recycles={phase.recycle_reasons}"
        )
    return progress


def expected_checkpoint_keys(
    samples: list[CanarySample],
    cache_dir: Path,
    *,
    pgs_ids: list[str] | None = None,
    limit: int | None = None,
) -> set[str]:
    """Keys the completeness check expects after a planned run."""
    from just_prs.prs_catalog import PRSCatalog

    catalog = PRSCatalog(cache_dir=cache_dir)
    prepared, _skipped = prepare_public_samples(samples, cache_dir)
    ids = _catalog_pgs_ids(catalog, "GRCh38", pgs_ids, limit)
    scores_cache = cache_dir / "scores"
    fingerprint_cache: dict[str, str] = {}
    universe_fp: str | None = None
    universe_path = catalog._reference_universe_path("GRCh38")
    if universe_path is not None and Path(universe_path).exists():
        from just_prs.sample_scores.fingerprints import reference_universe_fingerprint

        universe_fp = reference_universe_fingerprint(Path(universe_path))
    planned_by_profile: dict[str, list[CheckpointWork]] = {}
    for profile in SCORE_PROFILES.values():
        planned_by_profile[profile.score_profile_id] = plan_checkpoints(
            ids,
            prepared,
            profile,
            scores_cache,
            fingerprint_cache,
            universe_fp=universe_fp,
        )
    return reconcile_checkpoint_keys(planned_by_profile, cache_dir)
