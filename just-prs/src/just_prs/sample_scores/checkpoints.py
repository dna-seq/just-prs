"""Immutable parquet checkpoint parts for public sample scoring.

Parquet is not appendable. The failed layout reloaded, concatenated, and
rewrote the whole runtime table after every handful of rows. This module
writes one atomic part per checkpoint, discovers completed work lazily from
sidecar metadata, quarantines corrupt/stale parts, and compacts once.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from just_prs.sample_scores.models import (
    RESTORED_PROFILE_ID,
    RuntimeResultRow,
    SCORE_ALGORITHM_VERSION,
    UNRESTORED_PROFILE_ID,
)
from just_prs.sample_scores.store import (
    RUNTIME_RESULTS_FILENAME,
    sample_scores_dir,
)
from just_prs.scoring import parquet_cache_is_readable

PARTS_SUBDIR = "parts/runtime"
CANARY_PARTS_SUBDIR = "parts/canary"
QUARANTINE_DIRNAME = "_quarantine"
META_SUFFIX = ".meta.json"
PART_SCHEMA_VERSION = 1

RUNTIME_KEY = (
    "sample_id",
    "pgs_id",
    "scoring_build",
    "score_profile_id",
    "scoring_fingerprint",
)

RUNTIME_RESULT_COLUMNS = list(RuntimeResultRow.model_fields)


class CheckpointMeta(BaseModel):
    """Sidecar describing one immutable runtime (or canary) part."""

    checkpoint_key: str
    schema_version: int = PART_SCHEMA_VERSION
    kind: str = "runtime"
    score_profile_id: str
    score_algorithm_version: str = SCORE_ALGORITHM_VERSION
    genome_build: str = "GRCh38"
    pgs_ids: list[str]
    scoring_set_fingerprint: str
    sample_set_genotype_fingerprint: str
    reference_universe_fingerprint: str | None = None
    n_rows: int
    n_ok: int = 0
    n_failed: int = 0
    written_at: str = ""

    def expected_path(self, cache_dir: Path) -> Path:
        return part_path(cache_dir, self.score_profile_id, self.checkpoint_key, kind=self.kind)


class CheckpointDiscovery(BaseModel):
    """Lazy resume view: valid parts plus quarantined/stale keys."""

    valid: list[CheckpointMeta] = Field(default_factory=list)
    completed_pgs_ids: dict[str, set[str]] = Field(default_factory=dict)
    quarantined: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)


def parts_dir(cache_dir: Path, profile_id: str, *, kind: str = "runtime") -> Path:
    root = sample_scores_dir(cache_dir)
    subdir = PARTS_SUBDIR if kind == "runtime" else CANARY_PARTS_SUBDIR
    return root / subdir / profile_id


def quarantine_dir(cache_dir: Path, *, kind: str = "runtime") -> Path:
    root = sample_scores_dir(cache_dir)
    subdir = PARTS_SUBDIR if kind == "runtime" else CANARY_PARTS_SUBDIR
    return root / subdir / QUARANTINE_DIRNAME


def part_path(
    cache_dir: Path,
    profile_id: str,
    checkpoint_key: str,
    *,
    kind: str = "runtime",
) -> Path:
    return parts_dir(cache_dir, profile_id, kind=kind) / f"{checkpoint_key}.parquet"


def meta_path(parquet_path: Path) -> Path:
    return Path(str(parquet_path) + ".meta.json")


def _meta_path_for(parquet_path: Path) -> Path:
    return meta_path(parquet_path)


def scoring_set_fingerprint(pgs_to_fp: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"scoring_set_v1\n")
    for pgs_id in sorted(pgs_to_fp):
        digest.update(f"{pgs_id}\t{pgs_to_fp[pgs_id]}\n".encode())
    return digest.hexdigest()


def sample_set_fingerprint(sample_id_to_geno: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"sample_set_v1\n")
    for sample_id in sorted(sample_id_to_geno):
        digest.update(f"{sample_id}\t{sample_id_to_geno[sample_id]}\n".encode())
    return digest.hexdigest()


def make_checkpoint_key(
    *,
    score_profile_id: str,
    score_algorithm_version: str,
    genome_build: str,
    pgs_ids: list[str],
    scoring_set_fp: str,
    sample_set_fp: str,
    reference_universe_fp: str | None,
) -> str:
    """Short deterministic filename key. Full identity lives in the sidecar."""
    digest = hashlib.sha256()
    digest.update(score_profile_id.encode())
    digest.update(b"\0")
    digest.update(score_algorithm_version.encode())
    digest.update(b"\0")
    digest.update(genome_build.encode())
    digest.update(b"\0")
    digest.update(",".join(pgs_ids).encode())
    digest.update(b"\0")
    digest.update(scoring_set_fp.encode())
    digest.update(b"\0")
    digest.update(sample_set_fp.encode())
    digest.update(b"\0")
    digest.update((reference_universe_fp or "").encode())
    short = digest.hexdigest()[:16]
    first = pgs_ids[0] if pgs_ids else "empty"
    last = pgs_ids[-1] if pgs_ids else "empty"
    return f"{first}_{last}_n{len(pgs_ids)}_{short}"


def _atomic_replace(tmp: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(dest)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _atomic_replace(tmp, path)
    return path


def write_parquet_atomic(frame: pl.DataFrame, dest: Path) -> Path:
    """Write parquet to a temp path, validate readability, then rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    frame.write_parquet(tmp)
    if not parquet_cache_is_readable(tmp):
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Wrote unreadable parquet: {tmp}")
    _atomic_replace(tmp, dest)
    return dest


def _validate_runtime_frame(frame: pl.DataFrame, meta: CheckpointMeta) -> None:
    missing = [col for col in RUNTIME_KEY if col not in frame.columns]
    if missing:
        raise ValueError(f"Part {meta.checkpoint_key} missing columns: {missing}")
    if frame.height != meta.n_rows:
        raise ValueError(
            f"Part {meta.checkpoint_key} row count {frame.height} != meta {meta.n_rows}"
        )
    if frame.height:
        dupes = frame.group_by(list(RUNTIME_KEY)).len().filter(pl.col("len") > 1)
        if dupes.height:
            raise ValueError(f"Part {meta.checkpoint_key} has duplicate runtime keys")
        part_ids = set(frame["pgs_id"].unique().to_list())
        if part_ids != set(meta.pgs_ids):
            raise ValueError(
                f"Part {meta.checkpoint_key} PGS IDs {sorted(part_ids)} "
                f"!= meta {meta.pgs_ids}"
            )


def write_runtime_part(
    rows: list[RuntimeResultRow] | pl.DataFrame,
    cache_dir: Path,
    meta: CheckpointMeta,
) -> Path:
    """Validate and atomically persist one checkpoint part + sidecar."""
    frame = (
        rows
        if isinstance(rows, pl.DataFrame)
        else pl.DataFrame([row.model_dump() for row in rows])
    )
    if "error" in frame.columns:
        frame = frame.with_columns(pl.col("error").cast(pl.Utf8))
    if not meta.written_at:
        meta.written_at = datetime.now(timezone.utc).isoformat()
    _validate_runtime_frame(frame, meta)
    dest = meta.expected_path(cache_dir)
    sidecar = _meta_path_for(dest)
    write_parquet_atomic(frame, dest)
    if not parquet_cache_is_readable(dest):
        dest.unlink(missing_ok=True)
        raise ValueError(f"Part failed readability after rename: {dest}")
    write_json_atomic(sidecar, meta.model_dump())
    return dest


def _quarantine(path: Path, cache_dir: Path, *, kind: str, reason: str) -> None:
    dest_dir = quarantine_dir(cache_dir, kind=kind)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"{path.stem}.{stamp}{path.suffix}"
    if path.exists():
        shutil.move(str(path), str(dest))
    sidecar = _meta_path_for(path)
    if sidecar.exists():
        shutil.move(str(sidecar), str(dest_dir / f"{sidecar.stem}.{stamp}{sidecar.suffix}"))
    note = dest_dir / f"{path.stem}.{stamp}.reason.txt"
    note.write_text(reason + "\n", encoding="utf-8")


def _load_meta(sidecar: Path) -> CheckpointMeta | None:
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return CheckpointMeta.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def iter_part_metas(
    cache_dir: Path,
    profile_id: str,
    *,
    kind: str = "runtime",
) -> Iterator[tuple[Path, CheckpointMeta | None]]:
    """Yield ``(parquet, meta_or_none)`` without reading parquet bytes."""
    directory = parts_dir(cache_dir, profile_id, kind=kind)
    if not directory.exists():
        return
    for parquet in sorted(directory.glob("*.parquet")):
        if parquet.name.endswith(".tmp"):
            continue
        yield parquet, _load_meta(_meta_path_for(parquet))


def discover_valid_parts(
    cache_dir: Path,
    profile_ids: Iterable[str] | None = None,
    *,
    kind: str = "runtime",
    expected: dict[str, CheckpointMeta] | None = None,
    expected_keys: set[str] | None = None,
    validate_parquet: bool = False,
) -> CheckpointDiscovery:
    """Lazy resume discovery. Optionally quarantine corrupt/stale parts.

    ``validate_parquet=False`` (default) only reads sidecars and existence.
    Set True before compaction so a truncated part is recomputed.
    """
    profiles = list(profile_ids) if profile_ids is not None else [
        UNRESTORED_PROFILE_ID,
        RESTORED_PROFILE_ID,
    ]
    discovery = CheckpointDiscovery()
    for profile_id in profiles:
        for parquet, meta in iter_part_metas(cache_dir, profile_id, kind=kind):
            if meta is None:
                _quarantine(parquet, cache_dir, kind=kind, reason="missing_or_invalid_sidecar")
                discovery.quarantined.append(parquet.name)
                continue
            if expected_keys is not None and meta.checkpoint_key not in expected_keys:
                discovery.stale.append(meta.checkpoint_key)
                continue
            if expected is not None:
                if meta.checkpoint_key not in expected:
                    _quarantine(parquet, cache_dir, kind=kind, reason="unexpected_checkpoint_key")
                    discovery.stale.append(meta.checkpoint_key)
                    continue
                want = expected[meta.checkpoint_key]
                if (
                    meta.scoring_set_fingerprint != want.scoring_set_fingerprint
                    or meta.sample_set_genotype_fingerprint != want.sample_set_genotype_fingerprint
                    or meta.reference_universe_fingerprint != want.reference_universe_fingerprint
                    or meta.score_algorithm_version != want.score_algorithm_version
                    or meta.pgs_ids != want.pgs_ids
                ):
                    _quarantine(parquet, cache_dir, kind=kind, reason="stale_fingerprint")
                    discovery.stale.append(meta.checkpoint_key)
                    continue
            if validate_parquet:
                if not parquet_cache_is_readable(parquet):
                    _quarantine(parquet, cache_dir, kind=kind, reason="unreadable_parquet")
                    discovery.quarantined.append(meta.checkpoint_key)
                    continue
                try:
                    frame = pl.scan_parquet(parquet).collect()
                    _validate_runtime_frame(frame, meta)
                except (pl.exceptions.ComputeError, OSError, ValueError) as exc:
                    _quarantine(parquet, cache_dir, kind=kind, reason=f"invalid_part:{exc}")
                    discovery.quarantined.append(meta.checkpoint_key)
                    continue
            elif not parquet.exists():
                discovery.quarantined.append(meta.checkpoint_key)
                continue
            discovery.valid.append(meta)
            discovery.completed_pgs_ids.setdefault(profile_id, set()).update(meta.pgs_ids)
    return discovery


def completed_pgs_for_profile(cache_dir: Path, profile_id: str) -> set[str]:
    return discover_valid_parts(cache_dir, [profile_id]).completed_pgs_ids.get(profile_id, set())


class FailedPartReopen(BaseModel):
    """How many failed checkpoint parts were opened for retry."""

    pgs_ids: list[str] = Field(default_factory=list)
    n_parts_quarantined: int = 0
    n_parts_rewritten: int = 0


def reopen_failed_parts(
    cache_dir: Path,
    profile_id: str,
    *,
    kind: str = "runtime",
) -> FailedPartReopen:
    """Drop failed rows so those PGS IDs are pending again.

    All-failed parts are quarantined. Mixed parts keep only PGS IDs whose
    rows are all ``ok``, so a 10-wide batch with one crash-recorded failure
    does not throw away the nine successful scores. Missing parts are
    untouched — resume still continues into uncached IDs.
    """
    report = FailedPartReopen()
    failed: set[str] = set()
    for parquet, meta in iter_part_metas(cache_dir, profile_id, kind=kind):
        if meta is None or meta.n_failed <= 0:
            continue
        if not parquet_cache_is_readable(parquet):
            _quarantine(parquet, cache_dir, kind=kind, reason="retry_failed_unreadable")
            report.n_parts_quarantined += 1
            failed.update(meta.pgs_ids)
            continue
        frame = pl.read_parquet(parquet)
        if "status" not in frame.columns:
            _quarantine(parquet, cache_dir, kind=kind, reason="retry_failed_missing_status")
            report.n_parts_quarantined += 1
            failed.update(meta.pgs_ids)
            continue
        failed_ids = {
            str(pgs_id)
            for pgs_id in frame.filter(pl.col("status") != "ok")["pgs_id"].unique().to_list()
        }
        if not failed_ids:
            continue
        failed.update(failed_ids)
        keep_ids = [pgs_id for pgs_id in meta.pgs_ids if pgs_id not in failed_ids]
        if not keep_ids:
            _quarantine(parquet, cache_dir, kind=kind, reason="retry_failed")
            report.n_parts_quarantined += 1
            continue
        kept = frame.filter(~pl.col("pgs_id").is_in(failed_ids))
        rewritten = meta.model_copy(
            update={
                "pgs_ids": keep_ids,
                "n_rows": kept.height,
                "n_ok": kept.height,
                "n_failed": 0,
                "written_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_runtime_part(kept, cache_dir, rewritten)
        report.n_parts_rewritten += 1
    report.pgs_ids = sorted(failed)
    return report


def compact_runtime_parts(
    cache_dir: Path,
    *,
    dest: Path | None = None,
    expected_keys: set[str] | None = None,
    kind: str = "runtime",
) -> Path:
    """Lazy-concat valid parts into one parquet, validate, atomically replace."""
    discovery = discover_valid_parts(
        cache_dir,
        kind=kind,
        expected_keys=expected_keys,
        validate_parquet=True,
    )
    if expected_keys is not None:
        found = {meta.checkpoint_key for meta in discovery.valid}
        missing = expected_keys - found
        if missing:
            raise ValueError(
                f"Cannot compact: {len(missing)} checkpoint(s) missing "
                f"(examples: {sorted(missing)[:5]})"
            )
    paths = [meta.expected_path(cache_dir) for meta in discovery.valid]
    target = dest if dest is not None else sample_scores_dir(cache_dir) / RUNTIME_RESULTS_FILENAME
    if not paths:
        empty = pl.DataFrame({col: [] for col in RUNTIME_RESULT_COLUMNS})
        return write_parquet_atomic(empty, target)
    # Lazy concat with relaxed schema so all-ok parts (error=null) unify
    # with failed parts (error=utf8). sink once.
    combined = pl.concat([pl.scan_parquet(path) for path in paths], how="diagonal_relaxed")
    tmp = target.with_name(target.name + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    combined.sink_parquet(tmp)
    if not parquet_cache_is_readable(tmp):
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Compacted parquet is unreadable: {tmp}")
    compacted = pl.scan_parquet(tmp)
    n_rows = int(compacted.select(pl.len()).collect().item())
    expected_rows = sum(meta.n_rows for meta in discovery.valid)
    if n_rows != expected_rows:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Compaction row count {n_rows} != sum of parts {expected_rows}"
        )
    keys = compacted.group_by(list(RUNTIME_KEY)).len().filter(pl.col("len") > 1).collect()
    if keys.height:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Compaction produced {keys.height} duplicate runtime keys")
    _atomic_replace(tmp, target)
    return target
