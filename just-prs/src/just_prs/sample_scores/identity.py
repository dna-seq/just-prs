"""Source-file and canonical genotype hashes for public-sample identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any

import polars as pl
from eliot import start_action

from just_prs.sample_scores.models import (
    HASH_SCHEMA_VERSION,
    NORMALIZATION_PROFILE_ID,
    SampleRecord,
)

_CHUNK = 1 << 20
_AUTOSOME = {str(i) for i in range(1, 23)}


def normalize_chrom(chrom: str) -> str:
    """Strip a ``chr`` prefix and canonicalize mitochondrial contig names."""
    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    upper = value.upper()
    if upper in {"M", "MT"}:
        return "MT"
    if upper in {"X", "Y"}:
        return upper
    return value


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    if chrom in _AUTOSOME:
        return (int(chrom), "")
    if chrom == "X":
        return (23, "")
    if chrom == "Y":
        return (24, "")
    if chrom == "MT":
        return (25, "")
    return (26, chrom)


def source_sha256(path: Path) -> str:
    """SHA-256 of the exact source bytes (VCF, gzip, or other)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _genotype_token(value: Any) -> str:
    if value is None:
        return "."
    if isinstance(value, (list, tuple)):
        alleles = [str(part) for part in value if part not in (None, "")]
        if not alleles:
            return "."
        return "/".join(sorted(alleles))
    text = str(value).strip()
    return text if text else "."


def genotype_sha256_v1(
    genotypes: pl.DataFrame | pl.LazyFrame,
    *,
    normalization_profile_id: str = NORMALIZATION_PROFILE_ID,
    hash_schema_version: int = HASH_SCHEMA_VERSION,
) -> str:
    """SHA-256 of the versioned canonical genotype stream.

    Stream format (UTF-8), after a one-line preamble:

        {chrom}\\t{pos}\\t{ref}\\t{alt}\\t{gt}\\n

    Rows are sorted by canonical chromosome, position, ref, alt. ``gt`` is
    sorted alleles joined by ``/``, or ``.`` when missing. Independent of
    path, gzip, parquet compression, and row-group layout.
    """
    frame = genotypes.collect() if isinstance(genotypes, pl.LazyFrame) else genotypes
    required = {"chrom", "pos", "ref", "alt"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Genotype frame missing columns for identity hash: {sorted(missing)}")
    gt_col = "genotype" if "genotype" in frame.columns else ("GT" if "GT" in frame.columns else None)
    if gt_col is None:
        raise ValueError("Genotype frame must have a 'genotype' or 'GT' column")

    work = frame.select(
        pl.col("chrom").cast(pl.Utf8).alias("chrom"),
        pl.col("pos").cast(pl.Int64).alias("pos"),
        pl.col("ref").cast(pl.Utf8).alias("ref"),
        pl.col("alt").cast(pl.Utf8).alias("alt"),
        pl.col(gt_col).alias("gt_raw"),
    )
    chroms = [normalize_chrom(value) for value in work["chrom"].to_list()]
    work = work.with_columns(pl.Series("chrom_norm", chroms))
    order = sorted(
        range(work.height),
        key=lambda idx: (
            _chrom_sort_key(chroms[idx]),
            int(work["pos"][idx]),
            str(work["ref"][idx]),
            str(work["alt"][idx]),
        ),
    )

    digest = hashlib.sha256()
    digest.update(
        f"genotype_sha256_v1\t{hash_schema_version}\t{normalization_profile_id}\n".encode()
    )
    chrom_col = work["chrom_norm"]
    pos_col = work["pos"]
    ref_col = work["ref"]
    alt_col = work["alt"]
    gt_raw = work["gt_raw"]
    for idx in order:
        line = (
            f"{chrom_col[idx]}\t{pos_col[idx]}\t{ref_col[idx]}\t"
            f"{alt_col[idx]}\t{_genotype_token(gt_raw[idx])}\n"
        )
        digest.update(line.encode())
    return digest.hexdigest()


def identity_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "sample_scores" / "identity_cache.json"


def _cache_key(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_identity_cache(cache_dir: Path) -> list[dict[str, Any]]:
    cache_file = identity_cache_path(cache_dir)
    if not cache_file.exists():
        return []
    return json.loads(cache_file.read_text())


def _write_identity_cache(cache_dir: Path, records: list[dict[str, Any]]) -> None:
    cache_file = identity_cache_path(cache_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(records, indent=2))


def _cached_digest(path: Path, cache_dir: Path, field: str, compute: Callable[[], str]) -> str:
    key = _cache_key(path)
    records = _read_identity_cache(cache_dir)
    for record in records:
        if (
            record.get("path") == key["path"]
            and record.get("size") == key["size"]
            and record.get("mtime_ns") == key["mtime_ns"]
            and record.get(field)
        ):
            return str(record[field])
    digest = compute()
    updated = [record for record in records if record.get("path") != key["path"]]
    previous = next((record for record in records if record.get("path") == key["path"]), {})
    updated.append({**previous, **key, field: digest})
    _write_identity_cache(cache_dir, updated)
    return digest


def cached_source_sha256(path: Path, cache_dir: Path) -> str:
    """Return ``source_sha256``, reusing a path/size/mtime cache entry when valid."""
    return _cached_digest(path, cache_dir, "source_sha256", lambda: source_sha256(path))


def cached_genotype_sha256(
    genotypes: pl.DataFrame | pl.LazyFrame,
    parquet_path: Path,
    cache_dir: Path,
) -> str:
    """Return ``genotype_sha256_v1``, reusing a parquet path/size/mtime cache entry."""
    return _cached_digest(
        parquet_path,
        cache_dir,
        "genotype_sha256_v1",
        lambda: genotype_sha256_v1(genotypes),
    )


def resolve_sample(
    samples: list[SampleRecord],
    *,
    source_path: Path | None = None,
    genotypes: pl.DataFrame | pl.LazyFrame | None = None,
    alias: str | None = None,
    cache_dir: Path | None = None,
) -> SampleRecord | None:
    """Return the registered sample whose hashes match, or None.

    An alias only selects candidates. A user-defined alias never proves identity.
    """
    with start_action(action_type="sample_scores:resolve_sample", alias=alias):
        candidates = samples
        if alias:
            from just_prs.sample_scores.models import PRIVATE_INGEST_ALIASES

            key = alias.strip().casefold()
            mapped = PRIVATE_INGEST_ALIASES.get(key)
            aliased = [
                sample
                for sample in samples
                if sample.sample_id.casefold() == key
                or (mapped is not None and sample.sample_id == mapped)
                or key in {item.casefold() for item in sample.aliases}
            ]
            if aliased:
                candidates = aliased

        if source_path is not None and source_path.exists():
            digest = (
                cached_source_sha256(source_path, cache_dir)
                if cache_dir is not None
                else source_sha256(source_path)
            )
            for sample in candidates:
                if sample.source_sha256 == digest:
                    return sample

        if genotypes is not None:
            digest = genotype_sha256_v1(genotypes)
            for sample in candidates:
                if sample.genotype_sha256_v1 == digest:
                    return sample
        return None
