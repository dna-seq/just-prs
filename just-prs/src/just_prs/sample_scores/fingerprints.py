"""Logical fingerprints for scoring files and the reference-allele universe."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from just_prs.prs import _normalize_scoring_columns, is_dosage_weight_format

SCORING_FINGERPRINT_VERSION = 1
SCORING_FILE_FINGERPRINT_VERSION = 1
UNIVERSE_FINGERPRINT_VERSION = 1


def scoring_file_fingerprint(path: Path) -> str:
    """SHA-256 of a scoring parquet's bytes. Never materializes the table.

    Used for checkpoint keys and lookup so planning 5k+ catalog files stays
    bounded. A byte change (weights, positions, or a re-parse) invalidates
    the key. In-memory logical identity is ``scoring_fingerprint``.
    """
    digest = hashlib.sha256()
    digest.update(f"scoring_file_fingerprint_v{SCORING_FILE_FINGERPRINT_VERSION}\n".encode())
    digest.update(path.name.encode() + b"\n")
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scoring_fingerprint(scoring: pl.DataFrame | pl.LazyFrame) -> str:
    """SHA-256 of the logical score definition (positions, alleles, weights).

    A PGS ID is not immutable. This digest changes when the scoring file's
    variant set or weights change, independent of parquet compression.
    """
    frame = scoring.collect() if isinstance(scoring, pl.LazyFrame) else scoring
    normalized = _normalize_scoring_columns(frame.lazy()).collect()
    columns = normalized.columns
    dosage = is_dosage_weight_format(columns)
    weight_cols = (
        ["dosage_0_weight", "dosage_1_weight", "dosage_2_weight"]
        if dosage
        else ["effect_weight"]
    )
    other = "other_allele" if "other_allele" in columns else None
    select_cols = ["chr_name_norm", "chr_pos_norm", "effect_allele", *weight_cols]
    if other:
        select_cols.append(other)
    rows = (
        normalized
        .select(select_cols)
        .sort(select_cols)
        .iter_rows()
    )
    digest = hashlib.sha256()
    digest.update(f"scoring_fingerprint_v{SCORING_FINGERPRINT_VERSION}\n".encode())
    for row in rows:
        digest.update(("\t".join("" if part is None else str(part) for part in row) + "\n").encode())
    return digest.hexdigest()


def reference_universe_fingerprint(universe: pl.DataFrame | pl.LazyFrame | Path) -> str:
    """SHA-256 of the published universe file or a streaming column digest.

    Never materializes the catalog-wide universe (~34M rows) in Python.
    """
    digest = hashlib.sha256()
    digest.update(f"reference_universe_fingerprint_v{UNIVERSE_FINGERPRINT_VERSION}\n".encode())
    if isinstance(universe, Path):
        digest.update(b"file\n")
        digest.update(universe.name.encode() + b"\n")
        with universe.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    lf = universe.lazy() if isinstance(universe, pl.DataFrame) else universe
    stats = lf.select(
        pl.len().alias("n"),
        pl.col("chrom").hash(10).sum().alias("chrom_h"),
        pl.col("pos").hash(10).sum().alias("pos_h"),
        pl.col("ref").hash(10).sum().alias("ref_h"),
    ).collect()
    digest.update(b"frame\n")
    digest.update(str(stats.to_dicts()[0]).encode())
    return digest.hexdigest()
