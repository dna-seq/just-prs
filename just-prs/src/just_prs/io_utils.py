"""Compression-agnostic file opening by content (magic bytes), not extension.

A recurring bug class in this codebase was choosing ``gzip.open`` vs ``open`` from
the filename suffix. That silently mis-handles a BGZF/gzip stream saved with a
plain ``.vcf`` name (corrupted header reads) — while polars-bio's reader sniffs
content and works fine, so the two paths disagree. These helpers decide by the
gzip magic bytes (``1f 8b``); BGZF is gzip-compatible, so ``gzip.open`` reads it.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

_GZIP_MAGIC = b"\x1f\x8b"


def is_gzip(path: Path | str) -> bool:
    """Return True if the file begins with the gzip/BGZF magic bytes."""
    with open(path, "rb") as fh:
        return fh.read(2) == _GZIP_MAGIC


def open_maybe_compressed(
    path: Path | str,
    mode: str = "rt",
    encoding: str | None = "utf-8",
    errors: str | None = None,
) -> IO:
    """Open a file, transparently decompressing gzip/BGZF detected by content.

    Extension-independent: a BGZF stream named ``.vcf`` (no ``.gz``) is opened
    decompressed, and a plain text file named ``.gz`` is opened as-is. Binary
    modes ignore ``encoding``/``errors``.
    """
    opener = gzip.open if is_gzip(path) else open
    if "b" in mode:
        return opener(path, mode)  # type: ignore[call-overload]
    return opener(path, mode, encoding=encoding, errors=errors)  # type: ignore[call-overload]
