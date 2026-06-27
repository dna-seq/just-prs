"""Tests for content-based (magic-byte) compression detection.

Regression for the gz bug: a BGZF/gzip stream named ``.vcf`` (no ``.gz``) must
still be opened decompressed. Extension-based opening read it as plain text and
returned a corrupted header -> genome-build detection silently failed.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from just_prs.io_utils import is_gzip, open_maybe_compressed
from just_prs.vcf import detect_genome_build


def _grch38_header() -> str:
    lines = ["##fileformat=VCFv4.2", "##reference=unknown"]
    for chrom, length in [("1", 248956422), ("2", 242193529), ("3", 198295559)]:
        lines.append(f"##contig=<ID={chrom},length={length}>")
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    lines.append("1\t100\t.\tA\tG\t.\tPASS\t.")
    return "\n".join(lines) + "\n"


def test_open_maybe_compressed_roundtrip(tmp_path: Path):
    text = "hello\nworld\n"
    plain = tmp_path / "p.txt"
    plain.write_text(text)
    gz_named_plain = tmp_path / "g.txt"  # gzip content, non-.gz name
    with gzip.open(gz_named_plain, "wt") as f:
        f.write(text)

    assert not is_gzip(plain)
    assert is_gzip(gz_named_plain)
    with open_maybe_compressed(plain, "rt") as fh:
        assert fh.read() == text
    with open_maybe_compressed(gz_named_plain, "rt") as fh:
        assert fh.read() == text  # decompressed despite the .txt name


def test_detect_build_plain_vcf(tmp_path: Path):
    p = tmp_path / "plain.vcf"
    p.write_text(_grch38_header())
    assert detect_genome_build(p) == "GRCh38"


def test_detect_build_gzip_disguised_as_vcf(tmp_path: Path):
    # The bug case: gzip content, but named .vcf (no .gz suffix).
    p = tmp_path / "disguised.vcf"
    with gzip.open(p, "wt") as f:
        f.write(_grch38_header())
    assert is_gzip(p)
    assert detect_genome_build(p) == "GRCh38"


def test_detect_build_gz_extension(tmp_path: Path):
    p = tmp_path / "normal.vcf.gz"
    with gzip.open(p, "wt") as f:
        f.write(_grch38_header())
    assert detect_genome_build(p) == "GRCh38"
