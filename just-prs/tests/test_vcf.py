"""Tests for VCF header handling."""

import gzip
from pathlib import Path

from just_prs.vcf import detect_genome_build


_GRCH38_HEADER = """\
##fileformat=VCFv4.2
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample
"""


def test_detect_genome_build_reads_gzip_by_magic_bytes(tmp_path: Path) -> None:
    """Gzipped uploads can arrive without a useful extension from the browser."""
    src = tmp_path / "sample.vcf"
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        fh.write(_GRCH38_HEADER)

    assert detect_genome_build(src) == "GRCh38"
