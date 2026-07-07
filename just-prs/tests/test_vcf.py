"""Tests for VCF header handling."""

import gzip
from pathlib import Path

from just_prs.vcf import detect_genome_build


_GRCH38_HEADER = """\
##fileformat=VCFv4.2
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample
"""


# DRAGEN emits no ``##reference=`` line; the build only appears inside the
# command-line path and the ``--variant-annotation-assembly`` flag.
_DRAGEN_HEADER = """\
##fileformat=VCFv4.2
##DRAGENCommandLine=<ID=dragen,Version="SW: 05.121.676",CommandLineOptions="--ht-reference=/staging/.../Hsapiens/hg38/seq/hg38.fa --variant-annotation-assembly GRCh38 --ref-dir /scratch/hg38-alt_masked/DRAGEN/9">
##contig=<ID=chr1,length=248956422>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample
"""

# No build-name header at all — only contig lengths to vote on.
_CONTIG_ONLY_GRCH37_HEADER = """\
##fileformat=VCFv4.2
##contig=<ID=1,length=249250621>
##contig=<ID=2,length=243199373>
##contig=<ID=3,length=198022430>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample
"""

# ``assembly=`` carried as a per-contig attribute (a common GATK/Picard layout).
_CONTIG_ASSEMBLY_HEADER = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=248956422,assembly=GRCh38>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample
"""


def test_detect_genome_build_reads_gzip_by_magic_bytes(tmp_path: Path) -> None:
    """Gzipped uploads can arrive without a useful extension from the browser."""
    src = tmp_path / "sample.vcf"
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        fh.write(_GRCH38_HEADER)

    assert detect_genome_build(src) == "GRCh38"


def test_detect_genome_build_from_dragen_command_line(tmp_path: Path) -> None:
    """DRAGEN VCFs lack ##reference= but name the build in the command line."""
    src = tmp_path / "dragen.vcf"
    src.write_text(_DRAGEN_HEADER, encoding="utf-8")

    assert detect_genome_build(src) == "GRCh38"


def test_detect_genome_build_from_contig_assembly_attribute(tmp_path: Path) -> None:
    """assembly= on a ##contig line is an explicit build signal."""
    src = tmp_path / "assembly.vcf"
    src.write_text(_CONTIG_ASSEMBLY_HEADER, encoding="utf-8")

    assert detect_genome_build(src) == "GRCh38"


def test_detect_genome_build_falls_back_to_contig_lengths(tmp_path: Path) -> None:
    """With no build-name header, contig lengths still resolve the build."""
    src = tmp_path / "contig_only.vcf"
    src.write_text(_CONTIG_ONLY_GRCH37_HEADER, encoding="utf-8")

    assert detect_genome_build(src) == "GRCh37"


def test_detect_genome_build_ignores_b37_substring(tmp_path: Path) -> None:
    """A path like .../lib37/ must not be read as the b37 build."""
    src = tmp_path / "decoy.vcf"
    src.write_text(
        "##fileformat=VCFv4.2\n"
        '##DRAGENCommandLine=<ID=dragen,CommandLineOptions="--ref-dir /opt/lib37/refs/hg38.fa">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n",
        encoding="utf-8",
    )

    # hg38 in the same line wins; b37 inside "lib37" is not a standalone token.
    assert detect_genome_build(src) == "GRCh38"
