"""Tests for VCF normalization quality filtering.

Regression coverage for the FILTER "." vs "" mismatch: polars-bio decodes the
VCF missing FILTER sentinel ("." in the spec) as an empty string "". A
``pass_filters`` config of ``["PASS", "."]`` must therefore still keep records
whose FILTER column is "", otherwise GATK HaplotypeCaller-style VCFs (every
record FILTER=".") normalize down to zero rows.
"""

from pathlib import Path

import polars as pl

from just_prs.normalize import VcfFilterConfig, _expand_pass_filters, normalize_vcf


_VCF = """\
##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=LowQual,Description="Low quality">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##contig=<ID=1,length=248956422>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample
1\t100\trs1\tA\tG\t60\t.\t.\tGT:DP\t0/1:30
1\t200\trs2\tC\tT\t60\t.\t.\tGT:DP\t1/1:40
1\t300\trs3\tG\tA\t60\tPASS\t.\tGT:DP\t0/1:50
1\t400\trs4\tT\tC\t60\tLowQual\t.\tGT:DP\t0/1:50
"""


def test_expand_pass_filters_treats_dot_and_empty_as_equivalent() -> None:
    assert set(_expand_pass_filters(["PASS", "."])) == {"PASS", ".", ""}
    assert set(_expand_pass_filters(["PASS", ""])) == {"PASS", ".", ""}
    assert set(_expand_pass_filters(["PASS"])) == {"PASS"}


def test_normalize_keeps_dot_filter_records(tmp_path: Path) -> None:
    """FILTER='.' records (decoded as '') survive a ["PASS", "."] filter."""
    src = tmp_path / "sample.vcf"
    src.write_text(_VCF)
    out = tmp_path / "sample.parquet"

    config = VcfFilterConfig(pass_filters=["PASS", "."])
    normalize_vcf(src, out, config=config)

    df = pl.read_parquet(out)
    # 3 of 4 rows pass: the two "." records + the PASS record; LowQual is dropped.
    assert df.height == 3
    assert sorted(df["rsid"].to_list()) == ["rs1", "rs2", "rs3"]


def test_normalize_filter_with_depth(tmp_path: Path) -> None:
    """FILTER expansion composes correctly with the min_depth gate."""
    src = tmp_path / "sample.vcf"
    src.write_text(_VCF)
    out = tmp_path / "sample.parquet"

    config = VcfFilterConfig(pass_filters=["PASS", "."], min_depth=35)
    normalize_vcf(src, out, config=config)

    df = pl.read_parquet(out)
    # rs1 (DP=30) drops on depth; rs4 drops on FILTER; rs2 (40) and rs3 (50) remain.
    assert sorted(df["rsid"].to_list()) == ["rs2", "rs3"]
