"""Which caller wrote a genotype file, and how it writes a man's X.

``1/1`` on a man's X means one copy from DeepVariant 1.1.0 (diploid everywhere)
and is not written at all by a sex-aware caller (DRAGEN writes it haploid and
declares ``##referenceSexKaryotype``). The normalized parquet records the caller,
its version, its ploidy model and the inferred karyotype so a reader can tell.
"""

from pathlib import Path

import polars as pl
import pytest

from just_prs.arrays import normalize_array
from just_prs.normalize import VcfFilterConfig, normalize_vcf, read_genotype_provenance
from just_prs.vcf import read_vcf_provenance

_BODY_HEADER = """\
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=1,length=248956422>
##contig=<ID=X,length=156040895>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample
"""


def _vcf(header: str, rows: list[str]) -> str:
    return "##fileformat=VCFv4.2\n" + header + _BODY_HEADER + "".join(r + "\n" for r in rows)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("##DeepVariant_version=1.1.0\n", {"caller": "DeepVariant", "caller_version": "1.1.0"}),
        (
            '##DRAGENCommandLine=<ID=dragen,Version="SW: 13.021.779.4.4.4, HW: 13.021.779",Date="x">\n'
            "##referenceSexKaryotype=XY\n",
            {"caller": "DRAGEN", "caller_version": "SW: 13.021.779.4.4.4, HW: 13.021.779", "declared_karyotype": "XY"},
        ),
        (
            '##GATKCommandLine=<ID=HaplotypeCaller,CommandLine="x",Version="4.2.0.0",Date="y">\n',
            {"caller": "GATK HaplotypeCaller", "caller_version": "4.2.0.0"},
        ),
        ("##bcftools_callVersion=1.10.2+htslib-1.10.2\n", {"caller": "bcftools call", "caller_version": "1.10.2+htslib-1.10.2"}),
        ("##source=strelka\n##source_version=2.9.10\n", {"caller": "strelka", "caller_version": "2.9.10"}),
        ("", {}),
    ],
)
def test_header_provenance(tmp_path: Path, header: str, expected: dict[str, str]) -> None:
    path = tmp_path / "s.vcf"
    path.write_text(_vcf(header, ["1\t100\t.\tA\tG\t50\tPASS\t.\tGT\t0/1"]))
    found = read_vcf_provenance(path)
    assert {k: found[k] for k in expected} == expected
    if not expected:
        assert "caller" not in found
    assert found["vcf_format"] == "VCFv4.2"


@pytest.mark.parametrize(
    ("header", "x_gt", "ploidy_model"),
    [
        ("##DeepVariant_version=1.1.0\n", "1/1", "diploid"),
        ("##DeepVariant_version=1.6.0\n", "1", "sex_aware"),
    ],
)
def test_normalized_parquet_records_caller_and_ploidy_model(
    tmp_path: Path, header: str, x_gt: str, ploidy_model: str
) -> None:
    src = tmp_path / "s.vcf"
    src.write_text(_vcf(header, [
        "1\t100\t.\tA\tG\t50\tPASS\t.\tGT\t0/1",
        f"X\t50000000\t.\tC\tT\t50\tPASS\t.\tGT\t{x_gt}",
    ]))
    out = normalize_vcf(src, tmp_path / "s.parquet", config=VcfFilterConfig(pass_filters=["PASS", "."]))
    provenance = read_genotype_provenance(out)
    assert provenance["caller"] == "DeepVariant"
    assert provenance["caller_version"] == header.split("=", 1)[1].strip()
    assert provenance["ploidy_model"] == ploidy_model
    # Two records are too few to call a karyotype; the key still says so.
    assert provenance["karyotype"] == "unknown"
    assert provenance["normalized_by"].startswith("just-prs ")
    assert pl.read_parquet(out).height == 2


def test_array_provenance(tmp_path: Path) -> None:
    src = tmp_path / "g.txt"
    src.write_text("# 23andMe\n# rsid\tchromosome\tposition\tgenotype\nrs1\t1\t100\tAG\nrs2\tX\t5000\tG\n")
    provenance = read_genotype_provenance(normalize_array(src, tmp_path / "g.parquet"))
    assert provenance["caller"] == "23andme array export"
    assert provenance["ploidy_model"] == "sex_aware"
    assert provenance["genome_build"] == "GRCh37"


def test_foreign_parquet_has_no_provenance(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    pl.DataFrame({"chrom": ["1"]}).write_parquet(path)
    assert read_genotype_provenance(path) == {}
