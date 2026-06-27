"""Opt-in real-data test: GRCh37 WGS reference-allele restoration recovery.

The GRCh37 reference-allele universe must recover coverage on a real
variant-only WGS VCF the same way the GRCh38 universe does — an absent score
locus whose REF is resolved from the universe scores as hom-ref instead of
falling into ``variants_unscorable_absent``.

Set ``PRS_TEST_GRCH37_WGS_VCF`` to a real GRCh37 whole-genome VCF (e.g. a
DeepVariant callset lifted hg38->hg19) to run. Network is required: the GRCh37
harmonized scoring files and the ``reference_allele_universe_GRCh37.parquet``
are pulled from HuggingFace on first use.
"""

import os
from pathlib import Path

import polars as pl
import pytest

from just_prs import PRSCatalog, VcfFilterConfig, normalize_vcf
from just_prs.scoring import resolve_cache_dir
from just_prs.vcf import detect_genome_build

# A genome-wide score (~6.9M variants): the OFF path matches only the variant
# sites present in the callset (~half); the WGS path recovers near-total coverage.
_GENOME_WIDE_PGS = "PGS000014"


@pytest.mark.skipif(
    not os.environ.get("PRS_TEST_GRCH37_WGS_VCF"),
    reason="set PRS_TEST_GRCH37_WGS_VCF to a real GRCh37 WGS VCF to run",
)
def test_grch37_wgs_restoration_recovers_coverage(tmp_path: Path) -> None:
    src = Path(os.environ["PRS_TEST_GRCH37_WGS_VCF"])
    assert src.exists(), f"PRS_TEST_GRCH37_WGS_VCF does not exist: {src}"

    # The fixture must actually be GRCh37, or the test is meaningless.
    assert detect_genome_build(src) == "GRCh37", "fixture VCF is not detected as GRCh37"

    norm = normalize_vcf(src, tmp_path / "wgs.parquet", config=VcfFilterConfig(pass_filters=["PASS", "."]))
    geno = pl.scan_parquet(norm)
    # A real WGS callset has millions of variant rows.
    assert pl.scan_parquet(norm).select(pl.len()).collect().item() > 1_000_000

    cat = PRSCatalog(cache_dir=resolve_cache_dir())

    off = cat.compute_prs(
        vcf_path="", pgs_id=_GENOME_WIDE_PGS, genome_build="GRCh37",
        genotypes_lf=geno, reference_restoration=False,
    )
    wgs = cat.compute_prs(
        vcf_path="", pgs_id=_GENOME_WIDE_PGS, genome_build="GRCh37",
        genotypes_lf=geno, reference_restoration=True,
    )

    # Same score, same variant universe -> identical denominator.
    assert off.variants_total == wgs.variants_total

    # OFF leaves a genome-wide score badly under-covered; WGS restoration recovers
    # near-total coverage by resolving absent loci as hom-ref.
    assert off.match_rate < 0.60, f"expected OFF under-coverage, got {off.match_rate:.3f}"
    assert wgs.match_rate > 0.99, f"expected WGS near-full coverage, got {wgs.match_rate:.3f}"

    # The unscorable-absent backlog collapses, and the recovered loci are
    # attributed to the universe tiers (panel + fasta).
    assert wgs.variants_unscorable_absent < off.variants_unscorable_absent / 100
    restored = (wgs.variants_ref_resolved_panel or 0) + (wgs.variants_ref_resolved_fasta or 0)
    assert restored > 0, "no loci attributed to the GRCh37 universe restoration tiers"

    # Weight-mass coverage tracks the count recovery (scale-free).
    assert wgs.weight_mass_coverage > off.weight_mass_coverage
