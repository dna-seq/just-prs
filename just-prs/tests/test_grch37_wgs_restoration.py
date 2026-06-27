"""Opt-in/fallback real-data test: WGS reference-allele restoration recovery.

The reference-allele universe must recover coverage on a real variant-only WGS
VCF — an absent score locus whose REF is resolved from the universe scores as
hom-ref instead of falling into ``variants_unscorable_absent``.

Set ``PRS_TEST_GRCH37_WGS_VCF`` to a real GRCh37 whole-genome VCF (e.g. a
DeepVariant callset lifted hg38->hg19) to force GRCh37 coverage. If unset, the
test falls back to cached built-in public VCF aliases (``livia``, then
``anton``), auto-downloading from Zenodo only on first use. Network is required
for first-use VCF/scoring-file/reference-universe downloads.
"""

import os
from pathlib import Path

import polars as pl
import pytest

from just_prs import PRSCatalog, VcfFilterConfig, normalize_vcf
from just_prs.cli import _resolve_vcf
from just_prs.scoring import resolve_cache_dir
from just_prs.vcf import detect_genome_build

# A genome-wide score (~6.9M variants): the OFF path matches only the variant
# sites present in the callset (~half); the WGS path recovers near-total coverage.
_GENOME_WIDE_PGS = "PGS000014"
_PUBLIC_WGS_ALIASES = ("livia", "anton")


def _resolve_wgs_fixture() -> tuple[Path, str]:
    """Return a real WGS VCF path plus a human-readable trace label."""
    env_path = os.environ.get("PRS_TEST_GRCH37_WGS_VCF")
    if env_path:
        return Path(env_path), "PRS_TEST_GRCH37_WGS_VCF"

    cache_dir = resolve_cache_dir()
    for alias in _PUBLIC_WGS_ALIASES:
        path = _resolve_vcf(alias, cache_dir)
        if path.exists():
            return path, f"built-in alias '{alias}'"

    pytest.skip("No PRS_TEST_GRCH37_WGS_VCF and no built-in public WGS alias could be resolved.")


def test_grch37_wgs_restoration_recovers_coverage(tmp_path: Path) -> None:
    src, fixture_source = _resolve_wgs_fixture()
    assert src.exists(), f"{fixture_source} does not exist: {src}"

    detected_build = detect_genome_build(src)
    if detected_build is None:
        pytest.skip(f"Could not detect genome build for {fixture_source}: {src}")
    if fixture_source == "PRS_TEST_GRCH37_WGS_VCF":
        assert detected_build == "GRCh37", "fixture VCF is not detected as GRCh37"
    print(f"WGS restoration fixture: {fixture_source} -> {src} ({detected_build})")

    norm = normalize_vcf(src, tmp_path / "wgs.parquet", config=VcfFilterConfig(pass_filters=["PASS", "."]))
    geno = pl.scan_parquet(norm)
    # A real WGS callset has millions of variant rows.
    assert pl.scan_parquet(norm).select(pl.len()).collect().item() > 1_000_000

    cat = PRSCatalog(cache_dir=resolve_cache_dir())

    off = cat.compute_prs(
        vcf_path="", pgs_id=_GENOME_WIDE_PGS, genome_build=detected_build,
        genotypes_lf=geno, reference_restoration=False,
    )
    wgs = cat.compute_prs(
        vcf_path="", pgs_id=_GENOME_WIDE_PGS, genome_build=detected_build,
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
