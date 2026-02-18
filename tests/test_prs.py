"""PRS computation tests using the antonkulaga.vcf test data."""

import math
from pathlib import Path

from just_prs.prs import compute_prs, compute_prs_batch
from just_prs.vcf import read_genotypes


def test_read_genotypes(vcf_path: Path) -> None:
    """Verify VCF reading produces expected columns and non-empty data."""
    lf = read_genotypes(vcf_path)
    df = lf.limit(100).collect()

    assert "chrom" in df.columns
    assert "pos" in df.columns
    assert "ref" in df.columns
    assert "alt" in df.columns
    assert "GT" in df.columns
    assert len(df) > 0

    assert not any(v.startswith("chr") for v in df["chrom"].to_list() if isinstance(v, str))


def test_compute_prs_pgs000001(vcf_path: Path, scoring_cache_dir: Path) -> None:
    """Compute PRS for PGS000001 (Breast cancer, 77 variants) on test VCF."""
    result = compute_prs(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        trait_reported="Breast cancer",
    )

    assert result.pgs_id == "PGS000001"
    assert result.trait_reported == "Breast cancer"
    assert result.variants_total == 77
    assert result.variants_matched > 0
    assert result.match_rate > 0.0
    assert math.isfinite(result.score)
    assert result.score != 0.0


def test_compute_prs_batch(vcf_path: Path, scoring_cache_dir: Path) -> None:
    """Compute multiple PRS scores in batch mode."""
    pgs_ids = ["PGS000001", "PGS000002"]
    results = compute_prs_batch(
        vcf_path=vcf_path,
        pgs_ids=pgs_ids,
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
    )

    assert len(results) == 2
    result_ids = {r.pgs_id for r in results}
    assert result_ids == {"PGS000001", "PGS000002"}

    for r in results:
        assert r.variants_matched > 0
        assert r.match_rate > 0.0
        assert math.isfinite(r.score)
