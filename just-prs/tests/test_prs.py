"""PRS computation tests using the antonkulaga.vcf test data."""

import math
from pathlib import Path

import pytest

from just_prs.prs import PRSEngine, compute_prs, compute_prs_batch, compute_prs_duckdb
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


def test_compute_prs_duckdb_pgs000001(vcf_path: Path, scoring_cache_dir: Path) -> None:
    """Compute PRS for PGS000001 with DuckDB engine and verify basic properties."""
    result = compute_prs_duckdb(
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


@pytest.mark.parametrize("pgs_id", ["PGS000001", "PGS000002", "PGS000003"])
def test_engine_parity(vcf_path: Path, scoring_cache_dir: Path, pgs_id: str) -> None:
    """Verify that polars and DuckDB engines produce identical results."""
    polars_result = compute_prs(
        vcf_path=vcf_path,
        scoring_file=pgs_id,
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id=pgs_id,
    )
    duckdb_result = compute_prs_duckdb(
        vcf_path=vcf_path,
        scoring_file=pgs_id,
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id=pgs_id,
    )

    assert polars_result.variants_total == duckdb_result.variants_total
    assert polars_result.variants_matched == duckdb_result.variants_matched
    assert polars_result.match_rate == pytest.approx(duckdb_result.match_rate)
    assert polars_result.score == pytest.approx(duckdb_result.score, abs=1e-10)

    if polars_result.has_allele_frequencies:
        assert duckdb_result.has_allele_frequencies
        assert polars_result.theoretical_mean == pytest.approx(duckdb_result.theoretical_mean, abs=1e-10)
        assert polars_result.theoretical_std == pytest.approx(duckdb_result.theoretical_std, abs=1e-10)
        assert polars_result.percentile == pytest.approx(duckdb_result.percentile, abs=0.01)


def test_prs_engine_enum() -> None:
    """Verify PRSEngine enum values."""
    assert PRSEngine.POLARS.value == "polars"
    assert PRSEngine.DUCKDB.value == "duckdb"
    assert PRSEngine("polars") == PRSEngine.POLARS
    assert PRSEngine("duckdb") == PRSEngine.DUCKDB


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
