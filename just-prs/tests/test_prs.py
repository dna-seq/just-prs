"""PRS computation tests using the antonkulaga.vcf test data."""

import math
from pathlib import Path

import polars as pl
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
        genotype_input_mode="plink_present_only",
    )
    duckdb_result = compute_prs_duckdb(
        vcf_path=vcf_path,
        scoring_file=pgs_id,
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id=pgs_id,
        genotype_input_mode="plink_present_only",
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


def test_variant_only_absent_hom_ref_inference(tmp_path: Path) -> None:
    """Absent loci in variant-only VCFs can be scored as hom-ref when reference is known."""
    geno_path = tmp_path / "variant_only.parquet"
    pl.DataFrame({
        "chrom": ["1"],
        "pos": [100],
        "ref": ["A"],
        "alt": ["G"],
        "GT": ["0/1"],
    }).write_parquet(geno_path)

    scoring = pl.DataFrame({
        "hm_chr": ["1", "1", "1"],
        "hm_pos": [100, 200, 300],
        "effect_allele": ["G", "T", "C"],
        "reference_allele": ["A", "T", None],
        "effect_weight": [1.0, 2.0, 3.0],
    }).lazy()

    expected_score = 5.0  # observed G dosage 1 (+1), absent T/T dosage 2 (+4), unknown ref skipped.
    polars_result = compute_prs(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path),
        genotype_input_mode="variant_only",
    )
    duckdb_result = compute_prs_duckdb(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_parquet=geno_path,
        genotype_input_mode="variant_only",
    )

    for result in (polars_result, duckdb_result):
        assert result.score == pytest.approx(expected_score)
        assert result.variants_observed == 1
        assert result.variants_assumed_hom_ref == 1
        assert result.variants_unscorable_absent == 1
        assert result.variants_matched == 2
        assert result.match_rate == pytest.approx(2 / 3)


def test_variant_only_differs_from_plink_present_only_when_ref_effect_absent(tmp_path: Path) -> None:
    """Present-only PLINK-compatible scoring skips absent loci that variant-only mode can infer."""
    geno_path = tmp_path / "variant_only.parquet"
    pl.DataFrame({
        "chrom": ["1"],
        "pos": [100],
        "ref": ["A"],
        "alt": ["G"],
        "GT": ["0/1"],
    }).write_parquet(geno_path)

    scoring = pl.DataFrame({
        "hm_chr": ["1", "1"],
        "hm_pos": [100, 200],
        "effect_allele": ["G", "T"],
        "reference_allele": ["A", "T"],
        "effect_weight": [1.0, 2.0],
    }).lazy()

    variant_only = compute_prs(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path),
        genotype_input_mode="variant_only",
    )
    present_only = compute_prs(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path),
        genotype_input_mode="plink_present_only",
    )

    assert variant_only.score == pytest.approx(5.0)
    assert present_only.score == pytest.approx(1.0)
    assert variant_only.variants_assumed_hom_ref == 1
    assert present_only.variants_assumed_hom_ref == 0


def test_weight_mass_coverage_math_and_parity(tmp_path: Path) -> None:
    """C_wt = matched |beta| mass / total |beta| mass, identical across engines.

    Same fixture as the hom-ref inference test: weights 1, 2, 3. The w=1 (observed)
    and w=2 (absent hom-ref) variants are matched; the w=3 variant has an unknown
    reference allele and is unscorable. So matched mass = 3, total mass = 6, and
    C_wt = 0.5 — while the count-based match_rate is 2/3. The two diverge on purpose.
    """
    geno_path = tmp_path / "variant_only.parquet"
    pl.DataFrame({
        "chrom": ["1"],
        "pos": [100],
        "ref": ["A"],
        "alt": ["G"],
        "GT": ["0/1"],
    }).write_parquet(geno_path)

    scoring = pl.DataFrame({
        "hm_chr": ["1", "1", "1"],
        "hm_pos": [100, 200, 300],
        "effect_allele": ["G", "T", "C"],
        "reference_allele": ["A", "T", None],
        "effect_weight": [1.0, 2.0, 3.0],
    }).lazy()

    polars_result = compute_prs(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path),
        genotype_input_mode="variant_only",
    )
    duckdb_result = compute_prs_duckdb(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_parquet=geno_path,
        genotype_input_mode="variant_only",
    )

    for result in (polars_result, duckdb_result):
        assert result.weight_mass_total == pytest.approx(6.0)
        assert result.weight_mass_matched == pytest.approx(3.0)
        assert result.weight_mass_coverage == pytest.approx(0.5)
        # C_wt is scale-free: it differs from the count match_rate (2/3) here.
        assert result.match_rate == pytest.approx(2 / 3)


def test_weight_mass_coverage_genoboost_surrogate(tmp_path: Path) -> None:
    """Per-dosage (GenoBoost) scores use max|dosage_k_weight| as the mass surrogate."""
    geno_path = tmp_path / "geno.parquet"
    pl.DataFrame({
        "chrom": ["1"],
        "pos": [100],
        "ref": ["A"],
        "alt": ["G"],
        "GT": ["0/1"],
    }).write_parquet(geno_path)

    # Variant A present (matched, mass = max(0.1,0.5,0.9) = 0.9); variant B absent
    # with unknown reference allele -> unscorable (mass = max(0,1,4) = 4).
    scoring = pl.DataFrame({
        "hm_chr": ["1", "1"],
        "hm_pos": [100, 200],
        "effect_allele": ["G", "T"],
        "reference_allele": ["A", None],
        "dosage_0_weight": [0.1, 0.0],
        "dosage_1_weight": [0.5, 1.0],
        "dosage_2_weight": [0.9, 4.0],
    }).lazy()

    polars_result = compute_prs(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path),
        genotype_input_mode="variant_only",
    )
    duckdb_result = compute_prs_duckdb(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_parquet=geno_path,
        genotype_input_mode="variant_only",
    )

    for result in (polars_result, duckdb_result):
        assert result.weight_mass_total == pytest.approx(4.9)
        assert result.weight_mass_matched == pytest.approx(0.9)
        assert result.weight_mass_coverage == pytest.approx(0.9 / 4.9)


def test_z_score_and_reference_stats_exposed(tmp_path: Path) -> None:
    """The true z-score and reference mean/std are exposed on PRSResult (F12)."""
    geno_path = tmp_path / "geno.parquet"
    pl.DataFrame({
        "chrom": ["1"],
        "pos": [100],
        "ref": ["A"],
        "alt": ["G"],
        "GT": ["0/1"],
    }).write_parquet(geno_path)

    scoring = pl.DataFrame({
        "hm_chr": ["1", "1"],
        "hm_pos": [100, 200],
        "effect_allele": ["G", "T"],
        "reference_allele": ["A", "A"],
        "effect_weight": [1.0, 2.0],
        "allelefrequency_effect": [0.3, 0.4],
    }).lazy()

    polars_result = compute_prs(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path),
        genotype_input_mode="variant_only",
    )
    duckdb_result = compute_prs_duckdb(
        vcf_path="",
        scoring_file=scoring,
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_parquet=geno_path,
        genotype_input_mode="variant_only",
    )

    for result in (polars_result, duckdb_result):
        assert result.has_allele_frequencies
        assert result.percentile_method == "theoretical"
        assert result.reference_mean == pytest.approx(result.theoretical_mean)
        assert result.reference_std == pytest.approx(result.theoretical_std)
        assert result.z_score is not None
        expected_z = (result.score - result.reference_mean) / result.reference_std
        assert result.z_score == pytest.approx(expected_z)

    assert polars_result.z_score == pytest.approx(duckdb_result.z_score, abs=1e-10)


def test_prs_engine_enum() -> None:
    """Verify PRSEngine enum values."""
    assert PRSEngine.POLARS.value == "polars"
    assert PRSEngine.DUCKDB.value == "duckdb"
    assert PRSEngine("polars") == PRSEngine.POLARS
    assert PRSEngine("duckdb") == PRSEngine.DUCKDB


def test_compute_prs_batch(vcf_path: Path, scoring_cache_dir: Path) -> None:
    """Compute multiple PRS scores in batch mode."""
    pgs_ids = ["PGS000001", "PGS000002"]
    batch = compute_prs_batch(
        vcf_path=vcf_path,
        pgs_ids=pgs_ids,
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
    )

    assert batch.n_total == 2
    assert batch.n_ok == 2
    assert batch.n_failed == 0
    assert len(batch.results) == 2
    result_ids = {r.pgs_id for r in batch.results}
    assert result_ids == {"PGS000001", "PGS000002"}

    for r in batch.results:
        assert r.variants_matched > 0
        assert r.match_rate > 0.0
        assert math.isfinite(r.score)
