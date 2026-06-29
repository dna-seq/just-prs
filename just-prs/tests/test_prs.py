"""PRS computation tests using the antonkulaga.vcf test data."""

import math
from pathlib import Path

import polars as pl
import pytest

from just_prs.prs import (
    PRSEngine,
    compute_prs,
    compute_prs_batch,
    compute_prs_duckdb,
    prepare_reference_universe,
)
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


def test_resolve_reference_fills_missing_ref_both_engines(tmp_path: Path) -> None:
    """resolve_reference fills null reference_allele from the universe parquet,
    dissolving variants_unscorable_absent and attributing the source per tier."""
    geno_path = tmp_path / "variant_only.parquet"
    pl.DataFrame({
        "chrom": ["1"],
        "pos": [100],
        "ref": ["A"],
        "alt": ["G"],
        "GT": ["0/1"],
    }).write_parquet(geno_path)

    scoring = pl.DataFrame({
        "hm_chr": ["1", "1", "1", "1"],
        "hm_pos": [100, 200, 300, 400],
        "effect_allele": ["G", "T", "C", "A"],
        "reference_allele": ["A", "T", None, None],  # 300 & 400 unknown
        "effect_weight": [1.0, 2.0, 3.0, 0.5],
    }).lazy()

    # Precomputed REF universe: panel resolves 300 (C), FASTA resolves 400 (A).
    universe_path = tmp_path / "reference_allele_universe.parquet"
    pl.DataFrame({
        "genome_build": ["GRCh38", "GRCh38"],
        "chrom": ["1", "1"],
        "pos": [300, 400],
        "ref": ["C", "A"],
        "ref_source": ["panel", "fasta"],
    }).write_parquet(universe_path)

    # Without resolution: 300 & 400 are unscorable (existing behaviour).
    off = compute_prs(
        vcf_path="", scoring_file=scoring, cache_dir=tmp_path, pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path), genotype_input_mode="variant_only",
    )
    assert off.score == pytest.approx(5.0)  # 1 (obs G) + 4 (hom-ref T/T)
    assert off.variants_unscorable_absent == 2
    assert off.variants_ref_resolved_panel == 0
    assert off.variants_ref_resolved_fasta == 0

    # With resolution: 300 (+6) and 400 (+1) recovered as hom-ref.
    polars_result = compute_prs(
        vcf_path="", scoring_file=scoring, cache_dir=tmp_path, pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path), genotype_input_mode="variant_only",
        reference_restoration=True, reference_universe_path=universe_path,
    )
    duckdb_result = compute_prs_duckdb(
        vcf_path="", scoring_file=scoring, cache_dir=tmp_path, pgs_id="PGSTEST",
        genotypes_parquet=geno_path, genotype_input_mode="variant_only",
        reference_restoration=True, reference_universe_path=universe_path,
    )

    for result in (polars_result, duckdb_result):
        assert result.score == pytest.approx(12.0)  # 1 + 4 + 6 + 1
        assert result.variants_unscorable_absent == 0
        assert result.variants_assumed_hom_ref == 3
        assert result.variants_ref_resolved_panel == 1
        assert result.variants_ref_resolved_fasta == 1
        assert result.variants_matched == 4


def test_reference_restoration_scope_restricts_to_position_set(tmp_path: Path) -> None:
    """A position-set scope fills only in-set absent loci; off-set absent stays unscorable
    (the array/chip semantics — untyped positions must not be phantom hom-ref)."""
    geno_path = tmp_path / "variant_only.parquet"
    pl.DataFrame({
        "chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["G"], "GT": ["0/1"],
    }).write_parquet(geno_path)

    scoring = pl.DataFrame({
        "hm_chr": ["1", "1", "1"],
        "hm_pos": [100, 300, 400],
        "effect_allele": ["G", "C", "A"],
        "reference_allele": ["A", None, None],  # 300 & 400 unknown
        "effect_weight": [1.0, 3.0, 0.5],
    }).lazy()

    universe_path = tmp_path / "reference_allele_universe.parquet"
    pl.DataFrame({
        "genome_build": ["GRCh38", "GRCh38"],
        "chrom": ["1", "1"], "pos": [300, 400], "ref": ["C", "A"],
        "ref_source": ["panel", "fasta"],
    }).write_parquet(universe_path)

    # Scope = only position 300 → 400 must remain unscorable despite being in the universe.
    scope = pl.DataFrame({"chrom": ["1"], "pos": [300]})
    for engine_kwargs in (
        {"genotypes_lf": pl.scan_parquet(geno_path)},
        {"genotypes_parquet": geno_path},
    ):
        fn = compute_prs if "genotypes_lf" in engine_kwargs else compute_prs_duckdb
        r = fn(
            vcf_path="", scoring_file=scoring, cache_dir=tmp_path, pgs_id="PGSTEST",
            genotype_input_mode="variant_only",
            reference_restoration=scope, reference_universe_path=universe_path,
            **engine_kwargs,
        )
        assert r.variants_ref_resolved_panel == 1   # 300 filled
        assert r.variants_ref_resolved_fasta == 0   # 400 off-scope, not filled
        assert r.variants_unscorable_absent == 1    # 400 stays unscorable
        assert r.score == pytest.approx(7.0)        # 1 (obs) + 6 (300 hom-ref) ; 400 dropped


def test_prepared_universe_handle_matches_path_both_engines(tmp_path: Path) -> None:
    """An injected ReferenceUniverse handle (dependency injection) produces results
    identical to passing reference_universe_path — the prepare-once fast path must
    not change the score or the per-tier accounting."""
    geno_path = tmp_path / "variant_only.parquet"
    pl.DataFrame({
        "chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["G"], "GT": ["0/1"],
    }).write_parquet(geno_path)

    scoring = pl.DataFrame({
        "hm_chr": ["1", "1", "1", "1"],
        "hm_pos": [100, 200, 300, 400],
        "effect_allele": ["G", "T", "C", "A"],
        "reference_allele": ["A", "T", None, None],
        "effect_weight": [1.0, 2.0, 3.0, 0.5],
    }).lazy()

    universe_path = tmp_path / "reference_allele_universe.parquet"
    pl.DataFrame({
        "genome_build": ["GRCh38", "GRCh38"],
        "chrom": ["1", "1"], "pos": [300, 400], "ref": ["C", "A"],
        "ref_source": ["panel", "fasta"],
    }).write_parquet(universe_path)

    handle = prepare_reference_universe(universe_path, genome_build="GRCh38")
    assert handle.n_positions == 2
    assert handle.scoped is False

    for fn, geno_kwargs in (
        (compute_prs, {"genotypes_lf": pl.scan_parquet(geno_path)}),
        (compute_prs_duckdb, {"genotypes_parquet": geno_path}),
    ):
        via_path = fn(
            vcf_path="", scoring_file=scoring, cache_dir=tmp_path, pgs_id="PGSTEST",
            genotype_input_mode="variant_only",
            reference_restoration=True, reference_universe_path=universe_path,
            **geno_kwargs,
        )
        via_handle = fn(
            vcf_path="", scoring_file=scoring, cache_dir=tmp_path, pgs_id="PGSTEST",
            genotype_input_mode="variant_only",
            reference_universe=handle,
            **geno_kwargs,
        )
        # Pure dependency injection: no path, restoration left at its False default —
        # the handle's presence is the intent.
        assert via_handle.score == pytest.approx(via_path.score)
        assert via_handle.score == pytest.approx(12.0)
        assert via_handle.variants_ref_resolved_panel == via_path.variants_ref_resolved_panel == 1
        assert via_handle.variants_ref_resolved_fasta == via_path.variants_ref_resolved_fasta == 1
        assert via_handle.variants_unscorable_absent == via_path.variants_unscorable_absent == 0


def test_prepared_universe_handle_respects_baked_scope(tmp_path: Path) -> None:
    """A handle prepared with a scope restricts the eligible REF set; off-scope absent
    loci stay unscorable even though they exist in the universe."""
    geno_path = tmp_path / "variant_only.parquet"
    pl.DataFrame({
        "chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["G"], "GT": ["0/1"],
    }).write_parquet(geno_path)
    scoring = pl.DataFrame({
        "hm_chr": ["1", "1", "1"],
        "hm_pos": [100, 300, 400],
        "effect_allele": ["G", "C", "A"],
        "reference_allele": ["A", None, None],
        "effect_weight": [1.0, 3.0, 0.5],
    }).lazy()
    universe = pl.DataFrame({
        "chrom": ["1", "1"], "pos": [300, 400], "ref": ["C", "A"],
        "ref_source": ["panel", "fasta"],
    })

    handle = prepare_reference_universe(
        universe, scope=pl.DataFrame({"chrom": ["1"], "pos": [300]})
    )
    assert handle.scoped is True
    assert handle.n_positions == 1

    r = compute_prs(
        vcf_path="", scoring_file=scoring, cache_dir=tmp_path, pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path), genotype_input_mode="variant_only",
        reference_universe=handle,
    )
    assert r.variants_ref_resolved_panel == 1   # 300 filled
    assert r.variants_ref_resolved_fasta == 0   # 400 off-scope, not filled
    assert r.variants_unscorable_absent == 1
    assert r.score == pytest.approx(7.0)


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


def _write_grch37_vcf_header(path: Path) -> None:
    """A minimal VCF whose contig lengths identify it as GRCh37 (>=3 votes)."""
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=249250621>\n"
        "##contig=<ID=2,length=243199373>\n"
        "##contig=<ID=3,length=198022430>\n"
        "##contig=<ID=4,length=191154276>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )


def test_detected_build_mismatch_flagged(tmp_path: Path) -> None:
    """A GRCh37 VCF scored against a GRCh38 build is detected and flagged (F4)."""
    vcf_file = tmp_path / "grch37.vcf"
    _write_grch37_vcf_header(vcf_file)

    geno_path = tmp_path / "geno.parquet"
    pl.DataFrame({
        "chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["G"], "GT": ["0/1"],
    }).write_parquet(geno_path)

    scoring = pl.DataFrame({
        "hm_chr": ["1"], "hm_pos": [100],
        "effect_allele": ["G"], "reference_allele": ["A"], "effect_weight": [1.0],
    }).lazy()

    # Build detection reads vcf_file's header; genotypes come from the parquet/lf.
    polars_result = compute_prs(
        vcf_path=vcf_file,
        scoring_file=scoring,
        genome_build="GRCh38",
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path),
        genotype_input_mode="variant_only",
    )
    duckdb_result = compute_prs_duckdb(
        vcf_path=vcf_file,
        scoring_file=scoring,
        genome_build="GRCh38",
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_parquet=geno_path,
        genotype_input_mode="variant_only",
    )

    for result in (polars_result, duckdb_result):
        assert result.detected_genome_build == "GRCh37"
        assert result.build_mismatch is True


def test_detected_build_match_no_flag(tmp_path: Path) -> None:
    """A GRCh37 VCF scored against GRCh37 is detected with no mismatch flag (F4)."""
    vcf_file = tmp_path / "grch37.vcf"
    _write_grch37_vcf_header(vcf_file)
    geno_path = tmp_path / "geno.parquet"
    pl.DataFrame({
        "chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["G"], "GT": ["0/1"],
    }).write_parquet(geno_path)
    scoring = pl.DataFrame({
        "hm_chr": ["1"], "hm_pos": [100],
        "effect_allele": ["G"], "reference_allele": ["A"], "effect_weight": [1.0],
    }).lazy()

    result = compute_prs(
        vcf_path=vcf_file,
        scoring_file=scoring,
        genome_build="GRCh37",
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path),
        genotype_input_mode="variant_only",
    )
    assert result.detected_genome_build == "GRCh37"
    assert result.build_mismatch is False


def test_no_build_detection_for_prenormalized_input(tmp_path: Path) -> None:
    """A header-less / non-VCF genotype input yields no detected build, no false mismatch."""
    geno_path = tmp_path / "geno.parquet"
    pl.DataFrame({
        "chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["G"], "GT": ["0/1"],
    }).write_parquet(geno_path)
    scoring = pl.DataFrame({
        "hm_chr": ["1"], "hm_pos": [100],
        "effect_allele": ["G"], "reference_allele": ["A"], "effect_weight": [1.0],
    }).lazy()

    result = compute_prs(
        vcf_path="",
        scoring_file=scoring,
        genome_build="GRCh38",
        cache_dir=tmp_path,
        pgs_id="PGSTEST",
        genotypes_lf=pl.scan_parquet(geno_path),
        genotype_input_mode="variant_only",
    )
    assert result.detected_genome_build is None
    assert result.build_mismatch is False


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
