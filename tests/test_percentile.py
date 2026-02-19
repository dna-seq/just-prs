"""Tests for theoretical percentile computation from allele frequencies.

Validates the allelefrequency_effect -> theoretical mean/SD -> percentile pipeline
using real PGS scoring files that contain allele frequency data, and cross-validates
PRS scores against PLINK2 for the same scores.
"""

import math
import subprocess
import tempfile
from pathlib import Path

import polars as pl
import pytest

from just_prs.prs import (
    _compute_theoretical_stats,
    _norm_cdf,
    _normalize_scoring_columns,
    _resolve_scoring,
    compute_prs,
)
from just_prs.scoring import download_scoring_file, parse_scoring_file

PGS_IDS_WITH_FREQ = ["PGS000004", "PGS000005", "PGS000006", "PGS000010", "PGS000011"]
PGS_IDS_WITHOUT_FREQ = ["PGS000001", "PGS000002"]


class TestTheoreticalStats:
    """Validate _compute_theoretical_stats on real scoring files."""

    def test_scores_with_freq_produce_stats(self, scoring_cache_dir: Path) -> None:
        """All PGS IDs known to have allelefrequency_effect yield valid stats."""
        for pgs_id in PGS_IDS_WITH_FREQ:
            scoring_lf = _resolve_scoring(pgs_id, "GRCh38", scoring_cache_dir)
            scoring_df = _normalize_scoring_columns(scoring_lf).collect()

            assert "allelefrequency_effect" in scoring_df.columns, (
                f"{pgs_id} should have allelefrequency_effect"
            )

            stats = _compute_theoretical_stats(scoring_df)
            assert stats is not None, f"{pgs_id} should produce theoretical stats"

            mean, std, n_with_freq = stats
            assert math.isfinite(mean), f"{pgs_id}: mean not finite"
            assert math.isfinite(std), f"{pgs_id}: std not finite"
            assert std > 0, f"{pgs_id}: std should be positive"
            assert n_with_freq > 0, f"{pgs_id}: should have variants with freq"
            assert n_with_freq == scoring_df.height, (
                f"{pgs_id}: all variants should have freq"
            )

    def test_scores_without_freq_return_none(self, scoring_cache_dir: Path) -> None:
        """PGS IDs without allelefrequency_effect yield None stats."""
        for pgs_id in PGS_IDS_WITHOUT_FREQ:
            scoring_lf = _resolve_scoring(pgs_id, "GRCh38", scoring_cache_dir)
            scoring_df = _normalize_scoring_columns(scoring_lf).collect()
            stats = _compute_theoretical_stats(scoring_df)
            assert stats is None, f"{pgs_id} should NOT produce theoretical stats"

    def test_mean_equals_manual_computation(self, scoring_cache_dir: Path) -> None:
        """Verify mean = sum(w_i * 2 * p_i) by computing manually row-by-row."""
        scoring_lf = _resolve_scoring("PGS000010", "GRCh38", scoring_cache_dir)
        scoring_df = _normalize_scoring_columns(scoring_lf).collect()

        stats = _compute_theoretical_stats(scoring_df)
        assert stats is not None
        mean_from_fn, std_from_fn, _ = stats

        weights = scoring_df["effect_weight"].to_list()
        freqs = scoring_df["allelefrequency_effect"].to_list()

        manual_mean = sum(w * 2.0 * p for w, p in zip(weights, freqs))
        manual_var = sum(w * w * 2.0 * p * (1.0 - p) for w, p in zip(weights, freqs))
        manual_std = math.sqrt(manual_var)

        assert abs(mean_from_fn - manual_mean) < 1e-10, (
            f"Mean mismatch: {mean_from_fn} vs {manual_mean}"
        )
        assert abs(std_from_fn - manual_std) < 1e-10, (
            f"Std mismatch: {std_from_fn} vs {manual_std}"
        )

    def test_std_is_positive_and_reasonable(self, scoring_cache_dir: Path) -> None:
        """SD should be positive and not astronomically large."""
        for pgs_id in PGS_IDS_WITH_FREQ:
            scoring_lf = _resolve_scoring(pgs_id, "GRCh38", scoring_cache_dir)
            scoring_df = _normalize_scoring_columns(scoring_lf).collect()
            stats = _compute_theoretical_stats(scoring_df)
            assert stats is not None
            _, std, _ = stats
            assert 0 < std < 100, f"{pgs_id}: SD={std} out of reasonable range"

    def test_empty_freq_column_returns_none(self) -> None:
        """DataFrame with allelefrequency_effect column but all nulls returns None."""
        df = pl.DataFrame({
            "effect_weight": [0.5, -0.3],
            "allelefrequency_effect": [None, None],
        })
        assert _compute_theoretical_stats(df) is None

    def test_boundary_frequencies_excluded(self) -> None:
        """Frequencies of exactly 0.0 or 1.0 are excluded (monomorphic sites)."""
        df = pl.DataFrame({
            "effect_weight": [0.5, -0.3, 0.1],
            "allelefrequency_effect": [0.0, 1.0, 0.5],
        })
        stats = _compute_theoretical_stats(df)
        assert stats is not None
        mean, std, n = stats
        assert n == 1, "Only the p=0.5 variant should be included"
        assert abs(mean - 0.1 * 2.0 * 0.5) < 1e-10
        expected_var = 0.1**2 * 2.0 * 0.5 * 0.5
        assert abs(std - math.sqrt(expected_var)) < 1e-10


class TestNormCdf:
    """Validate the _norm_cdf implementation against known values."""

    def test_cdf_at_zero(self) -> None:
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-10

    def test_cdf_symmetry(self) -> None:
        for z in [0.5, 1.0, 1.96, 2.5, 3.0]:
            assert abs(_norm_cdf(z) + _norm_cdf(-z) - 1.0) < 1e-6

    def test_known_values(self) -> None:
        assert abs(_norm_cdf(1.96) - 0.975) < 0.001
        assert abs(_norm_cdf(-1.96) - 0.025) < 0.001
        assert abs(_norm_cdf(1.0) - 0.8413) < 0.001
        assert abs(_norm_cdf(2.0) - 0.9772) < 0.001


class TestPercentileEndToEnd:
    """End-to-end tests: compute PRS with percentile on the Zenodo test VCF."""

    def test_prs_result_has_percentile_for_freq_scores(
        self, vcf_path: Path, scoring_cache_dir: Path
    ) -> None:
        """Scores with allele frequencies should produce a valid percentile.

        Percentile can be 0.0 or 100.0 for extreme z-scores (e.g. when only a
        small fraction of variants matched), so we only check 0 <= pct <= 100.
        """
        for pgs_id in PGS_IDS_WITH_FREQ:
            result = compute_prs(
                vcf_path=vcf_path,
                scoring_file=pgs_id,
                genome_build="GRCh38",
                cache_dir=scoring_cache_dir,
                pgs_id=pgs_id,
            )
            assert result.has_allele_frequencies, f"{pgs_id}: should have freq"
            assert result.theoretical_mean is not None, f"{pgs_id}: no mean"
            assert result.theoretical_std is not None, f"{pgs_id}: no std"
            assert result.theoretical_std > 0, f"{pgs_id}: std should be positive"
            assert result.percentile is not None, f"{pgs_id}: no percentile"
            assert 0 <= result.percentile <= 100, (
                f"{pgs_id}: percentile={result.percentile} should be in [0, 100]"
            )

    def test_prs_result_no_percentile_for_non_freq_scores(
        self, vcf_path: Path, scoring_cache_dir: Path
    ) -> None:
        """Scores without allele frequencies should have None percentile."""
        for pgs_id in PGS_IDS_WITHOUT_FREQ:
            result = compute_prs(
                vcf_path=vcf_path,
                scoring_file=pgs_id,
                genome_build="GRCh38",
                cache_dir=scoring_cache_dir,
                pgs_id=pgs_id,
            )
            assert not result.has_allele_frequencies
            assert result.theoretical_mean is None
            assert result.theoretical_std is None
            assert result.percentile is None

    def test_percentile_consistent_with_z_score(
        self, vcf_path: Path, scoring_cache_dir: Path
    ) -> None:
        """Verify percentile = Phi((score - mean) / std) * 100 exactly."""
        result = compute_prs(
            vcf_path=vcf_path,
            scoring_file="PGS000004",
            genome_build="GRCh38",
            cache_dir=scoring_cache_dir,
            pgs_id="PGS000004",
        )
        assert result.percentile is not None
        assert result.theoretical_mean is not None
        assert result.theoretical_std is not None

        z = (result.score - result.theoretical_mean) / result.theoretical_std
        expected_pct = round(_norm_cdf(z) * 100.0, 2)
        assert result.percentile == expected_pct

    def test_different_scores_give_different_percentiles(
        self, vcf_path: Path, scoring_cache_dir: Path
    ) -> None:
        """Different PGS IDs on the same VCF produce different percentiles."""
        percentiles: dict[str, float] = {}
        for pgs_id in PGS_IDS_WITH_FREQ[:3]:
            result = compute_prs(
                vcf_path=vcf_path,
                scoring_file=pgs_id,
                genome_build="GRCh38",
                cache_dir=scoring_cache_dir,
                pgs_id=pgs_id,
            )
            assert result.percentile is not None
            percentiles[pgs_id] = result.percentile

        unique_pcts = set(percentiles.values())
        assert len(unique_pcts) > 1, (
            f"Expected different percentiles across scores, got: {percentiles}"
        )


class TestPercentileVsPlink2:
    """Cross-validate PRS scores against PLINK2 for scores with allele frequencies.

    PLINK2 does not compute percentiles, so we verify:
    1. Our PRS score matches PLINK2 (raw score validation)
    2. The theoretical stats are mathematically consistent
    3. Percentile is in a reasonable range given the score
    """

    @pytest.mark.parametrize("pgs_id", PGS_IDS_WITH_FREQ)
    def test_prs_score_matches_plink2(
        self,
        pgs_id: str,
        vcf_path: Path,
        scoring_cache_dir: Path,
        plink2_path: Path,
    ) -> None:
        """Our PRS score matches PLINK2 --score for scores with allele frequencies."""
        scoring_path = download_scoring_file(
            pgs_id, scoring_cache_dir, genome_build="GRCh38"
        )

        our_result = compute_prs(
            vcf_path=vcf_path,
            scoring_file=scoring_path,
            genome_build="GRCh38",
            cache_dir=scoring_cache_dir,
            pgs_id=pgs_id,
        )

        assert our_result.has_allele_frequencies
        assert our_result.percentile is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            plink_score_path = _prepare_plink_scoring_file(scoring_path, work_dir)
            plink_score = _run_plink2_prs(
                plink2_path, vcf_path, plink_score_path, work_dir
            )

        if abs(plink_score) > 1e-10:
            relative_diff = abs(our_result.score - plink_score) / abs(plink_score)
            assert relative_diff < 0.05, (
                f"{pgs_id}: score mismatch ours={our_result.score:.6f} "
                f"plink2={plink_score:.6f} rel_diff={relative_diff:.4f}"
            )
        else:
            assert abs(our_result.score - plink_score) < 1e-6

    def test_percentile_summary_table(
        self,
        vcf_path: Path,
        scoring_cache_dir: Path,
        plink2_path: Path,
    ) -> None:
        """Build a summary table comparing PRS + percentile across all freq scores."""
        rows: list[dict[str, object]] = []

        for pgs_id in PGS_IDS_WITH_FREQ:
            scoring_path = download_scoring_file(
                pgs_id, scoring_cache_dir, genome_build="GRCh38"
            )

            our_result = compute_prs(
                vcf_path=vcf_path,
                scoring_file=scoring_path,
                genome_build="GRCh38",
                cache_dir=scoring_cache_dir,
                pgs_id=pgs_id,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                work_dir = Path(tmpdir)
                plink_score_path = _prepare_plink_scoring_file(scoring_path, work_dir)
                plink_score = _run_plink2_prs(
                    plink2_path, vcf_path, plink_score_path, work_dir
                )

            if abs(plink_score) > 1e-10:
                rel_diff = abs(our_result.score - plink_score) / abs(plink_score)
            else:
                rel_diff = abs(our_result.score - plink_score)

            rows.append({
                "pgs_id": pgs_id,
                "our_score": round(our_result.score, 6),
                "plink_score": round(plink_score, 6),
                "rel_diff": round(rel_diff, 6),
                "theo_mean": round(our_result.theoretical_mean, 6) if our_result.theoretical_mean is not None else None,
                "theo_std": round(our_result.theoretical_std, 6) if our_result.theoretical_std is not None else None,
                "percentile": our_result.percentile,
                "match_rate": round(our_result.match_rate, 3),
                "variants": our_result.variants_total,
            })

        summary = pl.DataFrame(rows)
        print("\n=== PRS + Percentile: just_prs vs PLINK2 ===")
        print(summary)

        for row in rows:
            assert float(row["rel_diff"]) < 0.05, (  # type: ignore[arg-type]
                f"{row['pgs_id']}: score mismatch rel_diff={row['rel_diff']}"
            )
            pct = row["percentile"]
            assert pct is not None and 0 <= float(pct) <= 100, (  # type: ignore[arg-type]
                f"{row['pgs_id']}: percentile={pct} out of range"
            )


def _prepare_plink_scoring_file(scoring_path: Path, output_dir: Path) -> Path:
    """Convert PGS Catalog scoring file to PLINK2 --score format."""
    lf = parse_scoring_file(scoring_path)
    df = lf.collect()

    columns = df.columns
    if "hm_chr" in columns and "hm_pos" in columns:
        chr_col, pos_col = "hm_chr", "hm_pos"
    else:
        chr_col, pos_col = "chr_name", "chr_position"

    plink_df = df.select([
        (pl.col(chr_col).cast(pl.Utf8) + pl.lit(":") + pl.col(pos_col).cast(pl.Utf8)).alias("ID"),
        pl.col("effect_allele").alias("A1"),
        pl.col("effect_weight").alias("WEIGHT"),
    ]).drop_nulls()

    plink_score_path = output_dir / "plink_score.txt"
    plink_df.write_csv(plink_score_path, separator="\t")
    return plink_score_path


def _run_plink2_prs(
    plink2_bin: Path,
    vcf_path: Path,
    plink_score_path: Path,
    work_dir: Path,
) -> float:
    """Run PLINK2 to compute PRS and return the score."""
    pgen_prefix = work_dir / "genotypes"
    subprocess.run(
        [
            str(plink2_bin),
            "--vcf", str(vcf_path),
            "--set-all-var-ids", "@:#",
            "--make-pgen",
            "--out", str(pgen_prefix),
            "--allow-extra-chr",
            "--autosome",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result_prefix = work_dir / "prs_result"
    subprocess.run(
        [
            str(plink2_bin),
            "--pfile", str(pgen_prefix),
            "--score", str(plink_score_path), "1", "2", "3",
            "header",
            "no-mean-imputation",
            "cols=+scoresums",
            "--out", str(result_prefix),
            "--allow-extra-chr",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    sscore_path = result_prefix.with_suffix(".sscore")
    sscore_df = pl.read_csv(sscore_path, separator="\t")

    score_col = None
    for col in sscore_df.columns:
        if "SCORESUM" in col.upper() or "SCORE1_SUM" in col.upper():
            score_col = col
            break
    if score_col is None:
        for col in sscore_df.columns:
            if "SCORE" in col.upper() and "AVG" not in col.upper():
                score_col = col
                break
    if score_col is None:
        raise ValueError(f"Could not find score column in {sscore_df.columns}")

    return float(sscore_df[score_col][0])
