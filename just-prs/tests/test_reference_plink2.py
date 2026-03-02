"""Integration tests for reference panel PLINK2 scoring.

These tests require:
  - The 1000G reference panel extracted at ~/.cache/just-prs/reference_panel/
  - A PLINK2 binary (auto-downloaded by the plink2_path fixture)
  - Network access to download PGS scoring files from the PGS Catalog FTP

Tests are skipped automatically if the reference panel is not available.
Run with: ``uv run pytest just-prs/tests/test_reference_plink2.py -v``
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from just_prs.reference import (
    SUPERPOPULATIONS,
    _prepare_plink2_score_input,
    _resolve_other_allele_col,
    compute_reference_prs_plink2,
    reference_panel_dir,
)
from just_prs.scoring import download_scoring_file, resolve_cache_dir

TEST_CACHE_DIR = resolve_cache_dir() / "test-data"

REF_DIR = reference_panel_dir()
REF_PANEL_AVAILABLE = (REF_DIR / "GRCh38_1000G_ALL.pgen").exists()

PGS_IDS_REFERENCE = ["PGS000001", "PGS000002", "PGS000004", "PGS000010"]


@pytest.fixture(scope="session")
def scoring_cache() -> Path:
    cache = resolve_cache_dir() / "scores"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture(scope="session")
def ref_dir() -> Path:
    if not REF_PANEL_AVAILABLE:
        pytest.skip("1000G reference panel not available")
    return REF_DIR


# ---------------------------------------------------------------------------
# _prepare_plink2_score_input unit tests
# ---------------------------------------------------------------------------

class TestPrepareScoreInput:
    """Tests for the variant ID construction in _prepare_plink2_score_input."""

    def test_generates_4part_ids_with_other_allele(self, scoring_cache: Path) -> None:
        """When other_allele is present, IDs should be chr:pos:allele1:allele2."""
        scoring_file = download_scoring_file("PGS000001", scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result = _prepare_plink2_score_input(scoring_file, "PGS000001", out_dir)
            assert result is not None

            df = pl.read_csv(result, separator="\t")
            assert "variant_id" in df.columns
            assert "effect_allele" in df.columns
            assert "effect_weight" in df.columns

            for vid in df["variant_id"].head(20).to_list():
                parts = vid.split(":")
                assert len(parts) == 4, f"Expected 4-part ID, got {vid}"
                assert parts[0].isdigit() or parts[0] in ("X", "Y", "MT"), f"Bad chrom in {vid}"
                assert parts[1].isdigit(), f"Bad position in {vid}"
                assert len(parts[2]) > 0, f"Empty allele1 in {vid}"
                assert len(parts[3]) > 0, f"Empty allele2 in {vid}"

    def test_emits_both_orderings(self, scoring_cache: Path) -> None:
        """Each variant should appear twice (fwd and rev allele ordering)."""
        scoring_file = download_scoring_file("PGS000001", scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result = _prepare_plink2_score_input(scoring_file, "PGS000001", out_dir)
            assert result is not None

            df = pl.read_csv(result, separator="\t")
            n_unique_positions = df["variant_id"].str.extract(r"^(\d+:\d+):").n_unique()
            assert df.height == n_unique_positions * 2, (
                f"Expected 2 rows per unique chr:pos, got {df.height} rows "
                f"for {n_unique_positions} positions"
            )

    def test_uses_hm_infer_other_allele(self, scoring_cache: Path) -> None:
        """PGS000010 has no other_allele column but has hm_inferOtherAllele."""
        scoring_file = download_scoring_file("PGS000010", scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result = _prepare_plink2_score_input(scoring_file, "PGS000010", out_dir)
            assert result is not None

            df = pl.read_csv(result, separator="\t")
            assert df.height > 0
            for vid in df["variant_id"].head(10).to_list():
                parts = vid.split(":")
                assert len(parts) == 4, f"Expected 4-part ID from hm_inferOtherAllele, got {vid}"

    def test_weights_are_numeric(self, scoring_cache: Path) -> None:
        """All effect_weight values should be valid floats."""
        scoring_file = download_scoring_file("PGS000001", scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result = _prepare_plink2_score_input(scoring_file, "PGS000001", out_dir)
            assert result is not None

            df = pl.read_csv(result, separator="\t")
            assert df["effect_weight"].null_count() == 0
            assert df["effect_weight"].dtype in (pl.Float64, pl.Float32, pl.Int64)


class TestResolveOtherAlleleCol:
    """Tests for _resolve_other_allele_col helper."""

    def test_prefers_other_allele(self) -> None:
        df = pl.DataFrame({"other_allele": ["C"], "hm_inferOtherAllele": ["G"]})
        assert _resolve_other_allele_col(df) == "other_allele"

    def test_falls_back_to_hm_infer(self) -> None:
        df = pl.DataFrame({"hm_inferOtherAllele": ["G"], "effect_allele": ["A"]})
        assert _resolve_other_allele_col(df) == "hm_inferOtherAllele"

    def test_falls_back_to_reference_allele(self) -> None:
        df = pl.DataFrame({"reference_allele": ["T"], "effect_allele": ["A"]})
        assert _resolve_other_allele_col(df) == "reference_allele"

    def test_returns_none_when_missing(self) -> None:
        df = pl.DataFrame({"effect_allele": ["A"], "effect_weight": [0.5]})
        assert _resolve_other_allele_col(df) is None


# ---------------------------------------------------------------------------
# compute_reference_prs_plink2 integration tests (need reference panel)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REF_PANEL_AVAILABLE, reason="1000G reference panel not available")
class TestComputeReferencePrsPlink2:
    """End-to-end integration tests for compute_reference_prs_plink2."""

    @pytest.mark.parametrize("pgs_id", PGS_IDS_REFERENCE)
    def test_scores_all_samples(
        self, pgs_id: str, ref_dir: Path, plink2_path: Path, scoring_cache: Path,
    ) -> None:
        """Each PGS ID should produce scores for all 3202 reference panel samples."""
        scoring_file = download_scoring_file(pgs_id, scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result_df = compute_reference_prs_plink2(
                pgs_id=pgs_id,
                scoring_file=scoring_file,
                ref_dir=ref_dir,
                out_dir=out_dir,
                plink2_bin=plink2_path,
                genome_build="GRCh38",
            )
            assert result_df.height == 3202, (
                f"{pgs_id}: expected 3202 samples, got {result_df.height}"
            )

    @pytest.mark.parametrize("pgs_id", PGS_IDS_REFERENCE)
    def test_has_required_columns(
        self, pgs_id: str, ref_dir: Path, plink2_path: Path, scoring_cache: Path,
    ) -> None:
        """Output DataFrame must have iid, score, superpop, population, pgs_id."""
        scoring_file = download_scoring_file(pgs_id, scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result_df = compute_reference_prs_plink2(
                pgs_id=pgs_id,
                scoring_file=scoring_file,
                ref_dir=ref_dir,
                out_dir=out_dir,
                plink2_bin=plink2_path,
                genome_build="GRCh38",
            )
            expected_cols = {"iid", "score", "superpop", "population", "pgs_id"}
            assert expected_cols == set(result_df.columns), (
                f"{pgs_id}: expected columns {expected_cols}, got {set(result_df.columns)}"
            )

    @pytest.mark.parametrize("pgs_id", PGS_IDS_REFERENCE)
    def test_nonzero_score_variance(
        self, pgs_id: str, ref_dir: Path, plink2_path: Path, scoring_cache: Path,
    ) -> None:
        """Scores should have non-trivial variance (not all identical)."""
        scoring_file = download_scoring_file(pgs_id, scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result_df = compute_reference_prs_plink2(
                pgs_id=pgs_id,
                scoring_file=scoring_file,
                ref_dir=ref_dir,
                out_dir=out_dir,
                plink2_bin=plink2_path,
                genome_build="GRCh38",
            )
            std = result_df["score"].std()
            assert std is not None and std > 1e-10, (
                f"{pgs_id}: score std is {std}, expected non-trivial variance"
            )

    def test_all_superpopulations_present(
        self, ref_dir: Path, plink2_path: Path, scoring_cache: Path,
    ) -> None:
        """PGS000001 output should contain all 5 superpopulations."""
        scoring_file = download_scoring_file("PGS000001", scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result_df = compute_reference_prs_plink2(
                pgs_id="PGS000001",
                scoring_file=scoring_file,
                ref_dir=ref_dir,
                out_dir=out_dir,
                plink2_bin=plink2_path,
                genome_build="GRCh38",
            )
            superpops = set(result_df["superpop"].unique().to_list())
            assert superpops == set(SUPERPOPULATIONS), (
                f"Expected {set(SUPERPOPULATIONS)}, got {superpops}"
            )

    def test_superpopulation_means_differ(
        self, ref_dir: Path, plink2_path: Path, scoring_cache: Path,
    ) -> None:
        """Different superpopulations should have different mean scores."""
        scoring_file = download_scoring_file("PGS000001", scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result_df = compute_reference_prs_plink2(
                pgs_id="PGS000001",
                scoring_file=scoring_file,
                ref_dir=ref_dir,
                out_dir=out_dir,
                plink2_bin=plink2_path,
                genome_build="GRCh38",
            )
            means = (
                result_df.group_by("superpop")
                .agg(pl.col("score").mean().alias("mean"))
                .sort("superpop")
            )
            mean_values = means["mean"].to_list()
            assert len(set(round(m, 6) for m in mean_values)) > 1, (
                f"All superpopulation means are identical: {mean_values}"
            )

    def test_pgs_id_column_is_consistent(
        self, ref_dir: Path, plink2_path: Path, scoring_cache: Path,
    ) -> None:
        """The pgs_id column should contain only the scored PGS ID."""
        scoring_file = download_scoring_file("PGS000001", scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result_df = compute_reference_prs_plink2(
                pgs_id="PGS000001",
                scoring_file=scoring_file,
                ref_dir=ref_dir,
                out_dir=out_dir,
                plink2_bin=plink2_path,
                genome_build="GRCh38",
            )
            unique_ids = result_df["pgs_id"].unique().to_list()
            assert unique_ids == ["PGS000001"], f"Unexpected pgs_id values: {unique_ids}"

    def test_aggregate_distributions_from_reference_scores(
        self, ref_dir: Path, plink2_path: Path, scoring_cache: Path,
    ) -> None:
        """End-to-end: score PGS000001 → aggregate → produces valid distributions."""
        from just_prs.reference import aggregate_distributions

        scoring_file = download_scoring_file("PGS000001", scoring_cache, genome_build="GRCh38")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result_df = compute_reference_prs_plink2(
                pgs_id="PGS000001",
                scoring_file=scoring_file,
                ref_dir=ref_dir,
                out_dir=out_dir,
                plink2_bin=plink2_path,
                genome_build="GRCh38",
            )
            dist_df = aggregate_distributions(result_df)
            assert dist_df.height == 5, f"Expected 5 rows (1 per superpop), got {dist_df.height}"
            assert set(dist_df["superpopulation"].to_list()) == set(SUPERPOPULATIONS)
            for row in dist_df.iter_rows(named=True):
                assert row["n"] > 0
                assert row["std"] > 0, f"Zero std for {row['superpopulation']}"
