"""Cross-engine validation: PLINK2, polars, and DuckDB produce the same PRS scores.

All three reference panel scoring engines must agree on per-sample scores
for the same PGS IDs. This test scores several PGS IDs with each engine
and verifies that sample-level scores match within floating-point tolerance.

Engines:
  - PLINK2: ``compute_reference_prs_plink2`` (external binary, ground truth)
  - polars: ``match_scoring_to_pvar`` + ``compute_reference_prs_polars`` with pvar DataFrame
  - DuckDB: ``_ResolvedRefPanel.match_scoring`` + ``compute_reference_prs_polars`` (current default)

Requires:
  - 1000G reference panel at ~/.cache/just-prs/reference_panel/
  - PLINK2 binary (auto-downloaded by conftest fixture)
  - Network access for PGS scoring file downloads
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from just_prs.reference import (
    _ResolvedRefPanel,
    compute_reference_prs_plink2,
    compute_reference_prs_polars,
    match_scoring_to_pvar,
    parse_pvar,
    read_pgen_genotypes,
    reference_panel_dir,
)
from just_prs.prs import _normalize_scoring_columns
from just_prs.scoring import download_scoring_file, parse_scoring_file, resolve_cache_dir

REF_DIR = reference_panel_dir()
REF_PANEL_AVAILABLE = (REF_DIR / "GRCh38_1000G_ALL.pgen").exists()

PGS_IDS = ["PGS000001", "PGS000003", "PGS000007"]


@pytest.fixture(scope="module")
def scoring_cache() -> Path:
    cache = resolve_cache_dir() / "scores"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture(scope="module")
def ref_dir() -> Path:
    if not REF_PANEL_AVAILABLE:
        pytest.skip("1000G reference panel not available")
    return REF_DIR


@pytest.fixture(scope="module")
def resolved_panel(ref_dir: Path) -> _ResolvedRefPanel:
    return _ResolvedRefPanel(ref_dir, genome_build="GRCh38")


def _score_with_duckdb(
    pgs_id: str,
    scoring_file: Path,
    ref_dir: Path,
    panel: _ResolvedRefPanel,
) -> pl.DataFrame:
    """Score using the DuckDB engine (current default path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        return compute_reference_prs_polars(
            pgs_id=pgs_id,
            scoring_file=scoring_file,
            ref_dir=ref_dir,
            out_dir=Path(tmpdir),
            genome_build="GRCh38",
            _panel=panel,
        )


def _score_with_polars_match(
    pgs_id: str,
    scoring_file: Path,
    panel: _ResolvedRefPanel,
) -> pl.DataFrame:
    """Score using the old polars match_scoring_to_pvar (loads full pvar)."""
    import numpy as np

    pvar_df = parse_pvar(panel.pvar_zst_path)
    scoring_lf = parse_scoring_file(scoring_file)
    scoring_df = _normalize_scoring_columns(scoring_lf).collect()

    matched = match_scoring_to_pvar(pvar_df, scoring_df)
    del pvar_df, scoring_df

    variant_indices = matched["variant_idx"].cast(pl.UInt32).to_numpy()
    geno = read_pgen_genotypes(
        pgen_path=panel.pgen_path,
        pvar_zst_path=panel.pvar_zst_path,
        variant_indices=variant_indices,
        n_samples=panel.psam_df.height,
        pvar_variant_ct=panel.pvar_variant_ct,
    )

    weights = matched["effect_weight"].to_numpy()
    is_alt = matched["effect_is_alt"].to_numpy()
    missing_mask = geno == -9
    geno_float = geno.astype(np.float64)
    del geno

    dosage = np.where(is_alt[:, np.newaxis], geno_float, 2.0 - geno_float)
    del geno_float
    dosage = np.where(missing_mask, 0.0, dosage)
    del missing_mask

    prs_sum = (dosage * weights[:, np.newaxis]).sum(axis=0)
    del dosage
    prs_avg = prs_sum / (2 * matched.height)

    return (
        pl.DataFrame({"iid": panel.psam_df["iid"].to_list(), "score": prs_avg.tolist()})
        .join(panel.psam_df, on="iid", how="inner")
        .with_columns(pl.lit(pgs_id).alias("pgs_id"))
    )


def _score_with_plink2(
    pgs_id: str,
    scoring_file: Path,
    ref_dir: Path,
    plink2_path: Path,
) -> pl.DataFrame:
    """Score using the PLINK2 binary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        return compute_reference_prs_plink2(
            pgs_id=pgs_id,
            scoring_file=scoring_file,
            ref_dir=ref_dir,
            out_dir=Path(tmpdir),
            plink2_bin=plink2_path,
            genome_build="GRCh38",
        )


@pytest.mark.skipif(not REF_PANEL_AVAILABLE, reason="1000G reference panel not available")
class TestCrossEngineScores:
    """Verify that all three scoring engines produce identical results."""

    @pytest.mark.parametrize("pgs_id", PGS_IDS)
    def test_duckdb_matches_polars(
        self,
        pgs_id: str,
        ref_dir: Path,
        resolved_panel: _ResolvedRefPanel,
        scoring_cache: Path,
    ) -> None:
        """DuckDB-based matching must produce the same per-sample scores as polars."""
        scoring_file = download_scoring_file(pgs_id, scoring_cache, genome_build="GRCh38")

        duckdb_df = _score_with_duckdb(pgs_id, scoring_file, ref_dir, resolved_panel)
        polars_df = _score_with_polars_match(pgs_id, scoring_file, resolved_panel)

        duckdb_scores = duckdb_df.sort("iid").select("iid", "score")
        polars_scores = polars_df.sort("iid").select("iid", "score")

        assert duckdb_scores["iid"].to_list() == polars_scores["iid"].to_list(), (
            f"{pgs_id}: sample IDs differ between DuckDB and polars"
        )

        duckdb_vals = duckdb_scores["score"].to_list()
        polars_vals = polars_scores["score"].to_list()

        max_diff = max(abs(d - p) for d, p in zip(duckdb_vals, polars_vals))
        assert max_diff < 1e-6, (
            f"{pgs_id}: DuckDB vs polars max score diff = {max_diff:.2e} (should be < 1e-6)"
        )

    @pytest.mark.parametrize("pgs_id", PGS_IDS)
    def test_duckdb_matches_plink2(
        self,
        pgs_id: str,
        ref_dir: Path,
        resolved_panel: _ResolvedRefPanel,
        plink2_path: Path,
        scoring_cache: Path,
    ) -> None:
        """DuckDB engine and PLINK2 binary must produce correlated per-sample scores.

        PLINK2 uses SUM scoring while our engine uses AVG (sum / 2*n_variants),
        and PLINK2 may match a different number of variants due to its own
        ID-based matching. So we compare rank correlation rather than exact values.
        """
        scoring_file = download_scoring_file(pgs_id, scoring_cache, genome_build="GRCh38")

        duckdb_df = _score_with_duckdb(pgs_id, scoring_file, ref_dir, resolved_panel)
        plink2_df = _score_with_plink2(pgs_id, scoring_file, ref_dir, plink2_path)

        merged = duckdb_df.select("iid", pl.col("score").alias("duckdb")).join(
            plink2_df.select("iid", pl.col("score").alias("plink2")),
            on="iid",
            how="inner",
        )

        assert merged.height > 0, f"{pgs_id}: no overlapping samples between DuckDB and PLINK2"
        assert merged.height == duckdb_df.height, (
            f"{pgs_id}: sample count mismatch DuckDB={duckdb_df.height} vs merged={merged.height}"
        )

        corr = merged.select(pl.corr("duckdb", "plink2")).item()
        assert corr is not None and corr > 0.999, (
            f"{pgs_id}: DuckDB vs PLINK2 Pearson r = {corr:.6f} (should be > 0.999)"
        )

    @pytest.mark.parametrize("pgs_id", PGS_IDS)
    def test_polars_matches_plink2(
        self,
        pgs_id: str,
        ref_dir: Path,
        resolved_panel: _ResolvedRefPanel,
        plink2_path: Path,
        scoring_cache: Path,
    ) -> None:
        """Old polars engine and PLINK2 binary must produce correlated per-sample scores."""
        scoring_file = download_scoring_file(pgs_id, scoring_cache, genome_build="GRCh38")

        polars_df = _score_with_polars_match(pgs_id, scoring_file, resolved_panel)
        plink2_df = _score_with_plink2(pgs_id, scoring_file, ref_dir, plink2_path)

        merged = polars_df.select("iid", pl.col("score").alias("polars")).join(
            plink2_df.select("iid", pl.col("score").alias("plink2")),
            on="iid",
            how="inner",
        )

        assert merged.height > 0, f"{pgs_id}: no overlapping samples"

        corr = merged.select(pl.corr("polars", "plink2")).item()
        assert corr is not None and corr > 0.999, (
            f"{pgs_id}: polars vs PLINK2 Pearson r = {corr:.6f} (should be > 0.999)"
        )

    def test_summary_table(
        self,
        ref_dir: Path,
        resolved_panel: _ResolvedRefPanel,
        plink2_path: Path,
        scoring_cache: Path,
    ) -> None:
        """Print a comparison summary table across all engines and PGS IDs."""
        rows: list[dict[str, object]] = []

        for pgs_id in PGS_IDS:
            scoring_file = download_scoring_file(pgs_id, scoring_cache, genome_build="GRCh38")

            duckdb_df = _score_with_duckdb(pgs_id, scoring_file, ref_dir, resolved_panel)
            polars_df = _score_with_polars_match(pgs_id, scoring_file, resolved_panel)
            plink2_df = _score_with_plink2(pgs_id, scoring_file, ref_dir, plink2_path)

            merged = (
                duckdb_df.select("iid", pl.col("score").alias("duckdb"))
                .join(polars_df.select("iid", pl.col("score").alias("polars")), on="iid")
                .join(plink2_df.select("iid", pl.col("score").alias("plink2")), on="iid")
            )

            duck_vs_polars_max = (
                merged.select((pl.col("duckdb") - pl.col("polars")).abs().max()).item()
            )
            duck_vs_plink2_corr = merged.select(pl.corr("duckdb", "plink2")).item()
            polars_vs_plink2_corr = merged.select(pl.corr("polars", "plink2")).item()

            rows.append({
                "pgs_id": pgs_id,
                "n_samples": merged.height,
                "duckdb_mean": round(merged["duckdb"].mean(), 8),
                "polars_mean": round(merged["polars"].mean(), 8),
                "plink2_mean": round(merged["plink2"].mean(), 8),
                "duck_vs_polars_max_diff": f"{duck_vs_polars_max:.2e}",
                "duck_vs_plink2_r": f"{duck_vs_plink2_corr:.6f}",
                "polars_vs_plink2_r": f"{polars_vs_plink2_corr:.6f}",
            })

        summary = pl.DataFrame(rows)
        print("\n\nCross-engine comparison summary:")
        print(summary)

        for row in rows:
            assert float(row["duck_vs_polars_max_diff"]) < 1e-6
            assert float(row["duck_vs_plink2_r"]) > 0.999
            assert float(row["polars_vs_plink2_r"]) > 0.999
