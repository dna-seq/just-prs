"""Tests for consumer-chip coverage of PGS scoring files.

Uses the real Illumina GSA manifest (downloaded + cached once) and real cached
scoring parquets. Skips gracefully when the required cached scoring files are
not present so a clean clone without a populated cache does not fail.
"""

from pathlib import Path

import polars as pl
import pytest

from just_prs.chip_coverage import (
    CHIPS_BY_ID,
    _scoring_positions_lf,
    chip_typed_positions,
    compute_chip_coverage,
    parse_gsa_manifest,
    download_chip_manifest,
)
from just_prs.scoring import resolve_cache_dir


@pytest.fixture(scope="module")
def cache_dir() -> Path:
    return resolve_cache_dir()


@pytest.fixture(scope="module")
def scores_dir(cache_dir: Path) -> Path:
    d = cache_dir / "scores"
    if not d.exists() or not any(d.glob("*_hmPOS_GRCh38.parquet")):
        pytest.skip("No cached GRCh38 scoring parquets; run the pipeline first.")
    return d


@pytest.fixture(scope="module")
def typed_positions(cache_dir: Path) -> pl.DataFrame:
    return chip_typed_positions("gsa_v3", cache_dir)


def test_manifest_parses_to_plausible_marker_count(cache_dir: Path) -> None:
    """The GSA v3 manifest should yield ~650K typed markers in GRCh38."""
    zip_path = download_chip_manifest("gsa_v3", cache_dir)
    df = parse_gsa_manifest(zip_path)
    # GSA v3 has ~654K loci; after dropping unmapped, expect 600K-660K.
    assert 600_000 <= df.height <= 660_000
    assert set(df.columns) == {"name", "chr_norm", "pos"}
    assert df["pos"].min() > 0
    # Autosomes + X/Y should all be represented.
    chroms = set(df["chr_norm"].to_list())
    assert {"1", "2", "22", "X"}.issubset(chroms)


def test_typed_positions_unique_and_cached(cache_dir: Path, typed_positions: pl.DataFrame) -> None:
    """Typed positions are unique (chr, pos) and survive a parquet round-trip."""
    assert typed_positions.height == typed_positions.unique(["chr_norm", "pos"]).height
    cache_path = cache_dir / "chip_manifests" / "gsa_v3_positions.parquet"
    assert cache_path.exists()
    reread = chip_typed_positions("gsa_v3", cache_dir)
    assert reread.height == typed_positions.height


def test_coverage_ratio_bounded_and_consistent(scores_dir: Path, cache_dir: Path) -> None:
    """Every coverage row has n_typed <= n_total and ratio in [0, 1]."""
    sample_files = sorted(scores_dir.glob("*_hmPOS_GRCh38.parquet"))[:50]
    # Restrict to a small sample by temporarily pointing at a subset dir is overkill;
    # instead compute over all and assert invariants (fast enough for a sample check
    # via the public function on the full set would be slow, so verify per-file here).
    typed = chip_typed_positions("gsa_v3", cache_dir).with_columns(pl.lit(True).alias("typed"))
    for parquet_path in sample_files:
        scored = _scoring_positions_lf(parquet_path)
        agg = (
            scored.join(typed.lazy(), on=["chr_norm", "pos"], how="left")
            .select(
                pl.len().alias("n_total"),
                pl.col("typed").fill_null(False).sum().alias("n_typed"),
            )
            .collect()
        )
        n_total = int(agg["n_total"][0])
        n_typed = int(agg["n_typed"][0])
        assert 0 <= n_typed <= n_total
        ratio = (n_typed / n_total) if n_total else 0.0
        assert 0.0 <= ratio <= 1.0


def test_sparse_score_has_higher_coverage_than_dense(scores_dir: Path, cache_dir: Path) -> None:
    """A sparse clumped score covers a larger fraction of its variants than a dense
    genome-wide score, because consumer arrays type specific tag SNPs, not all variants.

    PGS000001 (~80 variants) vs PGS000013 (~6.6M variants) is a stable contrast.
    """
    sparse_p = scores_dir / "PGS000001_hmPOS_GRCh38.parquet"
    dense_p = scores_dir / "PGS000013_hmPOS_GRCh38.parquet"
    if not sparse_p.exists() or not dense_p.exists():
        pytest.skip("Required PGS000001/PGS000013 scoring parquets not cached.")

    typed = chip_typed_positions("gsa_v3", cache_dir).with_columns(pl.lit(True).alias("typed"))

    def _ratio(p: Path) -> float:
        agg = (
            _scoring_positions_lf(p)
            .join(typed.lazy(), on=["chr_norm", "pos"], how="left")
            .select(
                pl.len().alias("n_total"),
                pl.col("typed").fill_null(False).sum().alias("n_typed"),
            )
            .collect()
        )
        n_total = int(agg["n_total"][0])
        return (int(agg["n_typed"][0]) / n_total) if n_total else 0.0

    sparse_ratio = _ratio(sparse_p)
    dense_ratio = _ratio(dense_p)
    assert sparse_ratio > dense_ratio
    # Sparse clumped scores are largely tag SNPs → high direct coverage.
    assert sparse_ratio >= 0.80
    # Dense genome-wide scores need imputation → low direct coverage.
    assert dense_ratio < 0.30


def test_compute_chip_coverage_schema(scores_dir: Path, cache_dir: Path, tmp_path: Path) -> None:
    """compute_chip_coverage over a tiny scores dir returns the expected schema."""
    # Build a 2-file mini scores dir via symlinks to keep it fast.
    mini = tmp_path / "scores"
    mini.mkdir()
    picked = sorted(scores_dir.glob("*_hmPOS_GRCh38.parquet"))[:2]
    if len(picked) < 2:
        pytest.skip("Need at least 2 cached scoring parquets.")
    for p in picked:
        (mini / p.name).symlink_to(p)

    df = compute_chip_coverage(scores_dir=mini, cache_dir=cache_dir)
    assert df.height == 2  # 2 scores x 1 chip
    expected_cols = {
        "pgs_id", "chip", "platform", "consumer_products", "build",
        "n_typed", "n_total", "coverage_ratio",
    }
    assert expected_cols == set(df.columns)
    assert df["chip"].unique().to_list() == ["gsa_v3"]
    assert df["build"].unique().to_list() == [CHIPS_BY_ID["gsa_v3"]["build"]]
