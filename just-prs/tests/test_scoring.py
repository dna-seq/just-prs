"""Tests for PGS scoring file download and parsing."""

from pathlib import Path

import polars as pl

from just_prs.scoring import download_scoring_file, load_scoring, parse_scoring_file


def test_download_scoring_file_pgs000001(scoring_cache_dir: Path) -> None:
    """Download PGS000001 harmonized GRCh38 scoring file."""
    path = download_scoring_file("PGS000001", scoring_cache_dir, genome_build="GRCh38")

    assert path.exists()
    assert path.suffix == ".gz"
    assert "PGS000001" in path.name
    assert path.stat().st_size > 0


def test_parse_scoring_file_pgs000001(scoring_cache_dir: Path) -> None:
    """Parse PGS000001 scoring file and verify structure."""
    path = download_scoring_file("PGS000001", scoring_cache_dir, genome_build="GRCh38")
    lf = parse_scoring_file(path)
    df = lf.collect()

    assert "effect_allele" in df.columns
    assert "effect_weight" in df.columns
    assert df["effect_weight"].dtype == pl.Float64

    has_position = ("hm_chr" in df.columns and "hm_pos" in df.columns) or (
        "chr_name" in df.columns and "chr_position" in df.columns
    )
    assert has_position, f"Missing position columns. Found: {df.columns}"

    assert len(df) == 77


def test_load_scoring_caches(scoring_cache_dir: Path) -> None:
    """Verify load_scoring downloads and caches the file."""
    lf = load_scoring("PGS000001", cache_dir=scoring_cache_dir, genome_build="GRCh38")
    df = lf.collect()

    assert len(df) == 77
    assert "effect_weight" in df.columns

    expected_file = scoring_cache_dir / "PGS000001_hmPOS_GRCh38.txt.gz"
    assert expected_file.exists()
