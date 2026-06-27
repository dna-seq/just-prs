"""Tests for PGS scoring file download and parsing."""

import gzip
from pathlib import Path

import polars as pl

from just_prs.scoring import download_scoring_file, load_scoring, parse_scoring_file


def test_parse_scoring_file_scientific_notation_positions(tmp_path: Path) -> None:
    """Regression: some harmonized files serialize integer position columns in
    scientific notation (e.g. ``7.2e+07``). A hard i64 parse rejects them, which
    silently drops the score from the reference-allele universe. The parser must
    read those columns as float and cast back to Int64. (Hit by PGS004941 /
    PGS005253 GRCh37, where 7.2e+07 / 7.7e+07 appear in chr_position.)"""
    content = (
        "#pgs_id=PGSTEST\n"
        "#genome_build=GRCh37\n"
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\thm_chr\thm_pos\n"
        "rs1\t1\t72000000\tA\tG\t0.1\t1\t72000000\n"
        "rs2\t1\t7.2e+07\tA\tG\t0.2\t1\t7.2e+07\n"
        "rs3\t2\t7.7e+07\tC\tT\t0.3\t2\t7.7e+07\n"
    )
    gz = tmp_path / "PGSTEST_hmPOS_GRCh37.txt.gz"
    with gzip.open(gz, "wt") as f:
        f.write(content)

    df = parse_scoring_file(gz).collect()

    assert df["chr_position"].dtype == pl.Int64
    assert df["hm_pos"].dtype == pl.Int64
    assert df["chr_position"].to_list() == [72_000_000, 72_000_000, 77_000_000]
    assert df["hm_pos"].to_list() == [72_000_000, 72_000_000, 77_000_000]


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
