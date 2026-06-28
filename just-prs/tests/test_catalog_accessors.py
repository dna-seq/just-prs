"""Offline tests for PRSCatalog accessors added for MCP/prs-ui (F21, F27).

Both use synthetic local parquets so no network/HF access is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from just_prs import PRSCatalog


def _write_cleaned_scores(cache_dir: Path) -> None:
    """Minimal cleaned metadata so PRSCatalog loads locally (no HF/FTP)."""
    meta = cache_dir / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "pgs_id": ["PGS000001", "PGS000002"],
            "genome_build": ["GRCh37", "GRCh38"],
            "trait_reported": ["type 2 diabetes", "height"],
        }
    ).write_parquet(meta / "scores.parquet")
    for f in ("performance.parquet", "best_performance.parquet"):
        pl.DataFrame({"pgs_id": ["PGS000001"]}).write_parquet(meta / f)
    # Development-ancestry parquet so the F19/F21 surface loads offline (no FTP rebuild).
    pl.DataFrame(
        {
            "pgs_id": ["PGS000001", "PGS000002"],
            "dev_ancestry_broad": ["European", "European"],
            "dev_ancestries": [["European"], ["East Asian", "European"]],
            "dev_n_ancestries": [1, 2],
            "dev_is_multi_ancestry": [False, True],
            "dev_ancestry_distribution": [
                '{"European": 1.0}',
                '{"European": 0.7, "East Asian": 0.3}',
            ],
            "dev_sample_size": [22627, 90000],
            "dev_gwas_sample_size": [22627, 90000],
            "dev_training_sample_size": [0, 15000],
            "gwas_ancestry_broad": ["European", "European"],
            "training_ancestry_broad": [None, "East Asian"],
        }
    ).write_parquet(meta / "score_development_ancestry.parquet")


# --- F21: is_harmonized on score_info_row -------------------------------------


def test_score_info_row_is_harmonized_build_relative(tmp_path: Path) -> None:
    _write_cleaned_scores(tmp_path)
    catalog = PRSCatalog(cache_dir=tmp_path)

    # No target build -> is_harmonized is not guessed.
    plain = catalog.score_info_row("PGS000001")
    assert plain is not None
    assert "is_harmonized" not in plain

    # GRCh37-native score scored against GRCh38 -> harmonized.
    harm = catalog.score_info_row("PGS000001", genome_build="GRCh38")
    assert harm["is_harmonized"] is True

    # GRCh38-native score against GRCh38 -> native, not harmonized.
    native = catalog.score_info_row("PGS000002", genome_build="GRCh38")
    assert native["is_harmonized"] is False

    assert catalog.score_info_row("PGS999999", genome_build="GRCh38") is None


# --- F27: reference_individual_scores -----------------------------------------


def _write_reference_scores(cache_dir: Path, pgs_id: str = "PGS000001") -> None:
    d = cache_dir / "reference_scores" / "1000g" / pgs_id
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "iid": ["s1", "s2", "s3", "s4"],
            "score": [0.1, 0.2, 0.3, 0.4],
            "superpop": ["EUR", "EUR", "AFR", "EAS"],
            "population": ["GBR", "FIN", "YRI", "JPT"],
            "pgs_id": [pgs_id] * 4,
        }
    ).write_parquet(d / "scores.parquet")


def test_reference_individual_scores_returns_per_individual(tmp_path: Path) -> None:
    _write_reference_scores(tmp_path)
    catalog = PRSCatalog(cache_dir=tmp_path)

    df = catalog.reference_individual_scores("PGS000001").collect()
    assert set(df.columns) == {"iid", "superpopulation", "prs_score"}
    assert df.height == 4
    assert sorted(df["prs_score"].to_list()) == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_reference_individual_scores_superpop_filter_and_case(tmp_path: Path) -> None:
    _write_reference_scores(tmp_path)
    catalog = PRSCatalog(cache_dir=tmp_path)

    eur = catalog.reference_individual_scores("PGS000001", superpopulation="EUR").collect()
    assert eur.height == 2
    assert set(eur["superpopulation"].to_list()) == {"EUR"}

    # pgs_id is upper-cased before path lookup.
    assert catalog.reference_individual_scores("pgs000001").collect().height == 4


def test_reference_individual_scores_missing_raises(tmp_path: Path) -> None:
    _write_reference_scores(tmp_path)
    catalog = PRSCatalog(cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="per-individual reference scores"):
        catalog.reference_individual_scores("PGS999999")


# --- F19/F21: development-ancestry surfacing ----------------------------------


def test_development_ancestry_accessor(tmp_path: Path) -> None:
    _write_cleaned_scores(tmp_path)
    catalog = PRSCatalog(cache_dir=tmp_path)

    all_dev = catalog.development_ancestry().collect()
    assert all_dev.height == 2
    assert "dev_ancestry_distribution" in all_dev.columns

    # pgs_id filter is upper-cased.
    one = catalog.development_ancestry("pgs000002").collect()
    assert one.height == 1
    assert one.row(0, named=True)["dev_is_multi_ancestry"] is True


def test_scores_join_carries_lean_dev_columns(tmp_path: Path) -> None:
    _write_cleaned_scores(tmp_path)
    catalog = PRSCatalog(cache_dir=tmp_path)

    s = catalog.scores().collect()
    for col in ("dev_ancestry_broad", "dev_sample_size", "dev_is_multi_ancestry"):
        assert col in s.columns
    # The full distribution string stays out of the wide scores grid.
    assert "dev_ancestry_distribution" not in s.columns
    row = s.filter(pl.col("pgs_id") == "PGS000001").row(0, named=True)
    assert row["dev_ancestry_broad"] == "European"
    assert row["dev_sample_size"] == 22627


def test_score_info_row_enriched_with_dev_ancestry(tmp_path: Path) -> None:
    _write_cleaned_scores(tmp_path)
    catalog = PRSCatalog(cache_dir=tmp_path)

    row = catalog.score_info_row("PGS000002")
    assert row is not None
    assert row["dev_ancestry_broad"] == "European"
    assert row["dev_is_multi_ancestry"] is True
    assert json.loads(row["dev_ancestry_distribution"]) == {"European": 0.7, "East Asian": 0.3}
    assert row["training_ancestry_broad"] == "East Asian"
