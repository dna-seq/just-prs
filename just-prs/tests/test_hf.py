"""Tests for HuggingFace metadata/link helpers."""

import polars as pl

from just_prs.hf import _scores_with_parquet_links


def test_scores_with_parquet_links_rewrites_ftp_link_to_hf_parquet() -> None:
    """Combined-repo metadata should expose parquet-first scoring links."""
    repo_id = "just-dna-seq/pgs-catalog"
    scores_df = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "genome_build": ["GRCh38"],
            "ftp_link": ["https://ftp.ebi.ac.uk/example/PGS000001_hmPOS_GRCh38.txt.gz"],
        }
    )

    out = _scores_with_parquet_links(scores_df, repo_id=repo_id)
    row = out.row(0, named=True)

    assert row["ftp_link_ebi"] == "https://ftp.ebi.ac.uk/example/PGS000001_hmPOS_GRCh38.txt.gz"
    assert row["scoring_parquet_filename"] == "PGS000001_hmPOS_GRCh38.parquet"
    assert row["scoring_parquet_path"] == "data/scores/PGS000001_hmPOS_GRCh38.parquet"
    assert row["ftp_link"] == (
        "https://huggingface.co/datasets/just-dna-seq/pgs-catalog/resolve/main/"
        "data/scores/PGS000001_hmPOS_GRCh38.parquet"
    )


def test_scores_with_parquet_links_handles_non_harmonized_builds() -> None:
    """Rows with non-harmonized builds should keep null parquet references."""
    repo_id = "just-dna-seq/pgs-catalog"
    scores_df = pl.DataFrame(
        {
            "pgs_id": ["PGS000999"],
            "genome_build": ["NR"],
        }
    )

    out = _scores_with_parquet_links(scores_df, repo_id=repo_id)
    row = out.row(0, named=True)

    assert row["scoring_parquet_filename"] is None
    assert row["scoring_parquet_path"] is None
    assert row["ftp_link"] is None
