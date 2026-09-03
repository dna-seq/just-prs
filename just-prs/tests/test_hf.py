"""Tests for HuggingFace metadata/link helpers."""

from pathlib import Path

import polars as pl
import pytest

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


# ---------------------------------------------------------------------------
# Flat HF pulls: zero copies, no nested data/ duplicate, no poisoned cache
# ---------------------------------------------------------------------------


def _nested_parquet_downloader(rows: int = 1):
    """Stand-in for hf_hub_download: lands the file nested, as HF really does.

    ``hf_hub_download(local_dir=X)`` replicates the repo layout under X and
    offers no flat option, which is why a relocation step exists at all.
    """
    def _dl(
        repo_id,
        filename,
        repo_type="dataset",
        local_dir=None,
        token=None,
        revision=None,
    ):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"pgs_id": ["PGS000001"] * rows}).write_parquet(dest)
        return str(dest)

    return _dl


def test_pull_flat_moves_and_prunes_leaving_one_copy(tmp_path, monkeypatch):
    """The artifact must land flat with the nested mirror gone — zero copies."""
    import just_prs.hf as hf_mod
    from just_prs.hf import _pull_flat

    local = tmp_path / "reference"
    local.mkdir()
    monkeypatch.setattr(hf_mod, "_hf_download_with_retry", _nested_parquet_downloader())

    target = _pull_flat("repo/id", "data/reference/universe.parquet", local, None)

    assert target == local / "universe.parquet"
    assert target.exists()
    assert not (local / "data").exists(), "nested HF mirror must be pruned"
    on_disk = [p for p in local.rglob("*") if p.is_file()]
    assert on_disk == [target], f"expected exactly one copy, found {on_disk}"


def test_pull_flat_refuses_to_publish_a_corrupt_download(tmp_path, monkeypatch):
    """Validation happens before the artifact takes the name others trust."""
    import just_prs.hf as hf_mod
    from just_prs.hf import _pull_flat

    local = tmp_path / "reference"
    local.mkdir()

    def _bad(
        repo_id,
        filename,
        repo_type="dataset",
        local_dir=None,
        token=None,
        revision=None,
    ):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PAR1truncated")
        return str(dest)

    monkeypatch.setattr(hf_mod, "_hf_download_with_retry", _bad)

    with pytest.raises(RuntimeError, match="failed validation"):
        _pull_flat("repo/id", "data/reference/universe.parquet", local, None)

    assert not (local / "universe.parquet").exists()
    assert not (local / "data" / "reference" / "universe.parquet").exists(), (
        "a rejected download must be removed so HF re-fetches instead of "
        "serving a bad cache hit"
    )


def test_needs_pull_distinguishes_missing_corrupt_and_good(tmp_path):
    """Existence is not enough: a corrupt artifact must be re-fetched."""
    from just_prs.hf import needs_pull

    missing = tmp_path / "absent.parquet"
    assert needs_pull(missing) is True

    good = tmp_path / "good.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(good)
    assert needs_pull(good) is False
    assert good.exists()

    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"PAR1junk")
    assert needs_pull(corrupt) is True
    assert not corrupt.exists(), "an unreadable artifact must be unlinked so it self-heals"


def test_needs_pull_validates_json_sidecars(tmp_path):
    """Audit summaries are JSON, not parquet — they get validated too."""
    from just_prs.hf import needs_pull

    good = tmp_path / "summary.json"
    good.write_text('{"panel": "1000g"}')
    assert needs_pull(good) is False

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert needs_pull(bad) is True
    assert not bad.exists()
