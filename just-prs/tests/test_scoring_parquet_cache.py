"""Tests for PGS scoring file parquet cache: roundtrip, schema, header, skip-download, PRS equivalence."""

import shutil
from pathlib import Path

import polars as pl
import pytest

from just_prs.ftp import download_scoring_as_parquet
from just_prs.scoring import (
    SCORING_FILE_SCHEMA,
    _scoring_parquet_cache_path,
    download_scoring_file,
    load_scoring,
    parse_scoring_file,
    read_scoring_header,
    scoring_parquet_path,
)

TEST_PGS_IDS = ["PGS000001", "PGS000002", "PGS000010", "PGS000013"]


@pytest.fixture(scope="module")
def parquet_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temp directory for parquet cache tests."""
    return tmp_path_factory.mktemp("parquet_cache_tests")


@pytest.fixture(scope="module")
def gz_and_parquet_dfs(
    scoring_cache_dir: Path,
    parquet_cache_dir: Path,
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame, Path, Path]]:
    """For each test PGS ID, download the .txt.gz, parse it (creating the
    parquet cache), and return (gz_df, parquet_df, gz_path, parquet_path).

    Uses a dedicated copy dir so we don't pollute the shared scoring cache
    with parquet files from other tests.
    """
    results: dict[str, tuple[pl.DataFrame, pl.DataFrame, Path, Path]] = {}

    for pgs_id in TEST_PGS_IDS:
        src_gz = download_scoring_file(pgs_id, scoring_cache_dir, genome_build="GRCh38")

        copy_dir = parquet_cache_dir / pgs_id
        copy_dir.mkdir(parents=True, exist_ok=True)
        gz_path = copy_dir / src_gz.name
        if not gz_path.exists():
            shutil.copy2(src_gz, gz_path)

        parquet_path = _scoring_parquet_cache_path(gz_path)
        if parquet_path.exists():
            parquet_path.unlink()

        from just_prs.scoring import _parse_gz_scoring_file
        gz_df, _ = _parse_gz_scoring_file(gz_path)

        _ = parse_scoring_file(gz_path)
        assert parquet_path.exists(), f"Parquet cache not created for {pgs_id}"

        parquet_df = pl.read_parquet(parquet_path)
        results[pgs_id] = (gz_df, parquet_df, gz_path, parquet_path)

    return results


@pytest.mark.parametrize("pgs_id", TEST_PGS_IDS)
def test_parquet_roundtrip_schema(
    pgs_id: str,
    gz_and_parquet_dfs: dict[str, tuple[pl.DataFrame, pl.DataFrame, Path, Path]],
) -> None:
    """Parquet cache has identical columns, dtypes, and row count as the .txt.gz parse."""
    gz_df, parquet_df, _, _ = gz_and_parquet_dfs[pgs_id]

    assert gz_df.columns == parquet_df.columns, (
        f"Column mismatch for {pgs_id}: {gz_df.columns} vs {parquet_df.columns}"
    )
    assert gz_df.height == parquet_df.height, (
        f"Row count mismatch for {pgs_id}: {gz_df.height} vs {parquet_df.height}"
    )

    for col in gz_df.columns:
        assert gz_df[col].dtype == parquet_df[col].dtype, (
            f"Dtype mismatch for {pgs_id}.{col}: {gz_df[col].dtype} vs {parquet_df[col].dtype}"
        )

    if "chr_name" in parquet_df.columns:
        assert parquet_df["chr_name"].dtype == pl.Utf8
    if "hm_chr" in parquet_df.columns:
        assert parquet_df["hm_chr"].dtype == pl.Utf8
    if "effect_weight" in parquet_df.columns:
        assert parquet_df["effect_weight"].dtype == pl.Float64
    if "hm_pos" in parquet_df.columns:
        assert parquet_df["hm_pos"].dtype == pl.Int64
    if "effect_allele" in parquet_df.columns:
        assert parquet_df["effect_allele"].dtype == pl.Utf8
    if "rsID" in parquet_df.columns:
        assert parquet_df["rsID"].dtype == pl.Utf8


@pytest.mark.parametrize("pgs_id", TEST_PGS_IDS)
def test_parquet_roundtrip_values(
    pgs_id: str,
    gz_and_parquet_dfs: dict[str, tuple[pl.DataFrame, pl.DataFrame, Path, Path]],
) -> None:
    """Parquet cache has identical values as the .txt.gz parse (no data loss)."""
    gz_df, parquet_df, _, _ = gz_and_parquet_dfs[pgs_id]

    assert gz_df.equals(parquet_df), (
        f"DataFrame values differ for {pgs_id} between .txt.gz parse and parquet cache"
    )


@pytest.mark.parametrize("pgs_id", TEST_PGS_IDS)
def test_header_metadata_preserved(
    pgs_id: str,
    gz_and_parquet_dfs: dict[str, tuple[pl.DataFrame, pl.DataFrame, Path, Path]],
) -> None:
    """PGS Catalog header metadata is preserved in parquet file-level metadata."""
    _, _, _, parquet_path = gz_and_parquet_dfs[pgs_id]

    header = read_scoring_header(parquet_path)
    assert header, f"Empty header metadata for {pgs_id}"
    assert "pgs_id" in header, f"Missing pgs_id in header for {pgs_id}"
    assert header["pgs_id"] == pgs_id

    assert "weight_type" in header or "genome_build" in header, (
        f"Missing expected metadata keys for {pgs_id}: {list(header.keys())}"
    )

    if pgs_id == "PGS000001":
        assert header.get("trait_reported", "").casefold() == "breast cancer"


@pytest.mark.parametrize("pgs_id", TEST_PGS_IDS[:2])
def test_parse_accepts_parquet_path(
    pgs_id: str,
    gz_and_parquet_dfs: dict[str, tuple[pl.DataFrame, pl.DataFrame, Path, Path]],
) -> None:
    """parse_scoring_file() works when given a .parquet path directly."""
    _, _, _, parquet_path = gz_and_parquet_dfs[pgs_id]

    lf = parse_scoring_file(parquet_path)
    df = lf.collect()
    assert df.height > 0
    assert "effect_weight" in df.columns


def test_parse_cache_hit(
    gz_and_parquet_dfs: dict[str, tuple[pl.DataFrame, pl.DataFrame, Path, Path]],
) -> None:
    """Second call to parse_scoring_file() hits the parquet cache."""
    _, _, gz_path, parquet_path = gz_and_parquet_dfs["PGS000001"]

    assert parquet_path.exists()
    lf = parse_scoring_file(gz_path)
    df = lf.collect()
    assert df.height == 77


def test_load_scoring_skips_download_when_parquet_exists(
    parquet_cache_dir: Path,
    gz_and_parquet_dfs: dict[str, tuple[pl.DataFrame, pl.DataFrame, Path, Path]],
) -> None:
    """load_scoring() returns data from parquet cache even when .txt.gz is absent."""
    pgs_id = "PGS000001"
    _, _, gz_path, parquet_path = gz_and_parquet_dfs[pgs_id]

    isolated_dir = parquet_cache_dir / "isolated_load_test"
    isolated_dir.mkdir(parents=True, exist_ok=True)
    dest_parquet = scoring_parquet_path(pgs_id, isolated_dir, "GRCh38")
    shutil.copy2(parquet_path, dest_parquet)

    gz_in_isolated = isolated_dir / gz_path.name
    if gz_in_isolated.exists():
        gz_in_isolated.unlink()

    lf = load_scoring(pgs_id, cache_dir=isolated_dir, genome_build="GRCh38")
    df = lf.collect()
    assert df.height == 77
    assert "effect_weight" in df.columns

    assert not gz_in_isolated.exists(), (
        "load_scoring() should not have downloaded .txt.gz when parquet exists"
    )


def test_prs_equivalence_gz_vs_parquet(
    scoring_cache_dir: Path,
    vcf_path: Path,
    parquet_cache_dir: Path,
    gz_and_parquet_dfs: dict[str, tuple[pl.DataFrame, pl.DataFrame, Path, Path]],
) -> None:
    """PRS computed from .txt.gz and parquet cache are identical."""
    from just_prs.prs import compute_prs

    pgs_id = "PGS000001"
    _, _, gz_path, parquet_path = gz_and_parquet_dfs[pgs_id]

    result_gz = compute_prs(
        vcf_path=vcf_path,
        scoring_file=gz_path,
        genome_build="GRCh38",
        pgs_id=pgs_id,
    )

    result_pq = compute_prs(
        vcf_path=vcf_path,
        scoring_file=parquet_path,
        genome_build="GRCh38",
        pgs_id=pgs_id,
    )

    assert result_gz.score == result_pq.score, (
        f"PRS scores differ: gz={result_gz.score}, parquet={result_pq.score}"
    )
    assert result_gz.variants_matched == result_pq.variants_matched
    assert result_gz.variants_total == result_pq.variants_total
    assert result_gz.match_rate == result_pq.match_rate


def test_scoring_parquet_path_computation() -> None:
    """scoring_parquet_path() computes the expected path without filesystem access."""
    cache_dir = Path("/tmp/test-cache")
    path = scoring_parquet_path("PGS000042", cache_dir, "GRCh38")
    assert path == cache_dir / "PGS000042_hmPOS_GRCh38.parquet"

    path37 = scoring_parquet_path("PGS000042", cache_dir, "GRCh37")
    assert path37 == cache_dir / "PGS000042_hmPOS_GRCh37.parquet"


def test_scoring_parquet_cache_path_helper() -> None:
    """_scoring_parquet_cache_path() returns correct sibling path."""
    gz = Path("/data/scores/PGS000001_hmPOS_GRCh38.txt.gz")
    pq = _scoring_parquet_cache_path(gz)
    assert pq == Path("/data/scores/PGS000001_hmPOS_GRCh38.parquet")


def test_download_scoring_as_parquet_replaces_corrupt_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing unreadable parquet caches are deleted and regenerated."""
    pgs_id = "PGS999999"
    corrupt_path = tmp_path / f"{pgs_id}.parquet"
    corrupt_path.write_bytes(b"not a parquet")

    replacement = pl.DataFrame({
        "chr_name": ["1"],
        "chr_position": [123],
        "effect_allele": ["A"],
        "effect_weight": [0.5],
    })

    def fake_stream_scoring_file(pgs_id: str, genome_build: str = "GRCh38") -> pl.LazyFrame:
        return replacement.lazy()

    monkeypatch.setattr("just_prs.ftp.stream_scoring_file", fake_stream_scoring_file)

    path = download_scoring_as_parquet(pgs_id, tmp_path)
    assert path == corrupt_path

    df = pl.read_parquet(path)
    assert df.select("pgs_id", "chr_name", "effect_weight").to_dicts() == [
        {"pgs_id": pgs_id, "chr_name": "1", "effect_weight": 0.5}
    ]


# ---------------------------------------------------------------------------
# Parquet-only managed-cache invariant (ensure_scoring_file)
# ---------------------------------------------------------------------------


def _fake_gz(cache_dir: Path, pgs_id: str, build: str = "GRCh38") -> Path:
    """Write a minimal but real PGS scoring .txt.gz into a managed cache dir."""
    import gzip

    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / f"{pgs_id}_hmPOS_{build}.txt.gz"
    body = (
        f"###PGS CATALOG SCORING FILE\n"
        f"#pgs_id={pgs_id}\n"
        f"#genome_build={build}\n"
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n"
        "rs1\t1\t1000\tA\tG\t0.1\n"
        "rs2\t2\t2000\tC\tT\t-0.2\n"
    )
    with gzip.open(gz_path, "wt") as fh:
        fh.write(body)
    return gz_path


def test_ensure_scoring_file_discards_gz_after_verified_conversion(tmp_path: Path) -> None:
    """The managed cache must end up holding parquet only."""
    from just_prs.scoring import ensure_scoring_file

    scores = tmp_path / "scores"
    gz_path = _fake_gz(scores, "PGS000001")

    result = ensure_scoring_file("PGS000001", scores, "GRCh38")

    assert result.suffix == ".parquet"
    assert result.exists()
    assert not gz_path.exists(), "the .txt.gz must not survive a verified conversion"
    assert {p.name for p in scores.iterdir()} == {"PGS000001_hmPOS_GRCh38.parquet"}
    assert pl.read_parquet(result).height == 2


def test_ensure_scoring_file_parquet_hit_makes_no_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached parquet must short-circuit before any download is attempted.

    This is the regression that made six CLI paths re-fetch a .txt.gz they
    already had as parquet: download_scoring_file only ever checked for the gz.
    """
    import just_prs.scoring as scoring_mod
    from just_prs.scoring import ensure_scoring_file

    scores = tmp_path / "scores"
    _fake_gz(scores, "PGS000001")
    first = ensure_scoring_file("PGS000001", scores, "GRCh38")

    def _explode(*args: object, **kwargs: object) -> Path:
        raise AssertionError("download attempted despite a readable parquet cache")

    monkeypatch.setattr(scoring_mod, "download_scoring_file", _explode)
    again = ensure_scoring_file("PGS000001", scores, "GRCh38")

    assert again == first


def test_ensure_scoring_file_rebuilds_corrupt_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated parquet must be discarded and re-fetched, not trusted."""
    import just_prs.scoring as scoring_mod
    from just_prs.scoring import ensure_scoring_file

    scores = tmp_path / "scores"
    corrupt = scores / "PGS000001_hmPOS_GRCh38.parquet"
    scores.mkdir(parents=True)
    corrupt.write_bytes(b"PAR1-not-really")

    monkeypatch.setattr(
        scoring_mod,
        "download_scoring_file",
        lambda pgs_id, out_dir, build: _fake_gz(out_dir, pgs_id, build),
    )
    result = ensure_scoring_file("PGS000001", scores, "GRCh38")

    assert pl.read_parquet(result).height == 2
    assert not (scores / "PGS000001_hmPOS_GRCh38.txt.gz").exists()


def test_parse_scoring_file_never_deletes_a_user_supplied_file(tmp_path: Path) -> None:
    """Safety regression: only the managed cache owns the gz lifecycle.

    ``prs compute --scoring-file my.txt.gz`` routes a user's own file straight
    into ``parse_scoring_file``; deleting it there would destroy user data.
    """
    user_dir = tmp_path / "my-files"
    gz_path = _fake_gz(user_dir, "PGS000001")

    lf = parse_scoring_file(gz_path)

    assert lf.collect().height == 2
    assert gz_path.exists(), "parse_scoring_file must never delete its input"
