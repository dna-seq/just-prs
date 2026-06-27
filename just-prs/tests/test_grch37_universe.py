"""GRCh37 reference-allele universe support.

Covers the build-parameterization that makes GRCh37 a first-class build for the
reference-allele universe lineage:

- ``hf.reference_allele_universe_filename`` (build-aware, GRCh38 unsuffixed).
- The GSA per-build ``manifests`` map (A2 = GRCh38, A1 = GRCh37) and the gate
  that now raises ``ValueError`` (not ``NotImplementedError``) only for a build
  the chip has no manifest for.
- ``array_scoring._resolve_array_restoration`` unlocking per build once a
  build-matched universe parquet exists, and degrading to a no-op otherwise.

All tests run without network. The real A1 (GRCh37) manifest download+parse is an
opt-in test (67 MB), gated behind ``JUST_PRS_DOWNLOAD_MANIFESTS=1`` so a clean
clone never pulls it.
"""

import contextlib
import os
from pathlib import Path

import polars as pl
import pytest

from just_prs.array_scoring import _resolve_array_restoration
from just_prs.chip_coverage import (
    CHIPS_BY_ID,
    Chip,
    chip_manifest_dir,
    chip_typed_positions,
    download_chip_manifest,
    parse_gsa_manifest,
)
from just_prs.hf import reference_allele_universe_filename


# ---------------------------------------------------------------------------
# Build-aware universe filename
# ---------------------------------------------------------------------------

def test_universe_filename_grch38_unsuffixed() -> None:
    """GRCh38 keeps the historical unsuffixed name (back-compat with the
    already-published HF artifact and existing caches)."""
    assert reference_allele_universe_filename("GRCh38") == "reference_allele_universe.parquet"
    # Default is GRCh38.
    assert reference_allele_universe_filename() == "reference_allele_universe.parquet"


def test_universe_filename_other_builds_suffixed() -> None:
    """Non-GRCh38 builds are ``_<build>``-suffixed so builds coexist on disk/HF."""
    assert reference_allele_universe_filename("GRCh37") == "reference_allele_universe_GRCh37.parquet"
    # The suffix is the verbatim build label.
    assert reference_allele_universe_filename("T2T") == "reference_allele_universe_T2T.parquet"


# ---------------------------------------------------------------------------
# GSA per-build manifest map
# ---------------------------------------------------------------------------

def test_gsa_v3_has_both_build_manifests() -> None:
    """GSA v3 now ships both the A2 (GRCh38) and A1 (GRCh37) CSV manifests."""
    manifests = CHIPS_BY_ID[Chip.GSA_V3]["manifests"]
    assert set(manifests) == {"GRCh38", "GRCh37"}
    # A2 = GRCh38, A1 = GRCh37 — the Illumina convention.
    assert "A2" in manifests["GRCh38"] and manifests["GRCh38"].endswith(".zip")
    assert "A1" in manifests["GRCh37"] and manifests["GRCh37"].endswith(".zip")
    for url in manifests.values():
        assert "GSA-24v3-0" in url


def test_chip_typed_positions_rejects_unknown_build_with_valueerror(tmp_path: Path) -> None:
    """A build the chip has no manifest for raises ``ValueError`` (not
    ``NotImplementedError``) *before* any download — GRCh37 is no longer gated."""
    with pytest.raises(ValueError, match="no manifest for build"):
        chip_typed_positions(Chip.GSA_V3, tmp_path, build="GRCh36")
    # GRCh37 is a supported build, so it does NOT raise the gate error.
    # (We don't call it here to avoid the 67 MB download — see the opt-in test.)
    assert "GRCh37" in CHIPS_BY_ID[Chip.GSA_V3]["manifests"]


def test_download_chip_manifest_rejects_unknown_build(tmp_path: Path) -> None:
    """``download_chip_manifest`` validates the build before fetching anything."""
    with pytest.raises(ValueError, match="no manifest for build"):
        download_chip_manifest(Chip.GSA_V3, tmp_path, build="hg18")


# ---------------------------------------------------------------------------
# array_scoring._resolve_array_restoration — the unlock
# ---------------------------------------------------------------------------

def _write_dummy_universe(cache_dir: Path, genome_build: str) -> Path:
    """Write a minimal universe parquet at the build-aware path (content unread
    by _resolve_array_restoration — it only checks existence)."""
    ref_dir = cache_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    path = ref_dir / reference_allele_universe_filename(genome_build)
    pl.DataFrame(
        {"genome_build": [genome_build], "chrom": ["1"], "pos": [1000], "ref": ["A"], "ref_source": ["panel"]}
    ).write_parquet(path)
    return path


def test_resolve_array_restoration_grch37_unlocks_with_universe(tmp_path: Path) -> None:
    """With a GRCh37 universe present, a GRCh37 array unlocks chip-scoped
    restoration: returns (Chip, build-matched universe path)."""
    path = _write_dummy_universe(tmp_path, "GRCh37")
    scope, universe_path = _resolve_array_restoration("gsa_v3", "GRCh37", tmp_path)
    assert scope == Chip.GSA_V3
    assert universe_path == path
    assert universe_path.name == "reference_allele_universe_GRCh37.parquet"


def test_resolve_array_restoration_grch38_uses_unsuffixed(tmp_path: Path) -> None:
    """GRCh38 resolves the historical unsuffixed universe filename."""
    path = _write_dummy_universe(tmp_path, "GRCh38")
    scope, universe_path = _resolve_array_restoration("gsa_v3", "GRCh38", tmp_path)
    assert scope == Chip.GSA_V3
    assert universe_path == path
    assert universe_path.name == "reference_allele_universe.parquet"


def test_resolve_array_restoration_noop_when_universe_absent(tmp_path: Path, monkeypatch) -> None:
    """No published universe for the build → no-op (False, None). The HF pull is
    stubbed to a miss so the test stays offline."""
    import just_prs.hf as hf_mod

    monkeypatch.setattr(hf_mod, "pull_reference_allele_universe", lambda *a, **k: None)
    scope, universe_path = _resolve_array_restoration("gsa_v3", "GRCh37", tmp_path)
    assert scope is False
    assert universe_path is None


def test_resolve_array_restoration_noop_for_unsupported_build(tmp_path: Path) -> None:
    """A build with no chip manifest degrades to a no-op without touching HF
    (even if a stray universe file were present)."""
    _write_dummy_universe(tmp_path, "GRCh36")
    scope, universe_path = _resolve_array_restoration("gsa_v3", "GRCh36", tmp_path)
    assert scope is False
    assert universe_path is None


def test_resolve_array_restoration_noop_for_unknown_chip(tmp_path: Path) -> None:
    """An unrecognized chip id degrades to a no-op."""
    scope, universe_path = _resolve_array_restoration("not_a_chip", "GRCh37", tmp_path)
    assert scope is False
    assert universe_path is None


# ---------------------------------------------------------------------------
# Opt-in: real A1 (GRCh37) manifest download + parse (67 MB)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("JUST_PRS_DOWNLOAD_MANIFESTS", "").strip() not in {"1", "true", "yes"},
    reason="Set JUST_PRS_DOWNLOAD_MANIFESTS=1 to download+parse the 67 MB GSA A1 manifest.",
)
def test_gsa_a1_grch37_manifest_parses() -> None:
    """The A1 (GRCh37) manifest parses to the same plausible marker count as A2,
    in GRCh37 coordinates, with the same CSV schema."""
    from just_prs.scoring import resolve_cache_dir

    cache_dir = resolve_cache_dir()
    zip_path = download_chip_manifest(Chip.GSA_V3, cache_dir, build="GRCh37")
    df = parse_gsa_manifest(zip_path)
    assert 600_000 <= df.height <= 660_000
    assert set(df.columns) == {"name", "chr_norm", "pos"}
    assert df["pos"].min() > 0
    assert {"1", "2", "22", "X"}.issubset(set(df["chr_norm"].to_list()))

    # The GRCh37 typed positions round-trip through the per-build parquet cache.
    positions = chip_typed_positions(Chip.GSA_V3, cache_dir, build="GRCh37")
    assert positions.height == positions.unique(["chr_norm", "pos"]).height
    assert (chip_manifest_dir(cache_dir) / "gsa_v3_GRCh37_positions.parquet").exists()


# ---------------------------------------------------------------------------
# Universe completeness gate (robustness #4) — _scoring_parquet_coverage
# ---------------------------------------------------------------------------

def test_scoring_parquet_coverage_counts_and_missing(tmp_path: Path) -> None:
    """Coverage is over the catalog id set; missing ids are reported; the glob is
    build-specific so a GRCh37 parquet never counts toward GRCh38 coverage."""
    from prs_pipeline.assets import _scoring_parquet_coverage

    scores = tmp_path / "scores"
    scores.mkdir()
    for pid in ("PGS000001", "PGS000002"):
        pl.DataFrame({"a": [1]}).write_parquet(scores / f"{pid}_hmPOS_GRCh38.parquet")
    catalog = ["PGS000001", "PGS000002", "PGS000003"]

    n_present, n_catalog, coverage, missing = _scoring_parquet_coverage(scores, "GRCh38", catalog)
    assert (n_present, n_catalog) == (2, 3)
    assert coverage == pytest.approx(2 / 3)
    assert missing == ["PGS000003"]

    # A GRCh37 parquet for the missing id does NOT close the GRCh38 gap.
    pl.DataFrame({"a": [1]}).write_parquet(scores / "PGS000003_hmPOS_GRCh37.parquet")
    _, _, coverage38, missing38 = _scoring_parquet_coverage(scores, "GRCh38", catalog)
    assert coverage38 == pytest.approx(2 / 3)
    assert missing38 == ["PGS000003"]
    # …but it counts for GRCh37.
    _, _, coverage37, missing37 = _scoring_parquet_coverage(scores, "GRCh37", ["PGS000003"])
    assert coverage37 == pytest.approx(1.0)
    assert missing37 == []


# ---------------------------------------------------------------------------
# Scoring-file download hardening — bounded retry + logged failure
# ---------------------------------------------------------------------------

class _FakeRemote:
    """Minimal fsspec file-like that yields its payload once then EOF."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._done = False

    def read(self, _n: int) -> bytes:
        if self._done:
            return b""
        self._done = True
        return self._data

    def __enter__(self) -> "_FakeRemote":
        return self

    def __exit__(self, *_a) -> bool:
        return False


def test_scoring_download_retries_then_succeeds(tmp_path: Path, monkeypatch) -> None:
    """A transient drop on the first attempts is retried; a later success wins."""
    import just_prs.ftp as ftp

    monkeypatch.setattr(ftp, "_SCORING_DOWNLOAD_ATTEMPTS", 3)
    monkeypatch.setattr(ftp.time, "sleep", lambda *_a: None)  # no real backoff in tests
    calls = {"n": 0}

    @contextlib.contextmanager
    def fake_open(_url, _mode):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("transient connection drop")
        yield _FakeRemote(b"weight\tdata\n")

    monkeypatch.setattr(ftp.fsspec, "open", fake_open)

    pgs, status = ftp._download_one_scoring_file("PGS000001", tmp_path, "GRCh37")
    assert (pgs, status) == ("PGS000001", "downloaded")
    assert calls["n"] == 3
    assert (tmp_path / "PGS000001_hmPOS_GRCh37.txt.gz").read_bytes() == b"weight\tdata\n"


def test_scoring_download_fails_after_attempts_and_logs(tmp_path: Path, monkeypatch) -> None:
    """Persistent failure returns 'failed' after N attempts, leaves no partial
    file, and logs the actual error (no longer swallowed silently)."""
    import just_prs.ftp as ftp

    monkeypatch.setattr(ftp, "_SCORING_DOWNLOAD_ATTEMPTS", 2)
    monkeypatch.setattr(ftp.time, "sleep", lambda *_a: None)
    calls = {"n": 0}
    logs: list[dict] = []

    @contextlib.contextmanager
    def always_fail(_url, _mode):
        calls["n"] += 1
        raise OSError("EBI timeout")
        yield  # pragma: no cover

    monkeypatch.setattr(ftp.fsspec, "open", always_fail)
    monkeypatch.setattr(ftp, "log_message", lambda **kw: logs.append(kw))

    pgs, status = ftp._download_one_scoring_file("PGS000002", tmp_path, "GRCh38")
    assert (pgs, status) == ("PGS000002", "failed")
    assert calls["n"] == 2
    assert not (tmp_path / "PGS000002_hmPOS_GRCh38.txt.gz").exists()
    assert logs and logs[0]["message_type"] == "ftp:scoring_download_failed"
    assert "EBI timeout" in logs[0]["error"]
