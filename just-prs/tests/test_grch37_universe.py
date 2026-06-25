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
