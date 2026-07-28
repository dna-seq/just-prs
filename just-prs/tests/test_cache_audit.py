"""Tests for the read-only cache audit / duplication canary.

Builds real cache trees on disk (no mocking) and asserts exact byte accounting
and flag behaviour.  The sizes are chosen so every class has a distinct total,
which makes a misclassification impossible to hide behind a passing sum.
"""

import json
import os
from pathlib import Path

import polars as pl
import pytest

from just_prs.cache_audit import (
    CACHE_MANIFEST,
    ArtifactClass,
    CacheProfile,
    FlagSeverity,
    format_bytes,
    scan_cache,
)

# Distinct sizes per class so totals identify their source unambiguously.
SZ_WARM = 1000
SZ_SECONDARY = 2000
SZ_GZ = 4000
SZ_TARBALL = 8000
SZ_NESTED = 16000
SZ_ROGUE = 32000
SZ_RUNTIME = 64000
SZ_PANEL = 128000
SZ_DEV = 256000


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _parquet(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"chrom": ["1"], "pos": [1], "ref": ["A"]}).write_parquet(path)
    return path


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A cache containing exactly one instance of every artifact class."""
    root = tmp_path / "just-prs"
    scores = root / "scores"

    _write(scores / "PGS000001_hmPOS_GRCh38.parquet", SZ_WARM)
    _write(scores / "PGS000002_hmPOS_GRCh38.parquet", SZ_WARM)
    _write(scores / "PGS000001_hmPOS_GRCh37.parquet", SZ_SECONDARY)
    _write(scores / "PGS000001_hmPOS_GRCh38.txt.gz", SZ_GZ)

    _write(root / "metadata" / "scores.parquet", SZ_RUNTIME)
    _write(root / "reference" / "data" / "reference" / "universe.parquet", SZ_NESTED)

    _write(root / "reference_panel" / "pgsc_1000G_v1" / "x.pgen", SZ_PANEL)
    _write(root / "reference_panel" / "pgsc_1000G_v1.tar.zst", SZ_TARBALL)

    _write(root / "test-data" / "sample.vcf", SZ_DEV)
    return root


def test_every_byte_is_attributed_to_a_class(cache: Path) -> None:
    """The per-class breakdown must partition the total exactly."""
    report = scan_cache(cache)

    assert sum(report.by_class.values()) == report.total_bytes
    assert sum(e.bytes for e in report.entries) == report.total_bytes

    on_disk = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
    assert report.total_bytes == on_disk
    assert report.n_files == sum(1 for p in cache.rglob("*") if p.is_file())


def test_classes_match_expected_sizes(cache: Path) -> None:
    """Each class total identifies exactly the files that should be in it."""
    report = scan_cache(cache, primary_build="GRCh38")
    by = report.by_class

    assert by[ArtifactClass.WARM_CATALOG] == 2 * SZ_WARM
    assert by[ArtifactClass.SECONDARY_BUILD] == SZ_SECONDARY
    assert by[ArtifactClass.LEAK] == SZ_GZ + SZ_TARBALL + SZ_NESTED
    assert by[ArtifactClass.REBUILD] == SZ_PANEL
    assert by[ArtifactClass.DEV] == SZ_DEV
    assert by[ArtifactClass.RUNTIME] == SZ_RUNTIME
    assert by[ArtifactClass.UNKNOWN] == 0


def test_primary_build_switch_moves_bytes_between_classes(cache: Path) -> None:
    """Which build is 'warm' is a parameter, not a hardcoded assumption."""
    as38 = scan_cache(cache, primary_build="GRCh38")
    as37 = scan_cache(cache, primary_build="GRCh37")

    assert as38.by_class[ArtifactClass.WARM_CATALOG] == 2 * SZ_WARM
    assert as37.by_class[ArtifactClass.WARM_CATALOG] == SZ_SECONDARY
    assert as37.by_class[ArtifactClass.SECONDARY_BUILD] == 2 * SZ_WARM
    assert as38.total_bytes == as37.total_bytes


def test_expected_and_delta_follow_the_profile(cache: Path) -> None:
    """Excess is everything outside the profile's expected classes."""
    warm = scan_cache(cache, profile=CacheProfile.WARM)
    assert warm.expected_bytes == 2 * SZ_WARM + SZ_RUNTIME
    assert warm.delta_bytes == warm.total_bytes - warm.expected_bytes

    rebuild = scan_cache(cache, profile=CacheProfile.REBUILD)
    # A rebuild host legitimately holds the panel and the second build.
    assert rebuild.expected_bytes == (
        2 * SZ_WARM + SZ_RUNTIME + SZ_SECONDARY + SZ_PANEL
    )
    assert rebuild.expected_bytes > warm.expected_bytes


@pytest.mark.parametrize(
    "code,severity",
    [
        ("scoring_gz_present", FlagSeverity.ERROR),
        ("hf_nested_data_dir", FlagSeverity.ERROR),
        ("panel_tarball_present", FlagSeverity.ERROR),
        ("secondary_build_scores", FlagSeverity.WARN),
        ("rebuild_artifacts_present", FlagSeverity.WARN),
        ("dev_artifacts_present", FlagSeverity.WARN),
    ],
)
def test_each_leak_raises_its_flag(cache: Path, code: str, severity: FlagSeverity) -> None:
    report = scan_cache(cache)
    flag = next((f for f in report.flags if f.code == code), None)
    assert flag is not None, f"{code} did not fire; got {[f.code for f in report.flags]}"
    assert flag.severity is severity
    assert flag.bytes > 0


def test_clean_warm_cache_has_no_errors(tmp_path: Path) -> None:
    """A cache holding only runtime + primary-build parquets is silent."""
    root = tmp_path / "clean"
    _write(root / "scores" / "PGS000001_hmPOS_GRCh38.parquet", SZ_WARM)
    _write(root / "metadata" / "scores.parquet", SZ_RUNTIME)
    _write(root / "liftover" / "hg19ToHg38.over.chain.gz", 10)

    report = scan_cache(root)

    assert not report.has_errors
    assert [f.code for f in report.flags if f.severity is not FlagSeverity.INFO] == []
    assert report.delta_bytes == 0


def test_rogue_directory_and_loose_file_are_flagged(cache: Path) -> None:
    """An artifact nobody planned for must surface, not pass silently.

    This is the canary's real job: the manifest cannot enumerate future
    mistakes, so anything unrecognised is reported by construction.
    """
    rogue = cache / "totally_legit_ml_weights"
    _write(rogue / "model.bin", SZ_ROGUE)
    _write(rogue / "nested" / "deeper" / "checkpoint.pt", SZ_ROGUE)
    _write(cache / "stray_blob.dat", SZ_ROGUE)

    report = scan_cache(cache)

    flag = next(f for f in report.flags if f.code == "unknown_path")
    assert flag.bytes == 3 * SZ_ROGUE, "must recurse into nested rogue subdirectories"
    assert flag.n_files == 3
    assert set(flag.paths) == {"totally_legit_ml_weights/", "stray_blob.dat"}
    assert report.by_class[ArtifactClass.UNKNOWN] == 3 * SZ_ROGUE


def test_stray_scoring_files_at_cache_root(cache: Path) -> None:
    """Scoring files outside scores/ are a leak, not warm catalog."""
    _write(cache / "PGS000337_hmPOS_GRCh37.parquet", SZ_ROGUE)

    report = scan_cache(cache)

    flag = next(f for f in report.flags if f.code == "stray_scores_at_cache_root")
    assert flag.bytes == SZ_ROGUE
    assert report.by_class[ArtifactClass.WARM_CATALOG] == 2 * SZ_WARM  # unchanged


def test_hardlinks_are_counted_once(tmp_path: Path) -> None:
    """Hardlinked duplicates share an inode and must not inflate the total."""
    root = tmp_path / "hl"
    original = _write(root / "scores" / "PGS000001_hmPOS_GRCh38.parquet", SZ_WARM)
    link = root / "scores" / "PGS000002_hmPOS_GRCh38.parquet"
    os.link(original, link)

    report = scan_cache(root)

    assert report.total_bytes == SZ_WARM
    assert report.n_files == 1


def test_catalog_coverage_reports_completeness(cache: Path) -> None:
    report = scan_cache(cache, primary_build="GRCh38")

    assert report.catalog.n_primary_parquet == 2
    assert report.catalog.warm_bytes == 2 * SZ_WARM
    assert 0 < report.catalog.ratio < 1
    assert any(f.code == "catalog_incomplete" for f in report.flags)


def test_missing_cache_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_cache(tmp_path / "does-not-exist")


def test_report_is_json_serialisable(cache: Path) -> None:
    """--json output must round-trip for monitoring consumers."""
    report = scan_cache(cache)
    parsed = json.loads(report.model_dump_json())

    assert parsed["total_bytes"] == report.total_bytes
    assert parsed["flags"][0]["severity"] in {s.value for s in FlagSeverity}
    assert parsed["by_class"]["leak"] == report.by_class[ArtifactClass.LEAK]


def test_manifest_covers_documented_cache_layout() -> None:
    """Every directory the codebase creates must be classified.

    A new cache subdirectory added without a manifest entry would be reported as
    UNKNOWN on every host — this pins the intended set so that surfaces as a
    test failure at development time instead of a warning in production.
    """
    documented = {
        "metadata", "percentiles", "reference", "ancestry", "chip_manifests",
        "liftover", "scores", "reference_panel", "reference_fasta",
        "reference_scores", "normalized", "genomes", "test-data", "results",
        "plink2", "scoring",
    }
    assert set(CACHE_MANIFEST) == documented


def test_format_bytes_scales() -> None:
    assert format_bytes(512) == "512 B"
    assert format_bytes(1024) == "1.0 K"
    assert format_bytes(5 * 1024**3) == "5.0 G"
