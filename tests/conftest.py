"""Shared test fixtures for just-prs tests."""

import platform
import shutil
import stat
import zipfile
from pathlib import Path

import httpx
import pytest

from just_prs.scoring import resolve_cache_dir

TEST_VCF_URL = "https://zenodo.org/records/18370498/files/antonkulaga.vcf?download=1"
TEST_CACHE_DIR = resolve_cache_dir() / "test-data"

PLINK2_BASE_URL = "https://s3.amazonaws.com/plink2-assets"
PLINK2_VERSION_DATE = "20260110"

PLINK2_PLATFORM_MAP: dict[tuple[str, str], str] = {
    ("Linux", "x86_64"): f"plink2_linux_x86_64_{PLINK2_VERSION_DATE}.zip",
    ("Linux", "aarch64"): f"plink2_linux_aarch64_{PLINK2_VERSION_DATE}.zip",
    ("Darwin", "arm64"): f"plink2_mac_arm64_{PLINK2_VERSION_DATE}.zip",
    ("Darwin", "x86_64"): f"plink2_mac_{PLINK2_VERSION_DATE}.zip",
    ("Windows", "AMD64"): f"plink2_win64_{PLINK2_VERSION_DATE}.zip",
}


def _download_plink2(cache_dir: Path) -> Path:
    """Download the PLINK2 binary for the current platform and return its path."""
    system = platform.system()
    machine = platform.machine()
    filename = PLINK2_PLATFORM_MAP.get((system, machine))
    if filename is None:
        pytest.skip(
            f"Unsupported platform for PLINK2 auto-download: {system}/{machine}"
        )

    binary_name = "plink2.exe" if system == "Windows" else "plink2"
    binary_path = cache_dir / binary_name

    if binary_path.exists():
        return binary_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{PLINK2_BASE_URL}/{filename}"
    zip_path = cache_dir / filename

    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in response.iter_bytes(chunk_size=65536):
                    f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(binary_name, cache_dir)

    if system != "Windows":
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    zip_path.unlink()
    return binary_path


@pytest.fixture(scope="session")
def vcf_path() -> Path:
    """Download the test VCF file once per session and return its path."""
    TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vcf_file = TEST_CACHE_DIR / "antonkulaga.vcf"

    if not vcf_file.exists():
        with httpx.Client(timeout=300.0, follow_redirects=True) as client:
            with client.stream("GET", TEST_VCF_URL) as response:
                response.raise_for_status()
                with vcf_file.open("wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)

    return vcf_file


@pytest.fixture(scope="session")
def scoring_cache_dir() -> Path:
    """Return a session-scoped cache directory for scoring files."""
    cache = TEST_CACHE_DIR / "scores"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture(scope="session")
def plink2_path() -> Path:
    """Return path to a plink2 binary, downloading it if not in PATH."""
    system_plink2 = shutil.which("plink2")
    if system_plink2 is not None:
        return Path(system_plink2)
    return _download_plink2(resolve_cache_dir() / "plink2")
