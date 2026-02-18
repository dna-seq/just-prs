"""Shared test fixtures for just-prs tests."""

from pathlib import Path

import httpx
import pytest

TEST_VCF_URL = "https://zenodo.org/records/18370498/files/antonkulaga.vcf?download=1"
TEST_CACHE_DIR = Path.home() / ".cache" / "just-prs" / "test-data"


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
