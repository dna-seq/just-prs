"""Dagster resources for the PRS reference panel pipeline."""

import os
import subprocess
from pathlib import Path

from dagster import ConfigurableResource
from dotenv import load_dotenv


class Plink2Resource(ConfigurableResource):
    """Manages the PLINK2 binary.

    Resolves the binary path from config, a ``PLINK2_BIN`` env var, or
    the just-prs cache directory where ``just_prs.plink`` auto-downloads it.
    """

    plink2_bin: str = ""

    def get_bin(self) -> Path:
        """Return the resolved Path to the plink2 binary."""
        if self.plink2_bin:
            return Path(self.plink2_bin)
        env_bin = os.environ.get("PLINK2_BIN", "")
        if env_bin:
            return Path(env_bin)
        # Fall back to the just-prs auto-downloaded plink2
        from just_prs.scoring import resolve_cache_dir
        cached = resolve_cache_dir() / "plink2" / "plink2"
        if cached.exists():
            return cached
        # Last resort: assume it is on PATH
        return Path("plink2")

    def check_available(self) -> bool:
        """Return True if plink2 is available and executable."""
        result = subprocess.run(
            [str(self.get_bin()), "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0


class CacheDirResource(ConfigurableResource):
    """Provides a configurable root cache directory.

    Defaults to the just-prs platform cache dir, overridden by ``PRS_CACHE_DIR``
    env var or explicit config.
    """

    cache_dir: str = ""

    def get_path(self) -> Path:
        """Return the resolved cache directory Path."""
        if self.cache_dir:
            return Path(self.cache_dir)
        raw = os.environ.get("PRS_CACHE_DIR", "")
        if raw:
            return Path(raw)
        from just_prs.scoring import resolve_cache_dir
        return resolve_cache_dir()


class HuggingFaceResource(ConfigurableResource):
    """Provides the HuggingFace API token resolved from config or environment."""

    token: str = ""
    percentiles_repo: str = "just-dna-seq/prs-percentiles"
    metadata_repo: str = "just-dna-seq/polygenic_risk_scores"

    def get_token(self) -> str | None:
        """Return the resolved HF token."""
        if self.token:
            return self.token
        load_dotenv()
        return os.environ.get("HF_TOKEN")
