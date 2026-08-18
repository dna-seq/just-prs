"""Dagster resources for the PRS reference panel pipeline."""

import os
from pathlib import Path

from dagster import ConfigurableResource
from dotenv import load_dotenv


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
    catalog_repo: str = "just-dna-seq/pgs-catalog"
    sample_scores_repo: str = "just-dna-seq/prs-sample-scores"

    def get_token(self) -> str | None:
        """Return the resolved HF token."""
        if self.token:
            return self.token
        load_dotenv()
        return os.environ.get("HF_TOKEN")

    def get_sample_scores_repo(self) -> str:
        """Return the public sample-score dataset repo (env override first)."""
        return os.environ.get("PRS_SAMPLE_SCORES_REPO") or self.sample_scores_repo
