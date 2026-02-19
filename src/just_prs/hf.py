"""HuggingFace Hub integration for pushing/pulling cleaned PGS metadata parquets."""

import os
from pathlib import Path

from dotenv import load_dotenv
from eliot import start_action
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_HF_REPO = "just-dna-seq/polygenic_risk_scores"
HF_DATA_PREFIX = "data"

CLEANED_PARQUET_FILES = [
    "scores.parquet",
    "performance.parquet",
    "best_performance.parquet",
]


def _resolve_token(token: str | None = None) -> str | None:
    """Resolve HF token: explicit arg > env var (with dotenv loaded)."""
    if token:
        return token
    load_dotenv()
    return os.environ.get("HF_TOKEN")


def push_cleaned_parquets(
    local_dir: Path,
    repo_id: str = DEFAULT_HF_REPO,
    token: str | None = None,
) -> None:
    """Upload cleaned parquet files from local_dir to HF dataset repo under data/.

    Args:
        local_dir: Directory containing scores.parquet, performance.parquet, best_performance.parquet
        repo_id: HuggingFace dataset repository ID
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.
    """
    resolved_token = _resolve_token(token)
    with start_action(action_type="hf:push_cleaned_parquets", repo_id=repo_id):
        api = HfApi(token=resolved_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=HF_DATA_PREFIX,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns="*.parquet",
        )


def pull_cleaned_parquets(
    local_dir: Path,
    repo_id: str = DEFAULT_HF_REPO,
    token: str | None = None,
) -> list[Path]:
    """Download cleaned parquet files from HF dataset repo into local_dir.

    Args:
        local_dir: Directory to save downloaded parquet files
        repo_id: HuggingFace dataset repository ID
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.

    Returns:
        List of paths to downloaded parquet files
    """
    resolved_token = _resolve_token(token)
    with start_action(action_type="hf:pull_cleaned_parquets", repo_id=repo_id):
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for filename in CLEANED_PARQUET_FILES:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{HF_DATA_PREFIX}/{filename}",
                repo_type="dataset",
                local_dir=local_dir,
                token=resolved_token,
            )
            target = local_dir / filename
            hf_cached = Path(path)
            if hf_cached != target:
                import shutil
                shutil.copy2(hf_cached, target)
            downloaded.append(target)
        return downloaded
