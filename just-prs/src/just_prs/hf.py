"""HuggingFace Hub integration for pushing/pulling cleaned PGS metadata parquets."""

import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from eliot import start_action
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_HF_REPO = "just-dna-seq/polygenic_risk_scores"
DEFAULT_HF_PERCENTILES_REPO = "just-dna-seq/prs-percentiles"
HF_DATA_PREFIX = "data"

CLEANED_PARQUET_FILES = [
    "scores.parquet",
    "performance.parquet",
    "best_performance.parquet",
]

REFERENCE_DISTRIBUTIONS_FILE = "reference_distributions.parquet"  # legacy fallback


def distributions_filename(panel: str = "1000g") -> str:
    """Return the panel-aware filename for a reference distributions parquet."""
    return f"{panel}_distributions.parquet"


def _resolve_token(token: str | None = None) -> str | None:
    """Resolve HF token: explicit arg > env var (with dotenv loaded)."""
    if token:
        return token
    load_dotenv()
    return os.environ.get("HF_TOKEN")


def _generate_dataset_card(local_dir: Path, repo_id: str) -> str:
    """Generate a HuggingFace dataset card (README.md) describing the cleaned PGS metadata."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    stats_lines: list[str] = []
    for filename in CLEANED_PARQUET_FILES:
        p = local_dir / filename
        if p.exists():
            df = pl.read_parquet(p)
            cols = ", ".join(f"`{c}`" for c in df.columns[:8])
            if len(df.columns) > 8:
                cols += f", ... ({len(df.columns)} total)"
            stats_lines.append(f"| `{filename}` | {df.height:,} | {len(df.columns)} | {cols} |")

    stats_table = "\n".join(stats_lines) if stats_lines else "| — | — | — | — |"

    return f"""---
license: cc-by-4.0
task_categories:
  - tabular-classification
  - tabular-regression
tags:
  - biology
  - genomics
  - polygenic-risk-scores
  - pgs-catalog
  - genetics
  - health
pretty_name: PGS Catalog — Cleaned Polygenic Risk Score Metadata
size_categories:
  - 10K<n<100K
source_datasets:
  - PGS Catalog (https://www.pgscatalog.org/)
---

# PGS Catalog — Cleaned Polygenic Risk Score Metadata

Cleaned and normalized metadata from the [PGS Catalog](https://www.pgscatalog.org/),
the open database of published **Polygenic Risk Scores (PRS)**.

This dataset is automatically built from the bulk FTP downloads provided by PGS Catalog
and processed through the [`just-prs`](https://github.com/longevity-genie/just-prs) cleanup pipeline.

**Last updated:** {now}

## Files

All files are in the `data/` directory in Apache Parquet format.

| File | Rows | Columns | Key columns |
|------|------|---------|-------------|
{stats_table}

## Cleanup Pipeline

The raw PGS Catalog CSVs undergo several transformations:

1. **Column renaming** — verbose PGS column names (e.g. `Polygenic Score (PGS) ID`) are mapped to
   short `snake_case` equivalents (`pgs_id`).
2. **Genome build normalization** — 9 raw build variants (`hg19`, `hg37`, `hg38`, `NCBI36`, `hg18`,
   `NCBI35`, `GRCh37`, `GRCh38`, `NR`) are mapped to canonical values: `GRCh37`, `GRCh38`, `GRCh36`, or `NR`.
3. **Metric string parsing** — performance metrics stored as strings like `"1.55 [1.52,1.58]"` or
   `"-0.7 (0.15)"` are parsed into structured numeric columns (`*_estimate`, `*_ci_lower`, `*_ci_upper`, `*_se`)
   for OR, HR, Beta, AUROC, and C-index.
4. **Performance flattening** — performance metrics are joined with evaluation sample sets to include
   sample size (`n_individuals`) and ancestry (`ancestry_broad`).
5. **Best performance selection** — `best_performance.parquet` contains one row per PGS ID, selecting
   the evaluation with the largest sample size (with a preference for European-ancestry cohorts).

## Table Descriptions

### `scores.parquet`

One row per polygenic score in PGS Catalog. Key columns:
- `pgs_id` — PGS Catalog identifier (e.g. PGS000001)
- `name` — score name
- `trait_reported` / `trait_efo` — reported and mapped trait labels
- `genome_build` — normalized genome build (GRCh37, GRCh38, GRCh36, or NR)
- `n_variants` — number of variants in the scoring file
- `weight_type` — type of variant weights (e.g. beta, OR, log(OR))
- `ftp_link` — direct FTP link to the scoring file

### `performance.parquet`

One row per performance evaluation, with parsed numeric metrics:
- `ppm_id` — performance metric identifier
- `pgs_id` — evaluated score
- `or_estimate`, `hr_estimate`, `beta_estimate`, `auroc_estimate`, `cindex_estimate` — parsed point estimates
- `*_ci_lower`, `*_ci_upper`, `*_se` — confidence intervals and standard errors
- `n_individuals`, `ancestry_broad` — evaluation sample characteristics

### `best_performance.parquet`

One row per PGS ID with the single best performance evaluation (largest sample, European-preferred).
Same columns as `performance.parquet`.

## Source & License

- **Source:** [PGS Catalog](https://www.pgscatalog.org/) (EBI / NHGRI)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Citation:** Lambert, S.A. et al. *The Polygenic Score Catalog as an open database for
  reproducibility and systematic evaluation.* Nature Genetics 53, 420–425 (2021).
  [doi:10.1038/s41588-021-00783-5](https://doi.org/10.1038/s41588-021-00783-5)
"""


def push_cleaned_parquets(
    local_dir: Path,
    repo_id: str = DEFAULT_HF_REPO,
    token: str | None = None,
) -> None:
    """Upload cleaned parquet files and dataset card from local_dir to HF dataset repo.

    Generates a README.md dataset card describing the contents, then uploads
    both the parquets (under data/) and the card to the repo root.

    Args:
        local_dir: Directory containing scores.parquet, performance.parquet, best_performance.parquet
        repo_id: HuggingFace dataset repository ID
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.
    """
    resolved_token = _resolve_token(token)
    with start_action(action_type="hf:push_cleaned_parquets", repo_id=repo_id):
        api = HfApi(token=resolved_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        readme_path = local_dir / "README.md"
        readme_path.write_text(_generate_dataset_card(local_dir, repo_id))
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=HF_DATA_PREFIX,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns="*.parquet",
        )


def pull_reference_distributions(
    local_dir: Path,
    repo_id: str = DEFAULT_HF_PERCENTILES_REPO,
    token: str | None = None,
    panel: str = "1000g",
) -> Path | None:
    """Download a panel-aware distributions parquet from the prs-percentiles HF dataset repo.

    Tries the panel-aware filename first (e.g. ``1000g_distributions.parquet``),
    then falls back to the legacy ``reference_distributions.parquet`` for
    backward compatibility.

    Args:
        local_dir: Directory to save the downloaded parquet file.
        repo_id: HuggingFace dataset repository ID for percentiles.
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.
        panel: Reference panel identifier (e.g. ``1000g``, ``hgdp_1kg``).

    Returns:
        Path to the downloaded file, or None if not available in the repo yet.
    """
    import logging
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    resolved_token = _resolve_token(token)
    panel_file = distributions_filename(panel)

    with start_action(action_type="hf:pull_reference_distributions", repo_id=repo_id, panel=panel):
        local_dir.mkdir(parents=True, exist_ok=True)

        for candidate in [panel_file, REFERENCE_DISTRIBUTIONS_FILE]:
            hf_path = f"{HF_DATA_PREFIX}/{candidate}"
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_path,
                    repo_type="dataset",
                    local_dir=local_dir,
                    token=resolved_token,
                )
            except (EntryNotFoundError, RepositoryNotFoundError):
                continue

            target = local_dir / panel_file
            hf_cached = Path(path)
            if hf_cached != target:
                import shutil
                shutil.copy2(hf_cached, target)
            return target

        logging.getLogger(__name__).debug(
            "Reference distributions not found on HF (%s) for panel %s", repo_id, panel,
        )
        return None


def push_reference_distributions(
    parquet_path: Path,
    repo_id: str = DEFAULT_HF_PERCENTILES_REPO,
    token: str | None = None,
    panel: str = "1000g",
) -> None:
    """Upload a distributions parquet to the prs-percentiles HF dataset repo.

    The file is uploaded as ``data/{panel}_distributions.parquet``.

    Args:
        parquet_path: Local path to the distributions parquet file.
        repo_id: HuggingFace dataset repository ID for percentiles.
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.
        panel: Reference panel identifier (e.g. ``1000g``, ``hgdp_1kg``).
    """
    resolved_token = _resolve_token(token)
    panel_file = distributions_filename(panel)
    with start_action(action_type="hf:push_reference_distributions", repo_id=repo_id, panel=panel):
        api = HfApi(token=resolved_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"{HF_DATA_PREFIX}/{panel_file}",
            repo_id=repo_id,
            repo_type="dataset",
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
