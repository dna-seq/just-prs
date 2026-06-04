"""HuggingFace Hub integration for pushing/pulling cleaned PGS metadata parquets."""

import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from eliot import start_action, Message
from huggingface_hub import HfApi, hf_hub_download
import huggingface_hub.constants as _hf_constants

DEFAULT_HF_PERCENTILES_REPO = "just-dna-seq/prs-percentiles"
DEFAULT_HF_CATALOG_REPO = "just-dna-seq/pgs-catalog"
HF_DATA_PREFIX = "data"

CLEANED_PARQUET_FILES = [
    "scores.parquet",
    "performance.parquet",
    "best_performance.parquet",
    "publications.parquet",
    "pgs_quality_scores.parquet",
]

REFERENCE_DISTRIBUTIONS_FILE = "reference_distributions.parquet"  # legacy fallback

HF_UPLOAD_TIMEOUT_SEC = 1800
HF_UPLOAD_MAX_RETRIES = 5
HF_UPLOAD_RETRY_DELAY_SEC = 30
HF_DOWNLOAD_TIMEOUT_SEC = 120
HF_DOWNLOAD_MAX_RETRIES = 4
HF_DOWNLOAD_BACKOFF_BASE = 2.0


def _configure_hf_timeouts(timeout_sec: int = HF_UPLOAD_TIMEOUT_SEC) -> None:
    """Raise HuggingFace Hub HTTP timeouts from the 10s default.

    The default 10-second timeout is far too low for large uploads
    (thousands of scoring parquets).  This must be called before any
    HfApi method that performs HTTP requests.
    """
    _hf_constants.DEFAULT_DOWNLOAD_TIMEOUT = timeout_sec
    _hf_constants.DEFAULT_REQUEST_TIMEOUT = timeout_sec
    _hf_constants.DEFAULT_ETAG_TIMEOUT = timeout_sec
    _hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = timeout_sec
    _hf_constants.HF_HUB_ETAG_TIMEOUT = timeout_sec


def distributions_filename(panel: str = "1000g") -> str:
    """Return the panel-aware filename for a reference distributions parquet."""
    return f"{panel}_distributions.parquet"


def _hf_download_with_retry(
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    local_dir: Path | None = None,
    token: str | None = None,
) -> str:
    """``hf_hub_download`` with retry on 429 / transient errors."""
    import logging
    from huggingface_hub.utils import HfHubHTTPError

    _configure_hf_timeouts(HF_DOWNLOAD_TIMEOUT_SEC)
    _logger = logging.getLogger(__name__)
    last_exc: Exception | None = None
    for attempt in range(HF_DOWNLOAD_MAX_RETRIES):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=local_dir,
                token=token,
            )
        except HfHubHTTPError as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in (429, 500, 502, 503, 504):
                raise
            if attempt < HF_DOWNLOAD_MAX_RETRIES - 1:
                delay = HF_DOWNLOAD_BACKOFF_BASE ** attempt
                _logger.warning(
                    "HF download %s/%s failed (%s), retry %d in %.0fs",
                    repo_id, filename, status, attempt + 1, delay,
                )
                time.sleep(delay)
        except (OSError, TimeoutError) as exc:
            last_exc = exc
            if attempt < HF_DOWNLOAD_MAX_RETRIES - 1:
                delay = HF_DOWNLOAD_BACKOFF_BASE ** attempt
                _logger.warning(
                    "HF download %s/%s failed (%s), retry %d in %.0fs",
                    repo_id, filename, exc, attempt + 1, delay,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"HF download {repo_id}/{filename} failed after {HF_DOWNLOAD_MAX_RETRIES} attempts: {last_exc}"
    )


def _resolve_token(token: str | None = None) -> str | None:
    """Resolve HF token: explicit arg > env var (with dotenv loaded)."""
    if token:
        return token
    load_dotenv()
    return os.environ.get("HF_TOKEN")


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
                path = _hf_download_with_retry(
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
    issue_report_path: Path | None = None,
    audit_summary_path: Path | None = None,
) -> None:
    """Upload a distributions parquet to the prs-percentiles HF dataset repo.

    The distribution file is uploaded as ``data/{panel}_distributions.parquet``.
    If provided, the distribution quality issue report is uploaded alongside it
    as ``data/{panel}_distribution_quality_issues.parquet`` for audit/debugging.
    If provided, the audit summary is uploaded as
    ``data/{panel}_distribution_audit_summary.json``.

    Args:
        parquet_path: Local path to the distributions parquet file.
        repo_id: HuggingFace dataset repository ID for percentiles.
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.
        panel: Reference panel identifier (e.g. ``1000g``, ``hgdp_1kg``).
        issue_report_path: Optional sidecar parquet with quarantined distribution issues.
        audit_summary_path: Optional compact JSON audit summary.
    """
    resolved_token = _resolve_token(token)
    panel_file = distributions_filename(panel)
    with start_action(action_type="hf:push_reference_distributions", repo_id=repo_id, panel=panel):
        _configure_hf_timeouts()
        api = HfApi(token=resolved_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"{HF_DATA_PREFIX}/{panel_file}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        if issue_report_path is not None and issue_report_path.exists():
            api.upload_file(
                path_or_fileobj=str(issue_report_path),
                path_in_repo=f"{HF_DATA_PREFIX}/{panel}_distribution_quality_issues.parquet",
                repo_id=repo_id,
                repo_type="dataset",
            )
        if audit_summary_path is not None and audit_summary_path.exists():
            api.upload_file(
                path_or_fileobj=str(audit_summary_path),
                path_in_repo=f"{HF_DATA_PREFIX}/{panel}_distribution_audit_summary.json",
                repo_id=repo_id,
                repo_type="dataset",
            )


def pull_chip_coverage(
    local_dir: Path,
    repo_id: str = DEFAULT_HF_PERCENTILES_REPO,
    token: str | None = None,
) -> Path | None:
    """Download chip_coverage.parquet from the prs-percentiles HF dataset repo.

    Args:
        local_dir: Directory to save the downloaded parquet file.
        repo_id: HuggingFace dataset repository ID for percentiles.
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.

    Returns:
        Path to the downloaded file, or None if not available in the repo yet.
    """
    import logging
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    resolved_token = _resolve_token(token)
    with start_action(action_type="hf:pull_chip_coverage", repo_id=repo_id):
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = _hf_download_with_retry(
                repo_id=repo_id,
                filename=f"{HF_DATA_PREFIX}/chip_coverage.parquet",
                repo_type="dataset",
                local_dir=local_dir,
                token=resolved_token,
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            logging.getLogger(__name__).debug(
                "chip_coverage.parquet not found on HF (%s)", repo_id,
            )
            return None
        target = local_dir / "chip_coverage.parquet"
        hf_cached = Path(path)
        if hf_cached != target:
            import shutil
            shutil.copy2(hf_cached, target)
        return target


def push_chip_coverage(
    parquet_path: Path,
    repo_id: str = DEFAULT_HF_PERCENTILES_REPO,
    token: str | None = None,
) -> None:
    """Upload a consumer-chip coverage parquet to the prs-percentiles HF dataset repo.

    The file is uploaded as ``data/chip_coverage.parquet``. It reports, per PGS ID
    and per consumer genotyping chip, how many of the score's variants are directly
    typed (vs requiring imputation). End users pull this to label each PRS
    "array-ready" or "imputation-required" for a given chip.

    Args:
        parquet_path: Local path to the chip coverage parquet file.
        repo_id: HuggingFace dataset repository ID for percentiles.
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.
    """
    resolved_token = _resolve_token(token)
    with start_action(action_type="hf:push_chip_coverage", repo_id=repo_id):
        _configure_hf_timeouts()
        api = HfApi(token=resolved_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"{HF_DATA_PREFIX}/chip_coverage.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )


def _generate_catalog_dataset_card(
    metadata_dir: Path,
    scores_dir: Path,
    repo_id: str,
) -> str:
    """Generate a HuggingFace dataset card for the combined PGS Catalog dataset.

    Includes statistics and timestamps for both cleaned metadata and scoring
    file parquets.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_human = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metadata_lines: list[str] = []
    n_unique_pgs = 0
    for filename in CLEANED_PARQUET_FILES:
        p = metadata_dir / filename
        if p.exists():
            df = pl.read_parquet(p)
            cols = ", ".join(f"`{c}`" for c in df.columns[:8])
            if len(df.columns) > 8:
                cols += f", ... ({len(df.columns)} total)"
            metadata_lines.append(
                f"| `metadata/{filename}` | {df.height:,} | {len(df.columns)} | {cols} |"
            )
            if filename == "scores.parquet" and "pgs_id" in df.columns:
                n_unique_pgs = df["pgs_id"].n_unique()

    metadata_table = "\n".join(metadata_lines) if metadata_lines else "| — | — | — | — |"

    scoring_parquets = sorted(scores_dir.glob("*_hmPOS_*.parquet"))
    scoring_parquets = [p for p in scoring_parquets if p.name != "conversion_failures.parquet"]
    n_scoring = len(scoring_parquets)
    total_scoring_bytes = sum(p.stat().st_size for p in scoring_parquets)
    total_scoring_gb = total_scoring_bytes / (1024 ** 3)

    genome_build = "GRCh38"
    if scoring_parquets:
        name = scoring_parquets[0].name
        if "GRCh37" in name:
            genome_build = "GRCh37"

    example_files = ", ".join(f"`{p.name}`" for p in scoring_parquets[:3])
    if n_scoring > 3:
        example_files += f", ... ({n_scoring:,} files total)"

    if n_scoring > 100_000:
        size_cat = "1M<n<10M"
    elif n_scoring > 10_000:
        size_cat = "100K<n<1M"
    elif n_scoring > 1_000:
        size_cat = "10K<n<100K"
    else:
        size_cat = "1K<n<10K"

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
  - polars
  - parquet
pretty_name: PGS Catalog — Scoring Files & Metadata (Parquet)
size_categories:
  - {size_cat}
source_datasets:
  - PGS Catalog (https://www.pgscatalog.org/)
---

# PGS Catalog — Scoring Files & Cleaned Metadata

Complete mirror of [PGS Catalog](https://www.pgscatalog.org/) scoring files converted to
Apache Parquet format, together with cleaned and normalised metadata tables.

Built automatically by the [`just-prs`](https://github.com/longevity-genie/just-prs) pipeline.

**Last updated:** {now_human}

## Release Statistics

| Metric | Value |
|--------|-------|
| Scoring file parquets | **{n_scoring:,}** |
| Unique PGS IDs (metadata) | **{n_unique_pgs:,}** |
| Genome build | **{genome_build}** |
| Total scoring data size | **{total_scoring_gb:.1f} GB** |
| Release timestamp | `{now}` |

## Files

### Metadata (`data/metadata/`)

Cleaned and normalised PGS Catalog metadata tables.

| File | Rows | Columns | Key columns |
|------|------|---------|-------------|
{metadata_table}

### Scoring files (`data/scores/`)

Individual scoring weight files, one parquet per PGS ID.
Each file contains variant-level effect weights with harmonised positions.

| Info | Value |
|------|-------|
| Total files | {n_scoring:,} |
| Naming pattern | `{{PGS_ID}}_hmPOS_{genome_build}.parquet` |
| Compression | zstd (level 9) |
| Examples | {example_files} |

## Metadata Cleanup Pipeline

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

## Scoring File Schema

Each scoring parquet contains variant-level data from the PGS Catalog harmonised files:

- `rsID` — dbSNP rsID
- `chr_name` / `chr_position` — chromosome and position (original)
- `effect_allele` / `other_allele` — alleles
- `effect_weight` — variant weight (beta, log-OR, etc.)
- `hm_chr` / `hm_pos` — harmonised chromosome and position ({genome_build})
- `hm_inferOtherAllele` — inferred other allele during harmonisation
- PGS header metadata is embedded as file-level Parquet metadata under the key `pgs_catalog_header`.
- In `data/metadata/scores.parquet`, `ftp_link` points to the parquet mirror in this
  dataset (`data/scores/...parquet`) for fast parquet-first loading; original EBI links
  are preserved in `ftp_link_ebi`.

## Usage

### Load metadata with polars

```python
import polars as pl
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="{repo_id}",
    filename="data/metadata/scores.parquet",
    repo_type="dataset",
)
scores = pl.read_parquet(path)
print(scores.head())
```

### Load a scoring file

```python
path = hf_hub_download(
    repo_id="{repo_id}",
    filename="data/scores/PGS000001_hmPOS_{genome_build}.parquet",
    repo_type="dataset",
)
weights = pl.read_parquet(path)
print(weights.head())
```

## Source & License

- **Source:** [PGS Catalog](https://www.pgscatalog.org/) (EBI / NHGRI)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Citation:** Lambert, S.A. et al. *The Polygenic Score Catalog as an open database for
  reproducibility and systematic evaluation.* Nature Genetics 53, 420-425 (2021).
  [doi:10.1038/s41588-021-00783-5](https://doi.org/10.1038/s41588-021-00783-5)
"""


def _parquet_filename_expr() -> pl.Expr:
    """Build scoring parquet filename from (pgs_id, genome_build) when possible."""
    valid_build = pl.col("genome_build").is_in(["GRCh37", "GRCh38"])
    return (
        pl.when(valid_build)
        .then(
            pl.col("pgs_id")
            + pl.lit("_hmPOS_")
            + pl.col("genome_build")
            + pl.lit(".parquet")
        )
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def _scores_with_parquet_links(scores_df: pl.DataFrame, repo_id: str) -> pl.DataFrame:
    """Return scores metadata with parquet-first links for the combined HF repo.

    - Preserves original EBI link in ``ftp_link_ebi`` (when present).
    - Sets ``ftp_link`` to the combined-repo parquet URL.
    - Adds ``scoring_parquet_filename`` and ``scoring_parquet_path`` columns.
    """
    out = scores_df
    if "ftp_link" in out.columns and "ftp_link_ebi" not in out.columns:
        out = out.rename({"ftp_link": "ftp_link_ebi"})

    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
    return out.with_columns(
        _parquet_filename_expr().alias("scoring_parquet_filename"),
    ).with_columns(
        pl.when(pl.col("scoring_parquet_filename").is_not_null())
        .then(pl.lit(f"{HF_DATA_PREFIX}/scores/") + pl.col("scoring_parquet_filename"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("scoring_parquet_path"),
    ).with_columns(
        pl.when(pl.col("scoring_parquet_path").is_not_null())
        .then(pl.lit(base_url) + pl.col("scoring_parquet_path"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("ftp_link"),
    )


def _upload_large_folder_with_retry(
    api: HfApi,
    repo_id: str,
    folder_path: Path,
    repo_type: str = "dataset",
    max_retries: int = HF_UPLOAD_MAX_RETRIES,
    base_delay: int = HF_UPLOAD_RETRY_DELAY_SEC,
) -> None:
    """Call ``api.upload_large_folder`` with automatic retry on network timeouts.

    ``upload_large_folder`` maintains a ``.cache/.huggingface/`` resumability
    cache inside ``folder_path``, so retrying after a transient timeout
    resumes from where it left off without re-uploading completed files.

    Uses exponential backoff: delay doubles each attempt (30s, 60s, 120s, 240s).
    """
    import httpx
    import httpcore

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_large_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(folder_path),
            )
            return
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout,
                httpcore.ReadTimeout, httpcore.WriteTimeout,
                httpx.RemoteProtocolError, ConnectionError, TimeoutError) as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                Message.log(
                    message_type="hf:upload_retry",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(exc),
                    retry_delay=delay,
                )
                time.sleep(delay)
            else:
                Message.log(
                    message_type="hf:upload_failed",
                    attempts=max_retries,
                    error=str(exc),
                )
    raise last_exc  # type: ignore[misc]


def push_pgs_catalog(
    metadata_dir: Path,
    scores_dir: Path,
    repo_id: str = DEFAULT_HF_CATALOG_REPO,
    token: str | None = None,
    staging_dir: Path | None = None,
) -> None:
    """Upload scoring file parquets and cleaned metadata to a combined HF dataset.

    Uploads cleaned metadata under ``data/metadata/`` and all scoring parquets
    under ``data/scores/``.  Generates a dataset card (README.md) with release
    statistics and timestamps.

    Scoring files (~5,000+ parquets) are uploaded via ``upload_large_folder``
    which chunks the upload into multiple commits and is resilient to network
    errors, avoiding the ``413 Payload Too Large`` failure that plain
    ``upload_folder`` hits when committing thousands of files in one shot.

    ``upload_large_folder`` does not support ``path_in_repo``, and it writes a
    ``.cache/.huggingface/`` resumability cache **inside** the folder being uploaded.
    To preserve that cache across crashes (so interrupted uploads can resume
    on re-run), the staging tree is written to a **persistent** directory
    derived from ``staging_dir`` rather than a throwaway ``TemporaryDirectory``.
    The staging tree mirrors the target repo layout::

        <staging_dir>/data/metadata/   ← cleaned metadata parquets
        <staging_dir>/data/scores/     ← hard-linked scoring parquets
        <staging_dir>/README.md
        <staging_dir>/.cache/.huggingface/  ← resumability cache (managed by HF)

    The staging dir is cleaned up **only after a successful upload** so that an
    interrupted run can resume from where it left off. It must not be placed
    inside ``scores_dir`` or ``metadata_dir``.

    Args:
        metadata_dir: Directory containing scores.parquet, performance.parquet,
            best_performance.parquet (cleaned metadata).
        scores_dir: Directory containing ``{PGS_ID}_hmPOS_{build}.parquet`` scoring files.
        repo_id: HuggingFace dataset repository ID.
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.
        staging_dir: Persistent staging directory for the upload tree and the
            HF resumability cache. If None, defaults to
            ``scores_dir.parent / "hf_catalog_upload_staging"``.
            Must NOT be a subdirectory of ``scores_dir`` or ``metadata_dir``.
    """
    resolved_token = _resolve_token(token)
    metadata_dir = metadata_dir.resolve()
    scores_dir = scores_dir.resolve()
    if staging_dir is None:
        staging_dir = scores_dir.parent / "hf_catalog_upload_staging"
    staging_dir = staging_dir.resolve()

    if staging_dir == metadata_dir or staging_dir == scores_dir:
        raise ValueError("staging_dir must be separate from metadata_dir and scores_dir")
    if metadata_dir in staging_dir.parents or scores_dir in staging_dir.parents:
        raise ValueError("staging_dir must not be inside metadata_dir or scores_dir")

    with start_action(
        action_type="hf:push_pgs_catalog",
        repo_id=repo_id,
        metadata_dir=str(metadata_dir),
        scores_dir=str(scores_dir),
        staging_dir=str(staging_dir),
    ):
        _configure_hf_timeouts()
        api = HfApi(token=resolved_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        staged_metadata_dir = staging_dir / HF_DATA_PREFIX / "metadata"
        staged_scores_dir = staging_dir / HF_DATA_PREFIX / "scores"
        staged_metadata_dir.mkdir(parents=True, exist_ok=True)
        staged_scores_dir.mkdir(parents=True, exist_ok=True)

        for name in CLEANED_PARQUET_FILES:
            src = metadata_dir / name
            dst = staged_metadata_dir / name
            if name == "scores.parquet":
                scores_df = pl.read_parquet(src)
                scores_parquet = _scores_with_parquet_links(scores_df, repo_id=repo_id)
                scores_parquet.write_parquet(dst)
            else:
                shutil.copy2(src, dst)

        readme_content = _generate_catalog_dataset_card(staged_metadata_dir, scores_dir, repo_id)
        readme_path = staging_dir / "README.md"
        readme_path.write_text(readme_content)

        source_scores = {
            src_file.name: src_file
            for src_file in scores_dir.glob("*_hmPOS_*.parquet")
            if src_file.name != "conversion_failures.parquet"
        }

        # Drop stale staged files so a resumed upload never publishes parquets
        # that were removed or renamed in the source directory.
        for staged_file in staged_scores_dir.glob("*_hmPOS_*.parquet"):
            if staged_file.name not in source_scores:
                staged_file.unlink()

        # Hard-link scoring parquets into the staged layout to avoid duplicating
        # gigabytes on disk. Refresh a staged file when the source inode changed
        # (e.g. source file was replaced) or when a copy fallback drifted.
        for src_file in source_scores.values():
            dst_file = staged_scores_dir / src_file.name
            refresh_dst = True
            if dst_file.exists():
                try:
                    refresh_dst = not dst_file.samefile(src_file)
                except OSError:
                    src_stat = src_file.stat()
                    dst_stat = dst_file.stat()
                    refresh_dst = (
                        dst_stat.st_size != src_stat.st_size
                        or dst_stat.st_mtime_ns != src_stat.st_mtime_ns
                    )
            if refresh_dst:
                if dst_file.exists():
                    dst_file.unlink()
                try:
                    dst_file.hardlink_to(src_file)
                except FileExistsError:
                    # Another Dagster run may have refreshed the shared staging
                    # tree between exists()/unlink() and hardlink_to().
                    if dst_file.exists() and dst_file.samefile(src_file):
                        continue
                    dst_file.unlink(missing_ok=True)
                    dst_file.hardlink_to(src_file)
                except OSError:
                    if dst_file.exists() and dst_file.samefile(src_file):
                        continue
                    shutil.copy2(src_file, dst_file)

        # upload_large_folder writes .cache/.huggingface/ resumability cache inside
        # staging_dir.  Keeping staging_dir persistent means a crashed upload
        # can resume on re-run without re-hashing or re-uploading completed files.
        # Retry on transient network timeouts — the resumability cache ensures
        # no work is repeated.
        _upload_large_folder_with_retry(
            api=api,
            repo_id=repo_id,
            folder_path=staging_dir,
        )

        # Upload succeeded — remove the staging tree so stale hard-links don't
        # linger on disk.  The .huggingface/ cache is no longer needed.
        shutil.rmtree(staging_dir, ignore_errors=True)


def pull_cleaned_parquets(
    local_dir: Path,
    repo_id: str = DEFAULT_HF_CATALOG_REPO,
    token: str | None = None,
) -> list[Path]:
    """Download cleaned metadata parquets from the combined PGS Catalog HF dataset.

    Pulls ``scores.parquet``, ``performance.parquet``, and
    ``best_performance.parquet`` from ``data/metadata/`` in the combined
    ``just-dna-seq/pgs-catalog`` repo.  Retries on 429 / transient errors.

    Args:
        local_dir: Directory to save downloaded parquet files.
        repo_id: HuggingFace dataset repository ID.
        token: HF API token. If None, loaded from .env / HF_TOKEN env var.

    Returns:
        List of paths to downloaded parquet files.
    """
    from huggingface_hub.errors import EntryNotFoundError

    resolved_token = _resolve_token(token)
    with start_action(action_type="hf:pull_cleaned_parquets", repo_id=repo_id):
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for filename in CLEANED_PARQUET_FILES:
            try:
                path = _hf_download_with_retry(
                    repo_id=repo_id,
                    filename=f"{HF_DATA_PREFIX}/metadata/{filename}",
                    repo_type="dataset",
                    local_dir=local_dir,
                    token=resolved_token,
                )
            except EntryNotFoundError:
                continue
            target = local_dir / filename
            hf_cached = Path(path)
            if hf_cached != target:
                shutil.copy2(hf_cached, target)
            downloaded.append(target)
        return downloaded
