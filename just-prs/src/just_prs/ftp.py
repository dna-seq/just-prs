"""Bulk download of PGS Catalog data via EBI FTP/HTTPS using fsspec."""

import gzip
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import fsspec
import polars as pl
from eliot import log_message, start_action

PGS_FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs"
PGS_SCORES_LIST_URL = f"{PGS_FTP_BASE}/pgs_scores_list.txt"
PGS_METADATA_BASE = f"{PGS_FTP_BASE}/metadata"

MetadataSheet = Literal[
    "scores",
    "publications",
    "efo_traits",
    "score_development_samples",
    "performance_metrics",
    "evaluation_sample_sets",
    "cohorts",
]

METADATA_FILES: dict[str, str] = {
    "scores": "pgs_all_metadata_scores.csv",
    "publications": "pgs_all_metadata_publications.csv",
    "efo_traits": "pgs_all_metadata_efo_traits.csv",
    "score_development_samples": "pgs_all_metadata_score_development_samples.csv",
    "performance_metrics": "pgs_all_metadata_performance_metrics.csv",
    "evaluation_sample_sets": "pgs_all_metadata_evaluation_sample_sets.csv",
    "cohorts": "pgs_all_metadata_cohorts.csv",
}



def list_all_pgs_ids() -> list[str]:
    """Fetch the complete list of PGS IDs from the EBI FTP in a single request.

    Returns:
        List of PGS ID strings (e.g. ['PGS000001', 'PGS000002', ...])
    """
    with start_action(action_type="ftp:list_all_pgs_ids"):
        with fsspec.open(PGS_SCORES_LIST_URL, "rt") as f:
            return [line.strip() for line in f if line.strip()]


def download_metadata_sheet(
    sheet: MetadataSheet,
    output_path: Path,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Download one PGS Catalog metadata CSV sheet and save it as parquet.

    These pre-built CSVs cover the entire catalog in a single HTTP request,
    making this far faster than paginating the REST API.

    Args:
        sheet: Which metadata sheet to download (e.g. 'scores', 'publications')
        output_path: Destination .parquet file path
        overwrite: If True, re-download and overwrite an existing parquet

    Returns:
        The loaded DataFrame
    """
    with start_action(action_type="ftp:download_metadata_sheet", sheet=sheet):
        if output_path.exists() and not overwrite:
            return pl.read_parquet(output_path)

        filename = METADATA_FILES[sheet]
        url = f"{PGS_METADATA_BASE}/{filename}"

        with fsspec.open(url, "rt") as f:
            df = pl.read_csv(f, infer_schema_length=10000, null_values=["", "NA", "None"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        return df


def download_all_metadata(
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, pl.DataFrame]:
    """Download all PGS Catalog metadata sheets and save each as a parquet file.

    Args:
        output_dir: Directory to write parquet files (one per sheet)
        overwrite: If True, re-download existing parquet files

    Returns:
        Dict mapping sheet name to its DataFrame
    """
    with start_action(action_type="ftp:download_all_metadata", output_dir=str(output_dir)):
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, pl.DataFrame] = {}
        for sheet in METADATA_FILES:
            output_path = output_dir / f"{sheet}.parquet"
            df = download_metadata_sheet(sheet, output_path, overwrite=overwrite)  # type: ignore[arg-type]
            results[sheet] = df
        return results


def _scoring_url(pgs_id: str, genome_build: str) -> str:
    """Build the HTTPS URL for a harmonized scoring file."""
    padded = pgs_id.upper()
    return (
        f"{PGS_FTP_BASE}/scores/{padded}/ScoringFiles/Harmonized/"
        f"{padded}_hmPOS_{genome_build}.txt.gz"
    )


def stream_scoring_file(
    pgs_id: str,
    genome_build: str = "GRCh38",
) -> pl.LazyFrame:
    """Stream a harmonized PGS scoring file directly from EBI FTP via fsspec.

    Reads the gzipped TSV without saving a local copy, skipping `#` comment
    header lines. Returns a LazyFrame for memory-efficient downstream use.

    Args:
        pgs_id: PGS Catalog score ID (e.g. 'PGS000001')
        genome_build: Genome build (GRCh37 or GRCh38)

    Returns:
        LazyFrame with scoring file columns
    """
    from just_prs.scoring import SCORING_FILE_SCHEMA

    with start_action(
        action_type="ftp:stream_scoring_file",
        pgs_id=pgs_id,
        genome_build=genome_build,
    ):
        url = _scoring_url(pgs_id, genome_build)
        with fsspec.open(url, "rb") as raw:
            with gzip.open(raw, "rt") as gz:
                data_lines = [line for line in gz if not line.startswith("#")]

        tsv_content = "".join(data_lines)

        col_header_line = data_lines[0] if data_lines else ""
        present_cols = {c.strip() for c in col_header_line.split("\t")}
        overrides = {k: v for k, v in SCORING_FILE_SCHEMA.items() if k in present_cols}

        df = pl.read_csv(
            io.StringIO(tsv_content),
            separator="\t",
            infer_schema_length=10000,
            null_values=["", "NA", "None"],
            schema_overrides=overrides,
        )
        return df.lazy()


def download_scoring_as_parquet(
    pgs_id: str,
    output_dir: Path,
    genome_build: str = "GRCh38",
    overwrite: bool = False,
) -> Path:
    """Stream a scoring file from EBI FTP and save it as parquet.

    Adds a `pgs_id` column so multiple scores can be concatenated later.

    Args:
        pgs_id: PGS Catalog score ID
        output_dir: Directory to write the parquet file
        genome_build: Genome build (GRCh37 or GRCh38)
        overwrite: If True, overwrite an existing parquet

    Returns:
        Path to the written parquet file
    """
    with start_action(
        action_type="ftp:download_scoring_as_parquet",
        pgs_id=pgs_id,
        genome_build=genome_build,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pgs_id}.parquet"
        if output_path.exists() and not overwrite:
            return output_path

        lf = stream_scoring_file(pgs_id, genome_build=genome_build)
        df = lf.with_columns(pl.lit(pgs_id).alias("pgs_id")).collect()
        df.write_parquet(output_path)
        return output_path


def bulk_download_scoring_parquets(
    output_dir: Path,
    genome_build: str = "GRCh38",
    pgs_ids: list[str] | None = None,
    overwrite: bool = False,
    progress_every: int = 100,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> list[Path]:
    """Download all (or a subset of) PGS scoring files as individual parquet files.

    When `pgs_ids` is None, the full list of IDs is fetched from
    `pgs_scores_list.txt` in a single request.

    Args:
        output_dir: Directory to write parquet files
        genome_build: Genome build (GRCh37 or GRCh38)
        pgs_ids: Explicit list of PGS IDs to download. If None, download all.
        overwrite: If True, overwrite existing parquet files
        progress_every: Log progress every N completed files.
        progress_callback: Optional callback for progress updates.

    Returns:
        List of paths to written (or already-existing) parquet files
    """
    with start_action(
        action_type="ftp:bulk_download_scoring_parquets",
        genome_build=genome_build,
    ):
        ids = pgs_ids if pgs_ids is not None else list_all_pgs_ids()
        written: list[Path] = []
        total = len(ids)
        
        import time
        last_log_time = time.monotonic()
        
        for i, pgs_id in enumerate(ids):
            path = download_scoring_as_parquet(
                pgs_id,
                output_dir=output_dir,
                genome_build=genome_build,
                overwrite=overwrite,
            )
            written.append(path)
            
            completed = i + 1
            current_time = time.monotonic()
            should_log = (completed % progress_every == 0) or (current_time - last_log_time > 15.0) or (completed == total)
            
            if should_log:
                last_log_time = current_time
                log_message(
                    message_type="ftp:bulk_download_parquets_progress",
                    completed=completed,
                    total=total,
                )
                if progress_callback is not None:
                    progress_callback({
                        "completed": completed,
                        "total": total,
                    })
                    
        return written


_DEFAULT_DOWNLOAD_WORKERS = 4
_DEFAULT_DOWNLOAD_PROGRESS_EVERY = 100


@dataclass
class BulkDownloadResult:
    """Summary of a bulk scoring file download."""
    total: int = 0
    downloaded: int = 0
    cached: int = 0
    parquet_cached: int = 0
    failed: int = 0
    failed_ids: list[str] = field(default_factory=list)


def _download_one_scoring_file(
    pgs_id: str,
    output_dir: Path,
    genome_build: str,
) -> tuple[str, str]:
    """Download a single scoring .txt.gz file via fsspec. Returns (pgs_id, status).

    Status is one of: "cached" (.txt.gz exists), "parquet_cached" (parquet
    cache exists so .txt.gz is not needed), "downloaded", or "failed".
    """
    filename = f"{pgs_id}_hmPOS_{genome_build}.txt.gz"
    output_path = output_dir / filename
    if output_path.exists():
        return pgs_id, "cached"

    parquet_path = output_dir / f"{pgs_id}_hmPOS_{genome_build}.parquet"
    if parquet_path.exists():
        return pgs_id, "parquet_cached"

    url = _scoring_url(pgs_id, genome_build)
    tmp_path = output_path.with_suffix(".tmp")
    try:
        with fsspec.open(url, "rb") as remote:
            with tmp_path.open("wb") as local:
                while True:
                    chunk = remote.read(65536)
                    if not chunk:
                        break
                    local.write(chunk)
        tmp_path.rename(output_path)
        return pgs_id, "downloaded"
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        return pgs_id, "failed"


def bulk_download_scoring_files(
    pgs_ids: list[str],
    output_dir: Path,
    genome_build: str = "GRCh38",
    max_workers: int | None = None,
    progress_every: int | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> BulkDownloadResult:
    """Bulk-download harmonized PGS scoring .txt.gz files via fsspec.

    Uses direct HTTPS URLs to the EBI FTP server (no REST API calls).
    Downloads concurrently with a configurable number of workers and
    skips files that already exist on disk.

    Concurrency and progress frequency can be set via env vars
    ``PRS_DOWNLOAD_WORKERS`` and ``PRS_DOWNLOAD_PROGRESS_EVERY``.

    Args:
        pgs_ids: List of PGS IDs to download.
        output_dir: Directory to save the .txt.gz files.
        genome_build: Genome build (GRCh37 or GRCh38).
        max_workers: Number of concurrent download threads.
            Defaults to ``PRS_DOWNLOAD_WORKERS`` env var or 4.
        progress_every: Log progress every N completed files.
            Defaults to ``PRS_DOWNLOAD_PROGRESS_EVERY`` env var or 100.

    Returns:
        BulkDownloadResult with counts of downloaded, cached, and failed files.
    """
    if max_workers is None:
        env_val = os.environ.get("PRS_DOWNLOAD_WORKERS", "").strip()
        max_workers = int(env_val) if env_val else _DEFAULT_DOWNLOAD_WORKERS
    if progress_every is None:
        env_val = os.environ.get("PRS_DOWNLOAD_PROGRESS_EVERY", "").strip()
        progress_every = int(env_val) if env_val else _DEFAULT_DOWNLOAD_PROGRESS_EVERY

    output_dir.mkdir(parents=True, exist_ok=True)
    result = BulkDownloadResult(total=len(pgs_ids))

    with start_action(
        action_type="ftp:bulk_download_scoring_files",
        genome_build=genome_build,
        n_ids=len(pgs_ids),
        max_workers=max_workers,
    ):
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_download_one_scoring_file, pgs_id, output_dir, genome_build): pgs_id
                for pgs_id in pgs_ids
            }
            import time
            last_log_time = time.monotonic()
            
            for future in as_completed(futures):
                pgs_id = futures[future]
                try:
                    _, status = future.result()
                except Exception:
                    status = "failed"
                if status == "cached":
                    result.cached += 1
                elif status == "parquet_cached":
                    result.parquet_cached += 1
                elif status == "downloaded":
                    result.downloaded += 1
                else:
                    result.failed += 1
                    result.failed_ids.append(pgs_id)

                completed += 1
                current_time = time.monotonic()
                time_elapsed = current_time - last_log_time
                
                # Log if we hit the file count threshold OR 15 seconds have passed
                should_log = (completed % progress_every == 0) or (time_elapsed > 15.0) or (completed == len(pgs_ids))
                
                if should_log:
                    last_log_time = current_time
                    log_message(
                        message_type="ftp:bulk_download_progress",
                        completed=completed,
                        total=len(pgs_ids),
                        downloaded=result.downloaded,
                        cached=result.cached,
                        parquet_cached=result.parquet_cached,
                        failed=result.failed,
                    )
                    if progress_callback is not None:
                        progress_callback({
                            "completed": completed,
                            "total": len(pgs_ids),
                            "downloaded": result.downloaded,
                            "cached": result.cached,
                            "parquet_cached": result.parquet_cached,
                            "failed": result.failed,
                        })

    return result
