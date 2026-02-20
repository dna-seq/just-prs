"""Bulk download of PGS Catalog data via EBI FTP/HTTPS using fsspec."""

import gzip
import io
from pathlib import Path
from typing import Literal

import fsspec
import polars as pl
from eliot import start_action

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
        df = pl.read_csv(
            io.StringIO(tsv_content),
            separator="\t",
            infer_schema_length=10000,
            null_values=["", "NA", "None"],
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
) -> list[Path]:
    """Download all (or a subset of) PGS scoring files as individual parquet files.

    When `pgs_ids` is None, the full list of IDs is fetched from
    `pgs_scores_list.txt` in a single request.

    Args:
        output_dir: Directory to write parquet files
        genome_build: Genome build (GRCh37 or GRCh38)
        pgs_ids: Explicit list of PGS IDs to download. If None, download all.
        overwrite: If True, overwrite existing parquet files

    Returns:
        List of paths to written (or already-existing) parquet files
    """
    with start_action(
        action_type="ftp:bulk_download_scoring_parquets",
        genome_build=genome_build,
    ):
        ids = pgs_ids if pgs_ids is not None else list_all_pgs_ids()
        written: list[Path] = []
        for pgs_id in ids:
            path = download_scoring_as_parquet(
                pgs_id,
                output_dir=output_dir,
                genome_build=genome_build,
                overwrite=overwrite,
            )
            written.append(path)
        return written
