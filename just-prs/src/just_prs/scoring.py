"""Download and parse PGS Catalog scoring files into Polars DataFrames."""

import gzip
import io
import os
from pathlib import Path

import httpx
import polars as pl
from eliot import start_action
from platformdirs import user_cache_dir

from just_prs.catalog import PGSCatalogClient

_APP_CACHE_DIR = Path(user_cache_dir("just-prs", appauthor=False))


def resolve_cache_dir() -> Path:
    """Return the root cache directory, respecting PRS_CACHE_DIR env var.

    Resolution order: PRS_CACHE_DIR env var > platformdirs user cache dir.
    On Linux: ``~/.cache/just-prs``, macOS: ``~/Library/Caches/just-prs``,
    Windows: ``%LOCALAPPDATA%/just-prs/Cache``.
    """
    raw = os.environ.get("PRS_CACHE_DIR", "")
    if raw:
        return Path(raw)
    return _APP_CACHE_DIR


DEFAULT_CACHE_DIR = resolve_cache_dir() / "scores"


def download_scoring_file(
    pgs_id: str,
    output_dir: Path,
    genome_build: str = "GRCh38",
) -> Path:
    """Download a harmonized PGS scoring file from the EBI FTP server.

    Args:
        pgs_id: PGS Catalog score ID (e.g. "PGS000001")
        output_dir: Directory to save the downloaded file
        genome_build: Genome build (GRCh37 or GRCh38)

    Returns:
        Path to the downloaded gzipped scoring file
    """
    with start_action(
        action_type="scoring:download",
        pgs_id=pgs_id,
        genome_build=genome_build,
        output_dir=str(output_dir),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{pgs_id}_hmPOS_{genome_build}.txt.gz"
        output_path = output_dir / filename

        if output_path.exists():
            return output_path

        with PGSCatalogClient() as client:
            url = client.get_score_download_url(pgs_id, genome_build)

        with httpx.Client(timeout=120.0, follow_redirects=True) as http:
            with http.stream("GET", url) as response:
                response.raise_for_status()
                with output_path.open("wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)

        return output_path


def parse_scoring_file(path: Path) -> pl.LazyFrame:
    """Parse a PGS Catalog scoring file (gzipped TSV with comment header) into a LazyFrame.

    Skips lines starting with '#' (metadata header). Reads the tab-delimited data
    section and casts columns to appropriate types.

    Args:
        path: Path to the gzipped scoring file (.txt.gz)

    Returns:
        LazyFrame with columns from the scoring file, including at minimum:
        chr_name (str), chr_position (int), effect_allele (str), effect_weight (float)
    """
    with start_action(action_type="scoring:parse", path=str(path)):
        with gzip.open(path, "rt") as f:
            header_lines: list[str] = []
            data_lines: list[str] = []
            for line in f:
                if line.startswith("#"):
                    header_lines.append(line)
                else:
                    data_lines.append(line)

        tsv_content = "".join(data_lines)
        df = pl.read_csv(
            io.StringIO(tsv_content),
            separator="\t",
            infer_schema_length=10000,
            null_values=["", "NA", "None"],
        )

        columns = df.columns
        if "rsID" not in columns and "id" in columns:
            df = df.rename({"id": "rsID"})
            columns = df.columns

        cast_map: dict[str, pl.DataType] = {}
        if "chr_name" in columns:
            cast_map["chr_name"] = pl.Utf8
        if "chr_position" in columns:
            cast_map["chr_position"] = pl.Int64
        if "effect_weight" in columns:
            cast_map["effect_weight"] = pl.Float64
        if "effect_allele" in columns:
            cast_map["effect_allele"] = pl.Utf8
        if "other_allele" in columns:
            cast_map["other_allele"] = pl.Utf8
        if "rsID" in columns:
            cast_map["rsID"] = pl.Utf8
        if "hm_chr" in columns:
            cast_map["hm_chr"] = pl.Utf8
        if "hm_pos" in columns:
            cast_map["hm_pos"] = pl.Int64
        if "allelefrequency_effect" in columns:
            cast_map["allelefrequency_effect"] = pl.Float64

        if cast_map:
            df = df.cast(cast_map)

        return df.lazy()


def load_scoring(
    pgs_id: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    genome_build: str = "GRCh38",
) -> pl.LazyFrame:
    """Download (if needed) and parse a PGS scoring file, with local caching.

    Args:
        pgs_id: PGS Catalog score ID
        cache_dir: Directory for caching downloaded files
        genome_build: Genome build (GRCh37 or GRCh38)

    Returns:
        LazyFrame with scoring file data
    """
    with start_action(
        action_type="scoring:load",
        pgs_id=pgs_id,
        genome_build=genome_build,
    ):
        path = download_scoring_file(pgs_id, cache_dir, genome_build)
        return parse_scoring_file(path)


