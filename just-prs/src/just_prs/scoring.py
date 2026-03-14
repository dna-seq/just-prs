"""Download and parse PGS Catalog scoring files into Polars DataFrames.

Includes a transparent parquet cache: on first parse of a ``.txt.gz`` file,
a ``.parquet`` sibling is written with zstd-9 compression and the PGS Catalog
header metadata embedded as file-level metadata.  Subsequent reads hit the
parquet cache (5-60x faster, no gzip decompression).

Column types are enforced via ``SCORING_FILE_SCHEMA`` — a comprehensive type
map derived from the `PGS Catalog scoring file specification
<https://www.pgscatalog.org/downloads/#dl_ftp_scoring>`_.
"""

import gzip
import io
import json
import os
from pathlib import Path

import httpx
import polars as pl
from eliot import log_message, start_action
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

# ---------------------------------------------------------------------------
# PGS Catalog scoring file specification — canonical column type map
# https://www.pgscatalog.org/downloads/#dl_ftp_scoring
# ---------------------------------------------------------------------------

SCORING_FILE_SCHEMA: dict[str, pl.DataType] = {
    # -- Variant description (required + optional) --
    "rsID": pl.Utf8,
    "chr_name": pl.Utf8,
    "chr_position": pl.Int64,
    "effect_allele": pl.Utf8,
    "other_allele": pl.Utf8,
    "reference_allele": pl.Utf8,  # v1.0 legacy name for other_allele
    "locus_name": pl.Utf8,
    "is_haplotype": pl.Utf8,
    "is_diplotype": pl.Utf8,
    "imputation_method": pl.Utf8,
    "variant_description": pl.Utf8,
    "inclusion_criteria": pl.Utf8,
    # -- Weight information --
    "effect_weight": pl.Float64,
    "is_interaction": pl.Utf8,
    "is_dominant": pl.Utf8,
    "is_recessive": pl.Utf8,
    "dosage_0_weight": pl.Float64,
    "dosage_1_weight": pl.Float64,
    "dosage_2_weight": pl.Float64,
    # -- Other information --
    "OR": pl.Float64,
    "HR": pl.Float64,
    "allelefrequency_effect": pl.Float64,
    # -- Harmonized columns --
    "hm_source": pl.Utf8,
    "hm_rsID": pl.Utf8,
    "hm_chr": pl.Utf8,
    "hm_pos": pl.Int64,
    "hm_inferOtherAllele": pl.Utf8,
    "hm_match_chr": pl.Utf8,
    "hm_match_pos": pl.Utf8,
}

_PARQUET_COMPRESSION = "zstd"
_PARQUET_COMPRESSION_LEVEL = 9
_PGS_HEADER_METADATA_KEY = "pgs_catalog_header"


# ---------------------------------------------------------------------------
# Header metadata parsing
# ---------------------------------------------------------------------------

def _parse_scoring_header(header_lines: list[str]) -> dict[str, str]:
    """Parse ``#key=value`` lines from a PGS Catalog scoring file header.

    Skips section separators (lines starting with ``##`` or ``###``) that
    do not contain ``=``.

    Returns:
        Dict mapping lowercase keys (without ``#`` prefix) to their values,
        e.g. ``{"pgs_id": "PGS000001", "genome_build": "GRCh38", ...}``.
    """
    metadata: dict[str, str] = {}
    for line in header_lines:
        stripped = line.strip().lstrip("#")
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if key and value:
            metadata[key] = value
    return metadata


def read_scoring_header(path: Path) -> dict[str, str]:
    """Read PGS Catalog header metadata from a scoring file or its parquet cache.

    When *path* points to a ``.parquet`` file, reads the embedded
    ``pgs_catalog_header`` file-level metadata (instant, no data scan).
    Otherwise falls back to gzip-decompressing the ``.txt.gz`` and parsing
    the ``#`` header lines.

    Args:
        path: Path to a ``.txt.gz`` scoring file or its ``.parquet`` cache.

    Returns:
        Dict of header key-value pairs (e.g. ``pgs_id``, ``genome_build``).
    """
    if str(path).endswith(".parquet") and path.exists():
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        raw = pf.schema_arrow.metadata or {}
        json_bytes = raw.get(_PGS_HEADER_METADATA_KEY.encode("utf-8"), b"")
        if json_bytes:
            return json.loads(json_bytes)

    parquet_sibling = _scoring_parquet_cache_path(path)
    if parquet_sibling.exists() and not str(path).endswith(".parquet"):
        return read_scoring_header(parquet_sibling)

    with gzip.open(path, "rt") as f:
        header_lines = [line for line in f if line.startswith("#")]
    return _parse_scoring_header(header_lines)


# ---------------------------------------------------------------------------
# Parquet cache path helpers
# ---------------------------------------------------------------------------

def _scoring_parquet_cache_path(gz_path: Path) -> Path:
    """Return the ``.parquet`` sibling path for a given ``.txt.gz`` scoring file."""
    name = gz_path.name
    if name.endswith(".txt.gz"):
        return gz_path.parent / (name[: -len(".txt.gz")] + ".parquet")
    return gz_path.with_suffix(".parquet")


def scoring_parquet_path(
    pgs_id: str,
    cache_dir: Path,
    genome_build: str = "GRCh38",
) -> Path:
    """Compute the expected parquet cache path for a PGS ID.

    This does NOT check whether the file exists — it merely computes the path.

    Args:
        pgs_id: PGS Catalog score ID (e.g. ``"PGS000001"``).
        cache_dir: Scores cache directory (e.g. ``resolve_cache_dir() / "scores"``).
        genome_build: Genome build (``GRCh37`` or ``GRCh38``).

    Returns:
        Expected parquet path, e.g. ``cache_dir / "PGS000001_hmPOS_GRCh38.parquet"``.
    """
    return cache_dir / f"{pgs_id}_hmPOS_{genome_build}.parquet"


# ---------------------------------------------------------------------------
# Core parsing + caching
# ---------------------------------------------------------------------------

def _parse_gz_scoring_file(path: Path) -> tuple[pl.DataFrame, dict[str, str]]:
    """Parse a gzipped PGS Catalog scoring file into a DataFrame + header dict.

    Applies ``SCORING_FILE_SCHEMA`` overrides for all columns present in the
    file, ensuring correct types regardless of polars auto-inference.

    Returns:
        Tuple of (DataFrame, header_metadata_dict).
    """
    with gzip.open(path, "rt") as f:
        header_lines: list[str] = []
        data_lines: list[str] = []
        for line in f:
            if line.startswith("#"):
                header_lines.append(line)
            else:
                data_lines.append(line)

    header_metadata = _parse_scoring_header(header_lines)

    if not data_lines:
        raise ValueError(f"Scoring file {path} contains no data lines")

    tsv_content = "".join(data_lines)

    col_header_line = data_lines[0]
    present_cols = {c.strip() for c in col_header_line.split("\t")}
    overrides = {k: v for k, v in SCORING_FILE_SCHEMA.items() if k in present_cols}

    df = pl.read_csv(
        io.StringIO(tsv_content),
        separator="\t",
        infer_schema_length=10000,
        null_values=["", "NA", "None"],
        schema_overrides=overrides,
    )

    if "rsID" not in df.columns and "id" in df.columns:
        df = df.rename({"id": "rsID"})

    return df, header_metadata


def _write_scoring_parquet(
    df: pl.DataFrame,
    header_metadata: dict[str, str],
    parquet_path: Path,
) -> None:
    """Write a scoring DataFrame as parquet with PGS header metadata embedded."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = df.to_arrow()

    existing_meta = table.schema.metadata or {}
    existing_meta[_PGS_HEADER_METADATA_KEY.encode("utf-8")] = json.dumps(
        header_metadata
    ).encode("utf-8")
    table = table.replace_schema_metadata(existing_meta)

    pq.write_table(
        table,
        parquet_path,
        compression=_PARQUET_COMPRESSION,
        compression_level=_PARQUET_COMPRESSION_LEVEL,
    )


def parse_scoring_file(path: Path) -> pl.LazyFrame:
    """Parse a PGS Catalog scoring file into a LazyFrame, with transparent parquet caching.

    Accepts either a ``.txt.gz`` or ``.parquet`` path.  When given a ``.txt.gz``:

    1. If a ``.parquet`` sibling already exists, returns ``pl.scan_parquet()``
       (fast path, ~90ms even for 10M-variant files).
    2. Otherwise parses the gzipped TSV with spec-driven schema overrides,
       writes a zstd-9 parquet cache with PGS header metadata embedded, and
       returns ``pl.scan_parquet()`` on the new cache.

    When given a ``.parquet`` path directly, returns ``pl.scan_parquet()``.

    Args:
        path: Path to a ``.txt.gz`` scoring file or a ``.parquet`` cache.

    Returns:
        LazyFrame with columns from the scoring file, including at minimum:
        effect_allele (str), effect_weight (float), and positional columns.
    """
    if str(path).endswith(".parquet"):
        log_message(
            message_type="scoring:parse_from_parquet",
            path=str(path),
        )
        return pl.scan_parquet(path)

    parquet_cache = _scoring_parquet_cache_path(path)
    if parquet_cache.exists():
        log_message(
            message_type="scoring:parse_cache_hit",
            gz_path=str(path),
            parquet_path=str(parquet_cache),
        )
        return pl.scan_parquet(parquet_cache)

    with start_action(action_type="scoring:parse_and_cache", path=str(path)):
        df, header_metadata = _parse_gz_scoring_file(path)

        try:
            _write_scoring_parquet(df, header_metadata, parquet_cache)
            log_message(
                message_type="scoring:parquet_cache_written",
                gz_path=str(path),
                parquet_path=str(parquet_cache),
                rows=df.height,
                gz_bytes=path.stat().st_size,
                parquet_bytes=parquet_cache.stat().st_size,
            )
            return pl.scan_parquet(parquet_cache)
        except Exception as exc:
            log_message(
                message_type="scoring:parquet_cache_write_failed",
                gz_path=str(path),
                parquet_path=str(parquet_cache),
                error=str(exc),
            )
            if parquet_cache.exists():
                parquet_cache.unlink()
            return df.lazy()


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


def load_scoring(
    pgs_id: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    genome_build: str = "GRCh38",
) -> pl.LazyFrame:
    """Download (if needed) and parse a PGS scoring file, with local caching.

    Checks for a parquet cache first — if it exists, the ``.txt.gz`` download
    is skipped entirely.

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
        parquet = scoring_parquet_path(pgs_id, cache_dir, genome_build)
        if parquet.exists():
            log_message(
                message_type="scoring:load_from_parquet_cache",
                pgs_id=pgs_id,
                parquet_path=str(parquet),
            )
            return pl.scan_parquet(parquet)

        path = download_scoring_file(pgs_id, cache_dir, genome_build)
        return parse_scoring_file(path)
