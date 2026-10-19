"""GWAS summary statistics: fetch harmonised files from the GWAS Catalog and normalize them.

The GWAS Catalog publishes full summary statistics for ~60,000 studies and
re-harmonises most of them to GRCh38 (``harmonised/*.h.tsv.gz``). Those files
come in two layouts that disagree on column names and orientation:

* **GWAS-SSF** (current): ``chromosome``, ``base_pair_location``,
  ``effect_allele``, ``other_allele``, ``beta``/``odds_ratio``/``hazard_ratio``,
  ``standard_error``, ``p_value``, ``rsid``, ``hm_code``. The main columns
  already hold the harmonised (GRCh38, reference-strand) values.
* **Legacy harmonised**: ``hm_chrom``, ``hm_pos``, ``hm_effect_allele``,
  ``hm_other_allele``, ``hm_beta``, ``hm_odds_ratio`` … alongside the
  original, un-harmonised columns. Only the ``hm_*`` columns are aligned.

:func:`normalize_gwas_sumstats` maps both onto one schema
(:data:`NORMALIZED_SUMSTATS_COLUMNS`) with a single DuckDB streaming pass, so a
multi-GB file never decompresses into Python memory. It also repairs the
recurring quirks seen in real deposits: a quantitative-trait beta filed under
``odds_ratio`` (negative "odds ratios"), a standard error that only survives
inside ``additional_info`` (``REGENIE_SE=``), or only as a confidence interval.

Normalization is lossless: every input row is kept, with ``hm_code`` and
``palindromic`` so downstream builders choose their own filters.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import duckdb
import httpx
import polars as pl
import yaml
from eliot import log_message, start_action
from pydantic import BaseModel

from just_prs.scoring import resolve_cache_dir

GWAS_CATALOG_SUMSTATS_BASE = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
GWAS_CATALOG_STUDY_API = "https://www.ebi.ac.uk/gwas/rest/api/studies/{gcst}"

#: Schema written by :func:`normalize_gwas_sumstats`. ``beta`` is always on the
#: additive/log scale (log-OR / log-HR for ratio statistics); ``chrom``/``pos``
#: are the harmonised GRCh38 coordinates (null when the variant could not be
#: harmonised).
NORMALIZED_SUMSTATS_COLUMNS: tuple[str, ...] = (
    "chrom", "pos", "rsid", "effect_allele", "other_allele", "beta", "se", "p",
    "eaf", "n", "info", "hm_code", "palindromic",
)

#: GWAS Catalog harmonisation codes for strand-ambiguous (A/T, C/G) variants,
#: whose orientation was inferred or assumed rather than observed.
PALINDROMIC_HM_CODES: frozenset[int] = frozenset(range(1, 10))

_Z_95 = 1.959963984540054
_HTTP_RETRIES = 8
_HTTP_BACKOFF = 5.0
_DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, read=300.0)


class SumstatsLayout(StrEnum):
    """Column layout of a GWAS Catalog harmonised summary-statistics file."""

    GWAS_SSF = "gwas_ssf"
    LEGACY_HARMONISED = "legacy_harmonised"


class GwasSumstatsSource(BaseModel):
    """A downloaded GWAS Catalog summary-statistics file plus its study metadata."""

    gcst: str
    url: str
    path: Path
    meta: dict[str, Any] = {}
    sample_size: int | None = None
    trait: str | None = None
    ancestry: list[str] = []
    genome_assembly: str | None = None


class SumstatsNormalization(BaseModel):
    """Row accounting for one :func:`normalize_gwas_sumstats` run."""

    source: Path
    output: Path
    layout: SumstatsLayout
    beta_source: str
    se_source: str
    n_rows: int
    n_harmonised: int
    n_palindromic: int
    n_missing_beta_or_se: int
    median_n: float | None = None


# ---------------------------------------------------------------------------
# GWAS Catalog discovery + download
# ---------------------------------------------------------------------------


def gwas_catalog_study_url(gcst: str) -> str:
    """Return the FTP-over-HTTPS directory of a GWAS Catalog study.

    Studies are bucketed in ranges of 1,000 accessions whose zero padding
    follows the accession width (``GCST006001-GCST007000`` vs
    ``GCST90165001-GCST90166000``).
    """
    match = re.fullmatch(r"GCST(\d+)", gcst.strip().upper())
    if match is None:
        raise ValueError(f"Not a GWAS Catalog study accession: {gcst!r}")
    digits = match.group(1)
    number = int(digits)
    low = (number - 1) // 1000 * 1000 + 1
    high = low + 999
    width = len(digits)
    bucket = f"GCST{low:0{width}d}-GCST{high:0{width}d}"
    return f"{GWAS_CATALOG_SUMSTATS_BASE}/{bucket}/GCST{digits}"


def _http_get(client: httpx.Client, url: str) -> httpx.Response:
    """GET with retry on DNS/connection failures, timeouts and 429/5xx."""
    last: Exception | None = None
    for attempt in range(_HTTP_RETRIES):
        try:
            response = client.get(url)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise httpx.HTTPStatusError(
                    f"{response.status_code} for {url}", request=response.request, response=response
                )
            response.raise_for_status()
            return response
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
                raise
            last = exc
            time.sleep(_HTTP_BACKOFF * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {_HTTP_RETRIES} attempts: {last}")


def find_harmonised_sumstats_url(gcst: str, client: httpx.Client | None = None) -> str:
    """Resolve the harmonised ``*.h.tsv.gz`` file of a study.

    File names differ between eras (``GCST007406.h.tsv.gz`` vs
    ``29227965-GCST006697-EFO_0007796.h.tsv.gz``), so the ``harmonised/``
    directory listing is the source of truth.
    """
    directory = f"{gwas_catalog_study_url(gcst)}/harmonised/"
    own = client is None
    http = client or httpx.Client(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True)
    try:
        listing = _http_get(http, directory).text
    finally:
        if own:
            http.close()
    names = sorted(set(re.findall(r'href="([^"/?]+\.h\.tsv\.gz)"', listing)))
    if not names:
        raise FileNotFoundError(
            f"No harmonised summary statistics for {gcst} at {directory}. "
            "The study may only have author-formatted files; check the directory manually."
        )
    return directory + names[0]


def _study_meta(http: httpx.Client, data_url: str) -> dict[str, Any]:
    """Parse the ``-meta.yaml`` sidecar (missing on some legacy deposits)."""
    try:
        text = _http_get(http, data_url + "-meta.yaml").text
    except httpx.HTTPStatusError:
        return {}
    parsed = yaml.safe_load(text)
    return parsed if isinstance(parsed, dict) else {}


def _sample_size_from_text(text: str | None) -> int | None:
    """Sum every integer in a free-text sample description (``"389,166 British …"``)."""
    if not text:
        return None
    total = sum(int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text))
    return total or None


def _study_api(http: httpx.Client, gcst: str) -> dict[str, Any]:
    try:
        return _http_get(http, GWAS_CATALOG_STUDY_API.format(gcst=gcst)).json()
    except (httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
        log_message(message_type="sumstats:study_api_unavailable", gcst=gcst, error=str(exc))
        return {}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_download(http: httpx.Client, url: str, dest: Path) -> None:
    """Resumable streaming download to ``dest`` via a ``.part`` file + atomic rename."""
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(_HTTP_RETRIES):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with http.stream("GET", url, headers=headers) as response:
                if response.status_code == 416:
                    break
                response.raise_for_status()
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with part.open(mode) as handle:
                    for chunk in response.iter_bytes(chunk_size=1 << 20):
                        handle.write(chunk)
            break
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError, httpx.ReadError) as exc:
            last = exc
            log_message(message_type="sumstats:download_retry", url=url, attempt=attempt + 1, error=str(exc))
            time.sleep(_HTTP_BACKOFF * (attempt + 1))
    else:
        raise RuntimeError(f"Download of {url} failed after {_HTTP_RETRIES} attempts: {last}")
    if not part.exists() or part.stat().st_size == 0:
        raise RuntimeError(f"Download of {url} produced an empty file")
    os.replace(part, dest)


def download_gwas_sumstats(
    gcst: str,
    dest_dir: Path | None = None,
    force: bool = False,
) -> GwasSumstatsSource:
    """Download a study's harmonised GRCh38 summary statistics and its metadata.

    The file is verified against the ``data_file_md5sum`` in the study's
    ``-meta.yaml`` when one is published. An existing file is reused unless it
    fails that check or ``force`` is set.

    Sample size is taken, in order, from the meta sidecar's ``samples`` block and
    from the GWAS Catalog REST ``initialSampleSize`` text. It can still be
    ``None``; :func:`normalize_gwas_sumstats` reports a per-variant median when
    the file has an ``n`` column.

    Args:
        gcst: GWAS Catalog study accession, e.g. ``"GCST007406"``.
        dest_dir: Target directory (default ``<cache>/sumstats/gwas_catalog``).
        force: Re-download even when a verified copy exists.
    """
    gcst = gcst.strip().upper()
    dest_dir = dest_dir or resolve_cache_dir() / "sumstats" / "gwas_catalog"
    dest_dir.mkdir(parents=True, exist_ok=True)
    with start_action(action_type="sumstats:download", gcst=gcst) as action, httpx.Client(
        timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True
    ) as http:
        url = find_harmonised_sumstats_url(gcst, http)
        dest = dest_dir / url.rsplit("/", 1)[-1]
        meta = _study_meta(http, url)
        expected_md5 = meta.get("data_file_md5sum")

        if force:
            dest.unlink(missing_ok=True)
        if dest.exists() and expected_md5 and _md5(dest) != expected_md5:
            log_message(message_type="sumstats:md5_mismatch_refetch", path=str(dest))
            dest.unlink()
        if not dest.exists():
            _stream_download(http, url, dest)
            if expected_md5 and _md5(dest) != expected_md5:
                dest.unlink()
                raise RuntimeError(f"MD5 mismatch for {url}; the partial download was discarded")

        samples = meta.get("samples") or []
        sample_size = sum(int(s.get("sample_size") or 0) for s in samples if isinstance(s, dict)) or None
        ancestry = sorted(
            {a for s in samples if isinstance(s, dict) for a in (s.get("sample_ancestry_category") or [])}
        )
        api = _study_api(http, gcst)
        if sample_size is None:
            sample_size = _sample_size_from_text(api.get("initialSampleSize"))
        trait = (api.get("diseaseTrait") or {}).get("trait") or (meta.get("trait_description") or [None])[0]
        source = GwasSumstatsSource(
            gcst=gcst,
            url=url,
            path=dest,
            meta=meta,
            sample_size=sample_size,
            trait=trait,
            ancestry=ancestry,
            genome_assembly=meta.get("genome_assembly"),
        )
        action.add_success_fields(path=str(dest), sample_size=sample_size, trait=trait)
        return source


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _read_header(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return handle.readline().rstrip("\n").split("\t")


def detect_sumstats_layout(columns: list[str]) -> SumstatsLayout:
    """Classify a harmonised file by its header."""
    cols = set(columns)
    if {"hm_chrom", "hm_pos", "hm_effect_allele"} <= cols:
        return SumstatsLayout.LEGACY_HARMONISED
    if {"chromosome", "base_pair_location", "effect_allele"} <= cols:
        return SumstatsLayout.GWAS_SSF
    raise ValueError(f"Unrecognised summary-statistics header: {columns[:12]}")


def _num(col: str) -> str:
    return f'TRY_CAST("{col}" AS DOUBLE)'


def _duckdb_connect(memory_limit: str | None) -> duckdb.DuckDBPyConnection:
    from just_prs.prs import _resolve_duckdb_memory_limit

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit or _resolve_duckdb_memory_limit()}'")
    con.execute("SET preserve_insertion_order = false")
    return con


def normalize_gwas_sumstats(
    source: Path,
    dest: Path | None = None,
    memory_limit: str | None = None,
    force: bool = False,
) -> SumstatsNormalization:
    """Rewrite a GWAS Catalog harmonised file into the normalized parquet schema.

    One streaming DuckDB pass reads the gzip TSV and writes zstd parquet, so
    peak memory stays bounded by ``memory_limit`` (default: the shared
    ``PRS_DUCKDB_MEMORY_LIMIT`` / percent-of-RAM policy) even for multi-GB files.

    Effect sizes are resolved per layout: ``beta`` when present; otherwise the
    ratio column on the log scale. A ratio column containing negative values is
    a mislabelled beta (seen in REGENIE deposits) and is used as-is. Standard
    errors fall back to ``REGENIE_SE=`` in ``additional_info`` and then to the
    95% confidence interval width.

    Args:
        source: A harmonised ``*.h.tsv.gz`` file.
        dest: Output parquet (default ``<cache>/sumstats/normalized/<stem>.parquet``).
        memory_limit: DuckDB memory cap, e.g. ``"4GB"``.
        force: Rebuild even when ``dest`` is newer than ``source``.
    """
    stem = source.name.removesuffix(".gz").removesuffix(".tsv").removesuffix(".h")
    dest = dest or resolve_cache_dir() / "sumstats" / "normalized" / f"{stem}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    columns = _read_header(source)
    layout = detect_sumstats_layout(columns)
    cols = set(columns)
    prefix = "hm_" if layout is SumstatsLayout.LEGACY_HARMONISED else ""

    def pick(name: str) -> str | None:
        for candidate in (f"{prefix}{name}", name):
            if candidate in cols:
                return candidate
        return None

    con = _duckdb_connect(memory_limit)
    try:
        reader = (
            "read_csv(?, delim='\t', header=true, all_varchar=true, quote='', "
            "nullstr=['NA', '', 'NaN', 'nan', '#NA'])"
        )
        if layout is SumstatsLayout.LEGACY_HARMONISED:
            chrom, pos, ea, oa = "hm_chrom", "hm_pos", "hm_effect_allele", "hm_other_allele"
            rsid = "COALESCE(\"hm_rsid\", CASE WHEN \"variant_id\" LIKE 'rs%' THEN \"variant_id\" END)"
        else:
            chrom, pos, ea, oa = "chromosome", "base_pair_location", "effect_allele", "other_allele"
            rsid_col = "rsid" if "rsid" in cols else None
            rsid = (
                f"COALESCE(\"{rsid_col}\", CASE WHEN \"variant_id\" LIKE 'rs%' THEN \"variant_id\" END)"
                if rsid_col
                else "CASE WHEN \"variant_id\" LIKE 'rs%' THEN \"variant_id\" END"
            )

        beta_col, or_col, hr_col = pick("beta"), pick("odds_ratio"), pick("hazard_ratio")
        ratio_col = or_col or hr_col
        ratio_is_beta = False
        if ratio_col is not None:
            negatives = con.execute(
                f"SELECT count(*) FILTER (WHERE {_num(ratio_col)} < 0) "
                f"FROM (SELECT \"{ratio_col}\" FROM {reader} LIMIT 200000)",
                [str(source)],
            ).fetchone()[0]
            ratio_is_beta = negatives > 0

        beta_parts: list[str] = []
        if beta_col:
            beta_parts.append(_num(beta_col))
        if ratio_col:
            beta_parts.append(_num(ratio_col) if ratio_is_beta else f"ln(NULLIF({_num(ratio_col)}, 0))")
        if not beta_parts:
            raise ValueError(f"{source}: no beta, odds_ratio or hazard_ratio column")
        beta_expr = f"COALESCE({', '.join(beta_parts)})" if len(beta_parts) > 1 else beta_parts[0]
        beta_source = beta_col or (f"{ratio_col} (mislabelled beta)" if ratio_is_beta else f"ln({ratio_col})")

        se_parts: list[str] = []
        se_sources: list[str] = []
        if "standard_error" in cols:
            se_parts.append(_num("standard_error"))
            se_sources.append("standard_error")
        if "additional_info" in cols:
            se_parts.append(
                "TRY_CAST(NULLIF(regexp_extract(\"additional_info\", 'REGENIE_SE=([0-9.eE+-]+)', 1), '') AS DOUBLE)"
            )
            se_sources.append("additional_info REGENIE_SE")
        ci_hi, ci_lo = pick("ci_upper"), pick("ci_lower")
        if ci_hi and ci_lo:
            if ratio_col and not beta_col and not ratio_is_beta:
                ci = f"(ln(NULLIF({_num(ci_hi)}, 0)) - ln(NULLIF({_num(ci_lo)}, 0)))"
            else:
                ci = f"({_num(ci_hi)} - {_num(ci_lo)})"
            se_parts.append(f"{ci} / {2 * _Z_95}")
            se_sources.append("95% CI width")
        se_expr = f"COALESCE({', '.join(se_parts)})" if se_parts else "NULL::DOUBLE"

        if "p_value" in cols:
            p_expr = _num("p_value")
        elif "neg_log_10_p_value" in cols:
            p_expr = f"pow(10, -{_num('neg_log_10_p_value')})"
        else:
            p_expr = "NULL::DOUBLE"
        eaf_col = pick("effect_allele_frequency")
        eaf_expr = _num(eaf_col) if eaf_col else "NULL::DOUBLE"
        n_expr = _num("n") if "n" in cols else "NULL::DOUBLE"
        if "info" in cols:
            info_expr = _num("info")
        elif "additional_info" in cols:
            info_expr = "TRY_CAST(NULLIF(regexp_extract(\"additional_info\", '(?:^|;)INFO=([0-9.eE+-]+)', 1), '') AS DOUBLE)"
        else:
            info_expr = "NULL::DOUBLE"
        hm_expr = 'TRY_CAST("hm_code" AS INTEGER)' if "hm_code" in cols else "NULL::INTEGER"

        if dest.exists() and not force and dest.stat().st_mtime >= source.stat().st_mtime:
            log_message(message_type="sumstats:normalized_cache_hit", path=str(dest))
        else:
            tmp = dest.with_name(dest.name + ".tmp")
            with start_action(action_type="sumstats:normalize", source=str(source), layout=layout.value):
                con.execute(
                    f"""
                    COPY (
                        SELECT
                            CASE WHEN c = '23' THEN 'X' ELSE c END AS chrom,
                            TRY_CAST("{pos}" AS BIGINT) AS pos,
                            {rsid} AS rsid,
                            upper("{ea}") AS effect_allele,
                            upper("{oa}") AS other_allele,
                            {beta_expr} AS beta,
                            {se_expr} AS se,
                            {p_expr} AS p,
                            {eaf_expr} AS eaf,
                            {n_expr} AS n,
                            {info_expr} AS info,
                            {hm_expr} AS hm_code,
                            (upper("{ea}") || upper("{oa}")) IN ('AT', 'TA', 'CG', 'GC') AS palindromic
                        FROM (SELECT *, regexp_replace("{chrom}", '^chr', '') AS c FROM {reader})
                    ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)
                    """,
                    [str(source)],
                )
            os.replace(tmp, dest)

        stats = con.execute(
            """
            SELECT count(*),
                   count(*) FILTER (WHERE pos IS NOT NULL),
                   count(*) FILTER (WHERE palindromic),
                   count(*) FILTER (WHERE beta IS NULL OR se IS NULL OR se <= 0),
                   median(n)
            FROM read_parquet(?)
            """,
            [str(dest)],
        ).fetchone()
    finally:
        con.close()

    result = SumstatsNormalization(
        source=source,
        output=dest,
        layout=layout,
        beta_source=beta_source,
        se_source=" > ".join(se_sources) or "none",
        n_rows=stats[0],
        n_harmonised=stats[1],
        n_palindromic=stats[2],
        n_missing_beta_or_se=stats[3],
        median_n=stats[4],
    )
    log_message(message_type="sumstats:normalized", **result.model_dump(mode="json"))
    return result


def scan_normalized_sumstats(path: Path) -> pl.LazyFrame:
    """Lazily open a normalized summary-statistics parquet."""
    return pl.scan_parquet(path)
