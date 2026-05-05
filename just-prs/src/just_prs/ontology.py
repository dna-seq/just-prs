"""Ontology helpers for trait-level risk metadata.

PGS Catalog traits may be keyed by EFO, MONDO, OBA, HP, or other ontology
prefixes, while prevalence and heritability sources are often EFO-centric.
This module builds explicit alias rows so downstream risk lookup can stay
data-driven and offline after metadata has been published.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import polars as pl

logger = logging.getLogger(__name__)

ONTOLOGY_ALIAS_COLUMNS: dict[str, pl.DataType] = {
    "canonical_efo_id": pl.Utf8,
    "mapped_from_id": pl.Utf8,
    "mapping_source": pl.Utf8,
}

SUPPORTED_ALIAS_PREFIXES: tuple[str, ...] = (
    "EFO",
    "MONDO",
    "OBA",
    "HP",
    "Orphanet",
    "NCIT",
    "MP",
)

_ONTOLOGY_ID_RE = re.compile(
    r"\b(EFO|MONDO|OBA|HP|Orphanet|NCIT|MP)[:_](\d+)\b",
    flags=re.IGNORECASE,
)
_ICD10_RE = re.compile(r"\bICD(?:10|10CM|10WHO|[- ]10)[:/]?([A-Z]\d{2}(?:\.\d+)?)\b", flags=re.IGNORECASE)


def normalize_trait_id(value: object) -> str | None:
    """Normalize ontology identifiers to the underscore form used in local parquets."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text or "#" in text:
        text = text.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    match = _ONTOLOGY_ID_RE.search(text)
    if match is None:
        return text.replace(":", "_")
    prefix = match.group(1)
    number = match.group(2)
    canonical_prefix = next(
        (p for p in SUPPORTED_ALIAS_PREFIXES if p.lower() == prefix.lower()),
        prefix,
    )
    return f"{canonical_prefix}_{number}"


def colon_trait_id(value: str) -> str:
    """Return a colon-form identifier for APIs that prefer CURIE syntax."""
    normalized = normalize_trait_id(value) or value
    if "_" not in normalized:
        return normalized
    prefix, suffix = normalized.split("_", 1)
    return f"{prefix}:{suffix}"


def extract_ontology_ids(value: object) -> list[str]:
    """Extract normalized ontology IDs from arbitrary OLS annotation values."""
    if value is None:
        return []
    text = str(value)
    ids: list[str] = []
    for match in _ONTOLOGY_ID_RE.finditer(text):
        normalized = normalize_trait_id(match.group(0))
        if normalized is not None and normalized not in ids:
            ids.append(normalized)
    return ids


def _ols_cache_file(cache_dir: Path, efo_id: str) -> Path:
    return cache_dir / f"{normalize_trait_id(efo_id) or efo_id}.json"


def trait_iri(value: str) -> str | None:
    """Return the common OLS IRI for a normalized ontology identifier."""
    normalized = normalize_trait_id(value)
    if normalized is None or "_" not in normalized:
        return None
    prefix, suffix = normalized.split("_", 1)
    if prefix == "EFO":
        return f"http://www.ebi.ac.uk/efo/{normalized}"
    if prefix in {"MONDO", "OBA", "HP", "MP", "NCIT"}:
        return f"http://purl.obolibrary.org/obo/{normalized}"
    if prefix == "Orphanet":
        return f"http://www.orpha.net/ORDO/{normalized}"
    return None


def query_ols_trait_xrefs(trait_id: str) -> dict[str, Any]:
    """Query OLS4 by IRI and return labels, ontology aliases, and ICD10 xrefs."""
    import httpx

    iri = trait_iri(trait_id)
    normalized = normalize_trait_id(trait_id)
    result: dict[str, Any] = {
        "trait_id": normalized,
        "label": None,
        "synonyms": [],
        "aliases": [],
        "icd10_codes": [],
    }
    if iri is None:
        return result

    url = f"https://www.ebi.ac.uk/ols4/api/terms?iri={quote(iri, safe='')}"
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=10.0)
    except httpx.HTTPError as exc:
        logger.debug("OLS4 request failed for %s: %s", normalized, exc)
        return result
    if resp.status_code != 200:
        logger.debug("OLS4 returned HTTP %s for %s", resp.status_code, normalized)
        return result
    terms = resp.json().get("_embedded", {}).get("terms", [])
    if not terms:
        return result

    term = terms[0]
    result["label"] = term.get("label")
    result["synonyms"] = term.get("synonyms") or []
    annotation = term.get("annotation", {})
    aliases: list[str] = []
    icd10_codes: list[str] = []

    values: list[object] = []
    for raw in annotation.values():
        values.extend(raw if isinstance(raw, list) else [raw])
    values.extend(term.get("obo_xref") or [])

    for value in values:
        for alias in extract_ontology_ids(value):
            if alias != normalized and alias not in aliases:
                aliases.append(alias)
        for match in _ICD10_RE.finditer(str(value)):
            code = match.group(1)
            if code not in icd10_codes:
                icd10_codes.append(code)

    result["aliases"] = aliases
    result["icd10_codes"] = icd10_codes
    return result


def query_ols_efo_xrefs(efo_id: str) -> list[str]:
    """Query OLS4 for xrefs attached to an EFO term and return normalized aliases."""
    import httpx

    normalized = normalize_trait_id(efo_id)
    if normalized is None or not normalized.startswith("EFO_"):
        return []

    data = query_ols_trait_xrefs(normalized)

    aliases: list[str] = []
    for alias in data.get("aliases", []):
        if alias != normalized and alias not in aliases:
            aliases.append(alias)
    return aliases


def build_efo_alias_map(
    efo_ids: list[str],
    cache_dir: Path | None = None,
    allow_network: bool = True,
) -> dict[str, list[str]]:
    """Build EFO -> alias-ID mapping, caching each EFO lookup on disk."""
    result: dict[str, list[str]] = {}
    normalized_ids = sorted({
        normalized
        for raw_id in efo_ids
        if (normalized := normalize_trait_id(raw_id)) is not None and normalized.startswith("EFO_")
    })

    for efo_id in normalized_ids:
        aliases: list[str] = []
        cache_file = _ols_cache_file(cache_dir, efo_id) if cache_dir is not None else None
        cache_loaded = False
        if cache_file is not None and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                logger.warning("Ignoring corrupt ontology alias cache file: %s", cache_file)
                cache_file.unlink(missing_ok=True)
            else:
                cache_loaded = True
                raw_aliases = cached.get("aliases", [])
                aliases = [
                    alias
                    for item in raw_aliases
                    if (alias := normalize_trait_id(item)) is not None and alias != efo_id
                ]
        if not cache_loaded and allow_network:
            aliases = query_ols_efo_xrefs(efo_id)
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({"efo_id": efo_id, "aliases": aliases}, indent=2))

        deduped: list[str] = []
        for alias in aliases:
            if alias not in deduped:
                deduped.append(alias)
        result[efo_id] = deduped

    return result


def ensure_ontology_alias_columns(df: pl.DataFrame, id_col: str = "efo_id") -> pl.DataFrame:
    """Ensure risk metadata contains alias provenance columns."""
    result = df
    if "canonical_efo_id" not in result.columns:
        result = result.with_columns(
            pl.when(pl.col(id_col).str.starts_with("EFO_"))
            .then(pl.col(id_col))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("canonical_efo_id")
        )
    if "mapped_from_id" not in result.columns:
        result = result.with_columns(pl.lit(None, dtype=pl.Utf8).alias("mapped_from_id"))
    if "mapping_source" not in result.columns:
        result = result.with_columns(pl.lit("direct").alias("mapping_source"))
    result = result.with_columns([
        pl.col("canonical_efo_id").cast(pl.Utf8),
        pl.col("mapped_from_id").cast(pl.Utf8),
        pl.col("mapping_source").cast(pl.Utf8),
    ])
    return result


def enrich_with_trait_aliases(
    df: pl.DataFrame,
    id_col: str = "efo_id",
    cache_dir: Path | None = None,
    allow_network: bool = True,
) -> pl.DataFrame:
    """Duplicate EFO-keyed risk metadata rows under equivalent ontology aliases."""
    if df.height == 0 or id_col not in df.columns:
        return ensure_ontology_alias_columns(df, id_col=id_col)

    base = ensure_ontology_alias_columns(df, id_col=id_col)
    efo_ids = [
        value
        for value in base.get_column(id_col).drop_nulls().unique().to_list()
        if (normalized := normalize_trait_id(value)) is not None and normalized.startswith("EFO_")
    ]
    alias_map = build_efo_alias_map(efo_ids, cache_dir=cache_dir, allow_network=allow_network)

    alias_rows: list[dict[str, Any]] = []
    for row in base.iter_rows(named=True):
        source_id = normalize_trait_id(row.get(id_col))
        if source_id is None or not source_id.startswith("EFO_"):
            continue
        for alias_id in alias_map.get(source_id, []):
            copied = dict(row)
            copied[id_col] = alias_id
            copied["canonical_efo_id"] = source_id
            copied["mapped_from_id"] = source_id
            copied["mapping_source"] = "ols4_xref"
            alias_rows.append(copied)

    if not alias_rows:
        return base

    aliases = pl.DataFrame(alias_rows, schema=base.schema)
    combined = pl.concat([base, aliases], how="vertical_relaxed")
    return combined.unique(maintain_order=True)


def _icd10_to_efo_map(efo_mappings_df: pl.DataFrame) -> dict[str, str]:
    """Build ICD10 -> EFO lookup from the EBISPOT EFO-UKB mapping table."""
    result: dict[str, str] = {}
    if efo_mappings_df.height == 0 or {"ukb_code", "efo_id"} - set(efo_mappings_df.columns):
        return result
    rows = efo_mappings_df.filter(
        pl.col("ukb_code").is_not_null() & pl.col("efo_id").is_not_null()
    ).select("ukb_code", "efo_id").unique()
    for row in rows.iter_rows(named=True):
        efo_id = normalize_trait_id(row["efo_id"])
        if efo_id is None:
            continue
        code = str(row["ukb_code"]).strip()
        if code:
            result[code] = efo_id
            result[code.split(".", 1)[0]] = efo_id
    return result


def enrich_with_requested_trait_aliases(
    df: pl.DataFrame,
    requested_traits_df: pl.DataFrame,
    efo_mappings_df: pl.DataFrame,
    id_col: str = "efo_id",
    requested_id_col: str = "trait_efo_id",
    cache_dir: Path | None = None,
    allow_network: bool = True,
) -> pl.DataFrame:
    """Add aliases for non-EFO PGS trait IDs that resolve to EFO-keyed metadata.

    This handles the common disease path where a PGS score uses MONDO but
    Pan-UKBB heritability is keyed by EFO via ICD10/UKB mappings.
    """
    if df.height == 0 or requested_traits_df.height == 0 or requested_id_col not in requested_traits_df.columns:
        return ensure_ontology_alias_columns(df, id_col=id_col)

    base = ensure_ontology_alias_columns(df, id_col=id_col)
    available_efo_ids = {
        normalized
        for value in base.get_column(id_col).drop_nulls().to_list()
        if (normalized := normalize_trait_id(value)) is not None and normalized.startswith("EFO_")
    }
    if not available_efo_ids:
        return base

    icd_to_efo = _icd10_to_efo_map(efo_mappings_df)
    if not icd_to_efo:
        return base

    alias_to_efo: dict[str, str] = {}
    raw_ids: list[str] = []
    for value in requested_traits_df.get_column(requested_id_col).drop_nulls().to_list():
        for part in str(value).split(","):
            normalized = normalize_trait_id(part.strip())
            if normalized is not None and normalized not in raw_ids:
                raw_ids.append(normalized)

    for trait_id in raw_ids:
        if trait_id.startswith("EFO_"):
            continue
        cache_file = _ols_cache_file(cache_dir, trait_id) if cache_dir is not None else None
        cached: dict[str, Any] | None = None
        if cache_file is not None and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                logger.warning("Ignoring corrupt ontology xref cache file: %s", cache_file)
                cache_file.unlink(missing_ok=True)
        xrefs = cached
        if xrefs is None and allow_network:
            xrefs = query_ols_trait_xrefs(trait_id)
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(xrefs, indent=2))
        if not xrefs:
            continue
        for code in xrefs.get("icd10_codes", []):
            candidates = [code, str(code).split(".", 1)[0]]
            for candidate in candidates:
                efo_id = icd_to_efo.get(candidate)
                if efo_id in available_efo_ids:
                    alias_to_efo[trait_id] = efo_id
                    break
            if trait_id in alias_to_efo:
                break

    if not alias_to_efo:
        return base

    alias_rows: list[dict[str, Any]] = []
    for row in base.iter_rows(named=True):
        source_id = normalize_trait_id(row.get(id_col))
        if source_id is None:
            continue
        for alias_id, efo_id in alias_to_efo.items():
            if source_id != efo_id:
                continue
            copied = dict(row)
            copied[id_col] = alias_id
            copied["canonical_efo_id"] = efo_id
            copied["mapped_from_id"] = efo_id
            copied["mapping_source"] = "ols4_icd10_to_efo"
            alias_rows.append(copied)

    if not alias_rows:
        return base
    aliases = pl.DataFrame(alias_rows, schema=base.schema)
    return pl.concat([base, aliases], how="vertical_relaxed").unique(maintain_order=True)


def expand_trait_ids_from_alias_columns(
    trait_ids: list[str],
    metadata_df: pl.DataFrame,
    id_col: str = "efo_id",
) -> list[str]:
    """Expand requested trait IDs using alias provenance columns already in metadata."""
    requested = [
        normalized
        for raw_id in trait_ids
        if (normalized := normalize_trait_id(raw_id)) is not None
    ]
    expanded: list[str] = list(dict.fromkeys(requested))
    if metadata_df.height == 0 or id_col not in metadata_df.columns:
        return expanded

    df = ensure_ontology_alias_columns(metadata_df, id_col=id_col)
    aliases = df.filter(pl.col(id_col).is_in(requested))
    if aliases.height > 0:
        for col_name in (id_col, "canonical_efo_id", "mapped_from_id"):
            for value in aliases.get_column(col_name).drop_nulls().to_list():
                normalized = normalize_trait_id(value)
                if normalized is not None and normalized not in expanded:
                    expanded.append(normalized)

    canonical_hits = df.filter(pl.col("canonical_efo_id").is_in(requested))
    if canonical_hits.height > 0:
        for value in canonical_hits.get_column(id_col).drop_nulls().to_list():
            normalized = normalize_trait_id(value)
            if normalized is not None and normalized not in expanded:
                expanded.append(normalized)

    return expanded


def alias_coverage_metadata(df: pl.DataFrame, id_col: str = "efo_id", prefix: str = "alias") -> dict[str, int | str]:
    """Summarize alias enrichment for Dagster/HF metadata."""
    if df.height == 0 or id_col not in df.columns:
        return {
            f"n_{prefix}_rows": 0,
            f"n_{prefix}_alias_rows": 0,
            f"n_{prefix}_canonical_traits": 0,
            f"{prefix}_ontology_prefixes": "",
        }
    enriched = ensure_ontology_alias_columns(df, id_col=id_col)
    alias_rows = enriched.filter(pl.col("mapping_source") != "direct")
    prefixes = sorted({
        str(value).split("_", 1)[0]
        for value in enriched.get_column(id_col).drop_nulls().to_list()
        if "_" in str(value)
    })
    return {
        f"n_{prefix}_rows": enriched.height,
        f"n_{prefix}_alias_rows": alias_rows.height,
        f"n_{prefix}_canonical_traits": enriched.get_column("canonical_efo_id").drop_nulls().n_unique(),
        f"{prefix}_ontology_prefixes": ",".join(prefixes),
    }
