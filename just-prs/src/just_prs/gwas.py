"""GWAS Catalog bulk data download and parsing.

Downloads the GWAS Catalog bulk studies TSV and trait-to-EFO mapping file,
parses case/control counts from free-text sample descriptions, and produces
a structured parquet with per-study metadata linked to EFO trait IDs.
"""

import io
import re
from pathlib import Path

import polars as pl
from eliot import start_action

GWAS_STUDIES_URL = "https://www.ebi.ac.uk/gwas/api/search/downloads/studies_new"
GWAS_TRAIT_MAPPINGS_URL = "https://www.ebi.ac.uk/gwas/api/search/downloads/trait_mappings"

_CASES_PATTERN = re.compile(
    r"([\d,]+)\s+(?:[\w\s]*?\s)?cases\b",
    re.IGNORECASE,
)
_CONTROLS_PATTERN = re.compile(
    r"([\d,]+)\s+(?:[\w\s]*?\s)?controls\b",
    re.IGNORECASE,
)


def _parse_int(s: str) -> int:
    """Parse a comma-separated integer string."""
    return int(s.strip().replace(",", ""))


def parse_sample_size(text: str | None) -> tuple[int | None, int | None]:
    """Extract total case and control counts from a GWAS Catalog sample size string.

    Handles formats like:
        "1,019 Chinese ancestry cases, 1,710 Chinese ancestry controls"
        "37 European ancestry cases, 36 European ancestry controls"
        "4,390 individuals"

    Uses an iterative approach: finds all "N ... cases" and "N ... controls"
    patterns, taking the number immediately preceding the keyword.

    Returns:
        Tuple of (total_cases, total_controls). Either may be None if not found.
    """
    if not text:
        return None, None

    total_cases = 0
    total_controls = 0
    found_cases = False
    found_controls = False

    for m in re.finditer(r"\b(\d[\d,]*\d|\d)\s+", text):
        num_str = m.group(1)
        rest = text[m.end():]
        first_keyword = re.match(r"[\w\s]*?\b(cases|controls)\b", rest, re.IGNORECASE)
        if first_keyword:
            keyword = first_keyword.group(1).lower()
            if keyword == "cases":
                found_cases = True
                total_cases += _parse_int(num_str)
            elif keyword == "controls":
                found_controls = True
                total_controls += _parse_int(num_str)

    return (total_cases if found_cases else None, total_controls if found_controls else None)


def download_gwas_studies(
    output_path: Path,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Download the GWAS Catalog bulk studies TSV and save as parquet.

    Parses case/control counts from the free-text ``INITIAL SAMPLE SIZE``
    column and adds structured ``n_cases`` and ``n_controls`` columns.

    Args:
        output_path: Destination .parquet file path.
        overwrite: If True, re-download even if parquet exists.

    Returns:
        DataFrame with parsed study metadata.
    """
    import httpx

    with start_action(action_type="gwas:download_studies"):
        if output_path.exists() and not overwrite:
            return pl.read_parquet(output_path)

        resp = httpx.get(GWAS_STUDIES_URL, follow_redirects=True, timeout=120.0)
        resp.raise_for_status()

        df = pl.read_csv(
            io.BytesIO(resp.content),
            separator="\t",
            infer_schema_length=10000,
            null_values=["", "NA", "None", "NR"],
            quote_char=None,
            truncate_ragged_lines=True,
        )

        sample_col = None
        for candidate in ("INITIAL SAMPLE SIZE", "INITIAL SAMPLE DESCRIPTION"):
            if candidate in df.columns:
                sample_col = candidate
                break

        if sample_col is not None:
            parsed = [parse_sample_size(v) for v in df[sample_col].to_list()]
            df = df.with_columns(
                pl.Series("n_cases", [p[0] for p in parsed], dtype=pl.Int64),
                pl.Series("n_controls", [p[1] for p in parsed], dtype=pl.Int64),
            )
        else:
            df = df.with_columns(
                pl.lit(None, dtype=pl.Int64).alias("n_cases"),
                pl.lit(None, dtype=pl.Int64).alias("n_controls"),
            )

        rename_map: dict[str, str] = {}
        for raw, clean in [
            ("STUDY ACCESSION", "study_accession"),
            ("PUBMEDID", "pubmed_id"),
            ("DISEASE/TRAIT", "trait"),
            ("INITIAL SAMPLE SIZE", "initial_sample_size"),
            ("INITIAL SAMPLE DESCRIPTION", "initial_sample_size"),
            ("DATE", "date"),
            ("FIRST AUTHOR", "first_author"),
            ("JOURNAL", "journal"),
        ]:
            if raw in df.columns:
                rename_map[raw] = clean

        if rename_map:
            df = df.rename(rename_map)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        return df


def download_gwas_trait_mappings(
    output_path: Path,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Download the GWAS Catalog trait-to-EFO mapping file and save as parquet.

    Args:
        output_path: Destination .parquet file path.
        overwrite: If True, re-download even if parquet exists.

    Returns:
        DataFrame with trait-to-EFO mappings.
    """
    import httpx

    with start_action(action_type="gwas:download_trait_mappings"):
        if output_path.exists() and not overwrite:
            return pl.read_parquet(output_path)

        resp = httpx.get(GWAS_TRAIT_MAPPINGS_URL, follow_redirects=True, timeout=120.0)
        resp.raise_for_status()

        df = pl.read_csv(
            io.BytesIO(resp.content),
            separator="\t",
            infer_schema_length=10000,
            null_values=["", "NA", "None"],
        )

        rename_map: dict[str, str] = {}
        for raw, clean in [
            ("Disease trait", "trait"),
            ("EFO term", "efo_label"),
            ("EFO URI", "efo_uri"),
            ("Parent term", "parent_term"),
        ]:
            if raw in df.columns:
                rename_map[raw] = clean
        if rename_map:
            df = df.rename(rename_map)

        if "efo_uri" in df.columns:
            df = df.with_columns(
                pl.col("efo_uri")
                .str.extract(r"([A-Za-z_]+\d+)$")
                .alias("efo_id")
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        return df


def build_gwas_trait_summary(
    studies_df: pl.DataFrame,
    mappings_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join GWAS studies with EFO trait mappings and aggregate per EFO trait.

    For each EFO trait, picks the study with the largest total sample size
    (n_cases + n_controls) and reports its case/control counts.

    Returns:
        DataFrame with columns: efo_id, efo_label, trait, n_cases, n_controls,
        n_total, study_accession, pubmed_id, case_fraction.
    """
    trait_col = "trait"
    if trait_col not in studies_df.columns:
        for candidate in ("DISEASE/TRAIT", "disease_trait"):
            if candidate in studies_df.columns:
                trait_col = candidate
                break

    if "efo_id" not in mappings_df.columns:
        return pl.DataFrame()

    studies_with_counts = studies_df.filter(
        pl.col("n_cases").is_not_null() & pl.col("n_controls").is_not_null()
    )

    if studies_with_counts.height == 0:
        return pl.DataFrame()

    studies_with_counts = studies_with_counts.with_columns(
        (pl.col("n_cases") + pl.col("n_controls")).alias("n_total")
    )

    join_cols_studies = [trait_col]
    join_cols_mappings = [trait_col if trait_col in mappings_df.columns else "trait"]

    joined = studies_with_counts.join(
        mappings_df.select(
            pl.col(join_cols_mappings[0]).alias("trait_join"),
            "efo_id",
            "efo_label",
        ).unique(),
        left_on=trait_col,
        right_on="trait_join",
        how="inner",
    )

    if joined.height == 0:
        return pl.DataFrame()

    best_per_efo = (
        joined
        .sort("n_total", descending=True)
        .group_by("efo_id")
        .first()
    )

    result_cols = ["efo_id", "efo_label"]
    if trait_col in best_per_efo.columns:
        result_cols.append(trait_col)
    for col in ("n_cases", "n_controls", "n_total", "study_accession", "pubmed_id"):
        if col in best_per_efo.columns:
            result_cols.append(col)

    result = best_per_efo.select([c for c in result_cols if c in best_per_efo.columns])

    if "n_cases" in result.columns and "n_total" in result.columns:
        result = result.with_columns(
            (pl.col("n_cases").cast(pl.Float64) / pl.col("n_total").cast(pl.Float64))
            .alias("case_fraction")
        )

    return result
