"""Consumer genotyping-array ingestion: 23andMe / AncestryDNA raw files → normalized Parquet.

Direct-to-consumer services (23andMe, AncestryDNA, MyHeritage, FamilyTreeDNA,
LivingDNA) export a simple per-marker text table rather than a VCF. This module
parses those raw files into the *same* normalized Parquet schema that
``normalize_vcf()`` produces, so ``compute_prs()`` / ``compute_prs_duckdb()``
consume array data with no engine changes.

Genotype encoding trick (keeps the existing GT/ref/alt dosage logic correct):
an array reports the two *observed* alleles directly (e.g. ``AG``), not VCF
ref/alt + GT indices. We map each call so the existing
``compute_dosage_expr`` counts the effect allele correctly:
    - heterozygous ``a1≠a2`` → ``ref=a1, alt=a2, GT="0/1"``
    - homozygous   ``a1==a2`` → ``ref=a1, alt=a1, GT="1/1"``
    - no-call               → ``GT="./.", genotype=[]``
Dosage = count of the effect allele among the two observed alleles, which this
encoding reproduces exactly for both engines.

Build: 23andMe v5 / AncestryDNA v2 (and the other current GSA-based kits) report
**GRCh37** coordinates. ``normalize_array`` therefore defaults to GRCh37, and the
caller must score against the matching GRCh37 harmonized scoring file (positions
are matched by chrom+pos). Pass ``genome_build="GRCh38"`` only for files already
lifted to GRCh38.

Known limitations (documented, not silently handled): strand flips (alleles
reported on the opposite strand from the harmonized scoring file won't match),
indels (``I``/``D`` 23andMe codes won't match sequence effect alleles), and
hemizygous male X/Y calls (a single observed allele is treated as homozygous).
"""

import gzip
import io
import zipfile
from pathlib import Path

import polars as pl
from eliot import log_message, start_action

# Tokens that denote a missing/no-call allele across vendors.
_MISSING_ALLELES = ["-", "0", ".", "?", ""]

# AncestryDNA encodes chromosomes numerically; 23andMe uses letters already.
# 25 = pseudoautosomal (on X), 26 = mitochondrial.
_CHR_NUMERIC_MAP = {"23": "X", "24": "Y", "25": "X", "26": "MT"}

ARRAY_FORMATS = ("23andme", "ancestrydna")


def _read_array_text(path: Path) -> str:
    """Read a raw array file as text, transparently handling .gz / .zip / plain."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
            if not members:
                raise ValueError(f"Empty zip archive: {path}")
            # Prefer a .txt/.csv member if multiple are present.
            data_members = [n for n in members if n.lower().endswith((".txt", ".csv"))]
            member = data_members[0] if data_members else members[0]
            return zf.read(member).decode("utf-8", errors="replace")
    if suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="replace")


def detect_array_format(text: str) -> str:
    """Detect the raw-array vendor format from file content.

    Returns ``"23andme"`` (4 columns: rsid, chromosome, position, genotype) or
    ``"ancestrydna"`` (5 columns: rsid, chromosome, position, allele1, allele2).
    """
    lower = text[:4000].lower()
    if "23andme" in lower:
        return "23andme"
    if "ancestry" in lower:
        return "ancestrydna"
    # Fall back to column count of the first non-comment, non-blank line.
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        n_cols = len(stripped.split("\t"))
        return "ancestrydna" if n_cols >= 5 else "23andme"
    raise ValueError("Could not detect array format: no data rows found.")


def _read_array_dataframe(text: str, array_format: str) -> pl.DataFrame:
    """Parse cleaned array text into a raw DataFrame with canonical column names."""
    df = pl.read_csv(
        io.StringIO(text),
        separator="\t",
        comment_prefix="#",
        has_header=False,
        infer_schema_length=0,  # all Utf8; we cast positions ourselves
        truncate_ragged_lines=True,
    )
    if array_format == "ancestrydna":
        names = ["rsid", "chromosome", "position", "allele1", "allele2"]
    else:
        names = ["rsid", "chromosome", "position", "genotype"]
    # Map the first N columns; ignore any trailing junk columns.
    df = df.select(df.columns[: len(names)])
    df.columns = names
    # Drop an embedded header row (AncestryDNA ships a plain "rsid ..." header).
    df = df.filter(pl.col("rsid").str.to_lowercase() != "rsid")
    return df


def _allele_columns(df: pl.DataFrame, array_format: str) -> pl.DataFrame:
    """Add normalized ``a1`` / ``a2`` allele columns from vendor-specific layout."""
    if array_format == "ancestrydna":
        return df.with_columns(
            pl.col("allele1").str.strip_chars().str.to_uppercase().alias("a1"),
            pl.col("allele2").str.strip_chars().str.to_uppercase().alias("a2"),
        )
    g = pl.col("genotype").str.strip_chars().str.to_uppercase()
    length = g.str.len_chars()
    return df.with_columns(
        g.str.slice(0, 1).alias("a1"),
        pl.when(length >= 2).then(g.str.slice(1, 1)).otherwise(g.str.slice(0, 1)).alias("a2"),
    )


def normalize_array(
    array_path: Path,
    output_path: Path,
    genome_build: str = "GRCh37",
    array_format: str | None = None,
) -> Path:
    """Parse a 23andMe / AncestryDNA raw file into a normalized genotype Parquet.

    The output schema matches ``normalize_vcf()`` (``chrom``, ``pos``, ``rsid``,
    ``ref``, ``alt``, ``GT``, ``genotype``) so it is a drop-in input to
    ``compute_prs()`` / ``compute_prs_duckdb()``.

    Args:
        array_path: Path to the raw array file (``.txt``, ``.txt.gz``, ``.csv``,
            or ``.zip``).
        output_path: Destination Parquet path (zstd-compressed).
        genome_build: Build of the array's coordinates. Default ``GRCh37`` (the
            build 23andMe v5 / AncestryDNA v2 report). Used only for logging /
            provenance; positions are written as-is and must match the scoring
            file build chosen at compute time.
        array_format: ``"23andme"`` or ``"ancestrydna"``; auto-detected when None.

    Returns:
        The *output_path* for convenience.
    """
    with start_action(
        action_type="array:normalize",
        array_path=str(array_path),
        output_path=str(output_path),
        genome_build=genome_build,
    ):
        text = _read_array_text(array_path)
        fmt = array_format or detect_array_format(text)
        if fmt not in ARRAY_FORMATS:
            raise ValueError(f"Unsupported array format: {fmt!r}. Known: {ARRAY_FORMATS}")

        df = _read_array_dataframe(text, fmt)
        df = _allele_columns(df, fmt)

        is_missing = (
            pl.col("a1").is_in(_MISSING_ALLELES)
            | pl.col("a2").is_in(_MISSING_ALLELES)
            | pl.col("a1").is_null()
            | pl.col("a2").is_null()
        )
        is_hom = pl.col("a1") == pl.col("a2")

        chrom = (
            pl.col("chromosome")
            .str.replace("(?i)^chr", "")
            .str.to_uppercase()
            .replace(_CHR_NUMERIC_MAP)
        )

        df = df.with_columns(
            chrom.alias("chrom"),
            pl.col("position").cast(pl.Int64, strict=False).alias("pos"),
            pl.col("rsid").cast(pl.Utf8).alias("rsid"),
            pl.col("a1").alias("ref"),
            pl.col("a2").alias("alt"),
            pl.when(is_missing)
            .then(pl.lit("./."))
            .when(is_hom)
            .then(pl.lit("1/1"))
            .otherwise(pl.lit("0/1"))
            .alias("GT"),
            pl.when(is_missing)
            .then(pl.lit([], dtype=pl.List(pl.Utf8)))
            .otherwise(pl.concat_list(["a1", "a2"]).list.sort())
            .alias("genotype"),
        )

        # Keep only mapped, parseable rows. Missing-genotype rows are dropped:
        # they contribute zero dosage anyway and would only bloat the join.
        df = df.filter(
            pl.col("pos").is_not_null()
            & (pl.col("pos") > 0)
            & is_missing.not_()
        ).select("chrom", "pos", "rsid", "ref", "alt", "GT", "genotype")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path, compression="zstd")

        log_message(
            message_type="array:normalize_complete",
            array_format=fmt,
            genome_build=genome_build,
            rows=df.height,
            output_path=str(output_path),
        )
        return output_path


# ---------------------------------------------------------------------------
# Chip generation detection
# ---------------------------------------------------------------------------

_CHIP_GENERATIONS = [
    # (marker_min, marker_max, vendor_format, chip_id, platform, label, ld_proxy)
    (630_000, 680_000, "23andme", "gsa_v3", "Illumina GSA v3.0", "23andMe v5", True),
    (550_000, 620_000, "23andme", "omniexpress", "Illumina OmniExpress+", "23andMe v4", False),
    (930_000, 1_050_000, "23andme", "omniexpress", "Illumina OmniExpress", "23andMe v3", False),
    (640_000, 700_000, "ancestrydna", "gsa_v3", "Illumina GSA v3.0", "AncestryDNA v2", True),
    (680_000, 750_000, "ancestrydna", "omniexpress", "Illumina OmniExpress", "AncestryDNA v1", False),
]


def detect_chip_generation(
    normalized_df_or_path: pl.DataFrame | Path,
    array_format: str | None = None,
) -> "ChipGeneration":
    """Infer chip generation from marker count and vendor format.

    Uses the number of markers after normalization (no-calls already dropped) to
    determine which chip platform was used. When the marker count doesn't match
    any known generation, falls back to GSA v3 for post-2017 counts and
    OmniExpress for larger counts.

    Args:
        normalized_df_or_path: Normalized array DataFrame or path to normalized
            parquet (output of ``normalize_array``).
        array_format: Vendor format (``"23andme"`` or ``"ancestrydna"``).
            Auto-detected from marker count if None.

    Returns:
        ChipGeneration with chip_id, platform, generation_label,
        ld_proxy_available, and marker_count.
    """
    from just_prs.models import ChipGeneration

    if isinstance(normalized_df_or_path, Path):
        marker_count = pl.scan_parquet(normalized_df_or_path).select(pl.len()).collect().item()
    else:
        marker_count = normalized_df_or_path.height

    for m_min, m_max, fmt, chip_id, platform, label, ld_avail in _CHIP_GENERATIONS:
        if m_min <= marker_count <= m_max:
            if array_format is None or array_format == fmt:
                return ChipGeneration(
                    chip_id=chip_id,
                    platform=platform,
                    generation_label=label,
                    ld_proxy_available=ld_avail,
                    marker_count=marker_count,
                )

    if marker_count > 800_000:
        return ChipGeneration(
            chip_id="omniexpress",
            platform="Illumina OmniExpress (estimated)",
            generation_label="Unknown (high marker count)",
            ld_proxy_available=False,
            marker_count=marker_count,
        )
    return ChipGeneration(
        chip_id="gsa_v3",
        platform="Illumina GSA v3.0 (estimated)",
        generation_label="Unknown (GSA-range marker count)",
        ld_proxy_available=True,
        marker_count=marker_count,
    )
