"""VCF normalization: read, normalize columns, compute genotype, apply quality filters, sink to Parquet."""

from pathlib import Path

import polars as pl
import polars_bio as pb
from eliot import Message, start_action
from pydantic import BaseModel, Field


class VcfFilterConfig(BaseModel):
    """Configurable quality filters for VCF normalization.

    All fields are optional. When ``None`` the corresponding filter is skipped.
    Can be loaded from YAML via ``VcfFilterConfig.model_validate(yaml.safe_load(...))``.
    """

    pass_filters: list[str] | None = Field(
        default=None,
        description='Keep rows where FILTER is in this list (e.g. ["PASS", "."]). Excludes RefCall / gVCF blocks.',
    )
    min_depth: int | None = Field(
        default=None, description="Keep rows where DP >= this value."
    )
    min_qual: float | None = Field(
        default=None, description="Keep rows where QUAL >= this value."
    )
    sex: str | None = Field(
        default=None,
        description='Sample sex ("Male" / "Female"). When Female, a warning is logged if chrY variants exist.',
    )


_DEFAULT_FORMAT_FIELDS: list[str] = ["GT", "DP"]


def _find_column_ci(schema_names: list[str], target: str) -> str | None:
    """Case-insensitive column lookup. Returns the actual column name or ``None``."""
    target_lower = target.lower()
    for name in schema_names:
        if name.lower() == target_lower:
            return name
    return None


def genotype_expr(
    gt_col: str = "GT", ref_col: str = "ref", alt_col: str = "alt"
) -> pl.Expr:
    """Build a ``List[Utf8]`` genotype column from GT indices + REF/ALT alleles.

    Allele indices in the GT string (e.g. ``"0/1"``) are mapped to actual allele
    strings using REF (index 0) and ALT (index 1+).  polars-bio encodes
    multi-allelic ALTs with ``|`` as separator (e.g. ``"A|G"``).

    The two resolved alleles are sorted alphabetically so that ``["A","T"]``
    and ``["T","A"]`` both become ``["A","T"]``.  Missing genotypes
    (``"./."``, ``"."`` , or any unparseable GT) yield an empty list.
    """
    gt = pl.col(gt_col)
    ref = pl.col(ref_col)
    alt = pl.col(alt_col)

    gt_normalized = gt.cast(pl.Utf8).str.replace_all(r"\|", "/")
    parts = gt_normalized.str.split("/")

    idx0_str = parts.list.get(0, null_on_oob=True)
    idx1_str = parts.list.get(1, null_on_oob=True)

    is_missing = (
        idx0_str.is_null()
        | idx1_str.is_null()
        | (idx0_str == ".")
        | (idx1_str == ".")
    )

    idx0 = idx0_str.cast(pl.Int64, strict=False)
    idx1 = idx1_str.cast(pl.Int64, strict=False)

    alleles = pl.concat_list([ref.cast(pl.Utf8)]).list.concat(
        alt.cast(pl.Utf8).str.split("|")
    )

    allele0 = alleles.list.get(idx0, null_on_oob=True)
    allele1 = alleles.list.get(idx1, null_on_oob=True)

    pair_unsorted = pl.concat_list([allele0, allele1])
    pair = pair_unsorted.list.sort()

    return (
        pl.when(is_missing | idx0.is_null() | idx1.is_null() | allele0.is_null() | allele1.is_null())
        .then(pl.lit([], dtype=pl.List(pl.Utf8)))
        .otherwise(pair)
        .alias("genotype")
    )


def normalize_vcf(
    vcf_path: Path,
    output_path: Path,
    config: VcfFilterConfig | None = None,
    format_fields: list[str] | None = None,
) -> Path:
    """Read a VCF, normalize columns, apply quality filters, and write Parquet.

    Args:
        vcf_path: Path to ``.vcf`` or ``.vcf.gz`` file.
        output_path: Destination Parquet path (written with zstd compression).
        config: Optional quality filter settings. ``None`` means no filtering.
        format_fields: FORMAT fields to include (default ``["GT", "DP"]``).

    Returns:
        The *output_path* for convenience.
    """
    if config is None:
        config = VcfFilterConfig()

    if format_fields is None:
        format_fields = list(_DEFAULT_FORMAT_FIELDS)
    if "GT" not in format_fields:
        format_fields = ["GT", *format_fields]

    with start_action(
        action_type="vcf:normalize",
        vcf_path=str(vcf_path),
        output_path=str(output_path),
        format_fields=format_fields,
    ):
        lf = pb.scan_vcf(
            str(vcf_path),
            info_fields=[],
            format_fields=format_fields,
            use_zero_based=False,
        )

        columns = lf.collect_schema().names()

        select_exprs: list[pl.Expr] = [
            pl.col("chrom").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chrom"),
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
            pl.col("start").cast(pl.Int64).alias("pos"),
        ]

        id_col = _find_column_ci(columns, "id")
        if id_col is not None:
            select_exprs.append(pl.col(id_col).cast(pl.Utf8).alias("rsid"))

        select_exprs.append(pl.col("ref").cast(pl.Utf8))
        select_exprs.append(pl.col("alt").cast(pl.Utf8))

        qual_col = _find_column_ci(columns, "qual")
        if qual_col is not None:
            select_exprs.append(pl.col(qual_col).cast(pl.Float64).alias("qual"))

        filter_col = _find_column_ci(columns, "filter")
        if filter_col is not None:
            select_exprs.append(pl.col(filter_col).cast(pl.Utf8).alias("filter"))

        for ff in format_fields:
            actual = _find_column_ci(columns, ff)
            if actual is not None:
                if ff.upper() == "DP":
                    select_exprs.append(pl.col(actual).cast(pl.Int64, strict=False).alias("DP"))
                elif ff.upper() == "GT":
                    select_exprs.append(pl.col(actual).cast(pl.Utf8).alias("GT"))
                else:
                    select_exprs.append(pl.col(actual).alias(ff))

        lf = lf.select(select_exprs)

        lf = lf.with_columns(genotype_expr())

        # DataFusion-backed LazyFrames need collect() before aggregations
        df = lf.collect()

        # --- quality filters (applied on the materialized DataFrame) ---
        if config.pass_filters is not None:
            if "filter" in df.columns:
                df = df.filter(pl.col("filter").is_in(config.pass_filters))

        if config.min_depth is not None:
            dp_col = _find_column_ci(df.columns, "DP")
            if dp_col is not None:
                df = df.filter(pl.col(dp_col).cast(pl.Int64, strict=False) >= config.min_depth)

        if config.min_qual is not None:
            q_col = _find_column_ci(df.columns, "qual")
            if q_col is not None:
                df = df.filter(pl.col(q_col).cast(pl.Float64, strict=False) >= config.min_qual)

        # --- chrY warning for females ---
        if config.sex is not None and config.sex.lower() == "female":
            chr_y_count = df.filter(pl.col("chrom") == "Y").height
            if chr_y_count > 0:
                Message.log(
                    message_type="vcf:chrY_warning_female",
                    chr_y_variants=chr_y_count,
                    note="chrY variants found for a female sample. They are kept but may indicate sample mix-up.",
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path, compression="zstd")

        Message.log(
            message_type="vcf:normalize_complete",
            rows=df.height,
            columns=df.columns,
            output_path=str(output_path),
        )

        return output_path
