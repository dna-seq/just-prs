"""Sex-chromosome karyotype of one sample, inferred from its genotypes.

A man has one X. Many callers do not write it that way: DeepVariant 1.1.0 writes
a man's non-PAR X and Y diploid (``1/1``, and ``0/1`` where a second X would be
needed), so the GT alone cannot say how many copies there are. The *pattern* can:

* **X copies** from the non-PAR X heterozygosity, relative to the same file's
  autosomes (so arrays and WGS share one threshold). One X has (almost) no hets;
  measured: anton (male, DeepVariant 1.1.0) 0.088, livia (female, DRAGEN 4.4) ~0.9.
  A sex-aware caller writes a man's X haploid (``"1"``), which decides it directly.
* **Y presence** from called non-PAR Y records relative to non-PAR X records:
  anton 0.063, a GSA-type array (Genotec 2024, male) 0.017, livia 0 (DRAGEN
  filters a woman's Y as ``PloidyConflict``; an array writes her Y markers ``--``).

Together: XX, XY, XXY (two X and a Y) or X0 (one X, no Y calls: a man whose Y was
filtered out of the input looks the same). XYY needs read depth and is not
called. Anything between the bands is ``unknown`` and changes nothing.

The karyotype sets a ploidy per region (:func:`ploidy_expr`): the engines then
count a diploid-written hemizygous ``1/1`` as one copy and treat a hemizygous
``0/1`` as a no-call. The GT is never rewritten.
"""

from __future__ import annotations

import enum
import functools
from pathlib import Path

import polars as pl
from pydantic import BaseModel

AUTOSOMES = [str(i) for i in range(1, 23)]

# Pseudoautosomal regions, 1-based inclusive. PAR positions are diploid in
# every karyotype (one copy on X plus one on Y in a man).
PAR_REGIONS: dict[str, dict[str, list[tuple[int, int]]]] = {
    "GRCh38": {
        "X": [(10_001, 2_781_479), (155_701_383, 156_030_895)],
        "Y": [(10_001, 2_781_479), (56_887_903, 57_217_415)],
    },
    "GRCh37": {
        "X": [(60_001, 2_699_520), (154_931_044, 155_260_560)],
        "Y": [(10_001, 2_649_520), (59_034_050, 59_373_566)],
    },
}

MITOCHONDRIAL = ["MT", "M"]

# Bands, calibrated on anton (XY) and livia (XX); see the module docstring.
MIN_X_RECORDS = 200
ONE_X_MAX_RELATIVE_HET = 0.25
TWO_X_MIN_RELATIVE_HET = 0.5
HAPLOID_X_MIN_FRACTION = 0.5
Y_PRESENT_MIN_RATIO = 0.01
Y_ABSENT_MAX_RATIO = 0.002


class Karyotype(enum.StrEnum):
    XX = "XX"
    XY = "XY"
    XXY = "XXY"
    X0 = "X0"
    UNKNOWN = "unknown"


class KaryotypeSource(enum.StrEnum):
    PROVIDED = "provided"
    INFERRED = "inferred"


class XChromosomeCoding(enum.StrEnum):
    """How a one-copy region (male non-PAR X, Y, MT) is counted.

    ``copies``: the number of copies of the effect allele, 0..1. The default.
    ``dosage_compensation``: 0..2, as PLINK2 ``--xchr-model 2``, many GWAS and
    the 1000G pgen code a man's X. It is a model (one active X in either sex),
    kept for parity checks against those tools.
    """

    COPIES = "copies"
    DOSAGE_COMPENSATION = "dosage_compensation"


class KaryotypeCall(BaseModel):
    """A karyotype and the evidence it was called from."""

    karyotype: Karyotype
    source: KaryotypeSource
    x_records: int = 0
    x_het_fraction: float | None = None
    autosome_het_fraction: float | None = None
    x_relative_het: float | None = None
    x_haploid_fraction: float | None = None
    y_records: int = 0
    y_to_x_ratio: float | None = None

    @property
    def x_copies(self) -> int | None:
        return {Karyotype.XX: 2, Karyotype.XXY: 2, Karyotype.XY: 1, Karyotype.X0: 1}.get(self.karyotype)

    @property
    def y_copies(self) -> int | None:
        return {Karyotype.XX: 0, Karyotype.X0: 0, Karyotype.XY: 1, Karyotype.XXY: 1}.get(self.karyotype)


def _par_expr(chrom: pl.Expr, pos: pl.Expr, contig: str, genome_build: str) -> pl.Expr:
    regions = PAR_REGIONS.get(genome_build, PAR_REGIONS["GRCh38"])[contig]
    in_par = pl.lit(False)
    for start, end in regions:
        in_par = in_par | pos.is_between(start, end)
    return (chrom == contig) & in_par


def ploidy_expr(
    karyotype: Karyotype,
    genome_build: str,
    chrom_col: str = "chr_name_norm",
    pos_col: str = "chr_pos_norm",
) -> pl.Expr:
    """Copies of the region a row sits in: 0, 1 or 2 (``Int8``).

    MT is always 1. Non-PAR X and Y follow the karyotype; PAR and autosomes are 2.
    ``unknown`` keeps 2 everywhere except MT, i.e. the GT as written.
    """
    chrom = pl.col(chrom_col).cast(pl.Utf8)
    pos = pl.col(pos_col).cast(pl.Int64)
    call = KaryotypeCall(karyotype=karyotype, source=KaryotypeSource.PROVIDED)
    x_copies = call.x_copies if call.x_copies is not None else 2
    y_copies = call.y_copies if call.y_copies is not None else 2
    non_par_x = (chrom == "X") & ~_par_expr(chrom, pos, "X", genome_build)
    non_par_y = (chrom == "Y") & ~_par_expr(chrom, pos, "Y", genome_build)
    return (
        pl.when(chrom.is_in(MITOCHONDRIAL)).then(pl.lit(1))
        .when(non_par_x).then(pl.lit(x_copies))
        .when(non_par_y).then(pl.lit(y_copies))
        .otherwise(pl.lit(2))
        .cast(pl.Int8)
    )


def ploidy_sql(karyotype: Karyotype, genome_build: str, chrom: str, pos: str) -> str:
    """DuckDB twin of :func:`ploidy_expr`."""
    call = KaryotypeCall(karyotype=karyotype, source=KaryotypeSource.PROVIDED)
    x_copies = call.x_copies if call.x_copies is not None else 2
    y_copies = call.y_copies if call.y_copies is not None else 2
    regions = PAR_REGIONS.get(genome_build, PAR_REGIONS["GRCh38"])

    def in_par(contig: str) -> str:
        return " OR ".join(f"({pos} BETWEEN {s} AND {e})" for s, e in regions[contig])

    mt = ", ".join(f"'{c}'" for c in MITOCHONDRIAL)
    return f"""(CASE
        WHEN {chrom} IN ({mt}) THEN 1
        WHEN {chrom} = 'X' AND NOT ({in_par('X')}) THEN {x_copies}
        WHEN {chrom} = 'Y' AND NOT ({in_par('Y')}) THEN {y_copies}
        ELSE 2 END)"""


def karyotype_evidence(genotypes_lf: pl.LazyFrame, genome_build: str = "GRCh38") -> pl.DataFrame:
    """Called-record counts per class (``auto``, ``x``, ``y``): ``n``, ``het``, ``haploid``.

    Reads ``chrom``, ``pos``, ``GT`` and, when present, ``filter`` (only PASS or
    ``.`` records count). Non-PAR X and Y only; ``chr`` prefixes are stripped.
    """
    columns = genotypes_lf.collect_schema().names()
    chrom = pl.col("chrom").cast(pl.Utf8).str.replace(r"(?i)^chr", "")
    pos = pl.col("pos").cast(pl.Int64)
    parts = pl.col("GT").cast(pl.Utf8).str.replace_all("|", "/", literal=True).str.split_exact("/", 1)
    first, second = parts.struct.field("field_0"), parts.struct.field("field_1")
    called = first.is_not_null() & (first != ".") & (second.is_null() | (second != "."))
    keep = called
    if "filter" in columns:
        keep = keep & (pl.col("filter").is_null() | pl.col("filter").is_in(["PASS", ".", ""]))
    cls = (
        pl.when(chrom.is_in(AUTOSOMES)).then(pl.lit("auto"))
        .when((chrom == "X") & ~_par_expr(chrom, pos, "X", genome_build)).then(pl.lit("x"))
        .when((chrom == "Y") & ~_par_expr(chrom, pos, "Y", genome_build)).then(pl.lit("y"))
    )
    return (
        genotypes_lf.filter(keep)
        .select(
            cls.alias("cls"),
            (second.is_not_null() & (first != second)).alias("het"),
            second.is_null().alias("haploid"),
        )
        .filter(pl.col("cls").is_not_null())
        .group_by("cls")
        .agg(pl.len().alias("n"), pl.col("het").sum(), pl.col("haploid").sum())
        .collect()
    )


def call_karyotype(evidence: pl.DataFrame) -> KaryotypeCall:
    """Apply the bands to :func:`karyotype_evidence` output."""
    rows = {r["cls"]: r for r in evidence.iter_rows(named=True)}
    zero = {"n": 0, "het": 0, "haploid": 0}
    auto, x, y = rows.get("auto", zero), rows.get("x", zero), rows.get("y", zero)
    x_n = int(x["n"])
    call = KaryotypeCall(karyotype=Karyotype.UNKNOWN, source=KaryotypeSource.INFERRED, x_records=x_n, y_records=int(y["n"]))
    if x_n < MIN_X_RECORDS:
        return call
    diploid_x = x_n - int(x["haploid"])
    x_het = int(x["het"]) / diploid_x if diploid_x else 0.0
    auto_het = int(auto["het"]) / int(auto["n"]) if int(auto["n"]) else None
    haploid_fraction = int(x["haploid"]) / x_n
    relative = x_het / auto_het if auto_het else None
    y_ratio = int(y["n"]) / x_n
    call = call.model_copy(update={
        "x_het_fraction": x_het,
        "autosome_het_fraction": auto_het,
        "x_relative_het": relative,
        "x_haploid_fraction": haploid_fraction,
        "y_to_x_ratio": y_ratio,
    })

    if haploid_fraction >= HAPLOID_X_MIN_FRACTION:
        x_copies: int | None = 1
    elif relative is None:
        x_copies = None
    elif relative <= ONE_X_MAX_RELATIVE_HET:
        x_copies = 1
    elif relative >= TWO_X_MIN_RELATIVE_HET:
        x_copies = 2
    else:
        x_copies = None

    if y_ratio >= Y_PRESENT_MIN_RATIO:
        has_y: bool | None = True
    elif y_ratio <= Y_ABSENT_MAX_RATIO:
        has_y = False
    else:
        has_y = None

    karyotype = {
        (2, False): Karyotype.XX,
        (1, True): Karyotype.XY,
        (2, True): Karyotype.XXY,
        (1, False): Karyotype.X0,
    }.get((x_copies, has_y), Karyotype.UNKNOWN)  # type: ignore[arg-type]
    return call.model_copy(update={"karyotype": karyotype})


def infer_karyotype(genotypes_lf: pl.LazyFrame, genome_build: str = "GRCh38") -> KaryotypeCall:
    """Infer the karyotype of the single sample in *genotypes_lf*."""
    return call_karyotype(karyotype_evidence(genotypes_lf, genome_build))


@functools.lru_cache(maxsize=64)
def _infer_karyotype_cached(path: str, mtime_ns: int, size: int, genome_build: str) -> KaryotypeCall:
    if path.endswith(".parquet"):
        return infer_karyotype(pl.scan_parquet(path), genome_build)
    from just_prs.vcf import read_genotypes

    return infer_karyotype(read_genotypes(Path(path)), genome_build)


def infer_karyotype_from_file(path: Path | str, genome_build: str = "GRCh38") -> KaryotypeCall:
    """:func:`infer_karyotype` on a genotype parquet or a VCF, cached per file version."""
    stat = Path(path).stat()
    return _infer_karyotype_cached(str(path), stat.st_mtime_ns, stat.st_size, genome_build)


def resolve_karyotype(
    sample_karyotype: Karyotype | KaryotypeCall | str | None,
    *,
    genotypes_lf: pl.LazyFrame | None = None,
    genotypes_parquet: Path | str | None = None,
    genome_build: str = "GRCh38",
) -> KaryotypeCall:
    """A provided karyotype as-is; ``None`` infers it from the genotypes.

    ``genotypes_parquet`` may also be a VCF path; a file is inferred once per
    version (cached), a LazyFrame on every call.
    """
    if isinstance(sample_karyotype, KaryotypeCall):
        return sample_karyotype
    if sample_karyotype is not None:
        return KaryotypeCall(karyotype=Karyotype(sample_karyotype), source=KaryotypeSource.PROVIDED)
    if genotypes_parquet is not None:
        return infer_karyotype_from_file(genotypes_parquet, genome_build)
    if genotypes_lf is not None:
        return infer_karyotype(genotypes_lf, genome_build)
    return KaryotypeCall(karyotype=Karyotype.UNKNOWN, source=KaryotypeSource.INFERRED)
