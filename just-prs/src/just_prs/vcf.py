"""VCF reading via polars-bio and genotype dosage extraction."""

import re
from pathlib import Path

import polars as pl
import polars_bio as pb
from eliot import start_action

from just_prs.io_utils import open_maybe_compressed

GRCH38_CONTIG_LENGTHS: dict[str, int] = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}

GRCH37_CONTIG_LENGTHS: dict[str, int] = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276,
    "5": 180915260, "6": 171115067, "7": 159138663, "8": 146364022,
    "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753,
    "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520,
    "21": 48129895, "22": 51304566, "X": 155270560, "Y": 59373566,
}

# ``b37`` is matched only as a standalone token so paths like ``.../lib37/`` or
# ``sub37`` don't masquerade as a build name; ``GRCh38``/``hg38`` and their 37
# counterparts are distinctive enough to match anywhere.
_BUILD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"GRCh38|hg38", re.IGNORECASE), "GRCh38"),
    (re.compile(r"GRCh37|hg19|\bb37\b", re.IGNORECASE), "GRCh37"),
]

# Header keys whose value names the build directly. ``##reference=`` is the VCF
# standard; the DRAGEN/aligner lines carry the build inside the reference path
# (``--ht-reference=.../hg38/...``) or assembly flag (``--variant-annotation-assembly GRCh38``)
# even when no ``##reference=`` line is emitted, as is the case for DRAGEN output.
_BUILD_HINT_PREFIXES: tuple[str, ...] = (
    "##reference=",
    "##assembly=",
    "##DRAGENCommandLine=",
    "##source=",
)

_CONTIG_RE = re.compile(r"##contig=<.*?ID=(?:chr)?(\w+).*?length=(\d+)", re.IGNORECASE)


def _build_from_hint_line(line: str) -> str | None:
    """Return the build named by a header hint line, or None if ambiguous."""
    for pattern, build in _BUILD_PATTERNS:
        if pattern.search(line):
            return build
    return None


def detect_genome_build(vcf_path: Path | str) -> str | None:
    """Attempt to detect the genome build from a VCF file header.

    Detection is layered, preferring explicit evidence over inference:

    1. Build names in known header keys — ``##reference=``, ``##assembly=``,
       and aligner command lines (``##DRAGENCommandLine=``, ``##source=``).
       DRAGEN VCFs omit ``##reference=`` and only name the build inside the
       reference path / ``--variant-annotation-assembly`` flag, so those lines
       are scanned too.
    2. ``##contig=`` chromosome lengths voted against GRCh37 / GRCh38 tables.

    Args:
        vcf_path: Path to VCF file (plain text or gzipped)

    Returns:
        "GRCh37", "GRCh38", or None if detection fails
    """
    vcf_path = Path(vcf_path)
    # Detect compression by content (magic bytes), not extension — a BGZF stream
    # named .vcf must still be read decompressed, or the header read is garbage.
    with open_maybe_compressed(vcf_path, "rt") as fh:
        contig_votes: dict[str, int] = {"GRCh37": 0, "GRCh38": 0}

        for line in fh:
            if not line.startswith("##"):
                break

            # ``assembly=`` also appears as a ``##contig`` attribute (e.g.
            # ``##contig=<ID=chr1,...,assembly=GRCh38>``), another explicit signal.
            if line.startswith(_BUILD_HINT_PREFIXES) or "assembly=" in line.lower():
                build = _build_from_hint_line(line)
                if build is not None:
                    return build

            m = _CONTIG_RE.match(line)
            if m:
                chrom = m.group(1)
                length = int(m.group(2))
                if GRCH38_CONTIG_LENGTHS.get(chrom) == length:
                    contig_votes["GRCh38"] += 1
                if GRCH37_CONTIG_LENGTHS.get(chrom) == length:
                    contig_votes["GRCh37"] += 1

        if contig_votes["GRCh38"] > contig_votes["GRCh37"] and contig_votes["GRCh38"] >= 3:
            return "GRCh38"
        if contig_votes["GRCh37"] > contig_votes["GRCh38"] and contig_votes["GRCh37"] >= 3:
            return "GRCh37"

    return None


def read_genotypes(vcf_path: Path | str) -> pl.LazyFrame:
    """Read genotypes from a VCF file (or a pre-normalized Parquet) for PRS scoring.

    Returns a LazyFrame with at least columns: chrom, pos, ref, alt, GT.
    The chrom column is normalized to remove 'chr' prefix for consistent matching.

    A ``.parquet`` path is read directly with ``scan_parquet`` — this covers both
    ``normalize_vcf()`` and ``normalize_array()`` outputs, so consumer-array data
    flows through the same compute path as VCFs.

    Args:
        vcf_path: Path to a VCF file (plain/gzipped) or a normalized ``.parquet``.

    Returns:
        LazyFrame with genotype data
    """
    if str(vcf_path).endswith(".parquet"):
        with start_action(action_type="vcf:read_genotypes_parquet", path=str(vcf_path)):
            lf = pl.scan_parquet(vcf_path)
            cols = lf.collect_schema().names()
            if "pos" not in cols and "start" in cols:
                lf = lf.rename({"start": "pos"})
            return lf
    with start_action(action_type="vcf:read_genotypes", vcf_path=str(vcf_path)):
        lf = pb.scan_vcf(
            str(vcf_path),
            format_fields=["GT"],
            use_zero_based=False,
        )
        lf = lf.select([
            pl.col("chrom").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chrom"),
            pl.col("start").cast(pl.Int64).alias("pos"),
            pl.col("ref").cast(pl.Utf8),
            pl.col("alt").cast(pl.Utf8),
            pl.col("GT").cast(pl.Utf8),
        ])
        return lf


def compute_dosage_expr(
    gt_col: str = "GT",
    ref_col: str = "ref",
    alt_col: str = "alt",
    effect_allele_col: str = "effect_allele",
) -> pl.Expr:
    """Build a Polars expression that computes effect allele dosage from a GT string.

    The GT field is a string like "0/0", "0/1", "1/1", "0|1", etc.
    Each allele index: 0 = ref, 1 = first alt, 2 = second alt, etc.

    Dosage = count of alleles in the genotype that match the effect allele.

    For biallelic sites:
    - If effect_allele == alt: dosage = number of '1' alleles in GT
    - If effect_allele == ref: dosage = number of '0' alleles in GT
    - Otherwise: dosage = 0 (allele not found)

    Args:
        gt_col: Name of the GT column
        ref_col: Name of the ref allele column
        alt_col: Name of the alt allele column
        effect_allele_col: Name of the effect allele column

    Returns:
        Polars expression computing dosage as Int64
    """
    gt = pl.col(gt_col)
    ref = pl.col(ref_col)
    alt = pl.col(alt_col)
    effect = pl.col(effect_allele_col)

    gt_normalized = gt.str.replace_all(r"\|", "/")
    allele_1 = gt_normalized.str.split("/").list.get(0)
    allele_2 = gt_normalized.str.split("/").list.get(1)

    allele_1_int = allele_1.cast(pl.Int64, strict=False)
    allele_2_int = allele_2.cast(pl.Int64, strict=False)

    is_missing = allele_1_int.is_null() | allele_2_int.is_null()
    effect_is_alt = effect == alt
    effect_is_ref = effect == ref

    dosage_when_alt = (
        allele_1_int.eq(1).cast(pl.Int64).fill_null(0)
        + allele_2_int.eq(1).cast(pl.Int64).fill_null(0)
    )
    dosage_when_ref = (
        allele_1_int.eq(0).cast(pl.Int64).fill_null(0)
        + allele_2_int.eq(0).cast(pl.Int64).fill_null(0)
    )

    return (
        pl.when(is_missing)
        .then(pl.lit(0))
        .when(effect_is_alt)
        .then(dosage_when_alt)
        .when(effect_is_ref)
        .then(dosage_when_ref)
        .otherwise(pl.lit(0))
        .cast(pl.Int64)
        .alias("dosage")
    )
