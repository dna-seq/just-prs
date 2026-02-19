"""VCF reading via polars-bio and genotype dosage extraction."""

import gzip
import re
from pathlib import Path

import polars as pl
import polars_bio as pb
from eliot import start_action

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

_BUILD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"GRCh38|hg38", re.IGNORECASE), "GRCh38"),
    (re.compile(r"GRCh37|hg19|b37", re.IGNORECASE), "GRCh37"),
]

_CONTIG_RE = re.compile(r"##contig=<.*?ID=(?:chr)?(\w+).*?length=(\d+)", re.IGNORECASE)


def detect_genome_build(vcf_path: Path | str) -> str | None:
    """Attempt to detect the genome build from a VCF file header.

    Inspects ``##reference=`` lines for known build names (GRCh38, hg19, etc.)
    and ``##contig=`` lines for chromosome lengths matching GRCh37 or GRCh38.

    Args:
        vcf_path: Path to VCF file (plain text or gzipped)

    Returns:
        "GRCh37", "GRCh38", or None if detection fails
    """
    vcf_path = Path(vcf_path)
    opener = gzip.open if vcf_path.name.endswith(".gz") else open

    with opener(vcf_path, "rt") as fh:  # type: ignore[arg-type]
        contig_votes: dict[str, int] = {"GRCh37": 0, "GRCh38": 0}

        for line in fh:
            if not line.startswith("##"):
                break

            if line.startswith("##reference="):
                for pattern, build in _BUILD_PATTERNS:
                    if pattern.search(line):
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
    """Read genotypes from a VCF file using polars-bio.

    Returns a LazyFrame with columns: chrom, start (1-based), ref, alt, GT.
    The chrom column is normalized to remove 'chr' prefix for consistent matching.

    Args:
        vcf_path: Path to VCF file (plain or gzipped)

    Returns:
        LazyFrame with genotype data
    """
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
