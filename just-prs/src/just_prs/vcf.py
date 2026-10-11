"""VCF reading via polars-bio and genotype dosage extraction."""

import hashlib
import re
from pathlib import Path
from threading import Lock

import polars as pl
import polars_bio as pb
from eliot import start_action

from just_prs.io_utils import open_maybe_compressed

_VCF_REGISTRATION_LOCK = Lock()
_REGISTERED_VCF_TABLES: set[str] = set()


def _vcf_registration_key(path: Path | str, info_fields: list[str] | None) -> str:
    """Build a stable table key that changes when a local VCF is replaced."""
    source = str(path)
    local_path = Path(path)
    if local_path.is_file():
        stat = local_path.stat()
        source = f"{local_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    fields = "*" if info_fields is None else ",".join(sorted(info_fields))
    digest = hashlib.sha256(f"{source}:{fields}".encode()).hexdigest()[:24]
    return f"just_prs_vcf_{digest}"


def scan_vcf(
    path: Path | str,
    *,
    info_fields: list[str] | None = None,
    format_fields: list[str] | None = None,
) -> pl.LazyFrame:
    """Return a VCF scan that is safe to execute concurrently.

    ``polars_bio.scan_vcf`` re-registers a stem-derived table whenever a LazyFrame
    executes. Concurrent scans of the same path therefore panic with "table already
    exists". Registering each source once under a content-aware unique name and
    querying that named table avoids the conflicting execution-time registration.

    polars-bio discovers FORMAT fields from the header for named registrations.
    Callers still project ``format_fields`` from the returned frame.
    """
    table_name = _vcf_registration_key(path, info_fields)
    with _VCF_REGISTRATION_LOCK:
        if table_name not in _REGISTERED_VCF_TABLES:
            pb.register_vcf(
                str(path),
                name=table_name,
                info_fields=info_fields,
                chunk_size=8,
                concurrent_fetches=1,
            )
            _REGISTERED_VCF_TABLES.add(table_name)
    # Named registration currently exposes DataFusion's native zero-based,
    # half-open coordinates even though register_vcf documents one-based output.
    # Restore the one-based start expected from scan_vcf(use_zero_based=False);
    # the half-open end already equals the corresponding one-based closed end.
    return pb.sql(f'SELECT * FROM "{table_name}"').with_columns(
        (pl.col("start") + 1).alias("start")
    )


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
        lf = scan_vcf(
            vcf_path,
            format_fields=["GT"],
        )
        lf = lf.select([
            pl.col("chrom").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chrom"),
            pl.col("start").cast(pl.Int64).alias("pos"),
            pl.col("ref").cast(pl.Utf8),
            pl.col("alt").cast(pl.Utf8),
            pl.col("GT").cast(pl.Utf8),
        ])
        return lf


def record_alleles_expr(ref_col: str = "ref", alt_col: str = "alt") -> pl.Expr:
    """``[REF, ALT1, ALT2, ...]`` for a VCF record, so a GT index selects its allele.

    polars-bio joins multi-allelic ALTs with ``|`` (``"G|T"``); VCF text and
    other producers use ``,``. Both are split, so allele index ``i`` of the GT
    string is ``list[i]``.
    """
    alts = (
        pl.col(alt_col).cast(pl.Utf8).fill_null("")
        .str.replace_all(",", "|", literal=True)
        .str.split("|")
    )
    return pl.concat_list([pl.col(ref_col).cast(pl.Utf8), alts])


def gt_allele_indices(gt_col: str = "GT") -> tuple[pl.Expr, pl.Expr]:
    """The allele indices of a GT string; the second is null for a haploid call.

    A haploid GT (``"1"``, written on male chrX/chrY and on chrM) is one allele,
    so it contributes one copy at most: a man has one X. A missing allele (``.``)
    or an unparseable index also gives ``None``; :func:`gt_no_call_expr` tells
    the two apart.
    """
    parts = pl.col(gt_col).cast(pl.Utf8).str.replace_all("|", "/", literal=True).str.split_exact("/", 1)
    first = parts.struct.field("field_0").cast(pl.Int64, strict=False)
    second = parts.struct.field("field_1").cast(pl.Int64, strict=False)
    return first, second


def gt_is_haploid_expr(gt_col: str = "GT") -> pl.Expr:
    """True for a one-allele GT (no ``/`` or ``|`` separator)."""
    return ~pl.col(gt_col).cast(pl.Utf8).str.contains(r"[/|]")


def gt_no_call_expr(gt_col: str = "GT") -> pl.Expr:
    """True for a missing GT, ``.``/``./.``, a half call (``./1``) or a bad index."""
    first, second = gt_allele_indices(gt_col)
    return (
        pl.col(gt_col).is_null()
        | first.is_null()
        | (second.is_null() & ~gt_is_haploid_expr(gt_col))
    )


def _called_alts_expr(alt_col: str) -> pl.Expr:
    """ALT alleles that are real sequence: no ``<NON_REF>``-style symbols, ``*`` or ``.``."""
    element = pl.element()
    return (
        pl.col(alt_col).cast(pl.Utf8).fill_null("")
        .str.replace_all(",", "|", literal=True)
        .str.split("|")
        .list.eval(
            element.filter(
                (element != "") & (element != ".") & (element != "*") & ~element.str.starts_with("<")
            )
        )
    )


def record_is_snv_expr(ref_col: str = "ref", alt_col: str = "alt") -> pl.Expr:
    """True when REF is one base and the record carries an SNV allele.

    A multi-allelic record mixing an SNV and an indel ALT (``REF=G, ALT=A|AA``)
    qualifies: the SNV allele is counted and the indel allele, compared as a
    string, never equals a single-base effect allele. A pure insertion
    (``REF=A, ALT=AT``) does not. A gVCF reference block (``ALT=<NON_REF>``)
    has no sequence ALT and qualifies, so its hom-ref call still scores.
    """
    alts = _called_alts_expr(alt_col)
    return (pl.col(ref_col).cast(pl.Utf8).str.len_chars() == 1) & (
        (alts.list.len() == 0) | alts.list.eval(pl.element().str.len_chars() == 1).list.any()
    )


def other_alleles_expr(
    candidates_col: str = "other_candidates", effect_allele_col: str = "effect_allele"
) -> pl.Expr:
    """The ``/``-separated other-allele candidates as a list, without the effect allele."""
    return (
        pl.col(candidates_col).cast(pl.Utf8).fill_null("").str.split("/")
        .list.set_difference(pl.concat_list([pl.col(effect_allele_col).cast(pl.Utf8), pl.lit("")]))
    )


def biallelic_snv_record_expr(ref_col: str = "ref", alt_col: str = "alt") -> pl.Expr:
    """One-base REF and one-character ALT: decided without splitting any list."""
    return (pl.col(ref_col).str.len_chars() == 1) & (pl.col(alt_col).str.len_chars() == 1)


def allele_compatible_expr(
    ref_col: str = "ref",
    alt_col: str = "alt",
    effect_allele_col: str = "effect_allele",
    other_candidates_col: str = "other_candidates",
    effect_is_snv_col: str = "effect_is_snv",
) -> pl.Expr:
    """Whether a VCF record describes the same variant as a scoring row.

    Position alone is not identity: an indel shares its anchor base with any SNV
    at that position, so a position-only join counts an SNV allele against an
    indel allele string.

    * An SNV row (single-base effect allele and other alleles) is scored only
      from a record with a one-base REF and an SNV allele (``record_is_snv_expr``).
      Its dose is the count of the effect base, so a third allele (scoring
      ``T/C``, sample ``T/G``) still counts the REF copy.
    * An indel row needs its effect allele among the record's alleles and, when
      an other allele is known (``other_allele`` or ``hm_inferOtherAllele``), one
      of those too. A record with fewer than two distinct alleles (a consumer
      array's homozygous ``ref=alt`` encoding) skips the other-allele check.

    The row-side columns come from ``prs._normalize_scoring_columns``. This
    splits lists, so engines apply it only to rows that
    :func:`biallelic_snv_record_expr` does not already decide.
    """
    alleles = record_alleles_expr(ref_col, alt_col)
    others = other_alleles_expr(other_candidates_col, effect_allele_col)
    no_other = others.list.len().fill_null(0) == 0
    other_present = others.list.set_intersection(alleles).list.len() > 0
    monomorphic = alleles.list.unique().list.len() < 2
    indel_ok = alleles.list.contains(pl.col(effect_allele_col)) & (no_other | monomorphic | other_present)
    return (
        pl.when(pl.col(effect_is_snv_col))
        .then(record_is_snv_expr(ref_col, alt_col))
        .otherwise(indel_ok)
        .fill_null(False)
    )


def biallelic_dosage_expr(
    gt_col: str = "GT",
    ref_col: str = "ref",
    alt_col: str = "alt",
    effect_allele_col: str = "effect_allele",
) -> pl.Expr:
    """:func:`compute_dosage_expr` for a record with one ALT, without list operations.

    Only valid where ALT holds a single allele; a multi-allelic record needs
    :func:`compute_dosage_expr`. The GT is parsed once here.
    """
    first, second = gt_allele_indices(gt_col)
    effect = pl.col(effect_allele_col)
    effect_is_ref = pl.col(ref_col) == effect
    effect_is_alt = pl.col(alt_col) == effect

    def copies(index: pl.Expr) -> pl.Expr:
        return (
            ((index == 0) & effect_is_ref).cast(pl.Int64) + ((index == 1) & effect_is_alt).cast(pl.Int64)
        ).fill_null(0)

    return (
        pl.when(gt_no_call_expr(gt_col))
        .then(pl.lit(0, dtype=pl.Int64))
        .otherwise(copies(first) + copies(second))
        .cast(pl.Int64)
        .alias("dosage")
    )


def compute_dosage_expr(
    gt_col: str = "GT",
    ref_col: str = "ref",
    alt_col: str = "alt",
    effect_allele_col: str = "effect_allele",
) -> pl.Expr:
    """Build a Polars expression that counts the effect allele in a GT string.

    Each GT index selects an allele from ``[REF, ALT1, ALT2, ...]`` (multi-allelic
    ALTs are split, see :func:`record_alleles_expr`), and the dosage is the number
    of selected alleles equal to the effect allele. A haploid GT has one allele,
    so it gives 0 or 1 (see :func:`gt_allele_indices`). A no-call gives 0; callers that need to tell a
    no-call from a true 0 use :func:`gt_no_call_expr`.

    This compares allele strings only. Whether the record is the scoring row's
    variant at all is :func:`allele_compatible_expr`'s job.

    Args:
        gt_col: Name of the GT column
        ref_col: Name of the ref allele column
        alt_col: Name of the alt allele column
        effect_allele_col: Name of the effect allele column

    Returns:
        Polars expression computing dosage as Int64
    """
    first, second = gt_allele_indices(gt_col)
    alleles = record_alleles_expr(ref_col, alt_col)
    effect = pl.col(effect_allele_col)
    allele_1 = alleles.list.get(first, null_on_oob=True)
    allele_2 = alleles.list.get(second, null_on_oob=True)
    dosage = (allele_1 == effect).cast(pl.Int64).fill_null(0) + (allele_2 == effect).cast(pl.Int64).fill_null(0)
    return (
        pl.when(gt_no_call_expr(gt_col))
        .then(pl.lit(0))
        .otherwise(dosage)
        .cast(pl.Int64)
        .alias("dosage")
    )
