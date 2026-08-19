"""Core PRS computation engine: variant matching, dosage computation, weighted sum."""

from __future__ import annotations

import enum
import gc
import math
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import polars as pl
from eliot import log_message, start_action


class PRSEngine(str, enum.Enum):
    """PRS computation engine selection."""
    POLARS = "polars"
    DUCKDB = "duckdb"


class GenotypeInputMode(str, enum.Enum):
    """How absent scoring loci should be interpreted."""
    AUTO = "auto"
    VARIANT_ONLY = "variant_only"
    ALL_SITES = "all_sites"
    PLINK_PRESENT_ONLY = "plink_present_only"

from just_prs.chip_coverage import Chip, chip_typed_positions
from just_prs.models import PRSResult
from just_prs.scoring import DEFAULT_CACHE_DIR, load_scoring, parse_scoring_file
from just_prs.vcf import compute_dosage_expr, detect_genome_build, read_genotypes

# Restoration scope: which absent scoring positions may be hom-ref filled.
#   False  -> off (no filling)
#   True   -> the whole reference-allele universe (WGS; the universe index IS the set)
#   Chip   -> chip-typed positions (array; eligible = chip set ∩ universe)
#   Path / pl.DataFrame -> a custom (chrom,pos) set (embedder escape hatch)
RestorationScope = bool | Chip | Path | pl.DataFrame


# Columns the universe REF lookup is reduced to (and the order callers can rely on).
_UNIVERSE_COLUMNS = ("chrom", "pos", "ref", "ref_source")


@dataclass(frozen=True)
class ReferenceUniverse:
    """A lazily-scanned reference-allele universe, resolved once and reused.

    The reference-allele universe is **catalog-wide and identical for every PGS
    ID** (~34M rows), so it is resolved once via :func:`prepare_reference_universe`
    and passed into ``compute_prs`` / ``compute_prs_duckdb`` / ``compute_prs_batch``
    via ``reference_universe=``. This is also the dependency-injection entry point:
    an embedder (e.g. just-dna-lite) can supply its own universe or subset.

    **The universe is never materialized in Python.** ``frame`` stays a
    ``LazyFrame`` and ``source_path`` is the parquet that the DuckDB engine joins
    with ``read_parquet``, so DuckDB streams (and spills) the REF set under its own
    memory limit instead of polars hashing 34M rows on the heap. Holding it as a
    ``DataFrame`` cost ~2.7 GB resident per worker and bought nothing: the DuckDB
    join of a 6.9M-variant score against the full universe runs in ~0.3 s.

    ``scope_frame`` is the optional small ``(chrom, pos)`` restriction (a chip's
    typed positions, or a custom set) applied as a SQL semi-join; ``scoped`` records
    that a restriction is in force. When ``scoped`` is False the whole universe is
    eligible (WGS).
    """

    frame: pl.LazyFrame
    genome_build: str
    scoped: bool = False
    source_path: Path | None = None
    scope_frame: pl.LazyFrame | None = None

    @property
    def n_positions(self) -> int:
        """Eligible ``(chrom, pos)`` REF positions — a streaming count, not a collect."""
        return int(self.frame.select(pl.len()).collect().item())


_INJECTED_UNIVERSE_DIR: Path | None = None


def _sink_injected_universe(universe_lf: pl.LazyFrame) -> Path:
    """Stream an injected universe frame to a temp parquet for DuckDB to scan.

    An embedder may inject a ``DataFrame``/``LazyFrame`` rather than a path. The
    DuckDB engine needs a file to ``read_parquet``, and sinking keeps the promise
    that the catalog-wide REF set is never held in Python memory.
    """
    global _INJECTED_UNIVERSE_DIR
    if _INJECTED_UNIVERSE_DIR is None:
        _INJECTED_UNIVERSE_DIR = Path(tempfile.mkdtemp(prefix="just-prs-universe-"))
    path = _INJECTED_UNIVERSE_DIR / f"universe_{uuid.uuid4().hex}.parquet"
    universe_lf.sink_parquet(path)
    return path


def prepare_reference_universe(
    source: Path | str | pl.DataFrame | pl.LazyFrame,
    *,
    genome_build: str = "GRCh38",
    scope: pl.DataFrame | pl.LazyFrame | None = None,
) -> ReferenceUniverse:
    """Resolve the reference-allele universe once into a lazy :class:`ReferenceUniverse`.

    Call this once before scoring many PGS IDs against the same sample, then pass
    the returned handle into ``compute_prs``/``compute_prs_duckdb``/
    ``compute_prs_batch`` (``reference_universe=``). Nothing is read here: the
    universe stays a ``LazyFrame`` over its parquet, and the DuckDB engine joins
    that parquet directly.

    Args:
        source: A path to the universe parquet, or an already-loaded
            ``pl.DataFrame``/``pl.LazyFrame`` (dependency-injection: an embedder may
            supply its own universe or subset). Must expose
            ``chrom, pos, ref, ref_source`` columns. A frame source is streamed to a
            temp parquet so DuckDB can scan it.
        genome_build: Build the universe is in — recorded on the handle for tracing.
        scope: Optional ``(chrom, pos)`` position set restricting the eligible REF
            set (chip ∩ universe). Accepts ``chrom`` or ``chr_norm``. Kept as a
            separate small frame and applied as a semi-join by the engine.

    Returns:
        A :class:`ReferenceUniverse` over the eligible REF set.
    """
    source_path: Path | None = None
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        universe_lf = pl.scan_parquet(source)
    elif isinstance(source, pl.DataFrame):
        universe_lf = source.lazy()
    elif isinstance(source, pl.LazyFrame):
        universe_lf = source
    else:
        raise TypeError(f"Unsupported reference-universe source: {type(source)!r}")

    universe_lf = universe_lf.filter(pl.col("ref").is_not_null()).select(
        pl.col("chrom").cast(pl.Utf8),
        pl.col("pos").cast(pl.Int64),
        pl.col("ref").cast(pl.Utf8),
        pl.col("ref_source").cast(pl.Utf8),
    )

    scope_lf: pl.LazyFrame | None = None
    if scope is not None:
        scope_lf = (scope.lazy() if isinstance(scope, pl.DataFrame) else scope).pipe(
            _select_chrom_pos
        )

    if source_path is None:
        source_path = _sink_injected_universe(universe_lf)
        universe_lf = pl.scan_parquet(source_path)

    frame = universe_lf
    if scope_lf is not None:
        frame = frame.join(scope_lf, on=["chrom", "pos"], how="semi")

    return ReferenceUniverse(
        frame=frame,
        genome_build=genome_build,
        scoped=scope_lf is not None,
        source_path=source_path,
        scope_frame=scope_lf,
    )


def _sql_literal_path(path: Path) -> str:
    """Single-quote a filesystem path for DuckDB ``read_parquet``."""
    return str(path.resolve()).replace("'", "''")


def _geno_table_name(index: int) -> str:
    return f"geno_{index}"


@dataclass
class PreparedGenotypeTables:
    """One DuckDB connection with genomes materialized as tables.

    The genomes do not change between PGS files. Load them once with
    :func:`prepare_genotype_tables`, then pass this handle into
    ``compute_prs_duckdb`` so only the scoring table is replaced. Close the
    handle (or use it as a context manager) when the worker exits.
    """

    conn: duckdb.DuckDBPyConnection
    tables: dict[str, str]
    modes: dict[str, GenotypeInputMode]
    memory_limit: str
    _closed: bool = field(default=False, repr=False)

    def table_name(self, key: str) -> str:
        try:
            return self.tables[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown genotype table key {key!r}; have {sorted(self.tables)}"
            ) from exc

    def mode_for(self, key: str) -> GenotypeInputMode:
        return self.modes[key]

    def close(self) -> None:
        if self._closed:
            return
        self.conn.close()
        self._closed = True

    def __enter__(self) -> PreparedGenotypeTables:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def prepare_genotype_tables(
    samples: Mapping[str, Path | str],
    *,
    memory_limit: str | None = None,
    genotype_input_mode: str | GenotypeInputMode = GenotypeInputMode.AUTO,
) -> PreparedGenotypeTables:
    """Materialize each genome parquet as a DuckDB table on one connection.

    Keeps the columns the scoring SQL reads: ``chrom``, ``pos``, ``GT``, ``ref``,
    ``alt``, and ``filter`` when present (AUTO mode still sees gVCF ``RefCall`` /
    ``NON_REF``). Later scores only replace the ``scoring`` relation.
    """
    if not samples:
        raise ValueError("prepare_genotype_tables requires at least one sample")
    mem_limit = memory_limit or _resolve_duckdb_memory_limit()
    conn = duckdb.connect(config={"memory_limit": mem_limit})
    try:
        conn.execute("SET arrow_large_buffer_size = true")
        tables: dict[str, str] = {}
        modes: dict[str, GenotypeInputMode] = {}
        requested = _normalize_genotype_input_mode(genotype_input_mode)
        for index, (key, raw_path) in enumerate(samples.items()):
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"Genotype parquet for {key!r} not found: {path}")
            geno_lf = _normalize_genotype_columns(pl.scan_parquet(path))
            modes[key] = (
                _infer_genotype_input_mode(geno_lf)
                if requested == GenotypeInputMode.AUTO
                else requested
            )
            raw_names = set(pl.scan_parquet(path).collect_schema().names())
            if "chrom" not in raw_names:
                raise ValueError(f"Genotype parquet for {key!r} has no chrom column: {path}")
            if "GT" not in raw_names:
                raise ValueError(f"Genotype parquet for {key!r} has no GT column: {path}")
            cols = ["chrom"]
            if "pos" in raw_names:
                cols.append("pos")
            elif "start" in raw_names:
                cols.append("start AS pos")
            else:
                raise ValueError(f"Genotype parquet for {key!r} has no pos/start column: {path}")
            cols.append("GT")
            for extra in ("ref", "alt", "filter"):
                if extra in raw_names:
                    cols.append(extra)
            select_sql = ", ".join(cols)
            table = _geno_table_name(index)
            conn.execute(
                f"CREATE TABLE {table} AS SELECT {select_sql} "
                f"FROM read_parquet('{_sql_literal_path(path)}')"
            )
            tables[key] = table
    except Exception:
        conn.close()
        raise
    log_message(
        message_type="prs:genotype_tables_prepared",
        n_samples=len(tables),
        memory_limit=mem_limit,
        keys=list(tables),
    )
    return PreparedGenotypeTables(
        conn=conn,
        tables=tables,
        modes=modes,
        memory_limit=mem_limit,
    )


def _normalize_genotype_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Ensure genotype LazyFrame has the columns expected by compute_prs.

    polars-bio produces ``start`` while compute_prs joins on ``pos``.
    This renames ``start`` → ``pos`` when the caller passes an external
    LazyFrame that hasn't been through ``read_genotypes()``.
    """
    cols = lf.collect_schema().names()
    if "pos" not in cols and "start" in cols:
        lf = lf.rename({"start": "pos"})
    return lf


def _resolve_scoring(
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str,
    cache_dir: Path,
) -> pl.LazyFrame:
    """Resolve a scoring file argument into a LazyFrame.

    Accepts a Path to a local file, a PGS ID string, or an existing LazyFrame.
    """
    if isinstance(scoring_file, pl.LazyFrame):
        return scoring_file
    if isinstance(scoring_file, Path):
        return parse_scoring_file(scoring_file)
    if isinstance(scoring_file, str) and scoring_file.upper().startswith("PGS"):
        return load_scoring(scoring_file, cache_dir=cache_dir, genome_build=genome_build)
    return parse_scoring_file(Path(scoring_file))


DOSAGE_WEIGHT_COLUMNS = ("dosage_0_weight", "dosage_1_weight", "dosage_2_weight")


def is_dosage_weight_format(columns: list[str]) -> bool:
    """Check if a scoring file uses per-dosage-level weights (GenoBoost format)."""
    return all(c in columns for c in DOSAGE_WEIGHT_COLUMNS) and "effect_weight" not in columns


def _normalize_scoring_columns(scoring_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize scoring file columns to use harmonized position columns when available.

    Harmonized files from PGS Catalog have hm_chr and hm_pos columns that should
    be preferred over the original chr_name and chr_position.

    Supports two weight formats:
    - Standard additive: ``effect_weight`` column
    - Per-dosage (GenoBoost): ``dosage_0_weight``, ``dosage_1_weight``,
      ``dosage_2_weight`` columns (non-linear scoring model)
    """
    columns = scoring_lf.collect_schema().names()

    rename_exprs: list[pl.Expr] = []

    if "hm_chr" in columns and "hm_pos" in columns:
        rename_exprs.append(
            pl.col("hm_chr").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chr_name_norm")
        )
        rename_exprs.append(pl.col("hm_pos").cast(pl.Int64).alias("chr_pos_norm"))
    elif "chr_name" in columns and "chr_position" in columns:
        rename_exprs.append(
            pl.col("chr_name").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chr_name_norm")
        )
        rename_exprs.append(pl.col("chr_position").cast(pl.Int64).alias("chr_pos_norm"))
    else:
        raise ValueError(
            f"Scoring file must have (hm_chr, hm_pos) or (chr_name, chr_position). "
            f"Found columns: {columns}"
        )

    if "effect_allele" not in columns:
        raise ValueError(f"Scoring file must have 'effect_allele' column. Found: {columns}")

    dosage_weight = is_dosage_weight_format(columns)

    if dosage_weight:
        for col in DOSAGE_WEIGHT_COLUMNS:
            rename_exprs.append(pl.col(col).cast(pl.Float64))
    elif "effect_weight" in columns:
        rename_exprs.append(pl.col("effect_weight").cast(pl.Float64))
    else:
        raise ValueError(
            f"Scoring file must have 'effect_weight' or dosage weight columns "
            f"(dosage_0_weight, dosage_1_weight, dosage_2_weight). Found: {columns}"
        )

    rename_exprs.append(pl.col("effect_allele").cast(pl.Utf8))

    if "other_allele" in columns:
        rename_exprs.append(pl.col("other_allele").cast(pl.Utf8))

    if "reference_allele" in columns:
        rename_exprs.append(pl.col("reference_allele").cast(pl.Utf8))
    else:
        rename_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("reference_allele"))

    if "allelefrequency_effect" in columns:
        rename_exprs.append(
            pl.col("allelefrequency_effect").cast(pl.Float64, strict=False)
        )

    return scoring_lf.select(rename_exprs)


def _normalize_genotype_input_mode(mode: str | GenotypeInputMode) -> GenotypeInputMode:
    """Validate and normalize genotype input mode values."""
    if isinstance(mode, GenotypeInputMode):
        return mode
    try:
        return GenotypeInputMode(str(mode))
    except ValueError as exc:
        valid = ", ".join(m.value for m in GenotypeInputMode)
        raise ValueError(f"Unknown genotype_input_mode {mode!r}; expected one of: {valid}") from exc


def _infer_genotype_input_mode(genotypes_lf: pl.LazyFrame) -> GenotypeInputMode:
    """Best-effort mode detection from normalized genotype rows.

    Most uploaded VCFs are variant-only. gVCF/all-sites inputs commonly contain
    reference blocks (e.g. ``<NON_REF>`` alleles or ``RefCall`` filters), so we
    detect those conservatively and otherwise default to variant-only semantics.
    """
    cols = genotypes_lf.collect_schema().names()
    sample_exprs: list[pl.Expr] = []
    if "alt" in cols:
        sample_exprs.append(pl.col("alt").cast(pl.Utf8).str.contains("NON_REF", literal=True).any().alias("has_non_ref"))
    if "filter" in cols:
        sample_exprs.append(pl.col("filter").cast(pl.Utf8).str.contains("RefCall", literal=True).any().alias("has_refcall"))
    if not sample_exprs:
        return GenotypeInputMode.VARIANT_ONLY
    sample = genotypes_lf.select(sample_exprs).limit(1).collect()
    if any(bool(sample[col][0]) for col in sample.columns):
        return GenotypeInputMode.ALL_SITES
    return GenotypeInputMode.VARIANT_ONLY


def _resolve_genotype_input_mode(
    mode: str | GenotypeInputMode,
    genotypes_lf: pl.LazyFrame,
) -> GenotypeInputMode:
    """Resolve ``auto`` into an executable genotype input mode."""
    normalized = _normalize_genotype_input_mode(mode)
    if normalized == GenotypeInputMode.AUTO:
        return _infer_genotype_input_mode(genotypes_lf)
    return normalized


def _gt_no_call_expr(gt_col: str = "GT") -> pl.Expr:
    """Return an expression identifying missing/no-call diploid genotypes."""
    gt = pl.col(gt_col).cast(pl.Utf8)
    normalized = gt.str.replace_all(r"\|", "/")
    parts = normalized.str.split("/")
    a0 = parts.list.get(0, null_on_oob=True)
    a1 = parts.list.get(1, null_on_oob=True)
    return gt.is_null() | (gt == ".") | a0.is_null() | a1.is_null() | (a0 == ".") | (a1 == ".")


def _compute_theoretical_stats(
    scoring_lf: pl.LazyFrame,
    schema_names: list[str] | None = None,
) -> tuple[float, float, int] | None:
    """Compute theoretical PRS mean and SD from allele frequencies in the scoring file.

    Under Hardy-Weinberg equilibrium and independent loci:
      E[dosage_i] = 2 * p_i
      Var[dosage_i] = 2 * p_i * (1 - p_i)
      E[PRS] = sum(w_i * 2 * p_i)
      Var[PRS] = sum(w_i^2 * 2 * p_i * (1 - p_i))

    Returns (mean, std, n_variants_with_freq) or None if allelefrequency_effect
    column is absent or has no valid values.
    """
    if schema_names is None:
        schema_names = scoring_lf.collect_schema().names()

    if "allelefrequency_effect" not in schema_names:
        return None
    if "effect_weight" not in schema_names:
        return None

    agg = (
        scoring_lf.filter(
            pl.col("allelefrequency_effect").is_not_null()
            & pl.col("effect_weight").is_not_null()
            & (pl.col("allelefrequency_effect") > 0.0)
            & (pl.col("allelefrequency_effect") < 1.0)
        )
        .select(
            (pl.col("effect_weight") * 2.0 * pl.col("allelefrequency_effect"))
            .sum()
            .alias("mean"),
            (
                pl.col("effect_weight").pow(2)
                * 2.0
                * pl.col("allelefrequency_effect")
                * (1.0 - pl.col("allelefrequency_effect"))
            )
            .sum()
            .alias("variance"),
            pl.len().alias("n_valid"),
        )
        .collect()
    )

    n_valid = int(agg["n_valid"][0])
    if n_valid == 0:
        return None

    mean = float(agg["mean"][0])
    variance = float(agg["variance"][0])
    std = math.sqrt(variance) if variance > 0 else 0.0
    return mean, std, n_valid


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


_VCF_SUFFIXES = (".vcf", ".vcf.gz", ".vcf.bgz")


def _detect_build_mismatch(
    vcf_path: Path | str, genome_build: str
) -> tuple[str | None, bool]:
    """Best-effort: detect the VCF's own genome build and flag a build mismatch (F4).

    Only attempts detection on a real, VCF-suffixed, existing file. Returns
    ``(None, False)`` for pre-normalized genotype inputs (parquet/array/empty path)
    where no VCF header is available — never guesses a mismatch it can't prove.
    """
    text = str(vcf_path)
    if not text or not text.endswith(_VCF_SUFFIXES):
        return None, False
    path = Path(vcf_path)
    if not path.exists():
        return None, False
    try:
        detected = detect_genome_build(path)
    except (OSError, ValueError):
        return None, False
    if detected is None:
        return None, False
    mismatch = detected != genome_build
    if mismatch:
        log_message(
            message_type="prs:build_mismatch",
            pgs_vcf_build=detected,
            scoring_build=genome_build,
            vcf_path=text,
        )
    return detected, mismatch


def _normalize_restoration_scope(
    scope: RestorationScope | None,
    cache_dir: Path,
    genome_build: str,
) -> bool | pl.LazyFrame:
    """Resolve a user-facing scope into ``True`` (whole universe), ``False`` (off),
    or a ``(chrom, pos)`` LazyFrame (restricted set).

    - ``False``/``None`` -> ``False`` (no restoration).
    - ``True`` -> ``True`` (the universe itself is the eligible set; no duplication).
    - ``Chip`` -> the chip's typed positions for ``genome_build`` (GSA ships both
      A2/GRCh38 and A1/GRCh37 manifests); if the chip has no manifest for that build
      it degrades to ``False`` + a log, never silently mis-fills.
    - ``Path`` / ``pl.DataFrame`` -> a custom set (accepts ``chrom`` or ``chr_norm``).
    """
    if scope is False or scope is None:
        return False
    if scope is True:
        return True
    if isinstance(scope, Chip):
        try:
            positions = chip_typed_positions(scope, cache_dir, build=genome_build)
        except (NotImplementedError, ValueError) as exc:
            log_message(
                message_type="prs:restoration_scope_unavailable",
                chip=str(scope),
                genome_build=genome_build,
                reason=str(exc),
            )
            return False
        return positions.lazy().pipe(_select_chrom_pos)
    if isinstance(scope, pl.DataFrame):
        return scope.lazy().pipe(_select_chrom_pos)
    if isinstance(scope, (str, Path)):
        return pl.scan_parquet(scope).pipe(_select_chrom_pos)
    raise ValueError(f"Unsupported restoration scope: {scope!r}")


def _select_chrom_pos(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize a position-set frame to unique ``(chrom, pos)`` (accepts chr_norm)."""
    cols = lf.collect_schema().names()
    chrom_col = "chrom" if "chrom" in cols else ("chr_norm" if "chr_norm" in cols else None)
    if chrom_col is None or "pos" not in cols:
        raise ValueError(f"Position set must have (chrom|chr_norm, pos); got {cols}")
    return lf.select(
        pl.col(chrom_col).cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chrom"),
        pl.col("pos").cast(pl.Int64),
    ).unique()


def _resolve_reference_universe(
    *,
    reference_universe: ReferenceUniverse | None,
    reference_universe_path: Path | str | None,
    reference_restoration: RestorationScope,
    resolved_mode: GenotypeInputMode,
    genome_build: str,
    cache_dir: Path,
) -> ReferenceUniverse | None:
    """Resolve the effective :class:`ReferenceUniverse` for one compute call.

    Restoration only engages in ``variant_only`` mode. A pre-built
    ``reference_universe`` handle takes precedence (dependency injection — the
    scope is already baked in). Otherwise, when restoration is requested and a
    universe path is available, the universe is parsed once from that path (this
    fallback is what keeps the un-injected single-call path correct, though it
    re-parses per call — inject a handle to avoid that). Returns ``None`` when
    restoration is off or unavailable.
    """
    if resolved_mode != GenotypeInputMode.VARIANT_ONLY:
        return None
    if reference_universe is not None:
        return reference_universe
    if reference_restoration is False or reference_universe_path is None:
        return None
    scope = _normalize_restoration_scope(reference_restoration, cache_dir, genome_build)
    if scope is False:
        return None
    sub = scope if isinstance(scope, pl.LazyFrame) else None
    return prepare_reference_universe(
        reference_universe_path, genome_build=genome_build, scope=sub
    )


def _apply_reference_resolution(
    scoring_norm: pl.LazyFrame,
    universe: ReferenceUniverse | None,
) -> pl.LazyFrame:
    """Fill a missing ``reference_allele`` from a prepared :class:`ReferenceUniverse`.

    ``universe`` is the eligible REF set (already scope-restricted — whole universe
    for WGS, chip ∩ universe for arrays). ``None`` means restoration is off.

    Adds a ``ref_resolved_source`` column (``panel``/``fasta``/null) for accounting;
    the column is always present so downstream aggregation can reference it. Only
    positions whose ``reference_allele`` was null/empty are filled; existing values win.

    This is the **polars-engine** path only. ``compute_prs_duckdb`` does the same
    fill in SQL (:func:`_duckdb_scoring_relation`) so neither side of the join is
    hashed on the Python heap; prefer that engine when restoration is on.

    The join is **flipped** so the small scoring side is hashed, not the 34M-row
    universe: the universe is first reduced (via a semi-join) to just the scoring
    positions, yielding a lookup of at most one row per scoring position, which is
    then left-joined back.
    """
    schema = scoring_norm.collect_schema().names()
    if universe is None:
        if "ref_resolved_source" not in schema:
            scoring_norm = scoring_norm.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("ref_resolved_source")
            )
        return scoring_norm

    universe_lf = universe.frame.select(
        pl.col("chrom").alias("_u_chrom"),
        pl.col("pos").alias("_u_pos"),
        pl.col("ref").alias("_u_ref"),
        pl.col("ref_source").alias("_u_src"),
    )
    # Hash the small scoring side: reduce the universe to scoring positions first.
    scoring_positions = scoring_norm.select(
        pl.col("chr_name_norm").alias("_u_chrom"),
        pl.col("chr_pos_norm").alias("_u_pos"),
    ).unique()
    ref_lookup = universe_lf.join(
        scoring_positions, on=["_u_chrom", "_u_pos"], how="semi"
    ).collect()

    ref_unknown = pl.col("reference_allele").is_null() | (
        pl.col("reference_allele").str.len_chars() == 0
    )
    fill_mask = ref_unknown & pl.col("_u_ref").is_not_null()
    return (
        scoring_norm.join(
            ref_lookup.lazy(),
            left_on=["chr_name_norm", "chr_pos_norm"],
            right_on=["_u_chrom", "_u_pos"],
            how="left",
        )
        .with_columns(
            pl.when(fill_mask)
            .then(pl.col("_u_ref"))
            .otherwise(pl.col("reference_allele"))
            .alias("reference_allele"),
            pl.when(fill_mask)
            .then(pl.col("_u_src"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("ref_resolved_source"),
        )
        .drop("_u_ref", "_u_src")
    )


def _assert_sample_build_matches(
    sample_build: str | None, genome_build: str, pgs_id: str
) -> None:
    """Guard against the silent cross-build dead-end.

    When the caller knows the sample's genome build and it differs from the build
    the scoring file is in, a join on ``(chrom, pos)`` would silently match ~0
    variants and return a meaningless score. Raise a clear error instead. A no-op
    when ``sample_build`` is None (build unknown) — never guesses.
    """
    if sample_build is None:
        return
    from just_prs.cleanup import BUILD_NORMALIZATION

    s = BUILD_NORMALIZATION.get(sample_build, sample_build)
    g = BUILD_NORMALIZATION.get(genome_build, genome_build)
    if s != g:
        raise ValueError(
            f"Genome-build mismatch for {pgs_id}: sample genotypes are {sample_build!r} "
            f"but scoring is {genome_build!r}. Matching on (chrom,pos) across builds would "
            f"silently score ~0 variants. Lift the sample to {genome_build} first — e.g. "
            f"compute_array_prs(..., target_build={genome_build!r}) for arrays, or "
            f"just_prs.liftover.lift_frame(...) for a VCF — or pass the matching build."
        )


def compute_prs(
    vcf_path: Path | str,
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    pgs_id: str = "unknown",
    trait_reported: str | None = None,
    genotypes_lf: pl.LazyFrame | None = None,
    genotype_input_mode: str | GenotypeInputMode = GenotypeInputMode.AUTO,
    maf_fill: bool = False,
    reference_restoration: RestorationScope = False,
    reference_universe_path: Path | str | None = None,
    reference_universe: ReferenceUniverse | None = None,
    sample_build: str | None = None,
) -> PRSResult:
    """Compute a polygenic risk score for a single VCF against a scoring file.

    Algorithm:
    1. Read genotypes from VCF (chrom, pos, ref, alt, GT)
    2. Parse/load scoring file (chr_name, chr_position, effect_allele, effect_weight)
    3. Normalize chromosome names (strip 'chr' prefix)
    4. Inner join on (chrom == chr_name, pos == chr_position)
    5. Compute dosage of effect allele from GT
    6. PRS = sum(effect_weight * dosage)
    7. If allelefrequency_effect is present, compute theoretical mean/SD
       and estimate population percentile.

    Args:
        vcf_path: Path to VCF file (ignored when *genotypes_lf* is provided)
        scoring_file: Path to scoring file, PGS ID string, or pre-loaded LazyFrame
        genome_build: Genome build for downloading scoring files
        cache_dir: Cache directory for downloaded scoring files
        pgs_id: PGS ID for result labeling
        trait_reported: Trait name for result labeling
        genotypes_lf: Pre-built genotypes LazyFrame with columns
            ``chrom, pos, ref, alt, GT`` (``start`` is accepted as an
            alias for ``pos``).  When provided, *vcf_path* is not read —
            useful for passing a normalized parquet via
            ``pl.scan_parquet()``.
        genotype_input_mode: How absent scoring loci are interpreted:
            ``auto`` (default), ``variant_only``, ``all_sites``, or
            ``plink_present_only``.
        maf_fill: When True and the scoring file has ``allelefrequency_effect``,
            substitute ``dosage = 2 * MAF`` for absent variants that would
            otherwise be unscorable. Tracked as ``variants_maf_filled``.

    Returns:
        PRSResult with computed score, match statistics, and optionally
        theoretical distribution stats and percentile.
    """
    _assert_sample_build_matches(sample_build, genome_build, pgs_id)
    with start_action(
        action_type="prs:compute",
        vcf_path=str(vcf_path),
        pgs_id=pgs_id,
        genome_build=genome_build,
    ):
        if genotypes_lf is None:
            genotypes_lf = read_genotypes(vcf_path)
        else:
            genotypes_lf = _normalize_genotype_columns(genotypes_lf)
        scoring_lf = _resolve_scoring(scoring_file, genome_build, cache_dir)
        scoring_norm = _normalize_scoring_columns(scoring_lf)

        schema_names = scoring_norm.collect_schema().names()
        variants_total = scoring_norm.select(pl.len()).collect().item()
        resolved_mode = _resolve_genotype_input_mode(genotype_input_mode, genotypes_lf)

        # Reference-allele resolution only affects the variant-only absent-locus
        # path; in other modes absent loci are never imputed hom-ref. A prepared
        # universe handle (dependency injection) is reused as-is; otherwise it is
        # parsed once from reference_universe_path.
        ref_universe = _resolve_reference_universe(
            reference_universe=reference_universe,
            reference_universe_path=reference_universe_path,
            reference_restoration=reference_restoration,
            resolved_mode=resolved_mode,
            genome_build=genome_build,
            cache_dir=cache_dir,
        )
        scoring_norm = _apply_reference_resolution(scoring_norm, ref_universe)

        if resolved_mode == GenotypeInputMode.VARIANT_ONLY:
            joined = scoring_norm.join(
                genotypes_lf,
                left_on=["chr_name_norm", "chr_pos_norm"],
                right_on=["chrom", "pos"],
                how="left",
            )
        else:
            joined = genotypes_lf.join(
                scoring_norm,
                left_on=["chrom", "pos"],
                right_on=["chr_name_norm", "chr_pos_norm"],
                how="inner",
            )

        joined = joined.with_columns(
            pl.col("GT").is_not_null().alias("is_present"),
            _gt_no_call_expr().alias("is_no_call"),
            compute_dosage_expr(
                gt_col="GT",
                ref_col="ref",
                alt_col="alt",
                effect_allele_col="effect_allele",
            )
        )

        dosage_weight = DOSAGE_WEIGHT_COLUMNS[0] in schema_names
        has_maf_col = "allelefrequency_effect" in schema_names
        do_maf_fill = maf_fill and has_maf_col and not dosage_weight

        # Per-variant weight mass for C_wt (weight-mass coverage). Standard additive
        # scores use |effect_weight|; per-dosage (GenoBoost) scores have no single beta,
        # so use the largest absolute per-dosage weight as the mass surrogate.
        if dosage_weight:
            variant_mass_expr = pl.max_horizontal(
                pl.col("dosage_0_weight").abs(),
                pl.col("dosage_1_weight").abs(),
                pl.col("dosage_2_weight").abs(),
            )
        else:
            variant_mass_expr = pl.col("effect_weight").abs()
        weight_mass_total = float(
            scoring_norm.select(variant_mass_expr.sum().alias("m")).collect().item() or 0.0
        )

        if resolved_mode == GenotypeInputMode.VARIANT_ONLY:
            ref_known = pl.col("reference_allele").is_not_null() & (pl.col("reference_allele").str.len_chars() > 0)
            absent = pl.col("is_present").not_()

            dosage_chain = (
                pl.when(pl.col("is_present") & pl.col("is_no_call").not_())
                .then(pl.col("dosage"))
                .when(absent & ref_known & (pl.col("effect_allele") == pl.col("reference_allele")))
                .then(pl.lit(2))
                .when(absent & ref_known)
                .then(pl.lit(0))
            )
            if do_maf_fill:
                maf_available = pl.col("allelefrequency_effect").is_not_null() & (pl.col("allelefrequency_effect") > 0.0) & (pl.col("allelefrequency_effect") < 1.0)
                dosage_chain = dosage_chain.when(absent & ref_known.not_() & maf_available).then(
                    (2.0 * pl.col("allelefrequency_effect")).cast(pl.Float64)
                )
            dosage_chain = dosage_chain.otherwise(pl.lit(None, dtype=pl.Float64))

            joined = joined.with_columns(dosage_chain.alias("resolved_dosage"))

            if do_maf_fill:
                joined = joined.with_columns(
                    (absent & ref_known.not_() & pl.col("allelefrequency_effect").is_not_null() & (pl.col("allelefrequency_effect") > 0.0) & (pl.col("allelefrequency_effect") < 1.0) & pl.col("resolved_dosage").is_not_null())
                    .alias("is_maf_filled")
                )
            else:
                joined = joined.with_columns(pl.lit(False).alias("is_maf_filled"))
        else:
            joined = joined.with_columns(
                pl.when(pl.col("is_no_call"))
                .then(pl.lit(None, dtype=pl.Int64))
                .otherwise(pl.col("dosage"))
                .alias("resolved_dosage"),
                pl.lit(False).alias("is_maf_filled"),
            )

        if dosage_weight:
            joined = joined.with_columns(
                pl.when(pl.col("resolved_dosage") == 0)
                .then(pl.col("dosage_0_weight"))
                .when(pl.col("resolved_dosage") == 1)
                .then(pl.col("dosage_1_weight"))
                .when(pl.col("resolved_dosage") == 2)
                .then(pl.col("dosage_2_weight"))
                .otherwise(pl.lit(0.0))
                .alias("weighted_dosage")
            )
        else:
            joined = joined.with_columns(
                (pl.col("effect_weight") * pl.col("resolved_dosage").fill_null(0)).alias("weighted_dosage")
            )

        absent_expr = pl.col("is_present").not_()
        ref_known_expr = pl.col("reference_allele").is_not_null() & (pl.col("reference_allele").str.len_chars() > 0)
        agg = joined.select(
            pl.col("is_present").cast(pl.Int64).sum().alias("variants_observed"),
            (pl.col("is_present") & pl.col("is_no_call").not_()).cast(pl.Int64).sum().alias("observed_called"),
            (absent_expr & ref_known_expr).cast(pl.Int64).sum().alias("variants_assumed_hom_ref"),
            (absent_expr & ref_known_expr.not_() & pl.col("is_maf_filled").not_()).cast(pl.Int64).sum().alias("variants_unscorable_absent"),
            (pl.col("is_present") & pl.col("is_no_call")).cast(pl.Int64).sum().alias("variants_no_call"),
            pl.col("is_maf_filled").cast(pl.Int64).sum().alias("variants_maf_filled"),
            (absent_expr & (pl.col("ref_resolved_source") == "panel")).cast(pl.Int64).sum().alias("variants_ref_resolved_panel"),
            (absent_expr & (pl.col("ref_resolved_source") == "fasta")).cast(pl.Int64).sum().alias("variants_ref_resolved_fasta"),
            pl.col("weighted_dosage").sum().alias("prs_score"),
            (
                pl.when(pl.col("resolved_dosage").is_not_null())
                .then(variant_mass_expr)
                .otherwise(0.0)
            ).sum().alias("weight_mass_matched"),
        ).collect()

        variants_observed = int(agg["variants_observed"][0] or 0)
        variants_assumed_hom_ref = int(agg["variants_assumed_hom_ref"][0] or 0)
        variants_unscorable_absent = int(agg["variants_unscorable_absent"][0] or 0)
        variants_no_call = int(agg["variants_no_call"][0] or 0)
        variants_maf_filled = int(agg["variants_maf_filled"][0] or 0)
        variants_ref_resolved_panel = int(agg["variants_ref_resolved_panel"][0] or 0)
        variants_ref_resolved_fasta = int(agg["variants_ref_resolved_fasta"][0] or 0)
        variants_matched = int(agg["observed_called"][0] or 0) + variants_assumed_hom_ref + variants_maf_filled

        if variants_matched == 0:
            prs_score = 0.0
        else:
            prs_score = float(agg["prs_score"][0] or 0.0)

        match_rate = variants_matched / variants_total if variants_total > 0 else 0.0
        weight_mass_matched = float(agg["weight_mass_matched"][0] or 0.0)
        weight_mass_coverage = (
            weight_mass_matched / weight_mass_total if weight_mass_total > 0 else None
        )

        has_freqs = False
        theoretical_mean: float | None = None
        theoretical_std: float | None = None
        percentile: float | None = None
        percentile_method: str | None = None
        z_score: float | None = None
        reference_mean: float | None = None
        reference_std: float | None = None

        stats = _compute_theoretical_stats(scoring_norm, schema_names)
        if stats is not None:
            mean, std, n_with_freq = stats
            has_freqs = True
            theoretical_mean = mean
            theoretical_std = std
            if std > 0:
                z = (prs_score - mean) / std
                percentile = round(_norm_cdf(z) * 100.0, 2)
                percentile_method = "theoretical"
                z_score = z
                reference_mean = mean
                reference_std = std
            log_message(
                message_type="prs:theoretical_stats",
                pgs_id=pgs_id,
                variants_with_frequency=n_with_freq,
                variants_total=variants_total,
                theoretical_mean=mean,
                theoretical_std=std,
                percentile=percentile,
            )

        detected_build, build_mismatch = _detect_build_mismatch(vcf_path, genome_build)

        return PRSResult(
            pgs_id=pgs_id,
            score=prs_score,
            variants_matched=variants_matched,
            variants_total=int(variants_total),
            match_rate=float(match_rate),
            variants_observed=variants_observed,
            variants_assumed_hom_ref=variants_assumed_hom_ref,
            variants_unscorable_absent=variants_unscorable_absent,
            variants_no_call=variants_no_call,
            variants_maf_filled=variants_maf_filled,
            variants_ref_resolved_panel=variants_ref_resolved_panel,
            variants_ref_resolved_fasta=variants_ref_resolved_fasta,
            weight_mass_matched=weight_mass_matched,
            weight_mass_total=weight_mass_total,
            weight_mass_coverage=weight_mass_coverage,
            genotype_input_mode=resolved_mode.value,
            detected_genome_build=detected_build,
            build_mismatch=build_mismatch,
            trait_reported=trait_reported,
            has_allele_frequencies=has_freqs,
            theoretical_mean=theoretical_mean,
            theoretical_std=theoretical_std,
            percentile=percentile,
            percentile_method=percentile_method,
            z_score=z_score,
            reference_mean=reference_mean,
            reference_std=reference_std,
        )


_DUCKDB_DOSAGE_SQL = """\
CASE
    WHEN g.GT IS NULL OR g.GT = './.' OR g.GT = '.' THEN 0
    WHEN s.effect_allele = g.alt THEN
        (CASE WHEN split_part(replace(replace(g.GT, '|', '/'), './', '0/'), '/', 1) = '1' THEN 1 ELSE 0 END
       + CASE WHEN split_part(replace(replace(g.GT, '|', '/'), './', '0/'), '/', 2) = '1' THEN 1 ELSE 0 END)
    WHEN s.effect_allele = g.ref THEN
        (CASE WHEN split_part(replace(replace(g.GT, '|', '/'), './', '0/'), '/', 1) = '0' THEN 1 ELSE 0 END
       + CASE WHEN split_part(replace(replace(g.GT, '|', '/'), './', '0/'), '/', 2) = '0' THEN 1 ELSE 0 END)
    ELSE 0
END"""

_DUCKDB_NO_CALL_SQL = """\
(g.GT IS NULL OR g.GT = './.' OR g.GT = '.'
 OR split_part(replace(g.GT, '|', '/'), '/', 1) = '.'
 OR split_part(replace(g.GT, '|', '/'), '/', 2) = '.')"""

_DUCKDB_REFERENCE_KNOWN_SQL = """\
(s.reference_allele IS NOT NULL AND length(s.reference_allele) > 0)"""

_DUCKDB_RESOLVED_DOSAGE_PRESENT_ONLY = f"""\
CASE
    WHEN {_DUCKDB_NO_CALL_SQL} THEN NULL
    ELSE ({_DUCKDB_DOSAGE_SQL})
END"""

_DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY = f"""\
CASE
    WHEN g.GT IS NOT NULL AND NOT ({_DUCKDB_NO_CALL_SQL}) THEN ({_DUCKDB_DOSAGE_SQL})
    WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} AND s.effect_allele = s.reference_allele THEN 2
    WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} THEN 0
    ELSE NULL
END"""

_DUCKDB_MAF_AVAILABLE_SQL = """\
(s.allelefrequency_effect IS NOT NULL AND s.allelefrequency_effect > 0.0 AND s.allelefrequency_effect < 1.0)"""

_DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY_MAF = f"""\
CASE
    WHEN g.GT IS NOT NULL AND NOT ({_DUCKDB_NO_CALL_SQL}) THEN ({_DUCKDB_DOSAGE_SQL})
    WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} AND s.effect_allele = s.reference_allele THEN 2
    WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} THEN 0
    WHEN g.GT IS NULL AND NOT ({_DUCKDB_REFERENCE_KNOWN_SQL}) AND {_DUCKDB_MAF_AVAILABLE_SQL} THEN 2.0 * s.allelefrequency_effect
    ELSE NULL
END"""

_DUCKDB_WEIGHTED_DOSAGE_ADDITIVE = f"""\
s.effect_weight * resolved_dosage"""

_DUCKDB_WEIGHTED_DOSAGE_GENOBOOST = """\
CASE resolved_dosage
    WHEN 0 THEN s.dosage_0_weight
    WHEN 1 THEN s.dosage_1_weight
    WHEN 2 THEN s.dosage_2_weight
    ELSE 0.0
END"""


_DEFAULT_DUCKDB_MEMORY_PERCENT = 75


def _resolve_duckdb_memory_limit() -> str:
    """Compute DuckDB per-connection memory limit.

    Resolution order:
      1. ``PRS_DUCKDB_MEMORY_LIMIT`` env var (e.g. ``"8GB"``) — used as-is.
      2. ``PRS_DUCKDB_MEMORY_PERCENT`` env var — percentage of total RAM.
      3. Default: 75% of total RAM.
    """
    import os

    import psutil

    explicit = os.environ.get("PRS_DUCKDB_MEMORY_LIMIT", "").strip()
    if explicit:
        return explicit

    total_bytes = psutil.virtual_memory().total
    pct_str = os.environ.get("PRS_DUCKDB_MEMORY_PERCENT", "").strip()
    pct = int(pct_str) if pct_str else _DEFAULT_DUCKDB_MEMORY_PERCENT
    limit_bytes = int(total_bytes * pct / 100)
    limit_gb = max(limit_bytes / (1024**3), 1.0)
    return f"{limit_gb:.1f}GB"


@dataclass
class _DuckDbScoreAgg:
    """Additive per-chunk totals from the DuckDB scoring join."""

    prs_score: float = 0.0
    observed_called: int = 0
    variants_observed: int = 0
    variants_assumed_hom_ref: int = 0
    variants_unscorable_absent: int = 0
    variants_no_call: int = 0
    variants_maf_filled: int = 0
    variants_ref_resolved_panel: int = 0
    variants_ref_resolved_fasta: int = 0
    weight_mass_matched: float = 0.0

    def add(self, other: _DuckDbScoreAgg) -> None:
        self.prs_score += other.prs_score
        self.observed_called += other.observed_called
        self.variants_observed += other.variants_observed
        self.variants_assumed_hom_ref += other.variants_assumed_hom_ref
        self.variants_unscorable_absent += other.variants_unscorable_absent
        self.variants_no_call += other.variants_no_call
        self.variants_maf_filled += other.variants_maf_filled
        self.variants_ref_resolved_panel += other.variants_ref_resolved_panel
        self.variants_ref_resolved_fasta += other.variants_ref_resolved_fasta
        self.weight_mass_matched += other.weight_mass_matched


def _scoring_weight_mass_total(scoring_norm: pl.LazyFrame, *, dosage_weight: bool) -> float:
    if dosage_weight:
        expr = pl.max_horizontal(
            pl.col("dosage_0_weight").abs(),
            pl.col("dosage_1_weight").abs(),
            pl.col("dosage_2_weight").abs(),
        ).sum()
    else:
        expr = pl.col("effect_weight").abs().sum()
    value = scoring_norm.select(expr.alias("mass")).collect().item()
    return float(value or 0.0)


def _scoring_theoretical_stats(
    scoring_norm: pl.LazyFrame,
    schema_names: list[str],
) -> tuple[bool, float | None, float | None]:
    if "allelefrequency_effect" not in schema_names or "effect_weight" not in schema_names:
        return False, None, None
    row = (
        scoring_norm.filter(
            pl.col("allelefrequency_effect").is_not_null()
            & pl.col("effect_weight").is_not_null()
            & (pl.col("allelefrequency_effect") > 0.0)
            & (pl.col("allelefrequency_effect") < 1.0)
        )
        .select(
            (pl.col("effect_weight") * 2.0 * pl.col("allelefrequency_effect"))
            .sum()
            .alias("mean"),
            (
                pl.col("effect_weight").pow(2)
                * 2.0
                * pl.col("allelefrequency_effect")
                * (1.0 - pl.col("allelefrequency_effect"))
            )
            .sum()
            .alias("variance"),
            pl.len().alias("n_valid"),
        )
        .collect()
    )
    n_valid = int(row["n_valid"][0])
    if n_valid <= 0:
        return False, None, None
    mean = float(row["mean"][0])
    variance = float(row["variance"][0])
    std = math.sqrt(variance) if variance > 0 else 0.0
    return True, mean, std


_DUCKDB_SCORING_RELATION = "scoring_resolved"


def _duckdb_universe_sql(
    conn: duckdb.DuckDBPyConnection, universe: ReferenceUniverse
) -> str:
    """SQL for the eligible REF set, read straight from the universe parquet.

    ``read_parquet`` keeps the ~34M-row REF set inside DuckDB, which streams it
    under the connection's ``memory_limit`` and spills to disk if needed — the same
    reason the reference-panel path scans the 75M-row ``.pvar`` instead of loading
    it into polars. A restoration scope is a semi-join against the registered
    position set; scopes are small and bounded (a chip is ~648K positions), unlike
    the universe itself.
    """
    scope_sql = ""
    if universe.scope_frame is not None:
        conn.register("restoration_scope", universe.scope_frame.collect().to_arrow())
        scope_sql = (
            " AND EXISTS (SELECT 1 FROM restoration_scope rs"
            " WHERE rs.chrom = u.chrom AND rs.pos = u.pos)"
        )
    return (
        "SELECT u.chrom AS _u_chrom, u.pos AS _u_pos,"
        " u.ref AS _u_ref, u.ref_source AS _u_src"
        f" FROM read_parquet('{universe.source_path}') u"
        f" WHERE u.ref IS NOT NULL{scope_sql}"
    )


def _duckdb_scoring_relation(
    conn: duckdb.DuckDBPyConnection, *, universe_sql: str | None
) -> None:
    """Expose the registered ``scoring`` rows as ``scoring_resolved``.

    With restoration on, a null/empty ``reference_allele`` is filled by a SQL LEFT
    JOIN against the universe relation, and ``ref_resolved_source`` records which
    tier supplied it. With restoration off the column is a typed null so the
    scoring SQL can reference it unconditionally.
    """
    if universe_sql is None:
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW {_DUCKDB_SCORING_RELATION} AS "
            "SELECT *, CAST(NULL AS VARCHAR) AS ref_resolved_source FROM scoring"
        )
        return
    conn.execute(f"""
        CREATE OR REPLACE TEMP VIEW {_DUCKDB_SCORING_RELATION} AS
        SELECT
            sc.* EXCLUDE (reference_allele),
            COALESCE(NULLIF(sc.reference_allele, ''), u._u_ref) AS reference_allele,
            CASE WHEN NULLIF(sc.reference_allele, '') IS NULL THEN u._u_src END
                AS ref_resolved_source
        FROM scoring sc
        LEFT JOIN ({universe_sql}) u
          ON u._u_chrom = sc.chr_name_norm AND u._u_pos = sc.chr_pos_norm
    """)


def _duckdb_score_registered(
    conn: duckdb.DuckDBPyConnection,
    *,
    join_sql: str,
    resolved_dosage_sql: str,
    counter_sql: str,
    weighted_sql: str,
    variant_mass_sql: str,
) -> _DuckDbScoreAgg:
    row = conn.execute(f"""
        WITH resolved AS (
            SELECT
                s.*,
                g.GT,
                {resolved_dosage_sql} AS resolved_dosage,
                {variant_mass_sql} AS variant_mass,
                {counter_sql}
            {join_sql}
            GROUP BY ALL
        )
        SELECT
            COALESCE(SUM({weighted_sql}), 0.0) AS prs_score,
            COALESCE(SUM(observed_called), 0) AS observed_called,
            COALESCE(SUM(variants_observed), 0) AS variants_observed,
            COALESCE(SUM(variants_assumed_hom_ref), 0) AS variants_assumed_hom_ref,
            COALESCE(SUM(variants_unscorable_absent), 0) AS variants_unscorable_absent,
            COALESCE(SUM(variants_no_call), 0) AS variants_no_call,
            COALESCE(SUM(variants_maf_filled), 0) AS variants_maf_filled,
            COALESCE(SUM(variants_ref_resolved_panel), 0) AS variants_ref_resolved_panel,
            COALESCE(SUM(variants_ref_resolved_fasta), 0) AS variants_ref_resolved_fasta,
            COALESCE(SUM(CASE WHEN resolved_dosage IS NOT NULL THEN variant_mass ELSE 0.0 END), 0.0) AS weight_mass_matched
        FROM resolved s
    """).fetchone()
    assert row is not None
    return _DuckDbScoreAgg(
        prs_score=float(row[0]),
        observed_called=int(row[1]),
        variants_observed=int(row[2]),
        variants_assumed_hom_ref=int(row[3]),
        variants_unscorable_absent=int(row[4]),
        variants_no_call=int(row[5]),
        variants_maf_filled=int(row[6]),
        variants_ref_resolved_panel=int(row[7]),
        variants_ref_resolved_fasta=int(row[8]),
        weight_mass_matched=float(row[9] or 0.0),
    )


def compute_prs_duckdb(
    vcf_path: Path | str,
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    pgs_id: str = "unknown",
    trait_reported: str | None = None,
    genotypes_parquet: Path | str | None = None,
    genotypes_lf: pl.LazyFrame | None = None,
    genotype_tables: PreparedGenotypeTables | None = None,
    genotype_table_key: str | None = None,
    memory_limit: str | None = None,
    genotype_input_mode: str | GenotypeInputMode = GenotypeInputMode.AUTO,
    maf_fill: bool = False,
    reference_restoration: RestorationScope = False,
    reference_universe_path: Path | str | None = None,
    reference_universe: ReferenceUniverse | None = None,
    sample_build: str | None = None,
) -> PRSResult:
    """Compute a polygenic risk score using DuckDB for the join and aggregation.

    Functionally equivalent to ``compute_prs()`` but uses DuckDB SQL instead of
    polars for the variant-matching join and weighted-sum aggregation. Scoring
    files larger than ``PRS_SCORING_JOIN_CHUNK_SIZE`` (default 250_000 rows)
    are joined in bounded chunks — the same idea as reference-panel genotype
    chunking — so a 9.5M-variant score cannot native-crash one giant join.
    ``check_memory_pressure`` runs before each chunk.

    Either *genotype_tables* (preferred when scoring many PGS IDs against the
    same genomes — tables stay loaded, only ``scoring`` is replaced),
    *genotypes_parquet* (DuckDB reads the file directly), or *genotypes_lf*
    (materialized to Arrow) must be provided. If none of those is given, the
    VCF is read via polars-bio and materialized to a temporary Arrow table.

    Args:
        vcf_path: Path to VCF file (used only when neither genotypes arg is provided)
        scoring_file: Path to scoring file, PGS ID string, or pre-loaded LazyFrame
        genome_build: Genome build for downloading scoring files
        cache_dir: Cache directory for downloaded scoring files
        pgs_id: PGS ID for result labeling
        trait_reported: Trait name for result labeling
        genotypes_parquet: Path to normalized genotypes parquet (best for DuckDB)
        genotypes_lf: Pre-built genotypes LazyFrame (collected to Arrow for DuckDB)
        genotype_tables: Prepared DuckDB tables (genomes loaded once). When set,
            ``genotype_table_key`` selects which table to join and the connection
            is not closed.
        genotype_table_key: Sample key in ``genotype_tables.tables``.
        memory_limit: DuckDB memory limit (e.g. ``"2GB"``). Falls back to
            ``PRS_DUCKDB_MEMORY_LIMIT`` / ``PRS_DUCKDB_MEMORY_PERCENT`` env vars,
            then 75% of total RAM.
        genotype_input_mode: How absent scoring loci are interpreted:
            ``auto`` (default), ``variant_only``, ``all_sites``, or
            ``plink_present_only``.
        maf_fill: When True and the scoring file has ``allelefrequency_effect``,
            substitute ``dosage = 2 * MAF`` for absent variants that would
            otherwise be unscorable. Tracked as ``variants_maf_filled``.

    Returns:
        PRSResult with computed score, match statistics, and optionally
        theoretical distribution stats and percentile.
    """
    _assert_sample_build_matches(sample_build, genome_build, pgs_id)
    with start_action(
        action_type="prs:compute_duckdb",
        vcf_path=str(vcf_path),
        pgs_id=pgs_id,
        genome_build=genome_build,
    ):
        scoring_lf = _resolve_scoring(scoring_file, genome_build, cache_dir)
        scoring_norm = _normalize_scoring_columns(scoring_lf)

        # Resolve the genotype input mode up front (cheap — schema + 1-row probe)
        # so reference restoration can be gated on variant_only before the scoring
        # frame is collected. ``geno_mode_lf`` is reused for DuckDB registration
        # below so the VCF/parquet is not read twice. A prepared table session
        # already resolved the mode when the genomes were loaded.
        shared_tables = genotype_tables
        geno_mode_lf: pl.LazyFrame | None = None
        if shared_tables is not None:
            if not genotype_table_key:
                raise ValueError("genotype_table_key is required when genotype_tables is set")
            resolved_mode = shared_tables.mode_for(genotype_table_key)
            requested_mode = _normalize_genotype_input_mode(genotype_input_mode)
            if requested_mode != GenotypeInputMode.AUTO:
                resolved_mode = requested_mode
        elif genotypes_parquet is not None:
            geno_mode_lf = _normalize_genotype_columns(pl.scan_parquet(genotypes_parquet))
            resolved_mode = _resolve_genotype_input_mode(genotype_input_mode, geno_mode_lf)
        elif genotypes_lf is not None:
            geno_mode_lf = _normalize_genotype_columns(genotypes_lf)
            resolved_mode = _resolve_genotype_input_mode(genotype_input_mode, geno_mode_lf)
        else:
            geno_mode_lf = read_genotypes(vcf_path)
            resolved_mode = _resolve_genotype_input_mode(genotype_input_mode, geno_mode_lf)

        # Resolve the REF universe handle (restoration only engages in variant_only
        # mode). The fill itself happens in SQL against the universe parquet — see
        # ``_duckdb_scoring_relation``, which also supplies the ``ref_resolved_source``
        # column (typed null when restoration is off) that the scoring SQL reads.
        ref_universe = _resolve_reference_universe(
            reference_universe=reference_universe,
            reference_universe_path=reference_universe_path,
            reference_restoration=reference_restoration,
            resolved_mode=resolved_mode,
            genome_build=genome_build,
            cache_dir=cache_dir,
        )

        from just_prs.memory import check_memory_pressure, scoring_join_chunk_size

        schema_names = scoring_norm.collect_schema().names()
        variants_total = int(scoring_norm.select(pl.len()).collect().item())
        check_memory_pressure(pgs_id)

        dosage_weight = DOSAGE_WEIGHT_COLUMNS[0] in schema_names
        weighted_sql = _DUCKDB_WEIGHTED_DOSAGE_GENOBOOST if dosage_weight else _DUCKDB_WEIGHTED_DOSAGE_ADDITIVE
        if dosage_weight:
            variant_mass_sql = "greatest(abs(s.dosage_0_weight), abs(s.dosage_1_weight), abs(s.dosage_2_weight))"
        else:
            variant_mass_sql = "abs(s.effect_weight)"

        weight_mass_total = _scoring_weight_mass_total(scoring_norm, dosage_weight=dosage_weight)
        chunk_size = scoring_join_chunk_size(variants_total)
        chunked = variants_total > chunk_size > 0

        mem_limit = memory_limit or (
            shared_tables.memory_limit if shared_tables is not None else _resolve_duckdb_memory_limit()
        )
        own_conn = shared_tables is None
        if shared_tables is not None:
            conn = shared_tables.conn
            geno_from = shared_tables.table_name(str(genotype_table_key))
        else:
            conn = duckdb.connect(config={"memory_limit": mem_limit})
            conn.execute("SET arrow_large_buffer_size = true")
            if genotypes_parquet is not None:
                geno_from = f"read_parquet('{_sql_literal_path(Path(genotypes_parquet))}')"
            else:
                assert geno_mode_lf is not None
                conn.register("genotypes_tbl", geno_mode_lf.collect().to_arrow())
                geno_from = "genotypes_tbl"
        try:

            has_maf_col_ddb = "allelefrequency_effect" in schema_names
            do_maf_fill_ddb = maf_fill and has_maf_col_ddb and not dosage_weight

            universe_sql = (
                _duckdb_universe_sql(conn, ref_universe) if ref_universe is not None else None
            )

            if resolved_mode == GenotypeInputMode.VARIANT_ONLY:
                join_sql = f"""
                    FROM {_DUCKDB_SCORING_RELATION} s
                    LEFT JOIN {geno_from} g
                      ON g.chrom = s.chr_name_norm AND g.pos = s.chr_pos_norm
                """
                if do_maf_fill_ddb:
                    resolved_dosage_sql = _DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY_MAF
                    counter_sql = f"""
                        SUM(CASE WHEN g.GT IS NOT NULL THEN 1 ELSE 0 END) AS variants_observed,
                        SUM(CASE WHEN g.GT IS NOT NULL AND NOT ({_DUCKDB_NO_CALL_SQL}) THEN 1 ELSE 0 END) AS observed_called,
                        SUM(CASE WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} THEN 1 ELSE 0 END) AS variants_assumed_hom_ref,
                        SUM(CASE WHEN g.GT IS NULL AND NOT ({_DUCKDB_REFERENCE_KNOWN_SQL}) AND NOT ({_DUCKDB_MAF_AVAILABLE_SQL}) THEN 1 ELSE 0 END) AS variants_unscorable_absent,
                        SUM(CASE WHEN g.GT IS NOT NULL AND {_DUCKDB_NO_CALL_SQL} THEN 1 ELSE 0 END) AS variants_no_call,
                        SUM(CASE WHEN g.GT IS NULL AND NOT ({_DUCKDB_REFERENCE_KNOWN_SQL}) AND {_DUCKDB_MAF_AVAILABLE_SQL} THEN 1 ELSE 0 END) AS variants_maf_filled,
                        SUM(CASE WHEN g.GT IS NULL AND s.ref_resolved_source = 'panel' THEN 1 ELSE 0 END) AS variants_ref_resolved_panel,
                        SUM(CASE WHEN g.GT IS NULL AND s.ref_resolved_source = 'fasta' THEN 1 ELSE 0 END) AS variants_ref_resolved_fasta
                    """
                else:
                    resolved_dosage_sql = _DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY
                    counter_sql = f"""
                        SUM(CASE WHEN g.GT IS NOT NULL THEN 1 ELSE 0 END) AS variants_observed,
                        SUM(CASE WHEN g.GT IS NOT NULL AND NOT ({_DUCKDB_NO_CALL_SQL}) THEN 1 ELSE 0 END) AS observed_called,
                        SUM(CASE WHEN g.GT IS NULL AND {_DUCKDB_REFERENCE_KNOWN_SQL} THEN 1 ELSE 0 END) AS variants_assumed_hom_ref,
                        SUM(CASE WHEN g.GT IS NULL AND NOT ({_DUCKDB_REFERENCE_KNOWN_SQL}) THEN 1 ELSE 0 END) AS variants_unscorable_absent,
                        SUM(CASE WHEN g.GT IS NOT NULL AND {_DUCKDB_NO_CALL_SQL} THEN 1 ELSE 0 END) AS variants_no_call,
                        0 AS variants_maf_filled,
                        SUM(CASE WHEN g.GT IS NULL AND s.ref_resolved_source = 'panel' THEN 1 ELSE 0 END) AS variants_ref_resolved_panel,
                        SUM(CASE WHEN g.GT IS NULL AND s.ref_resolved_source = 'fasta' THEN 1 ELSE 0 END) AS variants_ref_resolved_fasta
                    """
            else:
                join_sql = f"""
                    FROM {geno_from} g
                    JOIN {_DUCKDB_SCORING_RELATION} s
                      ON g.chrom = s.chr_name_norm AND g.pos = s.chr_pos_norm
                """
                resolved_dosage_sql = _DUCKDB_RESOLVED_DOSAGE_PRESENT_ONLY
                counter_sql = f"""
                    COUNT(*) AS variants_observed,
                    SUM(CASE WHEN NOT ({_DUCKDB_NO_CALL_SQL}) THEN 1 ELSE 0 END) AS observed_called,
                    0 AS variants_assumed_hom_ref,
                    0 AS variants_unscorable_absent,
                    SUM(CASE WHEN {_DUCKDB_NO_CALL_SQL} THEN 1 ELSE 0 END) AS variants_no_call,
                    0 AS variants_maf_filled,
                    0 AS variants_ref_resolved_panel,
                    0 AS variants_ref_resolved_fasta
                """

            score_kwargs = {
                "join_sql": join_sql,
                "resolved_dosage_sql": resolved_dosage_sql,
                "counter_sql": counter_sql,
                "weighted_sql": weighted_sql,
                "variant_mass_sql": variant_mass_sql,
            }
            agg = _DuckDbScoreAgg()
            if variants_total == 0:
                pass
            elif not chunked:
                scoring_df = scoring_norm.collect()
                conn.register("scoring", scoring_df.to_arrow())
                _duckdb_scoring_relation(conn, universe_sql=universe_sql)
                agg = _duckdb_score_registered(conn, **score_kwargs)
                del scoring_df
            else:
                log_message(
                    message_type="prs:duckdb_scoring_chunked",
                    pgs_id=pgs_id,
                    variants_total=variants_total,
                    chunk_size=chunk_size,
                )
                offset = 0
                while offset < variants_total:
                    check_memory_pressure(pgs_id)
                    take = min(chunk_size, variants_total - offset)
                    chunk_df = scoring_norm.slice(offset, take).collect()
                    conn.register("scoring", chunk_df.to_arrow())
                    _duckdb_scoring_relation(conn, universe_sql=universe_sql)
                    agg.add(_duckdb_score_registered(conn, **score_kwargs))
                    conn.unregister("scoring")
                    del chunk_df
                    gc.collect()
                    offset += take
        finally:
            if own_conn:
                conn.close()

        prs_score = agg.prs_score
        observed_called = agg.observed_called
        variants_observed = agg.variants_observed
        variants_assumed_hom_ref = agg.variants_assumed_hom_ref
        variants_unscorable_absent = agg.variants_unscorable_absent
        variants_no_call = agg.variants_no_call
        variants_maf_filled = agg.variants_maf_filled
        variants_ref_resolved_panel = agg.variants_ref_resolved_panel
        variants_ref_resolved_fasta = agg.variants_ref_resolved_fasta
        weight_mass_matched = agg.weight_mass_matched
        variants_matched = observed_called + variants_assumed_hom_ref + variants_maf_filled

        has_freqs = False
        theoretical_mean: float | None = None
        theoretical_std: float | None = None
        percentile: float | None = None
        percentile_method: str | None = None
        z_score: float | None = None
        reference_mean: float | None = None
        reference_std: float | None = None
        has_freqs, theoretical_mean, theoretical_std = _scoring_theoretical_stats(
            scoring_norm, schema_names
        )
        if has_freqs and theoretical_mean is not None and theoretical_std is not None:
            if theoretical_std > 0:
                z = (prs_score - theoretical_mean) / theoretical_std
                percentile = round(_norm_cdf(z) * 100.0, 2)
                percentile_method = "theoretical"
                z_score = z
                reference_mean = theoretical_mean
                reference_std = theoretical_std
            log_message(
                message_type="prs:theoretical_stats",
                pgs_id=pgs_id,
                variants_total=variants_total,
                theoretical_mean=theoretical_mean,
                theoretical_std=theoretical_std,
                percentile=percentile,
            )

        match_rate = variants_matched / variants_total if variants_total > 0 else 0.0
        weight_mass_coverage = (
            weight_mass_matched / weight_mass_total if weight_mass_total > 0 else None
        )

        detected_build, build_mismatch = _detect_build_mismatch(vcf_path, genome_build)

        return PRSResult(
            pgs_id=pgs_id,
            score=prs_score,
            variants_matched=variants_matched,
            variants_total=variants_total,
            match_rate=match_rate,
            variants_observed=variants_observed,
            variants_assumed_hom_ref=variants_assumed_hom_ref,
            variants_unscorable_absent=variants_unscorable_absent,
            variants_no_call=variants_no_call,
            variants_maf_filled=variants_maf_filled,
            variants_ref_resolved_panel=variants_ref_resolved_panel,
            variants_ref_resolved_fasta=variants_ref_resolved_fasta,
            weight_mass_matched=weight_mass_matched,
            weight_mass_total=weight_mass_total,
            weight_mass_coverage=weight_mass_coverage,
            genotype_input_mode=resolved_mode.value,
            detected_genome_build=detected_build,
            build_mismatch=build_mismatch,
            trait_reported=trait_reported,
            has_allele_frequencies=has_freqs,
            theoretical_mean=theoretical_mean,
            theoretical_std=theoretical_std,
            percentile=percentile,
            percentile_method=percentile_method,
            z_score=z_score,
            reference_mean=reference_mean,
            reference_std=reference_std,
        )


_CORRUPT_PARQUET_MARKERS = (
    "out of specification",
    "invalid thrift",
    "metadata size",
    "footer",
    "not a parquet",
)


def _is_corrupt_parquet_error(exc: BaseException) -> bool:
    """Check if an exception looks like a corrupt parquet read."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _CORRUPT_PARQUET_MARKERS)


def _remove_scoring_parquet_cache(
    pgs_id: str, cache_dir: Path, genome_build: str,
) -> bool:
    """Delete a corrupt scoring parquet cache file. Returns True if deleted."""
    from just_prs.scoring import scoring_parquet_path

    parquet = scoring_parquet_path(pgs_id, cache_dir, genome_build)
    if parquet.exists():
        try:
            parquet.unlink()
            return True
        except OSError:
            pass
    return False


def compute_prs_batch(
    vcf_path: Path | str,
    pgs_ids: list[str],
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    genotype_input_mode: str | GenotypeInputMode = GenotypeInputMode.AUTO,
    engine: PRSEngine | str = PRSEngine.DUCKDB,
    genotypes_lf: pl.LazyFrame | None = None,
    memory_limit: str | None = None,
    reference_restoration: RestorationScope = False,
    reference_universe_path: Path | str | None = None,
    reference_universe: ReferenceUniverse | None = None,
    sample_build: str | None = None,
) -> "PRSBatchResult":
    """Compute multiple PRS scores for a single VCF file.

    Memory-safe: uses DuckDB engine by default (spill-to-disk), runs
    ``gc.collect()`` after each score, continues on per-score errors
    instead of crashing, and auto-retries once on corrupt parquet caches.

    When reference restoration is requested and no ``reference_universe`` handle
    is injected, the universe is parsed **once** here (from
    ``reference_universe_path``) and reused for every score, so the ~34M-row
    universe is never re-parsed per score.

    Args:
        vcf_path: Path to VCF file
        pgs_ids: List of PGS Catalog score IDs
        genome_build: Genome build
        cache_dir: Cache directory for downloaded scoring files
        genotype_input_mode: How absent scoring loci are interpreted.
        engine: Computation engine — DUCKDB (default, spill-to-disk)
            or POLARS (in-memory).
        genotypes_lf: Pre-built genotypes LazyFrame. When provided,
            vcf_path is not re-read on each iteration.
        memory_limit: DuckDB per-connection memory limit (e.g. "4GB").
            Only used when engine is DUCKDB. Defaults to env var or
            75% of RAM.

    Returns:
        PRSBatchResult with successful results and per-ID outcome tracking.
    """
    import gc

    from just_prs.catalog import PGSCatalogClient
    from just_prs.models import PRSBatchOutcome, PRSBatchResult

    if isinstance(engine, str):
        engine = PRSEngine(engine)

    # All scores share one sample/scoring build — guard once up front.
    _assert_sample_build_matches(sample_build, genome_build, "batch")

    # Resolve the reference-allele universe once for the whole batch (it is
    # catalog-wide and identical for every score). An injected handle wins;
    # otherwise parse it once from the path so each per-score compute reuses the
    # in-memory table instead of re-parsing 34M rows.
    if reference_universe is None and reference_restoration is not False and reference_universe_path is not None:
        scope = _normalize_restoration_scope(reference_restoration, cache_dir, genome_build)
        if scope is not False:
            sub = scope if isinstance(scope, pl.LazyFrame) else None
            reference_universe = prepare_reference_universe(
                reference_universe_path, genome_build=genome_build, scope=sub
            )

    with start_action(
        action_type="prs:compute_batch",
        vcf_path=str(vcf_path),
        pgs_ids=pgs_ids,
        genome_build=genome_build,
        engine=engine.value,
    ):
        results: list[PRSResult] = []
        outcomes: list[PRSBatchOutcome] = []
        failed_ids: list[str] = []
        genotype_tables: PreparedGenotypeTables | None = None
        table_key: str | None = None
        parquet_path = Path(str(vcf_path))
        try:
            if (
                engine == PRSEngine.DUCKDB
                and genotypes_lf is None
                and parquet_path.suffix == ".parquet"
                and parquet_path.is_file()
            ):
                genotype_tables = prepare_genotype_tables(
                    {"sample": parquet_path},
                    memory_limit=memory_limit,
                    genotype_input_mode=genotype_input_mode,
                )
                table_key = "sample"

            with PGSCatalogClient() as client:
                for pgs_id in pgs_ids:
                    attempts = 1
                    try:
                        score_info = client.get_score(pgs_id)
                        trait = score_info.trait_reported

                        if engine == PRSEngine.DUCKDB:
                            result = compute_prs_duckdb(
                                vcf_path=vcf_path,
                                scoring_file=pgs_id,
                                genome_build=genome_build,
                                cache_dir=cache_dir,
                                pgs_id=pgs_id,
                                trait_reported=trait,
                                genotypes_parquet=str(vcf_path) if (genotypes_lf is None and str(vcf_path).endswith(".parquet")) else None,
                                genotypes_lf=genotypes_lf,
                                genotype_tables=genotype_tables,
                                genotype_table_key=table_key,
                                memory_limit=memory_limit,
                                genotype_input_mode=genotype_input_mode,
                                reference_restoration=reference_restoration,
                                reference_universe_path=reference_universe_path,
                                reference_universe=reference_universe,
                            )
                        else:
                            result = compute_prs(
                                vcf_path=vcf_path,
                                scoring_file=pgs_id,
                                genome_build=genome_build,
                                cache_dir=cache_dir,
                                pgs_id=pgs_id,
                                trait_reported=trait,
                                genotypes_lf=genotypes_lf,
                                genotype_input_mode=genotype_input_mode,
                                reference_restoration=reference_restoration,
                                reference_universe_path=reference_universe_path,
                                reference_universe=reference_universe,
                            )

                        results.append(result)
                        outcomes.append(PRSBatchOutcome(
                            pgs_id=pgs_id, status="ok", attempts=attempts,
                        ))

                    except Exception as exc:
                        if _is_corrupt_parquet_error(exc):
                            removed = _remove_scoring_parquet_cache(
                                pgs_id, cache_dir, genome_build,
                            )
                            if removed:
                                attempts = 2
                                try:
                                    log_message(
                                        message_type="prs:batch_cache_repair",
                                        pgs_id=pgs_id,
                                    )
                                    score_info = client.get_score(pgs_id)
                                    trait = score_info.trait_reported
                                    if engine == PRSEngine.DUCKDB:
                                        result = compute_prs_duckdb(
                                            vcf_path=vcf_path,
                                            scoring_file=pgs_id,
                                            genome_build=genome_build,
                                            cache_dir=cache_dir,
                                            pgs_id=pgs_id,
                                            trait_reported=trait,
                                            genotypes_parquet=str(vcf_path) if (genotypes_lf is None and str(vcf_path).endswith(".parquet")) else None,
                                            genotypes_lf=genotypes_lf,
                                            genotype_tables=genotype_tables,
                                            genotype_table_key=table_key,
                                            memory_limit=memory_limit,
                                            genotype_input_mode=genotype_input_mode,
                                            reference_restoration=reference_restoration,
                                            reference_universe_path=reference_universe_path,
                                            reference_universe=reference_universe,
                                        )
                                    else:
                                        result = compute_prs(
                                            vcf_path=vcf_path,
                                            scoring_file=pgs_id,
                                            genome_build=genome_build,
                                            cache_dir=cache_dir,
                                            pgs_id=pgs_id,
                                            trait_reported=trait,
                                            genotypes_lf=genotypes_lf,
                                            genotype_input_mode=genotype_input_mode,
                                            reference_restoration=reference_restoration,
                                            reference_universe_path=reference_universe_path,
                                            reference_universe=reference_universe,
                                        )
                                    results.append(result)
                                    outcomes.append(PRSBatchOutcome(
                                        pgs_id=pgs_id, status="cache_repaired",
                                        attempts=attempts,
                                    ))
                                    gc.collect()
                                    continue
                                except Exception as retry_exc:
                                    log_message(
                                        message_type="prs:batch_retry_failed",
                                        pgs_id=pgs_id,
                                        error=str(retry_exc),
                                    )
                                    failed_ids.append(pgs_id)
                                    outcomes.append(PRSBatchOutcome(
                                        pgs_id=pgs_id, status="failed",
                                        error=str(retry_exc), attempts=attempts,
                                    ))
                                    gc.collect()
                                    continue

                        log_message(
                            message_type="prs:batch_score_failed",
                            pgs_id=pgs_id,
                            error=str(exc),
                        )
                        failed_ids.append(pgs_id)
                        outcomes.append(PRSBatchOutcome(
                            pgs_id=pgs_id, status="failed",
                            error=str(exc), attempts=attempts,
                        ))

                    gc.collect()

            return PRSBatchResult(
                results=results,
                outcomes=outcomes,
                n_total=len(pgs_ids),
                n_ok=len(results),
                n_failed=len(failed_ids),
                failed_ids=failed_ids,
            )
        finally:
            if genotype_tables is not None:
                genotype_tables.close()
