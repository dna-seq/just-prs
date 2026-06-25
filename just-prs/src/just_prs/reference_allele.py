"""Reference-allele resolution for absent PGS scoring variants.

A plain WGS VCF only records rows where the sample differs from reference, so a
scoring variant *absent* from the callset is hom-ref there. ``compute_prs`` can
already impute hom-ref for an absent variant **when the scoring file carries a
``reference_allele``** — but genome-wide scoring files frequently omit it, which
is what strands ~half of every genome-wide score in ``variants_unscorable_absent``.

This module resolves the missing reference base at a set of genomic positions
from two independent sources, in priority order:

  1. **Panel tier** — the in-repo reference-panel ``.pvar`` (full, authoritative
     ``REF`` including indels). Free; no new download.
  2. **FASTA tier** — a single-base faidx lookup against the Ensembl primary
     assembly, for the long tail of positions not in the panel. Single-base
     (SNV) only: an absent variant gives no REF length, so multi-base / indel
     positions are left ``unresolved`` rather than mis-represented by one base.

It is consumed by the ``reference_allele_universe`` precompute (which builds a
small ``(genome_build, chrom, pos, ref, ref_source)`` parquet over the catalog's
scoring positions and uploads it to HuggingFace). The runtime engine never calls
this module — it reads that precomputed parquet.

Designed to be self-contained / extractable into a future ``just-vcf-ops`` lib:
it reuses ``parse_pvar`` / ``_pvar_parquet_cache_path`` / ``_resolve_duckdb_memory_limit``
from ``reference.py`` but holds no PRS-specific state.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from eliot import log_message, start_action

from just_prs.reference import (
    REFERENCE_FASTA_CHR1_LENGTH,
    ReferencePanelError,
    _pvar_parquet_cache_path,
    _require_duckdb,
    _require_pysam,
    _resolve_duckdb_memory_limit,
    parse_pvar,
)

RefAlleleSource = str  # one of: "panel", "fasta", "unresolved"

# Ensembl uses "MT" for the mitochondrion; normalized genotype chrom may be "M".
_MITO_ALIASES = ("MT", "M")
_RESULT_COLUMNS = ("genome_build", "chrom", "pos", "ref", "ref_source")


def _empty_result(genome_build: str) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "genome_build": pl.Utf8,
            "chrom": pl.Utf8,
            "pos": pl.Int64,
            "ref": pl.Utf8,
            "ref_source": pl.Utf8,
        }
    )


def _normalize_positions(positions_df: pl.DataFrame) -> pl.DataFrame:
    """Return unique (chrom, pos[, snv_only]) with normalized dtypes.

    ``snv_only`` (bool) gates the FASTA tier: when False, a single FASTA base
    cannot stand in for the variant's REF, so the position is left for the panel
    tier or unresolved. Defaults to True when the column is absent.
    """
    if "chrom" not in positions_df.columns or "pos" not in positions_df.columns:
        raise ValueError(
            f"positions_df must have 'chrom' and 'pos' columns; got {positions_df.columns}"
        )
    snv_expr = (
        pl.col("snv_only").cast(pl.Boolean)
        if "snv_only" in positions_df.columns
        else pl.lit(True).alias("snv_only")
    )
    return (
        positions_df.select(
            pl.col("chrom").cast(pl.Utf8).str.replace("(?i)^chr", "").alias("chrom"),
            pl.col("pos").cast(pl.Int64).alias("pos"),
            snv_expr.alias("snv_only"),
        )
        .filter(pl.col("pos").is_not_null() & (pl.col("pos") > 0))
        # If a position is SNV in one record but indel in another, treat it as
        # non-SNV (conservative — keeps a single FASTA base from mis-orienting it).
        .group_by("chrom", "pos")
        .agg(pl.col("snv_only").min().alias("snv_only"))
    )


def _resolve_panel(positions: pl.DataFrame, pvar_parquet_path: Path, memory_limit: str) -> pl.DataFrame:
    """Resolve REF from the panel pvar parquet via DuckDB (memory-efficient scan).

    Returns (chrom, pos, ref) for positions whose panel REF is unambiguous.
    """
    _require_duckdb()
    import duckdb

    con = duckdb.connect(config={"memory_limit": memory_limit})
    con.execute("SET arrow_large_buffer_size = true")
    con.execute("SET preserve_insertion_order = false")
    con.register("positions", positions.select("chrom", "pos").to_arrow())
    query = f"""
        SELECT s.chrom AS chrom, s.pos AS pos, min(p."REF") AS ref
        FROM positions s
        INNER JOIN '{pvar_parquet_path}' p
            ON p.chrom = s.chrom AND p."POS" = s.pos
        GROUP BY s.chrom, s.pos
        HAVING count(DISTINCT p."REF") = 1
    """
    try:
        out = con.sql(query).pl()
    finally:
        con.close()
    return out.with_columns(pl.col("ref").cast(pl.Utf8))


def _verify_fasta_build(fasta_path: Path, genome_build: str) -> None:
    """Fail loudly unless the FASTA's chr1 length matches the expected build.

    Contig *names* are identical across GRCh37/38, so length is the cheap,
    unambiguous build fingerprint. A silent build/contig mismatch would inject
    wrong REF bases — so a mismatch raises rather than guessing.
    """
    expected = REFERENCE_FASTA_CHR1_LENGTH.get(genome_build)
    if expected is None:
        raise ReferencePanelError(
            f"No chr1-length fingerprint registered for build {genome_build!r}; "
            f"cannot verify FASTA build before trusting REF bases."
        )
    _require_pysam()
    import pysam

    fa = pysam.FastaFile(str(fasta_path))
    try:
        refs = set(fa.references)
        if "1" not in refs:
            raise ReferencePanelError(
                f"FASTA {fasta_path} has no contig '1' (contigs look prefixed or "
                f"non-Ensembl: {sorted(refs)[:5]}...); refusing to trust REF bases."
            )
        actual = fa.get_reference_length("1")
    finally:
        fa.close()
    if actual != expected:
        raise ReferencePanelError(
            f"FASTA chr1 length {actual} != expected {expected} for build "
            f"{genome_build!r}; build/contig mismatch — refusing to inject REF bases."
        )


def _resolve_fasta(positions: pl.DataFrame, fasta_path: Path, genome_build: str) -> pl.DataFrame:
    """Resolve single-base REF from the FASTA for SNV-only positions.

    Verifies the build first. Returns (chrom, pos, ref) for positions whose base
    is a clean A/C/G/T; everything else is omitted (caller marks unresolved).
    """
    snv = positions.filter(pl.col("snv_only"))
    if snv.height == 0:
        return positions.head(0).select("chrom", "pos").with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("ref")
        )

    _verify_fasta_build(fasta_path, genome_build)
    _require_pysam()
    import pysam

    fa = pysam.FastaFile(str(fasta_path))
    valid = {"A", "C", "G", "T"}
    chroms: list[str] = []
    poss: list[int] = []
    refs: list[str] = []
    try:
        available = set(fa.references)
        mito_contig = next((m for m in _MITO_ALIASES if m in available), None)
        for chrom, pos in snv.select("chrom", "pos").iter_rows():
            contig = chrom
            if chrom in _MITO_ALIASES:
                if mito_contig is None:
                    continue
                contig = mito_contig
            if contig not in available:
                continue
            base = fa.fetch(contig, pos - 1, pos).upper()
            if len(base) == 1 and base in valid:
                chroms.append(chrom)
                poss.append(pos)
                refs.append(base)
    finally:
        fa.close()

    return pl.DataFrame(
        {"chrom": chroms, "pos": poss, "ref": refs},
        schema={"chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8},
    )


def resolve_reference_alleles(
    positions_df: pl.DataFrame,
    genome_build: str = "GRCh38",
    *,
    panel_pvar_path: Path | None = None,
    fasta_path: Path | None = None,
    duckdb_memory_limit: str | None = None,
) -> pl.DataFrame:
    """Resolve the reference base at a set of genomic positions.

    Args:
        positions_df: DataFrame with ``chrom`` (str) and ``pos`` (i64), and an
            optional ``snv_only`` (bool) gate for the FASTA tier. ``chrom`` is
            normalized (``chr`` prefix stripped) internally.
        genome_build: Build label; keys the FASTA build fingerprint.
        panel_pvar_path: Reference-panel ``.pvar`` — either a parsed ``.parquet``
            (used directly) or a ``.pvar.zst`` (parsed via ``parse_pvar`` first).
            When None, the panel tier is skipped.
        fasta_path: Uncompressed Ensembl ``.fa`` (with sibling ``.fai``). When
            None, the FASTA tier is skipped.
        duckdb_memory_limit: Override for the DuckDB scan memory cap.

    Returns:
        DataFrame ``(genome_build, chrom, pos, ref, ref_source)`` with one row per
        unique input position. ``ref_source`` ∈ {``panel``, ``fasta``,
        ``unresolved``}; ``ref`` is null when unresolved.
    """
    with start_action(
        action_type="reference_allele:resolve",
        genome_build=genome_build,
        n_positions=positions_df.height,
        has_panel=panel_pvar_path is not None,
        has_fasta=fasta_path is not None,
    ):
        positions = _normalize_positions(positions_df)
        if positions.height == 0:
            return _empty_result(genome_build)

        memory_limit = duckdb_memory_limit or _resolve_duckdb_memory_limit()

        # --- Panel tier ---------------------------------------------------
        panel_res = positions.head(0).select("chrom", "pos").with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("ref")
        )
        if panel_pvar_path is not None:
            pvar_parquet = Path(panel_pvar_path)
            if pvar_parquet.suffix != ".parquet":
                # A .pvar.zst — ensure the parquet cache exists, then point at it.
                parse_pvar(pvar_parquet)
                pvar_parquet = _pvar_parquet_cache_path(pvar_parquet)
            panel_res = _resolve_panel(positions, pvar_parquet, memory_limit)

        resolved_panel = panel_res.with_columns(pl.lit("panel").alias("ref_source"))

        # --- FASTA tier (positions not resolved by the panel) -------------
        remaining = positions.join(
            panel_res.select("chrom", "pos"), on=["chrom", "pos"], how="anti"
        )
        if fasta_path is not None and remaining.height > 0:
            fasta_res = _resolve_fasta(remaining, Path(fasta_path), genome_build)
            resolved_fasta = fasta_res.with_columns(pl.lit("fasta").alias("ref_source"))
        else:
            resolved_fasta = pl.DataFrame(
                schema={"chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8, "ref_source": pl.Utf8}
            )

        resolved = pl.concat(
            [resolved_panel.select("chrom", "pos", "ref", "ref_source"), resolved_fasta],
            how="vertical",
        )

        # --- Unresolved remainder ----------------------------------------
        out = (
            positions.select("chrom", "pos")
            .join(resolved, on=["chrom", "pos"], how="left")
            .with_columns(
                pl.col("ref_source").fill_null("unresolved"),
                pl.lit(genome_build).alias("genome_build"),
            )
            .select("genome_build", "chrom", "pos", "ref", "ref_source")
            .sort("chrom", "pos")
        )

        counts = out.group_by("ref_source").len().to_dict(as_series=False)
        log_message(
            message_type="reference_allele:resolved",
            genome_build=genome_build,
            n_total=out.height,
            counts=dict(zip(counts["ref_source"], counts["len"])),
        )
        return out
