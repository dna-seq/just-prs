"""Per-variant PRS contributions (DuckDB).

Not on the reference-panel / ``score_and_push`` path — that job keeps using
``just_prs.reference``. This module only *reads* helpers from ``just_prs.prs``.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path

import duckdb
import polars as pl
from eliot import log_message, start_action
from pydantic import BaseModel, ConfigDict, Field

from just_prs.memory import check_memory_pressure, scoring_join_chunk_size
from just_prs.models import PRSResult
from just_prs.prs import (
    DOSAGE_WEIGHT_COLUMNS,
    GenotypeInputMode,
    PreparedGenotypeTables,
    ReferenceUniverse,
    RestorationScope,
    _DUCKDB_NO_CALL_SQL,
    _DUCKDB_REFERENCE_KNOWN_SQL,
    _DUCKDB_RESOLVED_DOSAGE_PRESENT_ONLY,
    _DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY,
    _DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY_MAF,
    _DUCKDB_SCORING_RELATION,
    _DUCKDB_WEIGHTED_DOSAGE_ADDITIVE,
    _DUCKDB_WEIGHTED_DOSAGE_GENOBOOST,
    _assert_sample_build_matches,
    _duckdb_scoring_relation,
    _duckdb_universe_sql,
    _normalize_genotype_columns,
    _normalize_genotype_input_mode,
    _normalize_scoring_columns,
    _resolve_duckdb_memory_limit,
    _resolve_genotype_input_mode,
    _resolve_reference_universe,
    _resolve_scoring,
    _sql_literal_path,
)
from just_prs.scoring import DEFAULT_CACHE_DIR
from just_prs.vcf import read_genotypes


class ContributionCutoffStats(BaseModel):
    """How the pre-join extract filter changed the scoring file.

    These filters apply only to ``extract_prs_contributions``. They never
    change ``compute_prs`` / ``compute_prs_duckdb``.
    """

    cutoff_pct: float | None = Field(
        default=None,
        description="Requested minimum as a percent of mean max-possible contribution",
    )
    top_n: int | None = Field(
        default=None,
        description="Requested keep-N by this sample's |contribution| after the join (None/0 = no cap)",
    )
    mean_max_contribution: float = Field(
        description="Mean of per-variant max |contribution| before the cutoff",
    )
    threshold: float | None = Field(
        default=None,
        description="Absolute keep threshold: (cutoff_pct/100) * mean_max_contribution",
    )
    variants_total: int = Field(description="Scoring rows before the cutoff")
    variants_kept: int = Field(description="Scoring rows after the cutoff (joined)")
    variants_dropped: int = Field(description="Scoring rows removed before the join")
    mass_total: float = Field(description="Sum of max |contribution| before the cutoff")
    mass_kept: float = Field(description="Sum of max |contribution| after the cutoff")
    mass_retained: float = Field(
        description="mass_kept / mass_total (1.0 when nothing was dropped)",
    )


class PRSContributionResult(BaseModel):
    """Per-variant contributions plus the same scalar score ``compute_prs_duckdb`` would return."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pgs_id: str
    score: float = Field(description="Sum of per-variant contribution (kept rows only)")
    cutoff: ContributionCutoffStats
    contributions: pl.LazyFrame = Field(
        description="Lazy scan of the contribution parquet (never fully collected here)",
    )
    output_path: Path = Field(description="Parquet written by DuckDB COPY")


def max_possible_contribution_expr(*, dosage_weight: bool) -> pl.Expr:
    """Per-variant maximum |contribution| if dosage is 0, 1, or 2.

    Additive scores: ``2 * |effect_weight|``. GenoBoost: the largest absolute
    per-dosage weight (the contribution *is* that weight).
    """
    if dosage_weight:
        return pl.max_horizontal(
            pl.col("dosage_0_weight").abs(),
            pl.col("dosage_1_weight").abs(),
            pl.col("dosage_2_weight").abs(),
        )
    return pl.col("effect_weight").abs() * 2.0


def apply_contribution_cutoff(
    scoring: pl.DataFrame,
    min_contribution_pct: float | None,
    *,
    dosage_weight: bool,
) -> tuple[pl.DataFrame, ContributionCutoffStats]:
    """Optional pre-join percent filter (extract only).

    Drops rows whose max |contribution| is below ``n%`` of the mean max.
    ``None`` / ``0`` leaves every scoring row. Sample top-N is applied
    *after* the genotype join, not here.

    Does not affect PRS score computation.
    """
    with_max = scoring.with_columns(
        max_possible_contribution_expr(dosage_weight=dosage_weight).alias("max_contribution")
    )
    variants_total = with_max.height
    mean_max = float(with_max["max_contribution"].mean() or 0.0)
    mass_total = float(with_max["max_contribution"].sum() or 0.0)

    if min_contribution_pct is not None and min_contribution_pct > 100.0:
        raise ValueError(
            f"min_contribution_pct must be in (0, 100]; got {min_contribution_pct}"
        )

    kept = with_max
    threshold: float | None = None
    cutoff_pct: float | None = None
    if min_contribution_pct is not None and min_contribution_pct > 0.0:
        cutoff_pct = float(min_contribution_pct)
        threshold = (min_contribution_pct / 100.0) * mean_max
        kept = kept.filter(pl.col("max_contribution") >= threshold)

    mass_kept = float(kept["max_contribution"].sum() or 0.0)
    variants_kept = kept.height
    return kept, ContributionCutoffStats(
        cutoff_pct=cutoff_pct,
        top_n=None,
        mean_max_contribution=mean_max,
        threshold=threshold,
        variants_total=variants_total,
        variants_kept=variants_kept,
        variants_dropped=variants_total - variants_kept,
        mass_total=mass_total,
        mass_kept=mass_kept,
        mass_retained=(mass_kept / mass_total) if mass_total > 0 else 1.0,
    )


def _normalize_top_n(top_n: int | None) -> int | None:
    """``None`` / ``0`` means keep every joined row. Negative is an error."""
    if top_n is None or int(top_n) == 0:
        return None
    if int(top_n) < 0:
        raise ValueError(f"top_n must be >= 0; got {top_n}")
    return int(top_n)


def _attach_identity_columns(raw: pl.LazyFrame, norm: pl.LazyFrame) -> pl.DataFrame:
    """Keep rsid / other_allele that ``_normalize_scoring_columns`` drops."""
    raw_cols = raw.collect_schema().names()
    extras: list[pl.Expr] = []
    if "hm_rsID" in raw_cols:
        extras.append(pl.col("hm_rsID").cast(pl.Utf8).alias("rsid"))
    elif "rsID" in raw_cols:
        extras.append(pl.col("rsID").cast(pl.Utf8).alias("rsid"))
    else:
        extras.append(pl.lit(None, dtype=pl.Utf8).alias("rsid"))
    extra_df = raw.select(extras).collect()
    norm_df = norm.collect()
    if extra_df.height != norm_df.height:
        raise ValueError(
            "Scoring identity columns drifted from the normalized frame "
            f"({extra_df.height} vs {norm_df.height} rows)"
        )
    out = norm_df.hstack(extra_df)
    if "other_allele" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("other_allele"))
    return out


def _status_sql(*, maf_fill: bool) -> str:
    no_call = _DUCKDB_NO_CALL_SQL.replace("g.GT", "s.GT")
    ref_known = _DUCKDB_REFERENCE_KNOWN_SQL
    if maf_fill:
        maf = (
            "(s.allelefrequency_effect IS NOT NULL AND s.allelefrequency_effect > 0.0 "
            "AND s.allelefrequency_effect < 1.0)"
        )
        return f"""
CASE
    WHEN s.GT IS NOT NULL AND NOT ({no_call}) THEN 'observed'
    WHEN s.GT IS NOT NULL THEN 'no_call'
    WHEN {ref_known} THEN 'restored_hom_ref'
    WHEN {maf} THEN 'maf_filled'
    ELSE 'unscorable'
END"""
    return f"""
CASE
    WHEN s.GT IS NOT NULL AND NOT ({no_call}) THEN 'observed'
    WHEN s.GT IS NOT NULL THEN 'no_call'
    WHEN {ref_known} THEN 'restored_hom_ref'
    ELSE 'unscorable'
END"""


def _contribution_select_sql(
    *,
    resolved_dosage_sql: str,
    weighted_sql: str,
    join_sql: str,
    status_sql: str,
    dosage_weight: bool,
) -> str:
    weight_cols = (
        "s.dosage_0_weight, s.dosage_1_weight, s.dosage_2_weight, "
        "CAST(NULL AS DOUBLE) AS effect_weight"
        if dosage_weight
        else "s.effect_weight"
    )
    return f"""
        WITH resolved AS (
            SELECT
                s.*,
                g.GT,
                g.ref AS sample_ref,
                g.alt AS sample_alt,
                {resolved_dosage_sql} AS resolved_dosage
            {join_sql}
        )
        SELECT
            s.chr_name_norm AS chrom,
            s.chr_pos_norm AS pos,
            s.rsid,
            s.effect_allele,
            s.other_allele,
            s.reference_allele,
            {weight_cols},
            s.sample_ref,
            s.sample_alt,
            s.GT,
            s.resolved_dosage AS dosage,
            ({weighted_sql}) AS contribution,
            s.max_contribution,
            {status_sql} AS status
        FROM resolved s
    """


def _copy_sql_to_parquet(conn: duckdb.DuckDBPyConnection, sql: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest_sql = _sql_literal_path(dest)
    conn.execute(f"COPY ({sql}) TO '{dest_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def _empty_contributions_frame(*, dosage_weight: bool) -> pl.DataFrame:
    weight_cols: dict[str, pl.DataType] = (
        {
            "dosage_0_weight": pl.Float64,
            "dosage_1_weight": pl.Float64,
            "dosage_2_weight": pl.Float64,
            "effect_weight": pl.Float64,
        }
        if dosage_weight
        else {"effect_weight": pl.Float64}
    )
    schema: dict[str, pl.DataType] = {
        "chrom": pl.Utf8,
        "pos": pl.Int64,
        "rsid": pl.Utf8,
        "effect_allele": pl.Utf8,
        "other_allele": pl.Utf8,
        "reference_allele": pl.Utf8,
        **weight_cols,
        "sample_ref": pl.Utf8,
        "sample_alt": pl.Utf8,
        "GT": pl.Utf8,
        "dosage": pl.Float64,
        "contribution": pl.Float64,
        "max_contribution": pl.Float64,
        "status": pl.Utf8,
        "contribution_share": pl.Float64,
    }
    return pl.DataFrame(schema=schema)


def _stamp_contribution_share(
    conn: duckdb.DuckDBPyConnection,
    raw_path: Path,
    dest: Path,
    *,
    top_n: int | None = None,
) -> tuple[float, int, float]:
    """Sort by this sample's |contribution|, optionally keep top-N, stamp share.

    Share is computed on the kept rows so it sums to 1. Returns
    ``(score, n_kept, mass_kept)`` where mass is sum of ``max_contribution``.
    """
    raw_sql = _sql_literal_path(raw_path)
    dest_sql = _sql_literal_path(dest)
    limit_sql = f"LIMIT {int(top_n)}" if top_n is not None and top_n > 0 else ""
    conn.execute(
        f"""
        COPY (
            SELECT
                ranked.*,
                CASE
                    WHEN SUM(ABS(contribution)) OVER () > 0
                    THEN ABS(contribution) / SUM(ABS(contribution)) OVER ()
                    ELSE 0.0
                END AS contribution_share
            FROM (
                SELECT *
                FROM read_parquet('{raw_sql}')
                ORDER BY
                    ABS(contribution) DESC NULLS LAST,
                    ABS(max_contribution) DESC NULLS LAST,
                    chrom,
                    pos
                {limit_sql}
            ) ranked
        ) TO '{dest_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    stats_row = conn.execute(
        f"""
        SELECT
            COALESCE(SUM(contribution), 0.0),
            COUNT(*),
            COALESCE(SUM(max_contribution), 0.0)
        FROM read_parquet('{dest_sql}')
        """
    ).fetchone()
    if stats_row is None:
        return 0.0, 0, 0.0
    return float(stats_row[0]), int(stats_row[1]), float(stats_row[2])


def _write_joined_contributions(
    conn: duckdb.DuckDBPyConnection,
    scoring_df: pl.DataFrame,
    *,
    select_sql: str,
    universe_sql: str | None,
    pgs_id: str,
    raw_path: Path,
) -> None:
    """Join scoring rows in the same chunk size as ``compute_prs_duckdb``."""
    variants_total = scoring_df.height
    chunk_size = scoring_join_chunk_size(variants_total)
    chunked = variants_total > chunk_size > 0
    if not chunked:
        check_memory_pressure(pgs_id)
        conn.register("scoring", scoring_df.to_arrow())
        _duckdb_scoring_relation(conn, universe_sql=universe_sql)
        _copy_sql_to_parquet(conn, select_sql, raw_path)
        conn.unregister("scoring")
        return

    parts_dir = raw_path.with_name(f"{raw_path.stem}_parts")
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    log_message(
        message_type="prs:extract_contributions_chunked",
        pgs_id=pgs_id,
        variants_total=variants_total,
        chunk_size=chunk_size,
    )
    offset = 0
    part_i = 0
    while offset < variants_total:
        check_memory_pressure(pgs_id)
        take = min(chunk_size, variants_total - offset)
        chunk_df = scoring_df.slice(offset, take)
        conn.register("scoring", chunk_df.to_arrow())
        _duckdb_scoring_relation(conn, universe_sql=universe_sql)
        _copy_sql_to_parquet(conn, select_sql, parts_dir / f"part_{part_i:05d}.parquet")
        conn.unregister("scoring")
        del chunk_df
        gc.collect()
        offset += take
        part_i += 1
    glob_sql = _sql_literal_path(parts_dir / "part_*.parquet")
    dest_sql = _sql_literal_path(raw_path)
    conn.execute(
        f"COPY (SELECT * FROM read_parquet('{glob_sql}')) "
        f"TO '{dest_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    shutil.rmtree(parts_dir)


def extract_prs_contributions(
    vcf_path: Path | str,
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str = "GRCh38",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    pgs_id: str = "unknown",
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
    min_contribution_pct: float | None = None,
    top_n: int | None = None,
    output: Path | str | None = None,
    sample_build: str | None = None,
) -> PRSContributionResult:
    """Join a scoring file to a genome and keep one row per scoring variant.

    Same DuckDB dosage / join rules as ``compute_prs_duckdb``. The scalar
    ``score`` is ``sum(contribution)`` over the *kept* rows, so with no
    extract filter it matches ``compute_prs_duckdb(...).score``.

    ``min_contribution_pct`` can still drop tiny-weight scoring rows
    before the join. ``top_n`` is this sample's N largest
    ``|contribution|`` *after* the genotype join. Neither changes
    ``compute_prs`` / ``compute_prs_duckdb``, which only return a
    scalar score plus match counts.
    """
    with start_action(
        action_type="prs:extract_contributions",
        vcf_path=str(vcf_path),
        pgs_id=pgs_id,
        genome_build=genome_build,
        min_contribution_pct=min_contribution_pct,
        top_n=top_n,
    ):
        _assert_sample_build_matches(sample_build, genome_build, pgs_id)
        scoring_raw = _resolve_scoring(scoring_file, genome_build, cache_dir)
        scoring_norm = _normalize_scoring_columns(scoring_raw)
        schema_names = scoring_norm.collect_schema().names()
        dosage_weight = DOSAGE_WEIGHT_COLUMNS[0] in schema_names

        scoring_df = _attach_identity_columns(scoring_raw, scoring_norm)
        requested_top_n = _normalize_top_n(top_n)
        scoring_df, cutoff = apply_contribution_cutoff(
            scoring_df,
            min_contribution_pct,
            dosage_weight=dosage_weight,
        )

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

        ref_universe = _resolve_reference_universe(
            reference_universe=reference_universe,
            reference_universe_path=reference_universe_path,
            reference_restoration=reference_restoration,
            resolved_mode=resolved_mode,
            genome_build=genome_build,
            cache_dir=cache_dir,
        )

        if output is None:
            output_path = Path(tempfile.mkdtemp(prefix="just-prs-contributions-")) / "contributions.parquet"
        else:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        weighted_sql = (
            _DUCKDB_WEIGHTED_DOSAGE_GENOBOOST if dosage_weight else _DUCKDB_WEIGHTED_DOSAGE_ADDITIVE
        )
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
            has_maf_col = "allelefrequency_effect" in scoring_df.columns
            do_maf_fill = maf_fill and has_maf_col and not dosage_weight
            universe_sql = (
                _duckdb_universe_sql(conn, ref_universe) if ref_universe is not None else None
            )

            if resolved_mode == GenotypeInputMode.VARIANT_ONLY:
                join_sql = f"""
                    FROM {_DUCKDB_SCORING_RELATION} s
                    LEFT JOIN {geno_from} g
                      ON g.chrom = s.chr_name_norm AND g.pos = s.chr_pos_norm
                """
                resolved_dosage_sql = (
                    _DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY_MAF
                    if do_maf_fill
                    else _DUCKDB_RESOLVED_DOSAGE_VARIANT_ONLY
                )
            else:
                join_sql = f"""
                    FROM {geno_from} g
                    JOIN {_DUCKDB_SCORING_RELATION} s
                      ON g.chrom = s.chr_name_norm AND g.pos = s.chr_pos_norm
                """
                resolved_dosage_sql = _DUCKDB_RESOLVED_DOSAGE_PRESENT_ONLY

            select_sql = _contribution_select_sql(
                resolved_dosage_sql=resolved_dosage_sql,
                weighted_sql=weighted_sql,
                join_sql=join_sql,
                status_sql=_status_sql(maf_fill=do_maf_fill),
                dosage_weight=dosage_weight,
            )

            if scoring_df.height == 0:
                _empty_contributions_frame(dosage_weight=dosage_weight).write_parquet(output_path)
                score = 0.0
                n_kept = 0
                mass_kept = 0.0
            else:
                raw_path = output_path.with_name(f"{output_path.stem}_raw.parquet")
                _write_joined_contributions(
                    conn,
                    scoring_df,
                    select_sql=select_sql,
                    universe_sql=universe_sql,
                    pgs_id=pgs_id,
                    raw_path=raw_path,
                )
                score, n_kept, mass_kept = _stamp_contribution_share(
                    conn, raw_path, output_path, top_n=requested_top_n
                )
                raw_path.unlink(missing_ok=True)
            cutoff = cutoff.model_copy(
                update={
                    "top_n": requested_top_n,
                    "variants_kept": n_kept,
                    "variants_dropped": cutoff.variants_total - n_kept,
                    "mass_kept": mass_kept,
                    "mass_retained": (
                        mass_kept / cutoff.mass_total if cutoff.mass_total > 0 else 1.0
                    ),
                }
            )
        finally:
            if own_conn:
                conn.close()

        log_message(
            message_type="prs:extract_contributions_done",
            pgs_id=pgs_id,
            score=score,
            variants_kept=cutoff.variants_kept,
            variants_dropped=cutoff.variants_dropped,
            mass_retained=cutoff.mass_retained,
            output_path=str(output_path),
        )
        return PRSContributionResult(
            pgs_id=pgs_id,
            score=score,
            cutoff=cutoff,
            contributions=pl.scan_parquet(output_path),
            output_path=output_path,
        )


def contributions_match_compute(result: PRSContributionResult, computed: PRSResult) -> bool:
    """True when extract and compute agree on the scalar score (no cutoff)."""
    if result.cutoff.variants_dropped != 0:
        return False
    return abs(result.score - computed.score) <= 1e-10
