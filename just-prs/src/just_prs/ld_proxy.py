"""LD-proxy substitution for consumer genotyping arrays.

When a PRS variant is not directly typed on a consumer array, an LD-proxy is a
nearby variant that IS on the array and is in high linkage disequilibrium (LD)
with the target. The proxy's observed dosage replaces the missing one, with the
PRS weight adjusted by the signed correlation: ``w_proxy = w_original * r_signed``.

This is the best-validated imputation-free method for improving array PRS accuracy
(Dite et al. 2018, PLOS ONE). It works well through ~50% missingness and requires
only a precomputed LD lookup table — no external binaries or genotype-level
imputation.

The LD table is precomputed from a reference panel (1000G) by the Dagster pipeline
and cached on HuggingFace. Consumer-side code only does a parquet join.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import polars as pl
from eliot import log_message, start_action

LD_PROXY_TABLE_COLUMNS = [
    "target_chr",
    "target_pos",
    "target_ref",
    "target_alt",
    "proxy_chr",
    "proxy_pos",
    "proxy_rsid",
    "proxy_ref",
    "proxy_alt",
    "r_squared",
    "r_signed",
]

LD_PROXY_TABLE_SCHEMA = {
    "target_chr": pl.Utf8,
    "target_pos": pl.Int64,
    "target_ref": pl.Utf8,
    "target_alt": pl.Utf8,
    "proxy_chr": pl.Utf8,
    "proxy_pos": pl.Int64,
    "proxy_rsid": pl.Utf8,
    "proxy_ref": pl.Utf8,
    "proxy_alt": pl.Utf8,
    "r_squared": pl.Float64,
    "r_signed": pl.Float64,
}

DEFAULT_MIN_R2 = 0.8
DEFAULT_PROXY_WINDOW_KB = 500
DEFAULT_LD_MAX_WORKERS = 1
DEFAULT_LD_CHUNK_SIZE_BP = 250_000
DEFAULT_LD_MAX_TARGETS_PER_CHUNK = 256
DEFAULT_LD_TARGET_BATCH_SIZE = 10_000
DEFAULT_LD_PROXY_FLUSH_ROWS = 50_000
DEFAULT_LD_MEMORY_LIMIT_PERCENT = 65
DEFAULT_LD_DUCKDB_MEMORY_GB = 4.0
DEFAULT_LD_REQUIRE_EXPLICIT_SCOPE = True
_LD_MEMORY_GB = 1024 ** 3


@dataclass(frozen=True)
class LDProxyBuildResult:
    """Summary for an LD-proxy table written directly to parquet."""

    path: Path
    n_untyped: int
    n_proxied: int
    mean_r2: float | None


@dataclass(frozen=True)
class LDProxyOutcome:
    """Per-PGS-ID outcome from LD-proxy batch generation."""

    pgs_id: str
    status: str
    path: Path | None = None
    n_untyped: int | None = None
    n_proxied: int | None = None
    mean_r2: float | None = None
    elapsed_sec: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class LDProxyBatchResult:
    """Summary for a resumable LD-proxy batch over PGS IDs."""

    panel: str
    chip: str
    build: str
    outcomes: list[LDProxyOutcome]
    quality_df: pl.DataFrame
    output_dir: Path

    @property
    def n_total(self) -> int:
        return len(self.outcomes)

    @property
    def n_ok(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "ok")

    @property
    def n_cached(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "cached")

    @property
    def n_failed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "failed")

    @property
    def coverage_ratio(self) -> float:
        return (self.n_ok + self.n_cached) / self.n_total if self.n_total > 0 else 0.0

    @property
    def paths(self) -> dict[str, str]:
        return {
            outcome.pgs_id: str(outcome.path)
            for outcome in self.outcomes
            if outcome.path is not None and outcome.status in {"ok", "cached"}
        }


LD_PROXY_QUALITY_SCHEMA = {
    "pgs_id": pl.Utf8,
    "status": pl.Utf8,
    "path": pl.Utf8,
    "n_untyped": pl.Int64,
    "n_proxied": pl.Int64,
    "mean_r2": pl.Float64,
    "elapsed_sec": pl.Float64,
    "error": pl.Utf8,
}


@dataclass(frozen=True)
class _ResolvedLDProxyInputs:
    """Reference/chip inputs shared across per-PGS LD-proxy builds."""

    typed_df: pl.DataFrame
    pvar_zst_path: Path
    pvar_parquet_path: Path
    pvar_variant_ct: int
    pgen_path: Path
    n_samples: int
    allele_offsets: Any


def ld_proxy_table_path(cache_dir: Path, chip: str, build: str, panel: str = "1000g") -> Path:
    """Return the legacy local cache path for a combined LD-proxy table parquet."""
    return cache_dir / "percentiles" / f"{panel}_ld_proxy_{chip}_{build}.parquet"


def ld_proxy_dir(cache_dir: Path, chip: str, build: str, panel: str = "1000g") -> Path:
    """Return the canonical per-PGS LD-proxy cache directory."""
    return cache_dir / "percentiles" / "ld_proxy" / panel / chip / build


def ld_proxy_pgs_path(
    cache_dir: Path,
    pgs_id: str,
    chip: str,
    build: str,
    panel: str = "1000g",
) -> Path:
    """Return the canonical per-PGS LD-proxy parquet path."""
    return ld_proxy_dir(cache_dir, chip, build, panel) / f"{pgs_id.strip().upper()}.parquet"


def ld_proxy_quality_path(
    cache_dir: Path,
    chip: str,
    build: str,
    panel: str = "1000g",
) -> Path:
    """Return the per-PGS LD-proxy quality/outcome parquet path."""
    return ld_proxy_dir(cache_dir, chip, build, panel) / "_quality.parquet"


def validate_ld_proxy_table(df: pl.DataFrame) -> None:
    """Validate that a DataFrame has the required LD proxy table columns."""
    missing = set(LD_PROXY_TABLE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"LD proxy table missing columns: {missing}")
    if df.height == 0:
        return
    n_bad_r2 = df.filter(
        (pl.col("r_squared") < 0.0) | (pl.col("r_squared") > 1.0)
    ).height
    if n_bad_r2 > 0:
        raise ValueError(f"LD proxy table has {n_bad_r2} rows with r_squared outside [0, 1]")
    n_bad_signed = df.filter(
        (pl.col("r_signed") < -1.0) | (pl.col("r_signed") > 1.0)
    ).height
    if n_bad_signed > 0:
        raise ValueError(f"LD proxy table has {n_bad_signed} rows with r_signed outside [-1, 1]")


def apply_ld_proxies(
    scoring_lf: pl.LazyFrame,
    genotypes_lf: pl.LazyFrame,
    ld_table: pl.LazyFrame | pl.DataFrame | Path,
    min_r2: float = DEFAULT_MIN_R2,
) -> tuple[pl.LazyFrame, int, float | None]:
    """Apply LD-proxy substitution to a scoring LazyFrame.

    For each scoring variant that is absent from the genotypes, look up a proxy
    in the LD table. If found (r² >= min_r2), substitute the proxy's dosage and
    adjust the effect weight: ``w_proxy = w_original * r_signed``.

    Args:
        scoring_lf: Normalized scoring LazyFrame (must have ``chr_name_norm``,
            ``chr_pos_norm``, ``effect_allele``, ``effect_weight``).
        genotypes_lf: Normalized genotypes LazyFrame (must have ``chrom``,
            ``pos``, ``ref``, ``alt``, ``GT``).
        ld_table: LD proxy lookup table (LazyFrame, DataFrame, or parquet Path).
        min_r2: Minimum r² threshold for proxy acceptance.

    Returns:
        Tuple of (enhanced_scoring_lf, n_proxied, mean_r2):
        - enhanced_scoring_lf has proxy-adjusted weights and proxy position info
          for variants that were substituted
        - n_proxied: number of variants recovered via proxy
        - mean_r2: mean r² of the used proxies (None if n_proxied == 0)
    """
    with start_action(action_type="ld_proxy:apply", min_r2=min_r2):
        if isinstance(ld_table, Path):
            ld_lf = pl.scan_parquet(ld_table)
        elif isinstance(ld_table, pl.DataFrame):
            ld_lf = ld_table.lazy()
        else:
            ld_lf = ld_table

        ld_lf = ld_lf.filter(pl.col("r_squared") >= min_r2)

        geno_pos_df = genotypes_lf.select("chrom", "pos").unique().collect()

        typed_set_lf = geno_pos_df.rename({"chrom": "chr_name_norm", "pos": "chr_pos_norm"}).lazy()

        scoring_cols = scoring_lf.collect_schema().names()

        typed = scoring_lf.join(
            typed_set_lf,
            on=["chr_name_norm", "chr_pos_norm"],
            how="semi",
        )

        untyped = scoring_lf.join(
            typed_set_lf,
            on=["chr_name_norm", "chr_pos_norm"],
            how="anti",
        )

        proxied = untyped.join(
            ld_lf,
            left_on=["chr_name_norm", "chr_pos_norm"],
            right_on=["target_chr", "target_pos"],
            how="inner",
        )

        proxy_geno_lf = geno_pos_df.rename({"chrom": "proxy_chr", "pos": "proxy_pos"}).lazy()
        proxied_with_geno = proxied.join(
            proxy_geno_lf,
            on=["proxy_chr", "proxy_pos"],
            how="semi",
        )

        stats = proxied_with_geno.select(
            pl.len().alias("n_proxied"),
            pl.col("r_squared").mean().alias("mean_r2"),
        ).collect()

        n_proxied = int(stats["n_proxied"][0] or 0)
        mean_r2 = float(stats["mean_r2"][0]) if n_proxied > 0 and stats["mean_r2"][0] is not None else None

        has_effect_weight = "effect_weight" in scoring_cols

        if has_effect_weight:
            proxied_adjusted = proxied_with_geno.with_columns(
                (pl.col("effect_weight") * pl.col("r_signed")).alias("effect_weight"),
                pl.col("proxy_pos").alias("chr_pos_norm"),
                pl.col("proxy_chr").alias("chr_name_norm"),
            ).select(scoring_cols)
        else:
            proxied_adjusted = proxied_with_geno.with_columns(
                pl.col("proxy_pos").alias("chr_pos_norm"),
                pl.col("proxy_chr").alias("chr_name_norm"),
            ).select(scoring_cols)

        ld_targets = ld_lf.select(
            pl.col("target_chr").alias("chr_name_norm"),
            pl.col("target_pos").alias("chr_pos_norm"),
        ).unique()
        still_untyped = untyped.join(
            ld_targets,
            on=["chr_name_norm", "chr_pos_norm"],
            how="anti",
        )

        enhanced = pl.concat([typed, proxied_adjusted, still_untyped], how="diagonal_relaxed")

        log_message(
            message_type="ld_proxy:applied",
            n_proxied=n_proxied,
            mean_r2=mean_r2,
        )

        return enhanced, n_proxied, mean_r2


def _pearson_r_one_vs_many(
    target: "Any", candidates: "Any",
) -> "Any":
    """Vectorized Pearson r between one target row and a matrix of candidate rows.

    Uses a fast matrix-multiplication path when no candidate has NaN at positions
    where the target is valid (typical for 1000G with <1% missingness). Falls back
    to per-candidate correlation for the rare rows with missing data.

    Args:
        target: 1D float64 numpy array, shape (n_samples,). NaN = missing.
        candidates: 2D float64 numpy array, shape (k, n_samples). NaN = missing.

    Returns:
        1D float64 numpy array, shape (k,), Pearson r per candidate.
    """
    import numpy as np

    valid_t = ~np.isnan(target)
    n_valid = int(valid_t.sum())
    k = candidates.shape[0]

    if n_valid < 50:
        return np.zeros(k)

    results = np.zeros(k)
    t_at_valid = target[valid_t]
    C_at_valid = candidates[:, valid_t]

    any_nan = np.any(np.isnan(C_at_valid), axis=1)
    clean = ~any_nan
    n_clean = int(clean.sum())

    if n_clean > 0:
        t_mean = t_at_valid.mean()
        t_c = t_at_valid - t_mean
        t_var = float((t_c * t_c).mean())
        t_std = t_var ** 0.5

        if t_std > 0:
            C_clean = C_at_valid[clean]
            C_means = C_clean.mean(axis=1, keepdims=True)
            C_c = C_clean - C_means
            C_stds = np.sqrt((C_c * C_c).mean(axis=1))

            numer = (C_c @ t_c) / n_valid
            good = C_stds > 0
            clean_r = np.zeros(n_clean)
            clean_r[good] = numer[good] / (t_std * C_stds[good])
            results[clean] = clean_r

    for di in np.where(any_nan)[0]:
        c = candidates[di]
        valid = valid_t & ~np.isnan(c)
        nv = int(valid.sum())
        if nv < 50:
            continue
        tv = target[valid]
        cv = c[valid]
        ts = float(tv.std())
        cs = float(cv.std())
        if ts == 0 or cs == 0:
            continue
        results[di] = float(np.corrcoef(tv, cv)[0, 1])

    return results


def _pearson_r_many_vs_many(
    targets: "Any", candidates: "Any",
) -> "Any":
    """Correlation matrix between multiple targets and multiple candidates.

    Computes Pearson r for every (target, candidate) pair in one matrix multiply.
    Much faster than calling ``_pearson_r_one_vs_many`` per target when targets
    share the same candidate set (the typical case within a genomic chunk).

    Args:
        targets: 2D float64 array, shape (n_targets, n_samples). NaN = missing.
        candidates: 2D float64 array, shape (n_candidates, n_samples). NaN = missing.

    Returns:
        2D float64 array, shape (n_targets, n_candidates), Pearson r values.
    """
    import numpy as np

    n_targets = targets.shape[0]
    n_cand = candidates.shape[0]
    n_samples = targets.shape[1]

    if n_samples < 50:
        return np.zeros((n_targets, n_cand))

    has_nan = np.any(np.isnan(targets)) or np.any(np.isnan(candidates))

    if not has_nan:
        T_means = targets.mean(axis=1, keepdims=True)
        T_c = targets - T_means
        T_stds = np.sqrt((T_c * T_c).mean(axis=1))

        C_means = candidates.mean(axis=1, keepdims=True)
        C_c = candidates - C_means
        C_stds = np.sqrt((C_c * C_c).mean(axis=1))

        # (n_targets, n_samples) @ (n_samples, n_candidates) → (n_targets, n_candidates)
        cov_matrix = (T_c @ C_c.T) / n_samples

        denom = np.outer(T_stds, C_stds)
        good = denom > 0
        result = np.zeros((n_targets, n_cand))
        result[good] = cov_matrix[good] / denom[good]
        return result

    result = np.zeros((n_targets, n_cand))
    for i in range(n_targets):
        result[i] = _pearson_r_one_vs_many(targets[i], candidates)
    return result


def _ensure_pvar_parquet(pvar_zst_path: Path) -> Path:
    """Ensure the pvar parquet cache exists and return its path.

    Follows the same naming convention as ``reference._pvar_parquet_cache_path``.
    If the cache does not exist, calls ``parse_pvar`` to build it (one-time cost).
    """
    parquet_path = pvar_zst_path.parent / (
        pvar_zst_path.stem.replace(".pvar", "") + "_pvar.parquet"
    )
    if not parquet_path.exists():
        from just_prs.reference import parse_pvar
        parse_pvar(pvar_zst_path)
    return parquet_path


def _ld_duckdb_memory_limit() -> str:
    """Resolve DuckDB's LD-staging memory limit.

    The LD pipeline may run under a much lower hard process/cgroup cap than the
    rest of the PRS code. DuckDB must know that smaller budget; otherwise it can
    plan for the general ``PRS_DUCKDB_MEMORY_LIMIT`` (often tens of GB) and then
    fail when the OS cap is hit. Default to 4 GB, capped to at most 25% of the LD
    hard limit when one is configured.
    """
    import os

    explicit = os.environ.get("PRS_LD_DUCKDB_MEMORY_LIMIT", "").strip()
    if explicit:
        return explicit

    hard_limit = _ld_memory_limit_bytes()
    if hard_limit is None:
        return os.environ.get("PRS_DUCKDB_MEMORY_LIMIT", "").strip() or f"{DEFAULT_LD_DUCKDB_MEMORY_GB:.1f}GB"

    duckdb_bytes = min(
        int(DEFAULT_LD_DUCKDB_MEMORY_GB * _LD_MEMORY_GB),
        max(int(hard_limit * 0.25), 512 * 1024 * 1024),
    )
    return f"{duckdb_bytes / _LD_MEMORY_GB:.1f}GB"


def _ld_duckdb_connect(temp_dir: Path | None = None) -> "Any":
    """Open a DuckDB connection with LD-specific low-memory settings."""
    import duckdb

    memory_limit = _ld_duckdb_memory_limit()
    config: dict[str, str] = {"memory_limit": memory_limit}
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        config["temp_directory"] = str(temp_dir)

    con = duckdb.connect(config=config)
    con.execute("SET arrow_large_buffer_size = true")
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = false")
    _log_ld_stage(
        "duckdb_connect",
        duckdb_memory_limit=memory_limit,
        duckdb_temp_dir=str(temp_dir) if temp_dir is not None else "",
    )
    return con


def _sql_literal(path: Path | str) -> str:
    """Return a SQL string literal for a filesystem path."""
    return str(path).replace("'", "''")


def _ld_memory_limit_bytes() -> int | None:
    """Return the cooperative LD-proxy RSS limit, or None when disabled."""
    import os

    if os.environ.get("PRS_LD_DISABLE_MEMORY_GUARD", "").strip().lower() in {"1", "true", "yes"}:
        return None

    explicit_gb = os.environ.get("PRS_LD_MEMORY_LIMIT_GB", "").strip()
    if explicit_gb:
        return int(float(explicit_gb) * _LD_MEMORY_GB)

    import psutil

    pct_str = os.environ.get("PRS_LD_MEMORY_LIMIT_PERCENT", "").strip()
    pct = int(pct_str) if pct_str else DEFAULT_LD_MEMORY_LIMIT_PERCENT
    return int(psutil.virtual_memory().total * pct / 100)


def _check_ld_memory_budget(stage: str, extra_bytes: int = 0) -> None:
    """Abort before an LD-proxy step can push the machine into OOM.

    This is intentionally cooperative rather than a hard OS limit: it lets
    Dagster record a normal asset failure with a clear message. It is checked
    before the largest predictable allocations (allele offsets and genotype
    chunks). DuckDB still gets its own memory_limit/spill directory separately.
    """
    limit = _ld_memory_limit_bytes()
    if limit is None:
        return

    import psutil

    rss = psutil.Process().memory_info().rss
    projected = rss + extra_bytes
    if projected <= limit:
        return

    raise MemoryError(
        "LD proxy memory guard stopped the build before system OOM: "
        f"{stage} would project RSS to {projected / _LD_MEMORY_GB:.1f} GB "
        f"over the configured limit {limit / _LD_MEMORY_GB:.1f} GB. "
        "Lower PRS_LD_TARGET_BATCH_SIZE, PRS_LD_MAX_TARGETS_PER_CHUNK, "
        "or PRS_LD_CHUNK_SIZE_BP; or raise PRS_LD_MEMORY_LIMIT_GB if the "
        "machine has enough free RAM."
    )


def _ld_memory_log_fields() -> dict[str, float]:
    """Return lightweight process/system memory fields for LD progress logs."""
    import psutil

    proc = psutil.Process()
    vm = psutil.virtual_memory()
    return {
        "rss_gb": round(proc.memory_info().rss / _LD_MEMORY_GB, 3),
        "available_gb": round(vm.available / _LD_MEMORY_GB, 3),
        "system_used_percent": round(float(vm.percent), 1),
    }


def _log_ld_stage(stage: str, **fields: Any) -> None:
    """Log an LD-proxy stage with current RSS so abrupt OOMs are localizable."""
    log_message(
        message_type="ld_proxy:stage",
        stage=stage,
        **_ld_memory_log_fields(),
        **fields,
    )


@contextmanager
def _ld_hard_memory_limit() -> Iterator[None]:
    """Temporarily apply an OS-level address-space cap to the LD build.

    The cooperative RSS checks are useful for clear messages, but they cannot
    stop sudden allocations inside native code (pgenlib, NumPy, Arrow, DuckDB).
    On POSIX systems, ``RLIMIT_AS`` makes those allocations fail inside the
    process before the kernel OOM killer can freeze the whole machine.
    """
    if _ld_memory_limit_bytes() is None:
        yield
        return

    try:
        import resource
    except ImportError:
        # Non-POSIX platforms do not expose RLIMIT_AS; cooperative checks still run.
        yield
        return

    limit = _ld_memory_limit_bytes()
    if limit is None:
        yield
        return

    original_soft, original_hard = resource.getrlimit(resource.RLIMIT_AS)
    hard_limit = original_hard
    if hard_limit != resource.RLIM_INFINITY:
        limit = min(limit, int(hard_limit))

    if original_soft != resource.RLIM_INFINITY:
        limit = min(limit, int(original_soft))

    resource.setrlimit(resource.RLIMIT_AS, (limit, original_hard))
    try:
        log_message(
            message_type="ld_proxy:hard_memory_limit_set",
            limit_gb=round(limit / _LD_MEMORY_GB, 2),
        )
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (original_soft, original_hard))


def _query_chrom_pvar(
    chrom: str,
    positions: list[int],
    pvar_parquet_path: Path,
    duckdb_temp_dir: Path | None = None,
) -> pl.DataFrame:
    """Query pvar parquet for specific positions on one chromosome via DuckDB.

    Returns only the matching rows — never loads the full 75M-row pvar into memory.
    """
    pos_df = pl.DataFrame({"qpos": positions}).cast({"qpos": pl.Int64})

    con = _ld_duckdb_connect(duckdb_temp_dir)
    con.register("qpositions", pos_df.to_arrow())

    arrow_tbl = con.execute(
        f"""
        SELECT p."POS", p."REF", p."ALT", p."ID", p.variant_idx
        FROM read_parquet('{_sql_literal(pvar_parquet_path)}') p
        INNER JOIN qpositions q ON p."POS" = q.qpos
        WHERE CAST(p.chrom AS VARCHAR) = $1
        """,
        [chrom],
    ).fetch_arrow_table()
    con.close()

    return pl.from_arrow(arrow_tbl)


def _resolve_ld_proxy_inputs(
    chip: str,
    build: str,
    ref_dir: Path,
    cache_dir: Path,
    duckdb_temp_dir: Path | None = None,
) -> _ResolvedLDProxyInputs:
    """Resolve reference-panel and chip inputs shared by LD-proxy builds."""
    from just_prs.chip_coverage import chip_typed_positions
    from just_prs.reference import parse_psam

    _log_ld_stage("load_chip_typed_positions_start", chip=chip)
    typed_positions = chip_typed_positions(chip, cache_dir)
    _log_ld_stage(
        "load_chip_typed_positions_done",
        chip=chip,
        n_typed_positions=int(typed_positions.height),
    )

    typed_df = pl.DataFrame({
        "chr": typed_positions["chr_norm"],
        "pos": typed_positions["pos"],
    })

    pvar_zst_path = ref_dir / f"{build}_1000G_ALL.pvar.zst"
    if not pvar_zst_path.exists():
        pvar_zst_path = next(ref_dir.glob("*.pvar.zst"))

    _log_ld_stage("ensure_pvar_parquet_start", pvar_zst_path=str(pvar_zst_path))
    pvar_parquet_path = _ensure_pvar_parquet(pvar_zst_path)
    _log_ld_stage("ensure_pvar_parquet_done", pvar_parquet_path=str(pvar_parquet_path))

    _log_ld_stage("count_pvar_variants_start", pvar_parquet_path=str(pvar_parquet_path))
    con = _ld_duckdb_connect(duckdb_temp_dir)
    pvar_variant_ct: int = con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql_literal(pvar_parquet_path)}')"
    ).fetchone()[0]  # type: ignore[index]
    con.close()
    _log_ld_stage("count_pvar_variants_done", pvar_variant_ct=int(pvar_variant_ct))

    _log_ld_stage("parse_psam_start")
    psam = parse_psam(next(ref_dir.glob("*.psam")))
    pgen_path = next(ref_dir.glob("*.pgen"))
    n_samples = psam.height
    _log_ld_stage(
        "parse_psam_done",
        n_samples=int(n_samples),
        pgen_path=str(pgen_path),
    )

    from just_prs.reference import (
        _allele_offsets_cache_path,
        _build_allele_offsets_cache,
        _load_allele_idx_offsets,
    )

    offsets_cache = _allele_offsets_cache_path(pvar_zst_path)
    if not offsets_cache.exists():
        _log_ld_stage("build_allele_offsets_cache_start", offsets_cache=str(offsets_cache))
        _build_allele_offsets_cache(pvar_zst_path)
        _log_ld_stage("build_allele_offsets_cache_done", offsets_cache=str(offsets_cache))
    _check_ld_memory_budget(
        stage="loading pgen allele offsets",
        extra_bytes=(pvar_variant_ct + 1) * 8,
    )
    _log_ld_stage(
        "load_allele_offsets_start",
        offsets_cache=str(offsets_cache),
        estimated_offsets_gb=round(((pvar_variant_ct + 1) * 8) / _LD_MEMORY_GB, 3),
    )
    allele_offsets = _load_allele_idx_offsets(offsets_cache, variant_ct=pvar_variant_ct)
    _log_ld_stage("load_allele_offsets_done")

    return _ResolvedLDProxyInputs(
        typed_df=typed_df,
        pvar_zst_path=pvar_zst_path,
        pvar_parquet_path=pvar_parquet_path,
        pvar_variant_ct=pvar_variant_ct,
        pgen_path=pgen_path,
        n_samples=n_samples,
        allele_offsets=allele_offsets,
    )


def _validate_ld_proxy_parquet(path: Path) -> LDProxyBuildResult:
    """Validate an LD-proxy parquet and return summary statistics."""
    df = pl.read_parquet(path)
    validate_ld_proxy_table(df)
    n_proxied = df.height
    mean_r2 = float(df["r_squared"].mean()) if n_proxied > 0 else None
    return LDProxyBuildResult(
        path=path,
        n_untyped=0,
        n_proxied=n_proxied,
        mean_r2=mean_r2,
    )


def _output_is_fresh(output_path: Path, input_paths: list[Path]) -> bool:
    """Return True when output exists, parses, and is newer than all inputs."""
    if not output_path.exists():
        return False
    try:
        lf = pl.scan_parquet(output_path)
        lf.collect_schema()
        lf.head(1).collect()
    except Exception:
        output_path.unlink(missing_ok=True)
        return False
    output_mtime = output_path.stat().st_mtime
    existing_inputs = [path for path in input_paths if path.exists()]
    return all(output_mtime >= path.stat().st_mtime for path in existing_inputs)


def _ld_outcomes_df(outcomes: list[LDProxyOutcome]) -> pl.DataFrame:
    """Convert LD-proxy outcomes to a typed Polars DataFrame."""
    rows = [
        {
            "pgs_id": outcome.pgs_id,
            "status": outcome.status,
            "path": str(outcome.path) if outcome.path is not None else None,
            "n_untyped": outcome.n_untyped,
            "n_proxied": outcome.n_proxied,
            "mean_r2": outcome.mean_r2,
            "elapsed_sec": outcome.elapsed_sec,
            "error": outcome.error,
        }
        for outcome in outcomes
    ]
    if not rows:
        return pl.DataFrame(schema=LD_PROXY_QUALITY_SCHEMA)
    return pl.DataFrame(rows, schema=LD_PROXY_QUALITY_SCHEMA)


def _process_chromosome(
    chrom: str,
    target_positions_sorted: list[int],
    typed_positions_on_chrom: list[int],
    pgen_path: Path,
    pvar_parquet_path: Path,
    n_samples: int,
    pvar_variant_ct: int,
    allele_offsets: "Any",
    window_bp: int,
    min_r2: float,
    max_chunk_span_bp: int,
    max_targets_per_chunk: int,
    output_dir: Path | None = None,
    duckdb_temp_dir: Path | None = None,
    flush_rows: int = DEFAULT_LD_PROXY_FLUSH_ROWS,
) -> list[dict] | dict[str, Any]:
    """Process one chromosome: one PgenReader open, chunked reads + correlation.

    Opens the PgenReader once, then groups targets into position-bounded chunks.
    Each chunk reads its genotypes from the already-open reader — no repeated
    open/close overhead. Memory per chunk: ``~chunk_variants × n_samples × 4``.

    ``allele_offsets`` (the ~variant_ct-length cumulative allele-index array,
    ~600 MB for 1000G) is built once by the caller and shared read-only across
    all worker threads, so adding workers does not multiply that cost.
    """
    import numpy as np

    from just_prs.reference import _require_pgenlib

    _require_pgenlib()
    import pgenlib

    all_query_pos = sorted(set(target_positions_sorted) | set(typed_positions_on_chrom))
    chrom_pvar = _query_chrom_pvar(chrom, all_query_pos, pvar_parquet_path, duckdb_temp_dir)
    if chrom_pvar.height == 0:
        if output_dir is not None:
            return {"paths": [], "n_proxied": 0, "sum_r2": 0.0}
        return []

    target_pos_set = set(target_positions_sorted)
    typed_pos_set = set(typed_positions_on_chrom)

    target_pvar = (
        chrom_pvar.filter(pl.col("POS").is_in(list(target_pos_set)))
        .group_by("POS").first().sort("POS")
    )
    typed_pvar = (
        chrom_pvar.filter(pl.col("POS").is_in(list(typed_pos_set)))
        .group_by("POS").first().sort("POS")
    )
    del chrom_pvar

    if target_pvar.height == 0 or typed_pvar.height == 0:
        if output_dir is not None:
            return {"paths": [], "n_proxied": 0, "sum_r2": 0.0}
        return []

    typed_pos_np = typed_pvar["POS"].to_numpy()
    typed_pos_order = np.argsort(typed_pos_np)
    typed_pos_sorted = typed_pos_np[typed_pos_order]
    typed_vidx_np = typed_pvar["variant_idx"].to_numpy()
    typed_meta = typed_pvar.select("POS", "REF", "ALT", "ID", "variant_idx").to_dicts()

    target_meta = target_pvar.select("POS", "REF", "ALT", "variant_idx").to_dicts()

    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_start = 0
    for t in target_meta:
        if not cur:
            cur.append(t)
            cur_start = t["POS"]
        elif t["POS"] - cur_start <= max_chunk_span_bp and len(cur) < max_targets_per_chunk:
            cur.append(t)
        else:
            chunks.append(cur)
            cur = [t]
            cur_start = t["POS"]
    if cur:
        chunks.append(cur)

    result_rows: list[dict] = []
    part_paths: list[Path] = []
    part_idx = 0
    n_proxied = 0
    sum_r2 = 0.0

    def _flush_rows() -> None:
        nonlocal part_idx
        if output_dir is None or not result_rows:
            return
        part_path = output_dir / f"ld_proxy_chr{chrom}_{part_idx:05d}.parquet"
        pl.DataFrame(result_rows, schema=LD_PROXY_TABLE_SCHEMA).write_parquet(part_path)
        part_paths.append(part_path)
        result_rows.clear()
        part_idx += 1

    with pgenlib.PgenReader(
        str(pgen_path).encode("utf-8"),
        raw_sample_ct=n_samples,
        variant_ct=pvar_variant_ct,
        allele_idx_offsets=allele_offsets,
    ) as greader:
        actual_samples = greader.get_raw_sample_ct()

        for chunk in chunks:
            chunk_min = chunk[0]["POS"]
            chunk_max = chunk[-1]["POS"]
            read_lo = chunk_min - window_bp
            read_hi = chunk_max + window_bp

            cand_lo = int(np.searchsorted(typed_pos_sorted, read_lo, side="left"))
            cand_hi = int(np.searchsorted(typed_pos_sorted, read_hi, side="right"))
            if cand_lo >= cand_hi:
                continue

            chunk_target_vidxs = np.array(
                [t["variant_idx"] for t in chunk], dtype=np.uint32,
            )
            chunk_cand_vidxs = np.array(
                [typed_vidx_np[typed_pos_order[i]] for i in range(cand_lo, cand_hi)],
                dtype=np.uint32,
            )
            all_indices = np.unique(np.concatenate([chunk_target_vidxs, chunk_cand_vidxs]))

            sort_order = np.argsort(all_indices)
            sorted_indices = all_indices[sort_order]
            estimated_chunk_bytes = (
                len(sorted_indices) * actual_samples * 5  # int8 read buffer + float32 genotype matrix
                + (len(chunk_target_vidxs) + len(chunk_cand_vidxs)) * actual_samples * 8  # centered matrices/temporaries
                + len(chunk_target_vidxs) * len(chunk_cand_vidxs) * 8  # r matrix
                + 256 * 1024 * 1024  # numpy / Arrow / Python overhead cushion
            )
            _check_ld_memory_budget(
                stage=f"chromosome {chrom} genotype chunk ({len(sorted_indices)} variants)",
                extra_bytes=estimated_chunk_bytes,
            )

            try:
                geno_buf = np.empty((len(sorted_indices), actual_samples), dtype=np.int8)
                greader.read_list(sorted_indices, geno_buf)
                unsort_order = np.argsort(sort_order)
                geno_raw = geno_buf[unsort_order]
            except Exception:
                continue

            idx_to_row: dict[int, int] = {int(v): i for i, v in enumerate(all_indices)}
            geno_f = geno_raw.astype(np.float32)
            geno_f[geno_f < 0] = np.nan
            del geno_raw, geno_buf

            n_cand_in_chunk = cand_hi - cand_lo
            cand_geno_rows_full = np.array(
                [idx_to_row[int(typed_vidx_np[typed_pos_order[i]])]
                 for i in range(cand_lo, cand_hi)],
                dtype=np.intp,
            )
            candidates_geno_full = geno_f[cand_geno_rows_full]

            valid_targets: list[tuple[dict, int]] = []
            for t_meta in chunk:
                row_idx = idx_to_row.get(t_meta["variant_idx"])
                if row_idx is not None:
                    valid_targets.append((t_meta, row_idx))

            if not valid_targets:
                del geno_f, idx_to_row
                continue

            target_rows = np.array([vt[1] for vt in valid_targets], dtype=np.intp)
            targets_geno = geno_f[target_rows]

            r_matrix = _pearson_r_many_vs_many(targets_geno, candidates_geno_full)

            for ti, (t_meta, _) in enumerate(valid_targets):
                t_pos = t_meta["POS"]
                t_lo = int(np.searchsorted(typed_pos_sorted, t_pos - window_bp, side="left"))
                t_hi = int(np.searchsorted(typed_pos_sorted, t_pos + window_bp, side="right"))
                if t_lo >= t_hi:
                    continue

                local_lo = t_lo - cand_lo
                local_hi = t_hi - cand_lo
                local_lo = max(0, local_lo)
                local_hi = min(n_cand_in_chunk, local_hi)
                if local_lo >= local_hi:
                    continue

                window_r = r_matrix[ti, local_lo:local_hi]
                best_in_window = int(np.argmax(np.abs(window_r)))
                best_r = float(window_r[best_in_window])
                best_r2 = best_r * best_r

                if best_r2 >= min_r2:
                    best_orig_idx = int(typed_pos_order[cand_lo + local_lo + best_in_window])
                    proxy = typed_meta[best_orig_idx]
                    n_proxied += 1
                    sum_r2 += best_r2
                    result_rows.append({
                        "target_chr": chrom,
                        "target_pos": t_pos,
                        "target_ref": t_meta.get("REF", ""),
                        "target_alt": t_meta.get("ALT", ""),
                        "proxy_chr": chrom,
                        "proxy_pos": proxy["POS"],
                        "proxy_rsid": proxy.get("ID", ""),
                        "proxy_ref": proxy.get("REF", ""),
                        "proxy_alt": proxy.get("ALT", ""),
                        "r_squared": round(best_r2, 6),
                        "r_signed": round(best_r, 6),
                    })
                    if output_dir is not None and len(result_rows) >= flush_rows:
                        _flush_rows()

            del geno_f, idx_to_row, r_matrix

    _flush_rows()
    if output_dir is not None:
        return {"paths": part_paths, "n_proxied": n_proxied, "sum_r2": sum_r2}
    return result_rows


def build_ld_proxy_table(
    chip: str,
    build: str,
    ref_dir: Path,
    cache_dir: Path,
    panel: str = "1000g",
    scoring_variants_lf: pl.LazyFrame | None = None,
    window_kb: int = DEFAULT_PROXY_WINDOW_KB,
    min_r2: float = DEFAULT_MIN_R2,
    progress_callback: "callable | None" = None,
    max_workers: int | None = None,
    chunk_size_bp: int | None = None,
    max_targets_per_chunk: int | None = None,
    output_path: Path | None = None,
    resolved_inputs: _ResolvedLDProxyInputs | None = None,
) -> pl.DataFrame | LDProxyBuildResult:
    """Compute an LD-proxy table from a reference panel.

    For each PRS variant not typed on the chip, find the best proxy among
    chip-typed variants within a genomic window, using Pearson correlation
    of reference panel genotypes.

    Memory-efficient: queries the pvar parquet via DuckDB (never loads the
    full 75M-row DataFrame), processes targets in position-bounded chunks,
    and parallelises chromosomes with ``ThreadPoolExecutor``.

    This is a pipeline-side function (requires pgenlib and the reference panel).

    Args:
        chip: Chip identifier (e.g. ``"gsa_v3"``).
        build: Genome build (``"GRCh37"`` or ``"GRCh38"``).
        ref_dir: Reference panel directory (contains .pgen/.pvar.zst/.psam).
        cache_dir: Cache directory for scoring files and chip manifests.
        panel: Panel identifier (``"1000g"`` or ``"hgdp_1kg"``).
        scoring_variants_lf: Optional LazyFrame with ``chr_name_norm``,
            ``chr_pos_norm`` columns. If None, collected from all cached
            scoring parquets for the build.
        window_kb: Search window in kb around each target variant.
        min_r2: Minimum r² threshold for including a proxy.
        progress_callback: Optional progress callback.
        max_workers: Thread pool size (chromosomes run in parallel). Defaults
            to 1 (conservative). The large allele-offsets array is shared across
            threads, so the only per-worker memory is the small per-chunk
            genotype matrix (tens of MB); raising this is safe given RAM headroom
            and speeds up the build roughly per added core (pgenlib/numpy release
            the GIL during reads and matmuls). Override with ``PRS_LD_MAX_WORKERS``.
        chunk_size_bp: Max genomic span per target chunk in bp.
            Defaults to 250kb.
            Override with ``PRS_LD_CHUNK_SIZE_BP`` env var.
        max_targets_per_chunk: Maximum number of target variants per chunk.
            Override with ``PRS_LD_MAX_TARGETS_PER_CHUNK`` env var.
        output_path: Optional parquet path. When provided, proxy rows are streamed
            through temporary parquet parts into this final file and a summary is
            returned instead of materializing the whole table in memory.
        resolved_inputs: Optional shared reference/chip inputs. Batch callers
            pass this to avoid rebuilding pvar/psam/allele-offset state per PGS.

    Returns:
        DataFrame with the LD proxy table schema.
    """
    with start_action(
        action_type="ld_proxy:build_table",
        chip=chip,
        build=build,
        panel=panel,
        window_kb=window_kb,
        min_r2=min_r2,
    ):
        import os
        import tempfile
        from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

        with tempfile.TemporaryDirectory(prefix="just_prs_ld_proxy_") as tmp_raw:
            tmp_dir = Path(tmp_raw)
            duckdb_temp_dir = tmp_dir / "duckdb"
            parts_dir = tmp_dir / "parts"
            parts_dir.mkdir(parents=True, exist_ok=True)

            _log_ld_stage(
                "start_build",
                chip=chip,
                build=build,
                panel=panel,
                temp_dir=str(tmp_dir),
                output_path=str(output_path) if output_path is not None else "",
            )

            scoring_tmp = tmp_dir / "scoring_positions.parquet"
            if scoring_variants_lf is None:
                raise ValueError(
                    "LD-proxy builds now require an explicit per-PGS scoring_variants_lf. "
                    "Use build_ld_proxy_for_pgs_id() or build_ld_proxy_batch() so full-catalog "
                    "coverage is reached by resumable per-PGS units, not a catalog-wide union."
                )

            _log_ld_stage("sink_supplied_scoring_variants_start", build=build)
            scoring_variants_lf.select(
                pl.col("chr_name_norm").alias("chr"),
                pl.col("chr_pos_norm").alias("pos"),
            ).sink_parquet(scoring_tmp)
            scoring_stats = pl.scan_parquet(scoring_tmp).select(pl.len().alias("n")).collect()
            _log_ld_stage(
                "sink_supplied_scoring_variants_done",
                build=build,
                n_scoring_positions=int(scoring_stats["n"][0]),
                scoring_tmp=str(scoring_tmp),
            )

            resolved = resolved_inputs or _resolve_ld_proxy_inputs(
                chip=chip,
                build=build,
                ref_dir=ref_dir,
                cache_dir=cache_dir,
                duckdb_temp_dir=duckdb_temp_dir,
            )
            typed_df = resolved.typed_df
            pvar_parquet_path = resolved.pvar_parquet_path
            pvar_variant_ct = resolved.pvar_variant_ct
            pgen_path = resolved.pgen_path
            n_samples = resolved.n_samples
            allele_offsets = resolved.allele_offsets

            window_bp = window_kb * 1000
            max_chunk_span = max(1, chunk_size_bp or int(
                os.environ.get("PRS_LD_CHUNK_SIZE_BP", str(DEFAULT_LD_CHUNK_SIZE_BP))
            ))
            max_targets = max(1, max_targets_per_chunk or int(
                os.environ.get("PRS_LD_MAX_TARGETS_PER_CHUNK", str(DEFAULT_LD_MAX_TARGETS_PER_CHUNK))
            ))
            target_batch_size = max(1, int(
                os.environ.get("PRS_LD_TARGET_BATCH_SIZE", str(DEFAULT_LD_TARGET_BATCH_SIZE))
            ))
            flush_rows = int(os.environ.get("PRS_LD_PROXY_FLUSH_ROWS", str(DEFAULT_LD_PROXY_FLUSH_ROWS)))

            con = _ld_duckdb_connect(duckdb_temp_dir)
            con.register("typed", typed_df.to_arrow())

            _log_ld_stage("plan_untyped_targets_start")
            chromosomes_rows = con.execute(
                f"""
                SELECT DISTINCT s.chr
                FROM read_parquet('{_sql_literal(scoring_tmp)}') s
                WHERE NOT EXISTS (
                    SELECT 1 FROM typed t WHERE t.chr = s.chr AND t.pos = s.pos
                )
                ORDER BY s.chr
                """
            ).fetchall()
            chromosomes = [r[0] for r in chromosomes_rows]

            untyped_counts_by_chrom: dict[str, int] = {}
            for chrom in chromosomes:
                count_row = con.execute(
                    f"""
                    SELECT count(*)
                    FROM read_parquet('{_sql_literal(scoring_tmp)}') s
                    WHERE s.chr = $1
                      AND NOT EXISTS (
                        SELECT 1 FROM typed t WHERE t.chr = s.chr AND t.pos = s.pos
                      )
                    """,
                    [chrom],
                ).fetchone()
                n_targets = int(count_row[0]) if count_row is not None else 0
                if n_targets > 0:
                    untyped_counts_by_chrom[chrom] = n_targets

            n_untyped = sum(untyped_counts_by_chrom.values())
            n_scoring_row = con.execute(
                f"SELECT count(*) FROM read_parquet('{_sql_literal(scoring_tmp)}')"
            ).fetchone()
            n_scoring = int(n_scoring_row[0]) if n_scoring_row is not None else n_untyped
            _log_ld_stage(
                "plan_untyped_targets_done",
                n_chromosomes=len(untyped_counts_by_chrom),
                n_untyped=int(n_untyped),
                n_scoring=int(n_scoring),
                top_untyped_chromosomes=str(
                    sorted(untyped_counts_by_chrom.items(), key=lambda item: item[1], reverse=True)[:5]
                ),
            )
            log_message(
                message_type="ld_proxy:untyped_variants",
                n_untyped=n_untyped,
                n_total_scoring=n_scoring,
                n_typed_on_chip=typed_df.height,
            )

            env_workers = os.environ.get("PRS_LD_MAX_WORKERS")
            n_workers = max_workers or (int(env_workers) if env_workers else DEFAULT_LD_MAX_WORKERS)
            n_workers = max(1, min(n_workers, len(untyped_counts_by_chrom))) if untyped_counts_by_chrom else 1
            total_batches = sum(
                (n_targets + target_batch_size - 1) // target_batch_size
                for n_targets in untyped_counts_by_chrom.values()
            )

            log_message(
                message_type="ld_proxy:start_parallel",
                n_chromosomes=len(untyped_counts_by_chrom),
                n_target_batches=total_batches,
                target_batch_size=target_batch_size,
                n_workers=n_workers,
                max_chunk_span_bp=max_chunk_span,
                max_targets_per_chunk=max_targets,
                pvar_variant_ct=pvar_variant_ct,
                streaming_output=output_path is not None,
            )

            all_rows: list[dict] = []
            all_part_paths: list[Path] = []
            total_proxied = 0
            total_r2 = 0.0

            def _typed_positions_for_chrom(chrom: str) -> list[int]:
                rows = con.execute(
                    "SELECT pos FROM typed WHERE chr = $1 ORDER BY pos",
                    [chrom],
                ).fetchall()
                return [int(r[0]) for r in rows]

            def _untyped_target_batch(chrom: str, limit: int, offset: int) -> list[int]:
                rows = con.execute(
                    f"""
                    SELECT DISTINCT s.pos
                    FROM read_parquet('{_sql_literal(scoring_tmp)}') s
                    WHERE s.chr = $1
                      AND NOT EXISTS (
                        SELECT 1 FROM typed t WHERE t.chr = s.chr AND t.pos = s.pos
                      )
                    ORDER BY s.pos
                    LIMIT {int(limit)} OFFSET {int(offset)}
                    """,
                    [chrom],
                ).fetchall()
                return [int(r[0]) for r in rows]

            def _task_specs() -> "Any":
                for chrom, n_targets in untyped_counts_by_chrom.items():
                    chrom_typed = _typed_positions_for_chrom(chrom)
                    if not chrom_typed:
                        continue
                    for batch_idx, offset in enumerate(range(0, n_targets, target_batch_size)):
                        chrom_targets = _untyped_target_batch(chrom, target_batch_size, offset)
                        if chrom_targets:
                            yield chrom, batch_idx, chrom_typed, chrom_targets

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                task_iter = iter(_task_specs())
                futures: dict[Future, str] = {}

                def _submit_next() -> bool:
                    try:
                        chrom, batch_idx, chrom_typed, chrom_targets = next(task_iter)
                    except StopIteration:
                        return False
                    _check_ld_memory_budget(stage=f"submitting chromosome {chrom} batch {batch_idx}")
                    future = pool.submit(
                        _process_chromosome,
                        chrom=chrom,
                        target_positions_sorted=chrom_targets,
                        typed_positions_on_chrom=chrom_typed,
                        pgen_path=pgen_path,
                        pvar_parquet_path=pvar_parquet_path,
                        n_samples=n_samples,
                        pvar_variant_ct=pvar_variant_ct,
                        allele_offsets=allele_offsets,
                        window_bp=window_bp,
                        min_r2=min_r2,
                        max_chunk_span_bp=max_chunk_span,
                        max_targets_per_chunk=max_targets,
                        output_dir=parts_dir if output_path is not None else None,
                        duckdb_temp_dir=duckdb_temp_dir,
                        flush_rows=flush_rows,
                    )
                    futures[future] = f"{chrom}:{batch_idx}"
                    return True

                for _ in range(n_workers):
                    if not _submit_next():
                        break

                completed = 0
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        chrom_label = futures.pop(future)
                        completed += 1
                        chrom_n_proxied = 0
                        try:
                            chrom_result = future.result()
                            if output_path is not None:
                                assert isinstance(chrom_result, dict)
                                chrom_paths = chrom_result["paths"]
                                chrom_n_proxied = int(chrom_result["n_proxied"])
                                all_part_paths.extend(chrom_paths)
                                total_proxied += chrom_n_proxied
                                total_r2 += float(chrom_result["sum_r2"])
                            else:
                                assert isinstance(chrom_result, list)
                                all_rows.extend(chrom_result)
                                chrom_n_proxied = len(chrom_result)
                                total_proxied = len(all_rows)
                            log_message(
                                message_type="ld_proxy:target_batch_done",
                                chromosome=chrom_label,
                                n_proxies=chrom_n_proxied,
                                progress=f"{completed}/{total_batches}",
                            )
                        except Exception as exc:
                            log_message(
                                message_type="ld_proxy:target_batch_failed",
                                chromosome=chrom_label,
                                error=str(exc),
                            )
                            raise
                        if progress_callback:
                            progress_callback({
                                "chromosome": chrom_label,
                                "chrom_index": completed,
                                "n_chromosomes": total_batches,
                                "n_proxies_found": total_proxied,
                            })
                        _submit_next()
            con.close()

            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if all_part_paths:
                    pl.concat([pl.scan_parquet(path) for path in all_part_paths]).sink_parquet(output_path)
                else:
                    pl.DataFrame(schema=LD_PROXY_TABLE_SCHEMA).write_parquet(output_path)
                mean_r2 = total_r2 / total_proxied if total_proxied > 0 else None
                log_message(
                    message_type="ld_proxy:build_complete",
                    n_untyped=n_untyped,
                    n_proxied=total_proxied,
                    proxy_rate=total_proxied / max(n_untyped, 1),
                    mean_r2=mean_r2,
                    output_path=str(output_path),
                )
                return LDProxyBuildResult(
                    path=output_path,
                    n_untyped=n_untyped,
                    n_proxied=total_proxied,
                    mean_r2=mean_r2,
                )

            result = pl.DataFrame(all_rows, schema=LD_PROXY_TABLE_SCHEMA)

            log_message(
                message_type="ld_proxy:build_complete",
                n_untyped=n_untyped,
                n_proxied=result.height,
                proxy_rate=result.height / max(n_untyped, 1),
                mean_r2=float(result["r_squared"].mean()) if result.height > 0 else None,
            )

            return result


def scoring_variants_for_pgs_ids(
    cache_dir: Path,
    build: str,
    pgs_ids: list[str],
) -> pl.LazyFrame:
    """Build a scoring-position LazyFrame for a bounded set of PGS IDs.

    This is the safe path for one-score or small pilot LD builds. It avoids the
    full-catalog DuckDB staging table entirely.
    """
    from just_prs.chip_coverage import _normalize_chr_expr
    from just_prs.scoring import scoring_parquet_path

    scores_dir = cache_dir / "scores"
    frames: list[pl.LazyFrame] = []
    missing: list[str] = []
    unusable: list[str] = []
    for raw_id in pgs_ids:
        pgs_id = raw_id.strip().upper()
        if not pgs_id:
            continue
        parquet_path = scoring_parquet_path(pgs_id, scores_dir, build)
        if not parquet_path.exists():
            missing.append(pgs_id)
            continue
        schema = pl.scan_parquet(parquet_path).collect_schema().names()
        if "hm_chr" not in schema or "hm_pos" not in schema:
            unusable.append(pgs_id)
            continue
        frames.append(
            pl.scan_parquet(parquet_path).select(
                _normalize_chr_expr("hm_chr").alias("chr_name_norm"),
                pl.col("hm_pos").cast(pl.Int64).alias("chr_pos_norm"),
            )
        )

    if missing:
        raise FileNotFoundError(
            f"Missing scoring parquet(s) for {build}: {', '.join(missing)} in {scores_dir}"
        )
    if unusable:
        raise ValueError(
            f"Scoring parquet(s) missing hm_chr/hm_pos for {build}: {', '.join(unusable)}"
        )
    if not frames:
        raise ValueError("No PGS IDs supplied for LD proxy scoring variants.")

    _log_ld_stage(
        "selected_pgs_scoring_variants_ready",
        build=build,
        pgs_ids=",".join(pgs_ids),
        n_pgs_ids=len(frames),
    )
    return pl.concat(frames).unique()


def build_ld_proxy_for_pgs_id(
    pgs_id: str,
    chip: str,
    build: str,
    ref_dir: Path,
    cache_dir: Path,
    panel: str = "1000g",
    output_path: Path | None = None,
    resolved_inputs: _ResolvedLDProxyInputs | None = None,
    window_kb: int = DEFAULT_PROXY_WINDOW_KB,
    min_r2: float = DEFAULT_MIN_R2,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    max_workers: int | None = None,
    chunk_size_bp: int | None = None,
    max_targets_per_chunk: int | None = None,
) -> LDProxyBuildResult:
    """Build one per-PGS LD-proxy parquet.

    This is the atomic unit for both interactive single-score runs and full
    catalog precomputation. It reads only the requested PGS scoring variants and
    writes one validated parquet atomically.
    """
    normalized_id = pgs_id.strip().upper()
    if not normalized_id:
        raise ValueError("PGS ID must be non-empty.")

    final_path = output_path or ld_proxy_pgs_path(cache_dir, normalized_id, chip, build, panel)
    tmp_path = final_path.with_name(f".{final_path.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    scoring_lf = scoring_variants_for_pgs_ids(cache_dir, build, [normalized_id])
    result = build_ld_proxy_table(
        chip=chip,
        build=build,
        ref_dir=ref_dir,
        cache_dir=cache_dir,
        panel=panel,
        scoring_variants_lf=scoring_lf,
        window_kb=window_kb,
        min_r2=min_r2,
        progress_callback=progress_callback,
        max_workers=max_workers,
        chunk_size_bp=chunk_size_bp,
        max_targets_per_chunk=max_targets_per_chunk,
        output_path=tmp_path,
        resolved_inputs=resolved_inputs,
    )
    if not isinstance(result, LDProxyBuildResult):
        raise TypeError("Expected LDProxyBuildResult when output_path is set.")

    validated = _validate_ld_proxy_parquet(tmp_path)
    tmp_path.replace(final_path)
    return LDProxyBuildResult(
        path=final_path,
        n_untyped=result.n_untyped,
        n_proxied=validated.n_proxied,
        mean_r2=validated.mean_r2,
    )


def build_ld_proxy_batch(
    pgs_ids: list[str],
    chip: str,
    build: str,
    ref_dir: Path,
    cache_dir: Path,
    panel: str = "1000g",
    skip_existing: bool = True,
    window_kb: int = DEFAULT_PROXY_WINDOW_KB,
    min_r2: float = DEFAULT_MIN_R2,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    max_workers: int | None = None,
    chunk_size_bp: int | None = None,
    max_targets_per_chunk: int | None = None,
) -> LDProxyBatchResult:
    """Build per-PGS LD-proxy parquets with shared setup and resumability."""
    import tempfile
    import time

    from just_prs.scoring import scoring_parquet_path

    normalized_ids = list(dict.fromkeys(pid.strip().upper() for pid in pgs_ids if pid.strip()))
    if not normalized_ids:
        raise ValueError("No PGS IDs supplied for LD-proxy batch.")

    output_dir = ld_proxy_dir(cache_dir, chip, build, panel)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = ld_proxy_quality_path(cache_dir, chip, build, panel)
    scores_dir = cache_dir / "scores"
    pvar_zst_path = ref_dir / f"{build}_1000G_ALL.pvar.zst"
    if not pvar_zst_path.exists():
        pvar_candidates = list(ref_dir.glob("*.pvar.zst"))
        if pvar_candidates:
            pvar_zst_path = pvar_candidates[0]

    outcomes: list[LDProxyOutcome] = []
    with start_action(
        action_type="ld_proxy:build_batch",
        chip=chip,
        build=build,
        panel=panel,
        n_pgs_ids=len(normalized_ids),
    ):
        ids_to_build: list[tuple[int, str]] = []
        for idx, pgs_id in enumerate(normalized_ids, start=1):
            start_time = time.monotonic()
            output_path = ld_proxy_pgs_path(cache_dir, pgs_id, chip, build, panel)
            scoring_path = scoring_parquet_path(pgs_id, scores_dir, build)
            if skip_existing and _output_is_fresh(output_path, [scoring_path, pvar_zst_path]):
                cached = _validate_ld_proxy_parquet(output_path)
                elapsed = time.monotonic() - start_time
                outcomes.append(LDProxyOutcome(
                    pgs_id=pgs_id,
                    status="cached",
                    path=output_path,
                    n_untyped=None,
                    n_proxied=cached.n_proxied,
                    mean_r2=cached.mean_r2,
                    elapsed_sec=elapsed,
                ))
                if progress_callback is not None:
                    progress_callback({
                        "pgs_id": pgs_id,
                        "index": idx,
                        "total": len(normalized_ids),
                        "status": "cached",
                        "n_ok": sum(1 for outcome in outcomes if outcome.status in {"ok", "cached"}),
                        "n_failed": sum(1 for outcome in outcomes if outcome.status == "failed"),
                    })
            else:
                ids_to_build.append((idx, pgs_id))

        if not ids_to_build:
            quality_df = _ld_outcomes_df(outcomes)
            quality_df.write_parquet(quality_path)
            return LDProxyBatchResult(
                panel=panel,
                chip=chip,
                build=build,
                outcomes=outcomes,
                quality_df=quality_df,
                output_dir=output_dir,
            )

        with tempfile.TemporaryDirectory(prefix="just_prs_ld_proxy_batch_") as tmp_raw:
            tmp_dir = Path(tmp_raw)
            resolved = _resolve_ld_proxy_inputs(
                chip=chip,
                build=build,
                ref_dir=ref_dir,
                cache_dir=cache_dir,
                duckdb_temp_dir=tmp_dir / "duckdb",
            )

            for idx, pgs_id in ids_to_build:
                start_time = time.monotonic()
                output_path = ld_proxy_pgs_path(cache_dir, pgs_id, chip, build, panel)

                try:
                    result = build_ld_proxy_for_pgs_id(
                        pgs_id=pgs_id,
                        chip=chip,
                        build=build,
                        ref_dir=ref_dir,
                        cache_dir=cache_dir,
                        panel=panel,
                        output_path=output_path,
                        resolved_inputs=resolved,
                        window_kb=window_kb,
                        min_r2=min_r2,
                        progress_callback=progress_callback,
                        max_workers=max_workers,
                        chunk_size_bp=chunk_size_bp,
                        max_targets_per_chunk=max_targets_per_chunk,
                    )
                    elapsed = time.monotonic() - start_time
                    outcomes.append(LDProxyOutcome(
                        pgs_id=pgs_id,
                        status="ok",
                        path=result.path,
                        n_untyped=result.n_untyped,
                        n_proxied=result.n_proxied,
                        mean_r2=result.mean_r2,
                        elapsed_sec=elapsed,
                    ))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    elapsed = time.monotonic() - start_time
                    outcomes.append(LDProxyOutcome(
                        pgs_id=pgs_id,
                        status="failed",
                        path=output_path,
                        elapsed_sec=elapsed,
                        error=f"{type(exc).__name__}: {exc}",
                    ))

                quality_df = _ld_outcomes_df(outcomes)
                quality_df.write_parquet(quality_path)
                latest = outcomes[-1]
                log_message(
                    message_type="ld_proxy:pgs_done",
                    pgs_id=pgs_id,
                    status=latest.status,
                    progress=f"{idx}/{len(normalized_ids)}",
                    n_proxied=latest.n_proxied or 0,
                    elapsed_sec=latest.elapsed_sec,
                )
                if progress_callback is not None:
                    progress_callback({
                        "pgs_id": pgs_id,
                        "index": idx,
                        "total": len(normalized_ids),
                        "status": latest.status,
                        "n_ok": sum(1 for outcome in outcomes if outcome.status in {"ok", "cached"}),
                        "n_failed": sum(1 for outcome in outcomes if outcome.status == "failed"),
                    })

    quality_df = _ld_outcomes_df(outcomes)
    quality_df.write_parquet(quality_path)
    log_message(
        message_type="ld_proxy:batch_complete",
        panel=panel,
        chip=chip,
        build=build,
        n_total=len(outcomes),
        n_ok=sum(1 for outcome in outcomes if outcome.status == "ok"),
        n_cached=sum(1 for outcome in outcomes if outcome.status == "cached"),
        n_failed=sum(1 for outcome in outcomes if outcome.status == "failed"),
    )
    return LDProxyBatchResult(
        panel=panel,
        chip=chip,
        build=build,
        outcomes=outcomes,
        quality_df=quality_df,
        output_dir=output_dir,
    )
