"""Synthetic tests for LD-proxy substitution (no network, no reference panel).

Tests cover:
- LD proxy table schema validation
- Proxy lookup with known r² values
- Weight adjustment: w_proxy = w_original * r_signed
- Min r² threshold rejection
- End-to-end apply_ld_proxies on a scoring LazyFrame
"""

from pathlib import Path

import polars as pl
import pytest

import numpy as np

from just_prs.ld_proxy import (
    DEFAULT_MIN_R2,
    LD_PROXY_TABLE_COLUMNS,
    LD_PROXY_TABLE_SCHEMA,
    LDProxyBatchResult,
    LDProxyOutcome,
    _pearson_r_one_vs_many,
    apply_ld_proxies,
    build_ld_proxy_batch,
    ld_proxy_dir,
    ld_proxy_pgs_path,
    ld_proxy_quality_path,
    ld_proxy_table_path,
    merge_ld_proxy_tables,
    validate_ld_proxy_table,
)


def _make_ld_table(rows: list[dict]) -> pl.DataFrame:
    """Build a synthetic LD proxy table DataFrame."""
    return pl.DataFrame(rows, schema=LD_PROXY_TABLE_SCHEMA)


def _make_scoring_lf(variants: list[dict]) -> pl.LazyFrame:
    """Build a minimal scoring LazyFrame."""
    defaults = {
        "chr_name_norm": pl.Utf8,
        "chr_pos_norm": pl.Int64,
        "effect_allele": pl.Utf8,
        "other_allele": pl.Utf8,
        "effect_weight": pl.Float64,
        "allelefrequency_effect": pl.Float64,
    }
    for v in variants:
        for k in defaults:
            v.setdefault(k, None)
    return pl.DataFrame(variants).cast(defaults).lazy()


def _make_genotypes_lf(positions: list[tuple[str, int]]) -> pl.LazyFrame:
    """Build a minimal genotypes LazyFrame with positions that are 'typed'."""
    return pl.DataFrame({
        "chrom": [p[0] for p in positions],
        "pos": [p[1] for p in positions],
        "ref": ["A"] * len(positions),
        "alt": ["G"] * len(positions),
        "GT": ["0/1"] * len(positions),
    }).lazy()


def test_proxy_table_schema_validation() -> None:
    good = _make_ld_table([{
        "target_chr": "1", "target_pos": 1000, "target_ref": "A", "target_alt": "G",
        "proxy_chr": "1", "proxy_pos": 1500, "proxy_rsid": "rs99", "proxy_ref": "C", "proxy_alt": "T",
        "r_squared": 0.85, "r_signed": 0.92,
    }])
    validate_ld_proxy_table(good)

    bad = good.drop("r_squared")
    with pytest.raises(ValueError, match="missing columns"):
        validate_ld_proxy_table(bad)


def test_proxy_table_has_all_columns() -> None:
    assert set(LD_PROXY_TABLE_COLUMNS) == set(LD_PROXY_TABLE_SCHEMA.keys())


def test_ld_proxy_table_path() -> None:
    p = ld_proxy_table_path(Path("/cache"), "gsa_v3", "GRCh37", "1000g")
    assert p == Path("/cache/percentiles/1000g_ld_proxy_gsa_v3_GRCh37.parquet")


def test_ld_proxy_pgs_path() -> None:
    p = ld_proxy_pgs_path(Path("/cache"), "pgs000001", "gsa_v3", "GRCh38", "1000g")
    assert p == Path("/cache/percentiles/ld_proxy/1000g/gsa_v3/GRCh38/PGS000001.parquet")


def test_ld_proxy_batch_skips_cached_without_resolving_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    out_path = ld_proxy_pgs_path(cache_dir, "PGS000001", "gsa_v3", "GRCh38", "1000g")
    out_path.parent.mkdir(parents=True)
    _make_ld_table([{
        "target_chr": "1", "target_pos": 1000, "target_ref": "A", "target_alt": "G",
        "proxy_chr": "1", "proxy_pos": 1500, "proxy_rsid": "rs99",
        "proxy_ref": "C", "proxy_alt": "T",
        "r_squared": 0.90, "r_signed": 0.95,
    }]).write_parquet(out_path)

    def _fail_resolve(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cached batch should not resolve reference inputs")

    monkeypatch.setattr("just_prs.ld_proxy._resolve_ld_proxy_inputs", _fail_resolve)

    result = build_ld_proxy_batch(
        pgs_ids=["PGS000001"],
        chip="gsa_v3",
        build="GRCh38",
        ref_dir=ref_dir,
        cache_dir=cache_dir,
        skip_existing=True,
    )

    assert result.n_cached == 1
    assert result.n_ok == 0
    assert result.n_failed == 0
    assert result.coverage_ratio == 1.0
    assert ld_proxy_quality_path(cache_dir, "gsa_v3", "GRCh38", "1000g").exists()


def test_apply_proxies_weight_adjustment() -> None:
    """Verify w_proxy = w_original * r_signed for a single proxied variant."""
    scoring = _make_scoring_lf([
        {"chr_name_norm": "1", "chr_pos_norm": 1000, "effect_allele": "G",
         "other_allele": "A", "effect_weight": 2.0},
    ])

    genotypes = _make_genotypes_lf([("1", 1500)])

    ld_table = _make_ld_table([{
        "target_chr": "1", "target_pos": 1000, "target_ref": "A", "target_alt": "G",
        "proxy_chr": "1", "proxy_pos": 1500, "proxy_rsid": "rs99",
        "proxy_ref": "C", "proxy_alt": "T",
        "r_squared": 0.90, "r_signed": -0.95,
    }])

    enhanced, n_proxied, mean_r2 = apply_ld_proxies(scoring, genotypes, ld_table)

    assert n_proxied == 1
    assert mean_r2 == pytest.approx(0.90)

    result = enhanced.collect()
    assert result.height == 1
    assert result["effect_weight"][0] == pytest.approx(2.0 * -0.95)
    assert result["chr_pos_norm"][0] == 1500


def test_apply_proxies_no_proxy_below_threshold() -> None:
    """Proxies with r² below min_r2 are rejected."""
    scoring = _make_scoring_lf([
        {"chr_name_norm": "1", "chr_pos_norm": 1000, "effect_allele": "G",
         "other_allele": "A", "effect_weight": 1.0},
    ])
    genotypes = _make_genotypes_lf([("1", 1500)])

    ld_table = _make_ld_table([{
        "target_chr": "1", "target_pos": 1000, "target_ref": "A", "target_alt": "G",
        "proxy_chr": "1", "proxy_pos": 1500, "proxy_rsid": "rs99",
        "proxy_ref": "C", "proxy_alt": "T",
        "r_squared": 0.50, "r_signed": 0.70,
    }])

    _, n_proxied, mean_r2 = apply_ld_proxies(scoring, genotypes, ld_table, min_r2=0.8)
    assert n_proxied == 0
    assert mean_r2 is None


def test_apply_proxies_typed_variants_unchanged() -> None:
    """Variants already typed on the array pass through without modification."""
    scoring = _make_scoring_lf([
        {"chr_name_norm": "1", "chr_pos_norm": 1000, "effect_allele": "G",
         "other_allele": "A", "effect_weight": 3.0},
        {"chr_name_norm": "1", "chr_pos_norm": 2000, "effect_allele": "T",
         "other_allele": "C", "effect_weight": 1.5},
    ])

    genotypes = _make_genotypes_lf([("1", 1000), ("1", 2000)])
    ld_table = _make_ld_table([])

    enhanced, n_proxied, mean_r2 = apply_ld_proxies(scoring, genotypes, ld_table)

    assert n_proxied == 0
    result = enhanced.collect().sort("chr_pos_norm")
    assert result["effect_weight"].to_list() == [3.0, 1.5]


def test_apply_proxies_mixed_typed_and_untyped() -> None:
    """Mix of typed, proxied, and unresolved variants."""
    scoring = _make_scoring_lf([
        {"chr_name_norm": "1", "chr_pos_norm": 1000, "effect_allele": "G",
         "other_allele": "A", "effect_weight": 1.0},
        {"chr_name_norm": "1", "chr_pos_norm": 2000, "effect_allele": "T",
         "other_allele": "C", "effect_weight": 2.0},
        {"chr_name_norm": "1", "chr_pos_norm": 3000, "effect_allele": "A",
         "other_allele": "G", "effect_weight": 3.0},
    ])

    genotypes = _make_genotypes_lf([("1", 1000), ("1", 2500)])

    ld_table = _make_ld_table([{
        "target_chr": "1", "target_pos": 2000, "target_ref": "C", "target_alt": "T",
        "proxy_chr": "1", "proxy_pos": 2500, "proxy_rsid": "rs_proxy",
        "proxy_ref": "A", "proxy_alt": "G",
        "r_squared": 0.85, "r_signed": 0.92,
    }])

    enhanced, n_proxied, mean_r2 = apply_ld_proxies(scoring, genotypes, ld_table)

    assert n_proxied == 1
    assert mean_r2 == pytest.approx(0.85)

    result = enhanced.collect().sort("chr_pos_norm")
    assert result.height == 3

    typed = result.filter(pl.col("chr_pos_norm") == 1000)
    assert typed["effect_weight"][0] == 1.0

    proxied = result.filter(pl.col("chr_pos_norm") == 2500)
    assert proxied["effect_weight"][0] == pytest.approx(2.0 * 0.92)

    unresolved = result.filter(pl.col("chr_pos_norm") == 3000)
    assert unresolved["effect_weight"][0] == 3.0


class TestPearsonROneVsMany:
    """Verify the vectorized Pearson r helper used by batched LD computation."""

    def test_perfect_correlation(self) -> None:
        target = np.array([0.0, 1.0, 2.0, 1.0, 0.0] * 20, dtype=np.float64)
        candidates = np.stack([target, -target, target * 2.0])
        r = _pearson_r_one_vs_many(target, candidates)
        assert r[0] == pytest.approx(1.0, abs=1e-10)
        assert r[1] == pytest.approx(-1.0, abs=1e-10)
        assert r[2] == pytest.approx(1.0, abs=1e-10)

    def test_uncorrelated(self) -> None:
        rng = np.random.default_rng(42)
        target = rng.choice([0, 1, 2], size=500).astype(np.float64)
        candidate = rng.choice([0, 1, 2], size=500).astype(np.float64)
        r = _pearson_r_one_vs_many(target, candidate.reshape(1, -1))
        assert abs(r[0]) < 0.15

    def test_nan_handling_fast_path(self) -> None:
        """No NaN at valid-target positions → fast matrix path."""
        target = np.array([0.0, 1.0, 2.0, 1.0] * 25, dtype=np.float64)
        cand = target * 0.5 + 0.5
        r = _pearson_r_one_vs_many(target, cand.reshape(1, -1))
        assert r[0] == pytest.approx(1.0, abs=1e-10)

    def test_nan_handling_slow_path(self) -> None:
        """NaN in candidate at target-valid positions → per-pair fallback."""
        target = np.array([0.0, 1.0, 2.0, 1.0, 0.0] * 20, dtype=np.float64)
        cand = target.copy()
        cand[0] = np.nan
        r = _pearson_r_one_vs_many(target, cand.reshape(1, -1))
        assert r[0] == pytest.approx(1.0, abs=1e-6)

    def test_too_few_valid_returns_zero(self) -> None:
        target = np.full(100, np.nan, dtype=np.float64)
        target[:30] = [0.0, 1.0, 2.0] * 10
        cand = np.ones(100, dtype=np.float64)
        r = _pearson_r_one_vs_many(target, cand.reshape(1, -1))
        assert r[0] == 0.0

    def test_zero_variance_returns_zero(self) -> None:
        target = np.array([1.0, 2.0, 0.0, 1.0] * 25, dtype=np.float64)
        cand = np.ones(100, dtype=np.float64)
        r = _pearson_r_one_vs_many(target, cand.reshape(1, -1))
        assert r[0] == 0.0

    def test_matches_numpy_corrcoef(self) -> None:
        """Cross-validate against numpy.corrcoef on random data."""
        rng = np.random.default_rng(123)
        n = 500
        target = rng.choice([0, 1, 2], size=n).astype(np.float64)
        candidates = rng.choice([0, 1, 2], size=(5, n)).astype(np.float64)
        r = _pearson_r_one_vs_many(target, candidates)
        for i in range(5):
            expected = np.corrcoef(target, candidates[i])[0, 1]
            assert r[i] == pytest.approx(expected, abs=1e-10)


def _proxy_row(t_chr: str, t_pos: int, p_pos: int, r2: float, r_signed: float) -> dict:
    return {
        "target_chr": t_chr, "target_pos": t_pos, "target_ref": "A", "target_alt": "G",
        "proxy_chr": t_chr, "proxy_pos": p_pos, "proxy_rsid": f"rs{p_pos}",
        "proxy_ref": "C", "proxy_alt": "T", "r_squared": r2, "r_signed": r_signed,
    }


def test_materialize_offchip_universe(tmp_path: Path) -> None:
    """Universe → off-chip parquet: build filter, chr-norm, dedup, anti-join typed."""
    from just_prs.ld_proxy import _materialize_offchip_universe

    universe = pl.DataFrame({
        "genome_build": ["GRCh38", "GRCh38", "GRCh38", "GRCh38", "GRCh37"],
        "chrom":        ["chr1",   "1",      "1",      "2",      "1"],
        "pos":          [100,      200,      200,      300,      999],
        "ref":          ["A",      "C",      "C",      "G",      "T"],
        "ref_source":   ["panel"] * 5,
    })
    uni_path = tmp_path / "reference_allele_universe.parquet"
    universe.write_parquet(uni_path)

    # Chip types (1, 100) — that position must be excluded as on-chip.
    typed_df = pl.DataFrame({"chr": ["1"], "pos": [100]})
    out_path = tmp_path / "offchip.parquet"

    n = _materialize_offchip_universe(uni_path, typed_df, "GRCh38", out_path)

    off = pl.read_parquet(out_path)
    assert set(off.columns) == {"chr_name_norm", "chr_pos_norm"}
    got = set(zip(off["chr_name_norm"].to_list(), off["chr_pos_norm"].to_list()))
    # GRCh37 row dropped; chr1:100 on-chip dropped; (1,200) deduped.
    assert got == {("1", 200), ("2", 300)}
    assert n == 2


def test_batch_result_skipped_accounting() -> None:
    """Capped (skipped) scores are counted separately and excluded from coverage."""
    outcomes = [
        LDProxyOutcome(pgs_id="PGS_A", status="ok", n_proxied=10),
        LDProxyOutcome(pgs_id="PGS_B", status="skipped", n_untyped=6_000_000, n_proxied=0),
        LDProxyOutcome(pgs_id="PGS_C", status="cached", n_proxied=5),
        LDProxyOutcome(pgs_id="PGS_D", status="failed", error="boom"),
    ]
    r = LDProxyBatchResult(
        panel="1000g", chip="gsa_v3", build="GRCh38",
        outcomes=outcomes, quality_df=pl.DataFrame(), output_dir=Path("/x"),
    )
    assert r.n_total == 4
    assert r.n_ok == 1
    assert r.n_cached == 1
    assert r.n_failed == 1
    assert r.n_skipped == 1
    # coverage excludes the skipped score from the denominator:
    # (ok + cached) / (total - skipped) = 2 / 3.
    assert r.coverage_ratio == pytest.approx(2 / 3)


def test_merge_ld_proxy_tables_lossless_dedup(tmp_path: Path) -> None:
    """Per-PGS intermediates merge into one position-deduplicated table, losslessly.

    Two PGS share target (1, 1000) with the byte-identical proxy row (proxy choice
    is position-deterministic), and each has one unique target. The merged table
    must have exactly one row per distinct (target_chr, target_pos), and equal the
    direct position-deduped union of all input rows.
    """
    cache = tmp_path / "cache"
    intermediates = ld_proxy_dir(cache, "gsa_v3", "GRCh38", "1000g")
    intermediates.mkdir(parents=True)

    shared = _proxy_row("1", 1000, 1500, 0.90, 0.95)
    _make_ld_table([shared, _proxy_row("1", 2000, 2400, 0.88, -0.80)]).write_parquet(
        intermediates / "PGS000001.parquet"
    )
    _make_ld_table([shared, _proxy_row("2", 3000, 3300, 0.82, 0.91)]).write_parquet(
        intermediates / "PGS000002.parquet"
    )
    # A _quality sidecar must be excluded from the merge.
    pl.DataFrame({"pgs_id": ["PGS000001"], "status": ["ok"]}).write_parquet(
        intermediates / "_quality.parquet"
    )

    out_path = ld_proxy_table_path(cache, "gsa_v3", "GRCh38", "1000g")
    result = merge_ld_proxy_tables(intermediates, out_path)

    assert result.path == out_path
    assert out_path.exists()
    assert result.n_sources == 2  # _quality.parquet excluded
    assert result.n_input_rows == 4  # 2 + 2 before dedup
    assert result.n_rows == 3  # (1,1000), (1,2000), (2,3000)
    assert result.n_targets == 3
    assert result.dedup_ratio == pytest.approx(0.25)

    merged = pl.read_parquet(out_path)
    validate_ld_proxy_table(merged)
    # One row per distinct target — equals the position-deduped union.
    assert merged.height == merged.select("target_chr", "target_pos").n_unique()
    expected_targets = {("1", 1000), ("1", 2000), ("2", 3000)}
    got_targets = set(zip(merged["target_chr"].to_list(), merged["target_pos"].to_list()))
    assert got_targets == expected_targets
    # The shared target kept its (identical) proxy row.
    shared_row = merged.filter((pl.col("target_chr") == "1") & (pl.col("target_pos") == 1000))
    assert shared_row["proxy_pos"][0] == 1500
    assert shared_row["r_signed"][0] == pytest.approx(0.95)


def test_merge_ld_proxy_tables_empty_dir(tmp_path: Path) -> None:
    """No intermediates → a schema-correct empty table is still written."""
    cache = tmp_path / "cache"
    intermediates = ld_proxy_dir(cache, "gsa_v3", "GRCh38", "1000g")
    intermediates.mkdir(parents=True)

    out_path = ld_proxy_table_path(cache, "gsa_v3", "GRCh38", "1000g")
    result = merge_ld_proxy_tables(intermediates, out_path)

    assert out_path.exists()
    assert result.n_rows == 0
    assert result.n_sources == 0
    assert result.mean_r2 is None
    empty = pl.read_parquet(out_path)
    assert empty.height == 0
    assert set(empty.columns) == set(LD_PROXY_TABLE_COLUMNS)


def test_merge_ld_proxy_tables_atomic_overwrite(tmp_path: Path) -> None:
    """Re-merging overwrites the prior table and leaves no .tmp residue."""
    cache = tmp_path / "cache"
    intermediates = ld_proxy_dir(cache, "gsa_v3", "GRCh38", "1000g")
    intermediates.mkdir(parents=True)
    _make_ld_table([_proxy_row("1", 1000, 1500, 0.90, 0.95)]).write_parquet(
        intermediates / "PGS000001.parquet"
    )
    out_path = ld_proxy_table_path(cache, "gsa_v3", "GRCh38", "1000g")

    merge_ld_proxy_tables(intermediates, out_path)
    _make_ld_table([_proxy_row("1", 2000, 2500, 0.85, 0.90)]).write_parquet(
        intermediates / "PGS000002.parquet"
    )
    result = merge_ld_proxy_tables(intermediates, out_path)

    assert result.n_rows == 2
    assert not out_path.with_name(out_path.name + ".tmp").exists()


def test_apply_proxies_from_parquet(tmp_path: Path) -> None:
    """LD table can be loaded from a parquet file."""
    ld_table = _make_ld_table([{
        "target_chr": "2", "target_pos": 5000, "target_ref": "A", "target_alt": "T",
        "proxy_chr": "2", "proxy_pos": 5100, "proxy_rsid": "rs_p",
        "proxy_ref": "G", "proxy_alt": "C",
        "r_squared": 0.95, "r_signed": 0.97,
    }])
    parquet_path = tmp_path / "ld_table.parquet"
    ld_table.write_parquet(parquet_path)

    scoring = _make_scoring_lf([
        {"chr_name_norm": "2", "chr_pos_norm": 5000, "effect_allele": "T",
         "other_allele": "A", "effect_weight": 1.0},
    ])
    genotypes = _make_genotypes_lf([("2", 5100)])

    enhanced, n_proxied, _ = apply_ld_proxies(scoring, genotypes, parquet_path)
    assert n_proxied == 1
    result = enhanced.collect()
    assert result["chr_pos_norm"][0] == 5100
