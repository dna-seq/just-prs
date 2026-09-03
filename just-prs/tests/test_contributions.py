"""Per-variant PRS contributions: cutoff math plus DuckDB extract vs compute."""

from pathlib import Path

import polars as pl
import pytest

from just_prs.contributions import (
    _attach_identity_columns,
    _normalize_top_n,
    apply_contribution_cutoff,
    extract_prs_contributions,
    max_possible_contribution_expr,
)
from just_prs.prs import _normalize_scoring_columns, _resolve_scoring, compute_prs_duckdb


def test_max_possible_contribution_additive() -> None:
    df = pl.DataFrame({"effect_weight": [0.5, -1.0]})
    out = df.select(max_possible_contribution_expr(dosage_weight=False).alias("m"))
    assert out["m"].to_list() == [1.0, 2.0]


def test_max_possible_contribution_genoboost() -> None:
    df = pl.DataFrame(
        {
            "dosage_0_weight": [0.1, -0.4],
            "dosage_1_weight": [0.2, 0.1],
            "dosage_2_weight": [-0.05, 0.3],
        }
    )
    out = df.select(max_possible_contribution_expr(dosage_weight=True).alias("m"))
    assert out["m"].to_list() == [0.2, 0.4]


def test_contribution_cutoff_drops_below_percent_of_mean() -> None:
    scoring = pl.DataFrame(
        {
            "effect_weight": [0.01, 0.1, 1.0, 0.001],
            "chr_name_norm": ["1", "1", "1", "1"],
            "chr_pos_norm": [1, 2, 3, 4],
        }
    )
    kept, stats = apply_contribution_cutoff(scoring, 10.0, dosage_weight=False)

    expected_max = [0.02, 0.2, 2.0, 0.002]
    mean_max = sum(expected_max) / 4
    threshold = 0.10 * mean_max
    expected_kept = [w * 2 for w in scoring["effect_weight"].to_list() if w * 2 >= threshold]

    assert stats.cutoff_pct == 10.0
    assert stats.variants_total == 4
    assert stats.variants_kept == len(expected_kept)
    assert stats.variants_dropped == 4 - len(expected_kept)
    assert stats.threshold == pytest.approx(threshold)
    assert stats.mean_max_contribution == pytest.approx(mean_max)
    assert set(kept["effect_weight"].to_list()) == {0.1, 1.0}
    assert stats.mass_retained < 1.0
    assert stats.mass_kept == pytest.approx(sum(expected_kept))


def test_contribution_cutoff_none_or_zero_keeps_all() -> None:
    scoring = pl.DataFrame({"effect_weight": [0.01, 1.0]})
    for pct in (None, 0.0):
        kept, stats = apply_contribution_cutoff(scoring, pct, dosage_weight=False)
        assert stats.variants_dropped == 0
        assert stats.variants_kept == 2
        assert stats.mass_retained == 1.0
        assert kept.height == 2


def test_contribution_cutoff_rejects_over_100() -> None:
    scoring = pl.DataFrame({"effect_weight": [1.0]})
    with pytest.raises(ValueError, match="min_contribution_pct"):
        apply_contribution_cutoff(scoring, 101.0, dosage_weight=False)


def test_normalize_top_n_none_or_zero_is_off() -> None:
    assert _normalize_top_n(None) is None
    assert _normalize_top_n(0) is None
    assert _normalize_top_n(5) == 5


def test_normalize_top_n_rejects_negative() -> None:
    with pytest.raises(ValueError, match="top_n"):
        _normalize_top_n(-1)


def test_extract_matches_compute_prs_duckdb(
    vcf_path: Path, scoring_cache_dir: Path, tmp_path: Path
) -> None:
    computed = compute_prs_duckdb(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        genotype_input_mode="plink_present_only",
    )
    extracted = extract_prs_contributions(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        genotype_input_mode="plink_present_only",
        output=tmp_path / "contrib.parquet",
    )
    contrib = extracted.contributions.collect()

    assert extracted.pgs_id == "PGS000001"
    assert extracted.cutoff.variants_total == 77
    assert extracted.cutoff.variants_kept == contrib.height
    assert extracted.cutoff.variants_dropped == 77 - contrib.height
    assert extracted.score == pytest.approx(computed.score, abs=1e-10)
    assert contrib.height == computed.variants_observed
    assert contrib["contribution"].fill_null(0.0).sum() == pytest.approx(computed.score, abs=1e-10)
    assert {"chrom", "pos", "rsid", "effect_allele", "effect_weight", "dosage", "contribution", "status"}.issubset(
        set(contrib.columns)
    )
    shares = contrib["contribution_share"].drop_nulls()
    if shares.len() > 0:
        assert shares.sum() == pytest.approx(1.0, abs=1e-9)
    abs_contrib = contrib["contribution"].fill_null(0.0).abs().to_list()
    assert abs_contrib == sorted(abs_contrib, reverse=True)
    assert extracted.output_path.exists()


def test_extract_cutoff_drops_low_weight_variants(
    vcf_path: Path, scoring_cache_dir: Path, tmp_path: Path
) -> None:
    scoring_raw = _resolve_scoring("PGS000001", "GRCh38", scoring_cache_dir)
    scoring = _attach_identity_columns(
        scoring_raw,
        _normalize_scoring_columns(scoring_raw),
    )
    maxes = scoring.select((pl.col("effect_weight").abs() * 2.0).alias("m"))
    mean_m = float(maxes["m"].mean() or 0.0)
    min_m = float(maxes["m"].min() or 0.0)
    assert mean_m > 0 and min_m < mean_m
    pct = (min_m / mean_m) * 100.0 + 1.0

    full = extract_prs_contributions(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        genotype_input_mode="plink_present_only",
        output=tmp_path / "full.parquet",
    )
    cut = extract_prs_contributions(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        genotype_input_mode="plink_present_only",
        min_contribution_pct=pct,
        output=tmp_path / "cut.parquet",
    )
    cut_df = cut.contributions.collect()

    assert cut.cutoff.variants_dropped > 0
    assert cut.cutoff.variants_kept < full.cutoff.variants_kept
    assert cut.cutoff.mass_retained < 1.0
    assert cut_df.height < full.contributions.collect().height
    assert cut.score != pytest.approx(full.score, abs=1e-10)
    assert cut_df["contribution"].fill_null(0.0).sum() == pytest.approx(cut.score, abs=1e-10)


def test_extract_top_n_does_not_change_compute(
    vcf_path: Path, scoring_cache_dir: Path, tmp_path: Path
) -> None:
    computed = compute_prs_duckdb(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        genotype_input_mode="plink_present_only",
    )
    full = extract_prs_contributions(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        genotype_input_mode="plink_present_only",
        output=tmp_path / "full.parquet",
    )
    extracted = extract_prs_contributions(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        genotype_input_mode="plink_present_only",
        top_n=5,
        output=tmp_path / "top5.parquet",
    )
    full_df = full.contributions.collect()
    top_df = extracted.contributions.collect()
    expected = full_df.sort(pl.col("contribution").abs(), descending=True).head(5)
    assert extracted.cutoff.top_n == 5
    assert extracted.cutoff.variants_kept == 5
    assert extracted.cutoff.variants_total == computed.variants_total
    assert top_df.height == 5
    assert top_df["chrom"].to_list() == expected["chrom"].to_list()
    assert top_df["pos"].to_list() == expected["pos"].to_list()
    assert top_df["contribution"].to_list() == pytest.approx(expected["contribution"].to_list())
    assert extracted.score == pytest.approx(
        expected["contribution"].fill_null(0.0).sum(), abs=1e-10
    )
    assert computed.variants_total > 5
    assert computed.score == pytest.approx(full.score, abs=1e-10)


def test_extract_chunked_matches_unchunked(
    vcf_path: Path,
    scoring_cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unchunked = extract_prs_contributions(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        genotype_input_mode="plink_present_only",
        output=tmp_path / "unchunked.parquet",
    )
    monkeypatch.setenv("PRS_SCORING_JOIN_CHUNK_SIZE", "10")
    chunked = extract_prs_contributions(
        vcf_path=vcf_path,
        scoring_file="PGS000001",
        genome_build="GRCh38",
        cache_dir=scoring_cache_dir,
        pgs_id="PGS000001",
        genotype_input_mode="plink_present_only",
        output=tmp_path / "chunked.parquet",
    )
    assert chunked.score == pytest.approx(unchunked.score, abs=1e-10)
    left = unchunked.contributions.collect().sort(["chrom", "pos", "effect_allele"])
    right = chunked.contributions.collect().sort(["chrom", "pos", "effect_allele"])
    assert left["contribution"].to_list() == pytest.approx(right["contribution"].to_list())
