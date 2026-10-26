"""PLINK2 --score input built from real harmonized scoring files.

PGS000012 has no ``other_allele``: 9,819 rows carry several
``hm_inferOtherAllele`` candidates, and the sample's ALT may be any of them.
PGS000013 has GRCh37 variants that lift over to one GRCh38 variant; PLINK2
aborts on a repeated (variant, allele) pair. Both broke the engine benchmark.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from just_prs.reference import _prepare_plink2_score_input
from just_prs.scoring import ensure_scoring_file, parse_scoring_file, resolve_cache_dir


def _scoring(pgs_id: str) -> Path:
    try:
        return ensure_scoring_file(pgs_id, resolve_cache_dir() / "scores", genome_build="GRCh38")
    except Exception as exc:  # offline and not cached
        pytest.skip(f"{pgs_id} unavailable: {exc}")


def test_every_inferred_other_allele_candidate_is_listed(tmp_path: Path) -> None:
    path = _scoring("PGS000012")
    raw = parse_scoring_file(path).collect()
    expected = (
        raw.filter(pl.col("hm_inferOtherAllele").is_not_null())
        .with_columns(pl.col("hm_inferOtherAllele").str.split("/").alias("o"))
        .explode("o", empty_as_null=True)
        .filter(pl.col("o") != pl.col("effect_allele"))
        .select(
            (
                pl.col("hm_chr").cast(pl.Utf8) + ":" + pl.col("hm_pos").cast(pl.Utf8)
                + ":" + pl.col("o") + ":" + pl.col("effect_allele")
            ).alias("variant_id")
        )
    )
    assert raw.filter(pl.col("hm_inferOtherAllele").str.contains("/")).height > 0

    score_input = pl.read_csv(
        _prepare_plink2_score_input(path, "PGS000012", tmp_path), separator="\t"
    )
    assert set(expected["variant_id"]) <= set(score_input["variant_id"])


def test_liftover_duplicate_pairs_are_summed_not_repeated(tmp_path: Path) -> None:
    path = _scoring("PGS000013")
    raw = parse_scoring_file(path).collect()
    score_input = pl.read_csv(
        _prepare_plink2_score_input(path, "PGS000013", tmp_path), separator="\t"
    )
    assert score_input.select(pl.struct("variant_id", "effect_allele").is_duplicated().any()).item() is False

    # Each scoring row's weight lands on its forward-order line exactly once, so
    # the forward lines carry the file's total weight.
    other = pl.col("other_allele").fill_null(pl.col("hm_inferOtherAllele"))
    forward_pairs = (
        raw.filter(other.is_not_null() & pl.col("hm_chr").is_not_null())
        .select(
            (
                pl.col("hm_chr").cast(pl.Utf8) + ":" + pl.col("hm_pos").cast(pl.Utf8)
                + ":" + other + ":" + pl.col("effect_allele")
            ).alias("variant_id"),
            pl.col("effect_allele").cast(pl.Utf8),
        )
        .unique()
    )
    forward = score_input.join(forward_pairs, on=["variant_id", "effect_allele"], how="semi")
    expected_total = raw.filter(other.is_not_null() & pl.col("hm_chr").is_not_null())["effect_weight"].sum()
    assert forward["effect_weight"].sum() == pytest.approx(expected_total, rel=1e-9)
