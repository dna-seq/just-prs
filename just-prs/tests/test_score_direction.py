"""Consuming the published score-direction curation (AGENTS.md "Consuming curated directions").

Every test reads the real tables from ``just-dna-seq/pgs-catalog`` ``data/metadata/`` through
``PRSCatalog``; expected values are recomputed from those tables, never hardcoded.
"""

import os
import random
import shutil
import statistics
import time
from pathlib import Path

import polars as pl
import pytest

from just_prs import PRSCatalog
from just_prs.enrich import _risk_level_from_percentile
from just_prs.prs_catalog import CURATION_DIR_ENV, CURATION_MAX_AGE_ENV
from just_prs.score_direction import (
    SCORE_DIRECTION_COLUMNS,
    CurationStatus,
    axis_percentile,
    direction_tone,
    higher_score_means,
    legacy_percentile_tone,
)
from just_prs.trait_search import matching_pgs_ids, score_search_terms
from just_prs.trait_summary import axis_aligned_rows, summarize_trait_rows

CACHE_DIR = PRSCatalog().cache_dir


@pytest.fixture(scope="module")
def catalog() -> PRSCatalog:
    return PRSCatalog(cache_dir=CACHE_DIR)


@pytest.fixture(scope="module")
def published(catalog: PRSCatalog) -> dict[str, pl.DataFrame]:
    tables = catalog.curation_tables()
    assert {"annotations", "review", "clusters", "search_terms"} <= set(tables)
    return tables


def test_score_directions_partition_the_published_tables(
    catalog: PRSCatalog, published: dict[str, pl.DataFrame]
) -> None:
    directions = catalog.score_directions()
    assert directions["pgs_id"].is_unique().all()
    by_status = {
        status: set(directions.filter(pl.col("curation_status") == status)["pgs_id"])
        for status in (s.value for s in CurationStatus)
    }
    annotations, review = published["annotations"], published["review"]
    assert by_status["resolved"] == set(annotations["pgs_id"])
    assert by_status["in_review"] == set(review.filter(pl.col("direction").is_not_null())["pgs_id"])
    assert by_status["no_direction"] == set(review.filter(pl.col("direction").is_null())["pgs_id"])
    # The review queue's reason is carried, the resolved rows have none.
    reasons = directions.join(review.select("pgs_id", "review_reason"), on="pgs_id", how="inner")
    assert (reasons["curation_review_reason"] == reasons["review_reason"]).all()
    assert directions.filter(pl.col("curation_status") == "resolved")["curation_review_reason"].null_count() == len(
        by_status["resolved"]
    )


def test_resolved_polarity_is_phenotype_sign_times_direction(catalog: PRSCatalog) -> None:
    resolved = catalog.score_directions().filter(pl.col("curation_status") == "resolved")
    expected = resolved["phenotype_sign"] * pl.Series(
        [1 if d == "increases" else -1 for d in resolved["direction"]], dtype=pl.Int8
    )
    assert (resolved["polarity"] == expected).all()


def test_scores_and_score_info_row_carry_the_direction(catalog: PRSCatalog) -> None:
    scores = catalog.scores(include_excluded=True).collect()
    assert set(SCORE_DIRECTION_COLUMNS) <= set(scores.columns)
    directions = catalog.score_directions()
    joined = scores.select("pgs_id", *SCORE_DIRECTION_COLUMNS).join(
        directions, on="pgs_id", how="inner", suffix="_direct"
    )
    # Every curated ID that is in the catalog got its row, column for column.
    assert set(joined["pgs_id"]) == set(directions["pgs_id"]) & set(scores["pgs_id"])
    for column in SCORE_DIRECTION_COLUMNS:
        assert joined[column].equals(joined[f"{column}_direct"]), column
    uncurated = scores.filter(~pl.col("pgs_id").is_in(directions["pgs_id"].to_list()))
    assert uncurated["curation_status"].null_count() == uncurated.height

    rng = random.Random(11)
    sample = rng.sample(sorted(joined["pgs_id"]), 5) + rng.sample(sorted(uncurated["pgs_id"]), 2)
    for pgs_id in sample:
        info = catalog.score_info_row(pgs_id)
        assert info is not None
        direct = catalog.score_direction(pgs_id) or {}
        assert {c: info[c] for c in SCORE_DIRECTION_COLUMNS} == {c: direct.get(c) for c in SCORE_DIRECTION_COLUMNS}


def test_curation_dir_override_and_empty_curation(
    catalog: PRSCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_build = tmp_path / "build"
    shutil.copytree(catalog.curation_dir, local_build, ignore=shutil.ignore_patterns(".*"))
    monkeypatch.setenv(CURATION_DIR_ENV, str(local_build))
    local = PRSCatalog(cache_dir=CACHE_DIR)
    assert local.curation_dir == local_build
    assert local.score_directions().equals(catalog.score_directions())
    assert not (local_build / ".pulled_at").exists(), "a local build must never be overwritten by a pull"

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(CURATION_DIR_ENV, str(empty))
    bare = PRSCatalog(cache_dir=CACHE_DIR)
    assert bare.score_directions().height == 0
    scores = bare.scores().collect()
    assert set(SCORE_DIRECTION_COLUMNS) <= set(scores.columns)
    assert scores["curation_status"].null_count() == scores.height


def test_curation_pull_honours_max_age(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CURATION_DIR_ENV, raising=False)
    monkeypatch.delenv(CURATION_MAX_AGE_ENV, raising=False)
    first = PRSCatalog(cache_dir=tmp_path)
    directory = first.refresh_curation()
    assert directory == tmp_path / "metadata" / "curation"
    stamp = directory / ".pulled_at"
    pulled_at = stamp.stat().st_mtime
    assert not (directory / "data").exists(), "pulls must land flat (cache invariant 2)"

    time.sleep(0.05)
    PRSCatalog(cache_dir=tmp_path).refresh_curation()
    assert stamp.stat().st_mtime == pulled_at, "a fresh copy is not re-pulled"

    # A long-running process re-checks once the max age (24 h) has passed since its last check.
    day_ago = time.time() - 25 * 3600
    os.utime(stamp, (day_ago, day_ago))
    first.refresh_curation()
    assert stamp.stat().st_mtime == day_ago, "checked less than a max age ago: no re-check"
    first._curation_checked_at = day_ago
    first.refresh_curation()
    assert stamp.stat().st_mtime > day_ago + 3600
    pulled_at = stamp.stat().st_mtime

    monkeypatch.setenv(CURATION_MAX_AGE_ENV, "0")
    again = PRSCatalog(cache_dir=tmp_path)
    again.refresh_curation()
    assert stamp.stat().st_mtime > pulled_at
    once = stamp.stat().st_mtime
    again.refresh_curation()
    assert stamp.stat().st_mtime == once, "max age 0 re-pulls once per catalog instance"
    assert again.score_directions().height == first.score_directions().height


def test_direction_tone_follows_valence_and_polarity(catalog: PRSCatalog) -> None:
    directions = catalog.score_directions()
    checked = 0
    for row in directions.iter_rows(named=True):
        status, valence, polarity = row["curation_status"], row["valence"], row["polarity"]
        high = direction_tone(97.0, row)
        if status != "resolved":
            assert high == "neutral"
        elif valence in ("neutral", "context_dependent"):
            assert high == "neutral"
        else:
            unwanted_up = (valence == "undesirable") == (polarity == 1)
            assert high == ("danger" if unwanted_up else "good")
            assert direction_tone(3.0, row) == ("good" if unwanted_up else "danger")
            checked += 1
        assert higher_score_means(row)
    assert checked > 0
    assert direction_tone(97.0, {}) == legacy_percentile_tone(97.0) == "danger"


def test_risk_level_wording_follows_the_curated_direction(catalog: PRSCatalog) -> None:
    resolved = catalog.score_directions().filter(
        (pl.col("curation_status") == "resolved") & pl.col("target").is_not_null()
    )
    for direction in ("increases", "decreases"):
        row = resolved.filter(pl.col("direction") == direction).row(0, named=True)
        label, colour = _risk_level_from_percentile(95.0, row)
        more = "more" if direction == "increases" else "less"
        assert label == f"High score: {more} {row['target']}"
        assert colour == {"danger": "red", "warning": "orange", "good": "green", "neutral": "gray"}[
            direction_tone(95.0, row)
        ]


def _mixed_cluster(directions: pl.DataFrame) -> str:
    resolved = directions.filter(pl.col("curation_status") == "resolved")
    counts = resolved.group_by("cluster").agg(
        (pl.col("polarity") == -1).sum().alias("down"), (pl.col("polarity") == 1).sum().alias("up")
    )
    mixed = counts.filter((pl.col("down") > 0) & (pl.col("up") > 0)).sort("down", "cluster", descending=[True, False])
    if mixed.height == 0:
        pytest.skip("the published curation has no concept with both polarities yet")
    return str(mixed["cluster"][0])


def test_trait_summary_aggregates_on_the_curated_axis(catalog: PRSCatalog) -> None:
    directions = catalog.score_directions()
    cluster = _mixed_cluster(directions)
    members = directions.filter(pl.col("cluster") == cluster).to_dicts()
    other = directions.filter(
        (pl.col("curation_status") == "resolved") & (pl.col("cluster") != cluster)
    ).row(0, named=True)
    uncurated = {"pgs_id": "PGS_UNCURATED"}
    rng = random.Random(3)
    rows = [
        {**row, "percentile": round(rng.uniform(1, 99), 1), "match_rate": 90.0, "quality_label": "High"}
        for row in [*members, other, uncurated]
    ]
    stats = summarize_trait_rows(rows, model_scope="usable")

    counted = [r for r in rows if r.get("curation_status") == "resolved" and r.get("cluster") == cluster]
    expected_axis = {r["pgs_id"]: axis_percentile(r["percentile"], r["polarity"]) for r in counted}
    assert stats.axis_aligned and stats.curated_cluster == cluster
    assert stats.pct_by_id == expected_axis
    assert stats.raw_pct_by_id == {r["pgs_id"]: float(r["percentile"]) for r in counted}
    assert stats.median_pct == pytest.approx(statistics.median(expected_axis.values()))
    assert set(stats.flipped_ids) == {r["pgs_id"] for r in counted if r["polarity"] == -1}
    assert {item["pgs_id"] for item in stats.not_counted} == {r["pgs_id"] for r in rows} - set(expected_axis)
    assert stats.n_total == len(rows) and stats.n_counted == len(counted)
    if stats.risk_phenotype:
        assert stats.risk_phenotype in {r["phenotype"] for r in counted}

    legacy = summarize_trait_rows(rows, model_scope="usable", curated_axis=False)
    assert not legacy.axis_aligned
    assert legacy.median_pct == pytest.approx(statistics.median(r["percentile"] for r in rows))

    # Charts and reports pre-align rows; the summary never flips them twice.
    aligned = summarize_trait_rows(axis_aligned_rows(rows), model_scope="usable")
    assert aligned.pct_by_id == stats.pct_by_id
    assert aligned.raw_pct_by_id == pytest.approx(stats.raw_pct_by_id)


def test_group_without_resolved_rows_keeps_face_value(catalog: PRSCatalog) -> None:
    review = catalog.score_directions().filter(pl.col("curation_status") != "resolved").head(4).to_dicts()
    rows = [{**row, "percentile": 10.0 * (i + 2), "match_rate": 90.0} for i, row in enumerate(review)]
    stats = summarize_trait_rows(rows)
    assert not stats.axis_aligned
    assert stats.not_counted == []
    assert stats.median_pct == pytest.approx(statistics.median(r["percentile"] for r in rows))
    assert axis_aligned_rows(rows) == rows


def test_search_finds_scores_by_curated_alias_only(
    catalog: PRSCatalog, published: dict[str, pl.DataFrame]
) -> None:
    scores = catalog.scores(include_excluded=True).collect()
    catalog_only = score_search_terms(scores)
    aliases = published["search_terms"].filter(pl.col("term_kind") == "alias").sort("cluster", "term")
    for alias_row in aliases.iter_rows(named=True):
        if matching_pgs_ids(catalog_only, alias_row["term"]):
            continue  # also a catalog name: does not prove the concept terms are used
        members = set(
            catalog.score_directions().filter(pl.col("cluster") == alias_row["cluster"])["pgs_id"]
        ) & set(scores["pgs_id"])
        if not members:
            continue
        found = catalog.search_pgs_ids(alias_row["term"])
        assert members <= found
        visible = set(catalog.search(alias_row["term"]).collect()["pgs_id"])
        assert found - set(catalog.excluded_catalog_pgs_ids()) <= visible
        return
    pytest.skip("no curated alias that is absent from catalog names yet")


def test_refresh_does_not_touch_cleaned_metadata(catalog: PRSCatalog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CURATION_DIR_ENV, raising=False)
    before = sorted(p.name for p in catalog.metadata_dir.glob("*.parquet"))
    catalog.refresh_curation(force=True)
    assert sorted(p.name for p in catalog.metadata_dir.glob("*.parquet")) == before


def _anton_parquet() -> Path:
    from just_prs.normalize import VcfFilterConfig, normalize_vcf
    from just_prs.scoring import resolve_cache_dir

    test_dir = resolve_cache_dir() / "test-data"
    vcf = test_dir / "antonkulaga.vcf"
    if not vcf.exists():
        pytest.skip(f"anton VCF not cached at {vcf}")
    out = test_dir / "antonkulaga_pass_normalized.parquet"
    if not out.exists() or out.stat().st_mtime < vcf.stat().st_mtime:
        normalize_vcf(vcf, out, config=VcfFilterConfig(pass_filters=["PASS", "."]))
    return out


def test_scored_genome_rows_carry_direction_and_align(catalog: PRSCatalog) -> None:
    """Score one up- and one down-pointing model of one concept on Anton's genome, one at a time."""
    from just_prs.enrich import enrich_prs_result

    directions = catalog.score_directions()
    cluster = _mixed_cluster(directions)
    candidates = (
        catalog.scores(genome_build="GRCh38")
        .filter((pl.col("cluster") == cluster) & (pl.col("curation_status") == "resolved"))
        .select("pgs_id", "polarity", "n_variants")
        .collect()
        .sort("n_variants", "pgs_id")
    )
    picked = [candidates.filter(pl.col("polarity") == sign).row(0, named=True)["pgs_id"] for sign in (1, -1)]
    genotypes = pl.scan_parquet(_anton_parquet())
    best_perf = catalog.best_performance().collect()
    rows: list[dict] = []
    for pgs_id in picked:  # sequential on purpose: never score genomes in parallel
        result = catalog.compute_prs(_anton_parquet(), pgs_id, genome_build="GRCh38", genotypes_lf=genotypes)
        enriched = enrich_prs_result(result, catalog, best_perf, genome_build="GRCh38")
        expected = catalog.score_direction(pgs_id) or {}
        assert {c: getattr(enriched, c) for c in SCORE_DIRECTION_COLUMNS} == {
            c: expected.get(c) for c in SCORE_DIRECTION_COLUMNS
        }
        if enriched.percentile is not None:
            tone = direction_tone(enriched.percentile, expected)
            assert enriched.risk_level_color == {
                "danger": "red", "warning": "orange", "good": "green", "neutral": "gray"
            }[tone]
            assert str(expected["higher_percentile_means"]) in enriched.risk_hint
        rows.append({**enriched.model_dump(), "match_rate": 90.0})

    with_pct = [r for r in rows if r["percentile"] is not None]
    if len(with_pct) < 2:
        pytest.skip("reference percentiles missing for the picked scores")
    stats = summarize_trait_rows(rows, model_scope="all")
    assert stats.axis_aligned and stats.curated_cluster == cluster
    expected_axis = {r["pgs_id"]: axis_percentile(r["percentile"], r["polarity"]) for r in with_pct}
    assert stats.pct_by_id == pytest.approx(expected_axis)
    assert stats.flipped_ids == [picked[1]]
