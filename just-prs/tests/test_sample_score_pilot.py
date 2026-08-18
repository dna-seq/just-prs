"""Real-data pilot for memory-safe public sample scoring.

Uses cached public genomes and scoring parquets. Skips when the warm cache
is absent so clean-clone CI stays cheap.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from just_prs.canary_audit import CanarySample, resolve_canary_vcf
from just_prs.sample_scores.checkpoints import part_path
from just_prs.sample_scores.engine import expected_checkpoint_keys, score_public_samples_pgs_major
from just_prs.sample_scores.pilot import select_pilot_pgs_ids
from just_prs.sample_scores.publish import PUBLIC_SAMPLE_SPECS
from just_prs.scoring import resolve_cache_dir


PUBLIC_LABELS = ("anton", "livia", "o-mom", "o-dad", "o-son1", "o-son2", "o-daughter")


def _available_public_samples(cache_dir: Path) -> list[CanarySample]:
    samples: list[CanarySample] = []
    for label in PUBLIC_LABELS:
        try:
            path = resolve_canary_vcf(label, cache_dir)
        except FileNotFoundError:
            continue
        if path.exists():
            samples.append(CanarySample(label=label, vcf_path=path, genome_build="GRCh38"))
    return samples


@pytest.fixture(scope="module")
def cache_dir() -> Path:
    return resolve_cache_dir()


def test_select_pilot_pgs_ids_covers_quantiles_and_failure(cache_dir: Path) -> None:
    scores = cache_dir / "scores"
    if not scores.exists():
        pytest.skip("Scoring cache is not present.")
    try:
        ids = select_pilot_pgs_ids(scores)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert "PGS999999" in ids
    assert len(ids) >= 4


def test_real_pilot_scores_available_public_genomes(cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = _available_public_samples(cache_dir)
    if len(samples) < 2:
        pytest.skip("Need at least two cached public genomes (anton/livia/o-family).")
    scores = cache_dir / "scores"
    try:
        pgs_ids = [pgs_id for pgs_id in select_pilot_pgs_ids(scores) if pgs_id != "PGS999999"]
        pgs_ids = pgs_ids[:3] + ["PGS999999"]
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    monkeypatch.setenv("PRS_SAMPLE_SCORE_INPROCESS", "1")
    monkeypatch.setenv("PRS_SAMPLE_SCORE_CHECKPOINT_SIZE", "2")
    # Isolate parts under the real cache but a dedicated subdir via PRS_CACHE_DIR
    # would lose scoring files. Score into the real cache; parts are resumable.
    progress = score_public_samples_pgs_major(
        samples,
        cache_dir,
        pgs_ids=pgs_ids,
        skip_existing=True,
        log=print,
    )
    assert progress.published_sample_ids
    assert set(progress.published_sample_ids) <= set(PUBLIC_SAMPLE_SPECS)
    from just_prs.sample_scores.checkpoints import discover_valid_parts

    expected = expected_checkpoint_keys(samples, cache_dir, pgs_ids=pgs_ids)
    found_metas = discover_valid_parts(cache_dir).valid
    found = {meta.checkpoint_key for meta in found_metas}
    assert expected <= found
    profiles = {meta.score_profile_id for meta in found_metas if meta.checkpoint_key in expected}
    from just_prs.sample_scores.models import RESTORED_PROFILE_ID, UNRESTORED_PROFILE_ID

    assert UNRESTORED_PROFILE_ID in profiles
    assert RESTORED_PROFILE_ID in profiles
    # The sentinel is unscorable by construction, so every sample must be recorded
    # as failed in BOTH profiles. Asserted from the parts rather than from this run's
    # counters, because `skip_existing` reports an already-failed phase as cached.
    sentinel_rows = pl.concat(
        [
            pl.read_parquet(
                part_path(cache_dir, meta.score_profile_id, meta.checkpoint_key),
                columns=["pgs_id", "sample_id", "score_profile_id", "status"],
            )
            for meta in found_metas
            if "PGS999999" in meta.pgs_ids
        ]
    ).filter(pl.col("pgs_id") == "PGS999999")
    assert set(sentinel_rows["status"].unique()) == {"failed"}
    for profile in (UNRESTORED_PROFILE_ID, RESTORED_PROFILE_ID):
        in_profile = sentinel_rows.filter(pl.col("score_profile_id") == profile)
        assert set(in_profile["sample_id"]) == set(progress.published_sample_ids)
    assert progress.n_ok + progress.n_failed + progress.n_cached > 0
    assert "stranger" not in progress.published_sample_ids
    from just_prs.memory import sample_score_memory_budget_bytes

    budget_mb = sample_score_memory_budget_bytes() / (1024 * 1024)
    if progress.peak_rss_mb:
        assert progress.peak_rss_mb < budget_mb

    # Truncated-part recovery on a real checkpoint. The victim must be a part this
    # pilot can actually rebuild — every PGS in it has to be inside `pgs_ids`, or the
    # rerun quarantines a production batch it was never asked to score and never
    # rewrites it. Parts are keyed on cache_dir alone, so the pilot shares the real
    # parts directory with the full catalog run.
    # The sentinel is excluded too: a batch whose only work is unscorable has nothing
    # to rewrite, so it would test the permanent-failure path, not part recovery.
    victim = next(
        meta
        for meta in found_metas
        if meta.checkpoint_key in expected
        and set(meta.pgs_ids) <= set(pgs_ids)
        and "PGS999999" not in meta.pgs_ids
    )
    parquet = part_path(cache_dir, victim.score_profile_id, victim.checkpoint_key)
    parquet.write_bytes(b"truncated")
    rediscovery = discover_valid_parts(cache_dir, validate_parquet=True)
    assert victim.checkpoint_key in rediscovery.quarantined
    progress2 = score_public_samples_pgs_major(
        samples,
        cache_dir,
        pgs_ids=pgs_ids,
        skip_existing=True,
    )
    found2 = {meta.checkpoint_key for meta in discover_valid_parts(cache_dir).valid}
    assert expected <= found2
    assert progress2.n_ok + progress2.n_failed + progress2.n_cached > 0
