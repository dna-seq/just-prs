"""Atomic checkpoint parts, quarantine, resume, and compaction."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from just_prs.sample_scores.checkpoints import (
    CheckpointMeta,
    compact_runtime_parts,
    completed_pgs_for_profile,
    discover_valid_parts,
    make_checkpoint_key,
    reopen_failed_parts,
    write_runtime_part,
)
from just_prs.sample_scores.completeness import CompletenessError, validate_runtime_results
from just_prs.sample_scores.models import RESTORED_PROFILE_ID, UNRESTORED_PROFILE_ID
from just_prs.sample_scores.store import write_samples
from just_prs.sample_scores.models import SampleRecord
from just_prs.sample_scores.publish import RuntimeResultRow


def _sample() -> SampleRecord:
    return SampleRecord(
        sample_id="anton",
        aliases=["Anton"],
        display_name="Anton",
        license="CC0",
        publication_allowed=True,
        consent_basis="public-domain",
        source_sha256="a" * 64,
        genotype_sha256_v1="b" * 64,
    )


def _runtime_row(sample: SampleRecord, pgs_id: str = "PGS000001") -> RuntimeResultRow:
    return RuntimeResultRow(
        sample_id=sample.sample_id,
        pgs_id=pgs_id,
        scoring_build="GRCh38",
        score_profile_id=UNRESTORED_PROFILE_ID,
        scoring_fingerprint="fp1",
        score=0.25,
        variants_matched=10,
        variants_total=12,
        match_rate=10 / 12,
        sample_genotype_sha256=sample.genotype_sha256_v1,
        trait_reported="body mass index",
    )


def _meta(pgs_ids: list[str], n_rows: int, key: str | None = None) -> CheckpointMeta:
    return CheckpointMeta(
        checkpoint_key=key or make_checkpoint_key(
            score_profile_id=UNRESTORED_PROFILE_ID,
            score_algorithm_version="1",
            genome_build="GRCh38",
            pgs_ids=pgs_ids,
            scoring_set_fp="s" * 64,
            sample_set_fp="g" * 64,
            reference_universe_fp=None,
        ),
        score_profile_id=UNRESTORED_PROFILE_ID,
        pgs_ids=pgs_ids,
        scoring_set_fingerprint="s" * 64,
        sample_set_genotype_fingerprint="g" * 64,
        n_rows=n_rows,
        n_ok=n_rows,
    )


def test_atomic_part_and_lazy_resume(tmp_path: Path) -> None:
    sample = _sample()
    meta = _meta(["PGS000001"], 1)
    path = write_runtime_part([_runtime_row(sample)], tmp_path, meta)
    assert path.exists()
    assert Path(str(path) + ".meta.json").exists()
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == {"PGS000001"}


def test_truncated_part_is_quarantined_and_recomputed(tmp_path: Path) -> None:
    sample = _sample()
    meta = _meta(["PGS000001"], 1)
    path = write_runtime_part([_runtime_row(sample)], tmp_path, meta)
    path.write_bytes(b"not-a-parquet")
    discovery = discover_valid_parts(
        tmp_path, [UNRESTORED_PROFILE_ID], validate_parquet=True
    )
    assert meta.checkpoint_key in discovery.quarantined
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == set()
    write_runtime_part([_runtime_row(sample)], tmp_path, meta)
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == {"PGS000001"}


def test_compaction_is_deterministic(tmp_path: Path) -> None:
    sample = _sample()
    write_runtime_part([_runtime_row(sample, "PGS000001")], tmp_path, _meta(["PGS000001"], 1, key="a_n1"))
    write_runtime_part([_runtime_row(sample, "PGS000002")], tmp_path, _meta(["PGS000002"], 1, key="b_n1"))
    dest_a = compact_runtime_parts(tmp_path)
    rows_a = pl.read_parquet(dest_a).sort("pgs_id")
    dest_b = compact_runtime_parts(tmp_path)
    rows_b = pl.read_parquet(dest_b).sort("pgs_id")
    assert rows_a.height == 2
    assert rows_a["pgs_id"].to_list() == rows_b["pgs_id"].to_list()
    assert rows_a["score"].to_list() == rows_b["score"].to_list()


def test_compaction_drops_withdrawn_catalog_ids(tmp_path: Path) -> None:
    """Old unrestored parts for retired PGS IDs must not fail the live matrix."""
    sample = _sample()
    write_samples([sample], tmp_path)
    write_runtime_part(
        [_runtime_row(sample, "PGS000001")],
        tmp_path,
        _meta(["PGS000001"], 1, key="live_unrestored"),
    )
    write_runtime_part(
        [_runtime_row(sample, "PGS005388")],
        tmp_path,
        _meta(["PGS005388"], 1, key="withdrawn_unrestored"),
    )
    restored = _runtime_row(sample, "PGS000001").model_copy(
        update={"score_profile_id": RESTORED_PROFILE_ID}
    )
    write_runtime_part(
        [restored],
        tmp_path,
        CheckpointMeta(
            checkpoint_key="live_restored",
            score_profile_id=RESTORED_PROFILE_ID,
            pgs_ids=["PGS000001"],
            scoring_set_fingerprint="s" * 64,
            sample_set_genotype_fingerprint="g" * 64,
            n_rows=1,
            n_ok=1,
        ),
    )
    dest = compact_runtime_parts(tmp_path, expected_pgs_ids=["PGS000001"])
    assert set(pl.read_parquet(dest)["pgs_id"].to_list()) == {"PGS000001"}
    report = validate_runtime_results(
        tmp_path,
        expected_sample_ids=["anton"],
        expected_pgs_ids=["PGS000001"],
    )
    assert report.passed is True
    assert report.n_rows == 2


def test_completeness_blocks_partial_matrix(tmp_path: Path) -> None:
    sample = _sample()
    write_samples([sample], tmp_path)
    write_runtime_part([_runtime_row(sample)], tmp_path, _meta(["PGS000001"], 1, key="only"))
    compact_runtime_parts(tmp_path)
    report = validate_runtime_results(
        tmp_path,
        expected_sample_ids=["anton"],
        expected_pgs_ids=["PGS000001"],
    )
    assert report.passed is False
    assert any("missing" in issue for issue in report.issues)
    with pytest.raises(CompletenessError):
        report.raise_if_failed()


def test_reopen_failed_parts_keeps_ok_and_retries_failed(tmp_path: Path) -> None:
    sample = _sample()
    ok = _runtime_row(sample, "PGS000001")
    failed = _runtime_row(sample, "PGS000002").model_copy(
        update={"status": "failed", "score": None, "error": "worker_sigsegv"}
    )
    mixed = _meta(["PGS000001", "PGS000002"], 2, key="mixed")
    mixed.n_ok = 1
    mixed.n_failed = 1
    write_runtime_part([ok, failed], tmp_path, mixed)

    crash = _runtime_row(sample, "PGS000003").model_copy(
        update={"status": "failed", "score": None, "error": "native crash"}
    )
    crash_meta = _meta(["PGS000003"], 1, key="crash")
    crash_meta.n_ok = 0
    crash_meta.n_failed = 1
    write_runtime_part([crash], tmp_path, crash_meta)

    later = _runtime_row(sample, "PGS000004")
    write_runtime_part([later], tmp_path, _meta(["PGS000004"], 1, key="ok"))

    report = reopen_failed_parts(tmp_path, UNRESTORED_PROFILE_ID)
    assert report.pgs_ids == ["PGS000002", "PGS000003"]
    assert report.n_parts_rewritten == 1
    assert report.n_parts_quarantined == 1
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == {
        "PGS000001",
        "PGS000004",
    }
