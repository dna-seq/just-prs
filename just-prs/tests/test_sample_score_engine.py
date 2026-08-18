"""PGS-major checkpoint planning and in-process worker recycle."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from just_prs.sample_scores.engine import (
    CheckpointWork,
    PreparedSample,
    WorkerWorkOrder,
    _should_log_progress,
    format_sample_score_progress,
    pack_checkpoint_id_batches,
    plan_checkpoints,
    run_phase,
    run_worker,
)
from just_prs.sample_scores.models import (
    SCORE_PROFILES,
    UNRESTORED_PROFILE_ID,
    SampleRecord,
)
from just_prs.sample_scores.checkpoints import CheckpointMeta, completed_pgs_for_profile


def _record(sample_id: str = "anton") -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        aliases=[],
        display_name=sample_id,
        license="CC0",
        publication_allowed=True,
        consent_basis="test",
        source_sha256="a" * 64,
        genotype_sha256_v1="b" * 64,
    )


def test_missing_scoring_file_gets_sentinel_fingerprint(tmp_path: Path) -> None:
    from just_prs.sample_scores.engine import _scoring_fingerprint_for

    digest = _scoring_fingerprint_for("PGS999999", "GRCh38", tmp_path / "scores", {})
    assert len(digest) == 64
    again = _scoring_fingerprint_for("PGS999999", "GRCh38", tmp_path / "scores", {})
    assert digest == again


def test_plan_checkpoints_is_pgs_major_not_sample_major(tmp_path: Path) -> None:
    scores = tmp_path / "scores"
    scores.mkdir()
    for pgs_id, weight in (("PGS000001", 0.1), ("PGS000002", 0.2), ("PGS000003", 0.3)):
        pl.DataFrame({
            "hm_chr": ["1"],
            "hm_pos": [10],
            "effect_allele": ["A"],
            "effect_weight": [weight],
            "other_allele": ["G"],
        }).write_parquet(scores / f"{pgs_id}_hmPOS_GRCh38.parquet")
    prepared = [
        PreparedSample(record=_record("anton"), parquet_path="a.parquet", vcf_path="a.vcf"),
        PreparedSample(record=_record("livia"), parquet_path="l.parquet", vcf_path="l.vcf"),
    ]
    planned = plan_checkpoints(
        ["PGS000001", "PGS000002", "PGS000003"],
        prepared,
        SCORE_PROFILES[UNRESTORED_PROFILE_ID],
        scores,
        {},
        universe_fp=None,
        batch_size=2,
    )
    assert len(planned) == 2
    assert planned[0].meta.pgs_ids == ["PGS000001", "PGS000002"]
    assert planned[0].meta.n_rows == 4
    assert planned[1].meta.pgs_ids == ["PGS000003"]
    assert planned[1].meta.n_rows == 2


def _write_scoring(scores: Path, pgs_id: str, n_rows: int) -> None:
    pl.DataFrame({
        "hm_chr": ["1"] * n_rows,
        "hm_pos": list(range(1, n_rows + 1)),
        "effect_allele": ["A"] * n_rows,
        "effect_weight": [0.1] * n_rows,
        "other_allele": ["G"] * n_rows,
    }).write_parquet(scores / f"{pgs_id}_hmPOS_GRCh38.parquet")


def test_pack_checkpoint_id_batches_keeps_smalls_and_splits_large() -> None:
    packed = pack_checkpoint_id_batches(
        ["PGS000001", "PGS000002", "PGS000003", "PGS000004", "PGS000005"],
        batch_size=2,
        large_ids={"PGS000003", "PGS000005"},
    )
    assert packed == [
        ["PGS000001", "PGS000002"],
        ["PGS000003"],
        ["PGS000004"],
        ["PGS000005"],
    ]


def test_plan_checkpoints_splits_large_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRS_SAMPLE_SCORE_LARGE_VARIANT_THRESHOLD", "5")
    scores = tmp_path / "scores"
    scores.mkdir()
    _write_scoring(scores, "PGS000001", 2)
    _write_scoring(scores, "PGS000002", 2)
    _write_scoring(scores, "PGS000003", 10)
    _write_scoring(scores, "PGS000004", 2)
    _write_scoring(scores, "PGS000005", 10)
    prepared = [
        PreparedSample(record=_record("anton"), parquet_path="a.parquet", vcf_path="a.vcf"),
        PreparedSample(record=_record("livia"), parquet_path="l.parquet", vcf_path="l.vcf"),
    ]
    logs: list[str] = []
    planned = plan_checkpoints(
        ["PGS000001", "PGS000002", "PGS000003", "PGS000004", "PGS000005"],
        prepared,
        SCORE_PROFILES[UNRESTORED_PROFILE_ID],
        scores,
        {},
        universe_fp=None,
        batch_size=2,
        log=logs.append,
    )
    assert [item.meta.pgs_ids for item in planned] == [
        ["PGS000001", "PGS000002"],
        ["PGS000003"],
        ["PGS000004"],
        ["PGS000005"],
    ]
    assert planned[1].meta.n_rows == 2
    assert any("2 large score(s) as singleton checkpoints" in line for line in logs)


def test_deferred_plan_starts_without_hashing_catalog(tmp_path: Path) -> None:
    from just_prs.sample_scores.engine import materialize_checkpoint_fingerprints

    scores = tmp_path / "scores"
    scores.mkdir()
    pl.DataFrame({
        "hm_chr": ["1"],
        "hm_pos": [10],
        "effect_allele": ["A"],
        "effect_weight": [0.1],
        "other_allele": ["G"],
    }).write_parquet(scores / "PGS000001_hmPOS_GRCh38.parquet")
    prepared = [
        PreparedSample(record=_record("anton"), parquet_path="a.parquet", vcf_path="a.vcf"),
    ]
    planned = plan_checkpoints(
        ["PGS000001"],
        prepared,
        SCORE_PROFILES[UNRESTORED_PROFILE_ID],
        tmp_path / "missing-scores",
        {},
        universe_fp=None,
        batch_size=1,
        defer_fingerprints=True,
    )
    assert planned[0].meta.scoring_set_fingerprint == "deferred"
    assert planned[0].scoring_fingerprints == {}
    filled = materialize_checkpoint_fingerprints(
        planned[0],
        scores,
        SCORE_PROFILES[UNRESTORED_PROFILE_ID],
        {},
        universe_fp=None,
    )
    assert filled.meta.scoring_set_fingerprint != "deferred"
    assert "PGS000001" in filled.scoring_fingerprints


def test_worker_recycles_and_parent_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from just_prs.sample_scores.engine import score_checkpoint
    from just_prs.sample_scores.models import RuntimeResultRow

    calls = {"n": 0}

    def _fake_score(work, prepared, **kwargs):
        calls["n"] += 1
        rows = []
        for pgs_id in work.meta.pgs_ids:
            for sample in prepared:
                rows.append(
                    RuntimeResultRow(
                        sample_id=sample.record.sample_id,
                        pgs_id=pgs_id,
                        scoring_build="GRCh38",
                        score_profile_id=UNRESTORED_PROFILE_ID,
                        scoring_fingerprint=work.scoring_fingerprints[pgs_id],
                        score=0.1,
                        variants_matched=1,
                        variants_total=1,
                        match_rate=1.0,
                        sample_genotype_sha256=sample.record.genotype_sha256_v1,
                    )
                )
        return rows

    monkeypatch.setattr("just_prs.sample_scores.engine.score_checkpoint", _fake_score)
    monkeypatch.setattr(
        "just_prs.sample_scores.engine.recycle_reason",
        lambda snap, budget_bytes=None: "memory_budget" if calls["n"] >= 1 else None,
    )

    prepared = [
        PreparedSample(record=_record("anton"), parquet_path="a.parquet", vcf_path="a.vcf"),
    ]
    works = []
    for pgs_id in ("PGS000001", "PGS000002"):
        meta = CheckpointMeta(
            checkpoint_key=f"key_{pgs_id}",
            score_profile_id=UNRESTORED_PROFILE_ID,
            pgs_ids=[pgs_id],
            scoring_set_fingerprint="s" * 64,
            sample_set_genotype_fingerprint="g" * 64,
            n_rows=1,
        )
        works.append(CheckpointWork(meta=meta, scoring_fingerprints={pgs_id: "fp"}))
    order = WorkerWorkOrder(
        cache_dir=str(tmp_path),
        scores_cache=str(tmp_path / "scores"),
        profile_id=UNRESTORED_PROFILE_ID,
        genome_build="GRCh38",
        reference_restoration=False,
        samples=prepared,
        checkpoints=works,
        just_prs_version="0.0.0",
        computed_at="2026-08-16T00:00:00+00:00",
    )
    report = run_worker(order)
    assert report.recycle_reason == "memory_budget"
    assert report.n_checkpoints == 1
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == {"PGS000001"}

    order.checkpoints = works[1:]
    monkeypatch.setattr(
        "just_prs.sample_scores.engine.recycle_reason",
        lambda snap, budget_bytes=None: None,
    )
    report2 = run_worker(order)
    assert report2.n_checkpoints == 1
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == {
        "PGS000001",
        "PGS000002",
    }


def test_worker_recycles_after_large_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from just_prs.sample_scores.models import RuntimeResultRow

    monkeypatch.setenv("PRS_SAMPLE_SCORE_LARGE_VARIANT_THRESHOLD", "5")
    scores = tmp_path / "scores"
    scores.mkdir()
    for pgs_id in ("PGS000001", "PGS000002"):
        _write_scoring(scores, pgs_id, 6)

    def _fake_score(work, prepared, **kwargs):
        rows = []
        for pgs_id in work.meta.pgs_ids:
            for sample in prepared:
                rows.append(
                    RuntimeResultRow(
                        sample_id=sample.record.sample_id,
                        pgs_id=pgs_id,
                        scoring_build="GRCh38",
                        score_profile_id=UNRESTORED_PROFILE_ID,
                        scoring_fingerprint=work.scoring_fingerprints[pgs_id],
                        score=0.1,
                        variants_matched=1,
                        variants_total=6,
                        match_rate=1.0,
                        sample_genotype_sha256=sample.record.genotype_sha256_v1,
                    )
                )
        return rows

    monkeypatch.setattr("just_prs.sample_scores.engine.score_checkpoint", _fake_score)
    monkeypatch.setattr(
        "just_prs.sample_scores.engine.recycle_reason",
        lambda snap, budget_bytes=None: None,
    )
    prepared = [
        PreparedSample(record=_record("anton"), parquet_path="a.parquet", vcf_path="a.vcf"),
    ]
    works = []
    for pgs_id in ("PGS000001", "PGS000002"):
        meta = CheckpointMeta(
            checkpoint_key=f"key_{pgs_id}",
            score_profile_id=UNRESTORED_PROFILE_ID,
            pgs_ids=[pgs_id],
            scoring_set_fingerprint="s" * 64,
            sample_set_genotype_fingerprint="g" * 64,
            n_rows=1,
        )
        works.append(CheckpointWork(meta=meta, scoring_fingerprints={pgs_id: "fp"}))
    order = WorkerWorkOrder(
        cache_dir=str(tmp_path),
        scores_cache=str(scores),
        profile_id=UNRESTORED_PROFILE_ID,
        genome_build="GRCh38",
        reference_restoration=False,
        samples=prepared,
        checkpoints=works,
        just_prs_version="0.0.0",
        computed_at="2026-08-16T00:00:00+00:00",
    )
    report = run_worker(order)
    assert report.recycle_reason == "large_score"
    assert report.n_checkpoints == 1
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == {"PGS000001"}


def test_run_phase_retries_failed_and_continues_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from just_prs.sample_scores.checkpoints import write_runtime_part
    from just_prs.sample_scores.models import RuntimeResultRow

    scored: list[str] = []

    def _fake_score(work, prepared, **kwargs):
        rows = []
        for pgs_id in work.meta.pgs_ids:
            scored.append(pgs_id)
            for sample in prepared:
                rows.append(
                    RuntimeResultRow(
                        sample_id=sample.record.sample_id,
                        pgs_id=pgs_id,
                        scoring_build="GRCh38",
                        score_profile_id=UNRESTORED_PROFILE_ID,
                        scoring_fingerprint=work.scoring_fingerprints.get(pgs_id, "fp"),
                        score=0.2,
                        variants_matched=1,
                        variants_total=1,
                        match_rate=1.0,
                        sample_genotype_sha256=sample.record.genotype_sha256_v1,
                    )
                )
        return rows

    monkeypatch.setenv("PRS_SAMPLE_SCORE_INPROCESS", "1")
    monkeypatch.setattr("just_prs.sample_scores.engine.score_checkpoint", _fake_score)
    monkeypatch.setattr(
        "just_prs.sample_scores.engine.recycle_reason",
        lambda snap, budget_bytes=None: None,
    )
    prepared = [
        PreparedSample(record=_record("anton"), parquet_path="a.parquet", vcf_path="a.vcf"),
    ]
    write_runtime_part(
        [
            RuntimeResultRow(
                sample_id="anton",
                pgs_id="PGS000001",
                scoring_build="GRCh38",
                score_profile_id=UNRESTORED_PROFILE_ID,
                scoring_fingerprint="fp",
                score=0.1,
                variants_matched=1,
                variants_total=1,
                match_rate=1.0,
                sample_genotype_sha256="b" * 64,
            )
        ],
        tmp_path,
        CheckpointMeta(
            checkpoint_key="ok",
            score_profile_id=UNRESTORED_PROFILE_ID,
            pgs_ids=["PGS000001"],
            scoring_set_fingerprint="s" * 64,
            sample_set_genotype_fingerprint="g" * 64,
            n_rows=1,
            n_ok=1,
        ),
    )
    write_runtime_part(
        [
            RuntimeResultRow(
                sample_id="anton",
                pgs_id="PGS000002",
                scoring_build="GRCh38",
                score_profile_id=UNRESTORED_PROFILE_ID,
                scoring_fingerprint="fp",
                status="failed",
                error="worker_sigsegv",
                sample_genotype_sha256="b" * 64,
            )
        ],
        tmp_path,
        CheckpointMeta(
            checkpoint_key="fail",
            score_profile_id=UNRESTORED_PROFILE_ID,
            pgs_ids=["PGS000002"],
            scoring_set_fingerprint="s" * 64,
            sample_set_genotype_fingerprint="g" * 64,
            n_rows=1,
            n_ok=0,
            n_failed=1,
        ),
    )
    planned = []
    for pgs_id in ("PGS000001", "PGS000002", "PGS000003"):
        planned.append(
            CheckpointWork(
                meta=CheckpointMeta(
                    checkpoint_key=f"deferred-{pgs_id}",
                    score_profile_id=UNRESTORED_PROFILE_ID,
                    pgs_ids=[pgs_id],
                    scoring_set_fingerprint="deferred",
                    sample_set_genotype_fingerprint="g" * 64,
                    n_rows=1,
                ),
                scoring_fingerprints={},
            )
        )
    logs: list[str] = []
    run_phase(
        SCORE_PROFILES[UNRESTORED_PROFILE_ID],
        planned,
        prepared,
        cache_dir=tmp_path,
        scores_cache=tmp_path / "scores",
        universe_path=None,
        universe_fp=None,
        traits={},
        just_prs_version="0.0.0",
        computed_at="2026-08-16T00:00:00+00:00",
        progress_every=1,
        retry_failed=True,
        log=logs.append,
    )
    assert scored == ["PGS000002", "PGS000003"]
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == {
        "PGS000001",
        "PGS000002",
        "PGS000003",
    }
    assert any("retrying 1 failed PGS" in line for line in logs)


def test_progress_line_names_profile_and_pgs_span() -> None:
    line = format_sample_score_progress(
        profile_id=UNRESTORED_PROFILE_ID,
        pgs_done=40,
        pgs_total=5385,
        checkpoint_done=4,
        checkpoints_total=539,
        pgs_ids=["PGS000031", "PGS000040"],
        n_ok=70,
        n_failed=0,
        n_samples=7,
        duration_sec=8.2,
        peak_rss_mb=4500,
        available_mb=52000,
    )
    assert "Sample scores grch38-wgs-pass-unrestored-v1:" in line
    assert "40/5385 PGS (0.7%)" in line
    assert "checkpoint 4/539 PGS000031..PGS000040" in line
    assert "+70 ok / +0 failed this batch (7 samples)" in line
    assert "peak=4500 MB" in line


def test_progress_logs_on_checkpoint_boundary_and_finish() -> None:
    assert _should_log_progress(10, 10, 5385, 10)
    assert not _should_log_progress(20, 10, 5385, 50)
    assert _should_log_progress(50, 10, 5385, 50)
    assert _should_log_progress(5385, 5, 5385, 50)


def test_run_phase_resumes_by_pgs_id_not_deferred_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from just_prs.sample_scores.models import RuntimeResultRow

    calls = {"n": 0}

    def _fake_score(work, prepared, **kwargs):
        calls["n"] += 1
        return [
            RuntimeResultRow(
                sample_id=sample.record.sample_id,
                pgs_id=pgs_id,
                scoring_build="GRCh38",
                score_profile_id=UNRESTORED_PROFILE_ID,
                scoring_fingerprint=work.scoring_fingerprints.get(pgs_id, "fp"),
                score=0.1,
                variants_matched=1,
                variants_total=1,
                match_rate=1.0,
                sample_genotype_sha256=sample.record.genotype_sha256_v1,
            )
            for pgs_id in work.meta.pgs_ids
            for sample in prepared
        ]

    monkeypatch.setenv("PRS_SAMPLE_SCORE_INPROCESS", "1")
    monkeypatch.setattr("just_prs.sample_scores.engine.score_checkpoint", _fake_score)
    monkeypatch.setattr(
        "just_prs.sample_scores.engine.recycle_reason",
        lambda snap, budget_bytes=None: "memory_budget" if calls["n"] >= 1 else None,
    )

    prepared = [
        PreparedSample(record=_record("anton"), parquet_path="a.parquet", vcf_path="a.vcf"),
    ]
    planned = []
    for pgs_id in ("PGS000001", "PGS000002"):
        planned.append(
            CheckpointWork(
                meta=CheckpointMeta(
                    checkpoint_key=f"deferred-{pgs_id}",
                    score_profile_id=UNRESTORED_PROFILE_ID,
                    pgs_ids=[pgs_id],
                    scoring_set_fingerprint="deferred",
                    sample_set_genotype_fingerprint="g" * 64,
                    n_rows=1,
                ),
                scoring_fingerprints={},
            )
        )
    logs: list[str] = []
    progress = run_phase(
        SCORE_PROFILES[UNRESTORED_PROFILE_ID],
        planned,
        prepared,
        cache_dir=tmp_path,
        scores_cache=tmp_path / "scores",
        universe_path=None,
        universe_fp=None,
        traits={},
        just_prs_version="0.0.0",
        computed_at="2026-08-16T00:00:00+00:00",
        progress_every=1,
        log=logs.append,
    )
    assert calls["n"] == 2
    assert progress.n_workers == 2
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == {
        "PGS000001",
        "PGS000002",
    }
    assert any("1/2 PGS" in line for line in logs)
    assert any("2/2 PGS" in line for line in logs)
    assert not any("Unknown/private" in line for line in logs)


def test_worker_exit_recycle_reason_maps_sigsegv() -> None:
    from just_prs.sample_scores.engine import worker_exit_recycle_reason

    assert worker_exit_recycle_reason(0) is None
    assert worker_exit_recycle_reason(-11) == "worker_sigsegv"
    assert worker_exit_recycle_reason(-9) == "worker_sigkill"
    assert worker_exit_recycle_reason(1) == "worker_exit_1"


def test_run_phase_isolates_native_crash_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from just_prs.sample_scores.engine import WorkerReport
    from just_prs.sample_scores.models import RuntimeResultRow

    def _ok_rows(work: CheckpointWork, prepared: list[PreparedSample]) -> list[RuntimeResultRow]:
        return [
            RuntimeResultRow(
                sample_id=sample.record.sample_id,
                pgs_id=pgs_id,
                scoring_build="GRCh38",
                score_profile_id=UNRESTORED_PROFILE_ID,
                scoring_fingerprint=work.scoring_fingerprints.get(pgs_id, "fp"),
                score=0.1,
                variants_matched=1,
                variants_total=1,
                match_rate=1.0,
                sample_genotype_sha256=sample.record.genotype_sha256_v1,
            )
            for pgs_id in work.meta.pgs_ids
            for sample in prepared
        ]

    def _fake_sub(order, work_path, report_path, log=None):
        head = tuple(order.checkpoints[0].meta.pgs_ids)
        if head == ("PGS000001", "PGS000002") or head == ("PGS000002",):
            return WorkerReport(
                profile_id=UNRESTORED_PROFILE_ID,
                recycle_reason="worker_sigsegv",
                error="worker exited -11 without a report",
            )
        from just_prs.sample_scores.engine import write_runtime_part

        work = order.checkpoints[0]
        rows = _ok_rows(work, order.samples)
        meta = work.meta.model_copy(update={"n_ok": len(rows), "n_failed": 0, "n_rows": len(rows)})
        write_runtime_part(rows, Path(order.cache_dir), meta)
        return WorkerReport(profile_id=UNRESTORED_PROFILE_ID, n_checkpoints=1, n_ok=len(rows))

    monkeypatch.setattr("just_prs.sample_scores.engine.inprocess_workers", lambda: False)
    monkeypatch.setattr("just_prs.sample_scores.engine._run_worker_subprocess", _fake_sub)

    prepared = [
        PreparedSample(record=_record("anton"), parquet_path="a.parquet", vcf_path="a.vcf"),
    ]
    planned = [
        CheckpointWork(
            meta=CheckpointMeta(
                checkpoint_key="batch",
                score_profile_id=UNRESTORED_PROFILE_ID,
                pgs_ids=["PGS000001", "PGS000002"],
                scoring_set_fingerprint="s" * 64,
                sample_set_genotype_fingerprint="g" * 64,
                n_rows=2,
            ),
            scoring_fingerprints={"PGS000001": "fp", "PGS000002": "fp"},
        ),
        CheckpointWork(
            meta=CheckpointMeta(
                checkpoint_key="later",
                score_profile_id=UNRESTORED_PROFILE_ID,
                pgs_ids=["PGS000003"],
                scoring_set_fingerprint="t" * 64,
                sample_set_genotype_fingerprint="g" * 64,
                n_rows=1,
            ),
            scoring_fingerprints={"PGS000003": "fp"},
        ),
    ]
    logs: list[str] = []
    progress = run_phase(
        SCORE_PROFILES[UNRESTORED_PROFILE_ID],
        planned,
        prepared,
        cache_dir=tmp_path,
        scores_cache=tmp_path / "scores",
        universe_path=None,
        universe_fp=None,
        traits={},
        just_prs_version="0.0.0",
        computed_at="2026-08-16T00:00:00+00:00",
        progress_every=1,
        log=logs.append,
    )
    done = completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID)
    assert done == {"PGS000001", "PGS000002", "PGS000003"}
    assert progress.n_failed == 1
    assert any("one at a time" in line for line in logs)
    assert any("recorded failure for PGS000002" in line for line in logs)
    hint = (
        tmp_path / "sample_scores" / "isolate" / UNRESTORED_PROFILE_ID
        / "PGS000001_PGS000002_n2.json"
    )
    assert hint.is_file()


def test_apply_isolation_hints_splits_before_retry(tmp_path: Path) -> None:
    from just_prs.sample_scores.engine import apply_isolation_hints, write_isolation_hint

    write_isolation_hint(
        tmp_path,
        UNRESTORED_PROFILE_ID,
        ["PGS000001", "PGS000002"],
        "worker_sigsegv",
    )
    planned = [
        CheckpointWork(
            meta=CheckpointMeta(
                checkpoint_key="batch",
                score_profile_id=UNRESTORED_PROFILE_ID,
                pgs_ids=["PGS000001", "PGS000002"],
                scoring_set_fingerprint="s" * 64,
                sample_set_genotype_fingerprint="g" * 64,
                n_rows=2,
            ),
            scoring_fingerprints={},
        ),
        CheckpointWork(
            meta=CheckpointMeta(
                checkpoint_key="later",
                score_profile_id=UNRESTORED_PROFILE_ID,
                pgs_ids=["PGS000003"],
                scoring_set_fingerprint="t" * 64,
                sample_set_genotype_fingerprint="g" * 64,
                n_rows=1,
            ),
            scoring_fingerprints={},
        ),
    ]
    assert apply_isolation_hints(planned, tmp_path, UNRESTORED_PROFILE_ID) == 1
    assert [item.meta.pgs_ids for item in planned] == [
        ["PGS000001"],
        ["PGS000002"],
        ["PGS000003"],
    ]


def test_reconcile_checkpoint_keys_accepts_isolated_parts(tmp_path: Path) -> None:
    from just_prs.sample_scores.engine import reconcile_checkpoint_keys
    from just_prs.sample_scores.checkpoints import write_runtime_part
    from just_prs.sample_scores.models import RuntimeResultRow

    planned_batch = CheckpointWork(
        meta=CheckpointMeta(
            checkpoint_key="planned-batch",
            score_profile_id=UNRESTORED_PROFILE_ID,
            pgs_ids=["PGS000001", "PGS000002"],
            scoring_set_fingerprint="s" * 64,
            sample_set_genotype_fingerprint="g" * 64,
            n_rows=2,
        ),
        scoring_fingerprints={},
    )
    isolated = CheckpointMeta(
        checkpoint_key="isolated-pgs000002",
        score_profile_id=UNRESTORED_PROFILE_ID,
        pgs_ids=["PGS000002"],
        scoring_set_fingerprint="t" * 64,
        sample_set_genotype_fingerprint="g" * 64,
        n_rows=1,
        n_failed=1,
    )
    write_runtime_part(
        [
            RuntimeResultRow(
                sample_id="anton",
                pgs_id="PGS000002",
                scoring_build="GRCh38",
                score_profile_id=UNRESTORED_PROFILE_ID,
                scoring_fingerprint="fp",
                status="failed",
                error="worker_sigsegv",
                sample_genotype_sha256="b" * 64,
            )
        ],
        tmp_path,
        isolated,
    )
    keys = reconcile_checkpoint_keys({UNRESTORED_PROFILE_ID: [planned_batch]}, tmp_path)
    assert "planned-batch" in keys
    assert "isolated-pgs000002" in keys

    covered = CheckpointMeta(
        checkpoint_key="isolated-pgs000001",
        score_profile_id=UNRESTORED_PROFILE_ID,
        pgs_ids=["PGS000001"],
        scoring_set_fingerprint="u" * 64,
        sample_set_genotype_fingerprint="g" * 64,
        n_rows=1,
        n_ok=1,
    )
    write_runtime_part(
        [
            RuntimeResultRow(
                sample_id="anton",
                pgs_id="PGS000001",
                scoring_build="GRCh38",
                score_profile_id=UNRESTORED_PROFILE_ID,
                scoring_fingerprint="fp",
                score=0.1,
                variants_matched=1,
                variants_total=1,
                match_rate=1.0,
                sample_genotype_sha256="b" * 64,
            )
        ],
        tmp_path,
        covered,
    )
    keys = reconcile_checkpoint_keys({UNRESTORED_PROFILE_ID: [planned_batch]}, tmp_path)
    assert "planned-batch" not in keys
    assert keys == {"isolated-pgs000001", "isolated-pgs000002"}
