"""Runtime numeric invariants and targeted invalid-part repair."""

from __future__ import annotations

from pathlib import Path

import pytest

from just_prs.sample_scores.checkpoints import (
    CheckpointMeta,
    compact_runtime_parts,
    completed_pgs_for_profile,
    make_checkpoint_key,
    reopen_invalid_parts,
    write_runtime_part,
)
from just_prs.sample_scores.completeness import (
    MATCH_RATE_TOLERANCE,
    WEIGHT_COVERAGE_TOLERANCE,
    CompletenessError,
    assert_finite_coverage,
    ok_row_invariant_issues,
    validate_runtime_results,
)
from just_prs.sample_scores.models import RESTORED_PROFILE_ID, UNRESTORED_PROFILE_ID, SampleRecord
from just_prs.sample_scores.publish import RuntimeResultRow
from just_prs.sample_scores.store import write_samples


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


def _ok_row(sample: SampleRecord, pgs_id: str = "PGS000001", **updates: object) -> RuntimeResultRow:
    payload: dict[str, object] = {
        "sample_id": sample.sample_id,
        "pgs_id": pgs_id,
        "scoring_build": "GRCh38",
        "score_profile_id": UNRESTORED_PROFILE_ID,
        "scoring_fingerprint": "fp-current",
        "score": 0.25,
        "variants_matched": 10,
        "variants_total": 12,
        "match_rate": 10 / 12,
        "variants_assumed_hom_ref": 2,
        "variants_ref_resolved_panel": 1,
        "variants_ref_resolved_fasta": 0,
        "weight_mass_matched": 1.0,
        "weight_mass_total": 2.0,
        "weight_mass_coverage": 0.5,
        "sample_genotype_sha256": sample.genotype_sha256_v1,
        "trait_reported": "body mass index",
    }
    payload.update(updates)
    return RuntimeResultRow.model_validate(payload)


def _meta(pgs_ids: list[str], n_rows: int, key: str) -> CheckpointMeta:
    return CheckpointMeta(
        checkpoint_key=key,
        score_profile_id=UNRESTORED_PROFILE_ID,
        pgs_ids=pgs_ids,
        scoring_set_fingerprint="s" * 64,
        sample_set_genotype_fingerprint="g" * 64,
        n_rows=n_rows,
        n_ok=n_rows,
    )


def test_impossible_counters_and_rates_fail() -> None:
    sample = _sample()
    assert ok_row_invariant_issues(_ok_row(sample).model_dump()) == []
    assert ok_row_invariant_issues(
        _ok_row(sample, variants_matched=13, match_rate=13 / 12).model_dump()
    )
    assert ok_row_invariant_issues(_ok_row(sample, match_rate=1.5).model_dump())
    assert ok_row_invariant_issues(
        _ok_row(sample, variants_ref_resolved_panel=3, variants_assumed_hom_ref=2).model_dump()
    )
    nan_row = _ok_row(sample).model_dump()
    nan_row["score"] = float("nan")
    assert ok_row_invariant_issues(nan_row)
    with pytest.raises(CompletenessError):
        assert_finite_coverage(_ok_row(sample, variants_matched=-1).model_dump())


def test_float_tolerance_is_explicit() -> None:
    assert MATCH_RATE_TOLERANCE == 1e-6
    assert WEIGHT_COVERAGE_TOLERANCE == 1e-6
    sample = _sample()
    almost = _ok_row(sample, match_rate=(10 / 12) + 5e-7)
    assert ok_row_invariant_issues(almost.model_dump()) == []
    drifted = _ok_row(sample, match_rate=(10 / 12) + 2e-6)
    assert any("match_rate inconsistent" in item for item in ok_row_invariant_issues(drifted.model_dump()))


def test_fingerprint_drift_and_profile_parity_fail(tmp_path: Path) -> None:
    sample = _sample()
    write_samples([sample], tmp_path)
    unrestored = _ok_row(sample, scoring_fingerprint="old")
    restored = _ok_row(
        sample,
        score_profile_id=RESTORED_PROFILE_ID,
        scoring_fingerprint="new",
    )
    write_runtime_part([unrestored], tmp_path, _meta(["PGS000001"], 1, "u1"))
    write_runtime_part(
        [restored],
        tmp_path,
        CheckpointMeta(
            checkpoint_key="r1",
            score_profile_id=RESTORED_PROFILE_ID,
            pgs_ids=["PGS000001"],
            scoring_set_fingerprint="s" * 64,
            sample_set_genotype_fingerprint="g" * 64,
            n_rows=1,
            n_ok=1,
        ),
    )
    compact_runtime_parts(tmp_path, expected_pgs_ids=["PGS000001"])
    report = validate_runtime_results(
        tmp_path,
        expected_sample_ids=["anton"],
        expected_pgs_ids=["PGS000001"],
        current_scoring_fingerprints={"PGS000001": "new"},
    )
    assert report.passed is False
    assert any("fingerprint mismatch" in issue for issue in report.issues)
    assert any("drifted from the current scoring snapshot" in issue for issue in report.issues)


def test_targeted_repair_invalidates_only_affected_parts(tmp_path: Path) -> None:
    sample = _sample()
    good = _ok_row(sample, "PGS000001")
    bad = _ok_row(sample, "PGS000002", variants_matched=99, variants_total=10, match_rate=9.9)
    stale = _ok_row(sample, "PGS000003", scoring_fingerprint="stale")
    write_runtime_part([good], tmp_path, _meta(["PGS000001"], 1, "good"))
    write_runtime_part([bad], tmp_path, _meta(["PGS000002"], 1, "bad"))
    write_runtime_part([stale], tmp_path, _meta(["PGS000003"], 1, "stale"))
    report = reopen_invalid_parts(
        tmp_path,
        UNRESTORED_PROFILE_ID,
        current_scoring_fingerprints={"PGS000001": "fp-current", "PGS000003": "fp-current"},
        row_issues=ok_row_invariant_issues,
    )
    assert set(report.pgs_ids) == {"PGS000002", "PGS000003"}
    assert report.n_parts_quarantined == 2
    assert completed_pgs_for_profile(tmp_path, UNRESTORED_PROFILE_ID) == {"PGS000001"}
    dest = compact_runtime_parts(tmp_path, expected_pgs_ids=["PGS000001"])
    rows = dest.read_bytes()
    dest_again = compact_runtime_parts(tmp_path, expected_pgs_ids=["PGS000001"])
    assert dest_again.read_bytes() == rows


def test_make_checkpoint_key_is_stable() -> None:
    kwargs = {
        "score_profile_id": UNRESTORED_PROFILE_ID,
        "score_algorithm_version": "1",
        "genome_build": "GRCh38",
        "pgs_ids": ["PGS000001", "PGS000002"],
        "scoring_set_fp": "s" * 64,
        "sample_set_fp": "g" * 64,
        "reference_universe_fp": None,
    }
    assert make_checkpoint_key(**kwargs) == make_checkpoint_key(**kwargs)
