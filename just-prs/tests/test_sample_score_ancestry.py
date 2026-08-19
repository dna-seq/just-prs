"""Published ancestry provenance and private-alias privacy."""

from __future__ import annotations

from pathlib import Path

import pytest

from just_prs.sample_scores.completeness import validate_runtime_results
from just_prs.sample_scores.models import (
    ANCESTRY_INFERENCE_VERSION,
    EXPECTED_PUBLIC_SAMPLE_ANCESTRY,
    SampleAncestryRecord,
    SampleRecord,
)
from just_prs.sample_scores.store import write_sample_ancestry, write_samples
from just_prs.scoring import resolve_cache_dir


def _sample(sample_id: str = "anton") -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        aliases=["Anton"] if sample_id == "anton" else [],
        display_name=sample_id,
        license="CC0",
        publication_allowed=True,
        consent_basis="public-domain",
        source_sha256="a" * 64,
        genotype_sha256_v1="b" * 64,
    )


def test_sample_ancestry_record_is_one_row_per_sample(tmp_path: Path) -> None:
    rows = [
        SampleAncestryRecord(
            sample_id=sample_id,
            genotype_sha256_v1="b" * 64,
            superpopulation=superpop,
            confidence=1.0,
            fine_population=fine,
            fine_confidence=0.55,
            ancestry_model_sha256="c" * 64,
            inference_version=ANCESTRY_INFERENCE_VERSION,
        )
        for sample_id, (superpop, fine) in EXPECTED_PUBLIC_SAMPLE_ANCESTRY.items()
    ]
    path = write_sample_ancestry(rows, tmp_path)
    assert path.name == "sample_ancestry.parquet"
    assert {row.sample_id for row in rows} == set(EXPECTED_PUBLIC_SAMPLE_ANCESTRY)
    assert all(row.superpopulation != "UNKNOWN" for row in rows)
    assert all(row.genotype_sha256_v1 == "b" * 64 for row in rows)
    assert all(row.ancestry_model_sha256 == "c" * 64 for row in rows)


def test_ancestry_gate_rejects_unknown_or_hash_mismatch(tmp_path: Path) -> None:
    sample = _sample("anton")
    write_samples([sample], tmp_path)
    write_sample_ancestry(
        [
            SampleAncestryRecord(
                sample_id="anton",
                genotype_sha256_v1="d" * 64,
                superpopulation="UNKNOWN",
                confidence=0.0,
            )
        ],
        tmp_path,
    )
    from just_prs.sample_scores.store import write_runtime_results
    from just_prs.sample_scores.models import UNRESTORED_PROFILE_ID, RESTORED_PROFILE_ID
    from just_prs.sample_scores.publish import RuntimeResultRow

    rows = [
        RuntimeResultRow(
            sample_id="anton",
            pgs_id="PGS000001",
            scoring_build="GRCh38",
            score_profile_id=profile,
            scoring_fingerprint="fp",
            score=0.1,
            variants_matched=1,
            variants_total=1,
            match_rate=1.0,
            sample_genotype_sha256="b" * 64,
        )
        for profile in (UNRESTORED_PROFILE_ID, RESTORED_PROFILE_ID)
    ]
    write_runtime_results(rows, tmp_path)
    report = validate_runtime_results(
        tmp_path,
        expected_sample_ids=["anton"],
        expected_pgs_ids=["PGS000001"],
        require_ancestry=True,
    )
    assert report.passed is False
    assert any("UNKNOWN" in issue or "hash mismatch" in issue for issue in report.issues)


def test_real_public_genome_ancestry_matches_recorded_calls() -> None:
    cache = resolve_cache_dir()
    parquet = cache / "normalized" / "sample_scores" / "anton.parquet"
    if not parquet.exists():
        pytest.skip("normalized public anton parquet is not in this cache")
    from just_prs.prs_catalog import PRSCatalog

    catalog = PRSCatalog(cache_dir=cache)
    lf = __import__("polars").scan_parquet(parquet)
    broad = catalog.infer_ancestry(
        genotypes_lf=lf, panel="1000g", sample_build="GRCh38", resolution="superpop"
    )
    fine = catalog.infer_ancestry(
        genotypes_lf=lf, panel="1000g", sample_build="GRCh38", resolution="population"
    )
    if broad.superpopulation == "UNKNOWN":
        pytest.skip("1000G ancestry model is not available")
    assert broad.superpopulation == "EUR"
    assert broad.confidence == 1.0
    assert fine.fine_population == "CEU"
