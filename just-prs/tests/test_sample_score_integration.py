"""Pinned staging, model analysis, and ancestry-selected trait summaries."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import polars as pl
import pytest

from just_prs.absolute_risk import estimate_absolute_risk
from just_prs.hf import (
    SampleScorePublishError,
    sample_score_commit_operations,
    sample_score_integration_commit_operations,
)
from just_prs.sample_scores.integration.build import build_model_analysis
from just_prs.sample_scores.integration.checks import (
    validate_model_analysis,
    validate_trait_summaries,
)
from just_prs.sample_scores.integration.docs import write_final_docs
from just_prs.sample_scores.integration.models import (
    EXCLUSION_DIST_UNAVAILABLE,
    EXCLUSION_FAILED,
    EXCLUSION_QUARANTINED,
    FINE_COHORT_LABELS,
)
from just_prs.sample_scores.integration.pins import (
    SAMPLE_SCORES_REPO,
    SAMPLE_SCORES_REVISION,
    SUPERPOPULATIONS,
    PinnedFile,
)
from just_prs.sample_scores.integration.staging import (
    StagingError,
    sources_dir,
    stage_pinned_sources,
)
from just_prs.sample_scores.integration.summaries import build_trait_summaries
from just_prs.sample_scores.models import RESTORED_PROFILE_ID, UNRESTORED_PROFILE_ID

_SHA = "a" * 64
_FP = "fp-shared"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _empty_issues() -> pl.DataFrame:
    return pl.DataFrame({"pgs_id": pl.Series([], dtype=pl.Utf8), "severity": pl.Series([], dtype=pl.Utf8)})


def _runtime_row(
    sample_id: str,
    pgs_id: str,
    profile: str,
    *,
    score: float,
    status: str = "ok",
    fingerprint: str = _FP,
    matched: int = 80,
    total: int = 100,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "pgs_id": pgs_id,
        "scoring_build": "GRCh38",
        "score_profile_id": profile,
        "scoring_fingerprint": fingerprint,
        "status": status,
        "error": None if status == "ok" else "boom",
        "score": score if status == "ok" else None,
        "variants_matched": matched if status == "ok" else 0,
        "variants_total": total,
        "match_rate": (matched / total) if status == "ok" else 0.0,
        "weight_mass_coverage": 0.9 if status == "ok" else None,
        "sample_genotype_sha256": _SHA,
    }


def write_analysis_sources(cache_dir: Path) -> Path:
    """Write a tiny but real staged snapshot under the integration sources dir."""
    dest = sources_dir(cache_dir)
    dest.mkdir(parents=True, exist_ok=True)
    profiles = (UNRESTORED_PROFILE_ID, RESTORED_PROFILE_ID)
    runtime_rows: list[dict[str, object]] = []
    for sample_id in ("anton", "livia"):
        for profile in profiles:
            runtime_rows.extend(
                [
                    _runtime_row(sample_id, "PGS000001", profile, score=12.0 if profile == UNRESTORED_PROFILE_ID else 13.0),
                    _runtime_row(sample_id, "PGS000002", profile, score=1.0 if profile == UNRESTORED_PROFILE_ID else 1.2),
                    _runtime_row(sample_id, "PGS000Q", profile, score=0.5),
                    _runtime_row(sample_id, "PGS000F", profile, score=0.0, status="failed"),
                    _runtime_row(sample_id, "PGS000Z", profile, score=3.0),
                    _runtime_row(sample_id, "PGS000N", profile, score=4.0),
                ]
            )
    pl.DataFrame(runtime_rows).write_parquet(dest / "runtime_results.parquet")
    pl.DataFrame(
        [
            {
                "sample_id": "anton",
                "aliases": ["Anton"],
                "display_name": "Anton",
                "family_id": "kulaga",
                "relationship_role": "self",
                "license": "CC0",
                "publication_allowed": True,
                "consent_basis": "public-domain",
                "source_sha256": _SHA,
                "genotype_sha256_v1": _SHA,
            },
            {
                "sample_id": "livia",
                "aliases": ["Livia"],
                "display_name": "Livia",
                "family_id": None,
                "relationship_role": None,
                "license": "CC-BY-4.0",
                "publication_allowed": True,
                "consent_basis": "cc-by",
                "source_sha256": _SHA,
                "genotype_sha256_v1": _SHA,
            },
        ]
    ).write_parquet(dest / "samples.parquet")
    pl.DataFrame(
        [
            {
                "sample_id": "anton",
                "genotype_sha256_v1": _SHA,
                "superpopulation": "EUR",
                "confidence": 1.0,
                "fine_population": "CEU",
                "fine_confidence": 0.75,
            },
            {
                "sample_id": "livia",
                "genotype_sha256_v1": _SHA,
                "superpopulation": "EUR",
                "confidence": 1.0,
                "fine_population": "IBS",
                "fine_confidence": 0.65,
            },
        ]
    ).write_parquet(dest / "sample_ancestry.parquet")
    pl.DataFrame(
        [
            {"pgs_id": "PGS000001", "name": "BMI A", "trait_reported": "BMI", "genome_build": "GRCh38", "n_variants": 100},
            {"pgs_id": "PGS000002", "name": "BMI B", "trait_reported": "BMI", "genome_build": "GRCh37", "n_variants": 100},
            {"pgs_id": "PGS000Q", "name": "Quarantine", "trait_reported": "BMI", "genome_build": "GRCh38", "n_variants": 100},
            {"pgs_id": "PGS000F", "name": "Failed", "trait_reported": "BMI", "genome_build": "GRCh38", "n_variants": 100},
            {"pgs_id": "PGS000Z", "name": "Zero std", "trait_reported": "BMI", "genome_build": "GRCh38", "n_variants": 100},
            {"pgs_id": "PGS000N", "name": "No dist", "trait_reported": "BMI", "genome_build": "GRCh38", "n_variants": 100},
        ]
    ).write_parquet(dest / "scores.parquet")
    pl.DataFrame(
        [
            {"pgs_id": "PGS000001", "or_estimate": 1.4, "auroc_estimate": 0.72},
            {"pgs_id": "PGS000002", "or_estimate": 1.3, "auroc_estimate": 0.70},
        ]
    ).write_parquet(dest / "best_performance.parquet")
    pl.DataFrame(
        [{"pgs_id": "PGS000Q", "exclude_from_catalog": True}]
    ).write_parquet(dest / "catalog_scoring_flags.parquet")
    dist_rows: list[dict[str, object]] = []
    for pgs_id, mean, std in (
        ("PGS000001", 20.0, 2.0),
        ("PGS000002", -2.0, 1.0),
        ("PGS000Q", 0.0, 1.0),
        ("PGS000F", 0.0, 1.0),
        ("PGS000Z", 3.0, 0.0),
    ):
        for pop in SUPERPOPULATIONS:
            shift = {"AFR": -1.0, "AMR": -0.5, "EAS": 0.2, "EUR": 0.0, "SAS": 0.4}[pop]
            dist_rows.append(
                {
                    "pgs_id": pgs_id,
                    "superpopulation": pop,
                    "mean": mean + shift,
                    "std": std,
                    "n": 500,
                }
            )
    pl.DataFrame(dist_rows).write_parquet(dest / "1000g_distributions.parquet")
    pl.DataFrame(
        [{"pgs_id": "PGS000Z", "severity": "ERROR", "issue": "zero_std"}]
    ).write_parquet(dest / "1000g_distribution_quality_issues.parquet")
    pl.DataFrame(
        [
            {"pgs_id": "PGS000001", "trait_id": "EFO_0001360", "relationship_source": "catalog", "trait_reported": "BMI"},
            {"pgs_id": "PGS000002", "trait_id": "EFO_0001360", "relationship_source": "catalog", "trait_reported": "BMI"},
            {"pgs_id": "PGS000Q", "trait_id": "EFO_0001360", "relationship_source": "catalog", "trait_reported": "BMI"},
            {"pgs_id": "PGS000F", "trait_id": "EFO_0001360", "relationship_source": "catalog", "trait_reported": "BMI"},
            {"pgs_id": "PGS000001", "trait_id": "MONDO_0005148", "relationship_source": "catalog", "trait_reported": "T2D"},
            {"pgs_id": "PGS000002", "trait_id": "HP_9999999", "relationship_source": "catalog", "trait_reported": "looks like BMI"},
        ]
    ).write_parquet(dest / "score_trait_links.parquet")
    pl.DataFrame(
        [
            {"trait_id": "EFO_0001360", "label": "body mass index"},
            {"trait_id": "MONDO_0005148", "label": "type 2 diabetes mellitus"},
            {"trait_id": "HP_9999999", "label": "body mass index"},
        ]
    ).write_parquet(dest / "traits.parquet")
    pl.DataFrame(
        {
            "trait_id": ["EFO_0001360"],
            "guideline_id": ["g1"],
            "prs_actionability_status": ["not_assessed"],
            "condition_actionability_status": ["not_assessed"],
            "context_resolution_status": ["not_assessed"],
        }
    ).write_parquet(dest / "actionability.parquet")
    pl.DataFrame(
        {"trait_id": ["EFO_0001360"], "context_class": ["measurement"]}
    ).write_parquet(dest / "trait_contexts.parquet")
    pl.DataFrame(
        {"paper_id": ["pmid:1"], "pmid": ["1"], "doi": [None], "pgp_id": [None]}
    ).write_parquet(dest / "papers.parquet")
    pl.DataFrame(
        {"pgs_id": ["PGS000001"], "paper_id": ["pmid:1"], "relationship_type": ["development"]}
    ).write_parquet(dest / "score_paper_links.parquet")
    pl.DataFrame(
        [
            {
                "canonical_efo_id": "EFO_0001360",
                "mapped_from_id": "MONDO_0005148",
                "efo_id": "EFO_0001360",
                "prevalence": 0.12,
                "prevalence_type": "lifetime",
                "source": "seed",
                "confidence": "high",
                "ancestry": "EUR",
            }
        ]
    ).write_parquet(dest / "trait_prevalence.parquet")
    pl.DataFrame(
        [
            {
                "canonical_efo_id": "EFO_0001360",
                "mapped_from_id": "MONDO_0005148",
                "efo_id": "EFO_0001360",
                "h2_liability": 0.2,
                "h2_observed": None,
                "source": "panukbb",
                "ancestry": "EUR",
                "confidence": "high",
                "method": "ldsc",
            }
        ]
    ).write_parquet(dest / "trait_heritability.parquet")
    return dest


def _expected_percentile(score: float, mean: float, std: float) -> float:
    z = (score - mean) / std
    return round(0.5 * (1.0 - math.erf(-z / math.sqrt(2.0))) * 100.0, 2)


def test_offline_staging_fails_closed_on_missing_pin(tmp_path: Path) -> None:
    pin = PinnedFile(
        repo_id=SAMPLE_SCORES_REPO,
        revision=SAMPLE_SCORES_REVISION,
        repo_path="data/missing.parquet",
        sha256="b" * 64,
        local_name="missing.parquet",
    )
    with pytest.raises(StagingError, match="offline staging miss"):
        stage_pinned_sources(tmp_path, allow_network=False, pins=(pin,))


def test_staging_copies_local_override_and_checks_hash(tmp_path: Path) -> None:
    payload = b"pinned-bytes"
    digest = _sha256_bytes(payload)
    src = tmp_path / "local.bin"
    src.write_bytes(payload)
    pin = PinnedFile(
        repo_id="example/repo",
        revision="abc",
        repo_path="data/note.txt",
        sha256=digest,
        local_name="note.txt",
    )
    index = stage_pinned_sources(
        tmp_path,
        allow_network=False,
        pins=(pin,),
        local_files={("example/repo", "data/note.txt"): src},
    )
    staged = sources_dir(tmp_path) / "note.txt"
    assert staged.read_bytes() == payload
    assert index[0].sha256 == digest

    (tmp_path / "wrong.bin").write_bytes(b"nope")
    with pytest.raises(StagingError, match="hash mismatch"):
        stage_pinned_sources(
            tmp_path,
            allow_network=False,
            pins=(pin,),
            local_files={("example/repo", "data/note.txt"): tmp_path / "wrong.bin"},
        )


def test_model_analysis_keeps_ineligible_rows_without_lookup_values(tmp_path: Path) -> None:
    write_analysis_sources(tmp_path)
    analysis = build_model_analysis(tmp_path)
    runtime = pl.read_parquet(sources_dir(tmp_path) / "runtime_results.parquet")
    assert analysis.height == runtime.height
    issues = validate_model_analysis(analysis, runtime=runtime)
    assert issues == []

    quarantined = analysis.filter(pl.col("pgs_id") == "PGS000Q")
    assert quarantined.height == 4
    assert quarantined["analysis_eligible"].to_list() == [False] * 4
    assert all(EXCLUSION_QUARANTINED in reasons for reasons in quarantined["exclusion_reasons"].to_list())

    failed = analysis.filter(pl.col("pgs_id") == "PGS000F")
    assert all(EXCLUSION_FAILED in reasons for reasons in failed["exclusion_reasons"].to_list())

    zero = analysis.filter(pl.col("pgs_id") == "PGS000Z")
    assert all(EXCLUSION_DIST_UNAVAILABLE in reasons or "reference_audit_error" in reasons for reasons in zero["exclusion_reasons"].to_list())
    missing = analysis.filter(pl.col("pgs_id") == "PGS000N")
    assert all(EXCLUSION_DIST_UNAVAILABLE in reasons for reasons in missing["exclusion_reasons"].to_list())

    ineligible = analysis.filter(~pl.col("analysis_eligible"))
    for row in ineligible.iter_rows(named=True):
        for metric in row["population_metrics"]:
            assert metric["percentile"] is None
            assert metric["z_score"] is None
            assert metric["available"] is False
        assert row["quality_label"] is None
        for trait in row["traits"] or []:
            for risk in trait.get("population_risks") or []:
                assert risk.get("absolute_risk") is None


def test_five_population_metrics_match_direct_calculation(tmp_path: Path) -> None:
    write_analysis_sources(tmp_path)
    analysis = build_model_analysis(tmp_path)
    row = analysis.filter(
        (pl.col("sample_id") == "anton")
        & (pl.col("pgs_id") == "PGS000001")
        & (pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
    ).row(0, named=True)
    assert row["selected_superpopulation"] == "EUR"
    assert row["closest_cohort"] == "CEU"
    assert row["closest_cohort_label"] == FINE_COHORT_LABELS["CEU"]
    assert [item["superpopulation"] for item in row["population_metrics"]] == list(SUPERPOPULATIONS)
    dist = pl.read_parquet(sources_dir(tmp_path) / "1000g_distributions.parquet")
    for metric in row["population_metrics"]:
        pop = metric["superpopulation"]
        stats = dist.filter((pl.col("pgs_id") == "PGS000001") & (pl.col("superpopulation") == pop)).row(0, named=True)
        expected = _expected_percentile(12.0, float(stats["mean"]), float(stats["std"]))
        assert metric["available"] is True
        assert metric["percentile"] == pytest.approx(expected)
        assert metric["std"] == pytest.approx(stats["std"])
        assert pop not in {"CEU", "IBS"}


def test_raw_score_order_is_not_used_across_pgs_ids(tmp_path: Path) -> None:
    write_analysis_sources(tmp_path)
    analysis = build_model_analysis(tmp_path)
    pair = analysis.filter(
        (pl.col("sample_id") == "anton")
        & (pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
        & pl.col("pgs_id").is_in(["PGS000001", "PGS000002"])
        & pl.col("analysis_eligible")
    )
    scores = {row["pgs_id"]: row["score"] for row in pair.iter_rows(named=True)}
    pcts = {}
    for row in pair.iter_rows(named=True):
        selected = next(item for item in row["population_metrics"] if item["superpopulation"] == "EUR")
        pcts[row["pgs_id"]] = selected["percentile"]
    assert scores["PGS000001"] > scores["PGS000002"]
    assert pcts["PGS000001"] < pcts["PGS000002"]


def test_ontology_alias_joins_and_similar_label_does_not(tmp_path: Path) -> None:
    write_analysis_sources(tmp_path)
    analysis = build_model_analysis(tmp_path)
    row = analysis.filter(
        (pl.col("sample_id") == "anton")
        & (pl.col("pgs_id") == "PGS000002")
        & (pl.col("analysis_eligible"))
        & (pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
    ).row(0, named=True)
    by_id = {trait["trait_id"]: trait for trait in row["traits"]}
    mondo = by_id["MONDO_0005148"] if "MONDO_0005148" in by_id else None
    # PGS000002 is linked to EFO BMI and a lookalike HP id, not MONDO.
    efo = by_id["EFO_0001360"]
    hp = by_id["HP_9999999"]
    efo_risk = next(item for item in efo["population_risks"] if item["superpopulation"] == "EUR")
    hp_risk = next(item for item in hp["population_risks"] if item["superpopulation"] == "EUR")
    assert efo_risk["prevalence"] == pytest.approx(0.12)
    assert efo_risk["absolute_risk"] is not None
    assert efo_risk["h2"] == pytest.approx(0.2)
    assert efo_risk["h2_scale"] == "liability"
    assert hp_risk["prevalence"] is None
    assert hp_risk["absolute_risk"] is None
    assert hp_risk["unavailable_reason"] == "prevalence_unavailable"
    assert mondo is None
    row_mondo = analysis.filter(
        (pl.col("pgs_id") == "PGS000001") & pl.col("analysis_eligible")
    ).row(0, named=True)
    mondo_trait = next(item for item in row_mondo["traits"] if item["trait_id"] == "MONDO_0005148")
    mondo_risk = next(item for item in mondo_trait["population_risks"] if item["superpopulation"] == "EUR")
    assert mondo_risk["prevalence"] == pytest.approx(0.12)


def test_risk_and_h2_null_is_not_zero(tmp_path: Path) -> None:
    write_analysis_sources(tmp_path)
    analysis = build_model_analysis(tmp_path)
    row = analysis.filter(
        (pl.col("pgs_id") == "PGS000001")
        & (pl.col("sample_id") == "anton")
        & (pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
    ).row(0, named=True)
    trait = next(item for item in row["traits"] if item["trait_id"] == "EFO_0001360")
    eur = next(item for item in trait["population_risks"] if item["superpopulation"] == "EUR")
    expected = estimate_absolute_risk(z_score=float((12.0 - 20.0) / 2.0), prevalence=0.12, or_estimate=1.4, auroc_estimate=0.72)
    assert expected is not None
    assert eur["absolute_risk"] == pytest.approx(expected.absolute_risk)
    assert eur["risk_ratio"] != 0
    missing = analysis.filter(pl.col("pgs_id") == "PGS000N").row(0, named=True)
    for trait in missing["traits"]:
        for risk in trait["population_risks"]:
            assert risk["absolute_risk"] is None
            assert risk["risk_ratio"] is None


def test_summaries_exclude_ineligible_and_match_canonical_aggregation(tmp_path: Path) -> None:
    write_analysis_sources(tmp_path)
    analysis = build_model_analysis(tmp_path)
    summaries = build_trait_summaries(analysis)
    assert summaries.filter(pl.col("percentile_population").is_in(["CEU", "IBS"])).height == 0
    assert "family_id" not in summaries.columns
    assert summaries.filter(pl.col("pgs_id").is_in(["PGS000Q", "PGS000F"]) if "pgs_id" in summaries.columns else pl.lit(False)).height == 0
    issues = validate_trait_summaries(summaries, analysis)
    assert issues == []
    bmi = summaries.filter(
        (pl.col("sample_id") == "anton")
        & (pl.col("trait_id") == "EFO_0001360")
        & (pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
    ).row(0, named=True)
    assert bmi["percentile_population"] == "EUR"
    assert bmi["n_quarantined"] >= 1
    assert bmi["n_failed"] >= 1
    eligible_ids = set(
        analysis.filter(
            (pl.col("sample_id") == "anton")
            & (pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
            & pl.col("analysis_eligible")
        )["pgs_id"].to_list()
    )
    assert "PGS000Q" not in eligible_ids
    assert bmi["n_eligible"] == len(
        [
            row
            for row in analysis.filter(
                (pl.col("sample_id") == "anton")
                & (pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
                & pl.col("analysis_eligible")
            ).iter_rows(named=True)
            if any(trait["trait_id"] == "EFO_0001360" for trait in row["traits"] or [])
        ]
    )
    unrestored = summaries.filter(pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
    restored = summaries.filter(pl.col("score_profile_id") == RESTORED_PROFILE_ID)
    assert unrestored.height == restored.height
    paired = summaries.filter(
        (pl.col("sample_id") == "anton")
        & (pl.col("trait_id") == "EFO_0001360")
        & (pl.col("score_profile_id") == RESTORED_PROFILE_ID)
    ).row(0, named=True)
    other = summaries.filter(
        (pl.col("sample_id") == "anton")
        & (pl.col("trait_id") == "EFO_0001360")
        & (pl.col("score_profile_id") == UNRESTORED_PROFILE_ID)
    ).row(0, named=True)
    assert paired["paired_profile_id"] == UNRESTORED_PROFILE_ID
    assert paired["delta_median_pct"] == pytest.approx(paired["median_pct"] - other["median_pct"])
    assert any("not genotype imputation" in text for text in paired["caveats"])


def test_docs_keep_family_descriptive_and_restoration_disclaimer(tmp_path: Path) -> None:
    write_analysis_sources(tmp_path)
    analysis = build_model_analysis(tmp_path)
    summaries = build_trait_summaries(analysis)
    out = tmp_path / "sample_scores"
    write_final_docs(
        out,
        analysis=analysis,
        summaries=summaries,
        sources=[],
        scoring_set_fingerprint="s",
        sample_set_fingerprint="t",
        reference_universe_fingerprint="u",
        n_pgs_ids=6,
        pgs_without_distribution=["PGS000N", "PGS000Z"],
        omitted_catalog_tables=["score_development_ancestry.parquet"],
        parent_revision=SAMPLE_SCORES_REVISION,
    )
    readme = (out / "README.md").read_text(encoding="utf-8")
    agents = (out / "AGENTS.md").read_text(encoding="utf-8")
    analysis_guide = (out / "ANALYSIS.md").read_text(encoding="utf-8")
    assert "cannot estimate heritability" in readme
    assert "not genotype imputation" in readme
    assert "drop(\"source_revision\")" in readme or "drop('source_revision')" in readme
    assert "CEU and IBS are nearest" in readme
    assert "not an individual's" in readme.lower() or "not an individual" in readme
    assert "ANALYSIS.md" in readme
    assert "oksana" not in readme.lower()
    assert "Family concordance is descriptive only" in agents
    assert "oksana" not in agents.lower()
    assert "not genotype imputation" in analysis_guide
    assert "drop(\"source_revision\")" in analysis_guide or "drop('source_revision')" in analysis_guide
    assert "Family concordance is descriptive only" in analysis_guide
    assert "ANALYSIS.md" in analysis_guide
    assert "oksana" not in analysis_guide.lower()
    manifest = (out / "manifest.json").read_text(encoding="utf-8")
    assert "sha256" not in manifest.split("outputs")[0] or "file_hashes" not in manifest
    assert SAMPLE_SCORES_REVISION in manifest


def test_integration_commit_is_exactly_owned_paths(tmp_path: Path) -> None:
    for name in (
        "model_analysis.parquet",
        "trait_summaries.parquet",
        "manifest.json",
        "README.md",
        "AGENTS.md",
        "ANALYSIS.md",
    ):
        (tmp_path / name).write_text("ok\n", encoding="utf-8")
    (tmp_path / "identity_cache.json").write_text("{}\n", encoding="utf-8")
    ops = sample_score_integration_commit_operations(tmp_path)
    paths = [str(op.path_in_repo) for op in ops]
    assert paths == [
        "data/model_analysis.parquet",
        "data/trait_summaries.parquet",
        "data/manifest.json",
        "README.md",
        "AGENTS.md",
        "ANALYSIS.md",
    ]
    with pytest.raises(SampleScorePublishError, match="forbidden"):
        sample_score_commit_operations(tmp_path, ("README.md",))
    with pytest.raises(SampleScorePublishError, match="forbidden"):
        sample_score_commit_operations(tmp_path, ("AGENTS.md",))
    with pytest.raises(SampleScorePublishError, match="forbidden"):
        sample_score_commit_operations(tmp_path, ("ANALYSIS.md",))
    with pytest.raises(SampleScorePublishError, match="forbidden"):
        sample_score_commit_operations(tmp_path, ("identity_cache.json",))
