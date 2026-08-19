"""Identity, profile matching, and precomputed PRS lookup."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from just_prs.models import PRSResult
from just_prs.sample_scores import (
    RESTORED_PROFILE_ID,
    UNRESTORED_PROFILE_ID,
    ComputationSource,
    PrecomputedMiss,
    PrecomputedPolicy,
    RuntimeResultRow,
    SampleRecord,
    canonical_sample_id,
    genotype_sha256_v1,
    is_publication_allowed,
    lookup_precomputed_prs,
    match_score_profile,
    resolve_official_prs,
    resolve_sample,
    runtime_row_from_result,
    scoring_fingerprint,
    source_sha256,
    upsert_runtime_results,
    write_runtime_results,
    write_samples,
)
from just_prs.sample_scores.models import SCORE_PROFILES
from just_prs.sample_scores.models import PRIVATE_INGEST_ALIASES, published_aliases
from just_prs.sample_scores.publish import PUBLIC_SAMPLE_SPECS


def _genotypes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["2", "1", "1"],
            "pos": [20, 10, 11],
            "ref": ["G", "A", "C"],
            "alt": ["T", "T", "G"],
            "genotype": [["G", "T"], ["A", "T"], ["C", "C"]],
        }
    )


def _sample(source_digest: str, genotype_digest: str) -> SampleRecord:
    return SampleRecord(
        sample_id="anton",
        aliases=["Anton"],
        display_name="Anton",
        license="CC0",
        publication_allowed=True,
        consent_basis="public-domain",
        source_sha256=source_digest,
        genotype_sha256_v1=genotype_digest,
    )


def _runtime_row(sample: SampleRecord, fingerprint: str = "fp1") -> RuntimeResultRow:
    return RuntimeResultRow(
        sample_id=sample.sample_id,
        pgs_id="PGS000001",
        scoring_build="GRCh38",
        score_profile_id=UNRESTORED_PROFILE_ID,
        scoring_fingerprint=fingerprint,
        score=0.25,
        variants_matched=10,
        variants_total=12,
        match_rate=10 / 12,
        sample_genotype_sha256=sample.genotype_sha256_v1,
        trait_reported="body mass index",
    )


def test_genotype_hash_is_order_independent() -> None:
    first = genotype_sha256_v1(_genotypes())
    shuffled = _genotypes().select(pl.all().shuffle(seed=1))
    assert genotype_sha256_v1(shuffled) == first


def test_genotype_hash_changes_when_a_call_changes() -> None:
    original = genotype_sha256_v1(_genotypes())
    changed = _genotypes().with_columns(
        pl.when(pl.col("pos") == 10)
        .then(pl.lit(["T", "T"]))
        .otherwise(pl.col("genotype"))
        .alias("genotype")
    )
    assert genotype_sha256_v1(changed) != original


def test_source_hash_hits_copied_file(tmp_path: Path) -> None:
    src = tmp_path / "a.vcf"
    copy = tmp_path / "renamed.vcf"
    src.write_bytes(b"##fileformat=VCFv4.2\n")
    copy.write_bytes(src.read_bytes())
    assert source_sha256(src) == source_sha256(copy)


def test_private_ingest_alias_resolves_locally_without_published_alias() -> None:
    geno = _genotypes()
    digest = genotype_sha256_v1(geno)
    sample = SampleRecord(
        sample_id="o-mom",
        aliases=published_aliases(["oksana", "mom", "o-mother"]),
        display_name="o-mom",
        license="CC-BY-4.0",
        publication_allowed=True,
        consent_basis="owner-authorized-derived-prs",
        source_sha256="e" * 64,
        genotype_sha256_v1=digest,
    )
    assert "oksana" not in sample.aliases
    assert resolve_sample([sample], genotypes=geno, alias="oksana") is not None


def test_alias_does_not_prove_identity(tmp_path: Path) -> None:
    geno = _genotypes()
    digest = genotype_sha256_v1(geno)
    other = tmp_path / "other.vcf"
    other.write_bytes(b"not-anton")
    sample = _sample(source_sha256(other), digest)
    sample.source_sha256 = "aaa" + "b" * 61
    assert resolve_sample([sample], source_path=other, alias="anton") is None
    assert resolve_sample([sample], genotypes=geno, alias="anton") is not None


def test_mislabeled_alias_misses_without_hash_match(tmp_path: Path) -> None:
    vcf = tmp_path / "livia.vcf"
    vcf.write_bytes(b"livia-bytes")
    sample = _sample("c" * 64, "d" * 64)
    assert resolve_sample([sample], source_path=vcf, alias="anton") is None


def test_match_score_profile_wgs_only() -> None:
    unrestored = match_score_profile(
        genome_build="GRCh38",
        reference_restoration=False,
        genotype_input_mode="auto",
    )
    restored = match_score_profile(
        genome_build="GRCh38",
        reference_restoration=True,
        genotype_input_mode="variant_only",
    )
    assert unrestored is not None
    assert unrestored.score_profile_id == UNRESTORED_PROFILE_ID
    assert restored is not None
    assert restored.reference_restoration is True
    assert match_score_profile(
        genome_build="GRCh38",
        reference_restoration=False,
        genotype_input_mode="all_sites",
    ) is None
    assert match_score_profile(
        genome_build="GRCh37",
        reference_restoration=False,
    ) is None


def test_scoring_file_fingerprint_is_bytes_not_rows(tmp_path: Path) -> None:
    from just_prs.sample_scores.fingerprints import scoring_file_fingerprint

    path = tmp_path / "PGS000001_hmPOS_GRCh38.parquet"
    pl.DataFrame({
        "hm_chr": ["1"],
        "hm_pos": [10],
        "effect_allele": ["A"],
        "effect_weight": [0.1],
    }).write_parquet(path)
    first = scoring_file_fingerprint(path)
    again = scoring_file_fingerprint(path)
    assert first == again
    assert len(first) == 64
    path.write_bytes(path.read_bytes() + b"\n")
    assert scoring_file_fingerprint(path) != first


def test_scoring_fingerprint_is_logical(tmp_path: Path) -> None:
    rows = {
        "hm_chr": ["1", "2"],
        "hm_pos": [10, 20],
        "effect_allele": ["A", "C"],
        "effect_weight": [0.1, -0.2],
        "other_allele": ["G", "T"],
    }
    first = scoring_fingerprint(pl.DataFrame(rows))
    shuffled = pl.DataFrame(rows).reverse()
    assert scoring_fingerprint(shuffled) == first
    changed = pl.DataFrame(rows).with_columns(pl.col("effect_weight") + 0.01)
    assert scoring_fingerprint(changed) != first


def test_lookup_hit_and_profile_miss(tmp_path: Path) -> None:
    vcf = tmp_path / "anton.vcf"
    vcf.write_bytes(b"anton-source")
    geno = _genotypes()
    sample = _sample(source_sha256(vcf), genotype_sha256_v1(geno))
    write_samples([sample], tmp_path)
    write_runtime_results([_runtime_row(sample)], tmp_path)

    hit = lookup_precomputed_prs(
        pgs_id="PGS000001",
        vcf_path=vcf,
        alias="anton",
        cache_dir=tmp_path,
        pull=False,
    )
    assert hit is not None
    assert hit.score == 0.25
    assert hit.computation_source == ComputationSource.PRECOMPUTED.value
    assert hit.score_profile_id == UNRESTORED_PROFILE_ID

    miss = lookup_precomputed_prs(
        pgs_id="PGS000001",
        vcf_path=vcf,
        reference_restoration=True,
        cache_dir=tmp_path,
        pull=False,
    )
    assert miss is None


def test_lookup_misses_unknown_genome(tmp_path: Path) -> None:
    vcf = tmp_path / "stranger.vcf"
    vcf.write_bytes(b"unknown")
    sample = _sample("e" * 64, "f" * 64)
    write_samples([sample], tmp_path)
    write_runtime_results([_runtime_row(sample)], tmp_path)
    assert lookup_precomputed_prs(
        pgs_id="PGS000001",
        vcf_path=vcf,
        cache_dir=tmp_path,
        pull=False,
    ) is None


def test_fingerprint_mismatch_misses(tmp_path: Path) -> None:
    vcf = tmp_path / "anton.vcf"
    vcf.write_bytes(b"anton-source")
    geno = _genotypes()
    sample = _sample(source_sha256(vcf), genotype_sha256_v1(geno))
    write_samples([sample], tmp_path)
    write_runtime_results([_runtime_row(sample, fingerprint="published")], tmp_path)
    scores = tmp_path / "scores"
    scores.mkdir()
    pl.DataFrame(
        {
            "hm_chr": ["1"],
            "hm_pos": [1],
            "effect_allele": ["A"],
            "effect_weight": [0.2],
        }
    ).write_parquet(scores / "PGS000001_hmPOS_GRCh38.parquet")
    assert lookup_precomputed_prs(
        pgs_id="PGS000001",
        vcf_path=vcf,
        cache_dir=tmp_path,
        scores_cache=scores,
        pull=False,
    ) is None


def test_resolve_official_prs_policy(tmp_path: Path) -> None:
    vcf = tmp_path / "anton.vcf"
    vcf.write_bytes(b"anton-source")
    geno = _genotypes()
    sample = _sample(source_sha256(vcf), genotype_sha256_v1(geno))
    write_samples([sample], tmp_path)
    write_runtime_results([_runtime_row(sample)], tmp_path)

    computed = PRSResult(
        pgs_id="PGS000001",
        score=9.0,
        variants_matched=1,
        variants_total=1,
        match_rate=1.0,
    )

    auto = resolve_official_prs(
        policy=PrecomputedPolicy.AUTO,
        pgs_id="PGS000001",
        vcf_path=vcf,
        cache_dir=tmp_path,
        compute=lambda: computed,
    )
    assert auto.score == 0.25

    forced = resolve_official_prs(
        policy=PrecomputedPolicy.OFF,
        pgs_id="PGS000001",
        vcf_path=vcf,
        cache_dir=tmp_path,
        compute=lambda: computed,
    )
    assert forced.score == 9.0
    assert forced.computation_source == ComputationSource.COMPUTED.value

    with pytest.raises(PrecomputedMiss):
        resolve_official_prs(
            policy=PrecomputedPolicy.REQUIRE,
            pgs_id="PGS999999",
            vcf_path=vcf,
            cache_dir=tmp_path,
            compute=lambda: computed,
        )


def test_canonical_sample_id_maps_oksana_only() -> None:
    assert canonical_sample_id("oksana") == "o-mom"
    assert canonical_sample_id("Anton") == "anton"
    assert canonical_sample_id("mom") == "o-mom"
    assert canonical_sample_id("stranger") == "stranger"


def test_private_and_unknown_labels_are_not_published() -> None:
    assert is_publication_allowed("anton") is True
    assert is_publication_allowed("livia") is True
    assert is_publication_allowed("oksana") is True
    assert is_publication_allowed("o-mom") is True
    assert is_publication_allowed("o-dad") is True
    assert is_publication_allowed("o-son1") is True
    assert is_publication_allowed("o-son2") is True
    assert is_publication_allowed("o-daughter") is True
    assert is_publication_allowed("stranger") is False
    assert PUBLIC_SAMPLE_SPECS["o-mom"].publication_allowed is True
    assert "oksana" in PUBLIC_SAMPLE_SPECS["o-mom"].aliases
    assert "oksana" not in published_aliases(PUBLIC_SAMPLE_SPECS["o-mom"].aliases)
    assert "oksana" in PRIVATE_INGEST_ALIASES


def test_runtime_row_from_result_keeps_coverage_counters() -> None:
    sample = _sample("a" * 64, "b" * 64)
    result = PRSResult(
        pgs_id="PGS000001",
        score=0.5,
        variants_matched=8,
        variants_total=10,
        match_rate=0.8,
        variants_observed=7,
        variants_assumed_hom_ref=1,
        variants_unscorable_absent=2,
        variants_no_call=0,
        weight_mass_matched=1.5,
        weight_mass_total=2.0,
        weight_mass_coverage=0.75,
        genotype_input_mode="variant_only",
    )
    row = runtime_row_from_result(
        result,
        sample=sample,
        profile=SCORE_PROFILES[UNRESTORED_PROFILE_ID],
        scoring_fingerprint_value="fp1",
        reference_universe_fp=None,
        just_prs_version="0.0.0",
        computed_at="2026-08-16T00:00:00+00:00",
    )
    assert row.variants_unscorable_absent == 2
    assert row.variants_assumed_hom_ref == 1
    assert row.weight_mass_coverage == 0.75
    assert row.sample_genotype_sha256 == sample.genotype_sha256_v1


def test_upsert_runtime_results_replaces_same_key(tmp_path: Path) -> None:
    sample = _sample("a" * 64, "b" * 64)
    first = _runtime_row(sample, fingerprint="fp1")
    first.score = 0.1
    upsert_runtime_results([first], tmp_path)
    second = _runtime_row(sample, fingerprint="fp1")
    second.score = 0.9
    upsert_runtime_results([second], tmp_path)
    other = _runtime_row(sample, fingerprint="fp2")
    other.pgs_id = "PGS000002"
    upsert_runtime_results([other], tmp_path)
    frame = pl.read_parquet(tmp_path / "sample_scores" / "runtime_results.parquet")
    assert frame.height == 2
    kept = frame.filter(pl.col("pgs_id") == "PGS000001")
    assert kept["score"][0] == 0.9


def test_lookup_skips_quarantined_pgs(tmp_path: Path) -> None:
    vcf = tmp_path / "anton.vcf"
    vcf.write_bytes(b"anton-source")
    geno = _genotypes()
    sample = _sample(source_sha256(vcf), genotype_sha256_v1(geno))
    write_samples([sample], tmp_path)
    write_runtime_results([_runtime_row(sample)], tmp_path)
    flags = tmp_path / "metadata"
    flags.mkdir()
    pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "flag": ["canary_collapsed_percentile"],
            "severity": ["ERROR"],
            "exclude_from_catalog": [True],
            "n_canary_samples": [2],
            "n_extreme": [2],
            "min_percentile": [0.0],
            "max_percentile": [0.0],
            "median_abs_z": [8.0],
            "median_match_rate": [0.2],
            "reason": ["collapsed"],
        }
    ).write_parquet(flags / "catalog_scoring_flags.parquet")
    assert lookup_precomputed_prs(
        pgs_id="PGS000001",
        vcf_path=vcf,
        cache_dir=tmp_path,
        pull=False,
    ) is None


def test_lookup_hits_both_profiles_when_fingerprints_match(tmp_path: Path) -> None:
    vcf = tmp_path / "anton.vcf"
    vcf.write_bytes(b"anton-source")
    geno = _genotypes()
    sample = _sample(source_sha256(vcf), genotype_sha256_v1(geno))
    write_samples([sample], tmp_path)
    unrestored = _runtime_row(sample, "fp-current")
    restored = unrestored.model_copy(
        update={"score_profile_id": RESTORED_PROFILE_ID, "score": 0.5}
    )
    write_runtime_results([unrestored, restored], tmp_path)
    hit_off = lookup_precomputed_prs(
        pgs_id="PGS000001",
        vcf_path=vcf,
        reference_restoration=False,
        cache_dir=tmp_path,
        pull=False,
    )
    hit_on = lookup_precomputed_prs(
        pgs_id="PGS000001",
        vcf_path=vcf,
        reference_restoration=True,
        cache_dir=tmp_path,
        pull=False,
    )
    assert hit_off is not None and hit_off.score == 0.25
    assert hit_on is not None and hit_on.score == 0.5
