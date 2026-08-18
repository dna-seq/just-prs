"""Publish public-genome PRS from the canary/pipeline job into the HF dataset.

Canary quarantine stays on its own thin ``canary_scores.parquet``. This module
scores registered public samples under the two published WGS profiles and
writes full ``RuntimeResultRow`` records. Thin canary rows are never converted
into runtime hits. Unknown or ``publication_allowed=False`` genomes may be
used for canary flags, but are never written to the published store.
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from eliot import start_action
from pydantic import BaseModel, Field

from just_prs.canary_audit import CanarySample
from just_prs.models import PRSResult
from just_prs.normalize import normalize_vcf
from just_prs.sample_scores.fingerprints import reference_universe_fingerprint
from just_prs.sample_scores.identity import (
    cached_genotype_sha256,
    cached_source_sha256,
)
from just_prs.sample_scores.models import (
    DEFAULT_SAMPLE_SCORES_REPO,
    HASH_SCHEMA_VERSION,
    NORMALIZATION_PROFILE_ID,
    PUBLIC_WGS_PASS_V1,
    SCORE_ALGORITHM_VERSION,
    SCORE_PROFILES,
    RuntimeResultRow,
    SampleRecord,
    ScoreProfile,
)
from just_prs.sample_scores.store import (
    RUNTIME_MANIFEST_FILENAME,
    load_runtime_results,
    load_samples,
    sample_scores_dir,
    write_runtime_results,
    write_samples,
)
from just_prs.scoring import resolve_cache_dir

RUNTIME_KEY = (
    "sample_id",
    "pgs_id",
    "scoring_build",
    "score_profile_id",
    "scoring_fingerprint",
)

LEGACY_SAMPLE_IDS: dict[str, str] = {
    "oksana": "o-mom",
}


class PublicSampleSpec(BaseModel):
    """Consent and license for a genome that may be published."""

    sample_id: str
    aliases: list[str] = Field(default_factory=list)
    display_name: str
    family_id: str | None = None
    relationship_role: str | None = None
    license: str
    publication_allowed: bool
    consent_basis: str
    source_url: str | None = None
    genome_build: str = "GRCh38"


def _spec(
    sample_id: str,
    *,
    display_name: str,
    license: str,
    publication_allowed: bool,
    consent_basis: str,
    aliases: list[str] | None = None,
    family_id: str | None = None,
    relationship_role: str | None = None,
    source_url: str | None = None,
) -> PublicSampleSpec:
    return PublicSampleSpec(
        sample_id=sample_id,
        aliases=aliases or [],
        display_name=display_name,
        family_id=family_id,
        relationship_role=relationship_role,
        license=license,
        publication_allowed=publication_allowed,
        consent_basis=consent_basis,
        source_url=source_url,
    )


PUBLIC_SAMPLE_SPECS: dict[str, PublicSampleSpec] = {
    "anton": _spec(
        "anton",
        aliases=["Anton", "antonkulaga"],
        display_name="Anton Kulaga",
        license="CC0-1.0",
        publication_allowed=True,
        consent_basis="public-domain-zenodo",
        source_url="https://zenodo.org/records/18370498",
    ),
    "livia": _spec(
        "livia",
        aliases=["Livia"],
        display_name="Livia Zaharia",
        license="CC-BY-4.0",
        publication_allowed=True,
        consent_basis="cc-by-4.0-zenodo",
        source_url="https://zenodo.org/records/19487816",
    ),
    "o-mom": _spec(
        "o-mom",
        aliases=["oksana", "mom", "o-mother"],
        display_name="o-mom",
        family_id="o-family",
        relationship_role="mother",
        license="CC-BY-4.0",
        publication_allowed=True,
        consent_basis="owner-authorized-derived-prs",
    ),
    "o-dad": _spec(
        "o-dad",
        aliases=["dad", "o-father"],
        display_name="o-dad",
        family_id="o-family",
        relationship_role="father",
        license="CC-BY-4.0",
        publication_allowed=True,
        consent_basis="owner-authorized-derived-prs",
    ),
    "o-son1": _spec(
        "o-son1",
        aliases=["son1"],
        display_name="o-son1",
        family_id="o-family",
        relationship_role="son",
        license="CC-BY-4.0",
        publication_allowed=True,
        consent_basis="owner-authorized-derived-prs",
    ),
    "o-son2": _spec(
        "o-son2",
        aliases=["son2"],
        display_name="o-son2",
        family_id="o-family",
        relationship_role="son",
        license="CC-BY-4.0",
        publication_allowed=True,
        consent_basis="owner-authorized-derived-prs",
    ),
    "o-daughter": _spec(
        "o-daughter",
        aliases=["daughter"],
        display_name="o-daughter",
        family_id="o-family",
        relationship_role="daughter",
        license="CC-BY-4.0",
        publication_allowed=True,
        consent_basis="owner-authorized-derived-prs",
    ),
}


@dataclass
class SampleScoreProgress:
    """Incremental publication counters for Dagster metadata."""

    n_total: int = 0
    n_ok: int = 0
    n_cached: int = 0
    n_failed: int = 0
    n_skipped_private: int = 0
    published_sample_ids: list[str] = field(default_factory=list)
    skipped_labels: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    recycle_reasons: list[str] = field(default_factory=list)


def canonical_sample_id(label: str) -> str:
    """Map a CLI/canary label to the public ``sample_id``.

    ``oksana`` is the only legacy rename; it becomes ``o-mom``. Unknown labels
    stay as-is and are not published unless a matching spec allows it.
    """
    key = label.strip().casefold()
    if key in LEGACY_SAMPLE_IDS:
        return LEGACY_SAMPLE_IDS[key]
    for spec in PUBLIC_SAMPLE_SPECS.values():
        if spec.sample_id.casefold() == key:
            return spec.sample_id
        if key in {alias.casefold() for alias in spec.aliases}:
            return spec.sample_id
    return label.strip()


def public_sample_spec(label: str) -> PublicSampleSpec | None:
    sample_id = canonical_sample_id(label)
    return PUBLIC_SAMPLE_SPECS.get(sample_id)


def is_publication_allowed(label: str) -> bool:
    spec = public_sample_spec(label)
    return bool(spec is not None and spec.publication_allowed)


def _plain_vcf_from_gzip(vcf_path: Path, dest: Path) -> Path:
    """Decompress a gzip/BGZF VCF so polars-bio can scan it as plain text."""
    if dest.exists() and dest.stat().st_mtime >= vcf_path.stat().st_mtime:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with gzip.open(vcf_path, "rb") as incoming, tmp.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    tmp.replace(dest)
    return dest


def normalized_public_parquet(cache_dir: Path, sample_id: str, vcf_path: Path) -> Path:
    """Normalize a public WGS VCF with ``public-wgs-pass-v1`` (PASS / .)."""
    out_dir = cache_dir / "normalized" / "sample_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}.parquet"
    if out_path.exists() and out_path.stat().st_mtime >= vcf_path.stat().st_mtime:
        return out_path
    config = PUBLIC_WGS_PASS_V1.to_filter_config()
    try:
        return normalize_vcf(vcf_path, out_path, config=config)
    except pl.exceptions.ComputeError as exc:
        if "invalid BGZF" not in str(exc):
            raise
        plain = _plain_vcf_from_gzip(
            vcf_path,
            out_dir / f"{sample_id}.plain.vcf",
        )
        return normalize_vcf(plain, out_path, config=config)


def build_sample_record(
    sample: CanarySample,
    cache_dir: Path,
    *,
    spec: PublicSampleSpec | None = None,
) -> SampleRecord | None:
    """Hash identity for a registered public sample. Returns None if unpublished."""
    resolved = spec or public_sample_spec(sample.label)
    if resolved is None or not resolved.publication_allowed:
        return None
    parquet = normalized_public_parquet(cache_dir, resolved.sample_id, sample.vcf_path)
    genotypes = pl.scan_parquet(parquet)
    n_variants = int(genotypes.select(pl.len()).collect().item())
    return SampleRecord(
        sample_id=resolved.sample_id,
        aliases=list(resolved.aliases),
        display_name=resolved.display_name,
        family_id=resolved.family_id,
        relationship_role=resolved.relationship_role,
        license=resolved.license,
        publication_allowed=resolved.publication_allowed,
        consent_basis=resolved.consent_basis,
        source_url=resolved.source_url,
        genome_build=resolved.genome_build,
        n_variants=n_variants,
        source_sha256=cached_source_sha256(sample.vcf_path, cache_dir),
        genotype_sha256_v1=cached_genotype_sha256(genotypes, parquet, cache_dir),
        normalization_profile_id=NORMALIZATION_PROFILE_ID,
        hash_schema_version=HASH_SCHEMA_VERSION,
    )


def runtime_row_from_result(
    result: PRSResult,
    *,
    sample: SampleRecord,
    profile: ScoreProfile,
    scoring_fingerprint_value: str,
    reference_universe_fp: str | None,
    just_prs_version: str,
    computed_at: str,
    status: str = "ok",
    error: str | None = None,
) -> RuntimeResultRow:
    """Project a live ``PRSResult`` into a publishable runtime row."""
    return RuntimeResultRow(
        sample_id=sample.sample_id,
        pgs_id=result.pgs_id,
        scoring_build=profile.genome_build,
        score_profile_id=profile.score_profile_id,
        scoring_fingerprint=scoring_fingerprint_value,
        status=status,
        error=error,
        score=result.score if status == "ok" else None,
        variants_matched=result.variants_matched,
        variants_total=result.variants_total,
        match_rate=result.match_rate,
        variants_observed=result.variants_observed,
        variants_assumed_hom_ref=result.variants_assumed_hom_ref,
        variants_unscorable_absent=result.variants_unscorable_absent,
        variants_no_call=result.variants_no_call,
        variants_maf_filled=result.variants_maf_filled,
        variants_ref_resolved_panel=result.variants_ref_resolved_panel,
        variants_ref_resolved_fasta=result.variants_ref_resolved_fasta,
        weight_mass_matched=result.weight_mass_matched,
        weight_mass_total=result.weight_mass_total,
        weight_mass_coverage=result.weight_mass_coverage,
        genotype_input_mode=result.genotype_input_mode,
        detected_genome_build=result.detected_genome_build,
        build_mismatch=result.build_mismatch,
        trait_reported=result.trait_reported,
        has_allele_frequencies=result.has_allele_frequencies,
        theoretical_mean=result.theoretical_mean,
        theoretical_std=result.theoretical_std,
        score_algorithm_version=SCORE_ALGORITHM_VERSION,
        reference_universe_fingerprint=reference_universe_fp,
        sample_genotype_sha256=sample.genotype_sha256_v1,
        just_prs_version=just_prs_version,
        computed_at=computed_at,
    )


def failed_runtime_row(
    *,
    sample: SampleRecord,
    pgs_id: str,
    profile: ScoreProfile,
    scoring_fingerprint_value: str,
    reference_universe_fp: str | None,
    just_prs_version: str,
    computed_at: str,
    error: str,
) -> RuntimeResultRow:
    return RuntimeResultRow(
        sample_id=sample.sample_id,
        pgs_id=pgs_id,
        scoring_build=profile.genome_build,
        score_profile_id=profile.score_profile_id,
        scoring_fingerprint=scoring_fingerprint_value,
        status="failed",
        error=error,
        reference_universe_fingerprint=reference_universe_fp,
        sample_genotype_sha256=sample.genotype_sha256_v1,
        just_prs_version=just_prs_version,
        computed_at=computed_at,
    )


def upsert_runtime_results(
    rows: list[RuntimeResultRow] | pl.DataFrame,
    cache_dir: Path | None = None,
) -> Path:
    """Append/replace runtime rows keyed by the published primary key."""
    incoming = (
        rows
        if isinstance(rows, pl.DataFrame)
        else pl.DataFrame([row.model_dump() for row in rows])
    )
    existing = load_runtime_results(cache_dir)
    if existing.is_empty() and incoming.is_empty():
        return write_runtime_results(incoming, cache_dir)
    combined = pl.concat([existing, incoming], how="diagonal_relaxed")
    if combined.height:
        combined = combined.unique(subset=list(RUNTIME_KEY), keep="last")
    return write_runtime_results(combined, cache_dir)


def merge_sample_records(
    incoming: list[SampleRecord],
    cache_dir: Path | None = None,
) -> list[SampleRecord]:
    by_id = {sample.sample_id: sample for sample in load_samples(cache_dir)}
    for sample in incoming:
        by_id[sample.sample_id] = sample
    records = list(by_id.values())
    write_samples(records, cache_dir)
    return records


def write_sample_scores_manifest(
    cache_dir: Path,
    *,
    repo_id: str = DEFAULT_SAMPLE_SCORES_REPO,
    reference_universe_fp: str | None = None,
    extra: dict[str, object] | None = None,
) -> Path:
    from just_prs import __version__

    samples = load_samples(cache_dir)
    results = load_runtime_results(cache_dir)
    n_ok = int(results.filter(pl.col("status") == "ok").height) if results.height else 0
    n_failed = int(results.filter(pl.col("status") != "ok").height) if results.height else 0
    payload: dict[str, object] = {
        "schema_version": 1,
        "repo_id": repo_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "just_prs_version": __version__,
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "score_algorithm_version": SCORE_ALGORITHM_VERSION,
        "hash_schema_version": HASH_SCHEMA_VERSION,
        "score_profiles": [profile.model_dump() for profile in SCORE_PROFILES.values()],
        "sample_ids": [sample.sample_id for sample in samples if sample.publication_allowed],
        "n_samples": sum(1 for sample in samples if sample.publication_allowed),
        "n_runtime_rows": results.height,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "reference_universe_fingerprint": reference_universe_fp,
        "evidence_tables": [],
        "note": (
            "Runtime scores only. Evidence, trait summaries, guidelines, and "
            "root README/AGENTS.md are a later publish onto this same repo. "
            "This file is runtime_manifest.json, not the combined manifest.json."
        ),
    }
    if extra:
        payload.update(extra)
    path = sample_scores_dir(cache_dir) / RUNTIME_MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def score_public_sample_catalog(
    samples: list[CanarySample],
    cache_dir: Path | None = None,
    *,
    pgs_ids: list[str] | None = None,
    limit: int | None = None,
    skip_existing: bool = True,
    retry_failed: bool = False,
    progress_every: int = 10,
    log: Callable[[str], None] | None = None,
) -> SampleScoreProgress:
    """Score publication-allowed samples under both WGS profiles (PGS-major).

    Writes atomic checkpoint parts. Compaction into ``runtime_results.parquet``
    is a separate step (``compact_public_sample_runtime``). Private or unknown
    labels are counted as skipped and never uploaded. ``progress_every``
    is the PGS-count interval for worker progress lines (default 10).
    ``retry_failed`` reopens failed parts and still continues into uncached IDs.
    """
    from just_prs.sample_scores.engine import score_public_samples_pgs_major

    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    return score_public_samples_pgs_major(
        samples,
        root,
        pgs_ids=pgs_ids,
        limit=limit,
        skip_existing=skip_existing,
        retry_failed=retry_failed,
        progress_every=progress_every,
        log=log,
    )


def compact_public_sample_runtime(
    cache_dir: Path | None = None,
    *,
    expected_keys: set[str] | None = None,
    repo_id: str = DEFAULT_SAMPLE_SCORES_REPO,
) -> Path:
    """Compact checkpoint parts once and write ``runtime_manifest.json``."""
    from just_prs.sample_scores.checkpoints import compact_runtime_parts
    from just_prs.sample_scores.fingerprints import reference_universe_fingerprint
    from just_prs.prs_catalog import PRSCatalog

    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    dest = compact_runtime_parts(root, expected_keys=expected_keys)
    universe_fp: str | None = None
    catalog = PRSCatalog(cache_dir=root)
    universe_path = catalog._reference_universe_path("GRCh38")
    if universe_path is not None and universe_path.exists():
        universe_fp = reference_universe_fingerprint(universe_path)
    write_sample_scores_manifest(root, repo_id=repo_id, reference_universe_fp=universe_fp)
    return dest


def publish_public_sample_scores(
    samples: list[CanarySample],
    cache_dir: Path | None = None,
    *,
    pgs_ids: list[str] | None = None,
    limit: int | None = None,
    skip_existing: bool = True,
    retry_failed: bool = False,
    progress_every: int = 10,
    log: Callable[[str], None] | None = None,
    push: bool = False,
    repo_id: str = DEFAULT_SAMPLE_SCORES_REPO,
    token: str | None = None,
) -> SampleScoreProgress:
    """Score publication-allowed samples and optionally push the dataset to HF."""
    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    with start_action(action_type="sample_scores:publish", repo_id=repo_id):
        progress = score_public_sample_catalog(
            samples,
            root,
            pgs_ids=pgs_ids,
            limit=limit,
            skip_existing=skip_existing,
            retry_failed=retry_failed,
            progress_every=progress_every,
            log=log,
        )
        if progress.published_sample_ids:
            compact_public_sample_runtime(root, repo_id=repo_id)
        if push and progress.published_sample_ids:
            from just_prs.hf import push_sample_score_runtime

            push_sample_score_runtime(sample_scores_dir(root), repo_id=repo_id, token=token)
        return progress
