"""High-level precomputed PRS lookup. Low-level engines stay network-free."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
from eliot import log_message, start_action

from just_prs.models import PRSResult
from just_prs.sample_scores.fingerprints import scoring_file_fingerprint
from just_prs.sample_scores.identity import resolve_sample
from just_prs.sample_scores.models import (
    DEFAULT_SAMPLE_SCORES_REPO,
    ComputationSource,
    PrecomputedPolicy,
    ScoreProfile,
    match_score_profile,
)
from just_prs.sample_scores.store import (
    ensure_local_dataset,
    load_samples,
    lookup_runtime_row,
)
from just_prs.scoring import parquet_cache_is_readable, resolve_cache_dir, scoring_parquet_path


class PrecomputedMiss(Exception):
    """Raised when ``PrecomputedPolicy.REQUIRE`` cannot return a published result."""


def _restoration_enabled(reference_restoration: object) -> bool:
    """WGS restoration only. A Chip scope has no published profile yet."""
    return reference_restoration is True


def _optional_scoring_fingerprint(
    pgs_id: str,
    genome_build: str,
    scores_cache: Path,
) -> str | None:
    path = scoring_parquet_path(pgs_id, scores_cache, genome_build)
    if not parquet_cache_is_readable(path):
        return None
    return scoring_file_fingerprint(path)


def lookup_precomputed_prs(
    *,
    pgs_id: str,
    genome_build: str = "GRCh38",
    vcf_path: Path | str | None = None,
    genotypes_lf: pl.LazyFrame | None = None,
    genotypes_parquet: str | Path | None = None,
    alias: str | None = None,
    reference_restoration: object = False,
    genotype_input_mode: str = "auto",
    maf_fill: bool = False,
    cache_dir: Path | None = None,
    scores_cache: Path | None = None,
    repo_id: str = DEFAULT_SAMPLE_SCORES_REPO,
    pull: bool = True,
) -> PRSResult | None:
    """Return a published ``PRSResult`` when identity, profile, and provenance match.

    Returns None on any miss (unknown genome, profile mismatch, fingerprint
    mismatch, missing local/HF dataset). Never writes an unknown genome.
    """
    root = cache_dir if cache_dir is not None else resolve_cache_dir()
    repo_id = os.environ.get("PRS_SAMPLE_SCORES_REPO") or repo_id
    profile: ScoreProfile | None = match_score_profile(
        genome_build=genome_build,
        reference_restoration=_restoration_enabled(reference_restoration),
        genotype_input_mode=genotype_input_mode,
        maf_fill=maf_fill,
    )
    if profile is None:
        return None

    with start_action(
        action_type="sample_scores:lookup",
        pgs_id=pgs_id,
        score_profile_id=profile.score_profile_id,
        alias=alias,
    ):
        try:
            ensure_local_dataset(root, repo_id=repo_id, pull=pull)
        except Exception as exc:
            log_message(
                message_type="sample_scores:dataset_unavailable",
                error=str(exc),
            )
            return None

        samples = load_samples(root)
        if not samples:
            return None

        source = Path(vcf_path) if vcf_path else None
        if source is not None and source.suffix == ".parquet":
            source = None
        genotypes: pl.LazyFrame | None = genotypes_lf
        if genotypes is None and genotypes_parquet:
            parquet = Path(genotypes_parquet)
            if parquet_cache_is_readable(parquet):
                genotypes = pl.scan_parquet(parquet)
        elif source is None and vcf_path and str(vcf_path).endswith(".parquet"):
            parquet = Path(vcf_path)
            if parquet_cache_is_readable(parquet):
                genotypes = pl.scan_parquet(parquet)

        sample = resolve_sample(
            samples,
            source_path=source if source is not None and source.exists() else None,
            genotypes=genotypes,
            alias=alias,
            cache_dir=root,
        )
        if sample is None or not sample.publication_allowed:
            return None

        from just_prs.canary_audit import catalog_flags_path, excluded_pgs_ids

        flags_path = catalog_flags_path(root)
        if parquet_cache_is_readable(flags_path):
            excluded = set(excluded_pgs_ids(pl.read_parquet(flags_path)))
            if pgs_id.upper() in excluded:
                return None

        fingerprint = _optional_scoring_fingerprint(
            pgs_id,
            genome_build,
            scores_cache if scores_cache is not None else root / "scores",
        )
        row = lookup_runtime_row(
            sample_id=sample.sample_id,
            pgs_id=pgs_id,
            scoring_build=genome_build,
            score_profile_id=profile.score_profile_id,
            scoring_fingerprint=fingerprint,
            cache_dir=root,
        )
        if row is None:
            return None
        if row.sample_genotype_sha256 != sample.genotype_sha256_v1:
            return None
        return row.to_prs_result(repo_id=repo_id)


def resolve_official_prs(
    *,
    policy: PrecomputedPolicy | str,
    pgs_id: str,
    compute,
    genome_build: str = "GRCh38",
    vcf_path: Path | str | None = None,
    genotypes_lf: pl.LazyFrame | None = None,
    genotypes_parquet: str | Path | None = None,
    alias: str | None = None,
    reference_restoration: object = False,
    genotype_input_mode: str = "auto",
    maf_fill: bool = False,
    cache_dir: Path | None = None,
    scores_cache: Path | None = None,
    repo_id: str = DEFAULT_SAMPLE_SCORES_REPO,
) -> PRSResult:
    """Lookup under ``policy``, otherwise call ``compute()`` and mark the source."""
    resolved = PrecomputedPolicy(policy)
    if resolved is not PrecomputedPolicy.OFF:
        hit = lookup_precomputed_prs(
            pgs_id=pgs_id,
            genome_build=genome_build,
            vcf_path=vcf_path,
            genotypes_lf=genotypes_lf,
            genotypes_parquet=genotypes_parquet,
            alias=alias,
            reference_restoration=reference_restoration,
            genotype_input_mode=genotype_input_mode,
            maf_fill=maf_fill,
            cache_dir=cache_dir,
            scores_cache=scores_cache,
            repo_id=repo_id,
            pull=resolved is not PrecomputedPolicy.OFF,
        )
        if hit is not None:
            return hit
        if resolved is PrecomputedPolicy.REQUIRE:
            raise PrecomputedMiss(
                f"No published PRS for {pgs_id} under the requested sample/profile"
            )
    result = compute()
    if result.computation_source is None:
        result.computation_source = ComputationSource.COMPUTED.value
    return result
