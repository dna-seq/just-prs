"""High-level PRS scoring for consumer genotyping arrays.

Chains: normalize → detect chip → LD-proxy substitution → MAF fill → score → quality tier.

This is the convenience entry point for array users. It handles the full pipeline
from a raw 23andMe/AncestryDNA file to an ``ArrayPRSResult`` with coverage-aware
quality tiers.
"""

from pathlib import Path

import polars as pl
from eliot import log_message, start_action

from just_prs.arrays import detect_chip_generation, normalize_array
from just_prs.chip_coverage import CHIPS_BY_ID, Chip
from just_prs.ld_proxy import apply_ld_proxies, ld_proxy_pgs_path
from just_prs.models import ArrayPRSResult, classify_coverage_tier
from just_prs.prs import RestorationScope, compute_prs, compute_prs_duckdb, PRSEngine
from just_prs.scoring import resolve_cache_dir


def _resolve_array_restoration(
    resolved_chip: str,
    genome_build: str,
    cache_dir: Path,
) -> tuple[RestorationScope, Path | None]:
    """Resolve chip-aware reference restoration for an array, build-gated.

    Hom-ref restoration at chip-typed positions is valid only when the chip has a
    typed-position manifest for the array build *and* a reference-allele universe is
    published for that build. Both GRCh38 (A2) and GRCh37 (A1) GSA manifests ship, so
    this unlocks per build as soon as the matching universe is available; it still
    degrades to ``(False, None)`` (a no-op) when the chip lacks a manifest for the
    build or the universe is not published yet (see docs/grch37-universe-build.md).
    Returns the scope + build-matched universe path.
    """
    from just_prs.hf import (
        pull_reference_allele_universe,
        reference_allele_universe_filename,
    )

    try:
        chip = Chip(resolved_chip)
    except ValueError:
        return False, None
    spec = CHIPS_BY_ID.get(chip)
    manifests: dict[str, str] = spec["manifests"] if spec else {}  # type: ignore[assignment]
    if not manifests or genome_build not in manifests:
        log_message(
            message_type="array_scoring:restoration_deferred",
            chip=str(chip),
            genome_build=genome_build,
            reason="no build-matched chip manifest",
        )
        return False, None

    ref_dir = cache_dir / "reference"
    candidate = ref_dir / reference_allele_universe_filename(genome_build)
    if not candidate.exists():
        try:
            pull_reference_allele_universe(ref_dir, genome_build=genome_build)
        except Exception as exc:  # offline / not published — degrade to no-op
            log_message(message_type="array_scoring:no_reference_universe", reason=str(exc))
    if not candidate.exists():
        return False, None
    return chip, candidate


def compute_array_prs(
    array_path: Path,
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str = "GRCh37",
    cache_dir: Path | None = None,
    pgs_id: str = "unknown",
    trait_reported: str | None = None,
    chip: str | None = None,
    ld_proxy: bool = True,
    maf_fill: bool = True,
    min_proxy_r2: float = 0.8,
    panel: str = "1000g",
    array_format: str | None = None,
    engine: str | PRSEngine = PRSEngine.DUCKDB,
    normalized_path: Path | None = None,
) -> ArrayPRSResult:
    """Compute a PRS from a consumer genotyping array file.

    Full pipeline: normalize → detect chip → LD-proxy substitution → MAF fill → score.

    Args:
        array_path: Path to raw array file (.txt, .txt.gz, .csv, .zip).
        scoring_file: Scoring file path, PGS ID string, or pre-loaded LazyFrame.
        genome_build: Build of the array coordinates (default ``"GRCh37"`` —
            all current DTC arrays report GRCh37).
        cache_dir: Cache directory. Resolved via ``resolve_cache_dir()`` if None.
        pgs_id: PGS ID for result labeling.
        trait_reported: Trait name for result labeling.
        chip: Chip identifier (e.g. ``"gsa_v3"``). Auto-detected if None.
        ld_proxy: Whether to apply LD-proxy substitution.
        maf_fill: Whether to fill remaining gaps with population MAF.
        min_proxy_r2: Minimum r² for LD-proxy acceptance.
        panel: Reference panel for LD tables (default ``"1000g"``).
        array_format: ``"23andme"`` or ``"ancestrydna"``; auto-detected if None.
        engine: PRS computation engine (``"DUCKDB"`` or ``"POLARS"``).
        normalized_path: Pre-normalized parquet path. If provided, skip normalization.

    Returns:
        ArrayPRSResult with coverage tiers, proxy stats, and the computed score.
    """
    resolved_cache = cache_dir or resolve_cache_dir()

    with start_action(
        action_type="array_scoring:compute",
        array_path=str(array_path),
        pgs_id=pgs_id,
        genome_build=genome_build,
        ld_proxy=ld_proxy,
        maf_fill=maf_fill,
    ):
        if normalized_path is not None and normalized_path.exists():
            norm_path = normalized_path
        else:
            norm_dir = resolved_cache / "normalized" / "arrays"
            norm_dir.mkdir(parents=True, exist_ok=True)
            norm_path = norm_dir / f"{array_path.stem}.parquet"
            if not norm_path.exists() or norm_path.stat().st_mtime < array_path.stat().st_mtime:
                normalize_array(array_path, norm_path, genome_build=genome_build, array_format=array_format)

        genotypes_lf = pl.scan_parquet(norm_path)

        chip_gen = detect_chip_generation(norm_path, array_format=array_format)
        resolved_chip = chip or chip_gen.chip_id

        log_message(
            message_type="array_scoring:chip_detected",
            chip_id=chip_gen.chip_id,
            platform=chip_gen.platform,
            generation_label=chip_gen.generation_label,
            ld_proxy_available=chip_gen.ld_proxy_available,
            marker_count=chip_gen.marker_count,
        )

        n_proxied = 0
        proxy_r2_mean = None
        scoring_lf_for_proxy = None

        if ld_proxy and chip_gen.ld_proxy_available:
            ld_path = ld_proxy_pgs_path(resolved_cache, pgs_id, resolved_chip, genome_build, panel)

            if not ld_path.exists():
                from just_prs.hf import pull_ld_proxy_pgs_table
                pulled = pull_ld_proxy_pgs_table(
                    pgs_id=pgs_id,
                    chip=resolved_chip,
                    build=genome_build,
                    local_dir=ld_path.parent,
                    panel=panel,
                )
                if pulled is not None:
                    ld_path = pulled

            if ld_path.exists():
                scoring_lf_for_proxy = _load_scoring_lf(scoring_file, genome_build, resolved_cache)
                if scoring_lf_for_proxy is not None:
                    enhanced_scoring, n_proxied, proxy_r2_mean = apply_ld_proxies(
                        scoring_lf=scoring_lf_for_proxy,
                        genotypes_lf=genotypes_lf,
                        ld_table=ld_path,
                        min_r2=min_proxy_r2,
                    )
                    scoring_file = enhanced_scoring
            elif not ld_path.exists():
                log_message(
                    message_type="array_scoring:no_ld_table",
                    pgs_id=pgs_id,
                    chip=resolved_chip,
                    build=genome_build,
                    panel=panel,
                )

        # Chip-aware reference restoration (hom-ref at chip-typed positions only),
        # build-gated: a no-op until a build-matched universe exists. Composes before
        # maf_fill — a restored hom-ref locus is ref-known, so the dosage chain takes
        # it ahead of the 2·MAF fill.
        reference_restoration, reference_universe_path = _resolve_array_restoration(
            resolved_chip, genome_build, resolved_cache
        )

        if engine == PRSEngine.DUCKDB or engine == "DUCKDB":
            base_result = compute_prs_duckdb(
                vcf_path=str(array_path),
                scoring_file=scoring_file,
                genome_build=genome_build,
                cache_dir=resolved_cache,
                pgs_id=pgs_id,
                trait_reported=trait_reported,
                genotypes_lf=genotypes_lf,
                genotypes_parquet=str(norm_path),
                maf_fill=maf_fill,
                reference_restoration=reference_restoration,
                reference_universe_path=reference_universe_path,
            )
        else:
            base_result = compute_prs(
                vcf_path=str(array_path),
                scoring_file=scoring_file,
                genome_build=genome_build,
                cache_dir=resolved_cache,
                pgs_id=pgs_id,
                trait_reported=trait_reported,
                genotypes_lf=genotypes_lf,
                maf_fill=maf_fill,
                reference_restoration=reference_restoration,
                reference_universe_path=reference_universe_path,
            )

        typed = base_result.variants_matched - base_result.variants_maf_filled - n_proxied
        total = base_result.variants_total
        coverage_ratio = typed / total if total > 0 else 0.0
        effective_cov = (typed + n_proxied + base_result.variants_maf_filled) / total if total > 0 else 0.0

        result = ArrayPRSResult(
            **{k: v for k, v in base_result.model_dump().items()},
            chip=resolved_chip,
            coverage_ratio=coverage_ratio,
            variants_proxied=n_proxied,
            effective_coverage=effective_cov,
            coverage_tier=classify_coverage_tier(effective_cov),
            proxy_r2_mean=proxy_r2_mean,
            score_uncorrected=base_result.score,
        )

        log_message(
            message_type="array_scoring:complete",
            pgs_id=pgs_id,
            chip=resolved_chip,
            coverage_ratio=coverage_ratio,
            effective_coverage=effective_cov,
            coverage_tier=result.coverage_tier,
            n_proxied=n_proxied,
            n_maf_filled=base_result.variants_maf_filled,
            score=base_result.score,
        )

        return result


def _load_scoring_lf(
    scoring_file: Path | pl.LazyFrame | str,
    genome_build: str,
    cache_dir: Path,
) -> pl.LazyFrame | None:
    """Load a scoring file as a LazyFrame for LD-proxy pre-processing."""
    if isinstance(scoring_file, pl.LazyFrame):
        return scoring_file

    from just_prs.scoring import load_scoring

    try:
        lf, _header = load_scoring(scoring_file, genome_build=genome_build, cache_dir=cache_dir)
        return lf
    except Exception:
        return None
