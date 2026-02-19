"""PRSCatalog: high-level class for PGS Catalog search, PRS computation, and percentile estimation.

Uses bulk FTP metadata (cleaned via cleanup pipeline) instead of per-score REST API calls,
giving instant access to score metadata and performance metrics for the entire catalog.

Cleaned parquets are persisted locally and can be synced to/from HuggingFace.
Loading priority: local cleaned parquet -> HF pull -> raw FTP download + cleanup.
"""

import logging
import math
from pathlib import Path

import polars as pl
from eliot import start_action

from just_prs.cleanup import (
    best_performance_per_score,
    clean_performance_metrics,
    clean_scores,
)
from just_prs.ftp import download_metadata_sheet
from just_prs.hf import pull_cleaned_parquets, push_cleaned_parquets
from just_prs.models import PRSResult
from just_prs.prs import compute_prs
from just_prs.scoring import DEFAULT_CACHE_DIR, resolve_cache_dir

logger = logging.getLogger(__name__)


class PRSCatalog:
    """High-level interface for PGS Catalog data, PRS computation, and percentile estimation.

    Lazily downloads and caches bulk FTP metadata on first access, then exposes
    cleaned LazyFrames for scores and performance metrics. Provides search,
    PRS computation, and percentile estimation without per-score REST API calls.

    Cleaned parquets are persisted locally under metadata_dir. When they are
    missing, the catalog first attempts to pull them from HuggingFace, and
    falls back to downloading raw CSVs from EBI FTP + running the cleanup pipeline.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or resolve_cache_dir()
        self._scores_lf: pl.LazyFrame | None = None
        self._perf_lf: pl.LazyFrame | None = None
        self._best_perf_lf: pl.LazyFrame | None = None

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def metadata_dir(self) -> Path:
        return self._cache_dir / "metadata"

    @property
    def raw_metadata_dir(self) -> Path:
        """Directory for raw (uncleaned) FTP parquet cache."""
        return self._cache_dir / "metadata" / "raw"

    def _has_cleaned_parquets(self) -> bool:
        """Check whether all 3 cleaned parquet files exist locally."""
        return all(
            (self.metadata_dir / f).exists()
            for f in ("scores.parquet", "performance.parquet", "best_performance.parquet")
        )

    def _try_pull_from_hf(self) -> bool:
        """Attempt to pull cleaned parquets from HuggingFace. Returns True on success."""
        try:
            pull_cleaned_parquets(self.metadata_dir)
            return self._has_cleaned_parquets()
        except Exception as exc:
            logger.debug("HF pull failed (will fall back to FTP): %s", exc)
            return False

    def _build_from_ftp(self) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        """Download raw CSVs from EBI FTP, run cleanup pipeline, and persist cleaned parquets."""
        raw_dir = self.raw_metadata_dir

        scores_df = download_metadata_sheet("scores", raw_dir / "scores.parquet")
        perf_df = download_metadata_sheet("performance_metrics", raw_dir / "performance_metrics.parquet")
        eval_df = download_metadata_sheet("evaluation_sample_sets", raw_dir / "evaluation_sample_sets.parquet")

        scores_lf = clean_scores(scores_df)
        perf_lf = clean_performance_metrics(perf_df, eval_df)
        best_perf_lf = best_performance_per_score(perf_lf)

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        scores_lf.collect().write_parquet(self.metadata_dir / "scores.parquet")
        perf_lf.collect().write_parquet(self.metadata_dir / "performance.parquet")
        best_perf_lf.collect().write_parquet(self.metadata_dir / "best_performance.parquet")

        return scores_lf, perf_lf, best_perf_lf

    def _load_all(self) -> None:
        """Load all 3 cleaned LazyFrames, using the fallback chain:
        local cleaned parquet -> HF pull -> raw FTP + cleanup.
        """
        if self._scores_lf is not None:
            return

        with start_action(action_type="prs_catalog:load_all"):
            if self._has_cleaned_parquets():
                self._scores_lf = pl.scan_parquet(self.metadata_dir / "scores.parquet")
                self._perf_lf = pl.scan_parquet(self.metadata_dir / "performance.parquet")
                self._best_perf_lf = pl.scan_parquet(self.metadata_dir / "best_performance.parquet")
                return

            if self._try_pull_from_hf():
                self._scores_lf = pl.scan_parquet(self.metadata_dir / "scores.parquet")
                self._perf_lf = pl.scan_parquet(self.metadata_dir / "performance.parquet")
                self._best_perf_lf = pl.scan_parquet(self.metadata_dir / "best_performance.parquet")
                return

            scores_lf, perf_lf, best_perf_lf = self._build_from_ftp()
            self._scores_lf = scores_lf
            self._perf_lf = perf_lf
            self._best_perf_lf = best_perf_lf

    def _ensure_scores(self) -> pl.LazyFrame:
        self._load_all()
        assert self._scores_lf is not None
        return self._scores_lf

    def _ensure_performance(self) -> pl.LazyFrame:
        self._load_all()
        assert self._perf_lf is not None
        return self._perf_lf

    def _ensure_best_performance(self) -> pl.LazyFrame:
        self._load_all()
        assert self._best_perf_lf is not None
        return self._best_perf_lf

    def build_cleaned_parquets(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Run the full cleanup pipeline and persist 3 cleaned parquet files.

        Downloads raw CSVs from EBI FTP (caching in raw/ subdir), applies the
        cleanup pipeline, and writes scores.parquet, performance.parquet, and
        best_performance.parquet to output_dir.

        Args:
            output_dir: Where to write the cleaned parquets. Defaults to self.metadata_dir.

        Returns:
            Dict mapping table name to its output path.
        """
        dest = output_dir or self.metadata_dir
        dest.mkdir(parents=True, exist_ok=True)
        raw_dir = self.raw_metadata_dir

        with start_action(action_type="prs_catalog:build_cleaned_parquets", output_dir=str(dest)):
            scores_df = download_metadata_sheet("scores", raw_dir / "scores.parquet", overwrite=True)
            perf_df = download_metadata_sheet("performance_metrics", raw_dir / "performance_metrics.parquet", overwrite=True)
            eval_df = download_metadata_sheet("evaluation_sample_sets", raw_dir / "evaluation_sample_sets.parquet", overwrite=True)

            scores_lf = clean_scores(scores_df)
            perf_lf = clean_performance_metrics(perf_df, eval_df)
            best_perf_lf = best_performance_per_score(perf_lf)

            paths: dict[str, Path] = {}
            for name, lf in [("scores", scores_lf), ("performance", perf_lf), ("best_performance", best_perf_lf)]:
                p = dest / f"{name}.parquet"
                lf.collect().write_parquet(p)
                paths[name] = p

            self._scores_lf = scores_lf
            self._perf_lf = perf_lf
            self._best_perf_lf = best_perf_lf

        return paths

    def push_to_hf(self, token: str | None = None, repo_id: str | None = None) -> None:
        """Push cleaned parquets to HuggingFace dataset repo.

        Builds them first if they don't exist locally.

        Args:
            token: HF API token. If None, loaded from .env / HF_TOKEN env var.
            repo_id: HF dataset repo ID. Defaults to just-dna-seq/polygenic_risk_scores.
        """
        if not self._has_cleaned_parquets():
            self.build_cleaned_parquets()

        kwargs: dict[str, str] = {}
        if repo_id is not None:
            kwargs["repo_id"] = repo_id
        push_cleaned_parquets(self.metadata_dir, token=token, **kwargs)

    def reload(self) -> None:
        """Force re-download of all metadata on next access."""
        self._scores_lf = None
        self._perf_lf = None
        self._best_perf_lf = None
        for p in self.metadata_dir.glob("*.parquet"):
            p.unlink()
        raw_dir = self.raw_metadata_dir
        if raw_dir.exists():
            for p in raw_dir.glob("*.parquet"):
                p.unlink()

    def scores(self, genome_build: str | None = None) -> pl.LazyFrame:
        """Return cleaned scores LazyFrame, optionally filtered by genome build.

        Args:
            genome_build: If provided, filter to scores matching this canonical build
                          (GRCh37, GRCh38, GRCh36). Scores with build=NR are excluded.

        Returns:
            Cleaned LazyFrame with columns: pgs_id, name, trait_reported, trait_efo,
            trait_efo_id, genome_build, n_variants, weight_type, pgp_id, pmid,
            ftp_link, release_date
        """
        lf = self._ensure_scores()
        if genome_build is not None:
            lf = lf.filter(pl.col("genome_build").eq(genome_build))
        return lf

    def performance(self, pgs_id: str | None = None) -> pl.LazyFrame:
        """Return cleaned performance metrics LazyFrame.

        Args:
            pgs_id: If provided, filter to metrics for this specific PGS ID.

        Returns:
            Cleaned LazyFrame with parsed numeric metric columns (or_estimate,
            auroc_estimate, etc.) and evaluation context (n_individuals, ancestry_broad).
        """
        lf = self._ensure_performance()
        if pgs_id is not None:
            lf = lf.filter(pl.col("pgs_id").eq(pgs_id))
        return lf

    def best_performance(self, pgs_id: str | None = None) -> pl.LazyFrame:
        """Return best performance metric per PGS ID (largest sample, European-preferred).

        Args:
            pgs_id: If provided, filter to the best metric for this specific PGS ID.
        """
        lf = self._ensure_best_performance()
        if pgs_id is not None:
            lf = lf.filter(pl.col("pgs_id").eq(pgs_id))
        return lf

    def search(
        self,
        query: str,
        genome_build: str | None = None,
    ) -> pl.LazyFrame:
        """Search scores by text query across pgs_id, name, trait_reported, and trait_efo.

        Case-insensitive substring match. Returns cleaned scores LazyFrame
        filtered to matching rows.

        Args:
            query: Search term (case-insensitive substring match)
            genome_build: Optional genome build filter (canonical form)
        """
        lf = self.scores(genome_build=genome_build)
        term = query.strip().lower()
        if not term:
            return lf
        return lf.filter(
            pl.col("pgs_id").str.to_lowercase().str.contains(term, literal=True)
            | pl.col("name").str.to_lowercase().str.contains(term, literal=True)
            | pl.col("trait_reported").str.to_lowercase().str.contains(term, literal=True)
            | pl.col("trait_efo").str.to_lowercase().str.contains(term, literal=True)
        )

    def score_info_row(self, pgs_id: str) -> dict[str, object] | None:
        """Get cleaned score metadata for a single PGS ID as a dict.

        Returns None if the PGS ID is not found.
        """
        rows = (
            self._ensure_scores()
            .filter(pl.col("pgs_id").eq(pgs_id.upper()))
            .collect()
        )
        if rows.height == 0:
            return None
        return rows.row(0, named=True)

    def compute_prs(
        self,
        vcf_path: Path | str,
        pgs_id: str,
        genome_build: str = "GRCh38",
    ) -> PRSResult:
        """Compute PRS for a VCF file against a single PGS score.

        Looks up trait_reported from cached metadata instead of making a REST API call.
        """
        with start_action(
            action_type="prs_catalog:compute_prs",
            pgs_id=pgs_id,
            genome_build=genome_build,
        ):
            info = self.score_info_row(pgs_id)
            trait = info["trait_reported"] if info else None

            return compute_prs(
                vcf_path=vcf_path,
                scoring_file=pgs_id,
                genome_build=genome_build,
                cache_dir=self._cache_dir / "scores",
                pgs_id=pgs_id,
                trait_reported=trait,
            )

    def compute_prs_batch(
        self,
        vcf_path: Path | str,
        pgs_ids: list[str],
        genome_build: str = "GRCh38",
    ) -> list[PRSResult]:
        """Compute PRS for a VCF file against multiple PGS scores.

        Uses cached metadata for trait lookup instead of per-score REST API calls.
        """
        with start_action(
            action_type="prs_catalog:compute_prs_batch",
            pgs_ids=pgs_ids,
            genome_build=genome_build,
        ):
            results: list[PRSResult] = []
            for pgs_id in pgs_ids:
                result = self.compute_prs(vcf_path, pgs_id, genome_build=genome_build)
                results.append(result)
            return results

    def percentile(
        self,
        prs_score: float,
        pgs_id: str,
        mean: float = 0.0,
        std: float | None = None,
    ) -> float | None:
        """Estimate the population percentile for a given PRS score.

        If mean/std are provided (e.g. from a reference cohort), uses them directly.
        Otherwise, estimates the PRS distribution spread from the AUROC of the best
        available performance metric via Cohen's d.

        This is an approximation. Exact percentiles require a matched reference cohort.

        Args:
            prs_score: The computed PRS value
            pgs_id: PGS Catalog Score ID
            mean: Assumed population mean of PRS (default 0.0)
            std: Population standard deviation of PRS. If None, estimated from AUROC.

        Returns:
            Estimated percentile (0-100), or None if std cannot be determined.
        """
        if std is not None and std > 0:
            z = (prs_score - mean) / std
            return round(_norm_cdf(z) * 100.0, 2)

        best_df = self.best_performance(pgs_id=pgs_id).collect()
        if best_df.height == 0:
            return None

        row = best_df.row(0, named=True)
        auroc = row.get("auroc_estimate")
        if auroc is None:
            return None

        auroc = float(auroc)
        if auroc <= 0.5 or auroc >= 1.0:
            return None

        d = _auroc_to_cohens_d(auroc)
        if d is None or d <= 0:
            return None

        # Cohen's d = (mu_case - mu_control) / sigma_pooled
        # Assuming sigma_pooled = 1, the population (mixed) distribution has
        # variance ≈ 1 + d^2/4  (from the mixture of two unit-variance normals
        # separated by d). We use this as the effective population SD.
        effective_std = math.sqrt(1.0 + d * d / 4.0)
        z = (prs_score - mean) / effective_std
        return round(_norm_cdf(z) * 100.0, 2)


def _auroc_to_cohens_d(auroc: float) -> float | None:
    """Convert AUROC to Cohen's d using the formula: AUROC = Phi(d / sqrt(2)).

    Inverts to: d = sqrt(2) * Phi^{-1}(AUROC)
    """
    if auroc <= 0.5 or auroc >= 1.0:
        return None
    z = _norm_ppf(auroc)
    return math.sqrt(2) * z


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function).

    Rational approximation from Abramowitz & Stegun, accurate to ~4.5e-4.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    if p < 0.5:
        return -_norm_ppf(1.0 - p)

    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
