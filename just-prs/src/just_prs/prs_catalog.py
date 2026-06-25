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
from eliot import log_message, start_action

from just_prs.cleanup import (
    best_performance_per_score,
    clean_performance_metrics,
    clean_publications,
    clean_scores,
)
from just_prs.ftp import download_metadata_sheet, _atomic_write_parquet
from just_prs.hf import (
    distributions_filename,
    pull_chip_coverage,
    pull_cleaned_parquets,
    pull_reference_distributions,
)
from just_prs.models import (
    AbsoluteRisk,
    AbsoluteRiskBundle,
    PercentileResult,
    PerformanceInfo,
    PRSResult,
)
from just_prs.ontology import (
    enrich_with_requested_trait_aliases,
    enrich_with_trait_aliases,
    ensure_ontology_alias_columns,
    expand_trait_ids_from_alias_columns,
    normalize_trait_id,
)
from just_prs.prs import RestorationScope, compute_prs
from just_prs.reference import reference_distribution_audit_issues
from just_prs.scoring import DEFAULT_CACHE_DIR, resolve_cache_dir

logger = logging.getLogger(__name__)

_HARMONIZABLE_BUILDS: dict[str, set[str]] = {
    "GRCh38": {"GRCh37", "GRCh36", "NR"},
    "GRCh37": {"GRCh38", "GRCh36", "NR"},
}

_PUBLICATIONS_REQUIRED_COLUMNS = {"pgp_id"}

# Below this weight-mass coverage (C_wt), a percentile is treated as a likely
# low-coverage artifact rather than an authoritative population position.
MIN_RELIABLE_WEIGHT_MASS_COVERAGE: float = 0.20


def _performance_info_from_row(row: dict[str, object]) -> PerformanceInfo:
    """Build a lightweight PerformanceInfo from a cleaned best_performance row.

    Populates effect sizes (OR/HR/Beta) and classification metrics (AUROC/C-index)
    from the flattened numeric columns produced by the cleanup pipeline.
    """
    from just_prs.models import EffectSizeInfo

    def _eff(prefix: str, short: str) -> EffectSizeInfo | None:
        est = row.get(f"{prefix}_estimate")
        if est is None:
            return None
        return EffectSizeInfo(
            name_short=short,
            estimate=float(est),  # type: ignore[arg-type]
            ci_lower=row.get(f"{prefix}_ci_lower"),  # type: ignore[arg-type]
            ci_upper=row.get(f"{prefix}_ci_upper"),  # type: ignore[arg-type]
            se=row.get(f"{prefix}_se"),  # type: ignore[arg-type]
        )

    effect_sizes = [e for e in (_eff("or", "OR"), _eff("hr", "HR"), _eff("beta", "Beta")) if e]
    class_acc = [e for e in (_eff("auroc", "AUROC"), _eff("cindex", "C-index")) if e]
    n_ind = row.get("n_individuals")
    return PerformanceInfo(
        ppm_id=str(row.get("ppm_id") or ""),
        effect_sizes=effect_sizes,
        class_acc=class_acc,
        sample_number=int(n_ind) if n_ind is not None else None,  # type: ignore[arg-type]
        ancestry_broad=row.get("ancestry_broad"),  # type: ignore[arg-type]
    )


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
        self._publications_lf: pl.LazyFrame | None = None
        self._quality_lf: pl.LazyFrame | None = None
        self._chip_coverage_lf: pl.LazyFrame | None = None
        self._chip_coverage_loaded = False
        self._prevalence_lf: pl.LazyFrame | None = None
        self._heritability_lf: pl.LazyFrame | None = None
        self._ref_dist_cache: dict[str, pl.LazyFrame] = {}
        self._ref_dist_refresh_attempted: set[str] = set()
        self._ref_dist_source: dict[str, str] = {}

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
        """Check whether cleaned parquet files exist locally.

        Requires the 3 core files; publications.parquet is optional for
        backward compatibility with older caches.
        """
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

    def _build_from_ftp(self) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        """Download raw CSVs from EBI FTP, run cleanup pipeline, and persist cleaned parquets."""
        raw_dir = self.raw_metadata_dir

        scores_df = download_metadata_sheet("scores", raw_dir / "scores.parquet")
        perf_df = download_metadata_sheet("performance_metrics", raw_dir / "performance_metrics.parquet")
        eval_df = download_metadata_sheet("evaluation_sample_sets", raw_dir / "evaluation_sample_sets.parquet")
        pub_df = download_metadata_sheet("publications", raw_dir / "publications.parquet")

        scores_lf = clean_scores(scores_df)
        perf_lf = clean_performance_metrics(perf_df, eval_df)
        best_perf_lf = best_performance_per_score(perf_lf)
        pub_lf = clean_publications(pub_df)

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_parquet(scores_lf.collect(), self.metadata_dir / "scores.parquet")
        _atomic_write_parquet(perf_lf.collect(), self.metadata_dir / "performance.parquet")
        _atomic_write_parquet(best_perf_lf.collect(), self.metadata_dir / "best_performance.parquet")
        _atomic_write_parquet(pub_lf.collect(), self.metadata_dir / "publications.parquet")

        return scores_lf, perf_lf, best_perf_lf, pub_lf

    def _publication_schema_is_valid(self, lf: pl.LazyFrame) -> bool:
        """Return whether a publications LazyFrame supports PGP ID lookups."""
        columns = set(lf.collect_schema().names())
        return _PUBLICATIONS_REQUIRED_COLUMNS <= columns

    def _rebuild_publications(self) -> pl.LazyFrame | None:
        """Rebuild cleaned publications metadata from the raw cache or FTP."""
        raw_path = self.raw_metadata_dir / "publications.parquet"
        try:
            raw_df = download_metadata_sheet("publications", raw_path)
            pub_lf = clean_publications(raw_df)
            if not self._publication_schema_is_valid(pub_lf):
                logger.warning("Cleaned publications metadata lacks required columns")
                return None
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_parquet(pub_lf.collect(), self.metadata_dir / "publications.parquet")
            return pub_lf
        except Exception as exc:
            logger.warning("Unable to rebuild publications metadata: %s", exc)
            return None

    def _load_quality_from_cache(self) -> pl.LazyFrame | None:
        """Load optional quality scores parquet (synced from HF)."""
        qpath = self.metadata_dir / "pgs_quality_scores.parquet"
        if not qpath.exists():
            return None
        return pl.scan_parquet(qpath)

    def _load_chip_coverage(self) -> pl.LazyFrame | None:
        """Load consumer-chip coverage, pivoted to one row per PGS ID.

        Looks for ``percentiles/chip_coverage.parquet`` locally; on a miss, pulls
        it once from the prs-percentiles HF dataset. The long (pgs_id, chip) table
        is pivoted to wide per-chip columns ``{chip}_array_ready`` (bool) and
        ``{chip}_coverage`` (float) so it can be joined onto scores by pgs_id.
        Returns None when no coverage data is available (e.g. offline first run).
        """
        cov_path = self._cache_dir / "percentiles" / "chip_coverage.parquet"
        if not cov_path.exists():
            try:
                pull_chip_coverage(self._cache_dir / "percentiles")
            except Exception as exc:
                logger.debug("Chip coverage HF pull failed: %s", exc)
        if not cov_path.exists():
            return None
        try:
            long = pl.read_parquet(cov_path)
        except (pl.exceptions.ComputeError, OSError) as exc:
            logger.warning("Corrupt chip_coverage.parquet (%s); deleting.", exc)
            cov_path.unlink(missing_ok=True)
            return None
        if long.height == 0:
            return None
        wide = long.pivot(
            on="chip",
            index="pgs_id",
            values=["array_ready", "coverage_ratio"],
        )
        # polars names pivoted columns like ``array_ready_gsa_v3`` /
        # ``coverage_ratio_gsa_v3``; rename to ``{chip}_array_ready`` / ``{chip}_coverage``.
        chips = long["chip"].unique().to_list()
        rename_map: dict[str, str] = {}
        for chip in chips:
            for src, dst in (
                (f"array_ready_{chip}", f"{chip}_array_ready"),
                (f"coverage_ratio_{chip}", f"{chip}_coverage"),
            ):
                if src in wide.columns:
                    rename_map[src] = dst
        wide = wide.rename(rename_map)
        return wide.lazy()

    def _load_publications_from_cache(self) -> pl.LazyFrame | None:
        """Load optional cleaned publications metadata, rebuilding stale caches."""
        pub_path = self.metadata_dir / "publications.parquet"
        if not pub_path.exists():
            return self._rebuild_publications()

        pub_lf = pl.scan_parquet(pub_path)
        if self._publication_schema_is_valid(pub_lf):
            return pub_lf

        logger.info("Rebuilding stale publications metadata at %s", pub_path)
        return self._rebuild_publications()

    def _load_all(self) -> None:
        """Load cleaned LazyFrames, using the fallback chain:
        local cleaned parquet -> HF pull -> raw FTP + cleanup.
        """
        if self._scores_lf is not None:
            return

        with start_action(action_type="prs_catalog:load_all"):
            if self._has_cleaned_parquets():
                self._scores_lf = pl.scan_parquet(self.metadata_dir / "scores.parquet")
                self._perf_lf = pl.scan_parquet(self.metadata_dir / "performance.parquet")
                self._best_perf_lf = pl.scan_parquet(self.metadata_dir / "best_performance.parquet")
                self._publications_lf = self._load_publications_from_cache()
                self._quality_lf = self._load_quality_from_cache()
                return

            if self._try_pull_from_hf():
                self._scores_lf = pl.scan_parquet(self.metadata_dir / "scores.parquet")
                self._perf_lf = pl.scan_parquet(self.metadata_dir / "performance.parquet")
                self._best_perf_lf = pl.scan_parquet(self.metadata_dir / "best_performance.parquet")
                self._publications_lf = self._load_publications_from_cache()
                self._quality_lf = self._load_quality_from_cache()
                return

            scores_lf, perf_lf, best_perf_lf, pub_lf = self._build_from_ftp()
            self._scores_lf = scores_lf
            self._perf_lf = perf_lf
            self._best_perf_lf = best_perf_lf
            self._publications_lf = pub_lf
            self._quality_lf = self._load_quality_from_cache()

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
            pub_df = download_metadata_sheet("publications", raw_dir / "publications.parquet", overwrite=True)

            scores_lf = clean_scores(scores_df)
            perf_lf = clean_performance_metrics(perf_df, eval_df)
            best_perf_lf = best_performance_per_score(perf_lf)
            pub_lf = clean_publications(pub_df)

            paths: dict[str, Path] = {}
            for name, lf in [
                ("scores", scores_lf),
                ("performance", perf_lf),
                ("best_performance", best_perf_lf),
                ("publications", pub_lf),
            ]:
                p = dest / f"{name}.parquet"
                lf.collect().write_parquet(p)
                paths[name] = p

            self._scores_lf = scores_lf
            self._perf_lf = perf_lf
            self._best_perf_lf = best_perf_lf
            self._publications_lf = pub_lf

        return paths

    def push_to_hf(self, token: str | None = None, repo_id: str | None = None) -> None:
        """Push cleaned metadata and scoring parquets to the combined HF dataset.

        Builds cleaned metadata first if not present locally. Scoring parquets
        must already exist in ``<cache_dir>/scores/``.

        Args:
            token: HF API token. If None, loaded from .env / HF_TOKEN env var.
            repo_id: HF dataset repo ID. Defaults to just-dna-seq/pgs-catalog.
        """
        from just_prs.hf import push_pgs_catalog, DEFAULT_HF_CATALOG_REPO

        if not self._has_cleaned_parquets():
            self.build_cleaned_parquets()

        scores_dir = self._cache_dir / "scores"
        if not scores_dir.exists() or not any(scores_dir.glob("*_hmPOS_*.parquet")):
            raise FileNotFoundError(
                f"No scoring parquets found in {scores_dir}. "
                "Download scoring files first via 'prs catalog bulk download-scores' "
                "or the pipeline."
            )

        push_pgs_catalog(
            metadata_dir=self.metadata_dir,
            scores_dir=scores_dir,
            repo_id=repo_id or DEFAULT_HF_CATALOG_REPO,
            token=token,
        )

    @property
    def percentiles_dir(self) -> Path:
        """Directory for reference distribution parquets."""
        return self._cache_dir / "percentiles"

    def reference_distributions(self, panel: str = "1000g") -> pl.LazyFrame:
        """Return a reference distribution LazyFrame for the given panel.

        Columns: pgs_id, superpopulation, mean, std, n, median, p5, p25, p75, p95.
        Auto-pulls from HuggingFace (just-dna-seq/prs-percentiles) if not cached locally.

        Falls back to the legacy ``reference_distributions.parquet`` filename for
        backward compatibility when the panel-aware file is not found locally.

        Args:
            panel: Reference panel identifier (e.g. ``1000g``, ``hgdp_1kg``).
        """
        if panel in self._ref_dist_cache:
            return self._ref_dist_cache[panel]

        panel_file = distributions_filename(panel)
        local = self.percentiles_dir / panel_file
        legacy = self.percentiles_dir / "reference_distributions.parquet"

        if not local.exists():
            if panel == "1000g" and legacy.exists():
                local = legacy
                self._ref_dist_source[panel] = "local_legacy_cache"
            else:
                with start_action(action_type="prs_catalog:pull_reference_distributions", panel=panel):
                    pulled = pull_reference_distributions(self.percentiles_dir, panel=panel)
                    if pulled is None or not local.exists():
                        empty = pl.LazyFrame(
                            schema={
                                "pgs_id": pl.Utf8,
                                "superpopulation": pl.Utf8,
                                "mean": pl.Float64,
                                "std": pl.Float64,
                                "n": pl.Int64,
                                "median": pl.Float64,
                                "p5": pl.Float64,
                                "p25": pl.Float64,
                                "p75": pl.Float64,
                                "p95": pl.Float64,
                            }
                        )
                        self._ref_dist_cache[panel] = empty
                        self._ref_dist_source[panel] = "unavailable"
                        return empty
                    self._ref_dist_source[panel] = "hf_sync"
        else:
            self._ref_dist_source[panel] = "local_cache"

        df = pl.read_parquet(local)
        issue_path = local.with_name(f"{panel}_distribution_quality_issues.parquet")
        quality_path = local.with_name(f"{panel}_quality.parquet")
        if issue_path.exists():
            issue_df = pl.read_parquet(issue_path)
        else:
            quality_df = pl.read_parquet(quality_path) if quality_path.exists() else None
            issue_df = reference_distribution_audit_issues(df, quality_df)
        error_issue_df = issue_df.filter(pl.col("severity") == "ERROR")
        if error_issue_df.height > 0:
            bad_ids = error_issue_df.select("pgs_id").unique()
            df = df.join(bad_ids, on="pgs_id", how="anti")
            logger.warning(
                "Filtered %s untrustworthy reference PGS IDs from %s.",
                bad_ids.height,
                local,
            )
        lf = df.lazy()
        self._ref_dist_cache[panel] = lf
        return lf

    def _refresh_reference_distributions(self, panel: str = "1000g") -> None:
        """Pull the latest reference distributions parquet from HF and reload cache.

        Called on-demand when percentile lookup misses for a score. The refresh is
        guarded so we only attempt it once per panel per PRSCatalog instance.
        """
        if panel in self._ref_dist_refresh_attempted:
            return

        self._ref_dist_refresh_attempted.add(panel)
        with start_action(action_type="prs_catalog:refresh_reference_distributions", panel=panel):
            pulled = pull_reference_distributions(self.percentiles_dir, panel=panel)
            if pulled is not None:
                self._ref_dist_cache.pop(panel, None)
                self._ref_dist_source[panel] = "hf_sync"

    def refresh_reference_cache(self, panel: str = "1000g") -> None:
        """Force-pull reference distributions and audit sidecars from HuggingFace."""
        self._ref_dist_refresh_attempted.discard(panel)
        self._ref_dist_cache.pop(panel, None)
        self._refresh_reference_distributions(panel=panel)

    def reload(self) -> None:
        """Force re-download of all metadata on next access."""
        self._scores_lf = None
        self._perf_lf = None
        self._best_perf_lf = None
        self._publications_lf = None
        self._ref_dist_cache.clear()
        self._ref_dist_refresh_attempted.clear()
        self._ref_dist_source.clear()
        for p in self.metadata_dir.glob("*.parquet"):
            p.unlink()
        raw_dir = self.raw_metadata_dir
        if raw_dir.exists():
            for p in raw_dir.glob("*.parquet"):
                p.unlink()

    def scores(
        self,
        genome_build: str | None = None,
        include_harmonized: bool = True,
    ) -> pl.LazyFrame:
        """Return cleaned scores LazyFrame, optionally filtered by genome build.

        When ``pgs_quality_scores.parquet`` is available (synced from HF),
        ``synthetic_quality_score`` and ``quality_label`` are joined in.

        Args:
            genome_build: If provided, filter to scores matching this canonical build
                          (GRCh37, GRCh38, GRCh36). Scores with build=NR are excluded
                          unless ``include_harmonized`` is True.
            include_harmonized: When True (default), also include scores whose original
                          build differs from ``genome_build`` but for which harmonized
                          scoring files exist. An ``is_harmonized`` boolean column is
                          added: True for scores whose original build != requested build.
        """
        lf = self._ensure_scores()
        if genome_build is not None:
            native_filter = pl.col("genome_build").eq(genome_build)
            if include_harmonized:
                harmonizable = _HARMONIZABLE_BUILDS.get(genome_build, set())
                if harmonizable:
                    lf = lf.filter(native_filter | pl.col("genome_build").is_in(list(harmonizable)))
                else:
                    lf = lf.filter(native_filter)
            else:
                lf = lf.filter(native_filter)
            lf = lf.with_columns(
                (~pl.col("genome_build").eq(genome_build)).alias("is_harmonized"),
            )
        else:
            lf = lf.with_columns(pl.lit(False).alias("is_harmonized"))
        if self._quality_lf is not None:
            quality_cols = self._quality_lf.select(
                "pgs_id", "synthetic_score", "combined_quality_score", "quality_label",
            )
            lf = lf.join(quality_cols, on="pgs_id", how="left")
        if not self._chip_coverage_loaded:
            self._chip_coverage_lf = self._load_chip_coverage()
            self._chip_coverage_loaded = True
        if self._chip_coverage_lf is not None:
            lf = lf.join(self._chip_coverage_lf, on="pgs_id", how="left")
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

    def publications(self, pgp_id: str | None = None) -> pl.LazyFrame | None:
        """Return cleaned publications LazyFrame, optionally filtered by PGP ID.

        Returns None when the publications parquet is unavailable (older cache
        that predates publications support).

        Args:
            pgp_id: If provided, filter to this specific PGP ID.

        Returns:
            Cleaned LazyFrame with columns: pgp_id, pmid, doi, title, authors,
            journal, date_publication. None if publications data unavailable.
        """
        self._load_all()
        if self._publications_lf is None:
            return None
        lf = self._publications_lf
        if pgp_id is not None:
            lf = lf.filter(pl.col("pgp_id").eq(pgp_id))
        return lf

    def search(
        self,
        query: str,
        genome_build: str | None = None,
        include_harmonized: bool = True,
    ) -> pl.LazyFrame:
        """Search scores by text query across pgs_id, name, trait_reported, and trait_efo.

        Case-insensitive substring match. Returns cleaned scores LazyFrame
        filtered to matching rows.

        Args:
            query: Search term (case-insensitive substring match)
            genome_build: Optional genome build filter (canonical form)
            include_harmonized: Include harmonized (cross-build) scores (default True)
        """
        lf = self.scores(genome_build=genome_build, include_harmonized=include_harmonized)
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

    def _attach_performance(self, result: PRSResult) -> None:
        """Populate ``result.performance`` from the best evaluation metric (F11)."""
        best_df = self.best_performance(pgs_id=result.pgs_id).collect()
        if best_df.height > 0:
            result.performance = _performance_info_from_row(best_df.row(0, named=True))

    def _reference_universe_path(self) -> Path | None:
        """Resolve the cached reference-allele universe parquet, pulling on miss.

        Mirrors the ``_load_chip_coverage`` lazy-pull pattern: looks under
        ``reference/`` locally; on a miss, pulls once from the catalog HF dataset.
        Returns None when unavailable (e.g. offline first run) so callers fall
        back to the un-resolved behaviour.
        """
        from just_prs.hf import (
            REFERENCE_ALLELE_UNIVERSE_FILE,
            pull_reference_allele_universe,
        )

        ref_dir = self._cache_dir / "reference"
        path = ref_dir / REFERENCE_ALLELE_UNIVERSE_FILE
        if not path.exists():
            try:
                pull_reference_allele_universe(ref_dir)
            except Exception as exc:
                logger.debug("Reference-allele universe HF pull failed: %s", exc)
        return path if path.exists() else None

    def compute_prs(
        self,
        vcf_path: Path | str,
        pgs_id: str,
        genome_build: str = "GRCh38",
        genotype_input_mode: str = "auto",
        attach_performance: bool = False,
        genotypes_lf: pl.LazyFrame | None = None,
        reference_restoration: RestorationScope = False,
    ) -> PRSResult:
        """Compute PRS for a VCF file against a single PGS score.

        Looks up trait_reported from cached metadata instead of making a REST API call.
        When ``attach_performance`` is True, also populates ``PRSResult.performance``
        with the best evaluation metric (F11). When ``genotypes_lf`` is provided, the
        pre-normalized genotype frame is reused instead of re-reading ``vcf_path`` — this
        is the single-score mirror of ``compute_prs_batch``, so a normalized Parquet can
        be reused *and* best performance attached in one call (F23). ``reference_restoration``
        (``True`` for WGS, a ``Chip`` for arrays, or ``False`` to disable) fills missing
        reference alleles from the precomputed REF universe (pulled from HF on cache miss)
        so absent loci recover coverage within the chosen scope.
        """
        with start_action(
            action_type="prs_catalog:compute_prs",
            pgs_id=pgs_id,
            genome_build=genome_build,
        ):
            info = self.score_info_row(pgs_id)
            trait = info["trait_reported"] if info else None

            result = compute_prs(
                vcf_path=vcf_path,
                scoring_file=pgs_id,
                genome_build=genome_build,
                cache_dir=self._cache_dir / "scores",
                pgs_id=pgs_id,
                trait_reported=trait,
                genotypes_lf=genotypes_lf,
                genotype_input_mode=genotype_input_mode,
                reference_restoration=reference_restoration,
                reference_universe_path=self._reference_universe_path() if reference_restoration is not False else None,
            )
            if attach_performance:
                self._attach_performance(result)
            return result

    def compute_prs_batch(
        self,
        vcf_path: Path | str,
        pgs_ids: list[str],
        genome_build: str = "GRCh38",
        genotype_input_mode: str = "auto",
        engine: str = "duckdb",
        genotypes_lf: pl.LazyFrame | None = None,
        memory_limit: str | None = None,
        attach_performance: bool = False,
        reference_restoration: RestorationScope = False,
    ) -> "PRSBatchResult":
        """Compute PRS for a VCF file against multiple PGS scores.

        Memory-safe: uses DuckDB engine by default (spill-to-disk), runs
        gc.collect() after each score, continues on per-score errors, and
        auto-retries once on corrupt parquet caches. ``reference_restoration``
        (``True`` for WGS, a ``Chip`` for arrays, ``False`` to disable) fills
        missing reference alleles from the precomputed REF universe within the
        chosen scope (the universe is resolved once before the loop).

        Uses cached metadata for trait lookup instead of per-score REST API calls.
        """
        import gc

        from just_prs.models import PRSBatchOutcome, PRSBatchResult
        from just_prs.prs import (
            PRSEngine,
            _is_corrupt_parquet_error,
            _remove_scoring_parquet_cache,
            compute_prs_duckdb,
        )

        resolved_engine = PRSEngine(engine)
        cache = self._cache_dir / "scores"
        universe_path = self._reference_universe_path() if reference_restoration is not False else None

        with start_action(
            action_type="prs_catalog:compute_prs_batch",
            pgs_ids=pgs_ids,
            genome_build=genome_build,
            engine=engine,
        ):
            results: list[PRSResult] = []
            outcomes: list[PRSBatchOutcome] = []
            failed_ids: list[str] = []

            for pgs_id in pgs_ids:
                attempts = 1
                try:
                    info = self.score_info_row(pgs_id)
                    trait = info["trait_reported"] if info else None

                    if resolved_engine == PRSEngine.DUCKDB:
                        result = compute_prs_duckdb(
                            vcf_path=vcf_path,
                            scoring_file=pgs_id,
                            genome_build=genome_build,
                            cache_dir=cache,
                            pgs_id=pgs_id,
                            trait_reported=trait,
                            genotypes_parquet=str(vcf_path) if (genotypes_lf is None and str(vcf_path).endswith(".parquet")) else None,
                            genotypes_lf=genotypes_lf,
                            memory_limit=memory_limit,
                            genotype_input_mode=genotype_input_mode,
                            reference_restoration=reference_restoration,
                            reference_universe_path=universe_path,
                        )
                    else:
                        result = compute_prs(
                            vcf_path=vcf_path,
                            scoring_file=pgs_id,
                            genome_build=genome_build,
                            cache_dir=cache,
                            pgs_id=pgs_id,
                            trait_reported=trait,
                            genotypes_lf=genotypes_lf,
                            genotype_input_mode=genotype_input_mode,
                            reference_restoration=reference_restoration,
                            reference_universe_path=universe_path,
                        )

                    results.append(result)
                    outcomes.append(PRSBatchOutcome(
                        pgs_id=pgs_id, status="ok", attempts=attempts,
                    ))

                except Exception as exc:
                    if _is_corrupt_parquet_error(exc):
                        removed = _remove_scoring_parquet_cache(
                            pgs_id, cache, genome_build,
                        )
                        if removed:
                            attempts = 2
                            try:
                                log_message(
                                    message_type="prs_catalog:batch_cache_repair",
                                    pgs_id=pgs_id,
                                )
                                info = self.score_info_row(pgs_id)
                                trait = info["trait_reported"] if info else None
                                if resolved_engine == PRSEngine.DUCKDB:
                                    result = compute_prs_duckdb(
                                        vcf_path=vcf_path,
                                        scoring_file=pgs_id,
                                        genome_build=genome_build,
                                        cache_dir=cache,
                                        pgs_id=pgs_id,
                                        trait_reported=trait,
                                        genotypes_parquet=str(vcf_path) if (genotypes_lf is None and str(vcf_path).endswith(".parquet")) else None,
                                        genotypes_lf=genotypes_lf,
                                        memory_limit=memory_limit,
                                        genotype_input_mode=genotype_input_mode,
                                        reference_restoration=reference_restoration,
                                        reference_universe_path=universe_path,
                                    )
                                else:
                                    result = compute_prs(
                                        vcf_path=vcf_path,
                                        scoring_file=pgs_id,
                                        genome_build=genome_build,
                                        cache_dir=cache,
                                        pgs_id=pgs_id,
                                        trait_reported=trait,
                                        genotypes_lf=genotypes_lf,
                                        genotype_input_mode=genotype_input_mode,
                                        reference_restoration=reference_restoration,
                                        reference_universe_path=universe_path,
                                    )
                                results.append(result)
                                outcomes.append(PRSBatchOutcome(
                                    pgs_id=pgs_id, status="cache_repaired",
                                    attempts=attempts,
                                ))
                                gc.collect()
                                continue
                            except Exception as retry_exc:
                                log_message(
                                    message_type="prs_catalog:batch_retry_failed",
                                    pgs_id=pgs_id,
                                    error=str(retry_exc),
                                )
                                failed_ids.append(pgs_id)
                                outcomes.append(PRSBatchOutcome(
                                    pgs_id=pgs_id, status="failed",
                                    error=str(retry_exc), attempts=attempts,
                                ))
                                gc.collect()
                                continue

                    log_message(
                        message_type="prs_catalog:batch_score_failed",
                        pgs_id=pgs_id,
                        error=str(exc),
                    )
                    failed_ids.append(pgs_id)
                    outcomes.append(PRSBatchOutcome(
                        pgs_id=pgs_id, status="failed",
                        error=str(exc), attempts=attempts,
                    ))

                gc.collect()

            if attach_performance:
                for r in results:
                    self._attach_performance(r)

            return PRSBatchResult(
                results=results,
                outcomes=outcomes,
                n_total=len(pgs_ids),
                n_ok=len(results),
                n_failed=len(failed_ids),
                failed_ids=failed_ids,
            )

    def percentile(
        self,
        prs_score: float,
        pgs_id: str,
        ancestry: str = "EUR",
        mean: float = 0.0,
        std: float | None = None,
        panel: str = "1000g",
    ) -> tuple[float | None, str]:
        """Estimate population percentile for a given PRS score (3-tier fallback).

        Thin back-compat wrapper around :meth:`percentile_full` returning only
        ``(percentile | None, method_label)``. See :meth:`percentile_full` for the
        true z-score / reference mean-std and the coverage-aware reliability verdict.
        """
        res = self.percentile_full(
            prs_score, pgs_id, ancestry=ancestry, mean=mean, std=std, panel=panel
        )
        return res.percentile, res.method

    def percentile_full(
        self,
        prs_score: float,
        pgs_id: str,
        ancestry: str = "EUR",
        mean: float = 0.0,
        std: float | None = None,
        panel: str = "1000g",
        weight_mass_coverage: float | None = None,
    ) -> PercentileResult:
        """Estimate population percentile and expose the statistics behind it.

        Three-tier fallback (same as :meth:`percentile`):

        Tier 1 — Reference panel (best): look up (pgs_id, ancestry) in the pre-computed
            reference distributions pulled from just-dna-seq/prs-percentiles.
        Tier 2 — Theoretical (partial): allele-frequency mean/std passed explicitly.
        Tier 3 — AUROC approximation (rough): effective SD from Cohen's d.

        Unlike :meth:`percentile`, this returns the **true** ``z_score`` and the
        ``reference_mean`` / ``reference_std`` used — so callers no longer re-derive z
        by inverting the percentile (lossy at the 0/100 extremes). When
        ``weight_mass_coverage`` (C_wt) is supplied and below
        ``MIN_RELIABLE_WEIGHT_MASS_COVERAGE``, the result is flagged ``reliable=False``
        with a caveat so a deflated low-coverage score can't emit an authoritative
        extreme percentile.

        Args:
            prs_score: The computed PRS value.
            pgs_id: PGS Catalog Score ID.
            ancestry: 1000G superpopulation code (AFR, AMR, EAS, EUR, SAS).
            mean: Population mean used when ``std`` is provided (Tier 2) or for Tier 3.
            std: Population SD. When provided (>0), uses it directly (Tier 2).
            panel: Reference panel identifier for Tier 1 lookup.
            weight_mass_coverage: Optional C_wt for the reliability verdict.

        Returns:
            A :class:`PercentileResult`.
        """
        from just_prs.reference import ancestry_distribution_stats

        pct: float | None = None
        method = "unavailable"
        z: float | None = None
        ref_mean: float | None = None
        ref_std: float | None = None

        if std is not None and std > 0:
            z = (prs_score - mean) / std
            pct = round(_norm_cdf(z) * 100.0, 2)
            method = "theoretical"
            ref_mean, ref_std = mean, std
        else:
            ref_lf = self.reference_distributions(panel=panel)
            stats = ancestry_distribution_stats(pgs_id, ancestry, ref_lf)
            if stats is None:
                self._refresh_reference_distributions(panel=panel)
                ref_lf = self.reference_distributions(panel=panel)
                stats = ancestry_distribution_stats(pgs_id, ancestry, ref_lf)
            if stats is not None:
                ref_mean, ref_std = stats
                z = (prs_score - ref_mean) / ref_std
                pct = round(_norm_cdf(z) * 100.0, 2)
                method = "reference_panel"
            else:
                best_df = self.best_performance(pgs_id=pgs_id).collect()
                if best_df.height > 0:
                    auroc = best_df.row(0, named=True).get("auroc_estimate")
                    if auroc is not None and 0.5 < float(auroc) < 1.0:
                        d = _auroc_to_cohens_d(float(auroc))
                        if d is not None and d > 0:
                            effective_std = math.sqrt(1.0 + d * d / 4.0)
                            z = (prs_score - mean) / effective_std
                            pct = round(_norm_cdf(z) * 100.0, 2)
                            method = "auroc_approx"
                            ref_mean, ref_std = mean, effective_std

        reliable = True
        caveat = ""
        if (
            pct is not None
            and weight_mass_coverage is not None
            and weight_mass_coverage < MIN_RELIABLE_WEIGHT_MASS_COVERAGE
        ):
            reliable = False
            caveat = (
                f"Only {weight_mass_coverage * 100:.0f}% of this score's effect-weight mass "
                f"was matched in this genome (C_wt). The percentile is likely a low-coverage "
                f"artifact, not an authoritative population position."
            )

        # Echo the reference-panel ancestry/panel actually used (F19) — only the
        # reference_panel method draws on a named superpopulation distribution.
        used_ancestry = ancestry.upper() if method == "reference_panel" else None
        used_panel = panel if method == "reference_panel" else None

        return PercentileResult(
            percentile=pct,
            method=method,
            z_score=z,
            reference_mean=ref_mean,
            reference_std=ref_std,
            ancestry=used_ancestry,
            panel=used_panel,
            reliable=reliable,
            caveat=caveat,
        )

    def absolute_risk_from_score(
        self,
        pgs_id: str,
        score: float,
        ancestry: str = "EUR",
        sex: str | None = None,
        weight_mass_coverage: float | None = None,
        panel: str = "1000g",
    ) -> AbsoluteRiskBundle:
        """Chain a raw PRS score to an absolute-risk bundle (F12 convenience).

        Resolves the percentile via :meth:`percentile_full` (using the **true**
        z-score, not a percentile inversion) and feeds that z into
        :meth:`absolute_risk_bundle`. Returns an empty bundle if no percentile /
        z-score could be derived.
        """
        pr = self.percentile_full(
            score, pgs_id, ancestry=ancestry, panel=panel,
            weight_mass_coverage=weight_mass_coverage,
        )
        if pr.z_score is None:
            return AbsoluteRiskBundle()
        return self.absolute_risk_bundle(pgs_id, pr.z_score, sex=sex)

    def prevalence_table(self) -> pl.LazyFrame:
        """Return the prevalence LazyFrame, loading from cache or HF on first access.

        Falls back to an empty LazyFrame if the prevalence parquet is not available.
        """
        if self._prevalence_lf is not None:
            return self._prevalence_lf

        local = self.metadata_dir / "trait_prevalence.parquet"
        if local.exists():
            self._prevalence_lf = pl.scan_parquet(local)
            return self._prevalence_lf

        from just_prs.prevalence import pull_prevalence_from_hf

        pulled = pull_prevalence_from_hf(self.metadata_dir)
        if pulled is not None and local.exists():
            self._prevalence_lf = pl.scan_parquet(local)
            return self._prevalence_lf

        from just_prs.prevalence import _PREVALENCE_SCHEMA

        self._prevalence_lf = pl.LazyFrame(schema=_PREVALENCE_SCHEMA)
        return self._prevalence_lf

    def heritability_table(self) -> pl.LazyFrame:
        """Return the heritability LazyFrame, loading from cache or HF on first access.

        Contains multiple rows per EFO ID (one per ancestry × source × method).
        Falls back to an empty LazyFrame if the heritability parquet is not available.
        """
        if self._heritability_lf is not None:
            return self._heritability_lf

        local = self.metadata_dir / "trait_heritability.parquet"
        if local.exists():
            self._heritability_lf = pl.scan_parquet(local)
            return self._heritability_lf

        from just_prs.heritability import pull_heritability_from_hf, _HERITABILITY_SCHEMA

        pulled = pull_heritability_from_hf(self.metadata_dir)
        if pulled is not None and local.exists():
            self._heritability_lf = pl.scan_parquet(local)
            return self._heritability_lf

        self._heritability_lf = pl.LazyFrame(schema=_HERITABILITY_SCHEMA)
        return self._heritability_lf

    def heritability_for_trait(
        self,
        efo_ids: list[str],
        ancestry: str | None = None,
    ) -> list[dict]:
        """Look up heritability estimates for given EFO trait IDs.

        Returns all matching rows as dicts with keys: h2_liability, h2_observed,
        ancestry, source, confidence, source_detail, method.

        Args:
            efo_ids: List of EFO trait IDs to look up.
            ancestry: If provided, filter to this ancestry only.

        Returns:
            List of dicts, one per matching heritability estimate.
        """
        h2_df, expanded_ids = self._risk_metadata_with_aliases(
            self.heritability_table(),
            efo_ids,
            self.metadata_dir / "trait_heritability.parquet",
        )
        if h2_df.height == 0:
            return []

        filtered = h2_df.lazy().filter(pl.col("efo_id").is_in(expanded_ids))
        if ancestry is not None:
            filtered = filtered.filter(pl.col("ancestry").eq(ancestry))

        rows = filtered.collect()
        if rows.height == 0:
            return []

        return rows.to_dicts()

    def _risk_metadata_with_aliases(
        self,
        lf: pl.LazyFrame,
        trait_ids: list[str],
        local_path: Path,
    ) -> tuple[pl.DataFrame, list[str]]:
        """Return risk metadata with ontology aliases and expanded requested IDs."""
        requested = [
            normalized
            for trait_id in trait_ids
            if (normalized := normalize_trait_id(trait_id)) is not None
        ]
        if not requested:
            return pl.DataFrame(), []

        df = ensure_ontology_alias_columns(lf.collect())
        expanded = expand_trait_ids_from_alias_columns(requested, df)
        has_non_efo_request = any(not trait_id.startswith("EFO_") for trait_id in requested)
        needs_runtime_aliases = has_non_efo_request and set(expanded) == set(requested)

        if needs_runtime_aliases and df.height > 0:
            enriched = enrich_with_trait_aliases(
                df,
                cache_dir=self.raw_metadata_dir / "ontology_xrefs",
                allow_network=False,
            )
            if enriched.height > df.height:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                enriched.write_parquet(local_path)
                if local_path.name == "trait_heritability.parquet":
                    self._heritability_lf = pl.scan_parquet(local_path)
                elif local_path.name == "trait_prevalence.parquet":
                    self._prevalence_lf = pl.scan_parquet(local_path)
                df = enriched
                expanded = expand_trait_ids_from_alias_columns(requested, df)

        needs_icd_fallback = has_non_efo_request and set(expanded) == set(requested)
        if needs_icd_fallback and df.height > 0:
            from just_prs.heritability import download_efo_ukb_mappings

            mapping_path = self.raw_metadata_dir / "heritability" / "efo_ukb_mappings.parquet"
            mappings_df = download_efo_ukb_mappings(mapping_path, overwrite=False)
            requested_df = pl.DataFrame({"trait_efo_id": requested})
            enriched = enrich_with_requested_trait_aliases(
                df,
                requested_traits_df=requested_df,
                efo_mappings_df=mappings_df,
                cache_dir=self.raw_metadata_dir / "ontology_xrefs",
                allow_network=True,
            )
            if enriched.height > df.height:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                enriched.write_parquet(local_path)
                if local_path.name == "trait_heritability.parquet":
                    self._heritability_lf = pl.scan_parquet(local_path)
                elif local_path.name == "trait_prevalence.parquet":
                    self._prevalence_lf = pl.scan_parquet(local_path)
                df = enriched
                expanded = expand_trait_ids_from_alias_columns(requested, df)

        return df, expanded

    def absolute_risk_bundle(
        self,
        pgs_id: str,
        z_score: float,
        sex: str | None = None,
    ) -> AbsoluteRiskBundle:
        """Compute ALL available absolute risk estimates for a PGS score.

        Runs every method for which input data exists:
        - OR-per-SD (from PGS Catalog best_performance)
        - AUC-bivariate (from PGS Catalog best_performance)
        - h²-liability (for each ancestry/source in heritability table)

        Returns an AbsoluteRiskBundle with all estimates, a best pick, and
        agreement information. Returns an empty bundle if prevalence is unavailable.

        Args:
            pgs_id: PGS Catalog Score ID.
            z_score: PRS z-score (SDs from population mean).
            sex: Optional sex filter for sex-specific prevalence.

        Returns:
            AbsoluteRiskBundle with all available estimates.
        """
        from just_prs.absolute_risk import estimate_all_absolute_risks

        score_info = self.score_info_row(pgs_id)
        if score_info is None:
            return AbsoluteRiskBundle()

        efo_ids_raw = score_info.get("trait_efo_id")
        if efo_ids_raw is None:
            return AbsoluteRiskBundle()

        efo_ids = [e.strip() for e in str(efo_ids_raw).split(",")]
        if not efo_ids:
            return AbsoluteRiskBundle()

        prev_df, expanded_efo_ids = self._risk_metadata_with_aliases(
            self.prevalence_table(),
            efo_ids,
            self.metadata_dir / "trait_prevalence.parquet",
        )
        if prev_df.height == 0:
            return AbsoluteRiskBundle(
                heritability_status="table_unavailable",
                heritability_detail="Risk metadata tables are unavailable or empty.",
                heritability_trait_ids=efo_ids,
            )

        prev_lf = prev_df.lazy()
        prev_filter = prev_lf.filter(pl.col("efo_id").is_in(expanded_efo_ids))
        if sex is not None:
            sex_filtered = prev_filter.filter(
                pl.col("sex").eq(sex) | pl.col("sex").is_null()
            )
            prev_rows = sex_filtered.collect()
            if prev_rows.height == 0:
                prev_rows = prev_filter.collect()
        else:
            prev_rows = prev_filter.collect()

        if prev_rows.height == 0:
            return AbsoluteRiskBundle(
                heritability_status="not_checked",
                heritability_detail="No prevalence metadata matched this trait, so absolute risk could not be estimated.",
                heritability_trait_ids=expanded_efo_ids,
            )

        prev_row = prev_rows.row(0, named=True)
        prevalence = prev_row.get("prevalence")
        if prevalence is None or prevalence <= 0 or prevalence >= 1.0:
            return AbsoluteRiskBundle()

        prevalence_source = prev_row.get("source", "")
        prevalence_type = prev_row.get("prevalence_type", "lifetime")
        prev_confidence = prev_row.get("confidence", "moderate")

        best_df = self.best_performance(pgs_id=pgs_id).collect()
        or_estimate: float | None = None
        auroc_estimate: float | None = None
        effect_citation: str | None = None

        if best_df.height > 0:
            best_row = best_df.row(0, named=True)
            or_val = best_row.get("or_estimate")
            if or_val is not None:
                or_estimate = float(or_val)
            auroc_val = best_row.get("auroc_estimate")
            if auroc_val is not None:
                auroc_estimate = float(auroc_val)

            pgp_id = best_row.get("pgp_id")
            if pgp_id is not None:
                pub_lf = self.publications(pgp_id=pgp_id)
                if pub_lf is not None:
                    pub_rows = pub_lf.collect()
                    if pub_rows.height > 0:
                        pub = pub_rows.row(0, named=True)
                        authors = pub.get("authors", "")
                        journal = pub.get("journal", "")
                        pmid = pub.get("pmid")
                        first_author = str(authors).split(",")[0].strip() if authors else ""
                        parts = [first_author]
                        if journal:
                            parts.append(str(journal))
                        if pmid is not None:
                            parts.append(f"PMID: {pmid}")
                        effect_citation = ", ".join(p for p in parts if p)

        h2_df, expanded_h2_ids = self._risk_metadata_with_aliases(
            self.heritability_table(),
            efo_ids,
            self.metadata_dir / "trait_heritability.parquet",
        )
        if h2_df.height > 0:
            h2_matches = h2_df.filter(pl.col("efo_id").is_in(expanded_h2_ids))
            dedupe_cols = [
                col
                for col in ("canonical_efo_id", "source", "ancestry", "method", "h2_liability")
                if col in h2_matches.columns
            ]
            if dedupe_cols:
                h2_matches = h2_matches.unique(subset=dedupe_cols, maintain_order=True)
            h2_rows = h2_matches.to_dicts()
        else:
            h2_rows = []
        h2_estimates: list[dict] = []
        for h2_row in h2_rows:
            h2_lia = h2_row.get("h2_liability")
            if h2_lia is not None and h2_lia > 0:
                h2_estimates.append({
                    "h2_liability": h2_lia,
                    "ancestry": h2_row.get("ancestry"),
                    "source": h2_row.get("source", ""),
                    "confidence": h2_row.get("confidence", "moderate"),
                    "source_detail": h2_row.get("source_detail", ""),
                })

        if h2_df.height == 0:
            heritability_status = "table_unavailable"
            heritability_detail = "Heritability metadata is unavailable, so no h²-liability method could be computed."
        elif h2_estimates:
            heritability_status = "used"
            heritability_detail = f"Used {len(h2_estimates)} mapped h²-liability estimate(s)."
        else:
            heritability_status = "no_mapped_h2"
            heritability_detail = (
                "No mapped h²-liability estimate is available for this trait. "
                "The app checked exact trait IDs and ontology aliases."
            )

        base_caveats: list[str] = []
        if prevalence_source in ("gwas_catalog_cohort", "pgs_eval_cohort"):
            base_caveats.append("Cohort case fraction used as prevalence proxy, not true population prevalence")
        base_caveats.append("This is an estimate, not a clinical diagnosis")

        bundle = estimate_all_absolute_risks(
            z_score=z_score,
            prevalence=prevalence,
            or_estimate=or_estimate,
            auroc_estimate=auroc_estimate,
            h2_estimates=h2_estimates if h2_estimates else None,
            prevalence_source=prevalence_source,
            prevalence_type=prevalence_type,
            confidence=prev_confidence,
            effect_size_citation=effect_citation,
            caveats=base_caveats,
        )
        bundle.heritability_status = heritability_status
        bundle.heritability_detail = heritability_detail
        bundle.heritability_trait_ids = expanded_h2_ids
        return bundle

    def absolute_risk(
        self,
        pgs_id: str,
        z_score: float,
        sex: str | None = None,
    ) -> AbsoluteRisk | None:
        """Estimate absolute disease risk for a given PGS score and z-score.

        Joins scores -> best_performance -> prevalence and calls the risk
        estimator. Returns None if required data (prevalence, effect size)
        is unavailable.

        Args:
            pgs_id: PGS Catalog Score ID.
            z_score: PRS z-score (SDs from population mean).
            sex: Optional sex filter for sex-specific prevalence ('male'/'female').

        Returns:
            AbsoluteRisk model or None if estimation is not possible.
        """
        from just_prs.absolute_risk import estimate_absolute_risk

        score_info = self.score_info_row(pgs_id)
        if score_info is None:
            return None

        efo_ids_raw = score_info.get("trait_efo_id")
        if efo_ids_raw is None:
            return None

        efo_ids = [e.strip() for e in str(efo_ids_raw).split(",")]
        if not efo_ids:
            return None

        prev_lf = self.prevalence_table()
        prev_filter = prev_lf.filter(pl.col("efo_id").is_in(efo_ids))
        if sex is not None:
            sex_filtered = prev_filter.filter(
                pl.col("sex").eq(sex) | pl.col("sex").is_null()
            )
            prev_rows = sex_filtered.collect()
            if prev_rows.height == 0:
                prev_rows = prev_filter.collect()
        else:
            prev_rows = prev_filter.collect()

        if prev_rows.height == 0:
            return None

        prev_row = prev_rows.row(0, named=True)
        prevalence = prev_row.get("prevalence")
        if prevalence is None or prevalence <= 0 or prevalence >= 1.0:
            return None

        prevalence_source = prev_row.get("source", "")
        prevalence_type = prev_row.get("prevalence_type", "lifetime")
        confidence = prev_row.get("confidence", "moderate")

        best_df = self.best_performance(pgs_id=pgs_id).collect()
        or_estimate: float | None = None
        auroc_estimate: float | None = None
        effect_citation: str | None = None

        if best_df.height > 0:
            best_row = best_df.row(0, named=True)
            or_val = best_row.get("or_estimate")
            if or_val is not None:
                or_estimate = float(or_val)
            auroc_val = best_row.get("auroc_estimate")
            if auroc_val is not None:
                auroc_estimate = float(auroc_val)

            pgp_id = best_row.get("pgp_id")
            if pgp_id is not None:
                pub_lf = self.publications(pgp_id=pgp_id)
                if pub_lf is not None:
                    pub_rows = pub_lf.collect()
                    if pub_rows.height > 0:
                        pub = pub_rows.row(0, named=True)
                        authors = pub.get("authors", "")
                        title = pub.get("title", "")
                        journal = pub.get("journal", "")
                        pmid = pub.get("pmid")
                        first_author = str(authors).split(",")[0].strip() if authors else ""
                        parts = [first_author]
                        if journal:
                            parts.append(str(journal))
                        if pmid is not None:
                            parts.append(f"PMID: {pmid}")
                        effect_citation = ", ".join(p for p in parts if p)

        caveats: list[str] = []
        if prevalence_source in ("gwas_catalog_cohort", "pgs_eval_cohort"):
            caveats.append("Cohort case fraction used as prevalence proxy, not true population prevalence")
        caveats.append("This is an estimate, not a clinical diagnosis")

        return estimate_absolute_risk(
            z_score=z_score,
            prevalence=prevalence,
            or_estimate=or_estimate,
            auroc_estimate=auroc_estimate,
            prevalence_source=prevalence_source,
            prevalence_type=prevalence_type,
            confidence=confidence,
            effect_size_citation=effect_citation,
            caveats=caveats,
        )

    def reference_data_status(
        self,
        pgs_id: str,
        panel: str = "1000g",
        refresh_on_miss: bool = True,
    ) -> dict[str, object]:
        """Return availability/source status for precomputed reference percentiles.

        This reports whether a PGS ID has precomputed reference distributions in
        the panel file, which superpopulations are available, and where the panel
        file was resolved from (local cache, HF sync, or unavailable).
        """
        lf = self.reference_distributions(panel=panel)
        rows = (
            lf.filter(pl.col("pgs_id") == pgs_id)
            .select("superpopulation")
            .collect()
        )
        if rows.height == 0 and refresh_on_miss:
            self._refresh_reference_distributions(panel=panel)
            lf = self.reference_distributions(panel=panel)
            rows = (
                lf.filter(pl.col("pgs_id") == pgs_id)
                .select("superpopulation")
                .collect()
            )
        superpops: list[str] = []
        if rows.height > 0:
            superpops = sorted(set(rows["superpopulation"].to_list()))
        source_code = self._ref_dist_source.get(panel, "unknown")
        source_label = {
            "hf_sync": "HuggingFace prs-percentiles",
            "local_cache": "local percentiles cache",
            "local_legacy_cache": "local legacy percentiles cache",
            "unavailable": "no reference distributions file",
        }.get(source_code, "unknown")
        issue_path = self.percentiles_dir / f"{panel}_distribution_quality_issues.parquet"
        audit_warning_count = 0
        audit_error_count = 0
        audit_issues: list[str] = []
        audit_status = "not available"
        if issue_path.exists():
            issue_df = pl.read_parquet(issue_path)
            pgs_issues = issue_df.filter(pl.col("pgs_id") == pgs_id)
            audit_warning_count = pgs_issues.filter(pl.col("severity") == "WARN").height
            audit_error_count = pgs_issues.filter(pl.col("severity") == "ERROR").height
            if pgs_issues.height > 0:
                audit_issues = sorted(set(pgs_issues["issue"].to_list()))
            audit_status = (
                "error" if audit_error_count > 0 else
                "warning" if audit_warning_count > 0 else
                "pass"
            )
        return {
            "has_reference_data": len(superpops) > 0,
            "available_superpopulations": superpops,
            "source_code": source_code,
            "source_label": source_label,
            "panel": panel,
            "audit_status": audit_status,
            "audit_warning_count": audit_warning_count,
            "audit_error_count": audit_error_count,
            "audit_issues": audit_issues,
        }


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
