"""Tests for the PGS Catalog cleanup pipeline and PRSCatalog class.

Uses real downloaded PGS Catalog metadata (auto-cached) to validate
cleanup transforms, search, and percentile computation.
"""

import polars as pl
import pytest

from just_prs.cleanup import (
    best_performance_per_score,
    clean_performance_metrics,
    clean_scores,
    normalize_genome_build,
    parse_metric_string,
    rename_score_columns,
)
from just_prs.ftp import download_metadata_sheet
from just_prs.prs_catalog import PRSCatalog
from just_prs.scoring import resolve_cache_dir

CACHE_DIR = resolve_cache_dir()
METADATA_DIR = CACHE_DIR / "metadata"
RAW_METADATA_DIR = METADATA_DIR / "raw"


@pytest.fixture(scope="module")
def raw_scores_df() -> pl.DataFrame:
    cache_path = RAW_METADATA_DIR / "scores.parquet"
    return download_metadata_sheet("scores", cache_path)


@pytest.fixture(scope="module")
def raw_perf_df() -> pl.DataFrame:
    cache_path = RAW_METADATA_DIR / "performance_metrics.parquet"
    return download_metadata_sheet("performance_metrics", cache_path)


@pytest.fixture(scope="module")
def raw_eval_df() -> pl.DataFrame:
    cache_path = RAW_METADATA_DIR / "evaluation_sample_sets.parquet"
    return download_metadata_sheet("evaluation_sample_sets", cache_path)


@pytest.fixture(scope="module")
def catalog() -> PRSCatalog:
    return PRSCatalog(cache_dir=CACHE_DIR)


class TestParseMetricString:
    def test_estimate_with_ci(self) -> None:
        result = parse_metric_string("1.55 [1.52,1.58]")
        assert result["estimate"] == pytest.approx(1.55)
        assert result["ci_lower"] == pytest.approx(1.52)
        assert result["ci_upper"] == pytest.approx(1.58)
        assert result["se"] is None

    def test_estimate_with_se(self) -> None:
        result = parse_metric_string("-0.7 (0.15)")
        assert result["estimate"] == pytest.approx(-0.7)
        assert result["ci_lower"] is None
        assert result["ci_upper"] is None
        assert result["se"] == pytest.approx(0.15)

    def test_estimate_only(self) -> None:
        result = parse_metric_string("1.41")
        assert result["estimate"] == pytest.approx(1.41)
        assert result["ci_lower"] is None
        assert result["ci_upper"] is None
        assert result["se"] is None

    def test_none_input(self) -> None:
        result = parse_metric_string(None)
        assert result["estimate"] is None

    def test_unparseable_string(self) -> None:
        result = parse_metric_string("some text = 0.04")
        assert result["estimate"] is None


class TestCleanScores:
    def test_column_names_are_snake_case(self, raw_scores_df: pl.DataFrame) -> None:
        cleaned = clean_scores(raw_scores_df).collect()
        expected_cols = {
            "pgs_id", "name", "trait_reported", "trait_efo", "trait_efo_id",
            "genome_build", "n_variants", "weight_type", "pgp_id", "pmid",
            "ftp_link", "release_date",
        }
        assert set(cleaned.columns) == expected_cols

    def test_row_count_preserved(self, raw_scores_df: pl.DataFrame) -> None:
        cleaned = clean_scores(raw_scores_df).collect()
        assert cleaned.height == raw_scores_df.height

    def test_genome_build_normalized(self, raw_scores_df: pl.DataFrame) -> None:
        cleaned = clean_scores(raw_scores_df).collect()
        builds = set(cleaned["genome_build"].unique().to_list())
        assert builds <= {"GRCh37", "GRCh38", "GRCh36", "NR"}

    def test_no_hg19_hg38_variants(self, raw_scores_df: pl.DataFrame) -> None:
        cleaned = clean_scores(raw_scores_df).collect()
        builds = set(cleaned["genome_build"].unique().to_list())
        assert "hg19" not in builds
        assert "hg38" not in builds
        assert "hg37" not in builds
        assert "NCBI36" not in builds

    def test_pgs_ids_preserved(self, raw_scores_df: pl.DataFrame) -> None:
        cleaned = clean_scores(raw_scores_df).collect()
        raw_ids = set(raw_scores_df["Polygenic Score (PGS) ID"].to_list())
        cleaned_ids = set(cleaned["pgs_id"].to_list())
        assert raw_ids == cleaned_ids


class TestRenameScoreColumns:
    def test_only_mapped_columns_retained(self, raw_scores_df: pl.DataFrame) -> None:
        renamed = rename_score_columns(raw_scores_df.lazy()).collect()
        assert "Polygenic Score (PGS) ID" not in renamed.columns
        assert "pgs_id" in renamed.columns


class TestNormalizeGenomeBuild:
    def test_normalizes_hg_variants(self) -> None:
        df = pl.DataFrame({"genome_build": ["hg19", "hg37", "hg38", "GRCh37", "NR", "NCBI36"]})
        result = normalize_genome_build(df.lazy()).collect()
        expected = ["GRCh37", "GRCh37", "GRCh38", "GRCh37", "NR", "GRCh36"]
        assert result["genome_build"].to_list() == expected


class TestCleanPerformanceMetrics:
    def test_produces_parsed_metric_columns(
        self, raw_perf_df: pl.DataFrame, raw_eval_df: pl.DataFrame
    ) -> None:
        cleaned = clean_performance_metrics(raw_perf_df, raw_eval_df).collect()
        for prefix in ["or", "hr", "beta", "auroc", "cindex"]:
            assert f"{prefix}_estimate" in cleaned.columns
            assert f"{prefix}_ci_lower" in cleaned.columns

    def test_has_sample_size_and_ancestry(
        self, raw_perf_df: pl.DataFrame, raw_eval_df: pl.DataFrame
    ) -> None:
        cleaned = clean_performance_metrics(raw_perf_df, raw_eval_df).collect()
        assert "n_individuals" in cleaned.columns
        assert "ancestry_broad" in cleaned.columns

    def test_raw_metric_columns_dropped(
        self, raw_perf_df: pl.DataFrame, raw_eval_df: pl.DataFrame
    ) -> None:
        cleaned = clean_performance_metrics(raw_perf_df, raw_eval_df).collect()
        for raw_col in ["or_raw", "hr_raw", "beta_raw", "auroc_raw", "cindex_raw"]:
            assert raw_col not in cleaned.columns

    def test_row_count_preserved(
        self, raw_perf_df: pl.DataFrame, raw_eval_df: pl.DataFrame
    ) -> None:
        cleaned = clean_performance_metrics(raw_perf_df, raw_eval_df).collect()
        assert cleaned.height == raw_perf_df.height


class TestBestPerformancePerScore:
    def test_one_row_per_pgs_id(
        self, raw_perf_df: pl.DataFrame, raw_eval_df: pl.DataFrame
    ) -> None:
        cleaned = clean_performance_metrics(raw_perf_df, raw_eval_df)
        best = best_performance_per_score(cleaned).collect()
        n_unique_ids = best["pgs_id"].n_unique()
        assert best.height == n_unique_ids

    def test_covers_most_scored_pgs_ids(
        self, raw_perf_df: pl.DataFrame, raw_eval_df: pl.DataFrame
    ) -> None:
        cleaned = clean_performance_metrics(raw_perf_df, raw_eval_df)
        best = best_performance_per_score(cleaned).collect()
        assert best.height > 5000


class TestPRSCatalog:
    def test_scores_returns_cleaned_lazyframe(self, catalog: PRSCatalog) -> None:
        lf = catalog.scores()
        df = lf.collect()
        assert "pgs_id" in df.columns
        assert "genome_build" in df.columns
        assert df.height > 5000

    def test_scores_filter_by_build(self, catalog: PRSCatalog) -> None:
        df_38 = catalog.scores(genome_build="GRCh38").collect()
        df_37 = catalog.scores(genome_build="GRCh37").collect()
        assert df_38.height < df_37.height
        builds_38 = set(df_38["genome_build"].unique().to_list())
        assert builds_38 == {"GRCh38"}

    def test_search_by_trait(self, catalog: PRSCatalog) -> None:
        results = catalog.search("diabetes").collect()
        assert results.height > 50
        traits = results["trait_reported"].str.to_lowercase().to_list()
        assert any("diabetes" in t for t in traits)

    def test_search_by_pgs_id(self, catalog: PRSCatalog) -> None:
        results = catalog.search("PGS000001").collect()
        assert results.height >= 1
        assert "PGS000001" in results["pgs_id"].to_list()

    def test_search_with_build_filter(self, catalog: PRSCatalog) -> None:
        all_results = catalog.search("breast cancer").collect()
        filtered = catalog.search("breast cancer", genome_build="GRCh38").collect()
        assert filtered.height <= all_results.height
        assert filtered.height > 0

    def test_score_info_row(self, catalog: PRSCatalog) -> None:
        info = catalog.score_info_row("PGS000001")
        assert info is not None
        assert info["pgs_id"] == "PGS000001"
        assert info["trait_reported"] == "Breast cancer"
        assert info["n_variants"] == 77

    def test_score_info_row_missing(self, catalog: PRSCatalog) -> None:
        info = catalog.score_info_row("PGS999999")
        assert info is None

    def test_best_performance(self, catalog: PRSCatalog) -> None:
        best = catalog.best_performance(pgs_id="PGS000001").collect()
        assert best.height == 1
        assert "or_estimate" in best.columns
        assert "auroc_estimate" in best.columns
        assert "n_individuals" in best.columns

    def test_percentile_with_auroc(self, catalog: PRSCatalog) -> None:
        best_all = catalog.best_performance().collect()
        with_auroc = best_all.filter(pl.col("auroc_estimate").is_not_null())
        assert with_auroc.height > 0

        pgs_id = with_auroc["pgs_id"][0]
        pct_zero = catalog.percentile(0.0, pgs_id)
        assert pct_zero is not None
        assert pct_zero == pytest.approx(50.0, abs=0.5)

        pct_pos = catalog.percentile(1.0, pgs_id)
        pct_neg = catalog.percentile(-1.0, pgs_id)
        assert pct_pos is not None and pct_neg is not None
        assert pct_pos > pct_zero
        assert pct_neg < pct_zero

    def test_percentile_with_explicit_std(self, catalog: PRSCatalog) -> None:
        pct = catalog.percentile(0.0, "PGS000001", mean=0.0, std=1.0)
        assert pct == pytest.approx(50.0, abs=0.1)

        pct_high = catalog.percentile(1.96, "PGS000001", mean=0.0, std=1.0)
        assert pct_high is not None
        assert pct_high == pytest.approx(97.5, abs=0.5)

    def test_percentile_no_auroc_returns_none(self, catalog: PRSCatalog) -> None:
        best = catalog.best_performance(pgs_id="PGS000001").collect()
        auroc = best["auroc_estimate"][0] if best.height > 0 else None
        if auroc is None:
            pct = catalog.percentile(1.0, "PGS000001")
            assert pct is None
