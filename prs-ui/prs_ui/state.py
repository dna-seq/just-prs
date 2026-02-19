"""Application state for the PRS UI.

Four state classes:
- AppState(rx.State): shared vars (active_tab, genome_build, cache_dir, etc.)
- MetadataGridState(LazyFrameGridMixin, AppState): metadata browser + scoring viewer grid
- ComputeGridState(LazyFrameGridMixin, AppState): compute PRS score selection grid
- GenomicGridState(LazyFrameGridMixin, AppState): normalized VCF genomic data grid

All grid states inherit LazyFrameGridMixin (mixin=True) and AppState, so each
gets its own independent set of reactive grid vars while sharing AppState vars.
"""

import csv
import io
from pathlib import Path
from typing import Any

import polars as pl
import reflex as rx
from reflex_mui_datagrid import LazyFrameGridMixin
from reflex_mui_datagrid.lazyframe_grid import _get_cache, apply_filter_model

from just_prs.ftp import (
    METADATA_FILES,
    download_metadata_sheet,
    download_scoring_as_parquet,
    stream_scoring_file,
)
from just_prs.normalize import VcfFilterConfig, normalize_vcf
from just_prs.prs import compute_prs
from just_prs.prs_catalog import PRSCatalog
from just_prs.scoring import resolve_cache_dir
from just_prs.vcf import detect_genome_build

SHEET_NAMES: list[str] = list(METADATA_FILES.keys())

SHEET_LABELS: dict[str, str] = {
    "scores": "Scores",
    "publications": "Publications",
    "efo_traits": "EFO Traits",
    "score_development_samples": "Dev Samples",
    "performance_metrics": "Performance",
    "evaluation_sample_sets": "Eval Samples",
    "cohorts": "Cohorts",
}


_catalog = PRSCatalog(cache_dir=resolve_cache_dir())


class AppState(rx.State):
    """Shared app state: tab selection, genome build, cache dir."""

    selected_sheet: str = "scores"
    cache_dir: str = str(resolve_cache_dir())
    status_message: str = ""
    pgs_id_input: str = "PGS000001"
    genome_build: str = "GRCh38"
    active_tab: str = "compute"

    def set_pgs_id(self, value: str) -> None:
        self.pgs_id_input = value

    def set_genome_build(self, value: str) -> None:
        self.genome_build = value

    def set_active_tab(self, value: str) -> None:
        self.active_tab = value


class MetadataGridState(LazyFrameGridMixin, AppState):
    """Grid state for the metadata browser + scoring file viewer."""

    metadata_selected_ids: list[str] = []

    def load_sheet(self, sheet: str) -> Any:
        """Download (or load cached) a metadata sheet and display in the metadata grid."""
        self.selected_sheet = sheet
        self.status_message = f"Loading {SHEET_LABELS.get(sheet, sheet)}..."
        cache_path = Path(self.cache_dir) / "metadata" / f"{sheet}.parquet"
        df = download_metadata_sheet(sheet, cache_path)  # type: ignore[arg-type]
        lf = df.lazy()
        yield from self.set_lazyframe(lf, chunk_size=500)
        self.status_message = f"Loaded {SHEET_LABELS.get(sheet, sheet)} ({df.height} rows)"

    def load_scoring(self) -> Any:
        """Stream a scoring file for the given PGS ID and display in the metadata grid."""
        pgs_id = self.pgs_id_input.strip().upper()
        if not pgs_id:
            self.status_message = "Please enter a PGS ID."
            return
        self.status_message = f"Streaming {pgs_id} ({self.genome_build})..."
        yield
        lf = stream_scoring_file(pgs_id, genome_build=self.genome_build)
        yield from self.set_lazyframe(lf, chunk_size=500)
        row_count = lf.select(pl.len()).collect().item()
        self.status_message = f"Loaded {pgs_id} ({row_count} variants)"

    def handle_metadata_row_selection(self, model: dict) -> None:
        """Track selected PGS IDs from metadata grid checkbox selection."""
        selection_type: str = model.get("type", "include")
        raw_ids: list = model.get("ids", [])
        selected_row_ids: set[int] = {int(i) for i in raw_ids}

        pgs_ids: list[str] = []
        for row in self.lf_grid_rows:
            row_id = row.get("__row_id__")
            in_set = (int(row_id) in selected_row_ids) if row_id is not None else False
            if (selection_type == "include" and in_set) or (
                selection_type == "exclude" and not in_set
            ):
                pgs_id = row.get("Polygenic Score (PGS) ID")
                if pgs_id:
                    pgs_ids.append(str(pgs_id))
        self.metadata_selected_ids = pgs_ids

    def download_selected_scoring_files(self) -> Any:
        """Download scoring files for selected PGS IDs to the local cache."""
        if not self.metadata_selected_ids:
            self.status_message = "No scores selected."
            return
        total = len(self.metadata_selected_ids)
        output_dir = Path(self.cache_dir) / "scoring" / self.genome_build
        self.status_message = f"Saving {total} scoring file(s) to cache..."
        yield
        for i, pgs_id in enumerate(self.metadata_selected_ids, start=1):
            self.status_message = f"Saving {i}/{total}: {pgs_id}..."
            yield
            download_scoring_as_parquet(pgs_id, output_dir, genome_build=self.genome_build)
        self.status_message = f"Saved {total} scoring file(s) to {output_dir}"


class GenomicGridState(LazyFrameGridMixin, AppState):
    """Grid state for viewing normalized VCF genomic data.

    After a VCF is uploaded and normalized, the resulting Parquet is loaded
    into this grid for interactive browsing.  The normalized parquet path
    is also used by ``ComputeGridState`` to feed pre-filtered genotypes
    into PRS computation.
    """

    normalized_parquet_path: str = ""
    normalize_status: str = ""
    genomic_loaded: bool = False
    genomic_row_count: int = 0

    _normalizing: bool = False

    def normalize_uploaded_vcf(self, vcf_path: str, sex: str = "") -> Any:
        """Run VCF normalization and load the result into this grid."""
        if not vcf_path:
            self.normalize_status = "No VCF path provided."
            return
        self._normalizing = True
        self.normalize_status = "Normalizing VCF..."
        yield

        src = Path(vcf_path)
        out_dir = Path(self.cache_dir) / "normalized"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_name = src.stem.removesuffix(".vcf") + ".parquet"
        output_path = out_dir / parquet_name

        config = VcfFilterConfig(
            pass_filters=["PASS", "."],
            sex=sex if sex else None,
        )
        normalize_vcf(src, output_path, config=config)

        self.normalized_parquet_path = str(output_path)

        lf = pl.scan_parquet(output_path)
        yield from self.set_lazyframe(lf, chunk_size=500)
        row_count = lf.select(pl.len()).collect().item()
        self.genomic_row_count = row_count
        self.genomic_loaded = True
        self._normalizing = False
        self.normalize_status = f"Normalized: {row_count:,} variants"
        self.status_message = f"VCF normalized: {row_count:,} variants"


class ComputeGridState(LazyFrameGridMixin, AppState):
    """Independent grid state for the Compute PRS tab.

    Inherits shared vars (genome_build, cache_dir, status_message) from AppState.
    Gets its own independent LazyFrameGridMixin vars (lf_grid_rows, lf_grid_loaded, etc.)
    because LazyFrameGridMixin uses mixin=True.
    """

    selected_pgs_ids: list[str] = []
    vcf_filename: str = ""
    detected_build: str = ""
    build_detection_message: str = ""
    prs_results: list[dict] = []
    prs_computing: bool = False
    prs_progress: int = 0
    low_match_warning: bool = False
    compute_scores_loaded: bool = False

    _scores_initialized: bool = False
    _vcf_path: str = ""
    _compute_scores_lf: pl.LazyFrame | None = None

    def initialize(self) -> Any:
        """Auto-load cleaned scores on first page visit."""
        if self._scores_initialized:
            return
        self._scores_initialized = True
        yield from self.load_compute_scores()

    def set_genome_build(self, value: str) -> Any:
        """Set genome build and reload compute scores if already loaded."""
        self.genome_build = value
        if self.compute_scores_loaded:
            yield from self.load_compute_scores()

    def load_compute_scores(self) -> Any:
        """Load cleaned scores into the compute grid, filtered by genome build."""
        self.status_message = "Loading scores for selection..."
        yield
        lf = _catalog.scores(genome_build=self.genome_build)
        self._compute_scores_lf = lf
        self.compute_scores_loaded = True
        self.selected_pgs_ids = []
        yield from self.set_lazyframe(lf, chunk_size=500)
        total = lf.select(pl.len()).collect().item()
        self.status_message = f"Loaded {total} scores for {self.genome_build}"

    def handle_compute_row_selection(self, model: dict) -> None:
        """Track selected PGS IDs from compute grid checkbox selection.

        Handles MUI DataGrid v8 selection model:
        - {type: "include", ids: [...]} — only listed rows are selected
        - {type: "exclude", ids: [...]} — all rows EXCEPT listed are selected
        - {type: "exclude", ids: []} — all rows selected (header checkbox)
        """
        self.status_message = f"Selection event: {model}"

        selection_type: str = model.get("type", "include")
        raw_ids: list = model.get("ids", [])
        selected_row_ids: set[int] = {int(i) for i in raw_ids}

        if selection_type == "exclude" and not selected_row_ids:
            self.select_filtered_scores()
            return
        if selection_type == "include" and not selected_row_ids:
            self.selected_pgs_ids = []
            return

        pgs_ids: list[str] = []
        for row in self.lf_grid_rows:
            row_id = row.get("__row_id__")
            in_set = (int(row_id) in selected_row_ids) if row_id is not None else False
            if (selection_type == "include" and in_set) or (
                selection_type == "exclude" and not in_set
            ):
                pgs_id = row.get("pgs_id")
                if pgs_id:
                    pgs_ids.append(str(pgs_id))
        self.selected_pgs_ids = pgs_ids
        self.status_message = f"Selected {len(pgs_ids)} scores (from {len(self.lf_grid_rows)} loaded rows)"

    def select_filtered_scores(self) -> None:
        """Select PGS IDs matching the current grid filter (or all if no filter active)."""
        if self._compute_scores_lf is None:
            return
        lf = self._compute_scores_lf
        if self._lf_grid_filter and self._lf_grid_filter.get("items"):
            cache = _get_cache(self._lf_grid_cache_id) if self._lf_grid_cache_id else None
            schema = cache.schema if cache else None
            lf = apply_filter_model(lf, self._lf_grid_filter, schema)
        ids = lf.select("pgs_id").collect()["pgs_id"].to_list()
        self.selected_pgs_ids = ids
        self.status_message = f"Selected {len(ids)} scores"

    def deselect_all_scores(self) -> None:
        """Clear all selected PGS IDs."""
        self.selected_pgs_ids = []
        self.status_message = ""

    async def handle_vcf_upload(self, files: list[rx.UploadFile]) -> None:
        """Save an uploaded VCF file and attempt genome build detection."""
        if not files:
            return
        upload_file = files[0]
        filename = upload_file.filename or "uploaded.vcf"
        upload_dir = rx.get_upload_dir()
        dest = upload_dir / filename
        contents = await upload_file.read()
        dest.write_bytes(contents)

        self._vcf_path = str(dest)
        self.vcf_filename = filename

        detected = detect_genome_build(dest)
        if detected is not None:
            self.detected_build = detected
            self.genome_build = detected
            self.build_detection_message = f"Detected genome build: {detected}"
        else:
            self.detected_build = ""
            self.build_detection_message = (
                "Could not detect genome build from VCF header. "
                "Please select it manually."
            )
        self.prs_results = []
        self.low_match_warning = False
        self.status_message = f"Uploaded {filename}"

        return GenomicGridState.normalize_uploaded_vcf(str(dest))

    async def compute_selected_prs(self) -> Any:
        """Compute PRS for uploaded VCF against all selected PGS IDs.

        Uses PRSCatalog for metadata lookup (no REST API calls) and for
        performance metrics from pre-downloaded bulk metadata.
        When a normalized parquet exists (from GenomicGridState), uses it
        as a pre-filtered genotypes source instead of re-reading the raw VCF.
        """
        if not self._vcf_path:
            self.status_message = "Please upload a VCF file first."
            return
        if not self.selected_pgs_ids:
            self.status_message = "No PGS scores selected. Load and select scores above."
            return

        total = len(self.selected_pgs_ids)
        self.prs_computing = True
        self.prs_progress = 0
        self.prs_results = []
        self.low_match_warning = False
        self.status_message = f"Computing PRS for {total} score(s)..."
        yield

        cache = Path(self.cache_dir) / "scores"
        results: list[dict] = []
        any_low_match = False

        genomic_state = await self.get_state(GenomicGridState)
        normalized_path = genomic_state.normalized_parquet_path
        pre_genotypes: pl.LazyFrame | None = None
        if normalized_path and Path(normalized_path).exists():
            pre_genotypes = pl.scan_parquet(normalized_path)

        best_perf_df = _catalog.best_performance().collect()

        for i, pgs_id in enumerate(self.selected_pgs_ids, start=1):
            self.prs_progress = round(i / total * 100)
            self.status_message = f"Computing {i}/{total}: {pgs_id}..."
            yield

            info = _catalog.score_info_row(pgs_id)
            trait = info["trait_reported"] if info else None

            result = compute_prs(
                vcf_path=self._vcf_path,
                scoring_file=pgs_id,
                genome_build=self.genome_build,
                cache_dir=cache,
                pgs_id=pgs_id,
                trait_reported=trait,
                genotypes_lf=pre_genotypes,
            )

            match_pct = round(result.match_rate * 100, 1)
            if match_pct < 10:
                match_color = "red"
            elif match_pct < 50:
                match_color = "orange"
            else:
                match_color = "green"

            auroc_val: float | None = None
            ancestry_str = ""
            n_individuals: int | None = None
            perf_rows = best_perf_df.filter(pl.col("pgs_id") == pgs_id)
            effect_size_str = ""
            classification_str = ""
            if perf_rows.height > 0:
                p = perf_rows.row(0, named=True)
                effect_size_str = _format_effect_size(p)
                classification_str = _format_classification(p)
                auroc_val = p.get("auroc_estimate")
                ancestry_str = p.get("ancestry_broad") or ""
                n_individuals = p.get("n_individuals")

            interp = _interpret_prs_result(result.percentile, result.match_rate, auroc_val)

            row: dict[str, Any] = {
                "pgs_id": result.pgs_id,
                "trait": result.trait_reported or "",
                "score": round(result.score, 6),
                "percentile": f"{result.percentile:.1f}" if result.percentile is not None else "",
                "has_allele_frequencies": result.has_allele_frequencies,
                "match_rate": match_pct,
                "match_color": match_color,
                "variants_matched": result.variants_matched,
                "variants_total": result.variants_total,
                "effect_size": effect_size_str,
                "classification": classification_str,
                "auroc": f"{auroc_val:.3f}" if auroc_val is not None else "",
                "quality_label": interp["quality_label"],
                "quality_color": interp["quality_color"],
                "summary": interp["summary"],
                "ancestry": ancestry_str,
                "n_individuals": n_individuals if n_individuals is not None else 0,
            }

            if result.match_rate < 0.1:
                any_low_match = True

            results.append(row)

        self.prs_results = results
        self.low_match_warning = any_low_match
        self.prs_computing = False
        self.prs_progress = 100
        self.status_message = f"Computed {total} PRS score(s)"

    def download_prs_results_csv(self) -> Any:
        """Build a CSV from prs_results and trigger a browser download."""
        if not self.prs_results:
            return
        columns = [
            "pgs_id", "trait", "score", "percentile", "auroc", "quality_label",
            "match_rate", "variants_matched", "variants_total",
            "effect_size", "classification", "ancestry",
            "n_individuals", "summary",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in self.prs_results:
            writer.writerow(row)
        return rx.download(data=buf.getvalue(), filename="prs_results.csv")


def _format_effect_size(perf_row: dict[str, Any]) -> str:
    """Format the best available effect size metric from a cleaned performance row."""
    for prefix, label in [("or", "OR"), ("hr", "HR"), ("beta", "Beta")]:
        est = perf_row.get(f"{prefix}_estimate")
        if est is not None:
            ci_lo = perf_row.get(f"{prefix}_ci_lower")
            ci_hi = perf_row.get(f"{prefix}_ci_upper")
            se = perf_row.get(f"{prefix}_se")
            result = f"{label}={est:.2f}"
            if ci_lo is not None and ci_hi is not None:
                result += f" [{ci_lo:.2f}-{ci_hi:.2f}]"
            elif se is not None:
                result += f" (SE={se:.2f})"
            return result
    return ""


def _format_classification(perf_row: dict[str, Any]) -> str:
    """Format the best available classification metric from a cleaned performance row."""
    for prefix, label in [("auroc", "AUROC"), ("cindex", "C-index")]:
        est = perf_row.get(f"{prefix}_estimate")
        if est is not None:
            ci_lo = perf_row.get(f"{prefix}_ci_lower")
            ci_hi = perf_row.get(f"{prefix}_ci_upper")
            result = f"{label}={est:.3f}"
            if ci_lo is not None and ci_hi is not None:
                result += f" [{ci_lo:.3f}-{ci_hi:.3f}]"
            return result
    return ""


def _classify_model_quality(
    match_rate: float,
    auroc: float | None,
) -> tuple[str, str]:
    """Classify overall model quality from match rate and AUROC.

    Returns (label, color) for the quality badge.
    """
    if match_rate < 0.1:
        return "Very Low", "red"
    if auroc is not None:
        if match_rate >= 0.5 and auroc >= 0.7:
            return "High", "green"
        if match_rate >= 0.5 and auroc >= 0.6:
            return "Moderate", "yellow"
    if match_rate >= 0.5:
        return "Moderate", "yellow"
    return "Low", "orange"


def _interpret_prs_result(
    percentile: float | None,
    match_rate: float,
    auroc: float | None,
) -> dict[str, str]:
    """Produce human-readable interpretation of a single PRS result.

    Returns a dict with keys: quality_label, quality_color, summary.

    When a theoretical percentile is available (computed from allele frequencies
    in the scoring file), it is included in the summary as an approximate
    population position.
    """
    quality_label, quality_color = _classify_model_quality(match_rate, auroc)

    parts: list[str] = []

    if percentile is not None:
        parts.append(
            f"Estimated percentile: {percentile:.1f}% "
            f"(theoretical, from allele frequencies in the scoring file)."
        )

    if auroc is not None:
        if auroc >= 0.7:
            parts.append(f"Good predictive model (AUROC={auroc:.3f}).")
        elif auroc >= 0.6:
            parts.append(f"Moderate predictive model (AUROC={auroc:.3f}).")
        else:
            parts.append(f"Weak predictive model (AUROC={auroc:.3f}).")
    else:
        parts.append("No AUROC available to assess model accuracy.")

    if match_rate < 0.1:
        parts.append(f"Only {match_rate * 100:.0f}% of scoring variants matched — results may be unreliable.")
    elif match_rate < 0.5:
        parts.append(f"{match_rate * 100:.0f}% of scoring variants matched — interpret with caution.")
    else:
        parts.append(f"{match_rate * 100:.0f}% of scoring variants matched.")

    if percentile is None:
        parts.append(
            "No allele frequencies in scoring file — percentile not available. "
            "Compare your score to a matched reference cohort for meaningful interpretation."
        )

    return {
        "quality_label": quality_label,
        "quality_color": quality_color,
        "summary": " ".join(parts),
    }
