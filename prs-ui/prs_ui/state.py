"""Application state for the PRS UI, built on LazyFrameGridMixin."""

import os
from pathlib import Path
from typing import Any

import polars as pl
import reflex as rx
from reflex_mui_datagrid import LazyFrameGridMixin

from just_prs.ftp import (
    METADATA_FILES,
    download_metadata_sheet,
    download_scoring_as_parquet,
    stream_scoring_file,
)
from just_prs.prs import compute_prs
from just_prs.prs_catalog import PRSCatalog
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


def _resolve_cache_dir() -> Path:
    raw = os.environ.get("PRS_CACHE_DIR", "")
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "just-prs"


_catalog = PRSCatalog(cache_dir=_resolve_cache_dir())


class AppState(LazyFrameGridMixin):
    """Reactive state for browsing PGS Catalog metadata and scoring files."""

    selected_sheet: str = "scores"
    cache_dir: str = str(_resolve_cache_dir())
    status_message: str = ""
    pgs_id_input: str = "PGS000001"
    genome_build: str = "GRCh38"
    active_tab: str = "metadata"
    selected_pgs_ids: list[str] = []

    vcf_filename: str = ""
    detected_build: str = ""
    build_detection_message: str = ""
    prs_results: list[dict] = []
    prs_computing: bool = False
    low_match_warning: bool = False

    compute_scores_rows: list[dict[str, str]] = []
    compute_scores_loaded: bool = False
    compute_search_term: str = ""
    compute_total_count: int = 0
    compute_page: int = 0
    compute_page_size: int = 50

    _scores_initialized: bool = False
    _vcf_path: str = ""
    _compute_filtered_lf: pl.LazyFrame | None = None

    def initialize(self) -> Any:
        """Load the scores sheet once per session; skip on subsequent on_load calls."""
        if self._scores_initialized:
            return
        self._scores_initialized = True
        yield from self.load_sheet("scores")

    def load_sheet(self, sheet: str) -> Any:
        """Download (or load cached) a metadata sheet and display in the grid.

        The metadata tab shows raw PGS Catalog columns for general browsing.
        """
        self.selected_sheet = sheet
        self.status_message = f"Loading {SHEET_LABELS.get(sheet, sheet)}..."
        cache_path = Path(self.cache_dir) / "metadata" / f"{sheet}.parquet"
        df = download_metadata_sheet(sheet, cache_path)  # type: ignore[arg-type]
        lf = df.lazy()
        yield from self.set_lazyframe(lf, chunk_size=500)
        self.status_message = f"Loaded {SHEET_LABELS.get(sheet, sheet)} ({df.height} rows)"

    def load_scoring(self) -> Any:
        """Stream a scoring file for the given PGS ID and display in the grid."""
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

    def set_pgs_id(self, value: str) -> None:
        self.pgs_id_input = value

    def set_genome_build(self, value: str) -> Any:
        self.genome_build = value
        if self.compute_scores_loaded:
            yield from self.load_compute_scores()

    def set_active_tab(self, value: str) -> None:
        self.active_tab = value

    def _apply_search_filter(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply text search filter across key cleaned columns."""
        term = self.compute_search_term.strip().lower()
        if not term:
            return lf
        return lf.filter(
            pl.col("pgs_id").str.to_lowercase().str.contains(term, literal=True)
            | pl.col("name").str.to_lowercase().str.contains(term, literal=True)
            | pl.col("trait_reported").str.to_lowercase().str.contains(term, literal=True)
            | pl.col("trait_efo").str.to_lowercase().str.contains(term, literal=True)
        )

    def _lf_to_rows(self, df: pl.DataFrame) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for row in df.iter_rows(named=True):
            rows.append({
                "pgs_id": str(row.get("pgs_id", "")),
                "name": str(row.get("name", "")),
                "trait": str(row.get("trait_reported", "")),
                "build": str(row.get("genome_build", "")),
                "variants": str(row.get("n_variants", "")),
            })
        return rows

    def _refresh_compute_page(self) -> None:
        """Re-apply search filter and pagination, update visible rows."""
        if self._compute_filtered_lf is None:
            return
        searched = self._apply_search_filter(self._compute_filtered_lf)
        searched_df = searched.collect()
        self.compute_total_count = searched_df.height
        offset = self.compute_page * self.compute_page_size
        page_df = searched_df.slice(offset, self.compute_page_size)
        self.compute_scores_rows = self._lf_to_rows(page_df)

    def load_compute_scores(self) -> Any:
        """Load cleaned scores metadata filtered by the current genome build."""
        self.status_message = "Loading scores for selection..."
        yield
        self._compute_filtered_lf = _catalog.scores(genome_build=self.genome_build)
        self.compute_scores_loaded = True
        self.selected_pgs_ids = []
        self.compute_page = 0
        self.compute_search_term = ""
        self._refresh_compute_page()
        self.status_message = f"Found {self.compute_total_count} scores for {self.genome_build}"

    def set_compute_search_term(self, value: str) -> None:
        self.compute_search_term = value
        self.compute_page = 0
        self._refresh_compute_page()

    def compute_next_page(self) -> None:
        max_page = max(0, (self.compute_total_count - 1) // self.compute_page_size)
        if self.compute_page < max_page:
            self.compute_page += 1
            self._refresh_compute_page()

    def compute_prev_page(self) -> None:
        if self.compute_page > 0:
            self.compute_page -= 1
            self._refresh_compute_page()

    def toggle_compute_score(self, pgs_id: str) -> None:
        """Toggle selection of a PGS ID in the compute scores list."""
        if pgs_id in self.selected_pgs_ids:
            self.selected_pgs_ids = [p for p in self.selected_pgs_ids if p != pgs_id]
        else:
            self.selected_pgs_ids = [*self.selected_pgs_ids, pgs_id]

    def select_all_visible_scores(self) -> None:
        """Select all scores matching the current search filter (all pages)."""
        if self._compute_filtered_lf is None:
            return
        searched = self._apply_search_filter(self._compute_filtered_lf)
        ids = searched.select("pgs_id").collect()["pgs_id"].to_list()
        self.selected_pgs_ids = ids

    def deselect_all_scores(self) -> None:
        self.selected_pgs_ids = []

    def handle_scores_row_selection(self, model: dict) -> None:
        """Track selected PGS IDs when rows are checked in the scores grid.

        MUI DataGrid v8 passes { type: 'include'|'exclude', ids: [GridRowId, ...] }.
        GridRowId values may be strings even when the underlying data is integers,
        so we normalise both sides to int before comparing.
        """
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
        self.selected_pgs_ids = pgs_ids

    def download_selected_scoring_files(self) -> Any:
        """Download scoring files for selected PGS IDs to the local cache."""
        if not self.selected_pgs_ids:
            self.status_message = "No scores selected."
            return
        total = len(self.selected_pgs_ids)
        output_dir = Path(self.cache_dir) / "scoring" / self.genome_build
        self.status_message = f"Saving {total} scoring file(s) to cache..."
        yield
        for i, pgs_id in enumerate(self.selected_pgs_ids, start=1):
            self.status_message = f"Saving {i}/{total}: {pgs_id}..."
            yield
            download_scoring_as_parquet(pgs_id, output_dir, genome_build=self.genome_build)
        self.status_message = f"Saved {total} scoring file(s) to {output_dir}"

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

    def compute_selected_prs(self) -> Any:
        """Compute PRS for uploaded VCF against all selected PGS IDs.

        Uses PRSCatalog for metadata lookup (no REST API calls) and for
        performance metrics from pre-downloaded bulk metadata.
        """
        if not self._vcf_path:
            self.status_message = "Please upload a VCF file first."
            return
        if not self.selected_pgs_ids:
            self.status_message = "No PGS scores selected. Load and select scores above."
            return

        total = len(self.selected_pgs_ids)
        self.prs_computing = True
        self.prs_results = []
        self.low_match_warning = False
        self.status_message = f"Computing PRS for {total} score(s)..."
        yield

        cache = Path(self.cache_dir) / "scores"
        results: list[dict] = []
        any_low_match = False

        best_perf_df = _catalog.best_performance().collect()

        for i, pgs_id in enumerate(self.selected_pgs_ids, start=1):
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
            )

            match_pct = round(result.match_rate * 100, 1)
            if match_pct < 10:
                match_color = "red"
            elif match_pct < 50:
                match_color = "orange"
            else:
                match_color = "green"

            row: dict[str, Any] = {
                "pgs_id": result.pgs_id,
                "trait": result.trait_reported or "",
                "score": round(result.score, 6),
                "match_rate": match_pct,
                "match_color": match_color,
                "variants_matched": result.variants_matched,
                "variants_total": result.variants_total,
                "effect_size": "",
                "classification": "",
            }

            if result.match_rate < 0.1:
                any_low_match = True

            perf_rows = best_perf_df.filter(pl.col("pgs_id") == pgs_id)
            if perf_rows.height > 0:
                p = perf_rows.row(0, named=True)
                row["effect_size"] = _format_effect_size(p)
                row["classification"] = _format_classification(p)

            results.append(row)

        self.prs_results = results
        self.low_match_warning = any_low_match
        self.prs_computing = False
        self.status_message = f"Computed {total} PRS score(s)"


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
