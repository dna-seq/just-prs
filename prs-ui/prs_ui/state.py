"""Application state for the PRS UI.

State hierarchy:
- AppState(rx.State): shared vars (active_tab, genome_build, cache_dir, etc.)
- MetadataGridState(LazyFrameGridMixin, AppState): metadata browser + scoring viewer grid
- GenomicGridState(LazyFrameGridMixin, AppState): normalized VCF genomic data grid
- PRSComputeStateMixin(rx.State, mixin=True): reusable PRS computation logic
- ComputeGridState(PRSComputeStateMixin, LazyFrameGridMixin, AppState): standalone compute page

PRSComputeStateMixin is designed for reuse: any Reflex app can inherit it
into a concrete state class, provide genotype data via ``set_prs_genotypes_lf()``
(preferred, pass a polars LazyFrame) or ``prs_genotypes_path`` (fallback, string
path), and get full PRS computation with quality assessment.
"""

import csv
import io
import os
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
from just_prs.quality import (
    classify_model_quality,
    format_classification,
    format_effect_size,
    interpret_prs_result,
)
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


def _resolve_cache_dir() -> Path:
    raw = os.environ.get("PRS_CACHE_DIR", "")
    if raw:
        return Path(raw)
    return resolve_cache_dir()


_catalog = PRSCatalog(cache_dir=_resolve_cache_dir())


class AppState(rx.State):
    """Shared app state: tab selection, genome build, cache dir."""

    selected_sheet: str = "scores"
    cache_dir: str = str(_resolve_cache_dir())
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


class PRSComputeStateMixin(rx.State, mixin=True):
    """Reusable mixin for PRS score selection, computation, and result display.

    Designed for inheritance: any Reflex state class that also inherits
    ``LazyFrameGridMixin`` can mix this in to get the full PRS workflow.

    Input contract -- the host app must provide genotype data via one of
    (in order of preference):
    1. **LazyFrame (recommended)** -- call ``set_prs_genotypes_lf(lf)`` with a
       ``pl.scan_parquet()`` LazyFrame.  Memory-efficient and avoids redundant
       I/O when computing multiple scores.
    2. **Parquet path (fallback)** -- set ``prs_genotypes_path`` to a string.
       The mixin calls ``pl.scan_parquet()`` internally if no LazyFrame is set.

    The mixin reads ``genome_build``, ``cache_dir``, and ``status_message``
    from whatever parent state provides them (e.g. AppState).
    """

    selected_pgs_ids: list[str] = []
    prs_results: list[dict] = []
    prs_computing: bool = False
    prs_progress: int = 0
    low_match_warning: bool = False
    compute_scores_loaded: bool = False
    prs_genotypes_path: str = ""

    _scores_initialized: bool = False
    _compute_scores_lf: pl.LazyFrame | None = None
    _prs_genotypes_lf: pl.LazyFrame | None = None

    def set_prs_genotypes_lf(self, lf: pl.LazyFrame) -> None:
        """Provide a pre-loaded genotypes LazyFrame for PRS computation."""
        self._prs_genotypes_lf = lf

    def _get_genotypes_lf(self) -> pl.LazyFrame | None:
        """Resolve genotypes: explicit LazyFrame first, then parquet path."""
        if self._prs_genotypes_lf is not None:
            return self._prs_genotypes_lf
        if self.prs_genotypes_path and Path(self.prs_genotypes_path).exists():
            return pl.scan_parquet(self.prs_genotypes_path)
        return None

    def initialize_prs(self) -> Any:
        """Auto-load cleaned scores on first access."""
        if self._scores_initialized:
            return
        self._scores_initialized = True
        yield from self.load_compute_scores()

    def set_prs_genome_build(self, value: str) -> Any:
        """Set genome build and reload compute scores if already loaded."""
        self.genome_build = value  # type: ignore[attr-defined]
        if self.compute_scores_loaded:
            yield from self.load_compute_scores()

    def load_compute_scores(self) -> Any:
        """Load cleaned scores into the compute grid, filtered by genome build."""
        self.status_message = "Loading scores for selection..."  # type: ignore[attr-defined]
        yield
        lf = _catalog.scores(genome_build=self.genome_build)  # type: ignore[attr-defined]
        self._compute_scores_lf = lf
        self.compute_scores_loaded = True
        self.selected_pgs_ids = []
        yield from self.set_lazyframe(lf, chunk_size=500)  # type: ignore[attr-defined]
        total = lf.select(pl.len()).collect().item()
        self.status_message = f"Loaded {total} scores for {self.genome_build}"  # type: ignore[attr-defined]

    def handle_compute_row_selection(self, model: dict) -> None:
        """Track selected PGS IDs from compute grid checkbox selection.

        Handles MUI DataGrid v8 selection model:
        - {type: "include", ids: [...]} -- only listed rows are selected
        - {type: "exclude", ids: [...]} -- all rows EXCEPT listed are selected
        - {type: "exclude", ids: []} -- all rows selected (header checkbox)
        """
        self.status_message = f"Selection event: {model}"  # type: ignore[attr-defined]

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
        for row in self.lf_grid_rows:  # type: ignore[attr-defined]
            row_id = row.get("__row_id__")
            in_set = (int(row_id) in selected_row_ids) if row_id is not None else False
            if (selection_type == "include" and in_set) or (
                selection_type == "exclude" and not in_set
            ):
                pgs_id = row.get("pgs_id")
                if pgs_id:
                    pgs_ids.append(str(pgs_id))
        self.selected_pgs_ids = pgs_ids
        self.status_message = f"Selected {len(pgs_ids)} scores (from {len(self.lf_grid_rows)} loaded rows)"  # type: ignore[attr-defined]

    def select_filtered_scores(self) -> None:
        """Select PGS IDs matching the current grid filter (or all if no filter active)."""
        if self._compute_scores_lf is None:
            return
        lf = self._compute_scores_lf
        if self._lf_grid_filter and self._lf_grid_filter.get("items"):  # type: ignore[attr-defined]
            cache = _get_cache(self._lf_grid_cache_id) if self._lf_grid_cache_id else None  # type: ignore[attr-defined]
            schema = cache.schema if cache else None
            lf = apply_filter_model(lf, self._lf_grid_filter, schema)  # type: ignore[attr-defined]
        ids = lf.select("pgs_id").collect()["pgs_id"].to_list()
        self.selected_pgs_ids = ids
        self.status_message = f"Selected {len(ids)} scores"  # type: ignore[attr-defined]

    def deselect_all_scores(self) -> None:
        """Clear all selected PGS IDs."""
        self.selected_pgs_ids = []
        self.status_message = ""  # type: ignore[attr-defined]

    def compute_selected_prs(self) -> Any:
        """Compute PRS for all selected PGS IDs using available genotype data.

        Uses PRSCatalog for metadata lookup (no REST API calls) and for
        performance metrics from pre-downloaded bulk metadata.
        """
        if not self.selected_pgs_ids:
            self.status_message = "No PGS scores selected. Load and select scores above."  # type: ignore[attr-defined]
            return

        pre_genotypes = self._get_genotypes_lf()

        total = len(self.selected_pgs_ids)
        self.prs_computing = True
        self.prs_progress = 0
        self.prs_results = []
        self.low_match_warning = False
        self.status_message = f"Computing PRS for {total} score(s)..."  # type: ignore[attr-defined]
        yield

        cache = Path(self.cache_dir) / "scores"  # type: ignore[attr-defined]
        results: list[dict] = []
        any_low_match = False

        best_perf_df = _catalog.best_performance().collect()

        for i, pgs_id in enumerate(self.selected_pgs_ids, start=1):
            self.prs_progress = round(i / total * 100)
            self.status_message = f"Computing {i}/{total}: {pgs_id}..."  # type: ignore[attr-defined]
            yield

            info = _catalog.score_info_row(pgs_id)
            trait = info["trait_reported"] if info else None

            vcf_path = self.prs_genotypes_path or ""
            result = compute_prs(
                vcf_path=vcf_path,
                scoring_file=pgs_id,
                genome_build=self.genome_build,  # type: ignore[attr-defined]
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
                effect_size_str = format_effect_size(p)
                classification_str = format_classification(p)
                auroc_val = p.get("auroc_estimate")
                ancestry_str = p.get("ancestry_broad") or ""
                n_individuals = p.get("n_individuals")

            interp = interpret_prs_result(result.percentile, result.match_rate, auroc_val)

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
        self.status_message = f"Computed {total} PRS score(s)"  # type: ignore[attr-defined]

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


class ComputeGridState(PRSComputeStateMixin, LazyFrameGridMixin, AppState):
    """Concrete state for the standalone Compute PRS page.

    Adds VCF-upload-specific behavior on top of PRSComputeStateMixin.
    Inherits shared vars from AppState and grid vars from LazyFrameGridMixin.
    """

    vcf_filename: str = ""
    detected_build: str = ""
    build_detection_message: str = ""

    _vcf_path: str = ""

    def initialize(self) -> Any:
        """Auto-load cleaned scores on first page visit."""
        yield from self.initialize_prs()

    def set_genome_build(self, value: str) -> Any:
        """Set genome build and reload compute scores if already loaded."""
        yield from self.set_prs_genome_build(value)

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
        self.prs_genotypes_path = str(dest)

        detected = detect_genome_build(dest)
        needs_score_reload = False
        if detected is not None:
            self.detected_build = detected
            if detected != self.genome_build:
                needs_score_reload = self.compute_scores_loaded
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

        events: list = [GenomicGridState.normalize_uploaded_vcf(str(dest))]
        if needs_score_reload:
            events.append(ComputeGridState.load_compute_scores)
        return events

    async def compute_selected_prs(self) -> Any:
        """Override to resolve genotypes from GenomicGridState, then delegate."""
        if not self._vcf_path:
            self.status_message = "Please upload a VCF file first."
            return

        genomic_state = await self.get_state(GenomicGridState)
        normalized_path = genomic_state.normalized_parquet_path
        if normalized_path and Path(normalized_path).exists():
            self.prs_genotypes_path = normalized_path
            self._prs_genotypes_lf = pl.scan_parquet(normalized_path)
        else:
            self.prs_genotypes_path = self._vcf_path

        gen = PRSComputeStateMixin.compute_selected_prs(self)
        if gen is not None:
            for event in gen:
                yield event
