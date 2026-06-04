"""Concrete application states for the PRS UI demo app.

State hierarchy:
- AppState(rx.State): shared vars (active_tab, genome_build, cache_dir, etc.)
- MetadataGridState(LazyFrameGridMixin, AppState): metadata browser + scoring viewer grid
- GenomicGridState(LazyFrameGridMixin, AppState): normalized VCF genomic data grid
- ComputeGridState(PRSComputeStateMixin, LazyFrameGridMixin, AppState): standalone compute page
- TraitBrowserState(PRSComputeStateMixin, LazyFrameGridMixin, AppState): trait-based browser

The reusable ``PRSComputeStateMixin`` lives in ``prs_ui.mixin`` so that
consumer apps can import it without registering these demo-only states.
"""

from pathlib import Path
from typing import Any

import polars as pl
import reflex as rx
from reflex_mui_datagrid import LazyFrameGridMixin

from just_prs.ftp import (
    download_metadata_sheet,
    download_scoring_as_parquet,
    stream_scoring_file,
)
from just_prs.normalize import VcfFilterConfig, normalize_vcf
from just_prs.vcf import detect_genome_build

from prs_ui.mixin import (
    PRSComputeStateMixin,
    SHEET_LABELS,
    SHEET_NAMES,
    SUPERPOPULATION_LABELS,
    _catalog,
    _compute_score_column_overrides,
    _enrich_scores_for_grid,
    _resolve_cache_dir,
    _resolve_preloaded_vcf_path,
    _resolve_preselect_query,
)


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
        cache_path = Path(self.cache_dir) / "metadata" / "raw" / f"{sheet}.parquet"
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

    def handle_lf_grid_row_selection(self, model: dict) -> None:
        """Track selected PGS IDs from metadata grid checkbox selection.

        Overrides ``LazyFrameGridMixin.handle_lf_grid_row_selection`` so that
        ``lazyframe_grid()`` automatically calls this without needing an
        explicit ``on_row_selection_model_change`` kwarg.
        """
        self.lf_grid_row_selection_model = model  # type: ignore[assignment]
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
        yield from self.set_lazyframe(lf, chunk_size=500, column_overrides={
            "rsid": {
                "width": 140,
                "cellRendererType": "url",
                "cellRendererConfig": {
                    "baseUrl": "https://www.ncbi.nlm.nih.gov/snp/",
                    "color": "#1565c0",
                },
            },
        })
        row_count = lf.select(pl.len()).collect().item()
        self.genomic_row_count = row_count
        self.genomic_loaded = True
        self._normalizing = False
        self.normalize_status = f"Normalized: {row_count:,} variants"
        self.status_message = f"VCF normalized: {row_count:,} variants"


class ComputeGridState(PRSComputeStateMixin, LazyFrameGridMixin, AppState):
    """Concrete state for the standalone Compute PRS page.

    Adds VCF-upload-specific behavior on top of PRSComputeStateMixin.
    Inherits shared vars from AppState and grid vars from LazyFrameGridMixin.
    """

    vcf_filename: str = ""
    detected_build: str = ""
    build_detection_message: str = ""

    _vcf_path: str = ""
    _preloaded_vcf_initialized: bool = False

    async def initialize(self) -> Any:
        """Auto-load cleaned scores and optional preloaded VCF on first page visit."""
        for event in self.initialize_prs():
            yield event

        if self._preloaded_vcf_initialized or self._vcf_path:
            return
        self._preloaded_vcf_initialized = True

        preloaded_vcf = _resolve_preloaded_vcf_path()
        if preloaded_vcf is None:
            for event in self._preselect_scores_from_env():
                yield event
            return
        if not preloaded_vcf.exists():
            self.status_message = f"Configured preloaded VCF does not exist: {preloaded_vcf}"
            self.build_detection_message = "Configured preloaded VCF was not found."
            for event in self._preselect_scores_from_env():
                yield event
            return

        needs_score_reload = self._set_vcf_source(preloaded_vcf, label_prefix="Preloaded")
        genomic_state = await self.get_state(GenomicGridState)
        for event in genomic_state.normalize_uploaded_vcf(str(preloaded_vcf)):
            yield event
        if needs_score_reload:
            for event in self.load_compute_scores():
                yield event
        for event in self._preselect_scores_from_env():
            yield event

    def set_genome_build(self, value: str) -> Any:
        """Set genome build and reload compute scores if already loaded."""
        yield from self.set_prs_genome_build(value)

    def _set_vcf_source(self, path: Path, label_prefix: str) -> bool:
        """Record a VCF path, detect genome build, and reset dependent UI state."""
        self._vcf_path = str(path)
        self.vcf_filename = path.name
        self.prs_genotypes_path = str(path)

        detected = detect_genome_build(path)
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
        self.trait_summary_rows = []
        self.trait_summary_visible = False
        self.low_match_warning = False
        self.status_message = f"{label_prefix} {path.name}"
        return needs_score_reload

    def _preselect_scores_from_env(self) -> Any:
        """Apply optional startup score selection from ``PRS_UI_PRESELECT_QUERY``."""
        query = _resolve_preselect_query()
        if query:
            yield from self.filter_and_select_scores_by_query(query)

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

        needs_score_reload = self._set_vcf_source(dest, label_prefix="Uploaded")

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


def _build_trait_column_overrides() -> dict[str, dict[str, Any]]:
    """Column overrides for the trait browser grid."""
    _grade_col = {"width": 75, "cellRendererType": "badge"}
    return {
        "trait": {"minWidth": 200, "flex": 2},
        "n_models": {"width": 90, "headerName": "Models"},
        "n_high": {
            **_grade_col,
            "headerName": "High",
            "cellRendererConfig": {"color": "#2e7d32", "bgColor": "#e8f5e9"},
        },
        "n_normal": {
            **_grade_col,
            "headerName": "Normal",
            "cellRendererConfig": {"color": "#1565c0", "bgColor": "#e3f2fd"},
        },
        "n_moderate": {
            **_grade_col,
            "headerName": "Moderate",
            "cellRendererConfig": {"color": "#f57f17", "bgColor": "#fff3e0"},
        },
        "n_low": {
            **_grade_col,
            "headerName": "Low",
            "cellRendererConfig": {"color": "#c62828", "bgColor": "#ffebee"},
        },
        "avg_variants": {"width": 120},
        "min_variants": {"width": 110},
        "max_variants": {"width": 110},
        "pgs_ids": {"minWidth": 200, "flex": 1, "headerName": "PGS IDs"},
        "trait_efo_id": {
            "width": 150,
            "cellRendererType": "url",
            "cellRendererConfig": {
                "baseUrl": "http://www.ebi.ac.uk/efo/",
                "color": "#1565c0",
            },
        },
    }


class TraitBrowserState(PRSComputeStateMixin, LazyFrameGridMixin, AppState):
    """State for the trait-based PRS browser tab.

    Groups PGS Catalog scores by EFO trait and lets the user select
    traits instead of individual PGS IDs.  Selected traits are resolved
    to their constituent PGS IDs, and computation proceeds via the
    inherited ``PRSComputeStateMixin``.
    """

    vcf_filename: str = ""
    detected_build: str = ""
    build_detection_message: str = ""
    selected_traits: list[str] = []
    traits_loaded: bool = False

    _vcf_path: str = ""
    _trait_to_pgs: dict[str, list[str]] = {}
    _trait_scores_lf: pl.LazyFrame | None = None
    _traits_initialized: bool = False

    def _build_trait_df(self) -> pl.DataFrame:
        """Group scores by trait and return a flat summary DataFrame."""
        lf = _catalog.scores(genome_build=self.genome_build, include_harmonized=self.include_harmonized)  # type: ignore[attr-defined]
        lf = _enrich_scores_for_grid(lf, _catalog)
        self._trait_scores_lf = lf
        df = lf.select(
            "pgs_id", "trait_reported", "trait_efo", "trait_efo_id",
            "n_variants", "quality_label",
        ).collect()

        df = df.with_columns(
            pl.when(pl.col("trait_efo").is_not_null() & (pl.col("trait_efo") != ""))
            .then(pl.col("trait_efo"))
            .otherwise(pl.col("trait_reported"))
            .alias("trait"),
        )

        grouped = df.group_by("trait").agg(
            pl.col("pgs_id").count().alias("n_models"),
            pl.col("pgs_id").alias("_pgs_list"),
            pl.col("trait_efo_id").first().alias("trait_efo_id"),
            pl.col("n_variants").mean().cast(pl.Int64).alias("avg_variants"),
            pl.col("n_variants").min().alias("min_variants"),
            pl.col("n_variants").max().alias("max_variants"),
            (pl.col("quality_label") == "High").sum().cast(pl.Int64).alias("n_high"),
            (pl.col("quality_label") == "Normal").sum().cast(pl.Int64).alias("n_normal"),
            (pl.col("quality_label") == "Moderate").sum().cast(pl.Int64).alias("n_moderate"),
            (pl.col("quality_label") == "Low").sum().cast(pl.Int64).alias("n_low"),
        ).sort("n_models", descending=True)

        mapping: dict[str, list[str]] = {}
        for row in grouped.iter_rows(named=True):
            mapping[row["trait"]] = row["_pgs_list"]

        self._trait_to_pgs = mapping

        result = grouped.with_columns(
            pl.col("_pgs_list").list.join(", ").alias("pgs_ids"),
        ).drop("_pgs_list")

        for col in ("n_high", "n_normal", "n_moderate", "n_low"):
            result = result.with_columns(
                pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col),
            )

        result = result.select(
            "trait", "n_models", "n_high", "n_normal", "n_moderate", "n_low",
            "avg_variants", "min_variants", "max_variants",
            "pgs_ids", "trait_efo_id",
        )

        return result

    def load_traits(self) -> Any:
        """Load trait-grouped data into the grid."""
        self.status_message = "Loading traits..."  # type: ignore[attr-defined]
        yield
        trait_df = self._build_trait_df()
        self.traits_loaded = True
        self.selected_traits = []
        self.selected_pgs_ids = []
        yield from self.set_lazyframe(  # type: ignore[attr-defined]
            trait_df.lazy(),
            chunk_size=500,
            column_overrides=_build_trait_column_overrides(),
        )
        self.status_message = f"Loaded {trait_df.height} traits for {self.genome_build}"  # type: ignore[attr-defined]

    def initialize_traits(self) -> Any:
        """Auto-load traits on first access."""
        if self._traits_initialized:
            return
        self._traits_initialized = True
        yield from self.load_traits()

    def set_genome_build(self, value: str) -> Any:
        """Set genome build and reload traits."""
        self.genome_build = value  # type: ignore[attr-defined]
        if self.traits_loaded:
            yield from self.load_traits()

    def handle_lf_grid_row_selection(self, model: dict) -> None:
        """Track selected traits and resolve to PGS IDs."""
        self.lf_grid_row_selection_model = model  # type: ignore[assignment]

        selection_type: str = model.get("type", "include")
        raw_ids: list = model.get("ids", [])
        selected_row_ids: set[int] = {int(i) for i in raw_ids}

        if selection_type == "exclude" and not selected_row_ids:
            self._select_all_traits()
            return
        if selection_type == "include" and not selected_row_ids:
            self.selected_traits = []
            self.selected_pgs_ids = []
            return

        traits: list[str] = []
        for row in self.lf_grid_rows:  # type: ignore[attr-defined]
            row_id = row.get("__row_id__")
            in_set = (int(row_id) in selected_row_ids) if row_id is not None else False
            if (selection_type == "include" and in_set) or (
                selection_type == "exclude" and not in_set
            ):
                trait = row.get("trait")
                if trait:
                    traits.append(str(trait))

        self.selected_traits = traits
        self._resolve_pgs_ids_from_traits()

    def _select_all_traits(self) -> None:
        """Select all traits (and their PGS IDs)."""
        self.selected_traits = list(self._trait_to_pgs.keys())
        pgs_ids: list[str] = []
        for ids in self._trait_to_pgs.values():
            pgs_ids.extend(ids)
        self.selected_pgs_ids = pgs_ids
        self.status_message = (  # type: ignore[attr-defined]
            f"Selected all {len(self.selected_traits)} traits "
            f"({len(self.selected_pgs_ids)} PGS IDs)"
        )

    def select_filtered_traits(self) -> None:
        """Select traits matching the current grid filter."""
        if self._trait_scores_lf is None:
            return
        traits: list[str] = []
        for row in self.lf_grid_rows:  # type: ignore[attr-defined]
            trait = row.get("trait")
            if trait:
                traits.append(str(trait))
        self.selected_traits = traits
        self._resolve_pgs_ids_from_traits()

    def deselect_all_traits(self) -> None:
        """Clear all selected traits."""
        self.selected_traits = []
        self.selected_pgs_ids = []
        self.lf_grid_row_selection_model = {"type": "include", "ids": []}  # type: ignore[assignment]
        self.status_message = ""  # type: ignore[attr-defined]

    def _resolve_pgs_ids_from_traits(self) -> None:
        """Resolve selected traits to PGS IDs."""
        pgs_ids: list[str] = []
        for trait in self.selected_traits:
            pgs_ids.extend(self._trait_to_pgs.get(trait, []))
        self.selected_pgs_ids = pgs_ids
        self.status_message = (  # type: ignore[attr-defined]
            f"Selected {len(self.selected_traits)} trait(s) "
            f"({len(self.selected_pgs_ids)} PGS IDs)"
        )

    def _set_vcf_source(self, path: Path, label_prefix: str) -> bool:
        """Record a VCF path, detect genome build, and reset dependent state."""
        self._vcf_path = str(path)
        self.vcf_filename = path.name
        self.prs_genotypes_path = str(path)

        detected = detect_genome_build(path)
        needs_reload = False
        if detected is not None:
            self.detected_build = detected
            if detected != self.genome_build:  # type: ignore[attr-defined]
                needs_reload = self.traits_loaded
            self.genome_build = detected  # type: ignore[attr-defined]
            self.build_detection_message = f"Detected genome build: {detected}"
        else:
            self.detected_build = ""
            self.build_detection_message = (
                "Could not detect genome build from VCF header. "
                "Please select it manually."
            )
        self.prs_results = []
        self.trait_summary_rows = []
        self.trait_summary_visible = False
        self.low_match_warning = False
        self.status_message = f"{label_prefix} {path.name}"  # type: ignore[attr-defined]
        return needs_reload

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

        needs_reload = self._set_vcf_source(dest, label_prefix="Uploaded")

        events: list = [GenomicGridState.normalize_uploaded_vcf(str(dest))]
        if needs_reload:
            events.append(TraitBrowserState.load_traits)
        return events

    async def compute_selected_prs(self) -> Any:
        """Resolve genotypes from GenomicGridState, then compute and auto-group by trait."""
        if not self._vcf_path:
            self.status_message = "Please upload a VCF file first."  # type: ignore[attr-defined]
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

        if self.prs_results:
            self.build_trait_summary()
