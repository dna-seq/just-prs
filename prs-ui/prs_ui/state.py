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
from typing import Any, ClassVar

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
    compute_mode: str = "prs"

    def set_pgs_id(self, value: str) -> None:
        self.pgs_id_input = value

    def set_genome_build(self, value: str) -> None:
        self.genome_build = value

    def set_active_tab(self, value: str) -> None:
        self.active_tab = value

    def set_compute_mode(self, value: str | list[str]) -> None:
        """Switch the Compute PRS workbench between 'prs' and 'trait' selection."""
        self.compute_mode = value if isinstance(value, str) else (value[0] if value else "prs")


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
    """Detachable genotype source: VCF upload + normalization + preview grid.

    This is the *reference* genotype source for the prs-ui demo app.  It owns
    all VCF UI state (filename, detected build, normalized parquet) and feeds
    the normalized genotypes into any registered **consumer** state (a state
    that mixes in ``PRSComputeStateMixin``) via the loose-coupling hooks
    ``consumer.load_genotypes(path)`` and ``consumer.set_genome_build(build)``.

    Coupling is wiring-time, not hardcoded: register consumers by assigning
    ``GenomicGridState._consumer_states = [...]`` (done at the bottom of this
    module).  A host app such as just-dna-lite can ignore this source entirely
    and drive the same consumer hooks from a different source (e.g. a public
    genome selector) without touching the mixin.
    """

    #: Consumer state classes fed by this source.  Set at app-wiring time.
    _consumer_states: ClassVar[list[type]] = []

    normalized_parquet_path: str = ""
    normalize_status: str = ""
    vcf_normalizing: bool = False
    genomic_loaded: bool = False
    genomic_row_count: int = 0

    vcf_filename: str = ""
    detected_build: str = ""
    build_detection_message: str = ""

    _vcf_path: str = ""
    _preloaded_vcf_initialized: bool = False

    def _normalized_parquet_path(self, src: Path) -> Path:
        """Deterministic output path for a normalized VCF parquet."""
        out_dir = Path(self.cache_dir) / "normalized"
        return out_dir / (src.stem.removesuffix(".vcf") + ".parquet")

    def _set_vcf_source(self, path: Path, label_prefix: str) -> str:
        """Record a VCF path and detect its genome build.

        Returns the detected build (``""`` if undetected) so callers can fan
        the build out to consumer states.
        """
        self._vcf_path = str(path)
        self.vcf_filename = path.name

        detected = detect_genome_build(path)
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
        self.status_message = f"{label_prefix} {path.name}"
        return self.detected_build

    async def _push_to_consumers(self) -> Any:
        """Push normalized genotypes (and changed build) into every consumer.

        Consumers are mutated directly via ``get_state`` rather than by yielding
        cross-state ``EventSpec``s.  Yielding chained events *after* the long,
        blocking ``normalize_vcf()`` call triggers Reflex's "Cannot add a child
        to an EventFuture that is already done" error and stalls the event
        queue (which also makes grid checkbox selection sluggish/unresponsive).
        Direct mutation enqueues no child events and is reliable and ordered.
        """
        for consumer_cls in self._consumer_states:
            consumer = await self.get_state(consumer_cls)
            consumer.load_genotypes(self.normalized_parquet_path)
            if self.detected_build and self.detected_build != consumer.genome_build:
                for event in consumer.set_genome_build(self.detected_build):
                    yield event

    async def set_shared_genome_build(self, value: str) -> Any:
        """Manual genome-build override shared across all consumer states."""
        self.genome_build = value
        self.detected_build = ""
        for consumer_cls in self._consumer_states:
            consumer = await self.get_state(consumer_cls)
            for event in consumer.set_genome_build(value):
                yield event

    async def handle_vcf_upload(self, files: list[rx.UploadFile]) -> Any:
        """Save an uploaded VCF, normalize it, and feed all consumer states."""
        if not files:
            return
        upload_file = files[0]
        filename = upload_file.filename or "uploaded.vcf"
        upload_dir = rx.get_upload_dir()
        dest = upload_dir / filename

        self.vcf_filename = filename
        self.vcf_normalizing = True
        self.genomic_loaded = False
        self.normalize_status = f"Saving {filename}..."
        self.status_message = self.normalize_status
        yield

        contents = await upload_file.read()
        dest.write_bytes(contents)

        self._set_vcf_source(dest, label_prefix="Uploaded")
        for event in self.normalize_uploaded_vcf(str(dest)):
            yield event
        async for event in self._push_to_consumers():
            yield event

    async def initialize_source(self) -> Any:
        """Normalize an optional preloaded VCF and feed consumers on startup."""
        if self._preloaded_vcf_initialized or self._vcf_path:
            return
        self._preloaded_vcf_initialized = True

        preloaded_vcf = _resolve_preloaded_vcf_path()
        if preloaded_vcf is None:
            return
        if not preloaded_vcf.exists():
            self.status_message = f"Configured preloaded VCF does not exist: {preloaded_vcf}"
            self.build_detection_message = "Configured preloaded VCF was not found."
            return

        self._set_vcf_source(preloaded_vcf, label_prefix="Preloaded")
        for event in self.normalize_uploaded_vcf(str(preloaded_vcf)):
            yield event
        async for event in self._push_to_consumers():
            yield event

    def normalize_uploaded_vcf(self, vcf_path: str, sex: str = "") -> Any:
        """Normalize a VCF and load the preview grid (no consumer fan-out)."""
        if not vcf_path:
            self.normalize_status = "No VCF path provided."
            return
        self.vcf_normalizing = True
        self.genomic_loaded = False
        self.normalize_status = (
            "Normalizing VCF — this is the slow step; large files can take up to a "
            "minute or two. Please wait."
        )
        yield

        src = Path(vcf_path)
        output_path = self._normalized_parquet_path(src)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Content-aware cache: reuse an existing normalized parquet when it is at
        # least as recent as the source VCF.  Re-normalizing a large VCF on every
        # upload is the main cause of the "normalization never finishes" regression.
        cache_fresh = (
            output_path.exists()
            and output_path.stat().st_mtime >= src.stat().st_mtime
        )
        if not cache_fresh:
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
        self.vcf_normalizing = False
        self.normalize_status = f"Normalized: {row_count:,} variants"
        self.status_message = f"VCF normalized: {row_count:,} variants"


class ComputeGridState(PRSComputeStateMixin, LazyFrameGridMixin, AppState):
    """Concrete "By PRS" consumer state for the Compute PRS workbench.

    Genotypes are pushed in by a genotype source (the VCF upload via
    ``GenomicGridState``, or any other source in a host app) through the
    inherited ``load_genotypes(path)`` hook, so this state owns no VCF or
    upload logic and the base ``PRSComputeStateMixin.compute_selected_prs``
    is used unchanged.
    """

    prs_view_mode: str = "individual"

    def initialize(self) -> Any:
        """Auto-load cleaned scores and apply optional startup preselection."""
        self.prs_view_mode = "individual"
        for event in self.initialize_prs():
            yield event
        for event in self._preselect_scores_from_env():
            yield event

    def set_genome_build(self, value: str) -> Any:
        """Set genome build and reload compute scores if already loaded."""
        yield from self.set_prs_genome_build(value)

    def _preselect_scores_from_env(self) -> Any:
        """Apply optional startup score selection from ``PRS_UI_PRESELECT_QUERY``."""
        query = _resolve_preselect_query()
        if query:
            yield from self.filter_and_select_scores_by_query(query)


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
    """Concrete "By Trait" consumer state for the Compute PRS workbench.

    Groups PGS Catalog scores by EFO trait and lets the user select
    traits instead of individual PGS IDs.  Selected traits are resolved
    to their constituent PGS IDs, and computation proceeds via the
    inherited ``PRSComputeStateMixin``.  Genotypes are pushed in by a
    genotype source through the inherited ``load_genotypes(path)`` hook;
    this state owns no VCF or upload logic.
    """

    prs_view_mode: str = "grouped"

    selected_traits: list[str] = []
    traits_loaded: bool = False

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
            eager_value_options_row_limit=0,
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

    def compute_selected_prs(self) -> Any:
        """Compute PRS for the selected traits, then auto-group by trait.

        Genotypes are supplied beforehand via the inherited ``load_genotypes``
        hook, so this override only adds trait-summary auto-building on top of
        the base mixin computation.
        """
        gen = PRSComputeStateMixin.compute_selected_prs(self)
        if gen is not None:
            for event in gen:
                yield event

        if self.prs_results:
            self.build_trait_summary()


# App wiring: register the consumer states fed by the VCF genotype source.
# This is the only coupling point between the source and the consumers; a host
# app can register a different set (or drive the consumer hooks from its own
# source) without changing GenomicGridState or PRSComputeStateMixin.
GenomicGridState._consumer_states = [ComputeGridState, TraitBrowserState]
