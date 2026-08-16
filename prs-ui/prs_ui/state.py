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

import hashlib
import os
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import reflex as rx
from reflex_mui_datagrid import LazyFrameGridMixin
from reflex_mui_datagrid.lazyframe_grid import _get_cache, apply_filter_model

from just_prs.ftp import (
    download_metadata_sheet,
    stream_scoring_file,
)
from just_prs.normalize import VcfFilterConfig, normalize_vcf
from just_prs.scoring import ensure_scoring_file
from just_prs.vcf import detect_genome_build
from just_prs.viz import (
    FINE_POPULATION_LABELS,
    IGSR_POPULATION_URL,
    SUPERPOP_LABELS,
)

from prs_ui.mixin import (
    PRSComputeStateMixin,
    SHEET_LABELS,
    SHEET_NAMES,
    SUPERPOPULATION_LABELS,
    _catalog,
    loaded_grid_selection_model,
    merge_loaded_grid_selection,
    sample_color,
    sample_label_from_path,
    _ancestry_chip_text,
    _compute_score_column_overrides,
    _enrich_scores_for_grid,
    _resolve_cache_dir,
    _resolve_preloaded_vcf_path,
    _resolve_preselect_query,
)


def _workspace_root() -> Path:
    """Resolve the source workspace root for direct Reflex development runs."""
    candidates = [Path.cwd(), Path(__file__).resolve()]
    for start in candidates:
        for parent in (start, *start.parents):
            if (parent / "uv.lock").exists() and (parent / "pyproject.toml").exists():
                return parent
    return Path.cwd()


def _input_vcf_dir() -> Path:
    """Directory for user-provided VCF inputs, kept outside source packages."""
    data_root = os.environ.get("PRS_UI_DATA_DIR", "").strip()
    base = Path(data_root).expanduser() if data_root else _workspace_root() / "data"
    return base / "input" / "vcf"


def _has_gzip_magic(contents: bytes) -> bool:
    """Return True when uploaded bytes are gzip-compressed."""
    return contents.startswith(b"\x1f\x8b")


class AppState(rx.State):
    """Shared app state: tab selection, genome build, cache dir."""

    selected_sheet: str = "scores"
    cache_dir: str = str(_resolve_cache_dir())
    status_message: str = ""
    pgs_id_input: str = "PGS000001"
    genome_build: str = "GRCh38"
    active_tab: str = "trait"
    compute_mode: str = "trait"

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

    # Raw PGS Catalog sheets key the ID by the verbose column name.
    _METADATA_ID_FIELD: ClassVar[str] = "Polygenic Score (PGS) ID"

    def _sync_loaded_metadata_selection(self) -> None:
        """Re-project ``metadata_selected_ids`` onto the currently-loaded rows.

        The MUI selection model is keyed by the positional ``__row_id__`` the
        grid renumbers on every sort/filter; without re-syncing, the checkboxes
        clear on sort and the next click collapses the selection.
        """
        self.lf_grid_row_selection_model = loaded_grid_selection_model(  # type: ignore[assignment]
            self.lf_grid_rows, self.metadata_selected_ids, self._METADATA_ID_FIELD
        )

    def handle_lf_grid_row_selection(self, model: dict) -> None:
        """Track selected PGS IDs from metadata grid checkbox selection.

        Overrides ``LazyFrameGridMixin.handle_lf_grid_row_selection`` so that
        ``lazyframe_grid()`` automatically calls this.  Out-of-scope (off-page /
        filtered-out) selections are preserved so sort/filter never drops them.
        """
        if model.get("type", "include") == "exclude" and not model.get("ids", []):
            # "Select all" header checkbox: select every loaded row's ID.
            loaded = [
                str(row.get(self._METADATA_ID_FIELD))
                for row in self.lf_grid_rows
                if row.get(self._METADATA_ID_FIELD)
            ]
            self.metadata_selected_ids = list(
                dict.fromkeys([*self.metadata_selected_ids, *loaded])
            )
        else:
            self.metadata_selected_ids = merge_loaded_grid_selection(
                self.lf_grid_rows, self.metadata_selected_ids, model, self._METADATA_ID_FIELD
            )
        self._sync_loaded_metadata_selection()

    def handle_lf_grid_sort(self, sort_model: list) -> Any:
        """Apply a server-side sort, then restore the selection."""
        yield from LazyFrameGridMixin.handle_lf_grid_sort(self, sort_model)
        self._sync_loaded_metadata_selection()

    def handle_lf_grid_filter(self, filter_model: dict) -> Any:
        """Apply a server-side filter, then restore the selection."""
        yield from LazyFrameGridMixin.handle_lf_grid_filter(self, filter_model)
        self._sync_loaded_metadata_selection()

    def clear_lf_grid_filters(self) -> Any:
        """Clear all grid filters, then restore the selection."""
        yield from LazyFrameGridMixin.clear_lf_grid_filters(self)
        self._sync_loaded_metadata_selection()

    def handle_lf_grid_scroll_end(self, params: dict) -> Any:
        """Load the next scroll chunk, then mark selected rows checked."""
        yield from LazyFrameGridMixin.handle_lf_grid_scroll_end(self, params)
        self._sync_loaded_metadata_selection()

    def download_selected_scoring_files(self) -> Any:
        """Pre-warm the compute cache with scoring files for the selected PGS IDs.

        Writes to the **canonical** managed cache (``<cache>/scores``) via
        ``ensure_scoring_file``, so a later PRS computation reuses these files.
        This previously wrote ``<cache>/scoring/{build}/{pgs_id}.parquet`` — a
        layout with a different filename *and* a different schema (it injected a
        ``pgs_id`` column) that no reader ever consulted, so every download was
        stored twice and the compute path re-fetched anyway.
        """
        if not self.metadata_selected_ids:
            self.status_message = "No scores selected."
            return
        total = len(self.metadata_selected_ids)
        output_dir = Path(self.cache_dir) / "scores"
        self.status_message = f"Saving {total} scoring file(s) to cache..."
        yield
        for i, pgs_id in enumerate(self.metadata_selected_ids, start=1):
            self.status_message = f"Saving {i}/{total}: {pgs_id}..."
            yield
            ensure_scoring_file(pgs_id, output_dir, genome_build=self.genome_build)
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

    #: Multi-sample registry (CLI ``--vcf Label=path`` equivalent):
    #: [{"label", "filename", "vcf_path", "parquet_path", "build", "color",
    #:   "ancestry", "ancestry_confidence", "fine_population", "fine_confidence"}].
    vcf_samples: list[dict] = []

    ancestry_inferring: bool = False

    _vcf_path: str = ""
    _preloaded_vcf_initialized: bool = False

    @rx.var
    def vcf_sample_chips(self) -> list[dict]:
        """One display row per sample (label, color, build, ancestry, variants).

        Ancestry mirrors the CLI report's sample table: the super-population
        with its confidence, plus the closest 1000G cohort (a clickable IGSR
        population-page link with its own posterior when the code is a known
        1000G cohort).
        """
        rows: list[dict] = []
        for s in self.vcf_samples:
            superpop = str(s.get("ancestry") or "")
            confidence = float(s.get("ancestry_confidence") or 0.0)
            fine = str(s.get("fine_population") or "")
            fine_confidence = float(s.get("fine_confidence") or 0.0)
            fine_entry = FINE_POPULATION_LABELS.get(fine)
            rows.append({
                "label": str(s.get("label") or ""),
                "color": str(s.get("color") or ""),
                "filename": str(s.get("filename") or ""),
                "build": str(s.get("build") or ""),
                "ancestry": _ancestry_chip_text(s),
                "ancestry_label": (
                    f"{SUPERPOP_LABELS.get(superpop, superpop)} ({superpop})"
                    if superpop and superpop != "UNKNOWN"
                    else ""
                ),
                "ancestry_conf": (
                    f"{confidence:.0%}" if superpop and confidence > 0 else ""
                ),
                "fine_label": (
                    f"{fine_entry[0]} ({fine})" if fine_entry else fine
                ),
                "fine_conf": (
                    f"{fine_confidence:.0%}" if fine and fine_confidence > 0 else ""
                ),
                "fine_url": (
                    IGSR_POPULATION_URL.format(code=fine) if fine_entry else ""
                ),
                "fine_title": fine_entry[1] if fine_entry else "",
                "variants": (
                    f"{int(s.get('variants') or 0):,} variants"
                    if int(s.get("variants") or 0) > 0
                    else ""
                ),
            })
        return rows

    @rx.var
    def vcf_has_fine_population(self) -> bool:
        """True when any sample carries a closest-cohort call (footnote gate)."""
        return any(str(s.get("fine_population") or "") for s in self.vcf_samples)

    @rx.var
    def vcf_sample_count(self) -> int:
        """Number of loaded genome samples."""
        return len(self.vcf_samples)

    @rx.var
    def vcf_multi_sample(self) -> bool:
        """True when more than one genome sample is loaded (comparison mode)."""
        return len(self.vcf_samples) > 1

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
        """Push normalized genotype samples (and changed build) into every consumer.

        Consumers are mutated directly via ``get_state`` rather than by yielding
        cross-state ``EventSpec``s.  Yielding chained events *after* the long,
        blocking ``normalize_vcf()`` call triggers Reflex's "Cannot add a child
        to an EventFuture that is already done" error and stalls the event
        queue (which also makes grid checkbox selection sluggish/unresponsive).
        Direct mutation enqueues no child events and is reliable and ordered.

        All loaded samples are fanned out via ``load_samples`` — a single-VCF
        upload is simply a registry of one, so consumers behave exactly as the
        old single-sample ``load_genotypes`` path.
        """
        payload = [
            {
                "label": str(s.get("label") or ""),
                "path": str(s.get("parquet_path") or ""),
                "ancestry": str(s.get("ancestry") or ""),
                "ancestry_confidence": float(s.get("ancestry_confidence") or 0.0),
                "fine_population": str(s.get("fine_population") or ""),
                "fine_confidence": float(s.get("fine_confidence") or 0.0),
            }
            for s in self.vcf_samples
            if s.get("parquet_path")
        ]
        # ``load_samples`` sets ``selected_ancestry`` to the majority detected
        # superpopulation; sample rows already show each call, and the trait
        # dashboard Population dropdown is the override for card numbers.
        for consumer_cls in self._consumer_states:
            consumer = await self.get_state(consumer_cls)
            consumer.load_samples(payload)
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

    def _infer_sample_ancestry(self, parquet_path: Path, build: str) -> dict:
        """Autodetect population + subpopulation for one normalized sample.

        Mirrors the CLI's ``_infer_vcf_ancestry``: the 1000G model is projected
        twice on the same normalized frame — once at super-population resolution
        (the label + confidence that drive percentiles) and once at fine
        population resolution (the closest 1000G cohort, e.g. CEU/GBR —
        informational, with its own posterior).  Degrades to an empty dict when
        the ancestry model cannot be pulled (offline) or the call is UNKNOWN
        (coverage below the floor) so the upload flow never breaks on it.
        """
        try:
            genotypes_lf = pl.scan_parquet(parquet_path)
            inference = _catalog.infer_ancestry(
                genotypes_lf=genotypes_lf,
                sample_build=build or "GRCh38",
            )
            fine = _catalog.infer_ancestry(
                genotypes_lf=genotypes_lf,
                sample_build=build or "GRCh38",
                resolution="population",
            )
        except Exception as exc:  # HF model pull / liftover are external boundaries
            self.status_message = f"Ancestry autodetection unavailable: {exc}"
            return {}
        if inference.superpopulation == "UNKNOWN":
            return {}
        return {
            "ancestry": inference.superpopulation,
            "ancestry_confidence": float(inference.confidence),
            "fine_population": str(fine.fine_population or ""),
            "fine_confidence": (
                float(fine.confidence) if fine.fine_population else 0.0
            ),
        }

    def _register_sample(
        self,
        vcf_path: Path,
        parquet_path: Path,
        build: str,
        ancestry: dict | None = None,
    ) -> None:
        """Add (or refresh) a sample registry entry, re-coloring by position."""
        label = sample_label_from_path(str(vcf_path))
        registry = [
            s for s in self.vcf_samples
            if str(s.get("parquet_path") or "") != str(parquet_path)
            and str(s.get("label") or "") != label
        ]
        registry.append({
            "label": label,
            "filename": vcf_path.name,
            "vcf_path": str(vcf_path),
            "parquet_path": str(parquet_path),
            "build": build,
            # Set by normalize_uploaded_vcf just before registration.
            "variants": int(self.genomic_row_count or 0),
            **(ancestry or {}),
        })
        self.vcf_samples = [
            {**s, "color": sample_color(i)} for i, s in enumerate(registry)
        ]

    async def remove_sample(self, label: str) -> Any:
        """Remove one sample from the comparison and re-feed all consumers."""
        self.vcf_samples = [
            {**s, "color": sample_color(i)}
            for i, s in enumerate(
                s for s in self.vcf_samples if str(s.get("label") or "") != label
            )
        ]
        first = self.vcf_samples[0] if self.vcf_samples else None
        self.normalized_parquet_path = str(first.get("parquet_path") or "") if first else ""
        self.vcf_filename = str(first.get("filename") or "") if first else ""
        self._vcf_path = str(first.get("vcf_path") or "") if first else ""
        if not self.vcf_samples:
            self.genomic_loaded = False
            self.normalize_status = ""
            self.status_message = "All samples removed."
        else:
            self.status_message = (
                f"Removed {label}. {len(self.vcf_samples)} sample(s) loaded."
            )
        async for event in self._push_to_consumers():
            yield event

    async def clear_samples(self) -> Any:
        """Remove every loaded sample and reset the source."""
        self.vcf_samples = []
        self.normalized_parquet_path = ""
        self.vcf_filename = ""
        self._vcf_path = ""
        self.genomic_loaded = False
        self.normalize_status = ""
        self.detected_build = ""
        self.build_detection_message = ""
        self.status_message = "All samples removed."
        async for event in self._push_to_consumers():
            yield event

    async def handle_vcf_upload(self, files: list[rx.UploadFile]) -> Any:
        """Save uploaded VCF(s), normalize each, and feed all consumer states.

        Multiple files can be dropped at once (or added across several uploads)
        — each becomes a named, colored sample in the comparison, mirroring the
        CLI's repeated ``--vcf Label=path`` flags.
        """
        if not files:
            return
        upload_dir = _input_vcf_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)

        for upload_file in files:
            filename = Path(upload_file.filename or "uploaded.vcf").name

            self.vcf_filename = filename
            self.vcf_normalizing = True
            self.genomic_loaded = False
            self.normalize_status = f"Saving {filename}..."
            self.status_message = self.normalize_status
            yield

            contents = await upload_file.read()
            if _has_gzip_magic(contents) and not filename.casefold().endswith((".gz", ".bgz")):
                filename = f"{filename}.gz"
                self.vcf_filename = filename
            dest = upload_dir / filename

            # Skip writing identical content so the VCF mtime stays unchanged and
            # the downstream mtime-based normalization cache remains valid.
            new_hash = hashlib.md5(contents).digest()
            if dest.exists() and hashlib.md5(dest.read_bytes()).digest() == new_hash:
                pass  # identical file already on disk — keep its mtime
            else:
                dest.write_bytes(contents)

            build = self._set_vcf_source(dest, label_prefix="Uploaded")
            for event in self.normalize_uploaded_vcf(str(dest)):
                yield event

            self.ancestry_inferring = True
            self.normalize_status = f"Autodetecting ancestry for {filename}..."
            self.status_message = self.normalize_status
            yield
            ancestry = self._infer_sample_ancestry(
                Path(self.normalized_parquet_path), build
            )
            self.ancestry_inferring = False
            self._register_sample(
                dest, Path(self.normalized_parquet_path), build, ancestry
            )
            detected = _ancestry_chip_text(self.vcf_samples[-1])
            self.normalize_status = (
                f"Normalized {filename} — ancestry {detected}"
                if detected
                else f"Normalized {filename} — ancestry not detected"
            )
            self.status_message = self.normalize_status

        if self.vcf_multi_sample:
            labels = ", ".join(str(s.get("label") or "") for s in self.vcf_samples)
            self.status_message = f"Comparing {len(self.vcf_samples)} samples: {labels}"
        builds = {str(s.get("build") or "") for s in self.vcf_samples if s.get("build")}
        if len(builds) > 1:
            self.build_detection_message = (
                f"⚠ Mixed genome builds detected across samples ({', '.join(sorted(builds))}). "
                "All samples are scored against the selected build — percentiles for "
                "mismatched samples may be unreliable."
            )
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

        build = self._set_vcf_source(preloaded_vcf, label_prefix="Preloaded")
        for event in self.normalize_uploaded_vcf(str(preloaded_vcf)):
            yield event
        self.ancestry_inferring = True
        self.normalize_status = f"Autodetecting ancestry for {preloaded_vcf.name}..."
        yield
        ancestry = self._infer_sample_ancestry(
            Path(self.normalized_parquet_path), build
        )
        self.ancestry_inferring = False
        self._register_sample(
            preloaded_vcf, Path(self.normalized_parquet_path), build, ancestry
        )
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

    def _sync_loaded_trait_selection(self) -> None:
        """Re-project ``selected_traits`` onto the currently-loaded grid rows.

        The trait grid is keyed by the ``trait`` column, not ``pgs_id``, so the
        inherited pgs_id-based ``_sync_loaded_grid_selection`` matches no rows
        here and would silently clear the MUI selection model on every
        sort/filter (the durable ``selected_traits`` badge would stay correct
        while the checkboxes vanished, and the next click would collapse the
        selection).  This trait-keyed variant keeps the client checkboxes in
        sync with ``selected_traits`` after the loaded rows are reordered.
        """
        self.lf_grid_row_selection_model = loaded_grid_selection_model(  # type: ignore[assignment]
            self.lf_grid_rows, self.selected_traits, "trait"  # type: ignore[attr-defined]
        )

    def handle_lf_grid_row_selection(self, model: dict) -> None:
        """Merge grid checkbox changes into the durable ``selected_traits``.

        Mirrors ``PRSComputeStateMixin.handle_lf_grid_row_selection`` but keyed
        on the ``trait`` column: traits selected outside the currently-loaded
        scope (off-page / filtered out) are preserved, so changing the
        sort/filter never drops them.
        """
        if model.get("type", "include") == "exclude" and not model.get("ids", []):
            self._select_all_traits()
            self._sync_loaded_trait_selection()
            return

        self.selected_traits = merge_loaded_grid_selection(
            self.lf_grid_rows, self.selected_traits, model, "trait"  # type: ignore[attr-defined]
        )
        self._resolve_pgs_ids_from_traits()
        self._sync_loaded_trait_selection()

    def handle_lf_grid_sort(self, sort_model: list) -> Any:
        """Apply a server-side sort, then restore the trait selection."""
        yield from LazyFrameGridMixin.handle_lf_grid_sort(self, sort_model)
        self._sync_loaded_trait_selection()

    def handle_lf_grid_filter(self, filter_model: dict) -> Any:
        """Apply a server-side filter, then restore the trait selection."""
        yield from LazyFrameGridMixin.handle_lf_grid_filter(self, filter_model)
        self._sync_loaded_trait_selection()

    def clear_lf_grid_filters(self) -> Any:
        """Clear all grid filters, then restore the trait selection."""
        yield from LazyFrameGridMixin.clear_lf_grid_filters(self)
        self._sync_loaded_trait_selection()

    def handle_lf_grid_scroll_end(self, params: dict) -> Any:
        """Load the next scroll chunk, then mark selected trait rows checked."""
        yield from LazyFrameGridMixin.handle_lf_grid_scroll_end(self, params)
        self._sync_loaded_trait_selection()

    def _sync_loaded_trait_selection(self) -> None:
        """Project durable trait selection onto the currently loaded grid rows."""
        selected = set(self.selected_traits)
        ids: list[int] = []
        for row in self.lf_grid_rows:  # type: ignore[attr-defined]
            trait = str(row.get("trait") or "")
            row_id = row.get("__row_id__")
            if trait in selected and row_id is not None:
                ids.append(int(row_id))
        self.lf_grid_row_selection_model = {"type": "include", "ids": ids}  # type: ignore[assignment]

    def handle_lf_grid_filter(self, filter_model: dict) -> Any:
        """Apply trait-grid filters, then restore trait checkbox selection."""
        yield from LazyFrameGridMixin.handle_lf_grid_filter(self, filter_model)
        self._sync_loaded_trait_selection()

    def handle_lf_grid_sort(self, sort_model: list) -> Any:
        """Apply trait-grid sorting, then restore trait checkbox selection."""
        yield from LazyFrameGridMixin.handle_lf_grid_sort(self, sort_model)
        self._sync_loaded_trait_selection()

    def clear_lf_grid_filters(self) -> Any:
        """Clear trait-grid filters, then restore trait checkbox selection."""
        yield from LazyFrameGridMixin.clear_lf_grid_filters(self)
        self._sync_loaded_trait_selection()

    def handle_lf_grid_scroll_end(self, params: dict) -> Any:
        """Load another trait-grid chunk, then restore checkbox selection."""
        yield from LazyFrameGridMixin.handle_lf_grid_scroll_end(self, params)
        self._sync_loaded_trait_selection()

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
        """Select traits matching the current grid filter (additive)."""
        if self._trait_scores_lf is None:
            return
        loaded_traits = [
            str(row.get("trait"))
            for row in self.lf_grid_rows  # type: ignore[attr-defined]
            if row.get("trait")
        ]
        self.selected_traits = list(dict.fromkeys([*self.selected_traits, *loaded_traits]))
        self._resolve_pgs_ids_from_traits()
        self._sync_loaded_trait_selection()

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
