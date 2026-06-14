# Python Coding Standards & Best Practices

## Project Architecture

This project is a **uv workspace** with a non-published root wrapper and three subprojects:

- **`just-prs/src/just_prs/`** — Core library: PRS computation, PGS Catalog REST API client, FTP downloads, VCF reading, scoring file parsing. CLI entrypoint via Typer. Published to PyPI as `just-prs`.
- **`prs-ui/`** — Reflex web app for interactive PRS computation. Has its own `pyproject.toml` and depends on `just_prs`. Run with `uv run ui` or `uv run start` from workspace root or `uv run reflex run` from inside `prs-ui/`. Published to PyPI as `prs-ui`.
- **`prs-pipeline/`** — Dagster pipeline for computing PRS reference distributions from the 1000G panel. Has its own `pyproject.toml` and depends on `just_prs`. All pipeline commands (`run`, `catalog`, `launch`) default to launching the Dagster UI for monitoring. Use `--headless` on `run`/`catalog` for in-process execution without UI.

The workspace root (`pyproject.toml` at repo root) is a non-published wrapper named `just-prs-workspace`. It depends on all three subprojects and **must re-export all CLI entry points** from subprojects so that every command is available via `uv run <name>` from the workspace root. The pipeline CLI has three main commands: `pipeline run` (full pipeline with Dagster UI), `pipeline catalog` (catalog pipeline with Dagster UI), and `pipeline launch` (Dagster UI only, no specific job pre-selected). All three launch the Dagster UI by default. Use `--headless` on `run`/`catalog` for in-process execution without UI. Tests live in `just-prs/tests/`.

**ALL PIPELINE COMMANDS LAUNCH DAGSTER UI BY DEFAULT (CRITICAL).** `pipeline run`, `pipeline catalog`, and `pipeline launch` all start the Dagster webserver with monitoring UI. Headless in-process execution is only available via the explicit `--headless` flag on `run`/`catalog`. The Dagster UI URL (`http://<host>:<port>`) must always be printed prominently at startup.

**EVERY CLI THAT STARTS A SERVER MUST PRINT ITS URL (CRITICAL).** When any CLI command starts a web server or UI (Reflex UI via `uv run ui` or `uv run start`, Dagster UI via `pipeline run`/`catalog`/`launch`), the URL (`http://<host>:<port>`) must be printed prominently in the first lines of output so the user always knows where to open their browser.

**ALL CLIs LOAD `.env` VIA `python-dotenv` AT STARTUP (CRITICAL).** Both `prs_ui.cli` and `prs_pipeline.cli` call `load_dotenv()` before reading any configuration. Users override defaults by setting env vars in `.env` (see `.env.template`). Key config env vars: `PRS_UI_HOST`, `PRS_UI_PORT` (Reflex frontend, default `0.0.0.0:3000`), `PRS_UI_BACKEND_PORT` (Reflex backend, default `8000`; frontend and backend ports must both be passed explicitly to Reflex after port conflict resolution), `PRS_UI_DATA_DIR` (runtime data root for UI uploads, default `./data` from the directory where the launcher was invoked, so `uvx` runs never write into the installed package), `PRS_UI_PRESELECT_VCF` (optional local VCF path used only by `uv run preselect`), `PRS_UI_PRESELECT_QUERY` (optional `uv run preselect` text query that preselects matching PGS IDs, e.g. `diabetes`), `PRS_PIPELINE_HOST`, `PRS_PIPELINE_PORT` (Dagster UI, default `0.0.0.0:3010`), `PRS_CACHE_DIR`, `HF_TOKEN`, `PRS_PIPELINE_PANEL`, `PRS_PIPELINE_STARTUP_JOB`, `PRS_DUCKDB_MEMORY_LIMIT` (DuckDB per-connection memory cap for pvar joins, e.g. `"8GB"`; default: 50% of total RAM), `PRS_DUCKDB_MEMORY_PERCENT` (percentage of total RAM for DuckDB if `PRS_DUCKDB_MEMORY_LIMIT` is not set, default `50`), `PRS_GENO_CHUNK_SIZE` (override auto-sized genotype chunk, default auto), `PRS_MEMORY_SAFETY_PERCENT` (percent of total RAM kept as safety floor, default `10`), `PRS_MEMORY_SAFETY_MIN_MB` (minimum safety floor in MB, default `512`), `PRS_HARMONIZED_PENALTY` (quality penalty multiplier for harmonized cross-build PRS scores, float 0-1, default `0.90`; set to `1.0` to disable). When adding new configurable values, always read them from env vars with sensible defaults, document them in `.env.template`, and mention them in `AGENTS.md`.

**CRITICAL: All subproject CLI entry points must be registered in the workspace root `pyproject.toml` `[project.scripts]`.** When adding a new CLI entry point to any subproject, always add it to the root `pyproject.toml` as well. Users run commands from the workspace root with `uv run <script>`, and scripts not registered at the root level will not be found. Current entry points:

| Script | Entry point | Subproject |
|--------|-------------|------------|
| `prs` | `just_prs.cli:run` | `just-prs` |
| `just-prs` | `just_prs.cli:run` | `just-prs` |
| `ui` | `prs_ui.cli:launch_ui` | `prs-ui` |
| `start` | `prs_ui.cli:launch_ui` | `prs-ui` (alias of `ui`, general startup) |
| `preselect` | `prs_ui.cli:launch_preselect_ui` | `prs-ui` (loads configured test VCF and preselects matching scores) |
| `pipeline` | `prs_pipeline.cli:app` | `prs-pipeline` |

### Optional dependencies

`pgenlib` is **optional** — it is only needed for reference panel operations (`.pgen` file reading, batch scoring). `duckdb` is a **core dependency** since `compute_prs_duckdb()` (the default UI engine) uses it for variant-matching joins and weighted-sum aggregation. The core PRS computation from VCF, scoring file parsing, `PRSCatalog`, quality assessment, and all UI components require `duckdb` but work without `pgenlib`.

Install the `reference` extra when you need reference panel features:

```bash
pip install just-prs[reference]
# or in pyproject.toml:
"just-prs[reference]>=0.3.8"
```

**Why is pgenlib optional?** `pgenlib` requires C compilation and does not ship Windows wheels. Making it optional allows `just-prs` to be used on Windows (e.g. in `just-dna-lite`) for VCF-based PRS computation without requiring a C compiler. Functions that need `pgenlib` raise a clear `ImportError` with installation instructions when called without the extra.

**Windows: pgenlib is excluded by an environment marker (CRITICAL).** The `reference` extra is declared as `"pgenlib>=0.93.0; sys_platform != 'win32'"` in `just-prs/pyproject.toml`. This means even though both the workspace root and `prs-pipeline` depend on `just-prs[reference]`, a Windows `uv sync` / `uv run ui` resolves **without** pgenlib and never attempts the (failing) MSVC build. The bundled libdeflate C in pgenlib 0.94.0 does not compile with MSVC even when Visual C++ Build Tools are installed (it tries to build ARM-only sources on x64), so excluding it on Windows is the only robust option. The marker propagates everywhere `[reference]` is used, so the **entire workspace is installable on Windows** — only the reference-panel / `.pgen` features are unavailable there. Windows users who need reference scoring or the pipeline should use **WSL or Linux**. When changing reference dependencies, keep this `sys_platform != 'win32'` marker and run `uv lock`; verify with `uv export --python-platform x86_64-pc-windows-msvc` that pgenlib is absent on Windows and `uv tree --package just-prs` that it is present on Linux/macOS.

### Key modules

| Module | Purpose |
|--------|---------|
| `just_prs.prs_catalog` | **`PRSCatalog`** — high-level class for search, PRS computation, and percentile estimation using cleaned bulk metadata (no REST API calls). Persists cleaned parquets locally with HuggingFace sync; percentile lookup refreshes reference distributions from HF on miss; `reference_data_status()` reports whether precomputed reference data exists for a PGS ID, which superpopulations are available, and whether source is local cache vs HF sync. |
| `just_prs.cleanup` | Pure-function pipeline: genome build normalization, column renaming, metric string parsing, performance metric cleanup, publications cleanup |
| `just_prs.absolute_risk` | Absolute risk estimation from PRS z-scores and population prevalence. Two methods: OR-per-SD (`estimate_absolute_risk_or`) and AUC-bivariate-normal (`estimate_absolute_risk_auc`). Facade `estimate_absolute_risk` picks the best available method. See [methodology doc](docs/absolute-risk-methodology.md). |
| `just_prs.prevalence` | Prevalence data sourcing and consolidation. 3-tier merge: hand-curated seed CSV (Tier 1) > GWAS Catalog cohort fractions (Tier 2) > PGS eval cohort fractions (Tier 3). `build_prevalence_table()`, `pull_prevalence_from_hf()`, `push_prevalence_to_hf()`, `query_ols_xrefs()`, `build_efo_xrefs()`. |
| `just_prs.gwas` | GWAS Catalog bulk data download and parsing. `download_gwas_studies()` fetches the bulk TSV and parses case/control counts from free-text sample descriptions. `download_gwas_trait_mappings()` fetches trait-to-EFO mappings. `build_gwas_trait_summary()` joins and aggregates per-EFO-trait. |
| `just_prs.hf` | HuggingFace Hub integration: `pull_cleaned_parquets()` pulls cleaned metadata parquets from `just-dna-seq/pgs-catalog` (`data/metadata/`); `push_pgs_catalog()` uploads combined metadata+scores to `just-dna-seq/pgs-catalog` and rewrites `data/metadata/scores.parquet` to parquet-first scoring links (`ftp_link`) while preserving original EBI links in `ftp_link_ebi`. |
| `just_prs.normalize` | VCF normalization: `normalize_vcf()` reads VCF with polars-bio, strips chr prefix, renames id→rsid, computes genotype List[Str], applies configurable quality filters (FILTER, DP, QUAL), warns on chrY for females, sinks to zstd Parquet. `VcfFilterConfig` (Pydantic v2) holds filter settings. |
| `just_prs.prs` | `compute_prs()` / `compute_prs_duckdb()` / `compute_prs_batch()` — core PRS engines. Two engines: **polars** (lazy in-memory, default for API) and **DuckDB** (SQL, spills to disk, default in UI). `PRSEngine` str enum (`POLARS`, `DUCKDB`) for type-safe engine selection and `GenotypeInputMode` str enum (`AUTO`, `VARIANT_ONLY`, `ALL_SITES`, `PLINK_PRESENT_ONLY`) for absent-locus semantics. `compute_prs_duckdb()` accepts `genotypes_parquet` (preferred — DuckDB reads directly) or `genotypes_lf`, plus `memory_limit` param (falls back to `PRS_DUCKDB_MEMORY_LIMIT` env var, then 75% of RAM). Both engines support standard additive (`effect_weight`) and per-dosage (GenoBoost) weight formats, theoretical stats, and percentile computation. `is_dosage_weight_format()` detects the format; `_normalize_scoring_columns()` handles both transparently. |
| `just_prs.reference` | Reference panel utilities and pgen operations: `download_reference_panel()` (panel-aware: `panel="1000g"` or `"hgdp_1kg"`), `parse_pvar()` (parse .pvar.zst with parquet caching), `parse_psam()` (parse .psam sample files), `read_pgen_genotypes()` (extract genotypes from .pgen via pgenlib), `match_scoring_to_pvar()` (allele-aware variant matching via polars — standalone use only), `compute_reference_prs_polars()` (single-PGS scoring using pgenlib + numpy; uses DuckDB for variant matching against pvar parquet to avoid loading 75M rows into memory), `compute_reference_prs_batch()` (memory-efficient batch scoring: resolves panel once via `_ResolvedRefPanel`, uses DuckDB for variant matching, aggregates distributions per PGS ID immediately and discards raw scores, returns `BatchScoringResult`), `compute_reference_prs_plink2()` (legacy, for cross-validation), `aggregate_distributions()`, `distribution_quality_issues()` (one row per non-finite/zero-variance distribution anomaly for manual triage and exclusion decisions), `enrich_distributions()` (join distributions with cleaned metadata: traits, EFO, AUROC, OR, C-index, ancestry), `ancestry_percentile()`, `ReferencePanelError`. Panel-aware constants: `REFERENCE_PANELS` dict, `DEFAULT_PANEL = "1000g"`. Result models: `ScoringOutcome` (per-ID outcome), `BatchScoringResult` (panel, distributions_df, outcomes, quality_df, distribution_issues_df — no raw scores held in memory). `_ResolvedRefPanel` caches file paths, psam, and variant count once per batch; variant matching uses DuckDB to scan the pvar parquet (~434 MB on disk) without materializing 75M rows in polars (~6 GB). |
| `just_prs.chip_coverage` | Consumer-genotyping-chip coverage of PGS scoring files. `CHIPS` / `CHIPS_BY_ID` define supported chips (currently `gsa_v3` — Illumina Global Screening Array v3, the platform the current consumer market converged on: 23andMe v5, AncestryDNA v2, MyHeritage 2019+, FamilyTreeDNA v2, LivingDNA). `download_chip_manifest()` fetches the GSA **A2 (GRCh38)** manifest zip; `parse_gsa_manifest()` extracts ~648K typed `(chr_norm, pos)` markers; `chip_typed_positions()` caches unique positions to parquet; `compute_chip_coverage()` intersects each GRCh38 scoring parquet's `hm_chr`/`hm_pos` against the chip's typed positions and returns one row per `pgs_id × chip` with `n_typed`, `n_total`, `coverage_ratio`, and `array_ready` (bool: `coverage_ratio >= ARRAY_READY_THRESHOLD`, default 0.90, and the score has mapped coordinates). Answers "which PRS are array-ready vs imputation-required" per chip. **Coverage is a position-set intersection in a single build** — A2=GRCh38 means no liftover is needed against the GRCh38 harmonized scoring files. The GSA manifest is the shared platform core; it omits each vendor's custom add-on markers, so coverage is a slight under-estimate. Older arrays (Illumina OmniExpress — pre-2019 kits — and deCODEme Omni) are a different platform and not represented. **Imputation itself is per-individual and cannot be precomputed/downloaded** (it infers untyped genotypes from one person's observed alleles against a reference panel); only this static per-chip coverage map is precomputable. `PRSCatalog.scores()` left-joins the coverage (pivoted to wide per-chip columns `{chip}_array_ready` / `{chip}_coverage`) via `_load_chip_coverage()`, pulling `chip_coverage.parquet` from HF (`pull_chip_coverage`) on local miss — same lazy pattern as `quality_label`. |
| `just_prs.arrays` | Consumer genotyping-array ingestion. `normalize_array()` parses a 23andMe / AncestryDNA raw file (`.txt`/`.txt.gz`/`.csv`/`.zip`) into the **same** normalized Parquet schema as `normalize_vcf()` (`chrom`, `pos`, `rsid`, `ref`, `alt`, `GT`, `genotype`), so `compute_prs()` / `compute_prs_duckdb()` consume array data unchanged. `detect_array_format()` auto-detects vendor (4-col 23andMe vs 5-col AncestryDNA). **Encoding trick:** arrays report observed alleles directly, so het `a1≠a2` → `ref=a1,alt=a2,GT="0/1"` and hom `a1==a2` → `ref=alt=a,GT="1/1"`, which makes the existing GT/ref/alt dosage logic count the effect allele correctly. Defaults to `genome_build="GRCh37"` (the build 23andMe v5 / AncestryDNA v2 report) — the caller must score against the matching GRCh37 harmonized file. **Known limitations** (documented, not silently handled): strand flips, indels (`I`/`D` codes), and hemizygous male X/Y (treated as homozygous). |
| `just_prs.vcf` | VCF reading via `polars-bio`, genome build detection, dosage computation |
| `just_prs.scoring` | Download, parse, and cache PGS scoring files. `SCORING_FILE_SCHEMA` — comprehensive column type map from the PGS Catalog spec (30+ columns). `parse_scoring_file()` transparently reads/writes a parquet cache (zstd-9 compressed) alongside the `.txt.gz`, with header metadata embedded as file-level metadata. `scoring_parquet_path()` computes cache paths. `read_scoring_header()` reads PGS header metadata from parquet or `.txt.gz`. `load_scoring()` checks parquet cache first and skips `.txt.gz` download when it exists. |
| `just_prs.ftp` | Bulk FTP/HTTPS downloads of raw metadata sheets and scoring files via `fsspec` |
| `just_prs.catalog` | Synchronous REST API client (`PGSCatalogClient`) for PGS Catalog — used for individual lookups, not for bulk metadata |
| `just_prs.models` | Pydantic v2 models (`ScoreInfo`, `PRSResult`, `PerformanceInfo`, `AbsoluteRisk`, `PublicationInfo`, etc.) |
| `just_prs.quality` | Pure-logic quality assessment helpers: `classify_model_quality()`, `interpret_prs_result()`, `format_effect_size()`, `format_classification()`. No Reflex dependency -- shared between core library and UI. |
| `prs_ui.state` | Reflex `AppState` + grid states + `PRSComputeStateMixin(rx.State, mixin=True)`. The mixin encapsulates all PRS computation logic (score loading, selection, batch compute, trait summary aggregation, CSV export) and is the genotype **consumer** (genotypes are pushed in via its additive `load_genotypes(path)` hook). `GenomicGridState` is the detachable VCF **source** that normalizes an upload and fans the normalized parquet + detected build out to its registered `_consumer_states`. `ComputeGridState` (By PRS) and `TraitBrowserState` (By Trait) subclass the mixin as the two consumers in the single Compute workbench. |
| `prs_ui.components` | **Reusable UI components**: `prs_workbench(source_section, prs_state, trait_state, mode_state, trait_selector, ...)` (the unified single-tab layout: shared source + By PRS / By Trait sub-tabs), `vcf_source_section(source_state)` (compact VCF upload + collapsed normalized preview), `prs_shared_build_bar(source_state)` (one genome-build selector that fans out to all consumers), plus the per-state pieces `prs_section(state)`, `prs_scores_selector(state)`, `prs_results_table(state)`, `trait_summary_table(state)`, `prs_progress_section(state)`, `prs_build_selector(state)`, `prs_compute_button(state)`, `prs_engine_selector(state)`, `prs_ancestry_selector(state)`. Each takes a state class parameter so the same components work with any concrete state inheriting `PRSComputeStateMixin`. |
| `prs_ui.pages.*` | UI panels: `metadata` (grid browser), `scoring` (file viewer), `compute` (the unified Compute PRS workbench assembled from `prs_workbench` + `vcf_source_section`), `traits` (exposes the reusable `trait_selector` grid used by the workbench's By Trait sub-tab) |
| `prs_pipeline.runtime` | `ResourceReport` (Pydantic model) and `resource_tracker` context manager — tracks CPU%, peak memory, duration via `psutil` and logs to Dagster output metadata |
| `prs_pipeline.utils` | `resource_summary_hook` — Dagster `@success_hook` that aggregates per-asset resource metrics into a run-level summary |
| `prs_pipeline.checks` | Dagster `@asset_check` definitions for data quality validation. Checks run after asset materialization and surface in the Dagster UI. `ALL_ASSET_CHECKS` collects all checks for registration in `Definitions`. |

### Cleanup pipeline (`just_prs.cleanup`)

Raw PGS Catalog CSVs have data quality issues that `cleanup.py` fixes:
- **Genome build normalization**: 9 raw variants (hg19, hg37, hg38, NCBI36, hg18, NCBI35, GRCh37, GRCh38, NR) are mapped to canonical `GRCh37`, `GRCh38`, `GRCh36`, or `NR` via `BUILD_NORMALIZATION` dict.
- **Column renaming**: Verbose PGS column names (e.g. `Polygenic Score (PGS) ID`) become snake_case (`pgs_id`). The full mapping is `_SCORES_COLUMN_RENAME` / `_PERF_COLUMN_RENAME` / `_EVAL_COLUMN_RENAME` / `_PUBLICATIONS_COLUMN_RENAME`.
- **Metric string parsing**: Performance metrics stored as strings like `"1.55 [1.52,1.58]"` or `"-0.7 (0.15)"` are parsed into `{estimate, ci_lower, ci_upper, se}` via `parse_metric_string()`.
- **Performance flattening**: `clean_performance_metrics()` joins with evaluation sample sets and produces numeric columns for OR, HR, Beta, AUROC, and C-index. Evaluation sample sets now preserve `n_cases` and `n_controls` for prevalence estimation. `best_performance_per_score()` selects one row per PGS ID (largest sample, European-preferred).
- **Publications cleaning**: `clean_publications()` transforms raw `pgs_all_metadata_publications.csv` into snake_case with columns: `pgp_id`, `first_author`, `title`, `journal`, `year`, `doi`, `pmid`.

### PRSCatalog class (`just_prs.prs_catalog`)

`PRSCatalog` is the primary interface for working with PGS Catalog data. It produces and persists 4 cleaned parquet files (`scores.parquet`, `performance.parquet`, `best_performance.parquet`, `publications.parquet`) and loads them as LazyFrames. Loading uses a 3-tier fallback chain: local cleaned parquets -> HuggingFace pull -> raw FTP download + cleanup. Raw FTP parquets are cached separately in a `raw/` subdirectory to avoid collision with cleaned files.

Key methods: `scores()`, `search()`, `best_performance()`, `publications()`, `score_info_row()`, `compute_prs()`, `compute_prs_batch()`, `percentile()`, `absolute_risk()`, `reference_data_status()`, `build_cleaned_parquets()`, `push_to_hf()`. The `absolute_risk(pgs_id, z_score, sex=None)` method joins scores, best_performance, prevalence, and publications data to produce an `AbsoluteRisk` estimate using the best available method (OR-per-SD or AUC-bivariate-normal).

The package public API (`just_prs.__init__`) exports: `PRSCatalog`, `ReferencePanelError`, `AbsoluteRisk`, `normalize_vcf`, `VcfFilterConfig`, `resolve_cache_dir`, `classify_model_quality`, `interpret_prs_result`, `format_effect_size`, `format_classification`, `PRSEngine`, `GenotypeInputMode`, `compute_reference_prs_polars`, `compute_reference_prs_batch`, `download_reference_panel`, `reference_panel_dir`, `parse_pvar`, `parse_psam`, `read_pgen_genotypes`, `match_scoring_to_pvar`, `aggregate_distributions`, `distribution_quality_issues`, `reference_distribution_audit_issues`, `enrich_distributions`, `ancestry_percentile`, `ReferenceDistribution`, `ScoringOutcome`, `BatchScoringResult`, `REFERENCE_PANELS`, `DEFAULT_PANEL`, `__version__`, `__package_name__`.

The `prs-ui` package public API (`prs_ui.__init__`) exports: `PRSComputeStateMixin`, `prs_workbench`, `vcf_source_section`, `prs_shared_build_bar`, `prs_section`, `prs_scores_selector`, `prs_results_table`, `trait_summary_table`, `prs_progress_section`, `prs_build_selector`, `prs_engine_selector`, `prs_compute_button`, `prs_ancestry_selector`.

### HuggingFace sync (`just_prs.hf`)

Cleaned metadata parquets (including `publications.parquet` and `trait_prevalence.parquet`) are synced to/from the HuggingFace dataset repo `just-dna-seq/pgs-catalog` under the `data/metadata/` prefix. The HF token is resolved from: explicit argument > `.env` file (via `python-dotenv`) > `HF_TOKEN` environment variable. CLI commands: `just-prs catalog bulk clean-metadata`, `push-catalog`, `pull-hf`.

### UI architecture notes

- **Single Compute PRS tab with two sub-tabs.** The top-level tabs are `Compute PRS`, `Metadata Sheets`, `Scoring File` (the old separate `Browse by Trait` top-level tab was removed). The Compute PRS tab is a unified workbench (`prs_workbench`): one shared, compact, **detachable** genotype source (VCF upload) at the top, then native `rx.tabs` sub-tabs `Select by PRS` (individual scores) and `Select by Trait` (trait groups). Results are shown for the **active** sub-tab only — By PRS renders the individual `prs_results_table`, By Trait renders the trait-grouped `trait_summary_table`. The sub-tab is bound to `AppState.compute_mode` (`"prs"` / `"trait"`) via `set_compute_mode`.
- **Loose-coupling contract (genotype source ⇄ consumer).** The source never lives in the mixin; it pushes normalized genotypes into each consumer via the additive `load_genotypes(path)` hook (and optionally `set_genome_build(build)`). This keeps `PRSComputeStateMixin` swappable: a host app such as just-dna-lite can supply its own source (public genome, consumer-array file, pre-normalized parquet) without touching the consumers or the mixin.
- **State classes** with independent MUI DataGrids via `LazyFrameGridMixin` (which uses `mixin=True`). Each concrete mixin subclass gets its own independent set of reactive grid vars:
  - `AppState(rx.State)` — shared vars: `active_tab`, `compute_mode`, `genome_build`, `cache_dir`, `status_message`, `pgs_id_input`. Provides `set_compute_mode` for the By PRS / By Trait sub-tab switch.
  - `MetadataGridState(LazyFrameGridMixin, AppState)` — metadata browser + scoring file viewer grid.
  - `GenomicGridState(LazyFrameGridMixin, AppState)` — **the reference detachable VCF source.** Owns all VCF UI state (`vcf_filename`, `detected_build`, `build_detection_message`, `_vcf_path`, `vcf_normalizing`, normalized parquet + preview grid). `handle_vcf_upload()` saves + `normalize_uploaded_vcf()` runs `normalize_vcf()` (strip chr prefix, compute genotype, PASS filter), then `_push_to_consumers()` feeds the normalized parquet + detected build to every state in the `_consumer_states: ClassVar[list[type]]` registry (assigned at module bottom: `GenomicGridState._consumer_states = [ComputeGridState, TraitBrowserState]`). `initialize_source()` does the same for an optional preloaded VCF; `set_shared_genome_build()` fans a manual build change to all consumers. **Fan-out mutates consumers directly via `await self.get_state(...)`** — never by yielding cross-state `EventSpec`s after the blocking `normalize_vcf()` call (that triggers Reflex's "Cannot add a child to an EventFuture that is already done" error and stalls the event queue, which manifests as sluggish/broken grid checkbox selection).
  - `PRSComputeStateMixin(rx.State, mixin=True)` — **reusable** genotype-consumer mixin: score loading via `PRSCatalog`, row selection, batch PRS computation, quality assessment, CSV export. Accepts genotypes via the additive `load_genotypes(path)` hook (the loose-coupling entry point used by any source), or directly via `set_prs_genotypes_lf()` / `prs_genotypes_path`. Designed for embedding in any Reflex app.
  - `ComputeGridState(PRSComputeStateMixin, LazyFrameGridMixin, AppState)` — the **By PRS** consumer. Owns no VCF/upload logic (genotypes are pushed in). `prs_view_mode` is fixed to `"individual"`, so it always renders the individual results table and never builds a trait summary.
  - `TraitBrowserState(PRSComputeStateMixin, LazyFrameGridMixin, AppState)` — the **By Trait** consumer. Groups PGS Catalog scores by EFO trait, tracks selected traits, resolves them to PGS IDs. `prs_view_mode` is `"grouped"`; its `compute_selected_prs()` override calls the base mixin then `build_trait_summary()` so the grouped view is the output.
  - **Important**: `AppState` must NOT inherit from `LazyFrameGridMixin` — otherwise substates that also list the mixin create an unresolvable MRO diamond.
- **Reusable components** (`prs_ui.components`): Each component function accepts a `state` class parameter, so the same UI works with any concrete state inheriting `PRSComputeStateMixin`. `prs_workbench(...)` is the unified single-tab layout (pluggable `source_section`, `prs_state`, `trait_state`, `mode_state`, `trait_selector`, optional `build_bar`, plus forwarded `results_table_kwargs` / `trait_summary_kwargs`) — render it in a host app with your own `source_section` to reuse the whole By PRS / By Trait experience. `vcf_source_section(source_state)` is the reference compact upload (collapsed normalized-VCF preview); `prs_shared_build_bar(source_state)` is the one-control build selector that fans out. `prs_section(state)` remains the older single-state entry point (build selector, score grid, compute button, progress, results). `trait_summary_table(state)` can be used independently for trait-grouped views. The per-mode controls include an `All available populations` toggle to request per-superpopulation percentiles where reference distributions exist.
- **Bell curve sizing is configurable** in both result tables. `prs_results_table(state, bell_curve_height=360, bell_curve_max_width=1200, detail_height="auto", bell_curve_config=None)` and `trait_summary_table(state, bell_curve_height=380, bell_curve_max_width=1200, large_bell_curve_threshold=4, large_bell_curve_height=460, large_bell_curve_max_width=1600, detail_height="auto", bell_curve_config=None)` expose the chart dimensions. **`detail_height` defaults to `"auto"`** so the detail panel grows to fit the bell curve — never pass a fixed numeric height unless you specifically need internal scroll within the panel. `bell_curve_config` is a shallow-merged dict of extra renderer keys (`labelTiers`, `labelMinGapZ`, `bands`, `marginTop`, etc.) for full per-app overrides. `prs_section(state, results_table_kwargs=None, trait_summary_kwargs=None)` forwards those dicts so embedders never need to fork the sub-tables to bump chart size. Default bell curve dimensions are sized to fit alongside the side panel without changing the underlying renderer layout; do not override `marginTop`/`marginBottom`/`legendY`/`yAxisMax` defaults unless you specifically need more headroom (changing them alters the curve aspect ratio).
- The Metadata tab shows **raw** PGS Catalog columns for general-purpose browsing of all 7 sheets.
- The Compute tab (default tab) uses **cleaned** data from `PRSCatalog` with normalized genome builds and snake_case column names. Scores are loaded into the MUI DataGrid with server-side virtual scrolling — no manual pagination. By default, **harmonized scores are included** (`include_harmonized=True`): when a user selects GRCh38, all ~5,300 scores are shown (not just ~600 native GRCh38), with a "Source" badge column distinguishing "Native" (green) from "Harmonized" (orange). The "Original Build" column shows each score's development build. The "Include harmonized scores" checkbox lives in each sub-tab's per-mode controls (`_workbench_mode_controls`), since it is a per-consumer setting. Harmonized scores receive a configurable quality penalty (`PRS_HARMONIZED_PENALTY`, default 0.85) in `synthetic_quality_score()` because coordinate liftover may introduce minor mapping errors.
- VCF upload triggers automatic normalization via `GenomicGridState.normalize_uploaded_vcf()` which runs `normalize_vcf()` (strip chr prefix, compute genotype, PASS filter) and shows the result in a browsable, collapsed-by-default preview grid. The normalized parquet is then pushed into **both** consumers (`ComputeGridState` and `TraitBrowserState`) via `load_genotypes(path)`, so a single upload powers both the By PRS and By Trait sub-tabs. The genome build is shared the same way via `prs_shared_build_bar` → `set_shared_genome_build`.
- **Normalization is the slow step, not upload — and it is content-aware cached (CRITICAL).** `normalize_uploaded_vcf()` reuses an existing normalized parquet when it is at least as recent as the source VCF (mtime check), so re-uploading the same file is instant. **Do NOT wire `on_upload_progress` on the VCF dropzone:** `normalize_vcf()` is synchronous CPU-bound work that blocks the asyncio event loop, so buffered upload-progress events replay *after* normalization completes and re-set the "uploading" flag with nothing left to clear it — which is what made normalization appear to "never finish" and "fire twice". Feedback is a single boolean gate `GenomicGridState.vcf_normalizing` driving a spinner + **indeterminate** `rx.progress` (no `value`); never show a fake determinate percentage for normalization.
- **Selection grids are read-only until genotypes load.** Both `prs_scores_selector` and the trait `trait_selector` gate on `selection_ready = (state.prs_genotypes_path != "") & ~GenomicGridState.vcf_normalizing`: the checkbox column is hidden (`checkbox_selection=selection_ready`), the grid box is dimmed (`opacity 0.55`) and made non-interactive (`pointer_events="none"`), the Select/Clear buttons are disabled, and an explicit callout tells the user to upload a VCF first (switching to a "normalizing…" message while `vcf_normalizing`). Scores/traits still load and render on page open so the catalog is browsable; only selection is locked.
- PRS results include **quality assessment**: AUROC-based model quality labels (High/Moderate/Low/Very Low), effect sizes (OR/HR/Beta with CI), classification metrics (AUROC/C-index), evaluation population ancestry, and plain-English interpretation summaries. Results can be exported as CSV via `download_prs_results_csv()`.
- PRS result rows use **foldable detail panels** (reflex-mui-datagrid >= 0.2.0 `detail_columns`) to show interpretation, quality summary, population percentiles, reference source, and effect size inline below each row. The old separate "Detailed interpretation cards" section has been replaced by these inline expandable panels.
- PRS result rows show explicit **reference percentile status**: whether precomputed 1000G reference data exists (`precomputed(...)` vs `not precomputed`), which populations are available, and the source (`HuggingFace prs-percentiles` vs local cache). UI text clarifies these reference distributions are precomputed from reference panel scoring, not direct PGS Catalog API percentiles.
- `lazyframe_grid()` already sets `pagination=False`, `hide_footer=True`, and `on_row_selection_model_change` internally — do NOT pass any of these again or you get a duplicate kwarg error. To customize row selection handling, override `handle_lf_grid_row_selection` in the concrete state class.

### Running the UI

The web UI is a Reflex app in the `prs-ui/` workspace member:

```bash
uv sync --all-packages
uv run ui   # alias: uv run start
```

Or equivalently:

```bash
cd prs-ui
uv run reflex run
```

This launches a local web server (default http://0.0.0.0:3000 with backend on port 8000). The CLI uses Reflex's port conflict handling for both frontend and backend, passes both resolved ports explicitly to Reflex, and always prints the UI URL prominently at startup. The `ui` and `start` script entry points (aliases) are defined in both the root `pyproject.toml` (convenience) and `prs-ui/pyproject.toml`, pointing to `prs_ui.cli:launch_ui`.

### Computing PRS on a custom VCF

**Via CLI:**

```bash
# Normalize a VCF to Parquet first (optional, but recommended for repeated analyses)
prs normalize --vcf /path/to/your/sample.vcf.gz --pass-filters "PASS,." --min-depth 10

# Single score
prs compute --vcf /path/to/your/sample.vcf.gz --pgs-id PGS000001

# Multiple scores at once
prs compute --vcf /path/to/your/sample.vcf.gz --pgs-id PGS000001,PGS000002,PGS000003

# Explicit genome build + JSON output
prs compute --vcf /path/to/your/sample.vcf.gz --pgs-id PGS000001 --build GRCh37 --output results.json
```

**Regular VCF vs gVCF scoring semantics:** `compute_prs()` and `compute_prs_duckdb()` default to `genotype_input_mode="auto"`. Regular uploaded VCFs are treated as **variant-only**: an absent scoring locus can be inferred as homozygous-reference only when the scoring file provides an explicit `reference_allele`; if the reference allele is unknown, the locus is counted as `variants_unscorable_absent` and is not silently guessed. gVCF/all-sites/ref-block inputs are treated as `all_sites`, where an absent locus remains unavailable. Use `genotype_input_mode="plink_present_only"` when validating against PLINK2 on a PGEN converted from a variant-only VCF, because PLINK only scores variants present in that PGEN and does not infer absent loci as homozygous-reference. The UI exposes `variants_observed`, `variants_assumed_hom_ref`, `variants_unscorable_absent`, and `variants_no_call` so low coverage is not confused with ancestry mismatch.

**Via Python API (recommended for scripting):**

```python
from just_prs import PRSCatalog, normalize_vcf, VcfFilterConfig
from pathlib import Path

catalog = PRSCatalog()

# Normalize VCF to Parquet (strip chr prefix, compute genotype, quality filter)
config = VcfFilterConfig(pass_filters=["PASS", "."], min_depth=10)
parquet_path = normalize_vcf(Path("/path/to/your/sample.vcf.gz"), Path("normalized.parquet"), config=config)

# Search for scores related to a trait
scores = catalog.search("type 2 diabetes", genome_build="GRCh38").collect()
print(scores.select("pgs_id", "name", "trait_reported"))

# Compute PRS for a single score
result = catalog.compute_prs(vcf_path=Path("/path/to/your/sample.vcf.gz"), pgs_id="PGS000001")
print(f"Score: {result.score:.6f}, Match rate: {result.match_rate:.1%}")

# Batch computation for multiple scores
results = catalog.compute_prs_batch(
    vcf_path=Path("/path/to/your/sample.vcf.gz"),
    pgs_ids=["PGS000001", "PGS000002", "PGS000003"],
)
for r in results:
    print(f"{r.pgs_id}: score={r.score:.6f}, matched={r.variants_matched}/{r.variants_total}")

# Look up best evaluation performance for a score
best = catalog.best_performance(pgs_id="PGS000001").collect()
```

**Via Web UI:** Open the Compute PRS tab and upload your VCF once (drag-and-drop) into the shared source at the top; the genome build is auto-detected. Then pick a sub-tab: **Select by PRS** to check individual scores and compute an individual results table, or **Select by Trait** to select entire trait groups (e.g. "type 2 diabetes mellitus") — all associated PGS models are computed and automatically grouped into a trait summary with consensus bell curves, outlier detection, and quality breakdown. Both sub-tabs share the same uploaded VCF.

### PLINK2 binary format operations

The `just_prs.reference` module provides pure Python operations on PLINK2 binary format files (.pgen/.pvar.zst/.psam) using `pgenlib` + polars + numpy:

| PLINK2 command | just-prs equivalent | Description |
|---|---|---|
| `plink2 --make-just-pvar` | `parse_pvar(path)` | Parse .pvar.zst with parquet caching |
| `plink2 --make-just-psam` | `parse_psam(path)` | Parse .psam sample file |
| `plink2 --extract` (genotype read) | `read_pgen_genotypes(...)` | Read genotypes for selected variants |
| `plink2 --score` | `compute_reference_prs_polars(...)` | Full PRS scoring pipeline |

**Via CLI (`prs pgen` — works with any .pgen dataset):**

```bash
# Read variant table from .pvar.zst
prs pgen read-pvar /path/to/panel.pvar.zst

# Read sample table from .psam
prs pgen read-psam /path/to/panel.psam

# Extract genotypes for a genomic region
prs pgen genotypes panel.pgen panel.pvar.zst panel.psam --chrom 11 --start 69M --end 70M

# Score a PGS ID against any .pgen dataset
prs pgen score PGS000001 /path/to/pgen_dir/
```

**Via CLI (`prs reference` — reference panel operations):**

```bash
# Download a reference panel (~7 GB for 1000g, ~15 GB for hgdp_1kg)
prs reference download
prs reference download --panel hgdp_1kg

# Batch score all PGS IDs (primary workflow for building distributions)
prs reference score-batch
prs reference score-batch --pgs-ids PGS000001,PGS000002,PGS000003
prs reference score-batch --limit 50 --panel hgdp_1kg

# Score a single PGS ID (for testing)
prs reference score PGS000001

# Compare both engines side-by-side (cross-validation)
prs reference compare PGS000001

# Test multiple PGS IDs with automated validation
prs reference test-score
prs reference test-score --pgs-ids PGS000001,PGS000003,PGS000007
```

**Via Python API (batch scoring — primary workflow):**

```python
from pathlib import Path
from just_prs import compute_reference_prs_batch, reference_panel_dir
from just_prs.scoring import resolve_cache_dir

cache_dir = resolve_cache_dir()
ref_dir = reference_panel_dir(cache_dir, panel="1000g")

result = compute_reference_prs_batch(
    pgs_ids=["PGS000001", "PGS000002", "PGS000003"],
    ref_dir=ref_dir,
    cache_dir=cache_dir,
    genome_build="GRCh38",
    panel="1000g",
    skip_existing=True,
)
# result.distributions_df: pgs_id, superpopulation, mean, std, n, ...
# result.outcomes: list[ScoringOutcome] with status, error, timing per ID
# result.quality_df: polars DataFrame version of outcomes
# result.distribution_issues_df: distribution-level anomalies for manual exclusion triage
```

**Via Python API (single PGS scoring + building blocks):**

```python
from pathlib import Path
from just_prs import (
    compute_reference_prs_polars, reference_panel_dir,
    aggregate_distributions, parse_pvar, parse_psam, read_pgen_genotypes,
)

ref_dir = reference_panel_dir()
scoring_file = Path("~/.cache/just-prs/scores/PGS000001_hmPOS_GRCh38.txt.gz").expanduser()

scores_df = compute_reference_prs_polars(
    pgs_id="PGS000001",
    scoring_file=scoring_file,
    ref_dir=ref_dir,
    out_dir=Path("/tmp/pgs000001"),
    genome_build="GRCh38",
)
dist_df = aggregate_distributions(scores_df)

# Or use building blocks individually:
pvar_df = parse_pvar(ref_dir / "some_panel.pvar.zst")
psam_df = parse_psam(ref_dir / "some_panel.psam")
geno = read_pgen_genotypes(
    pgen_path=ref_dir / "some_panel.pgen",
    pvar_zst_path=ref_dir / "some_panel.pvar.zst",
    variant_indices=pvar_df.head(100)["variant_idx"].cast(pl.UInt32).to_numpy(),
    n_samples=psam_df.height,
)
```

**Via Python API (PLINK2 engine — for cross-validation, requires binary):**

```python
from pathlib import Path
from just_prs.reference import compute_reference_prs_plink2, reference_panel_dir

ref_dir = reference_panel_dir()
plink2_bin = Path("~/.cache/just-prs/plink2/plink2").expanduser()
scoring_file = Path("~/.cache/just-prs/scores/PGS000001_hmPOS_GRCh38.txt.gz").expanduser()

scores_df = compute_reference_prs_plink2(
    pgs_id="PGS000001",
    scoring_file=scoring_file,
    ref_dir=ref_dir,
    out_dir=Path("/tmp/pgs000001"),
    plink2_bin=plink2_bin,
    genome_build="GRCh38",
)
```

### Embedding the PRS UI in another Reflex app (e.g. just-dna-lite)

The PRS computation UI is packaged as reusable Reflex components. The genotype source is **loosely coupled**: a host app feeds normalized genotypes into one or more consumer states via the additive `load_genotypes(path)` hook, then renders either the whole `prs_workbench` (single tab, By PRS / By Trait sub-tabs) or the older single-state `prs_section`. The host supplies its **own** source — it does not have to use `vcf_source_section` / `GenomicGridState` (e.g. just-dna-lite can drive the same consumers from a public-genome selector).

```python
import polars as pl
import reflex as rx
from reflex_mui_datagrid import LazyFrameGridMixin
from prs_ui import PRSComputeStateMixin, prs_workbench


class MyAppState(rx.State):
    genome_build: str = "GRCh38"
    cache_dir: str = "/path/to/cache"
    status_message: str = ""
    compute_mode: str = "prs"

    def set_compute_mode(self, value: str | list[str]) -> None:
        self.compute_mode = value if isinstance(value, str) else (value[0] if value else "prs")


class ByPRSState(PRSComputeStateMixin, LazyFrameGridMixin, MyAppState):
    """By PRS consumer."""
    prs_view_mode: str = "individual"


class ByTraitState(PRSComputeStateMixin, LazyFrameGridMixin, MyAppState):
    """By Trait consumer (auto-builds the trait summary)."""
    prs_view_mode: str = "grouped"

    def compute_selected_prs(self):  # type: ignore[override]
        yield from PRSComputeStateMixin.compute_selected_prs(self)
        self.build_trait_summary()


def my_genome_source() -> rx.Component:
    """Host app's own source: it just needs to call consumer.load_genotypes(path)."""
    ...  # e.g. a public-genome dropdown whose handler does:
    #   for C in (ByPRSState, ByTraitState):
    #       consumer = await self.get_state(C)
    #       consumer.load_genotypes(parquet_path)
    #       for event in consumer.set_genome_build(build): yield event


def prs_page() -> rx.Component:
    return prs_workbench(
        source_section=my_genome_source(),
        prs_state=ByPRSState,
        trait_state=ByTraitState,
        mode_state=MyAppState,
        trait_selector=lambda: ...,  # your trait-selection grid (or reuse prs_ui.pages.traits.trait_selector)
        results_table_kwargs={"bell_curve_height": 360, "bell_curve_max_width": 1200},
        trait_summary_kwargs={"bell_curve_height": 460},
    )
```

Key integration points:
- **`load_genotypes(path)` is the loose-coupling contract.** Any source pushes a normalized genotypes parquet into a consumer with `consumer.load_genotypes(path)` (it sets `prs_genotypes_path`, rescans the LazyFrame, and resets stale results). Mutate consumers directly via `await self.get_state(...)` from the source handler — do NOT yield cross-state `EventSpec`s after a long blocking normalize (causes the "EventFuture already done" error and stalls the event queue).
- **LazyFrame is still the in-state input** -- `load_genotypes` resolves the path to a `pl.scan_parquet()` LazyFrame internally; you can also call `set_prs_genotypes_lf(lf)` directly. Memory-efficient and avoids re-reading on each computation.
- The host app's state must provide `genome_build`, `cache_dir`, and `status_message` vars (inherited from a shared parent or defined directly). For `prs_workbench`, the `mode_state` must provide `compute_mode` + `set_compute_mode`.
- Call `initialize_prs()` (By PRS) / `initialize_traits()` (By Trait) on page load to auto-load scores/traits into the grids.
- **`prs_workbench` is the whole reusable layout** (shared source + By PRS / By Trait sub-tabs via `rx.tabs`, per-mode controls, compute button, per-mode results). Individual sub-components (`prs_scores_selector`, `prs_results_table`, `trait_summary_table`, `prs_compute_button`, `prs_shared_build_bar`, etc.) can still be used independently for custom layouts, and the single-state `prs_section(state)` remains available. Host apps with their own genotype source can pass `normalizing=<state_var>` to `prs_workbench`, `prs_scores_selector`, `prs_compute_button`, and `trait_selector` so selection/compute controls stay disabled while that source prepares genotypes; the default remains `GenomicGridState.vcf_normalizing` for the standalone app.
- **Trait summary** is available via `trait_summary_table(state)` — groups PRS results by EFO trait and shows consensus bell curves, outlier detection, quality breakdown, and per-trait aggregated statistics. Call `state.build_trait_summary()` after `compute_selected_prs()` completes to populate it, or override `compute_selected_prs()` in a concrete state to auto-build (as `ByTraitState`/`TraitBrowserState` does).
- **Bell curve dimensions are first-class config**: `prs_workbench` and `prs_section` forward `results_table_kwargs` and `trait_summary_kwargs` to the underlying tables. Use `bell_curve_height` / `bell_curve_max_width` / `detail_height` for size, and `bell_curve_config={"labelTiers": 12, "bands": [...], ...}` to layer on any `bell_curve` renderer key. Defaults preserve the standard layout.

### Reflex-specific patterns (CRITICAL)

- **State var mixin classes MUST use `rx.State` with `mixin=True`**: Declare mixins as `class MyMixin(rx.State, mixin=True)` so vars are injected independently into each concrete subclass. Each subclass must also inherit from `rx.State` (or another non-mixin state class). `LazyFrameGridMixin` already uses `mixin=True`, so `AppState` and `ComputeGridState` each get their own `lf_grid_rows`, `lf_grid_loaded`, etc.

  ```python
  # CORRECT — mixin=True, each child gets independent vars
  class MyMixin(rx.State, mixin=True):
      my_count: int = 0
  class GridA(MyMixin, rx.State): ...
  class GridB(MyMixin, rx.State): ...
  # GridA.my_count and GridB.my_count are INDEPENDENT rx.Var objects

  # WRONG — without mixin=True, all children share the SAME vars
  class MyMixin(rx.State):
      my_count: int = 0
  class GridA(MyMixin): ...
  class GridB(MyMixin): ...

  # ALSO WRONG — plain Python mixin without rx.State, vars stay as raw types
  class MyMixin:
      my_count: int = 0
  class AppState(MyMixin, rx.State): ...
  ```
- **No keyword-only arguments in mixin event handler methods**: Reflex's `_copy_fn` copies `__defaults__` but not `__kwdefaults__`. Always use regular positional arguments with defaults in mixin event handlers.
- **`pagination=False` for scrollable grids**: `WrappedDataGrid` defaults to `pagination=True`. You MUST pass `pagination=False` and `hide_footer=True` to get a continuously scrollable grid. NOTE: `lazyframe_grid()` already does this internally — only pass these when using `data_grid()` directly.
- **Detail panel height MUST be `"auto"` (CRITICAL — bell curve visibility).** When `detail_height` is omitted or `None`, the datagrid JS computes a tiny fallback (`max(120, columns×32+24)` ≈ 184px for 5 columns) that clips bell curves configured at 360–460px. Always pass `detail_height="auto"` so the panel grows to fit its content. Never use a fixed numeric `detail_height` unless you specifically need internal scroll within the panel.
- **Grids with auto-height detail panels MUST use the viewport-bounded flex column layout (CRITICAL — prevents page-scroll regression).** When `detail_height="auto"`, the expanded detail panel grows inside the grid's virtual scroller. Without a constrained flex host, the grid can push the page scroll instead of scrolling internally. The required pattern:

  ```python
  # CORRECT — grid scrolls internally, detail panels expand freely
  rx.box(
      sibling_above,                          # flex_shrink="0"
      rx.box(
          data_grid_scroll_container(
              data_grid(..., height="100%"),   # fills the flex slot
          ),
          flex="1 1 0%",
          min_height="0",
          overflow="hidden",                   # prevents leakage
          width="100%",
      ),
      sibling_below,                          # flex_shrink="0"
      display="flex",
      flex_direction="column",
      height="calc(100vh - <chrome>px)",      # viewport-bounded
      min_height="0",
      width="100%",
  )

  # WRONG — grid has its own calc height but sits in an unconstrained vstack
  rx.vstack(
      data_grid_scroll_container(
          data_grid(..., height="calc(100vh - 380px)"),
      ),
  )
  ```

  Key rules: (1) the flex root has a viewport-bounded height; (2) the grid wrapper gets `flex="1 1 0%"`, `min_height="0"`, `overflow="hidden"`; (3) non-grid siblings get `flex_shrink="0"`; (4) the grid itself uses `height="100%"` (not a calc); (5) `data_grid_scroll_container` passes `height="100%"` through to maintain the chain. Do NOT put `calc(100vh - N)` on the grid directly when using auto detail panels — put it on the flex root.

### polars-bio caveats

- `polars-bio` uses DataFusion as its query engine for VCF reading. Multi-column aggregations on DataFusion-backed LazyFrames can fail with "all columns in a record batch must have the same length". **Always `.collect()` the joined LazyFrame first**, then compute aggregations on the materialized DataFrame.

---

## Data Directory Conventions

**Data must be strictly separated from code.** Generated data, downloaded files, uploaded files, and computation outputs must NEVER be written to the project root or source tree. This project works with genomic data (VCF files, scoring files) that can be hundreds of megabytes — committing them to git will break pushes to GitHub (100 MB limit) and bloat the repository permanently.

### Input data (`data/input/`)

User-provided input files (VCF uploads, custom scoring files, etc.) go to `data/input/`. This directory is gitignored. The Reflex UI must write uploaded files under `PRS_UI_DATA_DIR/input/` (default: `./data/input/` from the directory where `uv run ui` / `uvx` was invoked), never to `prs-ui/uploaded_files/`, site-packages, or any source package directory.

| Subdirectory | Contents |
|---|---|
| `data/input/vcf/` | User-uploaded VCF files |
| `data/input/scoring/` | User-provided custom scoring files |

### Output data (`data/output/`)

All CLI commands that produce data default to writing under `data/output/`:

| Subdirectory | Contents |
|---|---|
| `data/output/pgs_metadata/` | Bulk metadata parquets (raw and cleaned) from FTP / HF |
| `data/output/pgs_scores/` | Bulk-downloaded per-score parquet files |
| `data/output/scores/` | Individual scoring file downloads |
| `data/output/results/` | PRS computation results |

### Legacy output directory (`output/`)

For backward compatibility, `output/` is also gitignored. New code should prefer `data/output/`.

### Cache directory (cross-platform via `platformdirs`)

Long-lived cached data used by `PRSCatalog` and tests goes to the OS-appropriate user cache directory, resolved by `resolve_cache_dir()` (re-exported from `just_prs`). The base path is determined by `platformdirs.user_cache_dir("just-prs")`:

| OS | Default base path |
|---|---|
| Linux | `~/.cache/just-prs/` |
| macOS | `~/Library/Caches/just-prs/` |
| Windows | `%LOCALAPPDATA%\just-prs\Cache\` |

Override with the `PRS_CACHE_DIR` environment variable (or set it in `.env`).

| Subdirectory | Contents |
|---|---|
| `<cache>/metadata/` | Cleaned parquets (auto-populated by `PRSCatalog`) |
| `<cache>/metadata/raw/` | Raw FTP parquets cached by `PRSCatalog` |
| `<cache>/scores/` | Cached scoring files for PRS computation. Contains both `.txt.gz` (original download) and `.parquet` (spec-driven parquet cache with zstd-9 compression and embedded PGS header metadata). The pipeline `scoring_files_parquet` asset deletes `.txt.gz` after verified conversion to save disk space (~5.5 GB savings for the full catalog). |
| `<cache>/normalized/` | Normalized VCF parquets (auto-populated by the web UI) |
| `<cache>/reference_scores/{panel}/{pgs_id}/` | Per-ID reference panel scores (cached by `compute_reference_prs_batch`) |
| `<cache>/percentiles/` | Distribution and quality parquets (`{panel}_distributions.parquet`, `{panel}_quality.parquet`, `{panel}_distribution_quality_issues.parquet`) |
| `<cache>/test-data/` | Test VCF files and fixtures |
| `<cache>/plink2/` | Auto-downloaded PLINK2 binary |

### Rules

- **NEVER commit large data files.** VCF (`.vcf`, `.vcf.gz`), parquet (`.parquet`), gzipped data (`.gz`, `.bgz`), FASTA (`.fa`, `.fasta`), and BAM/CRAM files must NEVER be added to git. GitHub rejects files > 100 MB and large files in history are extremely difficult to remove.
- **CLI defaults** must always point to `data/output/<subdir>` (or `./output/<subdir>` for legacy), never `./` or `./pgs_metadata/` etc.
- **Library code** (`PRSCatalog`, `scoring.py`) must use `resolve_cache_dir()` from `just_prs.scoring` (or accept explicit paths). Never hardcode OS-specific cache paths.
- **Tests** must use `resolve_cache_dir() / "test-data"`, never write to the project tree.
- **UI uploaded files** must go to `PRS_UI_DATA_DIR/input/` (default invocation-directory `data/input/`), never inside `prs-ui/`, site-packages, or any source directory.
- **Never add data directories** (parquet, CSV, VCF, gz) to git. The `.gitignore` blocks `data/`, `output/`, `pgs_metadata/`, `pgs_scores/`, `scores/`, and `**/uploaded_files/`.

---

## uv Project Management

- **Dependency Management**: Use `uv sync` and `uv add`. NEVER use `uv pip install`.
- **Python execution**: In this uv workspace, always run Python through uv: use `uv run python ...` for one-off scripts, `uv run python -m pytest ...` for tests, and `uv run <script>` for registered CLIs. Never call bare `python` or `python3` from the shell, because it may bypass the workspace environment or fail on systems without a `python` shim.
- **Project Configuration**: Use `project.toml` as the single source of truth for dependencies and project metadata.
- **Versioning**: Do not hardcode versions in `__init__.py`; rely on `project.toml`.

---

## PyPI Publishing

Both `just-prs` and `prs-ui` are published to PyPI. The publish token is stored in `.env` as `UV_PUBLISH_TOKEN` (and `PYPI_TOKEN` alias).

**`uv publish` does NOT load `.env` automatically** — unlike the project CLIs which use `python-dotenv`. You must extract the token explicitly:

```bash
# Build both packages
uv build --package just-prs
uv build --package prs-ui

# Publish (extract token from .env since uv publish doesn't load dotenv)
export UV_PUBLISH_TOKEN=$(grep '^UV_PUBLISH_TOKEN=' .env | sed 's/^UV_PUBLISH_TOKEN=//' | tr -d '"')
uv publish dist/just_prs-<version>-py3-none-any.whl dist/just_prs-<version>.tar.gz
uv publish dist/prs_ui-<version>-py3-none-any.whl dist/prs_ui-<version>.tar.gz
```

**Release checklist:**
1. Bump versions in `just-prs/pyproject.toml` and/or `prs-ui/pyproject.toml`
2. Run `uv lock` to update the lockfile
3. Run the full test suite: `uv run python -m pytest just-prs/tests/ -v`
4. Commit, push, build, publish
5. Create a GitHub release: `gh release create v<version> --title "..." --notes "..."`

---

## Coding Standards

- **Type Hints**: Mandatory for all Python code to ensure type safety and better IDE support.
- **Pathlib**: Always use `pathlib.Path` for all file path operations. Avoid string-based path manipulation.
- **Imports**: Always use absolute imports. Avoid relative imports (e.g., `from . import utils`).
- **Error Handling**: Avoid nested `try-catch` blocks. Only catch exceptions that are truly unavoidable or where you have a specific recovery strategy.
- **CLI Tools**: Use the `Typer` library for all command-line interface tools.
- **Data Classes**: Use `Pydantic 2` for all data validation and settings management.
- **Logging**: Use `Eliot` for structured, action-based logging and tracking.
- **No Placeholders**: Never use temporary or custom local paths (e.g., `/my/custom/path/`) in committed code.
- **Refactoring**: Prioritize clean, modern code. Refactor aggressively and do not maintain legacy API functions unless explicitly required.
- **Terminal Warnings**: Pay close attention to terminal output. Deprecation warnings are critical hints that APIs need updating.

---

## Data Engineering (Polars)

- **Prefer Polars over Pandas**: Use `Polars` for data manipulation.
- **Efficiency**: Use `LazyFrame` (`scan_parquet`) and streaming (`sink_parquet`) for memory-efficient processing of large datasets.
- **Memory Optimization**: Pre-filter dataframes before performing joins to avoid unnecessary materialization.
- **polars-bio**: When working with DataFusion-backed LazyFrames (e.g. from `scan_vcf`), collect before aggregating multiple columns to avoid record batch length mismatches.

---

## Test Generation Guidelines

- **Real Data + Ground Truth**: Use actual source data (auto-download if necessary) and compute expected values at runtime rather than hardcoding them.
- **Deterministic Coverage**: Use fixed seeds for sampling and explicit filters to ensure tests are reproducible.
- **Meaningful Assertions**: 
    - Prefer relationship and aggregate checks (e.g., set equality, sums, means) over simple existence checks.
    - **Good**: `assert set(source_ids) == set(output_ids)`
    - **Bad**: `assert len(df) > 0`
- **No Mocking**: Do not mock data transformations. Run real pipelines to ensure integration integrity.
- **Verification**: Before claiming a test catches a bug, demonstrate the failure by running the buggy code against the test.

### Anti-Patterns to Avoid
- Testing only the "happy path" with trivial data.
- Hardcoding expected values that drift from the source data.
- Ignoring edge cases (nulls, empty strings, boundary values, malformed data).
- Redundant tests (e.g., checking `len()` if you are already checking set equality).


### Dagster 1.12.x+ API Notes & Gotchas

Many older Dagster tutorials use deprecated APIs. Keep these rules in mind for modern Dagster versions:

**Context Access:** `get_dagster_context()` does NOT exist. You must pass `context: AssetExecutionContext` explicitly to your functions.

**Metadata Logging:** `context.log.info()` does NOT accept a `metadata` keyword argument. Use `context.add_output_metadata()` separately.

**Run Logs:** `EventRecordsFilter` does NOT have a `run_ids` parameter. Instead, use `instance.all_logs(run_id, of_type=...)`.

**Asset Materializations:** Use `EventLogEntry.asset_materialization` (which returns `Optional[AssetMaterialization]`), not `DagsterEvent.asset_materialization`.

**Job Hooks:** The `hooks` parameter in `define_asset_job` must be a set, not a list (e.g., `hooks={my_hook}`).

**Asset Resolution:** Use `defs.resolve_all_asset_specs()` instead of the deprecated `defs.get_all_asset_specs()`.

**Asset Job Config:** Asset job config uses the `"ops"` key, not `"assets"`. Using `"assets"` causes a `DagsterInvalidConfigError`.

**CLI deprecation:** `dagster dev` is superseded by `dg dev`. The old command still works but emits a `SupersessionWarning`.

**Deprecation policy (CRITICAL):** treat all deprecation warnings as blockers for new changes in touched code. Investigate current upstream docs/APIs, update the implementation to non-deprecated APIs, and update `AGENTS.md` rules/examples so future changes do not reintroduce deprecated patterns.

**Definitions jobs deprecation warning fix:** do not pass unresolved asset jobs directly in `Definitions(jobs=[...])` when warning suggests resolution; resolve jobs by creating a temporary `Definitions` inside a function (local variable) to get the asset graph, then build the final `Definitions` with the resolved jobs. **CRITICAL: Dagster 1.12+ rejects multiple `Definitions` objects at module scope** — never assign a temporary `Definitions` to a module-level variable (even prefixed with `_`).

**Automation (CRITICAL — DO NOT USE `AutomationCondition`):** `AutomationCondition.on_missing()` and `.eager()` are **broken** for triggering initial materializations in Dagster 1.12. `on_missing()` on root assets silently produces 0 runs on every tick due to `InitialEvaluationCondition` canceling `SinceCondition`. `eager()` on root assets never fires (no upstream updates). `AutomationConditionSensorDefinition` starts `STOPPED` by default, and even when forced to `RUNNING`, the underlying conditions still produce 0 runs. The `dagster.yaml` `auto_materialize: enabled: true` is the legacy daemon and has no effect on the sensor system. **For startup, use a run-once bootstrap sensor** (`@dg.sensor` with `default_status=RUNNING`) that checks `instance.get_latest_materialization_event()` and submits a `RunRequest` with a `run_key` for deduplication. **For ongoing correctness, add a separate recompute sensor that triggers when upstream assets are newer than downstream outputs.** See the "Dagster Single-Command Startup Pattern" section below.

### Resource Tracking (MANDATORY)

**Always track CPU and RAM consumption** for all compute-heavy assets using `resource_tracker` from `prs_pipeline.runtime`:

```python
from prs_pipeline.runtime import resource_tracker

@asset
def my_asset(context: AssetExecutionContext) -> Output[Path]:
    with resource_tracker("my_asset", context=context):
        # ... compute-heavy code ...
        pass
```

**Important:** Always pass `context=context` to enable Dagster UI metadata. Without it, metrics only go to the Dagster logger.
This automatically logs to Dagster UI: `duration_sec`, `cpu_percent`, `peak_memory_mb`, `memory_delta_mb`.

### Run-Level Resource Summaries (MANDATORY)

All jobs must include the `resource_summary_hook` from `prs_pipeline.utils` to provide aggregated resource metrics at the run level:

```python
from prs_pipeline.utils import resource_summary_hook

my_job = define_asset_job(
    name="my_job",
    selection=AssetSelection.assets(...),
    hooks={resource_summary_hook},  # Note: must be a set, not a list
)
```

This hook logs a summary at the end of each successful run: Total Duration, Max Peak Memory, and Top memory consumers.

### Key files for resource tracking

| File | What it does |
|------|-------------|
| `prs_pipeline/runtime.py` | `ResourceReport` model, `resource_tracker` context manager (uses `psutil`) |
| `prs_pipeline/utils.py` | `resource_summary_hook` — aggregates per-asset metrics into a run-level summary |

### Asset checks (`prs_pipeline/checks.py`)

| Check | Asset | Severity | What it validates |
|-------|-------|----------|-------------------|
| `check_distributions_superpop_completeness` | `reference_scores` | ERROR | Every PGS ID has exactly 5 superpopulation rows |
| `check_distributions_no_inf_nan` | `reference_scores` | ERROR/WARN | No inf, NaN, or zero-std in distributions; writes `{panel}_distribution_quality_issues.parquet` for full manual triage |
| `check_distributions_quantile_ordering` | `reference_scores` | ERROR | p5 ≤ p25 ≤ median ≤ p75 ≤ p95 for all rows |
| `check_distributions_vs_raw_scores` | `reference_scores` | ERROR | Spot-checks 20 PGS IDs: distributions match re-aggregation from raw scores (catches stale data) |
| `check_distributions_sample_sizes` | `reference_scores` | ERROR | Sample sizes within 1000G panel range (400–1000) |
| `check_enriched_has_metadata_columns` | `hf_prs_percentiles` | ERROR/WARN | Enriched distributions have all required stats + metadata columns |
| `check_cleaned_metadata_quality` | `cleaned_pgs_metadata` | ERROR | Non-empty scores, normalized genome builds, best_performance exists |
| `check_chip_coverage_valid` | `chip_coverage` | ERROR | Coverage ratios in [0,1], `n_typed` ≤ `n_total`, every chip covers the same PGS-ID set, table non-empty |

Checks are included in job selections via `AssetSelection.checks_for_assets()` so they run automatically after their target asset materializes. Jobs that include checks: `full_pipeline`, `score_and_push`, `catalog_pipeline`, `chip_coverage_pipeline`, `metadata_pipeline`.

### Consumer-chip coverage pipeline assets

A separate lineage answers "which PRS are usable on raw consumer-array data (no imputation)?". The `chip_coverage_pipeline` job is lightweight — it reuses cached scoring parquets and never touches the reference panel.

| Asset | Group | What it does |
|-------|-------|-------------|
| `illumina_gsa_manifest` | `external` (SourceAsset) | Illumina GSA v3 A2 (GRCh38) manifest, ~70 MB zip / ~648K markers; URL in metadata |
| `chip_coverage` | `compute` | `compute_chip_coverage()` over all cached GRCh38 scoring parquets → `percentiles/chip_coverage.parquet` (one row per `pgs_id × chip`). Result: only ~2.4% of PGS are array-ready (≥80% direct coverage) on GSA; ~93.5% need imputation (median ~15%) — this is the data backing the UI's array-ready vs imputation-required labels |
| `hf_chip_coverage` | `upload` | Pushes `chip_coverage.parquet` to `just-dna-seq/prs-percentiles` (`data/chip_coverage.parquet`) via `push_chip_coverage()` |

### Prevalence and Absolute Risk Pipeline Assets

The metadata pipeline includes two additional assets for disease prevalence and absolute risk:

| Asset | Group | What it does |
|-------|-------|-------------|
| `gwas_studies` | `download` | Downloads GWAS Catalog bulk studies TSV + trait mappings, parses case/control counts from free-text sample descriptions via regex, produces `gwas_studies.parquet` |
| `trait_prevalence` | `compute` | Merges 3 tiers of prevalence data (seed CSV → GWAS cohort fractions → PGS eval cohorts), builds EFO cross-references via OLS4, produces `trait_prevalence.parquet`, synced to HF |
| `trait_heritability` | `download` | Downloads Pan-UKBB SNP heritability plus GWAS Atlas v20191115 as an archival 2019 fallback, maps source traits to EFO where possible, and must publish ontology-resolved aliases so MONDO/OBA/HP PGS traits can still find EFO-keyed h² estimates |
| `hf_pgs_catalog_risk_metadata` | `upload` | Uploads `trait_prevalence.parquet` and `trait_heritability.parquet` to `just-dna-seq/pgs-catalog` under `data/metadata/`; this is the HF source of truth for absolute-risk metadata |

The `hf_prs_percentiles` asset enriches distributions with absolute risk columns (`abs_risk_at_mean`, `abs_risk_method`, `abs_risk_prevalence`) using the `estimate_absolute_risk` facade. See [Absolute Risk Methodology](docs/absolute-risk-methodology.md) for the mathematical details.

Before publishing percentiles, `hf_prs_percentiles` must run `reference_distribution_audit_issues()` and quarantine every PGS ID with any `ERROR` issue from `{panel}_distributions.parquet`; `WARN` rows remain published but visible in the sidecar. The sidecar `{panel}_distribution_quality_issues.parquet` and compact `{panel}_distribution_audit_summary.json` are uploaded for audit/debugging. `PRSCatalog.reference_distributions()` also defensively filters untrustworthy PGS IDs on read, so stale local/HF parquets cannot expose bad percentiles to users.

Reference percentile audits must be quality-aware, not only numeric. Use `reference_distribution_audit_issues(distributions_df, quality_df)` before publishing or trusting cached/HF percentiles. It includes the distribution-shape checks plus per-PGS quality metadata checks for missing match counts, low reference-panel match rate, non-OK scoring status, sample-count mismatch, and stale mean/std aggregates. A finite, non-degenerate distribution is still suspicious if `{panel}_quality.parquet` has null `variants_total`, `variants_matched`, or `match_rate`, because the UI cannot tell whether the reference population had the same low-coverage problem as the user.

Reference percentile auditing has a first-class Dagster asset `reference_percentile_audit` and job `reference_percentile_audit_job`, launched with `uv run pipeline audit` (Dagster UI by default, `--headless` optional). The job name must not equal the asset/op name because Dagster requires unique op/graph definition names in a repository. It audits cached or HuggingFace-pulled percentile parquets and writes `{panel}_distribution_quality_issues.parquet` plus `{panel}_distribution_audit_summary.json` without recomputing reference scores. The audit job must log a clear pass/warn/fail summary and upload audit sidecars to `just-dna-seq/prs-percentiles` when `HF_TOKEN` is available; if no token is available, it should warn and keep local sidecars. The PRS UI exposes a `Refresh reference/audit cache` checkbox that force-pulls the latest percentile/audit sidecars before computing selected PRS results, so stale in-process annotations can be refreshed without restarting the app. Keep this job registered in `Definitions`, included in pipeline docs, and available as a CLI entrypoint whenever audit behavior changes.

**Ontology-resolved risk metadata is mandatory.** PGS Catalog `trait_efo_id` values are not always EFO IDs; many shipped scores use MONDO, OBA, HP, or other ontology prefixes. Risk metadata must therefore be resolved at the data layer, not patched only in the UI. The pipeline should build/persist ontology aliases for prevalence and heritability (EFO ⇄ MONDO/OBA/HP where OLS4 or source mappings support it), publish the enriched parquets via `hf_pgs_catalog_risk_metadata`, and log alias coverage in Dagster metadata. `PRSCatalog` should exact-match cached IDs first, then use the same ontology resolver as a fallback for old caches or newly observed IDs. The UI must explicitly show when an h²-liability estimate was used and when all heritability mapping failed; silent omission of h² methods is not acceptable.

**Explain h² and ontology mappings for citizen scientists.** UI and public docs must not assume users know what `h²`, EFO, MONDO, OBA, or HP mean. When showing h²-liability, explain that h² is population-level heritability: the fraction of trait variation statistically associated with genetic differences in a studied population, not an individual causal percentage. Explain that EFO/MONDO/OBA/HP are different biomedical vocabularies and ontology mapping lets the app recognize that IDs from different vocabularies may refer to the same or closely related trait. Prefer plain text like `No mapped heritability estimate is available for this trait` over internal-only phrases.

**GWAS Atlas is archival only.** `atlas.ctglab.nl` is still reachable but effectively frozen at Release 3 (`v20191115`, last curated August 2019). Keep it as a reproducible secondary fallback for LDSC SNP-heritability coverage, but do not describe it as current. Pan-UKBB remains the primary h² source; GWAS Atlas rows should be labeled as archival 2019 data and should not receive `high` confidence.

After changing risk metadata resolution, run `uv run pipeline catalog --no-cache` so Dagster UI opens for monitoring while old local risk assets are rewritten and enriched `trait_prevalence.parquet` / `trait_heritability.parquet` are uploaded to HuggingFace. Use `uv run pipeline catalog --headless --no-cache` only when explicitly running in a non-interactive/scripted context.

### Best Practices for Assets & IO

**Declarative Assets:** Prioritize Software-Defined Assets (SDA) over imperative ops. Include all assets in `Definitions(assets=[...])` for complete lineage visibility in the UI.

**Polars Integration:** Use `dagster-polars` with `PolarsParquetIOManager` for `pl.LazyFrame` assets to automatically get schema and row counts in the Dagster UI.

**Large Data / Streaming:** Use `lazy_frame.sink_parquet()` and NEVER `.collect().write_parquet()` on large data to avoid out-of-memory errors.

**Path Assets:** When returning a `Path` from an asset, add `"dagster/column_schema": polars_schema_to_table_schema(path)` to ensure schema visibility in the UI.

**Asset Checks:** Use `@asset_check` for validation and include them in your job via `AssetSelection.checks_for_assets(...)`.

### Execution & Concurrency Patterns

**Concurrency Limits:** Use `op_tags={"dagster/concurrency_key": "name"}` to limit parallel execution for resource-intensive assets.

**Timestamps:** Timestamps are on `RunRecord`, not `DagsterRun`. `run.start_time` will raise an `AttributeError`. Retrieve `instance.get_run_records()` and use `record.start_time`/`record.end_time` (Unix floats) or `record.create_timestamp` (datetime).

**Partition Keys for Runs:** `create_run_for_job` doesn't accept a direct `partition_key` parameter. Pass it via tags instead: `tags={"dagster/partition": partition_key}`.

**Dynamic Partitions Pattern:**
- Create partition def: `PARTS = DynamicPartitionsDefinition(name="files")`
- Discovery asset registers partitions: `context.instance.add_dynamic_partitions(PARTS.name, keys)`
- Partitioned assets use: `partitions_def=PARTS` and access `context.partition_key`
- Collector depends on partitioned output via `deps=[partitioned_asset]` and scans the filesystem/storage for results.

### Web UI / Asynchronous Execution Pattern

If you are running Dagster alongside a Web UI (like Reflex, FastAPI, etc.), use the Try-Daemon-With-Fallback pattern:

**Submission vs Execution:**
Attempt to submit the run to the daemon first: `instance.submit_run(run_id, workspace=None)`. If this fails (e.g., due to missing `ExternalPipelineOrigin` in web contexts), fall back to `job.execute_in_process()`.

**Rust/PyO3 Thread Safety:**
NEVER use `asyncio.to_thread()` or `asyncio.create_task()` with Dagster objects (it causes PyO3 panics: "Cannot drop pointer into Python heap without the thread being attached"). Use `loop.run_in_executor(None, sync_execution_function, ...)` for thread-safe background execution that doesn't block your UI.

**Orphaned Run Cleanup:**
If you use `execute_in_process` inside a web server process, runs will be abandoned (stuck in `STARTED` status) if the server restarts. Add startup cleanup logic targeting `DagsterRunStatus.NOT_STARTED`. Use `atexit` or signal handlers (`SIGTERM`/`SIGINT`) to mark active in-process runs as `CANCELED` on graceful server shutdown.

### Common Anti-Patterns to Avoid

- Using `dagster job execute` CLI: This is deprecated.
- Hardcoding asset names: Resolve them dynamically using defs.
- Suspended jobs holding locks: If a job crashes while querying local DBs (like DuckDB/SQLite), it can hold file locks. Handle connections properly via context managers or resources.
- Processing failures in exception handlers: Keep business logic out of exception handlers when executing runs. Catch the exception, register the failure, and cleanly proceed to your fallback mechanism.
- **Compute-heavy assets without `resource_tracker`** — if a process gets OOM-killed, there are no metrics to diagnose it. Always wrap with `resource_tracker(name, context=context)`.
- **Jobs without `resource_summary_hook`** — without it, run-level resource consumption is invisible. Always pass `hooks={resource_summary_hook}` to `define_asset_job`.
- **NEVER use `AutomationCondition` for hands-free pipeline launch.** `on_missing()` silently produces 0 runs on root assets, `eager()` never fires on root assets, and `AutomationConditionSensorDefinition` doesn't fix either issue. Use a run-once `@dg.sensor` instead.
- **NEVER use `auto_materialize: enabled: true` in `dagster.yaml`.** This is the legacy daemon and has no effect on `AutomationCondition` or sensors.
- **NEVER use `AutomationConditionSensorDefinition`** as a fix for the above — the underlying `AutomationCondition` logic is what's broken, not the sensor wrapper.
- **Default to caching; force re-materialization only via explicit `--no-cache`.** No `FORCE_RUN_ON_STARTUP` env vars, no timestamp-based run keys to bypass deduplication. If cached data exists, use it. The `--no-cache` flag (which sets `PRS_PIPELINE_NO_CACHE=1`) is the only way to bypass caches, and it must default to `False`.
- **NEVER use `os.execvp` for headless pipeline execution.** Use `job.execute_in_process()` for `--headless` runs. `os.execvp` is only for the default UI mode (used by all commands without `--headless`).

For a detailed overview of the pipelines in this project, see [Dagster Pipelines Documentation](docs/DAGSTER.md).

---

## Dagster Pipeline Robustness Policy (CRITICAL)

Standard bioinformatics pipeline engines (Nextflow, WDL/Cromwell, Snakemake) provide out-of-the-box guarantees: timestamp-based cache invalidation, automatic failure retry, interrupted-run resume, and completeness validation. Dagster does NOT provide these by default — sensors only check metadata events (a database flag), not actual data state. Every Dagster asset, sensor, and job in this project MUST implement the 8 guarantees below to match or exceed these standards.

### The 8 Robustness Guarantees

**1. Input-change invalidation (mtime-based).**
If an upstream input file changes (new scoring file downloaded, reference panel updated, metadata refreshed), downstream cached results that depended on the old input MUST be recomputed. Detection mechanism: compare input file mtime against output file mtime. If the input is newer than the output, the cache is stale — recompute. This matches `make`/Nextflow/Snakemake timestamp-based rebuild. In `skip_existing` checks, file existence alone is NOT sufficient — always verify that the output is at least as recent as its input.

**2. Interrupted-run recovery (gap detection).**
If a process crashes, is OOM-killed, or is interrupted mid-batch, the next run MUST detect incomplete work and resume from where it left off. Detection mechanism: compare the set of expected outputs (e.g. all PGS IDs in the EBI catalog) against the set of outputs that actually exist on disk. Any gap triggers recomputation of the missing items only — not a full re-run. The `completeness_sensor` implements this by scanning `reference_scores/{panel}/*/scores.parquet` against `list_all_pgs_ids()`.

**3. Automatic failure retry.**
If individual items in a batch fail (e.g. a PGS ID scoring throws an exception), those failures MUST be automatically retried on subsequent runs. Detection mechanism: read the quality/status report parquet from the previous run, extract items with `status == "failed"`, and resubmit them. After N consecutive retries with the same failure set (configurable via `PRS_PIPELINE_MAX_FAILURE_RETRIES`, default 3), stop retrying and log a permanent-failure warning — do not retry forever. The `failure_retry_sensor` implements this.

**4. Completeness validation before publishing.**
Before uploading results to an external destination (HuggingFace, S3, etc.), the asset MUST validate that the output has acceptable coverage. Detection: compare the number of unique entities in the output against the total expected (e.g. `distributions["pgs_id"].n_unique()` vs `len(list_all_pgs_ids())`). Log `coverage_ratio`, `n_scored`, `n_catalog_total`, `n_failed`, `n_missing` in output metadata. Warn (but still upload) if coverage is below a configurable threshold (`PRS_PIPELINE_MIN_COVERAGE`, default 0.90). NEVER silently upload a near-empty dataset.

**5. Content-aware cache validity (skip_existing).**
`skip_existing` checks MUST NOT only test for file existence. They must also verify that the cached output is not stale relative to its inputs. Minimum check: compare output file mtime against input file mtime. If input is newer, the cache is invalid and the item must be recomputed. This is the Nextflow `-resume` / Snakemake / Make default behavior.

**6. Rich batch metadata.**
Every asset that processes a batch of items MUST emit in its Dagster output metadata: `n_total` (items attempted), `n_ok` (succeeded), `n_failed` (failed), `n_cached` (reused from cache), `coverage_ratio` (`n_ok / n_total`). This makes the Dagster UI a self-documenting data catalog where you can see at a glance whether a run was complete or partial.

**7. Sensor intelligence hierarchy.**
Sensors MUST check data state in this priority order:
  1. Is a run already active for this job? Skip — never double-submit.
  2. Was this an explicit user command (`PRS_PIPELINE_FORCE_RUN=1`)? Submit unconditionally.
  3. Are there failed items from the last run that should be retried? Submit targeted retry.
  4. Are there missing items (catalog gap vs on-disk scores)? Submit full job with `skip_existing`.
  5. Have upstream inputs changed (fingerprint/mtime changed)? Submit full job.
  6. Are all items present and up-to-date? Skip — everything is healthy.

**8. Corrupted file detection and recovery.**
A file that exists on disk but cannot be parsed (truncated writes, OOM-killed mid-flush, invalid thrift headers, etc.) is **worse than a missing file** — existence checks pass, recomputation is skipped, but reading the file later raises an exception that crashes the entire run. Every place that reads a cached parquet MUST catch parse errors (`polars.exceptions.ComputeError` and similar) and treat the file as missing:
- In `skip_existing` loops (`compute_reference_prs_batch`): catch `_CorruptParquet` from `_aggregate_single_pgs`, delete the corrupt file, and fall through to recompute.
- In the `completeness_sensor`: validate each `scores.parquet` with `scan_parquet` + `collect_schema()` before counting it as "scored". Delete any corrupt file and count it as missing so the gap triggers recomputation.
- Any other cache reader that wraps `scan_parquet` or `read_parquet` in a `skip_existing` guard MUST apply the same try/delete/recompute pattern.
The `_CorruptParquet` sentinel exception in `just_prs.reference` distinguishes "bad file" from legitimate scoring failures so the two can be handled differently (delete+retry vs permanent failure after N retries).

### Robustness Anti-Patterns (NEVER)

- **NEVER** use Dagster materialization events as the sole indicator of completeness. Materialization means "the asset function returned" — it says nothing about how many items succeeded, failed, or were skipped.
- **NEVER** upload to an external destination without logging coverage metrics. Silent uploads of near-empty datasets are worse than no upload.
- **NEVER** skip a cached result without checking that its input is older than the output. Existence-only checks lead to stale data that is never refreshed.
- **NEVER** ignore failures from a previous run. If a quality report exists with `status == "failed"` entries, the next run must attempt to resolve them.
- **NEVER** rely on a single sensor for all orchestration concerns. Separate sensors for: startup/force-run, failure-retry, completeness-gap, and upstream-freshness. Each has different check intervals and submission logic.
- **NEVER** treat file existence as proof of a valid cache. A truncated write or OOM-killed process leaves a corrupt parquet that passes the existence check but crashes the run when read. Always validate readability (`scan_parquet` + `collect_schema()`) before skipping recomputation. Delete and recompute any file that fails to parse.

### Sensor Architecture

The pipeline uses 4 sensors with distinct responsibilities:

| Sensor | Interval | Purpose |
|--------|----------|---------|
| `startup_sensor` | 30s | Handles `PRS_PIPELINE_FORCE_RUN` and initial materialization check |
| `completeness_sensor` | 5min | Compares on-disk scored IDs vs EBI catalog; submits when gap exists |
| `failure_retry_sensor` | 15min | Reads quality parquet, retries failed IDs up to N times |
| `upstream_freshness_sensor` | 6h | Compares live HTTP fingerprint vs stored; submits when upstream changes |

### Quick Checklist — Before Finishing Any Dagster Asset

- [ ] `skip_existing` checks compare mtimes, not just existence.
- [ ] `skip_existing` cache readers validate parquet readability (`collect_schema()`), delete corrupt files, and fall through to recompute.
- [ ] Output metadata includes `n_total`, `n_ok`, `n_failed`, `n_cached`, `coverage_ratio`.
- [ ] Upload assets validate coverage and log `n_scored`, `n_catalog_total`, `n_missing`.
- [ ] Failures are recorded in a quality parquet that sensors can read for retry.
- [ ] The asset handles partial prior results (resumes from where it left off).

---

## Dagster Asset Lineage Rules (CRITICAL)

Incomplete lineage is a recurring mistake. Every time you write a Dagster asset, apply ALL of the rules below before finishing.

### 1. SourceAssets for every external data source

Any data that Dagster **downloads but does not produce** — an FTP tarball, a remote API, a HuggingFace dataset, an S3 bucket — must be declared as a `SourceAsset`. This gives it a visible node in the lineage graph.

```python
from dagster import SourceAsset

ebi_reference_panel = SourceAsset(
    key="ebi_reference_panel",
    group_name="external",
    description="1000G reference panel tarball at EBI FTP (~7 GB).",
    metadata={"url": REFERENCE_PANEL_URL},
)
```

Register every `SourceAsset` in `Definitions(assets=[...])`. Without this, the left edge of the graph is a dangling node with no visible origin.

### 1b. NEVER put SourceAsset keys in computed asset `deps`

SourceAssets are visualization-only nodes — Dagster never materializes them. If a computed asset lists a SourceAsset in `deps`, Dagster will see the SourceAsset as permanently missing, which can block job execution and sensor-based automation.

```python
# WRONG — ebi_reference_panel is a SourceAsset → deps check will fail
@asset(deps=["ebi_reference_panel"])
def reference_panel(...): ...

# CORRECT — no SourceAsset in deps; document the source URL in output metadata instead
@asset()
def reference_panel(context, ...):
    ...
    context.add_output_metadata({"source_url": REFERENCE_PANEL_URL})
```

SourceAssets remain in `Definitions(assets=[...])` as standalone lineage nodes. The computed-to-computed dep chain (`a → b → c`) is all that automation needs. SourceAssets become "orphan" visualization nodes in the UI — that is acceptable since each computed asset documents its download URL in output metadata.

### 2. Always declare `deps` when an asset reads filesystem side effects of another asset

If an asset scans a directory that was populated by another asset (common pattern: partitioned writer → non-partitioned aggregator), it MUST declare a `deps` relationship. Without `deps`, Dagster draws NO edge between them even though there is a real data dependency.

```python
from dagster import AssetDep, asset

@asset(
    deps=[AssetDep("per_pgs_scores")],   # <-- REQUIRED for lineage even if no data is loaded via AssetIn
    ...
)
def reference_distributions(...):
    # scans cache_dir/reference_scores/**/*.parquet populated by per_pgs_scores partitions
    ...
```

Rule: if you write "scan for parquets produced by X" anywhere in a docstring or comment, you MUST add `deps=[AssetDep("X")]` to the decorator.

### 3. Use `AssetIn` only when data is passed through the IOManager

`AssetIn` means "load the output of asset X via the IOManager and inject it as a function argument". Use it when:
- The upstream returns a `pl.DataFrame` / `pl.LazyFrame` and you use `PolarsParquetIOManager`
- The upstream returns a picklable Python object and the default IOManager is acceptable

Do NOT use `AssetIn` with `Path` or `Output[Path]` unless you have a custom `UPathIOManager`. Passing a `Path` via the default IOManager only works within the same Python process (in-memory pickle) and will silently break across runs or when using a persistent IOManager.

For `Path`-based data flow, use `deps` for lineage and reconstruct the path inside the downstream asset from a shared resource (e.g. `CacheDirResource`).

### 4. Separate assets into named groups by pipeline stage

Every `@asset` and `SourceAsset` must have a `group_name`. Use a consistent stage-based taxonomy so the Dagster UI shows a clear left-to-right graph:

| group_name | What belongs here |
|---|---|
| `external` | `SourceAsset`s for remote data (FTP, HF, S3, API) |
| `download` | Assets that fetch external data into local cache |
| `compute` | Transformation / scoring / aggregation assets |
| `upload` | Assets that push results to external destinations (HF, S3, DB) |

The final upload asset (e.g. `hf_prs_percentiles`) IS the representation of the remote dataset in the lineage — name it after the destination, not after the action.

### 5. SourceAssets must be registered in `Definitions`

```python
import dagster as dg

full_pipeline = dg.define_asset_job(
    name="full_pipeline",
    selection=["reference_panel", "reference_scores", "hf_prs_percentiles"],
)

defs = dg.Definitions(
    assets=[
        ebi_pgs_catalog_reference_panel,   # SourceAsset — must be listed
        ebi_pgs_catalog_scoring_files,     # SourceAsset — must be listed
        reference_panel,
        reference_scores,
        hf_prs_percentiles,
    ],
    sensors=[run_pipeline_on_startup],     # run-once sensor for auto-launch
    jobs=[full_pipeline],
    ...
)
```

Omitting a `SourceAsset` from `Definitions` makes it invisible in the UI even if assets declare `deps` on it. Omitting the run-once sensor means the pipeline won't auto-start on `dagster dev` launch.

### Quick checklist before finishing any Dagster asset file

- [ ] Every remote data origin has a `SourceAsset` with `group_name="external"` and a `metadata={"url": ...}`.
- [ ] Every `SourceAsset` is listed in `Definitions(assets=[...])`.
- [ ] Every asset that scans a directory written by another asset has `deps=[AssetDep("that_asset")]`.
- [ ] No `AssetIn` is used with `Output[Path]` unless a `UPathIOManager` is configured.
- [ ] Every `@asset` has `group_name` set (never omit it).
- [ ] The final destination asset (HF upload, S3 push, DB write) is named after the destination, not the action.
- [ ] All 4 smart sensors (`startup_sensor`, `completeness_sensor`, `failure_retry_sensor`, `upstream_freshness_sensor`) are registered in `Definitions(sensors=[...])` via `make_all_sensors()`.
- [ ] The sensor uses `run_key` to prevent duplicate submissions across ticks (fresh key only on retry after failure).
- [ ] Every asset checks on-disk cache and short-circuits if data already exists — no redundant downloads or computations.
- [ ] Every compute-heavy asset is wrapped with `resource_tracker(name, context=context)`.
- [ ] Every job has `hooks={resource_summary_hook}` for run-level resource aggregation.
- [ ] Assets that produce statistical data (distributions, aggregations) have `@asset_check`s in `checks.py` validating invariants.
- [ ] Jobs that include checked assets use `AssetSelection.checks_for_assets(...)` in their selection.
- [ ] All checks are collected in `ALL_ASSET_CHECKS` and registered in `Definitions(asset_checks=...)`.
- [ ] The `check_distributions_vs_raw_scores` spot-check guards against stale distributions after scoring engine changes.

### 6. Universal Dagster Principles (Mindset & Architecture)

- **Assets over Tasks (Declarative Mindset)**: Focus on *what data should exist*, not *how to run a function*. Dependencies are expressed as data assets, not task wiring (e.g., `asset_b` depends on `asset_a`, rather than `task_a >> task_b`). Lineage is automatic based on these data dependencies.
- **Dynamic Partitions for Entities**: When processing many independent entities (like Users/Samples in a web app), use Dynamic Partitions for targeted materialization, backfills, and deletions. However, when each entity processes in seconds and shares expensive setup (as with PGS ID scoring via the polars engine), a batch approach in a single asset with in-process iteration and error tracking (`compute_reference_prs_batch`) is more efficient than thousands of partitions + sensor orchestration.
- **Assets vs. Jobs Separation**: Use Software-Defined Assets (`@asset`) for the declarative data graph and lineage. Use Jobs (`define_asset_job` / `ops`) strictly as operational entry points (for sensors, schedules, CLI, or UI triggers).
- **Abstracted Storage (IO Managers & Resources)**: Avoid hardcoding paths inside asset logic. Either return a value and let an IO Manager write it, or reconstruct the path from a shared resource (like `CacheDirResource`). This prevents path conflicts and separates business logic from storage concerns.
- **Rich Metadata**: Always add meaningful output metadata (`context.add_output_metadata(...)`), such as row counts, file sizes, schema details, or external URLs. This turns Dagster into an inspectable data catalog rather than just a task runner.
- **Freshness Over Presence**: "Asset exists" is not sufficient. Sensors/schedules must compare upstream vs downstream materialization recency and trigger recompute when lineage indicates stale outputs.

---

## Dagster Pipeline Execution Model (CRITICAL)

### Core principle: never force re-materialization

Every pipeline command respects caches. If data is already on disk, it is reused. No CLI command or sensor should ever force a full re-run when cached data exists. The only triggers for re-computation are: (1) assets have never been materialized, or (2) an upstream asset has been freshly materialized (making downstream stale).

### CLI commands

All pipeline commands launch the Dagster UI by default. Headless mode requires explicit `--headless`.

| Command | What it does | When it re-materializes |
|---------|-------------|------------------------|
| `pipeline run` | Launches Dagster UI; startup sensor submits `full_pipeline` if assets are missing. | Each asset checks disk cache and short-circuits if data exists. |
| `pipeline run --headless` | Executes `full_pipeline` in-process (no UI). | Same cache-respecting behavior, but no UI monitoring. |
| `pipeline run --no-cache` | With or without `--headless`: bypasses all on-disk caches. | Metadata is re-downloaded, parquets are re-parsed, scores are recomputed. |
| `pipeline catalog` | Launches Dagster UI; startup sensor submits `catalog_pipeline`. | Same cache-respecting behavior. |
| `pipeline catalog --headless` | Executes `catalog_pipeline` in-process (no UI). | Same cache-respecting behavior, but no UI monitoring. |
| `pipeline launch` | Launches Dagster UI (no specific job pre-selected). | The sensor submits a job only if key assets are unmaterialized. |

**Caching is the default. Re-materialization is opt-in via `--no-cache`.** No `FORCE_RUN_ON_STARTUP`, no timestamp-based run keys to bypass deduplication. If the user explicitly passes `--no-cache`, assets re-download metadata and recompute scores. Without it, all on-disk caches are respected.

### Dagster UI (all commands)

All pipeline commands use `os.execvp` to replace the current process with `dagster dev`. The startup sensor (`startup_sensor`) checks whether key assets are materialized. If any are missing, it submits the job. If all are present, it skips. It never forces re-runs. Three additional sensors (`completeness_sensor`, `failure_retry_sensor`, `upstream_freshness_sensor`) handle gap detection, failure retry, and upstream change detection per the Robustness Policy above. The Dagster UI URL is always printed prominently at startup.

### `--headless` in-process execution

When `--headless` is passed to `pipeline run` or `pipeline catalog`, the command calls `job.execute_in_process()` which materializes assets in dependency order. Each asset is responsible for checking its own disk cache:

- **Fingerprint assets** (`ebi_scoring_files_fingerprint`, `ebi_reference_panel_fingerprint`): Always run — they're lightweight HTTP HEAD requests.
- **Download assets** (`scoring_files`, `reference_panel`, `raw_pgs_metadata`): Check if files exist on disk. `scoring_files` checks both `.txt.gz` and `.parquet` per-file and skips already-cached. `download_metadata_sheet()` returns cached parquet if it exists (`overwrite=False`).
- **Compute assets** (`scoring_files_parquet`, `cleaned_pgs_metadata`, `reference_scores`): Check if output already exists. `scoring_files_parquet` skips per-file if `.parquet` cache exists. `reference_scores` uses `skip_existing=True`.
- **Upload assets** (`hf_pgs_catalog`, `hf_prs_percentiles`): Always run — uploading IS the point. They re-upload even if data hasn't changed (HF handles dedup).

### Why `os.execvp` (not `subprocess.run` or `Popen`)

Dagster's daemon uses complex internal signal handling. When trapped inside a `subprocess.run()` or `Popen()`, SIGINT/SIGTERM do not propagate correctly and the daemon does not shut down cleanly. `os.execvp` **replaces** the current Python process with `dagster`, so Dagster becomes the primary process and owns all signal handling. Ctrl+C works correctly.

### Why startup sensor (DO NOT use `AutomationCondition`)

Because `os.execvp` replaces the current process, there is no opportunity to submit a job "after dagster starts". **Do NOT use `AutomationCondition`** (`on_missing()`, `eager()`, or `AutomationConditionSensorDefinition`) for hands-free pipeline launch — they are broken for initial materialization in Dagster 1.12:

- `AutomationCondition.on_missing()` on root assets (no deps) silently produces **0 runs on every tick** — `InitialEvaluationCondition` resets `SinceCondition`, canceling the trigger permanently.
- `AutomationCondition.eager()` on root assets never fires (no upstream updates exist to trigger it).
- `AutomationConditionSensorDefinition` starts `STOPPED` by default. Even when forced to `RUNNING` with `default_status=DefaultSensorStatus.RUNNING`, the conditions above still produce 0 runs.
- `dagster.yaml` `auto_materialize: enabled: true` is the legacy daemon — it has no effect on the `AutomationCondition` sensor system.

**Reliable pattern is a run-once bootstrap sensor** that checks materialization status and submits a job only when assets are missing:

```python
import dagster as dg

@dg.sensor(
    job_name="full_pipeline",
    default_status=dg.DefaultSensorStatus.RUNNING,
    minimum_interval_seconds=60,
)
def run_pipeline_on_startup(context: dg.SensorEvaluationContext) -> dg.SensorResult | dg.SkipReason:
    check_keys = [
        dg.AssetKey("scoring_files"),
        dg.AssetKey("reference_scores"),
        dg.AssetKey("hf_prs_percentiles"),
    ]
    missing = [k for k in check_keys if context.instance.get_latest_materialization_event(k) is None]
    if not missing:
        return dg.SkipReason("All pipeline assets already materialized.")

    # Don't submit if a run is already active
    active = context.instance.get_runs(
        filters=dg.RunsFilter(job_name="full_pipeline", statuses=[
            dg.DagsterRunStatus.STARTED, dg.DagsterRunStatus.NOT_STARTED, dg.DagsterRunStatus.QUEUED,
        ])
    )
    if active:
        return dg.SkipReason(f"Already in progress (run {active[0].run_id[:8]}).")

    # Use a fresh run_key on retry after failure so dedup doesn't block it
    last_runs = context.instance.get_runs(filters=dg.RunsFilter(job_name="full_pipeline"), limit=1)
    if last_runs and last_runs[0].status == dg.DagsterRunStatus.FAILURE:
        run_key = f"pipeline_startup_retry_{int(time.time())}"
    else:
        run_key = "pipeline_startup"

    return dg.SensorResult(
        run_requests=[dg.RunRequest(run_key=run_key, job_name="full_pipeline")],
    )
```

The `run_key="pipeline_startup"` prevents duplicate submissions. The sensor only generates a fresh key after a failure, allowing retries.

### Anti-patterns to NEVER use

- **`FORCE_RUN_ON_STARTUP` env var** — bypasses cache checks, causes unnecessary re-downloads. Use `--no-cache` instead (explicit, opt-in, defaults to off).
- **Timestamp-based run keys** (e.g. `f"startup_{int(time.time())}"`) — defeats Dagster's deduplication, forces re-runs every time.
- **Implicit force-run on startup** — the sensor should only submit when assets are missing. If the user wants to force, they pass `--no-cache` to `pipeline run` or `pipeline catalog`.
- **`os.execvp` for headless execution** — use `execute_in_process()` only for `--headless` runs. `os.execvp` is for the default UI mode (all commands without `--headless`).

### `dagster.yaml` template

```yaml
telemetry:
  enabled: false
```

Generate this file at `{DAGSTER_HOME}/dagster.yaml` if it does not exist. The `telemetry: enabled: false` setting prevents `RotatingFileHandler` crashes in Dagster's event log writer. Do NOT add `auto_materialize: enabled: true` — that is the legacy daemon approach and does not work with the sensor pattern above.

### Port cleanup helper

```python
def _kill_port(port: int) -> None:
    import subprocess, signal, time, os
    result = subprocess.run(["lsof", "-t", f"-iTCP:{port}"], capture_output=True, text=True)
    pids = [int(p) for p in result.stdout.strip().splitlines() if p.strip()]
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    if pids:
        time.sleep(1)
        result2 = subprocess.run(["lsof", "-t", f"-iTCP:{port}"], capture_output=True, text=True)
        for pid in [int(p) for p in result2.stdout.strip().splitlines() if p.strip()]:
            os.kill(pid, signal.SIGKILL)
```

### Orphaned run cleanup helper

```python
def _cancel_orphaned_runs() -> None:
    from dagster import DagsterInstance, DagsterRunStatus, RunsFilter
    with DagsterInstance.get() as instance:
        stuck = instance.get_run_records(
            filters=RunsFilter(statuses=[DagsterRunStatus.STARTED, DagsterRunStatus.NOT_STARTED])
        )
        for record in stuck:
            instance.report_run_canceled(record.dagster_run, message="Orphaned run from previous session")
```

### Thread-safety note for Web UI contexts

If a Web UI (Reflex, FastAPI) needs to trigger a Dagster job in the background, **never use `asyncio.to_thread()`**. Dagster's Rust/PyO3 internals panic when the GIL is released across asyncio threads. Use `loop.run_in_executor(None, sync_func)` instead.

---

## Learned User Preferences

- Pipeline sensors and assets must be as robust as Nextflow/WDL/Snakemake: materialization-only checks are unacceptable, and data completeness must be verified from actual files and quality reports.
- Pipeline commands with a UI must launch the UI by default, with headless execution only via explicit flags; every server-starting CLI must print its URL prominently and load `.env` at startup. When suggesting commands to the user, prefer the Dagster UI form (`uv run pipeline run`, `uv run pipeline run --no-cache`, `uv run pipeline audit`) unless the user explicitly asks for headless, non-interactive, CI, or script-friendly execution.
- Policy and persistent rules should be written first, then code should comply; for major features, update `AGENTS.md`, README, and methodology/design docs without being asked.
- Treat deprecation warnings in touched code as blockers: investigate current upstream APIs and update rules/examples so deprecated patterns do not return.
- Integration tests should use real requests and real data unless the user explicitly asks for mocks or the data is multi-gigabyte; PLINK comparison tests must be opt-in and default clean-clone test runs must not download PLINK2.
- Avoid full `collect()` on large LazyFrames when lazy aggregations or `pl.collect_all()` can stream the result.
- When redundant functionality or bugs live in user-maintained upstream libraries such as `reflex-mui-datagrid`, provide a prompt/fix for that project instead of local monkey-patching.
- UI must distinguish unavailable data from zero with explicit `N/A`; use green for favorable/low-risk and red only for alarming/high-risk results.
- Long UI text in detail panels must word-wrap, badges should stay short, and foldable datagrid detail panels are preferred over separate cards; expanded PRS views should start with the percentile/reference curve and explain AUROC, variant match, risk agreement, and h² in plain language.
- Datagrid detail panels with rich content (bell curves, metric cards) must use `detail_height="auto"` and the viewport-bounded flex column layout to ensure content is fully visible while preserving internal grid scrolling; never omit `detail_height` (the JS fallback formula produces ~184px which clips 360–460px bell curves) and never put `calc(100vh)` directly on the grid when using auto detail panels.
- Use full population names in UI display, with column grouping for per-population fields; abbreviations like AFR/EUR are for data/internal columns.
- Do not hardcode memory limits, resource caps, or arbitrary column widths; use RAM percentages/env overrides and content-aware column sizing.
- For public demos such as just-dna-lite, prefer an immutable public-genome mode: no user uploads on the server, only permissively licensed public genomes, with an FAQ guiding users to install locally for private data.

## Learned Workspace Facts

- Dagster 1.12+ rejects multiple module-scope `Definitions`, `AutomationCondition` is unreliable for initial materialization, jobs must use `in_process_executor` because DuckDB/pgenlib/Arrow are not fork-safe under `dagster dev` multiprocessing, and `@asset_check` is non-blocking by default (`severity=ERROR` is only a UI label) so checks that must prevent a bad upload need `blocking=True` to fail the run and skip downstream assets.
- The pipeline processes about 5,300 EBI PGS IDs; a full reference scoring run takes hours at roughly 0.18 IDs/second with peak RSS around 3-5 GB.
- `just-dna-seq/pgs-catalog` is the HF source of truth for cleaned metadata, scoring parquet mirrors, and risk metadata, while `just-dna-seq/prs-percentiles` stores precomputed reference population distributions; catalog uploads use Hugging Face `upload_large_folder`, where pre-uploaded blobs may be committed incrementally and reruns resume from staging cache unless `--no-cache` or cache deletion forces more work.
- Scoring parquet caches save about 5.5 GB versus raw `.txt.gz`; downloads must reject zero-byte/corrupt files, and parquet cache readers must treat unreadable/truncated files as missing and recompute; `PRSCatalog.percentile()` performs a one-time HF refresh on cache miss so newly computed reference distributions are picked up without manual cache cleanup.
- `reflex-mui-datagrid` lazy grids own row-selection handling internally; customize by overriding `handle_lf_grid_row_selection`, use native `detail_columns`, badge props, `column_overrides`, and `link_list` renderers for PRS UI links/details, and do not inflate `getRowHeight` for community detail panels because MUI already renders panels after rows; bell-curve rows that should mirror the demo’s beside-chart metric stack must populate `sideItems` (and column renderer keys like `sidePanelTitle` / width hints), not only the chart `summary` text.
- The 1000G `.pvar` `ID` is `CHROM:POS:REF:ALT`; `parse_pvar()` must preserve it, `compute_reference_prs_polars(match_mode="id")` supports PLINK-parity scoring, and PLINK2 parity uses `scoresums`.
- GenoBoost includes 15 per-dosage-weight scores handled by `is_dosage_weight_format()`; five known PGS IDs are permanently unscorable with SNP-based reference panels due to HLA/no-coordinate data or upstream allele defects.
- DuckDB joins against the 75M-row pvar parquet must use explicit `duckdb.connect()`, configured memory limits, closed connections, allele-length filtering, and `SET arrow_large_buffer_size = true` before Polars conversion.
- The stored `1000g_distributions.parquet` from Mar 17 2026 was generated with PLINK2 AVG-mode and is stale versus current SUM-mode raw scores; delete that distribution parquet to re-aggregate from cached per-PGS scores.
- Absolute-risk data combines prevalence and heritability sources; `hf_pgs_catalog_risk_metadata` publishes ontology-resolved `trait_prevalence.parquet` and `trait_heritability.parquet`, `PRSCatalog` pulls those from HF when local cache is missing and falls back through the shared ontology resolver when exact IDs miss, and the UI shows all available risk methods plus explicit h² used/unavailable status by default.
- Metadata browsing must cache raw PGS sheets under `metadata/raw/` so it cannot overwrite cleaned metadata; cleaned `publications.parquet` must contain `pgp_id`, and Reflex backend exception logs must escape Rich markup to avoid masking real errors.
