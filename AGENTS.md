# Python Coding Standards & Best Practices

## Project Architecture

This project has two packages managed by a single uv workspace:

- **`src/just_prs/`** — Core library: PRS computation, PGS Catalog REST API client, FTP downloads, VCF reading, scoring file parsing. CLI entrypoint via Typer.
- **`prs-ui/`** — Reflex web app for interactive PRS computation. Has its own `pyproject.toml` and depends on `just_prs`. Run from inside `prs-ui/` with `uv run reflex run`.

### Key modules

| Module | Purpose |
|--------|---------|
| `just_prs.prs_catalog` | **`PRSCatalog`** — high-level class for search, PRS computation, and percentile estimation using cleaned bulk metadata (no REST API calls). Persists cleaned parquets locally with HuggingFace sync. |
| `just_prs.cleanup` | Pure-function pipeline: genome build normalization, column renaming, metric string parsing, performance metric cleanup |
| `just_prs.hf` | HuggingFace Hub integration: `push_cleaned_parquets()` / `pull_cleaned_parquets()` for syncing cleaned metadata parquets to/from `just-dna-seq/polygenic_risk_scores` |
| `just_prs.normalize` | VCF normalization: `normalize_vcf()` reads VCF with polars-bio, strips chr prefix, renames id→rsid, computes genotype List[Str], applies configurable quality filters (FILTER, DP, QUAL), warns on chrY for females, sinks to zstd Parquet. `VcfFilterConfig` (Pydantic v2) holds filter settings. |
| `just_prs.prs` | `compute_prs()` / `compute_prs_batch()` — core PRS engine. `compute_prs()` accepts optional `genotypes_lf` LazyFrame to skip VCF re-reading when a normalized parquet is available. |
| `just_prs.vcf` | VCF reading via `polars-bio`, genome build detection, dosage computation |
| `just_prs.scoring` | Download and parse PGS scoring files (gzipped TSV with `#` header) |
| `just_prs.ftp` | Bulk FTP/HTTPS downloads of raw metadata sheets and scoring files via `fsspec` |
| `just_prs.catalog` | Synchronous REST API client (`PGSCatalogClient`) for PGS Catalog — used for individual lookups, not for bulk metadata |
| `just_prs.models` | Pydantic v2 models (`ScoreInfo`, `PRSResult`, `PerformanceInfo`, etc.) |
| `just_prs.quality` | Pure-logic quality assessment helpers: `classify_model_quality()`, `interpret_prs_result()`, `format_effect_size()`, `format_classification()`. No Reflex dependency -- shared between core library and UI. |
| `prs_ui.state` | Reflex `AppState` + grid states + `PRSComputeStateMixin(rx.State, mixin=True)`. The mixin encapsulates all PRS computation logic (score loading, selection, batch compute, CSV export) and is designed for reuse in any Reflex app. `ComputeGridState` subclasses it with VCF-upload-specific behavior for the standalone app. |
| `prs_ui.components` | **Reusable UI components**: `prs_section(state)`, `prs_scores_selector(state)`, `prs_results_table(state)`, `prs_progress_section(state)`, `prs_build_selector(state)`, `prs_compute_button(state)`. Each takes a state class parameter so the same components work with any concrete state inheriting `PRSComputeStateMixin`. |
| `prs_ui.pages.*` | UI panels: `metadata` (grid browser), `scoring` (file viewer), `compute` (standalone PRS workflow using reusable components + VCF upload + genomic data grid) |

### Cleanup pipeline (`just_prs.cleanup`)

Raw PGS Catalog CSVs have data quality issues that `cleanup.py` fixes:
- **Genome build normalization**: 9 raw variants (hg19, hg37, hg38, NCBI36, hg18, NCBI35, GRCh37, GRCh38, NR) are mapped to canonical `GRCh37`, `GRCh38`, `GRCh36`, or `NR` via `BUILD_NORMALIZATION` dict.
- **Column renaming**: Verbose PGS column names (e.g. `Polygenic Score (PGS) ID`) become snake_case (`pgs_id`). The full mapping is `_SCORES_COLUMN_RENAME` / `_PERF_COLUMN_RENAME` / `_EVAL_COLUMN_RENAME`.
- **Metric string parsing**: Performance metrics stored as strings like `"1.55 [1.52,1.58]"` or `"-0.7 (0.15)"` are parsed into `{estimate, ci_lower, ci_upper, se}` via `parse_metric_string()`.
- **Performance flattening**: `clean_performance_metrics()` joins with evaluation sample sets and produces numeric columns for OR, HR, Beta, AUROC, and C-index. `best_performance_per_score()` selects one row per PGS ID (largest sample, European-preferred).

### PRSCatalog class (`just_prs.prs_catalog`)

`PRSCatalog` is the primary interface for working with PGS Catalog data. It produces and persists 3 cleaned parquet files (`scores.parquet`, `performance.parquet`, `best_performance.parquet`) and loads them as LazyFrames. Loading uses a 3-tier fallback chain: local cleaned parquets -> HuggingFace pull -> raw FTP download + cleanup. Raw FTP parquets are cached separately in a `raw/` subdirectory to avoid collision with cleaned files.

Key methods: `scores()`, `search()`, `best_performance()`, `score_info_row()`, `compute_prs()`, `compute_prs_batch()`, `percentile()`, `build_cleaned_parquets()`, `push_to_hf()`.

The package public API (`just_prs.__init__`) exports: `PRSCatalog`, `normalize_vcf`, `VcfFilterConfig`, `resolve_cache_dir`, `classify_model_quality`, `interpret_prs_result`, `format_effect_size`, `format_classification`, `__version__`, `__package_name__`.

The `prs-ui` package public API (`prs_ui.__init__`) exports: `PRSComputeStateMixin`, `prs_section`, `prs_scores_selector`, `prs_results_table`, `prs_progress_section`, `prs_build_selector`, `prs_compute_button`.

### HuggingFace sync (`just_prs.hf`)

Cleaned metadata parquets are synced to/from the HuggingFace dataset repo `just-dna-seq/polygenic_risk_scores` under the `data/` prefix. The HF token is resolved from: explicit argument > `.env` file (via `python-dotenv`) > `HF_TOKEN` environment variable. CLI commands: `just-prs catalog bulk clean-metadata`, `push-hf`, `pull-hf`.

### UI architecture notes

- **Five state classes** with three independent MUI DataGrids via `LazyFrameGridMixin` (which uses `mixin=True`). Each concrete mixin subclass gets its own independent set of reactive grid vars:
  - `AppState(rx.State)` — shared vars: `active_tab`, `genome_build`, `cache_dir`, `status_message`, `pgs_id_input`
  - `MetadataGridState(LazyFrameGridMixin, AppState)` — metadata browser + scoring file viewer grid
  - `GenomicGridState(LazyFrameGridMixin, AppState)` — normalized VCF genomic data grid. After VCF upload, runs `normalize_vcf()` (strip chr prefix, compute genotype, apply PASS filter) and loads the resulting parquet into a browsable DataGrid. The normalized parquet path is also used by `ComputeGridState` for PRS computation.
  - `PRSComputeStateMixin(rx.State, mixin=True)` — **reusable** PRS computation mixin: score loading via `PRSCatalog`, row selection, batch PRS computation, quality assessment, CSV export. Accepts genotypes via LazyFrame (preferred, via `set_prs_genotypes_lf()`) or parquet path (`prs_genotypes_path`). Designed for embedding in any Reflex app.
  - `ComputeGridState(PRSComputeStateMixin, LazyFrameGridMixin, AppState)` — concrete state for the standalone Compute PRS page. Adds VCF upload + genome build detection on top of the mixin.
  - **Important**: `AppState` must NOT inherit from `LazyFrameGridMixin` — otherwise substates that also list the mixin create an unresolvable MRO diamond.
- **Reusable components** (`prs_ui.components`): Each component function accepts a `state` class parameter, so the same UI works with any concrete state inheriting `PRSComputeStateMixin`. The primary entry point is `prs_section(state)` which composes build selector, score grid, compute button, progress bar, and results table.
- The Metadata tab shows **raw** PGS Catalog columns for general-purpose browsing of all 7 sheets.
- The Compute tab (default tab) uses **cleaned** data from `PRSCatalog` with normalized genome builds and snake_case column names. Scores are loaded into the MUI DataGrid with server-side virtual scrolling — no manual pagination.
- VCF upload triggers automatic normalization via `GenomicGridState.normalize_uploaded_vcf()` which runs `normalize_vcf()` (strip chr prefix, compute genotype, PASS filter) and shows the result in a browsable genomic data grid. The normalized parquet is reused by `ComputeGridState` for PRS computation.
- PRS results include **quality assessment**: AUROC-based model quality labels (High/Moderate/Low/Very Low), effect sizes (OR/HR/Beta with CI), classification metrics (AUROC/C-index), evaluation population ancestry, and plain-English interpretation summaries. Results can be exported as CSV via `download_prs_results_csv()`.
- `lazyframe_grid()` already sets `pagination=False` and `hide_footer=True` internally — do NOT pass them again or you get a duplicate kwarg error.

### Running the UI

The web UI is a Reflex app in the `prs-ui/` workspace member. Since `prs-ui` is an optional workspace dependency, you must install all packages first:

```bash
uv sync --all-packages
uv run ui
```

Or equivalently:

```bash
cd prs-ui
uv run reflex run
```

This launches a local web server (default http://localhost:3000). The `ui` script entry point is defined in the root `pyproject.toml` and calls `just_prs.cli:launch_ui`, which checks that `prs-ui` is installed before starting the Reflex server.

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

**Via Web UI:** Open the Compute tab, upload your VCF (drag-and-drop), select genome build, load scores, check the ones you want, and click Compute.

### Embedding the PRS UI in another Reflex app (e.g. just-dna-lite)

The PRS computation UI is packaged as reusable Reflex components. A host app adds `prs-ui` (and `just-prs`) as dependencies, creates a concrete state class that mixes in `PRSComputeStateMixin`, provides a normalized genotypes LazyFrame, and renders the section. The preferred input method is a polars LazyFrame (memory-efficient, avoids redundant I/O):

```python
import polars as pl
import reflex as rx
from reflex_mui_datagrid import LazyFrameGridMixin
from prs_ui import PRSComputeStateMixin, prs_section


class MyAppState(rx.State):
    genome_build: str = "GRCh38"
    cache_dir: str = "/path/to/cache"
    status_message: str = ""


class PRSState(PRSComputeStateMixin, LazyFrameGridMixin, MyAppState):
    """Concrete PRS state for the host app."""

    def on_vcf_ready(self, parquet_path: str) -> None:
        """Called after the host app normalizes a VCF."""
        lf = pl.scan_parquet(parquet_path)
        self.set_prs_genotypes_lf(lf)
        self.prs_genotypes_path = parquet_path


def prs_page() -> rx.Component:
    return prs_section(PRSState)
```

Key integration points:
- **LazyFrame is the preferred input** -- call `set_prs_genotypes_lf(lf)` with a `pl.scan_parquet()` LazyFrame. This is memory-efficient and avoids re-reading the file on each PRS computation.
- `prs_genotypes_path` is the fallback string path. Set it alongside the LazyFrame so the mixin can pass it to `compute_prs()` if needed.
- The host app's state must provide `genome_build`, `cache_dir`, and `status_message` vars (inherited from a shared parent or defined directly).
- Call `initialize_prs()` on page load to auto-load PGS Catalog scores into the grid.
- Individual sub-components (`prs_scores_selector`, `prs_results_table`, `prs_compute_button`, etc.) can be used independently for custom layouts.

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

### polars-bio caveats

- `polars-bio` uses DataFusion as its query engine for VCF reading. Multi-column aggregations on DataFusion-backed LazyFrames can fail with "all columns in a record batch must have the same length". **Always `.collect()` the joined LazyFrame first**, then compute aggregations on the materialized DataFrame.

---

## Data Directory Conventions

**Data must be strictly separated from code.** Generated data, downloaded files, uploaded files, and computation outputs must NEVER be written to the project root or source tree. This project works with genomic data (VCF files, scoring files) that can be hundreds of megabytes — committing them to git will break pushes to GitHub (100 MB limit) and bloat the repository permanently.

### Input data (`data/input/`)

User-provided input files (VCF uploads, custom scoring files, etc.) go to `data/input/`. This directory is gitignored. The Reflex UI must write uploaded files here, never to `prs-ui/uploaded_files/` or any directory inside the source tree.

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
| `<cache>/scores/` | Cached scoring files for PRS computation |
| `<cache>/normalized/` | Normalized VCF parquets (auto-populated by the web UI) |
| `<cache>/test-data/` | Test VCF files and fixtures |
| `<cache>/plink2/` | Auto-downloaded PLINK2 binary |

### Rules

- **NEVER commit large data files.** VCF (`.vcf`, `.vcf.gz`), parquet (`.parquet`), gzipped data (`.gz`, `.bgz`), FASTA (`.fa`, `.fasta`), and BAM/CRAM files must NEVER be added to git. GitHub rejects files > 100 MB and large files in history are extremely difficult to remove.
- **CLI defaults** must always point to `data/output/<subdir>` (or `./output/<subdir>` for legacy), never `./` or `./pgs_metadata/` etc.
- **Library code** (`PRSCatalog`, `scoring.py`) must use `resolve_cache_dir()` from `just_prs.scoring` (or accept explicit paths). Never hardcode OS-specific cache paths.
- **Tests** must use `resolve_cache_dir() / "test-data"`, never write to the project tree.
- **UI uploaded files** must go to `data/input/`, never inside `prs-ui/` or any source directory.
- **Never add data directories** (parquet, CSV, VCF, gz) to git. The `.gitignore` blocks `data/`, `output/`, `pgs_metadata/`, `pgs_scores/`, `scores/`, and `**/uploaded_files/`.

---

## uv Project Management

- **Dependency Management**: Use `uv sync` and `uv add`. NEVER use `uv pip install`.
- **Project Configuration**: Use `project.toml` as the single source of truth for dependencies and project metadata.
- **Versioning**: Do not hardcode versions in `__init__.py`; rely on `project.toml`.

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


1. Dagster 1.12.x+ API Notes & Gotchas
Many older Dagster tutorials use deprecated APIs. Keep these rules in mind for modern Dagster versions:
Context Access: get_dagster_context() does NOT exist. You must pass context: AssetExecutionContext explicitly to your functions.
Metadata Logging: context.log.info() does NOT accept a metadata keyword argument. Use context.add_output_metadata() separately.
Run Logs: EventRecordsFilter does NOT have a run_ids parameter. Instead, use instance.all_logs(run_id, of_type=...).
Asset Materializations: Use EventLogEntry.asset_materialization (which returns Optional[AssetMaterialization]), not DagsterEvent.asset_materialization.
Job Hooks: The hooks parameter in define_asset_job must be a set, not a list (e.g., hooks={my_hook}).
Asset Resolution: Use defs.resolve_all_asset_specs() instead of the deprecated defs.get_all_asset_specs().
Asset Job Config: Asset job config uses the "ops" key, not "assets". Using "assets" causes a DagsterInvalidConfigError.
2. Best Practices for Assets & IO
Declarative Assets: Prioritize Software-Defined Assets (SDA) over imperative ops. Include all assets in Definitions(assets=[...]) for complete lineage visibility in the UI.
Polars Integration: Use dagster-polars with PolarsParquetIOManager for pl.LazyFrame assets to automatically get schema and row counts in the Dagster UI.
Large Data / Streaming: Use lazy_frame.sink_parquet() and NEVER .collect().write_parquet() on large data to avoid out-of-memory errors.
Path Assets: When returning a Path from an asset, add "dagster/column_schema": polars_schema_to_table_schema(path) to ensure schema visibility in the UI.
Asset Checks: Use @asset_check for validation and include them in your job via AssetSelection.checks_for_assets(...).
3. Execution & Concurrency Patterns
Concurrency Limits: Use op_tags={"dagster/concurrency_key": "name"} to limit parallel execution for resource-intensive assets.
Timestamps: Timestamps are on RunRecord, not DagsterRun. run.start_time will raise an AttributeError. Retrieve instance.get_run_records() and use record.start_time/record.end_time (Unix floats) or record.create_timestamp (datetime).
Partition Keys for Runs: create_run_for_job doesn't accept a direct partition_key parameter. Pass it via tags instead: tags={"dagster/partition": partition_key}.
Dynamic Partitions Pattern:
Create partition def: PARTS = DynamicPartitionsDefinition(name="files")
Discovery asset registers partitions: context.instance.add_dynamic_partitions(PARTS.name, keys)
Partitioned assets use: partitions_def=PARTS and access context.partition_key
Collector depends on partitioned output via deps=[partitioned_asset] and scans the filesystem/storage for results.
4. Web UI / Asynchronous Execution Pattern
If you are running Dagster alongside a Web UI (like Reflex, FastAPI, etc.), use the Try-Daemon-With-Fallback pattern:
Submission vs Execution:
Attempt to submit the run to the daemon first: instance.submit_run(run_id, workspace=None). If this fails (e.g., due to missing ExternalPipelineOrigin in web contexts), fall back to job.execute_in_process().
Rust/PyO3 Thread Safety:
NEVER use asyncio.to_thread() or asyncio.create_task() with Dagster objects (it causes PyO3 panics: "Cannot drop pointer into Python heap without the thread being attached"). Use loop.run_in_executor(None, sync_execution_function, ...) for thread-safe background execution that doesn't block your UI.
Orphaned Run Cleanup:
If you use execute_in_process inside a web server process, runs will be abandoned (stuck in STARTED status) if the server restarts.
Add startup cleanup logic targeting DagsterRunStatus.NOT_STARTED.
Use atexit or signal handlers (SIGTERM/SIGINT) to mark active in-process runs as CANCELED on graceful server shutdown.
5. Common Anti-Patterns to Avoid
Using dagster job execute CLI: This is deprecated.
Hardcoding asset names: Resolve them dynamically using defs.
Suspended jobs holding locks: If a job crashes while querying local DBs (like DuckDB/SQLite), it can hold file locks. Handle connections properly via context managers or resources.
Processing failures in exception handlers: Keep business logic out of exception handlers when executing runs. Catch the exception, register the failure, and cleanly proceed to your fallback mechanism.

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

### 1b. NEVER put SourceAsset keys in computed asset `deps` when using AutomationCondition

`AutomationCondition.eager()` expands to include `~any_deps_missing`. Dagster treats `SourceAsset`s as **always missing** (they're never materialized by Dagster). If a computed asset lists a SourceAsset in its `deps`, the `any_deps_missing` check will be permanently True, and `eager()` will **never fire**.

```python
# WRONG — ebi_reference_panel is a SourceAsset → blocks eager() forever
@asset(deps=["ebi_reference_panel"], automation_condition=AutomationCondition.eager())
def reference_panel(...): ...

# CORRECT — no SourceAsset in deps; document the source URL in output metadata instead
@asset(automation_condition=AutomationCondition.eager())
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
defs = dg.Definitions(
    assets=[
        ebi_reference_panel,   # SourceAsset — must be listed
        ebi_pgs_catalog,       # SourceAsset — must be listed
        reference_panel,
        per_pgs_scores,
        reference_distributions,
        hf_prs_percentiles,
    ],
    ...
)
```

Omitting a `SourceAsset` from `Definitions` makes it invisible in the UI even if assets declare `deps` on it.

### Quick checklist before finishing any Dagster asset file

- [ ] Every remote data origin has a `SourceAsset` with `group_name="external"` and a `metadata={"url": ...}`.
- [ ] Every `SourceAsset` is listed in `Definitions(assets=[...])`.
- [ ] Every asset that scans a directory written by another asset has `deps=[AssetDep("that_asset")]`.
- [ ] No `AssetIn` is used with `Output[Path]` unless a `UPathIOManager` is configured.
- [ ] Every `@asset` has `group_name` set (never omit it).
- [ ] The final destination asset (HF upload, S3 push, DB write) is named after the destination, not the action.

---

## Dagster Single-Command Startup Pattern (CRITICAL)

When building a CLI command that launches Dagster (`dagster dev`) AND auto-runs the pipeline, follow this exact sequence. **Never use `subprocess.run()` or `subprocess.Popen()` for Dagster itself.**

### Full pre-flight before `os.execvp`

```python
def launch_pipeline(host: str = "0.0.0.0", port: int = 3010) -> None:
    import os, signal, time, sys
    from pathlib import Path

    # 1. Set DAGSTER_HOME to a project-relative path (never default ~/.dagster)
    project_root = _find_project_root()
    dagster_home = project_root / "data" / "output" / "dagster"
    dagster_home.mkdir(parents=True, exist_ok=True)
    os.environ["DAGSTER_HOME"] = str(dagster_home)

    # 2. Generate dagster.yaml if missing
    _ensure_dagster_yaml(dagster_home)

    # 3. Kill any orphaned process holding the port
    _kill_port(port)

    # 4. Cancel stuck runs from previous sessions
    _cancel_orphaned_runs()

    # 5. Replace this process with dagster dev — NEVER use subprocess.run/Popen here
    dagster_bin = str(Path(sys.executable).parent / "dagster")
    os.execvp(dagster_bin, ["dagster", "dev", "-m", "my_pkg.definitions",
                            "--host", host, "--port", str(port)])
```

### Why `os.execvp` (not `subprocess.run` or `Popen`)

Dagster's daemon uses complex internal signal handling. When trapped inside a `subprocess.run()` or `Popen()`, SIGINT/SIGTERM do not propagate correctly and the daemon does not shut down cleanly. `os.execvp` **replaces** the current Python process with `dagster`, so Dagster becomes the primary process and owns all signal handling. Ctrl+C works correctly.

### Why `AutomationCondition` (not job submission after startup)

Because `os.execvp` replaces the current process, there is no opportunity to submit a job "after dagster starts". Instead, mark assets with automation conditions. The `AssetDaemon` (enabled by `auto_materialize: enabled: true` in `dagster.yaml`) picks them up automatically — no timing hacks, no subprocess coordination, no SQLite conflicts.

### `on_missing()` for root assets, `eager()` for downstream assets (CRITICAL)

`AutomationCondition.eager()` fires when upstream dependencies are updated OR the asset is "newly missing". However, for **root assets** (no deps), `eager()` does NOT fire on startup because its `SinceCondition` is reset by `InitialEvaluationCondition` on the same tick as the `newly_missing` trigger — they cancel out.

Use `AutomationCondition.on_missing()` for root assets (assets with no upstream deps). It reliably fires on the first daemon evaluation tick when the asset has never been materialized. Use `AutomationCondition.eager()` for all downstream assets (assets with deps) — it fires when any upstream dependency is updated.

```python
from dagster import AutomationCondition, asset

@asset(automation_condition=AutomationCondition.on_missing())
def root_download_asset(...): ...

@asset(
    deps=["root_download_asset"],
    automation_condition=AutomationCondition.eager(),
)
def downstream_compute_asset(...): ...
```

**Never use `eager()` on root assets** — it will silently do nothing and the pipeline won't start.

`SourceAsset`s (external data) must NOT have an automation condition — they are always considered fresh. Also never put SourceAsset keys in computed asset `deps` (see lineage rules above).

### `dagster.yaml` template (telemetry off, auto_materialize on)

```yaml
telemetry:
  enabled: false

auto_materialize:
  enabled: true
```

Generate this file at `{DAGSTER_HOME}/dagster.yaml` if it does not exist. The `telemetry: enabled: false` setting prevents `RotatingFileHandler` crashes in Dagster's event log writer.

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