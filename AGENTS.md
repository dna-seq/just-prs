# Python Coding Standards & Best Practices

## Project Architecture

This project is a **uv workspace** with a non-published root wrapper and three subprojects:

- **`just-prs/src/just_prs/`** — Core library: PRS computation, PGS Catalog REST API client, FTP downloads, VCF reading, scoring file parsing. CLI entrypoint via Typer. Published to PyPI as `just-prs`.
- **`prs-ui/`** — Reflex web app for interactive PRS computation. Has its own `pyproject.toml` and depends on `just_prs`. Run with `uv run ui` from workspace root or `uv run reflex run` from inside `prs-ui/`. Published to PyPI as `prs-ui`.
- **`prs-pipeline/`** — Dagster pipeline for computing PRS reference distributions from the 1000G panel. Has its own `pyproject.toml` and depends on `just_prs`. All pipeline commands (`run`, `catalog`, `launch`) default to launching the Dagster UI for monitoring. Use `--headless` on `run`/`catalog` for in-process execution without UI.

The workspace root (`pyproject.toml` at repo root) is a non-published wrapper named `just-prs-workspace`. It depends on all three subprojects and **must re-export all CLI entry points** from subprojects so that every command is available via `uv run <name>` from the workspace root. The pipeline CLI has three main commands: `pipeline run` (full pipeline with Dagster UI), `pipeline catalog` (catalog pipeline with Dagster UI), and `pipeline launch` (Dagster UI only, no specific job pre-selected). All three launch the Dagster UI by default. Use `--headless` on `run`/`catalog` for in-process execution without UI. Tests live in `just-prs/tests/`.

**ALL PIPELINE COMMANDS LAUNCH DAGSTER UI BY DEFAULT (CRITICAL).** `pipeline run`, `pipeline catalog`, and `pipeline launch` all start the Dagster webserver with monitoring UI. Headless in-process execution is only available via the explicit `--headless` flag on `run`/`catalog`. The Dagster UI URL (`http://<host>:<port>`) must always be printed prominently at startup.

**EVERY CLI THAT STARTS A SERVER MUST PRINT ITS URL (CRITICAL).** When any CLI command starts a web server or UI (Reflex UI via `uv run ui`, Dagster UI via `pipeline run`/`catalog`/`launch`), the URL (`http://<host>:<port>`) must be printed prominently in the first lines of output so the user always knows where to open their browser.

**ALL CLIs LOAD `.env` VIA `python-dotenv` AT STARTUP (CRITICAL).** Both `prs_ui.cli` and `prs_pipeline.cli` call `load_dotenv()` before reading any configuration. Users override defaults by setting env vars in `.env` (see `.env.template`). Key config env vars: `PRS_UI_HOST`, `PRS_UI_PORT` (Reflex UI, default `0.0.0.0:3000`), `PRS_PIPELINE_HOST`, `PRS_PIPELINE_PORT` (Dagster UI, default `0.0.0.0:3010`), `PRS_CACHE_DIR`, `HF_TOKEN`, `PRS_PIPELINE_PANEL`, `PRS_PIPELINE_STARTUP_JOB`. When adding new configurable values, always read them from env vars with sensible defaults, document them in `.env.template`, and mention them in `AGENTS.md`.

**CRITICAL: All subproject CLI entry points must be registered in the workspace root `pyproject.toml` `[project.scripts]`.** When adding a new CLI entry point to any subproject, always add it to the root `pyproject.toml` as well. Users run commands from the workspace root with `uv run <script>`, and scripts not registered at the root level will not be found. Current entry points:

| Script | Entry point | Subproject |
|--------|-------------|------------|
| `prs` | `just_prs.cli:run` | `just-prs` |
| `just-prs` | `just_prs.cli:run` | `just-prs` |
| `ui` | `prs_ui.cli:launch_ui` | `prs-ui` |
| `pipeline` | `prs_pipeline.cli:app` | `prs-pipeline` |

### Key modules

| Module | Purpose |
|--------|---------|
| `just_prs.prs_catalog` | **`PRSCatalog`** — high-level class for search, PRS computation, and percentile estimation using cleaned bulk metadata (no REST API calls). Persists cleaned parquets locally with HuggingFace sync; percentile lookup refreshes reference distributions from HF on miss; `reference_data_status()` reports whether precomputed reference data exists for a PGS ID, which superpopulations are available, and whether source is local cache vs HF sync. |
| `just_prs.cleanup` | Pure-function pipeline: genome build normalization, column renaming, metric string parsing, performance metric cleanup |
| `just_prs.hf` | HuggingFace Hub integration: `push_cleaned_parquets()` / `pull_cleaned_parquets()` for syncing cleaned metadata parquets to/from `just-dna-seq/polygenic_risk_scores`; `push_pgs_catalog()` uploads combined metadata+scores to `just-dna-seq/pgs-catalog` and rewrites `data/metadata/scores.parquet` to parquet-first scoring links (`ftp_link`) while preserving original EBI links in `ftp_link_ebi`. |
| `just_prs.normalize` | VCF normalization: `normalize_vcf()` reads VCF with polars-bio, strips chr prefix, renames id→rsid, computes genotype List[Str], applies configurable quality filters (FILTER, DP, QUAL), warns on chrY for females, sinks to zstd Parquet. `VcfFilterConfig` (Pydantic v2) holds filter settings. |
| `just_prs.prs` | `compute_prs()` / `compute_prs_batch()` — core PRS engine. `compute_prs()` accepts optional `genotypes_lf` LazyFrame to skip VCF re-reading when a normalized parquet is available. |
| `just_prs.reference` | Reference panel utilities and pgen operations: `download_reference_panel()` (panel-aware: `panel="1000g"` or `"hgdp_1kg"`), `parse_pvar()` (parse .pvar.zst with parquet caching), `parse_psam()` (parse .psam sample files), `read_pgen_genotypes()` (extract genotypes from .pgen via pgenlib), `match_scoring_to_pvar()` (allele-aware variant matching via polars — standalone use only), `compute_reference_prs_polars()` (single-PGS scoring using pgenlib + numpy; uses DuckDB for variant matching against pvar parquet to avoid loading 75M rows into memory), `compute_reference_prs_batch()` (memory-efficient batch scoring: resolves panel once via `_ResolvedRefPanel`, uses DuckDB for variant matching, aggregates distributions per PGS ID immediately and discards raw scores, returns `BatchScoringResult`), `compute_reference_prs_plink2()` (legacy, for cross-validation), `aggregate_distributions()`, `enrich_distributions()` (join distributions with cleaned metadata: traits, EFO, AUROC, OR, C-index, ancestry), `ancestry_percentile()`, `ReferencePanelError`. Panel-aware constants: `REFERENCE_PANELS` dict, `DEFAULT_PANEL = "1000g"`. Result models: `ScoringOutcome` (per-ID outcome), `BatchScoringResult` (panel, distributions_df, outcomes, quality_df — no raw scores held in memory). `_ResolvedRefPanel` caches file paths, psam, and variant count once per batch; variant matching uses DuckDB to scan the pvar parquet (~434 MB on disk) without materializing 75M rows in polars (~6 GB). |
| `just_prs.vcf` | VCF reading via `polars-bio`, genome build detection, dosage computation |
| `just_prs.scoring` | Download, parse, and cache PGS scoring files. `SCORING_FILE_SCHEMA` — comprehensive column type map from the PGS Catalog spec (30+ columns). `parse_scoring_file()` transparently reads/writes a parquet cache (zstd-9 compressed) alongside the `.txt.gz`, with header metadata embedded as file-level metadata. `scoring_parquet_path()` computes cache paths. `read_scoring_header()` reads PGS header metadata from parquet or `.txt.gz`. `load_scoring()` checks parquet cache first and skips `.txt.gz` download when it exists. |
| `just_prs.ftp` | Bulk FTP/HTTPS downloads of raw metadata sheets and scoring files via `fsspec` |
| `just_prs.catalog` | Synchronous REST API client (`PGSCatalogClient`) for PGS Catalog — used for individual lookups, not for bulk metadata |
| `just_prs.models` | Pydantic v2 models (`ScoreInfo`, `PRSResult`, `PerformanceInfo`, etc.) |
| `just_prs.quality` | Pure-logic quality assessment helpers: `classify_model_quality()`, `interpret_prs_result()`, `format_effect_size()`, `format_classification()`. No Reflex dependency -- shared between core library and UI. |
| `prs_ui.state` | Reflex `AppState` + grid states + `PRSComputeStateMixin(rx.State, mixin=True)`. The mixin encapsulates all PRS computation logic (score loading, selection, batch compute, CSV export) and is designed for reuse in any Reflex app. `ComputeGridState` subclasses it with VCF-upload-specific behavior for the standalone app. |
| `prs_ui.components` | **Reusable UI components**: `prs_section(state)`, `prs_scores_selector(state)`, `prs_results_table(state)`, `prs_progress_section(state)`, `prs_build_selector(state)`, `prs_compute_button(state)`. Each takes a state class parameter so the same components work with any concrete state inheriting `PRSComputeStateMixin`. |
| `prs_ui.pages.*` | UI panels: `metadata` (grid browser), `scoring` (file viewer), `compute` (standalone PRS workflow using reusable components + VCF upload + genomic data grid) |
| `prs_pipeline.runtime` | `ResourceReport` (Pydantic model) and `resource_tracker` context manager — tracks CPU%, peak memory, duration via `psutil` and logs to Dagster output metadata |
| `prs_pipeline.utils` | `resource_summary_hook` — Dagster `@success_hook` that aggregates per-asset resource metrics into a run-level summary |

### Cleanup pipeline (`just_prs.cleanup`)

Raw PGS Catalog CSVs have data quality issues that `cleanup.py` fixes:
- **Genome build normalization**: 9 raw variants (hg19, hg37, hg38, NCBI36, hg18, NCBI35, GRCh37, GRCh38, NR) are mapped to canonical `GRCh37`, `GRCh38`, `GRCh36`, or `NR` via `BUILD_NORMALIZATION` dict.
- **Column renaming**: Verbose PGS column names (e.g. `Polygenic Score (PGS) ID`) become snake_case (`pgs_id`). The full mapping is `_SCORES_COLUMN_RENAME` / `_PERF_COLUMN_RENAME` / `_EVAL_COLUMN_RENAME`.
- **Metric string parsing**: Performance metrics stored as strings like `"1.55 [1.52,1.58]"` or `"-0.7 (0.15)"` are parsed into `{estimate, ci_lower, ci_upper, se}` via `parse_metric_string()`.
- **Performance flattening**: `clean_performance_metrics()` joins with evaluation sample sets and produces numeric columns for OR, HR, Beta, AUROC, and C-index. `best_performance_per_score()` selects one row per PGS ID (largest sample, European-preferred).

### PRSCatalog class (`just_prs.prs_catalog`)

`PRSCatalog` is the primary interface for working with PGS Catalog data. It produces and persists 3 cleaned parquet files (`scores.parquet`, `performance.parquet`, `best_performance.parquet`) and loads them as LazyFrames. Loading uses a 3-tier fallback chain: local cleaned parquets -> HuggingFace pull -> raw FTP download + cleanup. Raw FTP parquets are cached separately in a `raw/` subdirectory to avoid collision with cleaned files.

Key methods: `scores()`, `search()`, `best_performance()`, `score_info_row()`, `compute_prs()`, `compute_prs_batch()`, `percentile()`, `reference_data_status()`, `build_cleaned_parquets()`, `push_to_hf()`.

The package public API (`just_prs.__init__`) exports: `PRSCatalog`, `ReferencePanelError`, `normalize_vcf`, `VcfFilterConfig`, `resolve_cache_dir`, `classify_model_quality`, `interpret_prs_result`, `format_effect_size`, `format_classification`, `compute_reference_prs_polars`, `compute_reference_prs_batch`, `download_reference_panel`, `reference_panel_dir`, `parse_pvar`, `parse_psam`, `read_pgen_genotypes`, `match_scoring_to_pvar`, `aggregate_distributions`, `enrich_distributions`, `ancestry_percentile`, `ReferenceDistribution`, `ScoringOutcome`, `BatchScoringResult`, `REFERENCE_PANELS`, `DEFAULT_PANEL`, `__version__`, `__package_name__`.

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
- **Reusable components** (`prs_ui.components`): Each component function accepts a `state` class parameter, so the same UI works with any concrete state inheriting `PRSComputeStateMixin`. The primary entry point is `prs_section(state)` which composes build selector, score grid, compute button, progress bar, and results table. The compute section includes an `All available populations` toggle to request per-superpopulation percentiles where reference distributions exist.
- The Metadata tab shows **raw** PGS Catalog columns for general-purpose browsing of all 7 sheets.
- The Compute tab (default tab) uses **cleaned** data from `PRSCatalog` with normalized genome builds and snake_case column names. Scores are loaded into the MUI DataGrid with server-side virtual scrolling — no manual pagination.
- VCF upload triggers automatic normalization via `GenomicGridState.normalize_uploaded_vcf()` which runs `normalize_vcf()` (strip chr prefix, compute genotype, PASS filter) and shows the result in a browsable genomic data grid. The normalized parquet is reused by `ComputeGridState` for PRS computation.
- PRS results include **quality assessment**: AUROC-based model quality labels (High/Moderate/Low/Very Low), effect sizes (OR/HR/Beta with CI), classification metrics (AUROC/C-index), evaluation population ancestry, and plain-English interpretation summaries. Results can be exported as CSV via `download_prs_results_csv()`.
- PRS result rows/cards show explicit **reference percentile status**: whether precomputed 1000G reference data exists (`precomputed(...)` vs `not precomputed`), which populations are available, and the source (`HuggingFace prs-percentiles` vs local cache). UI text clarifies these reference distributions are precomputed from reference panel scoring, not direct PGS Catalog API percentiles.
- `lazyframe_grid()` already sets `pagination=False` and `hide_footer=True` internally — do NOT pass them again or you get a duplicate kwarg error.

### Running the UI

The web UI is a Reflex app in the `prs-ui/` workspace member:

```bash
uv sync --all-packages
uv run ui
```

Or equivalently:

```bash
cd prs-ui
uv run reflex run
```

This launches a local web server (default http://0.0.0.0:3000). The CLI always prints the UI URL prominently at startup. The `ui` script entry point is defined in both the root `pyproject.toml` (convenience) and `prs-ui/pyproject.toml`, pointing to `prs_ui.cli:launch_ui`.

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
| `<cache>/scores/` | Cached scoring files for PRS computation. Contains both `.txt.gz` (original download) and `.parquet` (spec-driven parquet cache with zstd-9 compression and embedded PGS header metadata). The pipeline `scoring_files_parquet` asset deletes `.txt.gz` after verified conversion to save disk space (~5.5 GB savings for the full catalog). |
| `<cache>/normalized/` | Normalized VCF parquets (auto-populated by the web UI) |
| `<cache>/reference_scores/{panel}/{pgs_id}/` | Per-ID reference panel scores (cached by `compute_reference_prs_batch`) |
| `<cache>/percentiles/` | Distribution and quality parquets (`{panel}_distributions.parquet`, `{panel}_quality.parquet`) |
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
- **Upload assets** (`hf_pgs_catalog`, `hf_prs_percentiles`, `hf_polygenic_risk_scores`): Always run — uploading IS the point. They re-upload even if data hasn't changed (HF handles dedup).

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

- Pipeline sensors and assets must be at least as robust as standard bioinformatics pipeline engines (Nextflow, WDL/Cromwell, Snakemake) — materialization-only checks are unacceptable
- All pipeline CLI commands (`run`, `catalog`, `launch`) must launch the Dagster UI by default; headless is opt-in via `--headless`
- Every CLI that starts a server (Reflex UI, Dagster UI) must print its URL prominently in the first lines of output
- All CLIs must load `.env` via `python-dotenv` at startup; configurable values must be overridable via env vars
- Deprecation warnings in terminal output are critical blockers — investigate, update to non-deprecated APIs, and update AGENTS.md
- Never present headless/CLI-only mode as the default for anything that has a UI — the user expects visual monitoring
- Policy and rules must be written first (in AGENTS.md), then code must comply — not the other way around
- When the user says "the pipeline should compute scores for all PRS on reference population," they expect the full pipeline to do this automatically without extra commands
- The user is frustrated by naive automation that only checks surface-level state (e.g. "is it materialized?") instead of actual data completeness
- Never introduce full `collect()` on large LazyFrames when lazy aggregations suffice — use `pl.collect_all([lf1, lf2])` to stream multiple aggregations in one pass without materialising per-sample data

## Learned Workspace Facts

- Dagster 1.12+ rejects multiple `Definitions` objects at module scope — never assign a temporary `Definitions` to a module-level variable (even prefixed with `_`); wrap resolution in a function
- Dagster's SQLite storage produces `database is locked` errors under concurrent writes (multiple runs + daemon + sensors) — these are transient and do not stop runs, but PostgreSQL (`dagster-postgres`) is the production alternative
- Dagster's `AutomationCondition.on_missing()` and `.eager()` are broken for initial materialization in Dagster 1.12 — use run-once `@dg.sensor` instead
- The pipeline processes ~5,300 PGS IDs from the EBI catalog; a full scoring run takes several hours at ~0.18 IDs/second with peak RSS around 3–5 GB
- The `just-dna-seq/pgs-catalog` HuggingFace repo stores combined metadata+scores with parquet-first scoring links; `ftp_link` points to HF parquet mirrors and `ftp_link_ebi` preserves the original EBI URL
- The `just-dna-seq/prs-percentiles` HuggingFace repo stores precomputed reference population distributions
- The `PRS_PIPELINE_FORCE_RUN=1` env var is set by `pipeline run`/`catalog` to ensure the sensor submits the job when launching the Dagster UI
- `os.execvp` is used for Dagster UI mode because Dagster's daemon signal handling breaks under `subprocess.run()`/`Popen()`
- The scoring files parquet cache (zstd-9 compressed) saves ~5.5 GB vs raw `.txt.gz` for the full catalog
- `polars collect_schema()` reads only parquet footer metadata — data-page corruption (truncated writes, OOM-killed flushes) is only detected when `collect()` reads row groups; always wrap the full read sequence in try/except for `_CorruptParquet` sentinel
- Reference population distributions parquet schema: `pgs_id, superpopulation, mean, std, n, median, p5, p25, p75, p95` — one row per (pgs_id, superpopulation) combination
- `PRSCatalog.percentile()` does a one-time HF refresh on cache miss and retries lookup, so newly computed reference distributions are picked up without manual cache cleanup