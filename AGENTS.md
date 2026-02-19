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
| `just_prs.prs` | `compute_prs()` / `compute_prs_batch()` — core PRS engine |
| `just_prs.vcf` | VCF reading via `polars-bio`, genome build detection, dosage computation |
| `just_prs.scoring` | Download and parse PGS scoring files (gzipped TSV with `#` header) |
| `just_prs.ftp` | Bulk FTP/HTTPS downloads of raw metadata sheets and scoring files via `fsspec` |
| `just_prs.catalog` | Synchronous REST API client (`PGSCatalogClient`) for PGS Catalog — used for individual lookups, not for bulk metadata |
| `just_prs.models` | Pydantic v2 models (`ScoreInfo`, `PRSResult`, `PerformanceInfo`, etc.) |
| `prs_ui.state` | Reflex `AppState` — uses `PRSCatalog` for cleaned data in the Compute tab |
| `prs_ui.pages.*` | UI panels: `metadata` (grid browser), `scoring` (file viewer), `compute` (PRS workflow) |

### Cleanup pipeline (`just_prs.cleanup`)

Raw PGS Catalog CSVs have data quality issues that `cleanup.py` fixes:
- **Genome build normalization**: 9 raw variants (hg19, hg37, hg38, NCBI36, hg18, NCBI35, GRCh37, GRCh38, NR) are mapped to canonical `GRCh37`, `GRCh38`, `GRCh36`, or `NR` via `BUILD_NORMALIZATION` dict.
- **Column renaming**: Verbose PGS column names (e.g. `Polygenic Score (PGS) ID`) become snake_case (`pgs_id`). The full mapping is `_SCORES_COLUMN_RENAME` / `_PERF_COLUMN_RENAME` / `_EVAL_COLUMN_RENAME`.
- **Metric string parsing**: Performance metrics stored as strings like `"1.55 [1.52,1.58]"` or `"-0.7 (0.15)"` are parsed into `{estimate, ci_lower, ci_upper, se}` via `parse_metric_string()`.
- **Performance flattening**: `clean_performance_metrics()` joins with evaluation sample sets and produces numeric columns for OR, HR, Beta, AUROC, and C-index. `best_performance_per_score()` selects one row per PGS ID (largest sample, European-preferred).

### PRSCatalog class (`just_prs.prs_catalog`)

`PRSCatalog` is the primary interface for working with PGS Catalog data. It produces and persists 3 cleaned parquet files (`scores.parquet`, `performance.parquet`, `best_performance.parquet`) and loads them as LazyFrames. Loading uses a 3-tier fallback chain: local cleaned parquets -> HuggingFace pull -> raw FTP download + cleanup. Raw FTP parquets are cached separately in a `raw/` subdirectory to avoid collision with cleaned files.

Key methods: `scores()`, `search()`, `best_performance()`, `compute_prs()`, `percentile()`, `build_cleaned_parquets()`, `push_to_hf()`.

### HuggingFace sync (`just_prs.hf`)

Cleaned metadata parquets are synced to/from the HuggingFace dataset repo `just-dna-seq/polygenic_risk_scores` under the `data/` prefix. The HF token is resolved from: explicit argument > `.env` file (via `python-dotenv`) > `HF_TOKEN` environment variable. CLI commands: `just-prs catalog bulk clean-metadata`, `push-hf`, `pull-hf`.

### UI architecture notes

- **Three state classes** with two independent MUI DataGrids via `LazyFrameGridMixin` (which uses `mixin=True`). Each concrete mixin subclass gets its own independent set of reactive grid vars:
  - `AppState(rx.State)` — shared vars: `active_tab`, `genome_build`, `cache_dir`, `status_message`, `pgs_id_input`
  - `MetadataGridState(LazyFrameGridMixin, AppState)` — metadata browser + scoring file viewer grid
  - `ComputeGridState(LazyFrameGridMixin, AppState)` — compute PRS score selection grid
  - **Important**: `AppState` must NOT inherit from `LazyFrameGridMixin` — otherwise substates that also list the mixin create an unresolvable MRO diamond.
- The Metadata tab shows **raw** PGS Catalog columns for general-purpose browsing of all 7 sheets.
- The Compute tab (default tab) uses **cleaned** data from `PRSCatalog` with normalized genome builds and snake_case column names. Scores are loaded into the MUI DataGrid with server-side virtual scrolling — no manual pagination.
- `lazyframe_grid()` already sets `pagination=False` and `hide_footer=True` internally — do NOT pass them again or you get a duplicate kwarg error.

### Running the UI

The web UI is a Reflex app in the `prs-ui/` workspace member. To start it:

```bash
cd prs-ui
uv run reflex run
```

This launches a local web server (default http://localhost:3000). If `prs-ui` is not installed, run `uv sync --all-packages` from the workspace root first.

### Computing PRS on a custom VCF

**Via CLI:**

```bash
# Single score
prs compute --vcf /path/to/your/sample.vcf.gz --pgs-id PGS000001

# Multiple scores at once
prs compute --vcf /path/to/your/sample.vcf.gz --pgs-id PGS000001,PGS000002,PGS000003

# Explicit genome build + JSON output
prs compute --vcf /path/to/your/sample.vcf.gz --pgs-id PGS000001 --build GRCh37 --output results.json
```

**Via Python API (recommended for scripting):**

```python
from just_prs import PRSCatalog
from pathlib import Path

catalog = PRSCatalog()

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

# Estimate percentile
pct = catalog.percentile(prs_score=result.score, pgs_id="PGS000001")
print(f"Percentile: {pct:.1f}%")
```

**Via Web UI:** Open the Compute tab, upload your VCF (drag-and-drop), select genome build, load scores, check the ones you want, and click Compute.

### Reflex-specific patterns (CRITICAL)

- **State var mixin classes MUST use `rx.State` with `mixin=True`**: Declare mixins as `class MyMixin(rx.State, mixin=True)` so vars are injected independently into each concrete subclass. Each subclass must also inherit from `rx.State` (or another non-mixin state class). `LazyFrameGridMixin` already uses `mixin=True`, so `AppState` and `ComputeGridState` each get their own `lf_grid_rows`, `lf_grid_loaded`, etc.
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

### Cache directory (`~/.cache/just-prs/`)

Long-lived cached data used by `PRSCatalog` and tests goes to the user-level cache:

| Subdirectory | Contents |
|---|---|
| `~/.cache/just-prs/metadata/` | Cleaned parquets (auto-populated by `PRSCatalog`) |
| `~/.cache/just-prs/metadata/raw/` | Raw FTP parquets cached by `PRSCatalog` |
| `~/.cache/just-prs/scores/` | Cached scoring files for PRS computation |
| `~/.cache/just-prs/test-data/` | Test VCF files and fixtures |
| `~/.cache/just-prs/plink2/` | Auto-downloaded PLINK2 binary |

Override with `PRS_CACHE_DIR` environment variable.

### Rules

- **NEVER commit large data files.** VCF (`.vcf`, `.vcf.gz`), parquet (`.parquet`), gzipped data (`.gz`, `.bgz`), FASTA (`.fa`, `.fasta`), and BAM/CRAM files must NEVER be added to git. GitHub rejects files > 100 MB and large files in history are extremely difficult to remove.
- **CLI defaults** must always point to `data/output/<subdir>` (or `./output/<subdir>` for legacy), never `./` or `./pgs_metadata/` etc.
- **Library code** (`PRSCatalog`, `scoring.py`) must use `~/.cache/just-prs/` or accept explicit paths.
- **Tests** must use `~/.cache/just-prs/test-data/`, never write to the project tree.
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