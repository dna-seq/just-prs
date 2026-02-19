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

- The app uses a single `AppState` class that inherits `LazyFrameGridMixin` from `reflex-mui-datagrid`. This means there is **one MUI DataGrid** shared across the Metadata and Scoring tabs.
- The Compute PRS tab uses its own paginated table (not the MUI DataGrid) to avoid conflicts with the shared grid state. Score rows are stored as `list[dict[str, str]]` in `compute_scores_rows` with server-side search and pagination.
- The Metadata tab shows **raw** PGS Catalog columns for general-purpose browsing of all 7 sheets.
- The Compute tab uses **cleaned** data from `PRSCatalog` with normalized genome builds and snake_case column names. Genome build filtering happens at the data level (canonical builds), not via alias lists.

### polars-bio caveats

- `polars-bio` uses DataFusion as its query engine for VCF reading. Multi-column aggregations on DataFusion-backed LazyFrames can fail with "all columns in a record batch must have the same length". **Always `.collect()` the joined LazyFrame first**, then compute aggregations on the materialized DataFrame.

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