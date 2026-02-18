# Python Coding Standards & Best Practices

## Project Architecture

This project has two packages managed by a single uv workspace:

- **`src/just_prs/`** — Core library: PRS computation, PGS Catalog REST API client, FTP downloads, VCF reading, scoring file parsing. CLI entrypoint via Typer.
- **`prs-ui/`** — Reflex web app for interactive PRS computation. Has its own `pyproject.toml` and depends on `just_prs`. Run from inside `prs-ui/` with `uv run reflex run`.

### Key modules

| Module | Purpose |
|--------|---------|
| `just_prs.prs` | `compute_prs()` / `compute_prs_batch()` — core PRS engine |
| `just_prs.vcf` | VCF reading via `polars-bio`, genome build detection, dosage computation |
| `just_prs.scoring` | Download and parse PGS scoring files (gzipped TSV with `#` header) |
| `just_prs.ftp` | Bulk FTP/HTTPS downloads of metadata sheets and scoring files via `fsspec` |
| `just_prs.catalog` | Synchronous REST API client (`PGSCatalogClient`) for PGS Catalog |
| `just_prs.models` | Pydantic v2 models (`ScoreInfo`, `PRSResult`, `PerformanceInfo`, etc.) |
| `prs_ui.state` | Reflex `AppState` — single state class inheriting `LazyFrameGridMixin` |
| `prs_ui.pages.*` | UI panels: `metadata` (grid browser), `scoring` (file viewer), `compute` (PRS workflow) |

### UI architecture notes

- The app uses a single `AppState` class that inherits `LazyFrameGridMixin` from `reflex-mui-datagrid`. This means there is **one MUI DataGrid** shared across the Metadata and Scoring tabs.
- The Compute PRS tab uses its own paginated table (not the MUI DataGrid) to avoid conflicts with the shared grid state. Score rows are stored as `list[dict[str, str]]` in `compute_scores_rows` with server-side search and pagination.
- Genome build filtering uses `BUILD_ALIASES` to map `GRCh38 → [GRCh38, hg38]` and `GRCh37 → [GRCh37, hg19, hg37]`, plus always including `NR` (not reported) entries.

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