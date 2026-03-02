# Dagster Pipelines in just-prs

The `just-prs` workspace includes a Dagster-based data pipeline in the `prs-pipeline` subproject. This pipeline is responsible for large-scale data engineering tasks:

1. **Computing PRS reference distributions** by scoring the entire PGS Catalog against a reference panel (1000G or HGDP+1kGP).
2. **Downloading and cleaning metadata** from the PGS Catalog.

## Core Dagster Principles Used in This Project

1. **Assets over Tasks**: Dependencies are expressed as data assets, not task wiring. Instead of saying `task_a >> task_b`, `asset_b` simply declares it needs `asset_a`. Dagster figures out the execution order. The focus is on *what data should exist*, not *how to run a function*.
2. **Assets vs. Jobs**:
   - **Software-Defined Assets (SDAs)** form the core, declarative pipeline (`assets.py`). They are best for lineage tracking, data quality, and automation.
   - **Jobs** (`definitions.py`) are used as entry points to trigger specific sub-graphs of assets, initiated by CLI commands or UI actions.
3. **Abstracted Storage**: Pipeline outputs don't hardcode their absolute locations in the business logic. Paths are resolved consistently via resources (e.g., `CacheDirResource`).
4. **Metadata and Observability**: Every asset logs rich metadata (row counts, file paths, variant match rates). This makes the Dagster UI a complete data catalog, not just a task runner.

## Pipeline Architecture

The pipeline is modeled as Software-Defined Assets and is divided into two primary flows.

### 1. Reference Panel Pipeline (`prs_pipeline.assets`)

This pipeline computes population-level Polygenic Risk Score (PRS) distributions using a reference panel and PGS Catalog scoring files. The output is a per-score, per-superpopulation percentile table that lets end-users compare their personal PRS against a global reference.

Asset lineage (left to right):

```text
[external]                        [download]       [compute]              [upload]
ebi_pgs_catalog_reference_panel → reference_panel → reference_scores ──→ hf_prs_percentiles
ebi_pgs_catalog_scoring_files  ──────────────────↗                     ↗
                                  raw_pgs_metadata → cleaned_pgs_metadata
```

- **`ebi_pgs_catalog_reference_panel`** (SourceAsset): The remote reference panel tarball on the EBI PGS Catalog FTP server.
- **`ebi_pgs_catalog_scoring_files`** (SourceAsset): Remote PGS Catalog scoring files and metadata on the EBI FTP server.
- **`reference_panel`**: Downloads and extracts the reference panel binary files to local cache.
- **`reference_scores`**: Scores all PGS IDs against the reference panel in a single batch using `compute_reference_prs_batch()`. This replaces the previous partitioned approach (one Dagster partition per PGS ID + sensor orchestration). The batch function iterates in-process, tracks failures, and produces aggregated distributions.
- **`hf_prs_percentiles`**: Enriches the raw distribution statistics with cleaned PGS Catalog metadata (trait names, EFO terms, performance metrics like AUROC/OR/C-index, ancestry) via `enrich_distributions()`, then uploads the enriched parquet to HuggingFace (`just-dna-seq/prs-percentiles`). This creates a cross-pipeline dependency on `cleaned_pgs_metadata`, ensuring the published distributions parquet is self-contained.

The batch scoring approach was adopted because the polars engine scores each PGS ID in seconds (not minutes), and the expensive parts (pvar parsing, psam loading, allele offset cache) are shared across IDs within a single process. This eliminates the overhead of thousands of Dagster partitions and the complex sensor orchestration that was previously required.

### 2. Metadata Pipeline (`prs_pipeline.metadata_assets`)

This pipeline downloads, cleans, and publishes PGS Catalog metadata—the tables that describe *what* each Polygenic Score measures (trait, method, publication) and *how well* it performs.

Asset lineage (left to right):

```text
[download]          [compute]              [upload]
raw_pgs_metadata → cleaned_pgs_metadata → hf_polygenic_risk_scores
                                         ↘ hf_prs_percentiles (cross-pipeline)
```

- **`raw_pgs_metadata`**: Downloads three bulk metadata CSV sheets (scores, performance_metrics, evaluation_sample_sets) from the FTP server and saves them as Parquet. The FTP source URL is logged in output metadata.
- **`cleaned_pgs_metadata`**: Cleans and normalizes the raw metadata (genome builds, snake_case column names, metric parsing). Feeds into both `hf_polygenic_risk_scores` (metadata-only HF repo) and `hf_prs_percentiles` (enriched distributions).
- **`hf_polygenic_risk_scores`**: Uploads the cleaned parquets to HuggingFace (`just-dna-seq/polygenic_risk_scores`).

## Panel-Aware Naming

Distribution files are panel-aware. Each reference panel produces a separate distributions file:

| Panel | Filename in HuggingFace | Local cache path |
|-------|------------------------|-----------------|
| `1000g` | `data/1000g_distributions.parquet` | `<cache>/percentiles/1000g_distributions.parquet` |
| `hgdp_1kg` | `data/hgdp_1kg_distributions.parquet` | `<cache>/percentiles/hgdp_1kg_distributions.parquet` |

The panel is configured via the `PRS_PIPELINE_PANEL` environment variable (default: `1000g`).

## Jobs

All jobs include `hooks={resource_summary_hook}` for run-level resource aggregation.

| Job | Assets | Description |
|-----|--------|-------------|
| `full_pipeline` | `reference_panel`, `reference_scores`, `raw_pgs_metadata`, `cleaned_pgs_metadata`, `hf_prs_percentiles` | Full pipeline: download panel, score, download and clean metadata, enrich distributions, push. Auto-submitted by `run_pipeline_on_startup` sensor |
| `download_reference_data` | `reference_panel` | Download the reference panel from EBI FTP |
| `score_and_push` | `reference_scores`, `raw_pgs_metadata`, `cleaned_pgs_metadata`, `hf_prs_percentiles` | Batch-score, download/clean metadata, enrich, and push to HuggingFace |
| `metadata_pipeline` | `raw_pgs_metadata`, `cleaned_pgs_metadata`, `hf_polygenic_risk_scores` | End-to-end metadata pipeline |
| `clean_and_push_metadata` | `cleaned_pgs_metadata`, `hf_polygenic_risk_scores` | Re-clean and push when raw sheets are already cached |

## CLI Commands

The pipeline is operated via the `prs-pipeline` CLI (or `uv run pipeline` from the workspace root).

- **Launch the pipeline**:
  ```bash
  uv run pipeline launch
  ```
  Starts the Dagster dev server. An `AutomationConditionSensorDefinition` (with `default_status=RUNNING`) automatically materializes assets in dependency order: `reference_panel` (on_missing) → `reference_scores` (eager) → `hf_prs_percentiles` (eager). No manual triggering needed — the sensor evaluates every ~30 seconds.

  To test with a subset of PGS IDs:
  ```bash
  uv run pipeline launch --test 5
  uv run pipeline launch --test-ids PGS000001,PGS000013
  uv run pipeline launch --panel hgdp_1kg
  ```

- **Check scoring status**:
  ```bash
  uv run pipeline status
  uv run pipeline status --panel 1000g
  ```
  Reads the quality report parquet and shows per-status counts and failed IDs.

- **Clean up stuck runs**:
  ```bash
  uv run pipeline clean
  ```
  Cancels queued/stuck Dagster runs.

## Resource Tracking

Every compute-heavy asset is wrapped with `resource_tracker` from `prs_pipeline.runtime`, which uses `psutil` to capture:

| Metric | Description |
|--------|-------------|
| `duration_sec` | Wall-clock seconds for the tracked block |
| `cpu_percent` | CPU utilization during execution |
| `peak_memory_mb` | Maximum RSS (resident set size) in MB |
| `memory_delta_mb` | Change in RSS from start to end (positive = growth) |

These metrics are written to Dagster output metadata (visible in the asset materialization panel) and logged to the Dagster logger.

Every job has `hooks={resource_summary_hook}` which aggregates per-asset metrics at the end of each successful run, logging:
- Total duration across all assets
- Maximum peak memory (bottleneck identification)
- Average CPU
- Top 3 memory consumers

### Why this matters

The `reference_scores` asset can score 5,000+ PGS IDs in a single process, consuming significant memory. If the process gets OOM-killed, previously completed assets still have their metrics recorded in Dagster, giving a baseline for how much memory was consumed before the crash. Without resource tracking, an OOM crash leaves no diagnostic information.

### Usage

```python
from prs_pipeline.runtime import resource_tracker

@asset(group_name="compute")
def my_asset(context: AssetExecutionContext) -> Output[Path]:
    with resource_tracker("my_asset", context=context):
        # ... compute-heavy code ...
        pass
```

All jobs must include the hook:

```python
from prs_pipeline.utils import resource_summary_hook

my_job = define_asset_job(
    name="my_job",
    selection=["my_asset"],
    hooks={resource_summary_hook},
)
```

### Key files

| File | Purpose |
|------|---------|
| `prs_pipeline/runtime.py` | `ResourceReport` model, `resource_tracker` context manager |
| `prs_pipeline/utils.py` | `resource_summary_hook` (Dagster `@success_hook`) |

## Data Flow Principles

- **SourceAssets**: Any data Dagster observes but does not create (e.g., FTP files, reference tarballs) is explicitly modeled as a `SourceAsset`.
- **Batch over Partitions**: The `reference_scores` asset uses `compute_reference_prs_batch()` which iterates in-process rather than creating one Dagster run per PGS ID. Failures are tracked in the returned `BatchScoringResult` and persisted as a quality parquet.
- **Resource Configurations**: Environment settings (cache directories, HuggingFace tokens) are handled via Dagster Resources (`CacheDirResource`, `HuggingFaceResource`).
