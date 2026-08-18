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
4. **Never `AssetIn` filesystem side effects.** There is no Polars/UPath IO manager — the default FS IOManager pickles every `Output[...]` under `data/output/dagster/storage/`. Those pickles are wiped across sessions, so `AssetIn` on `Path`, path-dicts, DataFrames that also live on disk, or metadata lists fails with `DagsterExecutionLoadInputError`. Use `deps=[AssetDep(...)]` for lineage and reconstruct from `CacheDirResource` (the pattern `reference_allele_universe`, `reference_scores`, and all HF upload assets follow).
5. **Metadata and Observability**: Every asset logs rich metadata (row counts, file paths, variant match rates). This makes the Dagster UI a complete data catalog, not just a task runner.

## Pipeline Architecture

The pipeline is modeled as Software-Defined Assets and is divided into two primary flows.

### 1. Reference Panel Pipeline (`prs_pipeline.assets`)

This pipeline computes population-level Polygenic Risk Score (PRS) distributions using a reference panel and PGS Catalog scoring files. The output is a per-score, per-superpopulation percentile table that lets end-users compare their personal PRS against a global reference.

Asset lineage (left to right):

```text
[external]                        [download]                         [compute]                     [upload]
ebi_pgs_catalog_reference_panel    ebi_reference_panel_fingerprint → reference_panel ──────→ reference_scores ──→ hf_prs_percentiles
ebi_pgs_catalog_scoring_files  →   ebi_scoring_files_fingerprint  → scoring_files → scoring_files_parquet ↗       ↗
                                                                         raw_pgs_metadata → cleaned_pgs_metadata
```

- **`ebi_pgs_catalog_reference_panel`** (SourceAsset): The remote reference panel tarball on the EBI PGS Catalog FTP server.
- **`ebi_pgs_catalog_scoring_files`** (SourceAsset): Remote PGS Catalog scoring files and metadata on the EBI FTP server.
- **`ebi_reference_panel_fingerprint`**: Materialized remote fingerprint for the reference panel URL (HTTP metadata hash). Downstream assets depend on this, not directly on the SourceAsset.
- **`ebi_scoring_files_fingerprint`**: Materialized remote fingerprint for `pgs_scores_list.txt` (HTTP metadata + body hash). Used as a freshness dependency for scoring/metadata assets.
- **`scoring_files`**: Bulk-downloads all harmonized PGS scoring `.txt.gz` files from EBI FTP.
- **`scoring_files_parquet`**: Converts all downloaded `.txt.gz` scoring files to parquet caches with spec-driven schema overrides (`SCORING_FILE_SCHEMA` from `just_prs.scoring`) and zstd-9 compression. PGS Catalog header metadata is embedded as file-level metadata in each parquet. After verified conversion, the original `.txt.gz` is deleted to save disk space (~5.5 GB savings for the full catalog). Per-file failures are tracked without aborting the loop and written to `conversion_failures.parquet` for post-hoc error analysis. `reference_scores` depends on this asset.
- **`reference_panel`**: Downloads and extracts the reference panel binary files to local cache (`<cache>/reference_panel/...`). Returns `Output[Path]` for metadata only — downstream assets must **not** `AssetIn` that Path (the default FS IOManager pickle under `data/output/dagster/storage/reference_panel` is wiped across sessions and causes `DagsterExecutionLoadInputError`).
- **`reference_scores`**: Scores all PGS IDs against the reference panel in a single batch using `compute_reference_prs_batch()`. Declares `deps=[AssetDep("reference_panel")]` for lineage and reconstructs the panel path via `download_reference_panel()` / `reference_panel_dir()` from `CacheDirResource`. Reads from parquet caches produced by `scoring_files_parquet` (5-60x faster than decompressing `.txt.gz`). The batch function iterates in-process, tracks failures, and produces aggregated distributions.
- **`hf_prs_percentiles`**: Enriches the raw distribution statistics with cleaned PGS Catalog metadata (trait names, EFO terms, performance metrics like AUROC/OR/C-index, ancestry) via `enrich_distributions()`, then uploads the enriched parquet to HuggingFace (`just-dna-seq/prs-percentiles`). This creates a cross-pipeline dependency on `cleaned_pgs_metadata`, ensuring the published distributions parquet is self-contained.

The batch scoring approach was adopted because the polars engine scores each PGS ID in seconds (not minutes), and the expensive parts (pvar parsing, psam loading, allele offset cache) are shared across IDs within a single process. This eliminates the overhead of thousands of Dagster partitions and the complex sensor orchestration that was previously required.

### 2. Metadata Pipeline (`prs_pipeline.metadata_assets`)

This pipeline downloads, cleans, and publishes PGS Catalog metadata—the tables that describe *what* each Polygenic Score measures (trait, method, publication) and *how well* it performs.

Asset lineage (left to right):

```text
[download]          [compute]              [upload]
raw_pgs_metadata → cleaned_pgs_metadata → hf_pgs_catalog (+ scoring_files_parquet)
                                         ↘ hf_prs_percentiles (cross-pipeline)
```

- **`raw_pgs_metadata`**: Downloads three bulk metadata CSV sheets (scores, performance_metrics, evaluation_sample_sets) from the FTP server and saves them as Parquet. The FTP source URL is logged in output metadata.
- **`cleaned_pgs_metadata`**: Cleans and normalizes the raw metadata (genome builds, snake_case column names, metric parsing). Feeds into `hf_pgs_catalog` (combined metadata + scoring files) and `hf_prs_percentiles` (enriched distributions).
- **`hf_pgs_catalog`**: Uploads cleaned metadata and scoring file parquets to HuggingFace (`just-dna-seq/pgs-catalog`). The `just-prs` library pulls cleaned metadata from this repo on first use via `PRSCatalog`.

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
| `full_pipeline` | `reference_panel`, `scoring_files`, `scoring_files_parquet`, `reference_scores`, `raw_pgs_metadata`, `cleaned_pgs_metadata`, `hf_prs_percentiles` | Full pipeline: download panel + scoring files, convert to parquet, score, download and clean metadata, enrich distributions, push. Auto-submitted by `run_pipeline_on_startup` sensor |
| `download_reference_data` | `reference_panel` | Download the reference panel from EBI FTP |
| `score_and_push` | `scoring_files`, `scoring_files_parquet`, `reference_scores`, `raw_pgs_metadata`, `cleaned_pgs_metadata`, `hf_prs_percentiles` | Download scoring files, convert to parquet, batch-score, download/clean metadata, enrich, and push to HuggingFace |
| `reference_percentile_audit_job` | `reference_percentile_audit` | Audit cached or HuggingFace reference percentile distributions and write sidecars without recomputing reference scores |
| `public_sample_scores_job` | `public_sample_score_parts`, `public_sample_runtime_results`, `public_sample_canary_audit`, `hf_public_sample_runtime` | Memory-safe PGS-major scoring of caller-supplied `--vcf` genomes (unrestored then restored), one compaction + blocking completeness check, canary quarantine from unrestored public-wgs-pass-v1 rows, and runtime-only HF upload (`samples.parquet` / `runtime_results.parquet` / `runtime_manifest.json`). Scores ≥1M variants are singleton checkpoints; DuckDB joins them in `PRS_SCORING_JOIN_CHUNK_SIZE` slices (same idea as 1000G genotype chunks) and the worker recycles after each. Does not recompute 1000G scores or write evidence/final docs. A native worker death isolates the crashing PGS, records a failed row, and continues |
| `sample_score_evidence_job` | `sample_score_evidence`, `hf_sample_score_evidence` | Build catalog-level evidence tables (traits, papers, guidelines, actionability, trait_contexts, record_search_terms) from cleaned catalog cache + public guideline adapters and upload them. Does not score genomes and does not wait for `runtime_results` |
| `ld_proxy_pipeline` | `ld_proxy_table`, `hf_ld_proxy_table` | Build consumer-array LD proxy tables as one parquet per PGS ID. Full-catalog coverage is a resumable per-PGS batch with shared reference-panel setup, not one catalog-wide union table |
| `metadata_pipeline` | `raw_pgs_metadata`, `cleaned_pgs_metadata` | End-to-end metadata pipeline (download + clean; push via catalog_pipeline) |

## CLI Commands

The pipeline is operated via the `prs-pipeline` CLI (or `uv run pipeline` from the workspace root).

- **Launch the pipeline**:
  ```bash
  uv run pipeline launch
  ```
  Starts the Dagster dev server. The startup sensor (`run_pipeline_on_startup`) is a **bootstrap trigger**, not a freshness policy: it submits `full_pipeline` on startup (by default via `--run-now`) and avoids duplicate in-flight runs.

  For freshness in proper production pipelines, keep a **separate recompute sensor/schedule** that re-triggers when upstream lineage is newer than downstream outputs. Do not rely on "assets exist" checks alone.

  To only start the UI without submitting a startup run:
  ```bash
  uv run pipeline launch --no-run-now
  ```

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

- **Audit reference percentiles**:
  ```bash
  uv run pipeline audit
  uv run pipeline audit --headless
  uv run pipeline audit --test
  uv run pipeline sample-scores
  uv run pipeline sample-scores --vcf anton --vcf livia --vcf o-mom=/path/to/mom.vcf.gz
  uv run pipeline sample-scores --vcf anton --vcf livia --retry-failed
  uv run pipeline sample-scores --headless
  uv run pipeline canary-audit --vcf anton --vcf livia  # alias
  ```
  Launches the Dagster UI by default and submits `reference_percentile_audit_job`, which audits cached or HuggingFace-pulled `{panel}_distributions.parquet` plus `{panel}_quality.parquet` when available. It logs pass/warn/fail PGS-ID counts, writes `{panel}_distribution_quality_issues.parquet` and `{panel}_distribution_audit_summary.json`, and uploads those sidecars to HuggingFace when `HF_TOKEN` is available, all without recomputing reference scores.

  `uv run pipeline sample-scores --vcf ... --vcf ...` launches the Dagster UI and submits `public_sample_scores_job` (`canary-audit` is an alias). The 1000G `completeness_sensor` / `failure_retry_sensor` stay quiet while this CLI job is requested so they cannot submit `score_and_push` and steal the slot. Pass `--vcf` at least twice (path, alias, or `Label=path`). The job is four assets: PGS-major checkpoint parts (unrestored then restored, resumable atomic parquet parts), one compaction to `runtime_results.parquet` with a blocking completeness check, canary flags derived from unrestored `public-wgs-pass-v1` rows, and a runtime-only Hugging Face upload (`samples.parquet` / `runtime_results.parquet` / `runtime_manifest.json` plus catalog flags and percentile audit sidecars). Scores with ≥ `PRS_SAMPLE_SCORE_LARGE_VARIANT_THRESHOLD` variants (default 1M; the PGS005172–PGS005197 cluster is ~9.5M each) are planned as singleton checkpoints; `compute_prs_duckdb` joins them in `PRS_SCORING_JOIN_CHUNK_SIZE` slices and the worker recycles after each so a 10-wide batch of genome-wide scores cannot native-crash. Scoring workers emit a Dagster log line after each checkpoint (`Sample scores {profile}: N/M PGS …`), forwarded from the subprocess so the compute log is not empty. Unpublished `--vcf` labels are logged only when present. It does **not** recompute 1000G reference scores, upload evidence tables, or write the combined root `manifest.json`. Use `--pgs-ids` / `--limit` for a pilot; `--retry-failed` to re-score failed checkpoint rows while keeping successful cache and continuing into missing IDs; `--no-cache` to wipe parts and rescore everything.

- **Publish catalog evidence (no scoring)**:
  ```bash
  uv run pipeline evidence
  uv run pipeline evidence --headless
  uv run pipeline evidence --offline
  ```
  Launches the Dagster UI by default and submits `sample_score_evidence_job`. Reads cleaned catalog cache plus public guideline adapters (USPSTF, ClinGen, WHO/CDC, NICE public pages) and writes traits / papers / guidelines / three-status actionability / extra-clinical contexts. Uploads those tables plus schema-generated root `README.md` and `AGENTS.md` to `just-dna-seq/prs-sample-scores`. Skips missing `runtime_results.parquet`. Does **not** invent sample scores. `--offline` sets `PRS_EVIDENCE_ALLOW_NETWORK=0`.

- **Build LD proxy tables for consumer arrays**:
  ```bash
  uv run pipeline ld-proxy --pgs-ids PGS000001
  uv run pipeline ld-proxy --limit 5
  uv run pipeline ld-proxy --full-catalog
  ```
  Launches the Dagster UI by default and submits `ld_proxy_pipeline`. The output layout is `<cache>/percentiles/ld_proxy/{panel}/{chip}/{build}/{pgs_id}.parquet`, mirrored on HuggingFace as `data/ld_proxy/{panel}/{chip}/{build}/{pgs_id}.parquet`. `--full-catalog` is the eventual publishing path, but it still runs as a resumable loop over per-PGS files with mtime-aware `skip_existing`, per-ID failure isolation, and a `_quality.parquet` sidecar. Use OS memory containment such as `systemd-run --user --scope -p MemoryMax=24G ...` for long full-catalog runs.

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

End-of-asset resource metrics are not a substitute for in-loop progress. Hours-long jobs (`public_sample_scores_job`, `reference_scores`) must emit `context.log.info` every `PRS_PIPELINE_PROGRESS_EVERY` items (default 10) with done/total, percent, current ID span, and timing. If scoring runs in a subprocess, the parent must forward flushed stdout into that logger; otherwise the Dagster compute log stays empty.

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

- **SourceAssets + Fingerprints**: External origins are modeled as `SourceAsset`s for provenance, while downstream freshness dependencies use materialized fingerprint assets to avoid "missing forever" behavior.
- **Batch over Partitions**: The `reference_scores` asset uses `compute_reference_prs_batch()` which iterates in-process rather than creating one Dagster run per PGS ID. Failures are tracked in the returned `BatchScoringResult` and persisted as a quality parquet.
- **Resource Configurations**: Environment settings (cache directories, HuggingFace tokens) are handled via Dagster Resources (`CacheDirResource`, `HuggingFaceResource`).
- **Freshness over Presence**: "Materialized" does not imply "up-to-date." If upstream assets are newer, downstream compute/upload assets must be recomputed by sensor/schedule policy.
