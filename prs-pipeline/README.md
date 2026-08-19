# prs-pipeline

Dagster pipeline for computing PRS reference distributions from the 1000 Genomes reference panel.

## Overview

This pipeline downloads the PGS Catalog 1000G reference panel (~7 GB), computes polygenic risk scores
for all 2,504 reference individuals across all PGS Catalog scores, aggregates per-superpopulation
distribution statistics, and pushes `reference_distributions.parquet` to HuggingFace
(`just-dna-seq/prs-percentiles`).

End users of `just-prs` automatically pull this tiny parquet via `PRSCatalog.reference_distributions()`.

## Running

```bash
cd prs-pipeline
uv run dagster dev -m prs_pipeline.definitions
```

Then open http://localhost:3000 in your browser.

## Assets

### Scoring & Distribution Pipeline

| Asset | Group | Description |
|-------|-------|-------------|
| `ebi_reference_panel_fingerprint` | download | HTTP fingerprint for freshness tracking of the remote reference panel |
| `ebi_scoring_files_fingerprint` | download | HTTP fingerprint for the remote scoring file manifest |
| `scoring_files` | download | Bulk-download all harmonized PGS scoring `.txt.gz` files from EBI FTP |
| `scoring_files_parquet` | compute | Convert all `.txt.gz` scoring files to spec-driven parquet caches (zstd-9, embedded headers). Deletes `.txt.gz` after verified conversion to save ~5.5 GB disk space. Tracks per-file failures in `conversion_failures.parquet` |
| `reference_panel` | download | Download + extract reference panel binary files (.pgen/.pvar/.psam) |
| `reference_scores` | compute | Score all PGS IDs against the reference panel via `compute_reference_prs_batch()` |
| `reference_percentile_audit` | compute | Audit cached or HuggingFace reference percentile parquets, log pass/warn/fail counts, and write/upload quality sidecars without recomputing scores |
| `public_sample_score_parts` | compute | PGS-major unrestored then restored scoring of `--vcf` genomes into atomic checkpoint parts. Scores ≥1M variants are singleton checkpoints; DuckDB joins them in bounded slices; workers recycle at the env memory budget or after a large score. Does not recompute 1000G scores |
| `public_sample_runtime_results` | compute | Compact checkpoint parts once into `runtime_results.parquet` and enforce the complete sample×PGS×profile outcome matrix |
| `public_sample_canary_audit` | compute | Derive public canary rows from unrestored `public-wgs-pass-v1` runtime; score unknown/private `--vcf` separately; write catalog flags and percentile audit sidecars |
| `hf_public_sample_runtime` | upload | Atomic allowlisted commit of runtime-owned files (`samples.parquet`, `runtime_results.parquet`, `runtime_manifest.json`, `sample_ancestry.parquet`) plus flags/sidecars. Never uploads evidence, root docs, identity cache, or checkpoint parts |
| `sample_score_evidence` | compute | Catalog-level evidence tables from one pinned catalog snapshot. Writes `evidence_manifest.json`. Does not score genomes or write final docs |
| `hf_sample_score_evidence` | upload | Atomic allowlisted commit of the nine evidence parquets plus `evidence_manifest.json`. Never uploads runtime scores or root README/AGENTS |
| `sample_score_integration` | compute | Stage pinned runtime/evidence/catalog/percentile snapshots and build `model_analysis.parquet` plus ancestry-selected `trait_summaries.parquet` and final docs. Does not rescore genomes |
| `hf_sample_score_dataset` | upload | Atomic six-path commit of model_analysis, trait_summaries, final manifest.json, README.md, AGENTS.md, and ANALYSIS.md |
| `hf_prs_percentiles` | upload | Enrich distributions with metadata and absolute risk, push to HuggingFace |

### Metadata & Prevalence Pipeline

| Asset | Group | Description |
|-------|-------|-------------|
| `raw_pgs_metadata` | download | Download PGS Catalog bulk metadata sheets (scores, performance, evaluation, publications) |
| `cleaned_pgs_metadata` | compute | Clean and normalize metadata, produce parquets including `publications.parquet` |
| `gwas_studies` | download | Download GWAS Catalog bulk studies + trait mappings, parse case/control from free-text |
| `trait_prevalence` | compute | Merge 3-tier prevalence data (seed CSV → GWAS cohorts → PGS eval cohorts) into `trait_prevalence.parquet` |
| `hf_pgs_catalog` | upload | Push cleaned metadata + prevalence to HuggingFace |

For details on how absolute risk estimation works, see the [methodology document](../docs/absolute-risk-methodology.md).
