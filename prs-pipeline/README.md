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
