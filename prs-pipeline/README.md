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

| Asset | Description |
|-------|-------------|
| `pgs_id_partitions` | Discover all PGS IDs and register as dynamic partitions |
| `reference_panel` | Download + extract `pgsc_1000G_v1.tar.zst` (~7 GB, once) |
| `per_pgs_scores` | PLINK2 `--score` on reference panel for each PGS ID (partitioned) |
| `reference_distributions` | Aggregate per-individual scores to per-superpopulation stats |
| `push_distributions_to_hf` | Upload to `just-dna-seq/prs-percentiles` on HuggingFace |
