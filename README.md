# just-prs

[![PyPI version](https://badge.fury.io/py/just-prs.svg)](https://badge.fury.io/py/just-prs)

A [Polars](https://pola.rs/)-bio based tool to compute **Polygenic Risk Scores (PRS)** from the [PGS Catalog](https://www.pgscatalog.org/).

## Web UI

An interactive [Reflex](https://reflex.dev/) web application for browsing PGS Catalog data and computing PRS scores.

![PRS Compute UI — upload VCF, select scores, compute PRS](images/PRS_screenshot.jpg)

### Setup

```bash
# From the workspace root — install all packages first
uv sync --all-packages

# Then launch the UI
cd prs-ui
uv run reflex run
```

The UI opens at http://localhost:3000 with three tabs:

### Compute PRS (default tab)

1. **Upload a VCF** — drag-and-drop or browse; genome build is auto-detected from `##reference` and `##contig` headers
2. **Load Scores** — fetches PGS Catalog scores metadata, pre-filtered by detected (or manually selected) genome build
3. **Select scores** — use checkboxes to pick individual scores, or "Select Filtered" to select everything matching the current filter
4. **Compute** — click **Compute PRS** to run PRS for each selected score. Results show PRS score, match rate, matched/total variants, effect sizes, and classification metrics

### Metadata Sheets

Browse all 7 PGS Catalog metadata sheets in a MUI DataGrid with filtering and sorting. Select rows and download their scoring files with **Download Selected**.

### Scoring File

Stream any harmonized scoring file by PGS ID directly from EBI FTP and view it in the grid.

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `PRS_CACHE_DIR` | `~/.cache/just-prs` | Root directory for cached metadata and scoring files |

## Features

- **`PRSCatalog`** — search scores, compute PRS, and estimate percentiles using cleaned bulk metadata (no REST API calls needed)
- **Cleanup pipeline** — normalizes genome builds, renames columns to snake_case, parses performance metrics into structured numeric fields
- **HuggingFace sync** — cleaned metadata parquets published to [just-dna-seq/polygenic_risk_scores](https://huggingface.co/datasets/just-dna-seq/polygenic_risk_scores) and auto-downloaded on first use
- **Bulk download** the entire PGS Catalog metadata (~5,000+ scores) via EBI FTP
- Compute PRS for one or many scores against a VCF file
- All data saved as **Parquet** for fast downstream analysis with Polars
- [Validated against PLINK2](docs/validation.md) with floating-point precision agreement

## Installation

Requires Python >= 3.14. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

**From PyPI:**

```bash
pip install just-prs
```

**From source:**

```bash
git clone https://github.com/antonkulaga/just-prs
cd just-prs
uv sync
```

For the optional web UI: `pip install just-prs[ui]` or `uv sync --all-packages` when developing from source.

The CLI is available as both `just-prs` and `prs`.

## Quick Start

### CLI

```bash
# Compute PRS for a single score
prs compute --vcf sample.vcf.gz --pgs-id PGS000001

# Multiple scores at once
prs compute --vcf sample.vcf.gz --pgs-id PGS000001,PGS000002,PGS000003

# Search the catalog
prs catalog scores search --term "breast cancer"
```

### Python

```python
from just_prs import PRSCatalog
from pathlib import Path

catalog = PRSCatalog()

# Search for scores
results = catalog.search("type 2 diabetes", genome_build="GRCh38").collect()

# Compute PRS
result = catalog.compute_prs(vcf_path=Path("sample.vcf.gz"), pgs_id="PGS000001")
print(f"Score: {result.score:.6f}, Match rate: {result.match_rate:.1%}")

# Batch computation
results = catalog.compute_prs_batch(
    vcf_path=Path("sample.vcf.gz"),
    pgs_ids=["PGS000001", "PGS000002", "PGS000003"],
)

# Percentile estimation
pct = catalog.percentile(prs_score=result.score, pgs_id="PGS000001")
```

## Documentation

- [CLI Reference](docs/cli.md) — full command-line usage for `prs compute`, `prs catalog`, and bulk downloads
- [Python API](docs/python-api.md) — `PRSCatalog`, FTP downloads, REST client, cleanup pipeline, HuggingFace sync
- [PLINK2 Validation](docs/validation.md) — accuracy benchmarks against PLINK2 `--score`
- [Cleanup Pipeline](docs/cleanup-pipeline.md) — genome build normalization, column renaming, metric parsing

## Data sources

- PGS Catalog REST API: <https://www.pgscatalog.org/rest/>
- EBI FTP bulk downloads: <https://ftp.ebi.ac.uk/pub/databases/spot/pgs/>
- PGS Catalog download documentation: <https://www.pgscatalog.org/downloads/>
- Cleaned metadata parquets on HuggingFace: <https://huggingface.co/datasets/just-dna-seq/polygenic_risk_scores>
