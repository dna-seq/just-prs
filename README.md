# just-prs

[![PyPI version](https://badge.fury.io/py/just-prs.svg)](https://badge.fury.io/py/just-prs)

A [Polars](https://pola.rs/)-bio based tool to compute **Polygenic Risk Scores (PRS)** from the [PGS Catalog](https://www.pgscatalog.org/).

## Web UI

An interactive [Reflex](https://reflex.dev/) web application for browsing PGS Catalog data and computing PRS scores.

![PRS Compute UI — upload VCF, select scores, compute PRS](images/PRS_screenshot.jpg)

### Setup

```bash
# From the workspace root — install all packages (including prs-ui)
uv sync --all-packages

# Launch the UI (shortcut defined in pyproject.toml)
uv run ui

# Or equivalently, from the prs-ui directory:
cd prs-ui
uv run reflex run
```

The UI opens at http://localhost:3000 with three tabs:

### Compute PRS (default tab)

1. **Upload a VCF** — drag-and-drop or browse; genome build is auto-detected from `##reference` and `##contig` headers. VCF is normalized (chr prefix stripped, genotype computed, quality filtered) with a visible progress bar and a green callout on completion showing variant count
2. **Load Scores** — fetches PGS Catalog scores metadata, pre-filtered by detected (or manually selected) genome build
3. **Select scores** — use checkboxes to pick individual scores, or "Select Filtered" to select everything matching the current filter
4. **Compute** — click **Compute PRS** to run PRS for each selected score. A progress bar tracks completion across scores. Results table shows PRS score, AUROC (model accuracy), quality assessment, evaluation population/ancestry, match rate, matched/total variants, and effect sizes. Each result includes an interpretation card with a plain-English summary of model quality
5. **Download CSV** — export all computed results to a CSV file via the **Download CSV** button above the results table

### Metadata Sheets

Browse all 7 PGS Catalog metadata sheets in a MUI DataGrid with filtering and sorting. Select rows and download their scoring files with **Download Selected**.

### Scoring File

Stream any harmonized scoring file by PGS ID directly from EBI FTP and view it in the grid.

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `PRS_CACHE_DIR` | OS-dependent (via `platformdirs`) | Root directory for cached metadata and scoring files |

## Features

- **`PRSCatalog`** — search scores, compute PRS, and look up evaluation performance using cleaned bulk metadata (no REST API calls needed)
- **VCF normalization** — `normalize_vcf()` strips chr prefix, renames id→rsid, computes genotype from GT, applies configurable quality filters (FILTER, DP, QUAL), warns on chrY for females, and writes zstd-compressed Parquet
- **Quality assessment** — PRS results include AUROC-based model quality labels, effect sizes (OR/HR/Beta), classification metrics (AUROC/C-index), and plain-English interpretation summaries
- **CSV export** — download computed PRS results as CSV from the web UI or programmatically
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

# Normalize a VCF to Parquet (strip chr prefix, compute genotype, quality filter)
prs normalize --vcf sample.vcf.gz --pass-filters "PASS,." --min-depth 10

# Search the catalog
prs catalog scores search --term "breast cancer"
```

### Python

```python
from just_prs import PRSCatalog, normalize_vcf, VcfFilterConfig
from pathlib import Path

catalog = PRSCatalog()

# Normalize a VCF to Parquet (strip chr prefix, compute genotype, quality filters)
config = VcfFilterConfig(pass_filters=["PASS", "."], min_depth=10)
normalize_vcf(Path("sample.vcf.gz"), Path("sample.parquet"), config=config)

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

# Look up best evaluation performance for a score
best = catalog.best_performance(pgs_id="PGS000001").collect()
```

## Testing

The project includes an extensive integration test suite that runs against real genomic data and external tools -- no mocked data or synthetic fixtures. All tests are reproducible on any Linux, macOS, or Windows machine.

```bash
uv run pytest tests/ -v
```

| Test suite | What it validates | Data source |
|---|---|---|
| `test_plink.py` | PRS scores match [PLINK2](https://www.cog-genomics.org/plink/2.0/) `--score` within floating-point precision for 5 GRCh38 scores | Real whole-genome VCF from Zenodo; PLINK2 auto-downloaded |
| `test_percentile.py` | Theoretical mean/SD from allele frequencies, percentile computation, and cross-validation against PLINK2 for 5 scores with allele frequency data | Real PGS scoring files with `allelefrequency_effect` |
| `test_prs.py` | End-to-end PRS computation (single and batch) on a real VCF | Zenodo test VCF |
| `test_cleanup.py` | Full cleanup pipeline: column renaming, genome build normalization, metric string parsing, performance flattening, `PRSCatalog` search/percentile on live catalog data | Real PGS Catalog bulk metadata (~5,000+ scores) via EBI FTP |
| `test_scoring.py` | Scoring file download, parsing, and caching | Real PGS000001 harmonized scoring file |
| `test_catalog.py` | REST API client: score lookup, trait search, download URL resolution | Live PGS Catalog REST API |

Key properties of the test suite:

- **PLINK2 cross-validation** -- scores are compared against the gold-standard PLINK2 `--score` command with relative differences below 5e-7 ([details](docs/validation.md))
- **Real data throughout** -- test VCF auto-downloaded from Zenodo, PLINK2 binary auto-downloaded for the host platform, scoring files fetched from EBI FTP
- **Percentile verification** -- theoretical statistics computed from allele frequencies are validated against manual row-by-row computation, and percentiles are checked for mathematical consistency (CDF symmetry, known quantiles)
- **No mocking** -- all tests run real pipelines against real data to catch integration issues

## Documentation

- [CLI Reference](docs/cli.md) — full command-line usage for `prs compute`, `prs normalize`, `prs catalog`, and bulk downloads
- [Python API](docs/python-api.md) — `PRSCatalog`, VCF normalization, FTP downloads, REST client, cleanup pipeline, HuggingFace sync
- [PLINK2 Validation](docs/validation.md) — accuracy benchmarks against PLINK2 `--score`
- [Cleanup Pipeline](docs/cleanup-pipeline.md) — genome build normalization, column renaming, metric parsing

## Data sources

- PGS Catalog REST API: <https://www.pgscatalog.org/rest/>
- EBI FTP bulk downloads: <https://ftp.ebi.ac.uk/pub/databases/spot/pgs/>
- PGS Catalog download documentation: <https://www.pgscatalog.org/downloads/>
- Cleaned metadata parquets on HuggingFace: <https://huggingface.co/datasets/just-dna-seq/polygenic_risk_scores>
