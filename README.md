# just-prs

[![PyPI version](https://badge.fury.io/py/just-prs.svg)](https://badge.fury.io/py/just-prs)

A [Polars](https://pola.rs/)-bio based tool to compute **Polygenic Risk Scores (PRS)** from the [PGS Catalog](https://www.pgscatalog.org/).

## Features

- **`PRSCatalog`** — high-level class for searching scores, computing PRS, and estimating percentiles using cleaned bulk metadata (no REST API calls needed)
- **Cleanup pipeline** — normalizes genome builds (hg19/hg38/NCBI36 → GRCh37/GRCh38/GRCh36), renames columns to snake_case, parses performance metric strings into structured numeric fields
- **HuggingFace sync** — cleaned metadata parquets are published to [just-dna-seq/polygenic_risk_scores](https://huggingface.co/datasets/just-dna-seq/polygenic_risk_scores) and auto-downloaded on first use
- Compute PRS for one or many scores against a VCF file
- Search and inspect PGS Catalog scores and traits via the REST API
- **Bulk download** the entire PGS Catalog metadata (all ~5,000+ scores) via EBI FTP — one HTTP request per sheet, not hundreds of API pages
- Stream harmonized scoring files directly from EBI FTP without storing intermediate `.gz` files
- All data saved as **Parquet** for fast, efficient downstream analysis with Polars

## Validation against PLINK2

Our PRS computation is validated against [PLINK2](https://www.cog-genomics.org/plink/2.0/) `--score` on real genomic data. The integration test suite downloads a whole-genome VCF from Zenodo, computes PRS for multiple GRCh38 scores using both `just-prs` and PLINK2, and asserts agreement:

| PGS ID | just-prs | PLINK2 | Relative diff | Variants matched |
|--------|----------|--------|---------------|-----------------|
| PGS000001 | 0.030123 | 0.030123 | 6.5e-7 | 51 / 77 |
| PGS000002 | -0.137089 | -0.137089 | 1.1e-7 | 51 / 77 |
| PGS000003 | 0.588127 | 0.588127 | 8.1e-9 | 51 / 77 |
| PGS000004 | -0.7158 | -0.7158 | 3.1e-16 | 170 / 313 |
| PGS000005 | -0.8903 | -0.8903 | 5.0e-16 | 170 / 313 |

All differences are within floating-point precision. PLINK2 is auto-downloaded if not already installed, so the tests run on any Linux, macOS, or Windows machine:

```bash
uv run pytest tests/test_plink.py -v
```

## Installation

Requires Python ≥ 3.14. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

**From PyPI:**

```bash
uv add just-prs
# or
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

## CLI Reference

### Top-level commands

```
prs --help
prs compute --help
prs catalog --help
```

---

### `prs compute` — Compute PRS for a VCF

```bash
prs compute --vcf sample.vcf.gz --pgs-id PGS000001
prs compute --vcf sample.vcf.gz --pgs-id PGS000001,PGS000002,PGS000003
prs compute --vcf sample.vcf.gz --pgs-id PGS000001 --build GRCh37 --output results.json
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--vcf / -v` | — | Path to VCF file (required) |
| `--pgs-id / -p` | — | Comma-separated PGS ID(s) (required) |
| `--build / -b` | `GRCh38` | Genome build |
| `--cache-dir` | `~/.cache/just-prs/scores` | Cache directory for scoring files |
| `--output / -o` | — | Save results as JSON |

---

### `prs catalog scores` — Search and inspect scores (REST API)

```bash
prs catalog scores list                        # first 100 scores
prs catalog scores list --all                  # every score in catalog
prs catalog scores search --term "breast cancer"
prs catalog scores info PGS000001
```

### `prs catalog traits` — Search and inspect traits (REST API)

```bash
prs catalog traits search --term "diabetes"
prs catalog traits info EFO_0001645
```

### `prs catalog download` — Download a single scoring file

Downloads the harmonized `.txt.gz` scoring file for one score and caches it locally.

```bash
prs catalog download PGS000001
prs catalog download PGS000001 --output-dir ./my_scores --build GRCh37
```

---

### `prs catalog bulk` — Bulk FTP downloads (fast, parquet output)

These commands use the [EBI FTP HTTPS mirror](https://ftp.ebi.ac.uk/pub/databases/spot/pgs/) via **fsspec** to download pre-built catalog-wide files directly — far faster than paginating the REST API.

#### `prs catalog bulk metadata` — All catalog metadata as parquet

Downloads the PGS Catalog bulk metadata CSVs and converts each to a parquet file.
The full catalog (~5,000+ scores) downloads in seconds as a single HTTP request per sheet.

```bash
# Download all 7 metadata sheets → ./output/pgs_metadata/*.parquet
prs catalog bulk metadata

# Download only the scores sheet
prs catalog bulk metadata --sheet scores

# Specify output directory; force re-download
prs catalog bulk metadata --output-dir /data/pgs --overwrite
```

Available sheets:

| Sheet | Contents |
|-------|----------|
| `scores` | All PGS scores and their metadata |
| `publications` | Publication sources for each PGS |
| `efo_traits` | Ontology-mapped trait information |
| `score_development_samples` | GWAS and training samples |
| `performance_metrics` | Evaluation performance metrics |
| `evaluation_sample_sets` | Evaluation sample set descriptions |
| `cohorts` | Cohort information |

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir / -o` | `./output/pgs_metadata` | Directory for parquet output |
| `--sheet / -s` | all sheets | Single sheet name to download |
| `--overwrite` | `False` | Re-download existing files |

#### `prs catalog bulk scores` — All scoring files as parquet

Streams each harmonized scoring file from EBI FTP and saves it as a parquet file
(with an added `pgs_id` column). No intermediate `.gz` files are written to disk.

```bash
# Download ALL ~5,000+ scoring files (GRCh38) → ./output/pgs_scores/PGS######.parquet
prs catalog bulk scores

# Download a specific subset
prs catalog bulk scores --ids PGS000001,PGS000002,PGS000003

# GRCh37 build, custom output dir
prs catalog bulk scores --build GRCh37 --output-dir /data/scores

# Force re-download of existing files
prs catalog bulk scores --ids PGS000001 --overwrite
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir / -o` | `./output/pgs_scores` | Directory for parquet output |
| `--build / -b` | `GRCh38` | Genome build (`GRCh37` or `GRCh38`) |
| `--ids` | all | Comma-separated PGS IDs to download |
| `--overwrite` | `False` | Re-download existing parquet files |

#### `prs catalog bulk clean-metadata` — Build cleaned metadata parquets

Downloads raw metadata from EBI FTP, runs the cleanup pipeline (genome build normalization, column renaming, metric parsing, performance flattening), and saves three cleaned parquet files.

```bash
# Build cleaned parquets → ./output/pgs_metadata/
prs catalog bulk clean-metadata

# Custom output directory
prs catalog bulk clean-metadata --output-dir /data/cleaned
```

Output files:

| File | Contents |
|------|----------|
| `scores.parquet` | All PGS scores with snake_case columns, normalized genome builds |
| `performance.parquet` | Performance metrics joined with evaluation samples, parsed numeric columns |
| `best_performance.parquet` | One best row per PGS ID (largest sample, European-preferred) |

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir / -o` | `./output/pgs_metadata` | Directory for cleaned parquet output |

#### `prs catalog bulk push-hf` — Push cleaned parquets to HuggingFace

Uploads cleaned metadata parquets to a HuggingFace dataset repository. Builds them first if not already present. Token is read from `.env` file or `HF_TOKEN` environment variable.

```bash
# Push to default repo (just-dna-seq/polygenic_risk_scores)
prs catalog bulk push-hf

# Push from a custom directory to a custom repo
prs catalog bulk push-hf --output-dir /data/cleaned --repo my-org/my-dataset
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir / -o` | `./output/pgs_metadata` | Directory containing cleaned parquets |
| `--repo / -r` | `just-dna-seq/polygenic_risk_scores` | HuggingFace dataset repo ID |

#### `prs catalog bulk pull-hf` — Pull cleaned parquets from HuggingFace

Downloads cleaned metadata parquets from a HuggingFace dataset repository. Useful for bootstrapping a local cache without running the cleanup pipeline.

```bash
# Pull to default directory
prs catalog bulk pull-hf

# Pull to custom directory from custom repo
prs catalog bulk pull-hf --output-dir /data/cleaned --repo my-org/my-dataset
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir / -o` | `./output/pgs_metadata` | Directory to save pulled parquets |
| `--repo / -r` | `just-dna-seq/polygenic_risk_scores` | HuggingFace dataset repo ID |

#### `prs catalog bulk ids` — List all PGS IDs

Fetches `pgs_scores_list.txt` from EBI FTP (one request) and prints every PGS ID.

```bash
prs catalog bulk ids
prs catalog bulk ids | wc -l    # count total scores
```

---

## Python API

### Bulk FTP downloads (`just_prs.ftp`)

```python
from just_prs.ftp import (
    list_all_pgs_ids,
    download_metadata_sheet,
    download_all_metadata,
    stream_scoring_file,
    download_scoring_as_parquet,
    bulk_download_scoring_parquets,
)
from pathlib import Path

# Full ID list in one request
ids = list_all_pgs_ids()  # ['PGS000001', 'PGS000002', ...]

# All score metadata as a Polars DataFrame, saved to parquet
df = download_metadata_sheet("scores", Path("./output/pgs_metadata/scores.parquet"))

# All 7 sheets at once
sheets = download_all_metadata(Path("./output/pgs_metadata"))

# Stream a scoring file as a LazyFrame (no local .gz written)
lf = stream_scoring_file("PGS000001", genome_build="GRCh38")

# Download one scoring file as parquet (adds pgs_id column)
path = download_scoring_as_parquet("PGS000001", Path("./output/pgs_scores"))

# Bulk download a list (or all) scoring files as parquet
paths = bulk_download_scoring_parquets(Path("./output/pgs_scores"), pgs_ids=["PGS000001", "PGS000002"])
paths = bulk_download_scoring_parquets(Path("./output/pgs_scores"))  # all ~5000+
```

### REST API client (`just_prs.catalog`)

```python
from just_prs.catalog import PGSCatalogClient

with PGSCatalogClient() as client:
    score = client.get_score("PGS000001")
    results = client.search_scores("breast cancer", limit=10)
    trait = client.get_trait("EFO_0001645")
    for score in client.iter_all_scores(page_size=100):
        print(score.id, score.trait_reported)
```

### PRSCatalog — search, compute, and percentile (`just_prs.prs_catalog`)

`PRSCatalog` is the recommended high-level interface. It persists 3 cleaned parquet files locally and loads them on access using a 3-tier fallback chain: local files -> HuggingFace pull -> raw FTP download + cleanup. All lookups, searches, and PRS computations use cleaned data with no per-score REST API calls.

```python
from just_prs import PRSCatalog

catalog = PRSCatalog()  # uses ~/.cache/just-prs by default

# Browse cleaned scores (genome builds normalized, snake_case columns)
scores_df = catalog.scores(genome_build="GRCh38").collect()

# Search across pgs_id, name, trait_reported, and trait_efo
results = catalog.search("breast cancer", genome_build="GRCh38").collect()

# Get cleaned metadata for a single score
info = catalog.score_info_row("PGS000001")  # dict or None

# Best performance metric per score (largest sample, European-preferred)
best = catalog.best_performance(pgs_id="PGS000001").collect()

# Compute PRS (trait lookup from cached metadata, not REST API)
result = catalog.compute_prs(vcf_path="sample.vcf.gz", pgs_id="PGS000001")
print(result.score, result.match_rate)

# Batch computation
results = catalog.compute_prs_batch(
    vcf_path="sample.vcf.gz",
    pgs_ids=["PGS000001", "PGS000002"],
)

# Percentile estimation (AUROC-based or explicit mean/std)
pct = catalog.percentile(prs_score=1.5, pgs_id="PGS000014")
pct = catalog.percentile(prs_score=1.5, pgs_id="PGS000014", mean=0.0, std=1.0)

# Build cleaned parquets explicitly (download from FTP + cleanup)
paths = catalog.build_cleaned_parquets(output_dir=Path("./output/pgs_metadata"))
# {'scores': Path('output/pgs_metadata/scores.parquet'), 'performance': ..., 'best_performance': ...}

# Push cleaned parquets to HuggingFace
catalog.push_to_hf()  # token from .env / HF_TOKEN
catalog.push_to_hf(token="hf_...", repo_id="my-org/my-dataset")
```

### HuggingFace sync (`just_prs.hf`)

```python
from just_prs.hf import push_cleaned_parquets, pull_cleaned_parquets
from pathlib import Path

# Push cleaned parquets to HF dataset repo
push_cleaned_parquets(Path("./output/pgs_metadata"))  # default: just-dna-seq/polygenic_risk_scores

# Pull cleaned parquets from HF
downloaded = pull_cleaned_parquets(Path("./local_cache"))
# [Path('local_cache/scores.parquet'), Path('local_cache/performance.parquet'), ...]
```

### Cleanup pipeline (`just_prs.cleanup`)

The cleanup functions can be used independently of `PRSCatalog`:

```python
from just_prs.cleanup import clean_scores, clean_performance_metrics, parse_metric_string
from just_prs.ftp import download_metadata_sheet
from pathlib import Path

# Clean scores: rename columns, normalize genome builds
raw_df = download_metadata_sheet("scores", Path("./output/pgs_metadata/scores_raw.parquet"))
cleaned_lf = clean_scores(raw_df)  # LazyFrame with snake_case columns

# Parse a metric string
parse_metric_string("1.55 [1.52,1.58]")
# {'estimate': 1.55, 'ci_lower': 1.52, 'ci_upper': 1.58, 'se': None}

# Clean performance metrics: parse strings, join with evaluation samples
perf_df = download_metadata_sheet("performance_metrics", Path("./output/pgs_metadata/perf.parquet"))
eval_df = download_metadata_sheet("evaluation_sample_sets", Path("./output/pgs_metadata/eval.parquet"))
cleaned_perf_lf = clean_performance_metrics(perf_df, eval_df)
```

### Low-level PRS computation (`just_prs.prs`)

```python
from pathlib import Path
from just_prs.prs import compute_prs, compute_prs_batch

result = compute_prs(
    vcf_path=Path("sample.vcf.gz"),
    scoring_file="PGS000001",   # PGS ID, local path, or LazyFrame
    genome_build="GRCh38",
)
print(result.score, result.match_rate)

results = compute_prs_batch(
    vcf_path=Path("sample.vcf.gz"),
    pgs_ids=["PGS000001", "PGS000002"],
)
```

## Web UI (`prs-ui`)

An interactive [Reflex](https://reflex.dev/) web application for browsing PGS Catalog data and computing PRS scores.

![PRS Compute UI — upload VCF, select scores, compute PRS](images/PRS_screenshot.jpg)

### Running the UI

```bash
# From the workspace root — install all packages first
uv sync --all-packages

# Then launch the UI
cd prs-ui
uv run reflex run
```

The UI opens at http://localhost:3000 and has three tabs:

### Compute PRS (default tab)

End-to-end PRS computation workflow:

1. **Upload a VCF** — drag-and-drop or browse; genome build is auto-detected from `##reference` and `##contig` headers
2. **Load Scores** — fetches the PGS Catalog scores metadata, pre-filtered by the detected (or manually selected) genome build. Scores are shown in a searchable table
3. **Select scores** — use checkboxes to pick individual scores, or "Select Filtered" to select everything matching the current filter
4. **Compute** — click **Compute PRS** to run PRS for each selected score against the uploaded VCF. Results show PRS score, match rate, matched/total variants, effect sizes, and classification metrics from PGS Catalog evaluation studies

### Metadata Sheets

Browse all 7 PGS Catalog metadata sheets (Scores, Publications, EFO Traits, etc.) in a MUI DataGrid with server-side filtering and sorting. Select rows with checkboxes and download their scoring files to the local cache with a single **Download Selected** button.

### Scoring File

Stream any harmonized scoring file by PGS ID directly from EBI FTP and view it in the grid. Select the genome build (GRCh37 / GRCh38) before loading.

### Configuration

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `PRS_CACHE_DIR` | `~/.cache/just-prs` | Root directory for cached metadata and scoring files |

---

## Data sources

- PGS Catalog REST API: <https://www.pgscatalog.org/rest/>
- EBI FTP bulk downloads: <https://ftp.ebi.ac.uk/pub/databases/spot/pgs/>
- PGS Catalog download documentation: <https://www.pgscatalog.org/downloads/>
- Cleaned metadata parquets on HuggingFace: <https://huggingface.co/datasets/just-dna-seq/polygenic_risk_scores>
