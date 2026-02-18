# just-prs

A [Polars](https://pola.rs/)-bio based tool to compute **Polygenic Risk Scores (PRS)** from the [PGS Catalog](https://www.pgscatalog.org/).

## Features

- Compute PRS for one or many scores against a VCF file
- Search and inspect PGS Catalog scores and traits via the REST API
- **Bulk download** the entire PGS Catalog metadata (all ~5,000+ scores) via EBI FTP — one HTTP request per sheet, not hundreds of API pages
- Stream harmonized scoring files directly from EBI FTP without storing intermediate `.gz` files
- All data saved as **Parquet** for fast, efficient downstream analysis with Polars

## Installation

Requires Python ≥ 3.14. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/antonkulaga/just-prs
cd just-prs
uv sync
```

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
# Download all 7 metadata sheets → ./pgs_metadata/*.parquet
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

The `scores` sheet automatically parses the three ancestry distribution columns
(`Ancestry Distribution (%) - Source of Variant Associations (GWAS)`,
`Ancestry Distribution (%) - Score Development/Training`,
`Ancestry Distribution (%) - PGS Evaluation`) from their raw
`Population:percent|Population:percent` string encoding into proper
`List(Struct({population: String, percent: Float64}))` Polars columns,
enabling direct filtering and aggregation without string manipulation.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir / -o` | `./pgs_metadata` | Directory for parquet output |
| `--sheet / -s` | all sheets | Single sheet name to download |
| `--overwrite` | `False` | Re-download existing files |

#### `prs catalog bulk scores` — All scoring files as parquet

Streams each harmonized scoring file from EBI FTP and saves it as a parquet file
(with an added `pgs_id` column). No intermediate `.gz` files are written to disk.

```bash
# Download ALL ~5,000+ scoring files (GRCh38) → ./pgs_scores/PGS######.parquet
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
| `--output-dir / -o` | `./pgs_scores` | Directory for parquet output |
| `--build / -b` | `GRCh38` | Genome build (`GRCh37` or `GRCh38`) |
| `--ids` | all | Comma-separated PGS IDs to download |
| `--overwrite` | `False` | Re-download existing parquet files |

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
df = download_metadata_sheet("scores", Path("scores.parquet"))

# All 7 sheets at once
sheets = download_all_metadata(Path("./pgs_metadata"))

# Stream a scoring file as a LazyFrame (no local .gz written)
lf = stream_scoring_file("PGS000001", genome_build="GRCh38")

# Download one scoring file as parquet (adds pgs_id column)
path = download_scoring_as_parquet("PGS000001", Path("./scores"))

# Bulk download a list (or all) scoring files as parquet
paths = bulk_download_scoring_parquets(Path("./scores"), pgs_ids=["PGS000001", "PGS000002"])
paths = bulk_download_scoring_parquets(Path("./scores"))  # all ~5000+
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

### PRS computation (`just_prs.prs`)

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

```bash
cd prs-ui
uv run reflex run
```

The UI has three tabs:

### Metadata Sheets

Browse all 7 PGS Catalog metadata sheets (Scores, Publications, EFO Traits, etc.) in a MUI DataGrid with server-side filtering and sorting. Select rows with checkboxes and download their scoring files to the local cache with a single **Download Selected** button.

### Scoring File

Stream any harmonized scoring file by PGS ID directly from EBI FTP and view it in the grid. Select the genome build (GRCh37 / GRCh38) before loading.

### Compute PRS

End-to-end PRS computation workflow:

1. **Upload a VCF** — drag-and-drop or browse; genome build is auto-detected from `##reference` and `##contig` headers
2. **Load Scores** — fetches the PGS Catalog scores metadata, pre-filtered by the detected (or manually selected) genome build. Scores are shown in a paginated, searchable table
3. **Select scores** — use checkboxes to pick individual scores, or "Select All" to select everything matching the current search
4. **Compute** — runs PRS for each selected score against the uploaded VCF and shows results with match rates, effect sizes, and classification metrics from PGS Catalog evaluation studies

Configuration:

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `PRS_CACHE_DIR` | `~/.cache/just-prs` | Root directory for cached metadata and scoring files |

---

## Data sources

- PGS Catalog REST API: <https://www.pgscatalog.org/rest/>
- EBI FTP bulk downloads: <https://ftp.ebi.ac.uk/pub/databases/spot/pgs/>
- PGS Catalog download documentation: <https://www.pgscatalog.org/downloads/>
