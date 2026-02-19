# CLI Reference

The CLI is available as both `just-prs` and `prs`.

```
prs --help
prs compute --help
prs normalize --help
prs catalog --help
```

---

## `prs normalize` — Normalize a VCF to Parquet

Reads a VCF file, strips chr prefix from chromosomes, renames the id column to rsid, computes genotype from GT indices, applies configurable quality filters (FILTER values, minimum depth, minimum QUAL), and writes a zstd-compressed Parquet file.

```bash
prs normalize --vcf sample.vcf.gz
prs normalize --vcf sample.vcf.gz --output normalized.parquet
prs normalize --vcf sample.vcf.gz --pass-filters "PASS,." --min-depth 10 --min-qual 30
prs normalize --vcf sample.vcf.gz --sex Female
prs normalize --vcf sample.vcf.gz --format-fields "GT,DP,GQ"
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--vcf / -v` | — | Path to VCF file (required) |
| `--output / -o` | `data/output/results/<stem>.parquet` | Output Parquet path |
| `--pass-filters` | — | Comma-separated FILTER values to keep (e.g. `"PASS,."`) |
| `--min-depth` | — | Minimum DP (read depth) to keep a variant |
| `--min-qual` | — | Minimum QUAL score to keep a variant |
| `--sex` | — | Sample sex (`"Male"` or `"Female"`). Warns if Female has chrY variants |
| `--format-fields` | `GT,DP` | Comma-separated FORMAT fields to include |

Output columns: `chrom`, `pos`, `rsid`, `ref`, `alt`, `qual`, `filter`, `GT`, `DP`, `genotype` (List[Str] of resolved alleles, alphabetically sorted).

---

## `prs compute` — Compute PRS for a VCF

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
| `--cache-dir` | OS cache dir / `just-prs/scores` | Cache directory for scoring files |
| `--output / -o` | — | Save results as JSON |

---

## `prs catalog scores` — Search and inspect scores (REST API)

```bash
prs catalog scores list                        # first 100 scores
prs catalog scores list --all                  # every score in catalog
prs catalog scores search --term "breast cancer"
prs catalog scores info PGS000001
```

## `prs catalog traits` — Search and inspect traits (REST API)

```bash
prs catalog traits search --term "diabetes"
prs catalog traits info EFO_0001645
```

## `prs catalog download` — Download a single scoring file

Downloads the harmonized `.txt.gz` scoring file for one score and caches it locally.

```bash
prs catalog download PGS000001
prs catalog download PGS000001 --output-dir ./my_scores --build GRCh37
```

---

## `prs catalog bulk` — Bulk FTP downloads (fast, parquet output)

These commands use the [EBI FTP HTTPS mirror](https://ftp.ebi.ac.uk/pub/databases/spot/pgs/) via **fsspec** to download pre-built catalog-wide files directly — far faster than paginating the REST API.

### `prs catalog bulk metadata` — All catalog metadata as parquet

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

### `prs catalog bulk scores` — All scoring files as parquet

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

### `prs catalog bulk clean-metadata` — Build cleaned metadata parquets

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

### `prs catalog bulk push-hf` — Push cleaned parquets to HuggingFace

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

### `prs catalog bulk pull-hf` — Pull cleaned parquets from HuggingFace

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

### `prs catalog bulk ids` — List all PGS IDs

Fetches `pgs_scores_list.txt` from EBI FTP (one request) and prints every PGS ID.

```bash
prs catalog bulk ids
prs catalog bulk ids | wc -l    # count total scores
```
