# CLI Reference

The CLI is available as both `just-prs` and `prs`.

```
prs --help
prs compute --help
prs normalize --help
prs catalog --help
prs reference --help
prs pgen --help
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

## `prs pgen` — PLINK2 binary format operations

Pure Python tools for working with PLINK2 binary filesets (.pgen/.pvar.zst/.psam) via `pgenlib` + polars. These commands replace common PLINK2 operations while producing identical results — validated against PLINK2 with Pearson r = 1.0 across 3,202 samples (see [validation](validation.md)). No external binaries required.

| PLINK2 command | just-prs equivalent | Description |
|---|---|---|
| `plink2 --pfile ... --make-just-pvar` | `prs pgen read-pvar` | Parse .pvar.zst variant table |
| `plink2 --pfile ... --make-just-psam` | `prs pgen read-psam` | Parse .psam sample table |
| `plink2 --pfile ... --extract ...` | `prs pgen genotypes` | Extract genotypes for selected variants |
| `plink2 --pfile ... --score ...` | `prs pgen score` / `prs reference score` | Compute PRS for a scoring file |

### `prs pgen read-pvar` — Parse a .pvar.zst variant file

Decompresses and parses a .pvar.zst file into a variant table. Caches the result as a parquet file for fast subsequent reads (~0.5s vs ~7s for initial parse).

```bash
prs pgen read-pvar /path/to/panel.pvar.zst
prs pgen read-pvar /path/to/panel.pvar.zst --limit 50
prs pgen read-pvar /path/to/panel.pvar.zst --output variants.parquet
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `PVAR_PATH` (argument) | — | Path to .pvar.zst file (required) |
| `--limit / -n` | 20 | Max rows to display |
| `--output / -o` | — | Save full table as parquet |

### `prs pgen read-psam` — Parse a .psam sample file

Reads sample IDs and population labels from a PLINK2 .psam file.

```bash
prs pgen read-psam /path/to/panel.psam
prs pgen read-psam /path/to/panel.psam --output samples.parquet
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `PSAM_PATH` (argument) | — | Path to .psam file (required) |
| `--limit / -n` | 20 | Max rows to display |
| `--output / -o` | — | Save full table as parquet |

### `prs pgen genotypes` — Extract genotypes from a .pgen file

Reads genotype data directly from a .pgen binary file for specified genomic regions. Output values: 0 = hom-ref, 1 = het, 2 = hom-alt, -9 = missing.

```bash
prs pgen genotypes panel.pgen panel.pvar.zst panel.psam --chrom 11 --start 69000000 --end 70000000
prs pgen genotypes panel.pgen panel.pvar.zst panel.psam --chrom 1 --limit 50 --output geno.parquet
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `PGEN_PATH` (argument) | — | Path to .pgen file (required) |
| `PVAR_PATH` (argument) | — | Path to .pvar.zst file (required) |
| `PSAM_PATH` (argument) | — | Path to .psam file (required) |
| `--chrom / -c` | — | Filter to this chromosome |
| `--start` | — | Start position (inclusive) |
| `--end` | — | End position (inclusive) |
| `--limit / -n` | 100 | Max variants to extract |
| `--output / -o` | — | Save genotypes as parquet |

### `prs pgen score` — Score a PGS ID against any .pgen dataset

Computes PRS for a PGS Catalog score against any PLINK2 binary fileset. Unlike `prs reference score` (which targets the 1000G panel), this works with any .pgen/.pvar.zst/.psam dataset.

```bash
prs pgen score PGS000001 /path/to/pgen_dir/
prs pgen score PGS000001 /path/to/pgen_dir/ --build GRCh37 --output scores.parquet
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `PGS_ID` (argument) | — | PGS score ID, e.g. `PGS000001` (required) |
| `PGEN_DIR` (argument) | — | Directory containing .pgen/.pvar.zst/.psam files (required) |
| `--build / -b` | `GRCh38` | Genome build |
| `--output / -o` | — | Save scores as parquet |
| `--cache-dir` | OS cache dir | Override cache directory |

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

---

## `prs reference` — Reference panel operations

Score PGS IDs against population reference panels (1000 Genomes, HGDP+1kGP) using pgenlib + polars — no external PLINK2 binary required. Two panels are supported:

| Panel ID | Size | Description |
|----------|------|-------------|
| `1000g` (default) | ~7 GB | 1000 Genomes Project (3,202 individuals, 5 superpopulations) |
| `hgdp_1kg` | ~15 GB | HGDP + 1000 Genomes merged panel (more populations, better global coverage) |

The `score-plink2` and `compare` subcommands are retained for cross-validation against PLINK2 (which requires a PLINK2 binary).

### `prs reference download` — Download a reference panel

Downloads a reference panel tarball from the PGS Catalog FTP and extracts it.

```bash
prs reference download                          # default: 1000g panel (~7 GB)
prs reference download --panel hgdp_1kg         # HGDP + 1000G merged panel (~15 GB)
prs reference download --cache-dir /data/cache
prs reference download --overwrite
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--panel` | `1000g` | Reference panel to download (`1000g` or `hgdp_1kg`) |
| `--cache-dir` | OS cache dir | Override cache directory |
| `--overwrite` | `False` | Re-download even if already present |

### `prs reference score-batch` — Batch score multiple PGS IDs

Scores multiple PGS IDs against a reference panel in a single process. Downloads scoring files, computes PRS for each using pgenlib + polars, tracks failures and quality flags, and produces aggregated distribution statistics and a quality report. This is the primary command for building reference distributions.

```bash
# Score all ~5,000+ PGS IDs against the 1000G panel
prs reference score-batch

# Score specific PGS IDs
prs reference score-batch --pgs-ids PGS000001,PGS000002,PGS000003

# Score only the first 50 PGS IDs
prs reference score-batch --limit 50

# Score against a different panel
prs reference score-batch --panel hgdp_1kg

# Force re-scoring (ignore cached results)
prs reference score-batch --no-skip-existing

# Adjust the match rate threshold for quality flags
prs reference score-batch --match-threshold 0.2
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--pgs-ids / -p` | all PGS IDs | Comma-separated PGS IDs to score |
| `--limit / -n` | `0` (all) | Score only the first N PGS IDs |
| `--build / -b` | `GRCh38` | Genome build |
| `--panel` | `1000g` | Reference panel identifier (`1000g` or `hgdp_1kg`) |
| `--skip-existing / --no-skip-existing` | `--skip-existing` | Skip PGS IDs already scored |
| `--match-threshold` | `0.1` | Flag scores with match rate below this as `low_match` |
| `--cache-dir` | OS cache dir | Override cache directory |

Output files:

| File | Description |
|------|-------------|
| `<cache>/percentiles/{panel}_distributions.parquet` | Per-superpopulation distribution statistics for all scored PGS IDs |
| `<cache>/percentiles/{panel}_quality.parquet` | Quality report with status, match rate, variance, timing per PGS ID |
| `<cache>/reference_scores/{panel}/{pgs_id}/scores.parquet` | Per-individual scores for each PGS ID (cached for reuse) |

Quality status values: `ok`, `failed` (exception during scoring), `low_match` (match rate below threshold), `zero_variance` (all individuals scored identically).

### `prs reference score` — Score via pgenlib + polars (default)

Reads genotypes directly from the `.pgen` binary via `pgenlib`, matches scoring variants against `.pvar.zst` using polars, and computes dosage-weighted PRS in numpy.

```bash
prs reference score PGS000001
prs reference score PGS000001 --build GRCh37
prs reference score PGS000001 --cache-dir /data/cache
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `PGS_ID` (argument) | — | PGS score ID, e.g. `PGS000001` (required) |
| `--build / -b` | `GRCh38` | Genome build (`GRCh37` or `GRCh38`) |
| `--cache-dir` | OS cache dir | Override cache directory |

### `prs reference score-plink2` — Score via PLINK2 (cross-validation)

Uses the PLINK2 binary for `--score`. Retained for cross-validating against the pgenlib + polars engine. Requires a PLINK2 binary at `~/.cache/just-prs/plink2/plink2`.

```bash
prs reference score-plink2 PGS000001
prs reference score-plink2 PGS000001 --build GRCh37
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `PGS_ID` (argument) | — | PGS score ID, e.g. `PGS000001` (required) |
| `--build / -b` | `GRCh38` | Genome build (`GRCh37` or `GRCh38`) |
| `--cache-dir` | OS cache dir | Override cache directory |

### `prs reference compare` — Cross-validate engines

Runs both scoring engines (pgenlib + polars and PLINK2 `--score`) on the same PGS ID and reports per-superpopulation statistics, per-sample Pearson correlation (expected: 1.0), maximum absolute difference, and timing comparison. Useful for verifying that the pure Python engine produces identical results to PLINK2.

```bash
prs reference compare PGS000001
prs reference compare PGS000001 --build GRCh37
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `PGS_ID` (argument) | — | PGS score ID, e.g. `PGS000001` (required) |
| `--build / -b` | `GRCh38` | Genome build (`GRCh37` or `GRCh38`) |
| `--cache-dir` | OS cache dir | Override cache directory |

### `prs reference test-score` — Test scoring for multiple PGS IDs

Runs scoring for each PGS ID using the polars engine, validates the output (sample count, superpopulation coverage, score variance), and prints a pass/fail summary table. Exits with code 1 if any score fails validation.

```bash
# Test default set (PGS000001, PGS000002, PGS000004, PGS000010)
prs reference test-score

# Test specific IDs
prs reference test-score --pgs-ids PGS000001,PGS000003,PGS000007

# Custom build and cache dir
prs reference test-score --pgs-ids PGS000001 --build GRCh37 --cache-dir /data/cache
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--pgs-ids / -p` | `PGS000001,PGS000002,PGS000004,PGS000010` | Comma-separated PGS IDs to test |
| `--build / -b` | `GRCh38` | Genome build |
| `--cache-dir` | OS cache dir | Override cache directory |

Validation checks per PGS ID:
- Exactly 3,202 samples scored
- All 5 superpopulations present (AFR, AMR, EAS, EUR, SAS)
- Non-zero score variance (scores are not all identical)
