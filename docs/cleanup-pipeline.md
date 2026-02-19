# PGS Catalog Cleanup Pipeline

This document describes the data quality issues in the raw PGS Catalog bulk metadata
and how the cleanup pipeline (`just_prs.cleanup`) addresses them.

## Raw data quality issues

The PGS Catalog distributes pre-built CSV metadata files covering the entire catalog
(~5,000+ scores). These are fast to download but contain several inconsistencies
that make programmatic use difficult.

### 1. Genome build inconsistency

The `Original Genome Build` column has 9 distinct values representing only ~3 actual builds:

| Raw value | Canonical | Count (as of 2026-02) |
|-----------|-----------|----------------------|
| `GRCh37`  | GRCh37    | 3921                 |
| `hg19`    | GRCh37    | 378                  |
| `hg37`    | GRCh37    | 3                    |
| `GRCh38`  | GRCh38    | 421                  |
| `hg38`    | GRCh38    | 169                  |
| `NR`      | NR        | 344                  |
| `NCBI36`  | GRCh36    | 11                   |
| `hg18`    | GRCh36    | 3                    |
| `NCBI35`  | GRCh36    | 1                    |

The cleanup pipeline maps all variants to canonical forms (`GRCh37`, `GRCh38`, `GRCh36`, `NR`)
via `cleanup.BUILD_NORMALIZATION`.

### 2. Verbose column names

Raw PGS Catalog column names are long and contain parentheses, spaces, and special characters:

| Raw name | Cleaned name |
|----------|-------------|
| `Polygenic Score (PGS) ID` | `pgs_id` |
| `PGS Name` | `name` |
| `Reported Trait` | `trait_reported` |
| `Mapped Trait(s) (EFO label)` | `trait_efo` |
| `Mapped Trait(s) (EFO ID)` | `trait_efo_id` |
| `Original Genome Build` | `genome_build` |
| `Number of Variants` | `n_variants` |
| `Type of Variant Weight` | `weight_type` |
| `PGS Publication (PGP) ID` | `pgp_id` |
| `Publication (PMID)` | `pmid` |
| `FTP link` | `ftp_link` |
| `Release Date` | `release_date` |

Columns not in this list (development details, interaction terms, license text, ancestry
distribution strings, publication DOI, match-original boolean) are dropped by `clean_scores()`
as they are not needed for PRS computation or search.

### 3. Performance metrics as strings

The performance_metrics sheet stores metric values as human-readable strings with
varying formats:

| Format | Example | Parsed fields |
|--------|---------|---------------|
| Estimate with CI | `1.55 [1.52,1.58]` | estimate=1.55, ci_lower=1.52, ci_upper=1.58 |
| Estimate with SE | `-0.7 (0.15)` | estimate=-0.7, se=0.15 |
| Estimate only | `1.41` | estimate=1.41 |
| Free text | `Nagelkerke's R2 = 0.04` | Not parsed (returns None) |

The cleanup pipeline parses 5 metric columns (OR, HR, Beta, AUROC, C-index) into
4 numeric columns each (e.g. `or_estimate`, `or_ci_lower`, `or_ci_upper`, `or_se`),
producing 20 structured numeric columns from 5 string columns.

### 4. Multi-valued trait fields

507 scores have pipe-separated values in `Mapped Trait(s) (EFO label)` and
`Mapped Trait(s) (EFO ID)`, e.g.:
- `stroke|Ischemic stroke`
- `EFO_0000712|HP_0002140`

These are kept as-is in the cleaned LazyFrame since the pipe-separated format
still allows substring search and is sufficient for display/filtering.

## Architecture

```
ftp.download_metadata_sheet()     Raw CSV → DataFrame (cached as parquet)
        │
        ▼
cleanup.clean_scores()            Rename columns, normalize builds, select useful subset
cleanup.clean_performance_metrics()  Parse metrics, join with evaluation samples
cleanup.best_performance_per_score() One best row per PGS ID
        │
        ▼
PRSCatalog                        Lazy loading, search, PRS computation, percentile
```

- `cleanup.py` contains pure functions: no state, no caching, no I/O.
- `prs_catalog.py` handles caching and lazy loading, delegates transforms to `cleanup.py`.
- `ftp.py` handles raw downloads and parquet serialization.

## Percentile estimation

`PRSCatalog.percentile()` estimates population percentiles for a PRS score using
one of two approaches:

1. **Explicit mean/std** (preferred): When a reference cohort provides population
   statistics, pass `mean` and `std` directly. The percentile is `Phi((score - mean) / std) * 100`.

2. **AUROC-based estimation** (fallback): When only the best AUROC from evaluation
   studies is available, Cohen's d is estimated via `d = sqrt(2) * Phi^{-1}(AUROC)`.
   The effective population SD is derived from the mixture distribution:
   `sigma_eff = sqrt(1 + d^2/4)`. This is an approximation; exact percentiles
   require a matched reference cohort.

The normal CDF and its inverse are implemented without scipy using `math.erfc`
and the Abramowitz & Stegun rational approximation.
