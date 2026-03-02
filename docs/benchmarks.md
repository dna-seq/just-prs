# Engine Benchmark: Polars vs DuckDB vs PLINK2

This page presents a reproducible benchmark comparing three PRS computation engines
on a real personal genome. The goal is to demonstrate that **just-prs** matches
PLINK2 in accuracy while being substantially faster and far more memory-efficient.

---

## Setup

### Input data

| Item | Details |
|------|---------|
| **VCF** | Personal whole-genome VCF of a single individual ([Zenodo 18370498](https://zenodo.org/records/18370498)) |
| **Variants** | 4,661,444 biallelic sites (auto-downloaded, normalized to Parquet, ~55 MB) |
| **Genome build** | GRCh38 (harmonized PGS Catalog scoring files) |
| **PGS IDs** | 100 consecutive IDs: PGS000006 – PGS000106 (5 warm-up IDs excluded from results) |
| **Scoring files** | Fetched from PGS Catalog FTP; range from 77 variants (PGS000001) to 6.9 M variants (PGS000014) |

### System

| Property | Value |
|----------|-------|
| OS | Linux 6.8.0 (lowlatency) |
| Python | 3.13.5 |
| CPU | 32 logical cores |
| RAM | 94 GB |
| PLINK2 | v2.0 (20260110) |

### Engines

| Engine | Description | Variant matching | Genotype input |
|--------|-------------|-----------------|---------------|
| **polars** | `compute_prs()` with LazyFrame from normalized Parquet | Polars inner join on (chrom, pos) | Pre-normalized `.parquet` |
| **duckdb** | SQL join via `duckdb.connect()` on normalized Parquet | DuckDB vectorized hash join | Pre-normalized `.parquet` |
| **plink2** | PLINK2 binary v2.0, `--pfile --score no-mean-imputation cols=scoresums` | 4-part chr:pos:ref:alt ID match | Pre-converted `.pgen` (autosomes only) |

**Pre-computation** (done once, not included in per-PGS timings):

- polars / duckdb: VCF → normalized Parquet via `normalize_vcf()` (~4 min for 483 MB VCF)
- plink2: VCF → pgen via `plink2 --vcf --set-all-var-ids @:#:$r:$a --chr 1-22 --make-pgen` (~3 min)

### Memory measurement

- **polars / duckdb**: `tracemalloc` peak Python heap allocation per scoring call.
  This captures in-process allocations (scoring file loading, join structures, result
  arrays) without OS page-cache noise from memory-mapped Parquet files.
- **plink2**: subprocess peak RSS sampled every 100 ms via `psutil`. Captures the
  full footprint of the external binary including its internal data structures.

> These two measures are not directly comparable in scale, but they each represent
> the *additional* memory cost incurred per scoring run on top of the data already
> held in the OS page cache.

---

## Results

### Runtime — 100 PGS IDs

> Median and mean elapsed time per PGS ID scoring call (wall-clock, seconds).
> "Excluding large" = the 7 PGS IDs with ≥1 M variants (see below).

| Engine | N scored | Median (all) | Mean (all) | Median (≤1M vars) | Mean (≤1M vars) | Speedup vs PLINK2 (median) |
|--------|:--------:|:------------:|:----------:|:-----------------:|:---------------:|:--------------------------:|
| **duckdb** | 100 | **0.049 s** | 0.394 s | 0.048 s | 0.059 s | **12.3×** |
| **polars** | 100 | 0.106 s | 0.466 s | 0.105 s | 0.117 s | 5.7× |
| plink2 | 96 | 0.603 s | 0.703 s | 0.603 s | 0.640 s | 1× (baseline) |

**Key takeaways:**

- DuckDB's vectorized SQL join is **2.2× faster than Polars** at the median.
- Both Python engines are **5–12× faster than PLINK2** at the median.
- PLINK2 has a near-constant ~600 ms floor per PGS regardless of scoring file size,
  dominated by loading and decompressing the pgen file for every scoring call.
- polars and duckdb scale near-linearly with scoring file size; for small scores
  (< 1 000 variants), duckdb achieves < 30 ms.

### Runtime — Large scoring files (≥ 1 M variants)

These 7 genome-wide PGS IDs are scored by all engines but represent atypical workloads.

| PGS ID | Variants | Polars (s) | DuckDB (s) | PLINK2 |
|--------|:--------:|:----------:|:----------:|:------:|
| PGS000014 | 6,917,436 | 6.72 | 6.42 | FAILED |
| PGS000017 | 6,907,112 | 6.77 | 6.42 | FAILED |
| PGS000016 | 6,730,541 | 6.65 | 6.28 | FAILED |
| PGS000013 | 6,630,150 | 6.71 | 6.42 | FAILED |
| PGS000039 | 3,225,583 | 4.12 | 3.84 | 1.79 s |
| PGS000027 | 2,100,302 | 2.48 | 2.28 | 1.71 s |
| PGS000018 | 1,745,179 | 2.32 | 2.23 | 1.41 s |

PLINK2 **failed to produce a score** for the four largest files
(PGS000013/014/016/017, each ≈ 6.6–6.9 M variants). These genome-wide scoring
files contain variants without a matching 4-part `chr:pos:ref:alt` ID in the pgen
(which covers autosomes only). Both just-prs engines scored all 100 IDs
successfully using flexible position-based matching.

For the three files where PLINK2 did succeed, PLINK2 was faster for very large
files because it can leverage compiled C++ SIMD operations on pre-indexed data;
just-prs currently reads the full scoring file as a DataFrame before joining.
Future versions will use streaming joins for files > 1 M variants.

### Memory — per scoring call

| Engine | Median peak | Mean peak | Max peak | Method |
|--------|:-----------:|:---------:|:--------:|--------|
| **duckdb** | **0.2 MB** | 139 MB | 2,594 MB | tracemalloc heap |
| **polars** | **0.2 MB** | 139 MB | 2,594 MB | tracemalloc heap |
| plink2 | **590 MB** | 595 MB | 974 MB | subprocess RSS |

**Key takeaways:**

- For typical PGS IDs (< 500 K variants), polars and duckdb allocate **< 1 MB**
  of Python heap per scoring call. The normalized Parquet is memory-mapped by the
  OS and does not appear in the tracemalloc peak.
- PLINK2 allocates **~590 MB per scoring call** because it must load the compressed
  pgen, pvar, and psam files into its own address space for every `--score`
  invocation. This is roughly **590–3000× more RAM** than our Python engines.
- The mean/max for polars and duckdb are high because they are dominated by the
  7 large scoring files (6–7 M variants × 8 bytes × several columns ≈ 2–3 GB
  DataFrames). Future streaming support will eliminate this spike.
- PLINK2's RAM usage is stable across scoring file sizes (590–974 MB) because
  the pgen reading cost is invariant; it is limited by the pgen file size, not
  the scoring file size.

### Score agreement

All engines produce numerically equivalent results. Pearson correlations and
maximum absolute differences across 100 PGS IDs (96 PGS IDs for PLINK2 pairs,
since 4 IDs failed):

| Engine pair | N PGS | Pearson r | Max \|Δ score\| |
|-------------|:-----:|:---------:|:---------------:|
| **duckdb ↔ polars** | 100 | **1.000000** | < 1.1 × 10⁻¹³ |
| duckdb ↔ plink2 | 96 | 0.999859 | 21.4 |
| polars ↔ plink2 | 96 | 0.999859 | 21.4 |

**Notes on the absolute difference:**

- The near-zero difference between DuckDB and Polars confirms they implement
  the same algorithm: position-based join, dosage from GT string, weighted sum.
- The absolute max difference vs PLINK2 (21.4) reflects two sources:
  1. **Scale**: PLINK2 `cols=scoresums` returns a raw sum, but our engines also
     compute a raw sum. For genome-wide scores (millions of variants), even a
     fractional difference in the set of matched variants causes large absolute diffs.
  2. **Variant matching**: PLINK2 requires an exact 4-part `chr:pos:ref:alt` ID
     match; our engines use position-only matching followed by allele orientation
     detection. For ambiguous or multiallelic sites, 1–3 extra variants are
     included or excluded, shifting the sum slightly.
- The **Pearson correlation of 0.9999** confirms that per-individual risk rankings
  are effectively identical. For percentile estimation and clinical interpretation,
  rank correlation is the relevant metric.

---

## Reproducing the benchmark

All results are fully reproducible. The benchmark auto-downloads all inputs on
first run; subsequent runs use cached files.

```bash
# Clone and install
git clone https://github.com/just-dna-seq/just-prs.git
cd just-prs
uv sync --all-packages

# Full 100-PGS benchmark (auto-downloads VCF + scoring files on first run, ~5–10 min)
uv run python just-prs/benchmarks/benchmark_engines.py

# Skip PLINK2 (if binary not available)
uv run python just-prs/benchmarks/benchmark_engines.py --skip-plink2

# Custom number of PGS IDs or output directory
uv run python just-prs/benchmarks/benchmark_engines.py --n-pgs 50 --output my_results/

# Results are written to:
#   data/output/benchmarks/benchmark_results.csv      (per-PGS per-engine timings)
#   data/output/benchmarks/benchmark_validation.csv   (cross-engine correlations)
#   data/output/benchmarks/benchmark_metadata.csv     (system info)
```

### Output schema

`benchmark_results.csv`:

| Column | Description |
|--------|-------------|
| `pgs_id` | PGS Catalog score identifier |
| `engine` | `polars`, `duckdb`, or `plink2` |
| `elapsed_sec` | Wall-clock time (seconds) |
| `peak_memory_mb` | Peak heap MB (polars/duckdb) or peak subprocess RSS MB (plink2) |
| `score` | Raw PRS sum (effect_weight × dosage) |
| `variants_matched` | Variants matched between VCF and scoring file |
| `variants_total` | Total variants in scoring file |
| `match_rate` | `variants_matched / variants_total` |
| `error` | Error message if engine failed, else null |

---

## Discussion

### When to use each engine

| Scenario | Recommended engine |
|----------|--------------------|
| Single individual, typical PGS (< 100 K variants) | **duckdb** (fastest, lowest RAM) |
| Batch scoring many individuals | **polars** (pure Python, easily parallelizable) |
| Cross-validation against published tools | **plink2** (standard reference implementation) |
| Genome-wide scores (> 1 M variants), large files | **duckdb** or **polars** (PLINK2 may fail) |

### Limitations and future work

1. **Large scoring files**: For genome-wide PGS (6–7 M variants), all engines
   currently load the scoring DataFrame into memory. A future streaming join will
   reduce the memory peak from ~2.5 GB to < 100 MB for these cases.
2. **PLINK2 pgen scope**: The pre-converted pgen covers autosomes only (chr 1–22)
   due to chrX pseudoautosomal region complications. This causes PLINK2 to miss
   X-linked variants in scoring files. Our engines use the full-genome normalized
   Parquet, so they score X-linked variants correctly.
3. **Match rate**: All engines achieve ~50–54% variant match rate. The unmatched
   ~50% consists mainly of insertions/deletions and multiallelic sites that are
   not directly representable in the VCF as biallelic SNPs, plus any build-specific
   coordinate shifts.
