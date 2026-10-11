# Engine Benchmark: Polars vs DuckDB vs PLINK2

This page describes a reproducible benchmark of three PRS engines on one real
personal genome: just-prs's two engines (Polars and DuckDB) and PLINK2 as the
reference implementation. It measures agreement, runtime and memory.

> **Status (2026-10-11).** The timed rerun with the fixed engines and fixed
> benchmark inputs is queued; its results land in
> `just-prs/benchmarks/results/2026-10-11/` and replace the tables below. The
> numbers from the March 2026 run are kept at the end of this page for the
> record, with what was wrong with them.

---

## Setup

### Input data

| Item | Details |
|------|---------|
| **VCF** | Personal whole-genome VCF of one individual, DeepVariant 1.1.0 ([Zenodo 18370498](https://zenodo.org/records/18370498), `antonkulaga.vcf`, md5 `21598da4f77d8237577662ab89e8a55d`) |
| **Records scored** | PASS (or `.`) records on chromosomes 1–22: 4,516,566. All 4,661,444 PASS records, of which 76,657 are multi-allelic and 115,416 on chrX; DeepVariant RefCall records (1,418,300) are excluded |
| **Genome build** | GRCh38 (PGS Catalog harmonized scoring files, `hmPOS_GRCh38`) |
| **PGS IDs** | PGS000006 – PGS000106 without PGS000085 (no GRCh38 harmonized file): 100 IDs, listed in `pgs_ids.txt`. Warm-up: PGS000001–003, not counted |
| **Scoring files** | 77 to 6.9 M variants; checksums and harmonization dates in `scoring_files.csv` |

### Making the three engines read the same data

| Input | just-prs (Polars, DuckDB) | PLINK2 |
|---|---|---|
| Records | `normalize_vcf(pass_filters=["PASS", "."])`, then chromosomes 1–22 | `--vcf-filter --chr 1-22 --make-pgen`, IDs `@:#:$r:$a` |
| Allele matching | position join, then the allele check (`docs/vcf_flow.md`) | `CHROM:POS:A:B` IDs |
| Score input | the scoring file | `_prepare_plink2_score_input`: both allele orders, **every** `hm_inferOtherAllele` candidate, liftover-duplicate (variant, allele) pairs summed |
| Missing genotype | not counted (`variants_no_call`) | `no-mean-imputation` |
| Output | sum of `effect_weight × dosage` | `cols=scoresums` (`SCORE1_SUM`) |

chrX is left out of every engine: a PLINK2 pgen needs sample sex to code it,
and the two disagree on it by design. PLINK2 scores a male X call 0..2
(`--xchr-model 2`); just-prs counts a haploid call as the one copy it is.

### Engines

| Engine | Call | Genotype input |
|--------|------|----------------|
| **polars** | `compute_prs(genotypes_lf=pl.scan_parquet(...))` | normalized Parquet |
| **duckdb** | `compute_prs_duckdb(genotypes_parquet=...)` | normalized Parquet (DuckDB `read_parquet`) |
| **plink2** | `plink2 --pfile ... --score <input> 1 2 3 header no-mean-imputation cols=scoresums` | pre-converted pgen |

PLINK2 is v2.0.0-a.6.33LM (28 Feb 2026), the build the script downloads. Its
captured stdout/stderr and `.log` are kept per ID in `plink2_logs/`.

**Pre-computation** (done once, not in the per-PGS timings): VCF → normalized
Parquet with `normalize_vcf()`; VCF → pgen with `plink2 --vcf-filter`.

### Memory measurement

- **polars / duckdb**: process RSS, peak during the call minus the RSS before it,
  sampled every 10 ms with `psutil`. RSS includes the native Rust (Polars) and
  C++ (DuckDB) allocations.
- **plink2**: peak RSS of the subprocess, sampled every 100 ms.

The two numbers are not the same quantity: one is the extra memory a call costs
inside an already running Python process, the other is a whole process.

---

## Results

Pending the 2026-10-11 run (see the status note above).

---

## Why the March 2026 run disagreed with PLINK2

The March 2026 run reported r = 0.999859 and a largest difference of 21.4
(PGS000012), and PLINK2 failed on PGS000013/014/016/017. A per-variant
recomputation of both routes by a just-dna-lite review (2026-10-10)
reproduced every score and assigned every difference to a cause, with no
residual:

| Share of Σ\|Δ\| | Cause | Fixed by |
|---:|---|---|
| 71.4% | The PLINK2 score input used only the **first** `hm_inferOtherAllele` candidate, so 2,024 heterozygous effect=REF variants in 20 scores went unmatched (all of PGS000012's +21.44) | listing every candidate |
| 15.7% | The pgen was converted from the **unfiltered** VCF, so PLINK2 counted 1,497 DeepVariant RefCall 0/0 sites as two REF copies; just-prs reads PASS only | `--vcf-filter` |
| 9.6% | just-prs matched by **position only**: an SNV row took a dose from an indel record sharing its anchor base, and an indel row matched a different indel | allele-aware matching (just-prs 0.10.2) |
| 1.4% | just-prs compared the effect allele with the **joined** multi-allelic ALT string (`C\|G`) | ALT splitting (just-prs 0.10.2) |
| 1.9% | chrX, scored by just-prs but cut by `--chr 1-22` | both engines now score 1–22 |

The PLINK2 failures were not caused by unmatched IDs (those are skipped with a
warning). Each of the four files has pairs of GRCh37 variants that lift over to
the same GRCh38 variant; writing both allele orders repeated a (variant,
allele) line, and PLINK2 stops with `ALT1 allele for variant ... appears
multiple times`. The score input now sums such pairs, which is also what
just-prs computes when it scores both rows against one genotype.

The matched counts in the March table differed by far more than "1–3
variants": on PGS000012, just-prs matched 24,005 rows and PLINK2 20,328.

### March 2026 numbers (superseded)

just-prs commit 5add116, PLINK2 a6.32, 32 cores / 94 GB. Memory was then
measured with `tracemalloc` (Python heap only), which does not see native
allocations; the script now measures RSS.

| Engine | N scored | Median time | Mean time |
|--------|:--------:|:-----------:|:---------:|
| duckdb | 100 | 0.049 s | 0.394 s |
| polars | 100 | 0.106 s | 0.466 s |
| plink2 | 96 | 0.603 s | 0.703 s |

| Engine pair | N PGS | Pearson r | Max \|Δ score\| |
|-------------|:-----:|:---------:|:---------------:|
| duckdb ↔ polars | 100 | 1.000000 | < 1.1 × 10⁻¹³ |
| polars ↔ plink2 | 96 | 0.999859 | 21.4 |

---

## Reproducing the benchmark

The script downloads the VCF, the scoring files and PLINK2 on first use, then
reuses them.

```bash
uv sync --all-packages

# The published set: PGS000006-PGS000106 without PGS000085
uv run python just-prs/benchmarks/benchmark_engines.py --output just-prs/benchmarks/results/<date>

# Other IDs, or without PLINK2
uv run python just-prs/benchmarks/benchmark_engines.py --pgs-ids PGS000001-PGS000050 --exclude ""
uv run python just-prs/benchmarks/benchmark_engines.py --skip-plink2

# Attribute every remaining just-prs vs PLINK2 difference to a cause
uv run python just-prs/benchmarks/attribute_residual.py \
    --run just-prs/benchmarks/results/<date> \
    --genotypes <cache>/benchmark/vcf/antonkulaga_normalized_autosomes.parquet \
    --pvar <cache>/benchmark/pgen/benchmark_pgen_pass.pvar
```

### Outputs

| File | Contents |
|------|----------|
| `benchmark_results.csv` | one row per PGS ID × engine: time, memory, score, matched, `variants_allele_mismatch`, error |
| `benchmark_per_pgs.csv` | one row per PGS ID: each engine's score and matched count, and the deltas |
| `benchmark_validation.csv` | Pearson r and max \|Δ\| per engine pair |
| `benchmark_metadata.csv` | versions (incl. the just-prs git commit), PLINK2 build and flags, VCF md5, machine |
| `pgs_ids.txt`, `scoring_files.csv` | the exact IDs; scoring-file md5 and harmonization date |
| `plink2_logs/` | per ID: PLINK2 `.log` and captured console output; `vcf_to_pgen.log` |
| `residual_by_cause.csv`, `residual_per_pgs.csv`, `residual_per_variant.csv` | from `attribute_residual.py` |
