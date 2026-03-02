# Validation and Reproducibility Testing

`just-prs` provides a pure Python PRS scoring engine (pgenlib + polars + numpy) that produces **identical results** to the PLINK2 `--score` command. This is validated at two levels:

1. **Individual VCF scoring** — PRS computed from a real whole-genome VCF matches PLINK2 to within floating-point precision (relative differences < 5e-7)
2. **Reference panel scoring** — PRS computed across all 3,202 individuals of the 1000 Genomes reference panel matches PLINK2 with Pearson r = 1.0

Both validations run against real data with no mocking. The PLINK2 binary is auto-downloaded for the host platform during testing.

## Individual VCF PRS validation

The integration test suite downloads a whole-genome VCF from Zenodo, computes PRS for multiple GRCh38 scores using both `just-prs` and PLINK2, and asserts agreement:

| PGS ID | just-prs | PLINK2 | Relative diff | Variants matched |
|--------|----------|--------|---------------|-----------------|
| PGS000001 | 0.030123 | 0.030123 | 6.5e-7 | 51 / 77 |
| PGS000002 | -0.137089 | -0.137089 | 1.1e-7 | 51 / 77 |
| PGS000003 | 0.588127 | 0.588127 | 8.1e-9 | 51 / 77 |
| PGS000004 | -0.7158 | -0.7158 | 3.1e-16 | 170 / 313 |
| PGS000005 | -0.8903 | -0.8903 | 5.0e-16 | 170 / 313 |

All differences are within floating-point precision.

```bash
uv run pytest tests/test_plink.py -v
```

## Reference panel cross-validation (1000 Genomes)

The pgenlib + polars engine (`compute_reference_prs_polars`) is validated against PLINK2 `--score` on the PGS Catalog 1000 Genomes reference panel (3,202 samples, 5 superpopulations). For each tested PGS ID:

- **Per-sample Pearson correlation** is 1.0 (perfect agreement across all 3,202 individuals)
- **Maximum per-sample absolute difference** is < 1e-6
- **All 5 superpopulations** (AFR, AMR, EAS, EUR, SAS) are present with correct sample counts
- **Score variance** is non-trivial (scores are not all identical)
- **Distribution statistics** (mean, std, percentiles) are identical between engines

This validation can be run interactively via the CLI:

```bash
# Compare both engines side-by-side for any PGS ID
prs reference compare PGS000001

# Automated validation across multiple PGS IDs
prs reference test-score --pgs-ids PGS000001,PGS000002,PGS000004,PGS000010
```

The test suite covers 4 PGS IDs with different variant counts and allele patterns:

```bash
uv run pytest tests/test_reference_plink2.py -v
```

## Percentile cross-validation

Scores with allele frequency data (PGS000004, PGS000005, PGS000006, PGS000010, PGS000011) are additionally validated for:

- **Score agreement with PLINK2** — the raw PRS score matches PLINK2 within 5% relative difference
- **Theoretical statistics** — mean and standard deviation computed from allele frequencies match manual row-by-row computation to within 1e-10
- **Percentile consistency** — `percentile = Phi((score - mean) / std) * 100` is verified exactly
- **Boundary correctness** — monomorphic sites (allele frequency 0.0 or 1.0) are excluded from the theoretical distribution

Scores without allele frequencies (PGS000001, PGS000002) are verified to return `None` for percentile, rather than producing spurious values.

```bash
uv run pytest tests/test_percentile.py -v
```

## Cleanup pipeline validation

The cleanup pipeline is tested against the full live PGS Catalog metadata (~5,000+ scores downloaded via EBI FTP):

- **Column renaming** — all 12 verbose raw column names are mapped to snake_case, and unmapped columns are dropped
- **Genome build normalization** — all 9 raw build variants (hg19, hg37, hg38, NCBI36, hg18, NCBI35, GRCh37, GRCh38, NR) are mapped to canonical forms
- **Metric string parsing** — performance metric strings in 5 formats are parsed into 20 structured numeric columns
- **Performance flattening** — evaluation sample sets are joined, and one best row per PGS ID is selected by sample size with European-ancestry preference
- **PRSCatalog integration** — search, score info lookup, best performance, and percentile estimation are tested against the cleaned data

```bash
uv run pytest tests/test_cleanup.py -v
```

## Full test suite

All tests use real data and real tools. Nothing is mocked.

| Dependency | How it is obtained |
|---|---|
| Test VCF (whole-genome) | Auto-downloaded from [Zenodo](https://zenodo.org/records/18370498) on first run |
| PLINK2 binary | Auto-downloaded from [plink2-assets](https://s3.amazonaws.com/plink2-assets) for the host platform (Linux x86_64/aarch64, macOS arm64/x86_64, Windows) |
| 1000G reference panel | ~7 GB tarball from EBI FTP (for reference panel tests) |
| PGS scoring files | Fetched from EBI FTP and cached locally |
| PGS Catalog metadata | Bulk CSVs downloaded from EBI FTP (single HTTP request per sheet) |

Run the full suite:

```bash
uv run pytest tests/ -v
```
