# Validation and Reproducibility Testing

## PRS score validation against PLINK2

Our PRS computation is validated against [PLINK2](https://www.cog-genomics.org/plink/2.0/) `--score` on real genomic data. The integration test suite downloads a whole-genome VCF from Zenodo, computes PRS for multiple GRCh38 scores using both `just-prs` and PLINK2, and asserts agreement:

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

## Percentile cross-validation against PLINK2

Scores with allele frequency data (PGS000004, PGS000005, PGS000006, PGS000010, PGS000011) are additionally validated for:

- **Score agreement with PLINK2** -- the raw PRS score matches PLINK2 within 5% relative difference
- **Theoretical statistics** -- mean and standard deviation computed from allele frequencies match manual row-by-row computation to within 1e-10
- **Percentile consistency** -- `percentile = Phi((score - mean) / std) * 100` is verified exactly
- **Boundary correctness** -- monomorphic sites (allele frequency 0.0 or 1.0) are excluded from the theoretical distribution

Scores without allele frequencies (PGS000001, PGS000002) are verified to return `None` for percentile, rather than producing spurious values.

```bash
uv run pytest tests/test_percentile.py -v
```

## Cleanup pipeline validation

The cleanup pipeline is tested against the full live PGS Catalog metadata (~5,000+ scores downloaded via EBI FTP):

- **Column renaming** -- all 12 verbose raw column names are mapped to snake_case, and unmapped columns are dropped
- **Genome build normalization** -- all 9 raw build variants (hg19, hg37, hg38, NCBI36, hg18, NCBI35, GRCh37, GRCh38, NR) are mapped to canonical forms
- **Metric string parsing** -- performance metric strings in 5 formats are parsed into 20 structured numeric columns
- **Performance flattening** -- evaluation sample sets are joined, and one best row per PGS ID is selected by sample size with European-ancestry preference
- **PRSCatalog integration** -- search, score info lookup, best performance, and percentile estimation are tested against the cleaned data

```bash
uv run pytest tests/test_cleanup.py -v
```

## Full test suite

All tests use real data and real tools. Nothing is mocked.

| Dependency | How it is obtained |
|---|---|
| Test VCF (whole-genome) | Auto-downloaded from [Zenodo](https://zenodo.org/records/18370498) on first run |
| PLINK2 binary | Auto-downloaded from [plink2-assets](https://s3.amazonaws.com/plink2-assets) for the host platform (Linux x86_64/aarch64, macOS arm64/x86_64, Windows) |
| PGS scoring files | Fetched from EBI FTP and cached locally |
| PGS Catalog metadata | Bulk CSVs downloaded from EBI FTP (single HTTP request per sheet) |

Run the full suite:

```bash
uv run pytest tests/ -v
```
