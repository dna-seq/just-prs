# Validation against PLINK2

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
