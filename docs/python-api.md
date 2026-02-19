# Python API

## VCF normalization (`just_prs.normalize`)

`normalize_vcf()` reads a VCF file via polars-bio, normalizes column names, strips the `chr` prefix from chromosomes, computes a sorted `genotype` List[Str] from the GT field, applies configurable quality filters, and writes a zstd-compressed Parquet file. The result can be used directly for PRS computation via the `genotypes_lf` parameter of `compute_prs()`.

```python
from pathlib import Path
from just_prs import normalize_vcf, VcfFilterConfig

# Basic normalization with PASS filter
config = VcfFilterConfig(pass_filters=["PASS", "."])
normalize_vcf(Path("sample.vcf.gz"), Path("sample.parquet"), config=config)

# Full quality filtering
config = VcfFilterConfig(
    pass_filters=["PASS", "."],
    min_depth=10,
    min_qual=30.0,
    sex="Female",  # warns if chrY variants exist
)
output = normalize_vcf(
    Path("sample.vcf.gz"),
    Path("filtered.parquet"),
    config=config,
    format_fields=["GT", "DP", "GQ"],
)

# Use the normalized parquet for PRS computation
import polars as pl
from just_prs.prs import compute_prs

genotypes_lf = pl.scan_parquet(output)
result = compute_prs(
    vcf_path="sample.vcf.gz",
    scoring_file="PGS000001",
    genome_build="GRCh38",
    genotypes_lf=genotypes_lf,  # skips re-reading the VCF
)
```

`VcfFilterConfig` fields:

| Field | Type | Description |
|-------|------|-------------|
| `pass_filters` | `list[str] \| None` | Keep rows where FILTER is in this list (e.g. `["PASS", "."]`) |
| `min_depth` | `int \| None` | Keep rows where DP >= this value |
| `min_qual` | `float \| None` | Keep rows where QUAL >= this value |
| `sex` | `str \| None` | Sample sex (`"Male"` / `"Female"`). When Female, warns if chrY variants exist |

## PRSCatalog — search, compute, and percentile (`just_prs.prs_catalog`)

`PRSCatalog` is the recommended high-level interface. It persists 3 cleaned parquet files locally and loads them on access using a 3-tier fallback chain: local files -> HuggingFace pull -> raw FTP download + cleanup. All lookups, searches, and PRS computations use cleaned data with no per-score REST API calls.

```python
from just_prs import PRSCatalog

catalog = PRSCatalog()  # uses platformdirs user cache dir by default

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

### Genotype expression helper

`genotype_expr()` builds a Polars expression that resolves GT indices (e.g. `"0/1"`) into actual allele strings using REF/ALT columns, returning a sorted `List[Utf8]`. This is used internally by `normalize_vcf()` but can also be applied to any DataFrame with GT/ref/alt columns:

```python
from just_prs.normalize import genotype_expr
import polars as pl

df = pl.DataFrame({
    "GT": ["0/1", "1/1", "./."],
    "ref": ["A", "C", "G"],
    "alt": ["T", "G", "A"],
})
df = df.with_columns(genotype_expr())
# genotype: [["A","T"], ["G","G"], []]
```

## Low-level PRS computation (`just_prs.prs`)

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

## REST API client (`just_prs.catalog`)

```python
from just_prs.catalog import PGSCatalogClient

with PGSCatalogClient() as client:
    score = client.get_score("PGS000001")
    results = client.search_scores("breast cancer", limit=10)
    trait = client.get_trait("EFO_0001645")
    for score in client.iter_all_scores(page_size=100):
        print(score.id, score.trait_reported)
```

## Bulk FTP downloads (`just_prs.ftp`)

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

## HuggingFace sync (`just_prs.hf`)

```python
from just_prs.hf import push_cleaned_parquets, pull_cleaned_parquets
from pathlib import Path

# Push cleaned parquets to HF dataset repo
push_cleaned_parquets(Path("./output/pgs_metadata"))  # default: just-dna-seq/polygenic_risk_scores

# Pull cleaned parquets from HF
downloaded = pull_cleaned_parquets(Path("./local_cache"))
# [Path('local_cache/scores.parquet'), Path('local_cache/performance.parquet'), ...]
```

## Cleanup pipeline (`just_prs.cleanup`)

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
