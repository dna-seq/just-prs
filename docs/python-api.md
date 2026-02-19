# Python API

## PRSCatalog — search, compute, and percentile (`just_prs.prs_catalog`)

`PRSCatalog` is the recommended high-level interface. It persists 3 cleaned parquet files locally and loads them on access using a 3-tier fallback chain: local files -> HuggingFace pull -> raw FTP download + cleanup. All lookups, searches, and PRS computations use cleaned data with no per-score REST API calls.

```python
from just_prs import PRSCatalog

catalog = PRSCatalog()  # uses ~/.cache/just-prs by default

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
