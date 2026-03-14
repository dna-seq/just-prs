# Python API

## Recommended workflow: normalize once, compute via LazyFrame

The recommended workflow is to normalize a VCF to Parquet once, then use a polars LazyFrame for all subsequent PRS computations. This is memory-efficient and avoids re-reading the raw VCF for each score:

```python
import polars as pl
from pathlib import Path
from just_prs import PRSCatalog, normalize_vcf, VcfFilterConfig
from just_prs.prs import compute_prs

# 1. Normalize VCF to Parquet (one-time step)
config = VcfFilterConfig(pass_filters=["PASS", "."], min_depth=10)
parquet_path = normalize_vcf(Path("sample.vcf.gz"), Path("sample.parquet"), config=config)

# 2. Load as LazyFrame — reuse for all PRS computations
genotypes_lf = pl.scan_parquet(parquet_path)

# 3. Compute PRS (LazyFrame avoids re-reading the VCF)
result = compute_prs(
    vcf_path="sample.vcf.gz",
    scoring_file="PGS000001",
    genome_build="GRCh38",
    genotypes_lf=genotypes_lf,
)

# 4. Batch computation via PRSCatalog (also accepts genotypes LazyFrame)
catalog = PRSCatalog()
results = catalog.compute_prs_batch(
    vcf_path=Path("sample.vcf.gz"),
    pgs_ids=["PGS000001", "PGS000002", "PGS000003"],
)
```

## VCF normalization (`just_prs.normalize`)

`normalize_vcf()` reads a VCF file via polars-bio, normalizes column names, strips the `chr` prefix from chromosomes, computes a sorted `genotype` List[Str] from the GT field, applies configurable quality filters, and writes a zstd-compressed Parquet file. The returned path can be loaded as a LazyFrame for PRS computation.

```python
from pathlib import Path
from just_prs import normalize_vcf, VcfFilterConfig

# Basic normalization with PASS filter
config = VcfFilterConfig(pass_filters=["PASS", "."])
parquet_path = normalize_vcf(Path("sample.vcf.gz"), Path("sample.parquet"), config=config)

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

# Use the normalized parquet for PRS computation via LazyFrame (preferred)
import polars as pl
from just_prs.prs import compute_prs

genotypes_lf = pl.scan_parquet(output)
result = compute_prs(
    vcf_path="sample.vcf.gz",
    scoring_file="PGS000001",
    genome_build="GRCh38",
    genotypes_lf=genotypes_lf,
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
pct, method = catalog.percentile(prs_score=1.5, pgs_id="PGS000014")
pct, method = catalog.percentile(prs_score=1.5, pgs_id="PGS000014", panel="1000g")
pct, method = catalog.percentile(prs_score=1.5, pgs_id="PGS000014", mean=0.0, std=1.0)

# Reference distributions (panel-aware)
dist_lf = catalog.reference_distributions(panel="1000g")
dist_lf = catalog.reference_distributions(panel="hgdp_1kg")

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

## PLINK2 binary format operations (`just_prs.reference`)

Pure Python functions for reading PLINK2 binary files (.pgen/.pvar.zst/.psam), matching scoring variants, and computing PRS using `pgenlib` + polars + numpy. These functions replace common PLINK2 CLI operations while producing identical results (validated with Pearson r = 1.0 across 3,202 samples — see [validation](validation.md)).

**Advantages over PLINK2 CLI:**
- No external binary to install — works anywhere Python runs
- Returns polars DataFrames directly instead of text files
- Modular — use individual functions (`parse_pvar`, `read_pgen_genotypes`, `match_scoring_to_pvar`) independently for custom analyses
- Caches parsed `.pvar.zst` as parquet for ~14x faster subsequent reads (~0.5s vs ~7s)
- Reuses caches across multiple PGS IDs in batch scoring

All functions raise `ReferencePanelError` (importable from `just_prs`) when required files are missing.

### Parse variant files (`.pvar.zst`)

```python
from just_prs import parse_pvar

# Decompresses .pvar.zst and caches as parquet (~0.5s on subsequent reads)
pvar_df = parse_pvar(Path("/path/to/panel.pvar.zst"))
# DataFrame: variant_idx (u32), chrom (str), POS (i64), REF (str), ALT (str)

print(f"{pvar_df.height:,} variants loaded")
print(pvar_df.filter(pl.col("chrom") == "11").head())
```

### Parse sample files (`.psam`)

```python
from just_prs import parse_psam

psam_df = parse_psam(Path("/path/to/panel.psam"))
# DataFrame: iid (str), superpop (str), population (str)

print(psam_df.group_by("superpop").len())
```

### Read genotypes from `.pgen` files

```python
from just_prs import parse_pvar, parse_psam, read_pgen_genotypes
from pathlib import Path
import polars as pl

pvar_df = parse_pvar(Path("panel.pvar.zst"))
psam_df = parse_psam(Path("panel.psam"))

# Select variants of interest (e.g. chromosome 11, position range)
region = pvar_df.filter(
    (pl.col("chrom") == "11")
    & (pl.col("POS").is_between(69000000, 70000000))
)

# Read genotypes — returns int8 numpy array (variants x samples)
# Values: 0=hom-ref, 1=het, 2=hom-alt, -9=missing
geno = read_pgen_genotypes(
    pgen_path=Path("panel.pgen"),
    pvar_zst_path=Path("panel.pvar.zst"),
    variant_indices=region["variant_idx"].cast(pl.UInt32).to_numpy(),
    n_samples=psam_df.height,
)
print(f"Shape: {geno.shape}")  # (n_variants, n_samples)
```

### Match scoring file variants to `.pvar`

```python
from just_prs import parse_pvar, match_scoring_to_pvar
from just_prs.prs import _normalize_scoring_columns
from just_prs.scoring import parse_scoring_file

pvar_df = parse_pvar(Path("panel.pvar.zst"))
scoring_lf = parse_scoring_file(Path("PGS000001_hmPOS_GRCh38.txt.gz"))
scoring_df = _normalize_scoring_columns(scoring_lf).collect()

matched = match_scoring_to_pvar(pvar_df, scoring_df)
# DataFrame with effect_is_alt column indicating allele orientation
print(f"Matched {matched.height} of {scoring_df.height} scoring variants")
```

### Score a PGS against any .pgen dataset

```python
from just_prs import compute_reference_prs_polars
from pathlib import Path

scores_df = compute_reference_prs_polars(
    pgs_id="PGS000001",
    scoring_file=Path("PGS000001_hmPOS_GRCh38.txt.gz"),
    ref_dir=Path("/path/to/pgen_dir"),  # any dir with .pgen/.pvar.zst/.psam
    out_dir=Path("/tmp/output"),
    genome_build="GRCh38",
)
# DataFrame: iid, superpop, population, score, pgs_id
```

### Aggregate into population distributions

```python
from just_prs import aggregate_distributions

dist_df = aggregate_distributions(scores_df)
# DataFrame: pgs_id, superpopulation, mean, std, n, median, p5, p25, p75, p95
```

### Batch reference scoring (`compute_reference_prs_batch`)

Score multiple PGS IDs against a reference panel in a single call. Downloads scoring files, computes PRS for each, tracks failures and quality flags, and produces aggregated distributions:

```python
from pathlib import Path
from just_prs import compute_reference_prs_batch, reference_panel_dir
from just_prs.scoring import resolve_cache_dir

cache_dir = resolve_cache_dir()
ref_dir = reference_panel_dir(cache_dir, panel="1000g")

result = compute_reference_prs_batch(
    pgs_ids=["PGS000001", "PGS000002", "PGS000003"],
    ref_dir=ref_dir,
    cache_dir=cache_dir,
    genome_build="GRCh38",
    panel="1000g",
    skip_existing=True,
    match_rate_threshold=0.1,
)

# Inspect outcomes per PGS ID
for o in result.outcomes:
    print(f"{o.pgs_id}: {o.status} (n={o.n_samples}, mean={o.score_mean})")

# Aggregated distributions DataFrame
print(result.distributions_df)
# pgs_id, superpopulation, mean, std, n, median, p5, p25, p75, p95

# Quality report DataFrame
print(result.quality_df)
# pgs_id, status, n_samples, score_mean, score_std, elapsed_sec, error

# Files are also written to cache:
#   cache_dir/percentiles/1000g_distributions.parquet
#   cache_dir/percentiles/1000g_quality.parquet
#   cache_dir/reference_scores/1000g/{pgs_id}/scores.parquet
```

`BatchScoringResult` fields:

| Field | Type | Description |
|-------|------|-------------|
| `panel` | `str` | Reference panel identifier |
| `scores_df` | `pl.DataFrame` | All successful per-individual scores concatenated |
| `distributions_df` | `pl.DataFrame` | Aggregated per-superpopulation distribution stats |
| `outcomes` | `list[ScoringOutcome]` | Per-PGS-ID status, variant counts, timing, errors |
| `quality_df` | `pl.DataFrame` | Same data as outcomes but as a polars DataFrame |

Quality status values: `ok`, `failed`, `low_match`, `zero_variance`.

### Ancestry-matched percentile estimation

```python
from just_prs import ancestry_percentile
import polars as pl

distributions_lf = pl.scan_parquet("1000g_distributions.parquet")
pct = ancestry_percentile(
    prs_score=0.00123,
    pgs_id="PGS000001",
    superpopulation="EUR",
    distributions_lf=distributions_lf,
)
print(f"Percentile: {pct}%")
```

### Download and manage reference panels

Two reference panels are supported:

| Panel ID | Description |
|----------|-------------|
| `1000g` (default) | 1000 Genomes Project (3,202 individuals, 5 superpopulations) |
| `hgdp_1kg` | HGDP + 1000 Genomes merged panel |

```python
from just_prs import download_reference_panel, reference_panel_dir, REFERENCE_PANELS

# List available panels
for name, info in REFERENCE_PANELS.items():
    print(f"{name}: {info['description']}")

# Check where a panel would be stored
panel_dir = reference_panel_dir(panel="1000g")

# Download if not present (~7 GB for 1000g, ~15 GB for hgdp_1kg)
dest = download_reference_panel(panel="1000g")
```

## Scoring file parquet cache (`just_prs.scoring`)

PGS scoring files (`.txt.gz`) are transparently cached as parquet on first parse, with [PGS Catalog spec](https://www.pgscatalog.org/downloads/#dl_ftp_scoring)-driven schema overrides and embedded header metadata. Subsequent reads are 5-60x faster (no gzip decompression) and files are ~17% smaller with zstd-9 compression.

The cache is shared between the pipeline and individual `compute_prs()` / `load_scoring()` calls — whoever creates the parquet first wins, and all subsequent consumers benefit. When a parquet cache exists, `load_scoring()` skips the `.txt.gz` download entirely.

```python
from pathlib import Path
from just_prs.scoring import (
    SCORING_FILE_SCHEMA,
    parse_scoring_file,
    scoring_parquet_path,
    read_scoring_header,
    load_scoring,
)

# SCORING_FILE_SCHEMA: dict of column name → pl.DataType for all 30+ PGS Catalog columns
# Used automatically by parse_scoring_file() and stream_scoring_file()

# Parse a scoring file (creates parquet cache on first call)
lf = parse_scoring_file(Path("PGS000001_hmPOS_GRCh38.txt.gz"))
# Subsequent calls return pl.scan_parquet() on the cached parquet (fast path)

# Also works with a .parquet path directly
lf = parse_scoring_file(Path("PGS000001_hmPOS_GRCh38.parquet"))

# Check where the parquet cache would live for a PGS ID
pq_path = scoring_parquet_path("PGS000001", Path("~/.cache/just-prs/scores"), "GRCh38")
# Path('~/.cache/just-prs/scores/PGS000001_hmPOS_GRCh38.parquet')

# Read PGS Catalog header metadata (instant from parquet, fallback to .txt.gz)
header = read_scoring_header(Path("PGS000001_hmPOS_GRCh38.parquet"))
# {'pgs_id': 'PGS000001', 'trait_reported': 'Breast cancer', 'genome_build': 'GRCh38', ...}

# load_scoring() checks parquet cache first, skips .txt.gz download if cached
lf = load_scoring("PGS000001")  # no network I/O if parquet already exists
```

Cache layout in `<cache>/scores/`:

```text
PGS000001_hmPOS_GRCh38.txt.gz     # original download (may be deleted by pipeline)
PGS000001_hmPOS_GRCh38.parquet    # parquet cache with embedded header metadata
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

## Quality assessment helpers (`just_prs.quality`)

Pure functions for classifying PRS result quality, interpreting results, and formatting performance metrics. No Reflex dependency — usable from any script, notebook, or UI:

```python
from just_prs import classify_model_quality, interpret_prs_result, format_effect_size, format_classification

# Classify model quality from match rate + AUROC
label, color = classify_model_quality(match_rate=0.85, auroc=0.72)
# ("High", "green")

# Produce a human-readable interpretation
interp = interpret_prs_result(percentile=72.3, match_rate=0.85, auroc=0.72)
# {"quality_label": "High", "quality_color": "green", "summary": "Estimated percentile: 72.3% ..."}

# Format effect size from a cleaned performance row dict
effect = format_effect_size({"or_estimate": 1.55, "or_ci_lower": 1.52, "or_ci_upper": 1.58})
# "OR=1.55 [1.52-1.58]"

# Format classification metric
auroc = format_classification({"auroc_estimate": 0.721, "auroc_ci_lower": 0.690, "auroc_ci_upper": 0.752})
# "AUROC=0.721 [0.690-0.752]"
```

| Function | Input | Output |
|----------|-------|--------|
| `classify_model_quality(match_rate, auroc)` | Match rate (0-1), optional AUROC | `(label, color)` tuple: High/Moderate/Low/Very Low |
| `interpret_prs_result(percentile, match_rate, auroc)` | Percentile (optional), match rate, AUROC | Dict with `quality_label`, `quality_color`, `summary` |
| `format_effect_size(perf_row)` | Dict with OR/HR/Beta estimate + CI/SE keys | Formatted string like `"OR=1.55 [1.52-1.58]"` |
| `format_classification(perf_row)` | Dict with AUROC/C-index estimate + CI keys | Formatted string like `"AUROC=0.721 [0.690-0.752]"` |

## Reusable PRS UI components (`prs_ui`)

The `prs-ui` package provides reusable [Reflex](https://reflex.dev/) components for PRS computation. Each component function accepts a `state` class parameter, so the same components work with any concrete state inheriting `PRSComputeStateMixin`.

### Quick integration

```python
import polars as pl
import reflex as rx
from reflex_mui_datagrid import LazyFrameGridMixin
from prs_ui import PRSComputeStateMixin, prs_section


class MyAppState(rx.State):
    genome_build: str = "GRCh38"
    cache_dir: str = ""
    status_message: str = ""


class PRSState(PRSComputeStateMixin, LazyFrameGridMixin, MyAppState):
    """Concrete PRS state for the host app."""

    def load_genotypes(self, parquet_path: str) -> None:
        lf = pl.scan_parquet(parquet_path)
        self.set_prs_genotypes_lf(lf)  # preferred: provide a LazyFrame
        self.prs_genotypes_path = parquet_path


def prs_page() -> rx.Component:
    return prs_section(PRSState)
```

### Providing genotype data

The mixin resolves genotype data in this order:

1. **LazyFrame (preferred)** — call `set_prs_genotypes_lf(lf)` with `pl.scan_parquet(path)`. This is memory-efficient and avoids redundant I/O when computing multiple scores.
2. **Parquet path (fallback)** — set `prs_genotypes_path` to a string path. The mixin calls `pl.scan_parquet()` internally if no LazyFrame was provided.

### Available components

| Component | Description |
|-----------|-------------|
| `prs_section(state)` | Complete PRS section: build selector + score grid + compute button + progress + results |
| `prs_build_selector(state)` | Genome build dropdown (GRCh37/GRCh38) |
| `prs_scores_selector(state)` | MUI DataGrid for score selection with checkboxes, filtering, and "Select Filtered" |
| `prs_compute_button(state)` | Compute button with disclaimer callout |
| `prs_progress_section(state)` | Progress bar and status text during computation |
| `prs_results_table(state)` | Results table with quality badges, interpretation cards, and CSV download |

### State contract

The host app's state hierarchy must provide these reactive vars (usually via a shared parent state):

| Var | Type | Description |
|-----|------|-------------|
| `genome_build` | `str` | Current genome build (`"GRCh37"` or `"GRCh38"`) |
| `cache_dir` | `str` | Cache directory path (for scoring file downloads) |
| `status_message` | `str` | Status text displayed in progress sections |

The concrete state class must inherit from both `PRSComputeStateMixin` and `LazyFrameGridMixin` (in that order), plus any shared parent state.

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
