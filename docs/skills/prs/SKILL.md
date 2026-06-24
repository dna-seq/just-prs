---
name: prs
description: Compute polygenic risk scores, generate charts, and interpret results using just-prs. Use when the user asks about PRS, genetic risk, traits, PGS scores, bell curves, or percentile charts.
argument-hint: [query or PGS ID or trait name]
allowed-tools: Bash(uvx just-prs*) Bash(uv run prs *) Bash(uv run python *)
---

# just-prs: Polygenic Risk Score Computation & Visualization

You are an expert at computing and interpreting polygenic risk scores using `just-prs`.
The tool is available as a CLI (`uvx just-prs` for one-off use, `uv run prs` inside the workspace)
and as a Python API for charts and advanced workflows.

## Installation

For one-off use (no install needed):
```bash
uvx just-prs --help
uvx just-prs catalog scores search -t "diabetes"
```

For repeated use:
```bash
uv pip install just-prs          # altair charts (HTML/JSON) included
uv pip install "just-prs[viz]"   # adds vl-convert for PNG/SVG export
```

## Quick reference: CLI commands

### Search the PGS Catalog

```bash
# Search scores by trait, gene, or keyword
uvx just-prs catalog scores search -t "$ARGUMENTS" -n 25

# Get detailed info about a specific score
uvx just-prs catalog scores info PGS000001

# Search traits (returns EFO IDs for batch workflows)
uvx just-prs catalog traits search -t "type 2 diabetes"

# Get trait details including associated PGS IDs
uvx just-prs catalog traits info EFO_0004278
```

### Normalize a VCF

Normalization strips chr prefixes, computes genotype columns, applies quality filters, and writes Parquet.
Do this once per VCF; reuse the parquet for all subsequent scoring.

```bash
uvx just-prs normalize \
  --vcf /path/to/sample.vcf.gz \
  --output normalized.parquet \
  --pass-filters "PASS,." \
  --min-depth 10
```

### Compute PRS (individual scores)

`--vcf` accepts a file path or a named alias (see VCF aliases below).

```bash
# Single score
uvx just-prs compute --vcf /path/to/sample.vcf.gz --pgs-id PGS000001

# Multiple scores at once (comma-separated)
uvx just-prs compute --vcf /path/to/sample.vcf.gz --pgs-id PGS000001,PGS000002,PGS000003

# Using an alias instead of a full path
uvx just-prs compute --vcf anton --pgs-id PGS000001

# With explicit build and JSON output
uvx just-prs compute --vcf /path/to/sample.vcf.gz --pgs-id PGS000001 --build GRCh37 --output results.json
```

### Normalize a consumer genotyping array (23andMe, AncestryDNA)

```bash
uvx just-prs normalize-array \
  --array-path /path/to/23andme_raw.txt \
  --output normalized_array.parquet \
  --genome-build GRCh37
```

### Generate charts from the CLI

Altair is a core dependency — HTML and JSON chart export works out of the box with
`uvx just-prs`. For PNG/SVG raster export, use `uvx "just-prs[viz]"`.
Reference distributions are loaded from cache automatically (pulled from HuggingFace on first use).

All plot commands except `strip` accept `--vcf` (path or alias) to auto-compute PRS
and plot in one step. This is the easiest way to get a chart from a VCF.

PRS results are **cached per (VCF, PGS ID, build, ancestry)** — repeated plotting is
instant. Use `--no-cache` to force recomputation. Trait matching is **exact by default**
(case-insensitive); use `--fuzzy` to match all traits containing the query string.

```bash
# One-liner: compute + plot for a trait using an alias
uvx just-prs plot trait "type 1 diabetes" --vcf livia -o t1d.html --show-table
uvx just-prs plot trait BMI --vcf anton -o bmi.html --show-table

# Fuzzy trait matching (substring)
uvx just-prs plot trait thrombosis --vcf anton -o dvt.html --fuzzy --show-table

# Overlay all 5 population reference curves on the trait chart
uvx just-prs plot trait BMI --vcf anton -o bmi.html --show-table --all-ancestries

# Select specific populations to overlay
uvx just-prs plot trait BMI --vcf anton -o bmi.html --ancestries EUR,AFR,EAS

# Single PGS bell curve with auto-computed score
uvx just-prs plot bell-curve PGS000001 --vcf anton -o bell.html
uvx just-prs plot bell-curve PGS000001 -o bell.html -a AFR --user-score 0.274

# All five population curves overlaid (single PGS ID)
uvx just-prs plot multi-ancestry PGS000001 --vcf livia -o multi.html
uvx just-prs plot multi-ancestry PGS000001 -o multi.html --ancestries EUR,AFR,EAS

# Trait chart from pre-computed results JSON
uvx just-prs plot trait "type 2 diabetes" -o t2d.html --results my_results.json

# Percentile strip chart (requires computed results JSON)
uvx just-prs plot strip results.json -o strip.html --title "My PRS Report"

# Force recompute (bypass result cache)
uvx just-prs plot trait BMI --vcf anton -o bmi.html --no-cache

# PNG/SVG export requires the viz extra
uvx "just-prs[viz]" plot bell-curve PGS000001 -o bell.png
uvx "just-prs[viz]" plot trait BMI -o bmi.svg --show-table
```

Inside the workspace, use `uv run prs plot ...` instead of `uvx`.
Output format is auto-detected from file extension: `.html` (interactive with tooltips),
`.json` (Vega-Lite spec), `.png` / `.svg` (requires `just-prs[viz]`).

All plot commands accept `--width` and `--height` for sizing, `--panel` for reference panel
selection (1000g or hgdp_1kg), and `--cache-dir` to override the cache location.

### VCF aliases

Named shortcuts for frequently used VCF files. Built-in aliases `anton` and `livia`
point to the test genomes in the cache directory and **auto-download from Zenodo**
on first use (~482 MB and ~349 MB respectively).

```bash
# List all aliases
uvx just-prs alias list

# Add your own
uvx just-prs alias set mygenome /path/to/my/sample.vcf.gz

# Remove a user alias
uvx just-prs alias remove mygenome

# Use anywhere --vcf is accepted
uvx just-prs compute --vcf mygenome --pgs-id PGS000001
uvx just-prs plot trait BMI --vcf mygenome -o bmi.html
```

User aliases are stored in `~/.cache/just-prs/vcf_aliases.json`.

## Python API: batch computation + charts

For trait-level analysis and chart generation, use the Python API.
Always run Python through `uv run python` in the workspace.

### Batch PRS computation for a trait

```python
import polars as pl
from pathlib import Path
from just_prs import PRSCatalog

catalog = PRSCatalog()

# Search for scores related to a trait
scores = catalog.search("$ARGUMENTS", genome_build="GRCh38").collect()
pgs_ids = scores["pgs_id"].to_list()
print(f"Found {len(pgs_ids)} scores for '$ARGUMENTS'")

# Compute batch PRS
results = catalog.compute_prs_batch(
    vcf_path=Path("/path/to/sample.vcf.gz"),
    pgs_ids=pgs_ids[:15],  # top 15 by catalog order
)

# Build result dicts with percentiles
result_dicts = []
for r in results:
    pctl_info = catalog.percentile(r.score, r.pgs_id, superpopulation="EUR")
    result_dicts.append({
        "pgs_id": r.pgs_id,
        "score": r.score,
        "percentile": pctl_info.percentile if pctl_info else None,
        "z_score": pctl_info.z_score if pctl_info else None,
        "match_rate": r.match_rate,
    })

# Save as JSON for chart generation
import json
with open("results.json", "w") as f:
    json.dump(result_dicts, f, indent=2)
```

### Load reference distributions (needed for charts)

```python
import polars as pl
from just_prs.scoring import resolve_cache_dir

cache = resolve_cache_dir()
distributions = pl.read_parquet(cache / "percentiles/1000g_distributions.parquet")
quality = pl.read_parquet(cache / "percentiles/1000g_quality.parquet")
```

## Chart generation (`just_prs.viz`)

Altair is a core dependency. All chart functions return Altair chart objects.
Use `save_chart()` to export as HTML (interactive with tooltips) or JSON out of the box,
or PNG/SVG with `just-prs[viz]` (adds vl-convert-python).

### `plot_prs_bell_curve` — Single model, single ancestry

```python
from just_prs.viz import plot_prs_bell_curve, save_chart

chart = plot_prs_bell_curve(
    pgs_id="PGS000001",
    distributions_df=distributions,  # from 1000g_distributions.parquet
    user_score=0.274,                # from compute_prs result
    ancestry="EUR",                  # AFR, AMR, EAS, EUR, SAS
    width=560,                       # configurable
    height=220,                      # configurable
)
save_chart(chart, Path("bell_curve.png"))   # also .svg, .html, .json
```

Shows a Gaussian bell curve for the reference population with percentile markers (5th, 25th, 50th, 75th, 95th) and the user's score as a red vertical line with percentile label.

### `plot_prs_multi_ancestry` — All populations overlaid

```python
from just_prs.viz import plot_prs_multi_ancestry, save_chart

chart = plot_prs_multi_ancestry(
    pgs_id="PGS000001",
    distributions_df=distributions,
    user_score=0.274,                # optional
    ancestries=None,                 # None = all 5 populations
    width=560,
    height=220,
)
save_chart(chart, Path("multi_ancestry.html"))  # interactive is best here
```

Overlays 5 color-coded population curves (African, American, East Asian, European, South Asian) with the user's score as a dashed red line. Best viewed as interactive HTML.

### `plot_trait_scores` — Trait-level grouped analysis

The most powerful chart. Shows a standard N(0,1) reference bell curve with per-model dots
colored by quality tier. Each model's raw score is independently z-normalized using its own
reference distribution, so dots from different PGS models are comparable on the same axis.

```python
from just_prs.viz import plot_trait_scores, save_chart

chart = plot_trait_scores(
    trait="Body mass index",        # fuzzy match: "BMI" also works
    distributions_df=distributions,
    quality_df=quality,             # for variant counts
    user_results=result_dicts,      # list of dicts with pgs_id, score, percentile, match_rate
    ancestry="EUR",
    max_scores=25,                  # top N models by variant count
    width=600,                      # configurable
    height=250,                     # bell curve height (try 150 for compact)
    show_table=False,               # True to add model summary table below
    table_height=None,              # auto-sized from row count
)
save_chart(chart, Path("trait_BMI.png"))
save_chart(chart, Path("trait_BMI.html"))  # interactive with hover tooltips
```

**With summary table** (compact bell + detailed model info):
```python
chart = plot_trait_scores(
    trait="BMI",
    distributions_df=distributions,
    quality_df=quality,
    user_results=result_dicts,
    height=150,                     # shorter bell curve
    show_table=True,                # adds Percentile, Variants, Match%, Quality columns
)
save_chart(chart, Path("trait_BMI_detailed.png"))
```

The table shows each model sorted by percentile (descending), with columns:
PGS ID | Percentile | Variant count | Match rate | Quality tier dot.

**Multi-ancestry overlay** (all 5 population curves on one chart):
```python
chart = plot_trait_scores(
    trait="BMI",
    distributions_df=distributions,
    quality_df=quality,
    user_results=result_dicts,
    ancestries=["AFR", "AMR", "EAS", "EUR", "SAS"],  # or a subset
    height=250, show_table=True,
)
save_chart(chart, Path("trait_BMI_all_pops.html"))
```

When `ancestries` contains multiple populations, each gets its own color-coded bell curve
and a separate "Population" legend appears alongside the quality-dot legend.

### `plot_prs_percentile_strip` — Multi-score comparison strip

```python
from just_prs.viz import plot_prs_percentile_strip, save_chart

# Add trait labels to results for the y-axis
for r in result_dicts:
    r["trait"] = "BMI"
    r["label"] = f"BMI: {r['pgs_id']}"

chart = plot_prs_percentile_strip(
    results=result_dicts,           # list with pgs_id, percentile, trait, label
    title="My PRS Percentiles",
    width=560,
    height=None,                    # auto-sized from row count
)
save_chart(chart, Path("percentile_strip.png"))
```

Horizontal strip with colored risk bands (very low / below average / average / above average / high) and a dot per score at its percentile position.

### `save_chart` — Export to any format

```python
from just_prs.viz import save_chart
from pathlib import Path

save_chart(chart, Path("output.png"), scale_factor=2.0)  # high-res PNG
save_chart(chart, Path("output.svg"))                     # vector
save_chart(chart, Path("output.html"))                    # interactive tooltips
save_chart(chart, Path("output.json"))                    # Vega-Lite spec
```

Format is auto-detected from file extension. `scale_factor` only applies to PNG.

## Model quality tiers

| Tier | Criteria | Color |
|------|----------|-------|
| high | >=100K variants OR AUROC >= 0.7 | green |
| moderate | >=10K variants | blue |
| low | >=100 variants | orange |
| very_low | <100 variants | red |

## Ancestry codes

| Code | Population | Color in charts |
|------|-----------|----------------|
| AFR | African | amber |
| AMR | American (admixed) | sky blue |
| EAS | East Asian | teal |
| EUR | European | navy blue |
| SAS | South Asian | pink |

## Interpretation guidelines

When interpreting PRS results for a user:

1. **Match rate matters**: Below 30% match rate, the score is unreliable. Always report it.
2. **Percentile is relative**: 90th percentile means higher genetic predisposition than 90% of the reference population, not 90% chance of disease.
3. **Multiple models converge**: When many models agree on direction (e.g., all above 60th percentile for BMI), the signal is stronger than any single model.
4. **PRS is not diagnosis**: It reflects genetic predisposition in the studied population. Lifestyle, environment, and other factors are not captured.
5. **Ancestry context**: PRS models are often developed in European cohorts. Percentiles from non-matched ancestries should be flagged as less reliable.

## Complete workflow example

Here is a full workflow for computing and visualizing PRS for a trait:

```python
import polars as pl, json
from pathlib import Path
from just_prs import PRSCatalog
from just_prs.scoring import resolve_cache_dir

catalog = PRSCatalog()
cache = resolve_cache_dir()
dists = pl.read_parquet(cache / "percentiles/1000g_distributions.parquet")
quality = pl.read_parquet(cache / "percentiles/1000g_quality.parquet")

# 1. Find scores for the trait
trait = "$ARGUMENTS"
scores = catalog.search(trait, genome_build="GRCh38").collect()
pgs_ids = scores["pgs_id"].head(15).to_list()

# 2. Batch compute PRS
vcf = Path("/path/to/sample.vcf.gz")
results = catalog.compute_prs_batch(vcf_path=vcf, pgs_ids=pgs_ids)

# 3. Enrich with percentiles
result_dicts = []
for r in results:
    p = catalog.percentile(r.score, r.pgs_id, superpopulation="EUR")
    result_dicts.append({
        "pgs_id": r.pgs_id,
        "score": r.score,
        "percentile": p.percentile if p else None,
        "z_score": p.z_score if p else None,
        "match_rate": r.match_rate,
    })

# 4. Generate charts
from just_prs.viz import plot_trait_scores, plot_prs_percentile_strip, save_chart

out = Path("data/output/plots")

# Trait chart with table
chart = plot_trait_scores(trait, dists, quality_df=quality, user_results=result_dicts,
                          height=150, show_table=True)
save_chart(chart, out / f"trait_{trait.replace(' ', '_')}.png")
save_chart(chart, out / f"trait_{trait.replace(' ', '_')}.html")

# Strip chart
for r in result_dicts:
    r["trait"] = trait
    r["label"] = f"{r['pgs_id']}"
chart = plot_prs_percentile_strip(result_dicts, title=f"PRS: {trait}")
save_chart(chart, out / f"strip_{trait.replace(' ', '_')}.png")
```
