# Demo Trait Ranking

Traits recommended for showcasing PRS results, ranked by a weighted blend
of model quality, practical genotype match, cross-genome percentile stability,
absolute risk concordance, and PGS count per trait.

## Selection criteria

A trait makes the demo list only if it scores **70+** in **both** ranking
modes:

- **All genomes** (10 genomes as of May 2026) — tests stability across a
  diverse set of real-world WGS samples.
- **Public-only** (antonkulaga + livia) — the two permissively licensed
  genomes used in public demos where private data cannot be shown.

Traits are grouped by EFO/MONDO ontology ID so that free-text variants
(e.g. "Type 2 diabetes (T2D)" vs "(PheCode 250.2)") are merged into one
entry. Traits with fewer than 3 PGS IDs are excluded.

## Methodology

Each trait is scored as:

```
trait_score = mean_model_quality     * 0.30
            + mean_practical_match   * 0.15
            + mean_stability         * 0.10
            + mean_risk_concordance  * 0.15
            + quality_count_weight   * 0.30
```

All components are normalized to 0-100 before blending. `quality_count`
weights each PGS model by its quality tier — best (≥70): 4, normal
(≥55): 1, moderate (≥40): 0.4, low (<40): 0.1 — then log₂-normalizes
the sum (capped at 48). This rewards traits with many strong models
rather than just many models. When risk data is unavailable, the
remaining weights are redistributed proportionally.

Traits are sorted by `min(all-genomes, public-only)` so the weaker
mode determines rank.

See [PRS Quality Score Methodology](prs-quality-score.md) for the underlying
`synthetic_quality_score()` formula.

## Final demo trait list

Sorted by min(all-genomes, public-only) so the weaker mode dominates.
All traits have distinct EFO/MONDO ontology IDs (verified — no false
merges or missed splits).

| Rank | Trait                            | Ontology ID   | All genomes | Public-only | Min   | n_pgs |
|------|----------------------------------|---------------|-------------|-------------|-------|-------|
| 1    | Inflammatory bowel disease (IBD) | MONDO:0005265 | 84.3        | 87.6        | 84.3  | 13    |
| 2    | Type 1 diabetes (T1D)            | MONDO:0005147 | 83.9        | 84.3        | 83.9  | 15    |
| 3    | Prostate cancer                  | MONDO:0005159 | 78.7        | 84.4        | 78.7  | 18    |
| 4    | Myocardial infarction            | MONDO:0005068 | 74.3        | 74.3        | 74.3  | 7     |
| 5    | Chronic kidney disease (CKD)     | MONDO:0005300 | 76.3        | 73.7        | 73.7  | 14    |

## Notes on cardiac trait overlap

Several cardiac conditions appear independently because they have distinct
ontology IDs:

| Trait                          | Ontology ID   |
|--------------------------------|---------------|
| Congestive heart failure (CHF) | MONDO:0005009 |
| Heart failure                  | MONDO:0005252 |
| Myocardial infarction          | MONDO:0005068 |
| Angina pectoris                | EFO:0003913   |
| Coronary artery disease        | MONDO:0005010 |

These are clinically related but genetically distinct conditions with
separate PGS models and validation cohorts. Merging them would lose
meaningful signal. For a demo UI that groups by organ system, these could
be presented under a "Cardiovascular" category while preserving distinct
PRS results per condition.

## Catalog filter

The MUI DataGrid filter model for pre-selecting these 5 trait groups
is saved at [`docs/demo-trait-filter.json`](demo-trait-filter.json).
It has two parts:

- **`filter_model`**: flat OR on `trait_efo_id` (compatible with MUI
  DataGrid Community, which only supports one `logicOperator` level).
- **`quality_floor`**: `combined_quality_score >= 40` — applied
  programmatically before loading into the grid to drop PGS models
  that scored poorly across all practical signals.

The `combined_quality_score` and `quality_label` columns are available in
`PRSCatalog.scores()` via `pgs_quality_scores.parquet` synced from
HuggingFace (`just-dna-seq/pgs-catalog` under `data/metadata/`).

To apply it programmatically:

```python
import json, polars as pl
from pathlib import Path
from reflex_mui_datagrid.lazyframe_grid import apply_filter_model
from just_prs import PRSCatalog

config = json.loads(Path("docs/demo-trait-filter.json").read_text())
catalog = PRSCatalog()
lf = catalog.scores()

# Apply quality floor first
qf = config["quality_floor"]
lf = lf.filter(pl.col(qf["field"]) >= float(qf["value"]))

# Then apply trait filter via DataGrid filter model
demo_scores = apply_filter_model(lf, config["filter_model"]).collect()
```

## Traits that didn't make the cut

Alzheimer's disease (min 72.9), Gout (71.8), CHF (70.6), Angina (70.4),
Atrial fibrillation (70.2), and Breast cancer (70.6) passed the 70+
threshold in both modes but fell outside the top 5. Thyroid cancer and
Benign nodular goiter scored well on public genomes but lacked stability
across the full 10-genome set.

## Reproducing

```bash
# Both rankings
uv run python scripts/rate_prs_quality.py --top 50 --output data/output/ratings_all.parquet
uv run python scripts/rate_prs_quality.py --public-only --top 50 --output data/output/ratings_public.parquet

# Trait parquets are saved alongside as trait_demo_ranking_all.parquet
# and trait_demo_ranking_public.parquet
```
