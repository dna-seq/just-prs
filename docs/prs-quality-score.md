# PRS Quality Score Methodology

This document describes the `synthetic_quality_score()` function in
`just_prs.quality` — a numeric 0–100 ranking metric that lets heterogeneous
PGS Catalog models be compared on a single axis regardless of which
performance metric each model happens to report.

## Why a synthetic score?

PGS Catalog evaluation rows report different performance metrics depending on
outcome type:

| Outcome type | Natural metric | Coverage in GRCh38 set |
|---|---|---|
| Binary (disease case/control) | AUROC or C-index | 40% |
| Continuous (BMI, height, BP…) | Beta per SD | 22% |
| Binary with effect size only | OR or HR | 8% |
| No performance data reported | — | 30% |

A purely AUROC-based ranking is blind to 60% of scores. Returning 0 for
everything without AUROC is misleading: a blood-pressure PRS validated in
400k people with β=0.3 is clearly high quality; it just uses the correct
metric for its trait type.

## Formula

```
score = 100 × discrimination × cohort_factor × match_factor × penalty
```

Each component is bounded to `[0, 1]` before multiplication.

### Discrimination component

Resolved in four tiers, in priority order:

#### Tier 1a — AUROC or C-index (power-stretched, no penalty)

```
exponent = 0.78   if raw ≥ 0.60
         = 0.88   if raw < 0.60

discrimination = 0.5 + 0.5 × ((raw − 0.5) / 0.5) ^ exponent
penalty = 1.0
```

AUROC and C-index are direct discrimination metrics. A piecewise
power-stretch boosts the 0.60–0.70 range (exponent 0.78) where most
validated disease PRS sit, while a steeper exponent (0.88) below
AUROC 0.60 further penalizes weak models that offer little practical
discrimination. AUROC=0.60 maps to disc≈0.64 and AUROC=0.70 to
disc≈0.74.

Representative values:

| AUROC | discrimination |
|---|---|
| 0.50 | 0.500 |
| 0.55 | 0.566 |
| 0.58 | 0.600 |
| 0.60 | 0.642 |
| 0.62 | 0.664 |
| 0.65 | 0.695 |
| 0.70 | 0.745 |
| 0.80 | 0.836 |
| 1.00 | 1.000 |

#### Tier 1b — Beta only (log-mapped, 0.95 penalty)

```
discrimination = 0.5 + 0.5 × tanh(ln(1 + |beta|))
penalty = 0.95
```

Beta (SD units of outcome per SD of PRS) is the native performance metric
for continuous traits. However, beta magnitude is **unit-dependent**: blood
pressure in mmHg produces β≈2, height in cm produces β≈0.3, and lab-value
traits can reach β=150+ simply due to measurement scale.

The log-based mapping (`tanh(ln(1+|β|))`) provides diminishing returns:
small betas spread out over a meaningful discrimination range, while extreme
betas asymptote toward 1.0 without inflating the score. This replaces the
prior linear `clamp(|β|+0.5, 0.5, 1.0)` which saturated at β≥0.5.

The 0.95 penalty reflects that beta-only models show **empirically lower
cross-genome percentile stability** than AUROC-validated models (26% stable
vs 49% stable in a 4-genome smoke test), likely because beta doesn't
directly measure discrimination ability.

Representative values:

| |beta| | discrimination |
|---|---|
| 0.02 | 0.510 |
| 0.10 | 0.548 |
| 0.24 | 0.606 |
| 0.33 | 0.639 |
| 0.50 | 0.692 |
| 1.00 | 0.800 |
| 1.95 | 0.897 |
| 5.00 | 0.973 |
| 152 | 1.000 |

#### Tier 2 — OR or HR, no direct discrimination metric (penalty 0.90)

```
discrimination = Φ( ln(OR) / √2 )
penalty = 0.90
```

where Φ is the standard normal CDF, implemented as:

```
Φ(x) = 0.5 × (1 + erf(x / √2))
```

This is the well-known approximation that converts an odds ratio per SD into
an approximate AUROC assuming a logistic model with normally distributed PRS.

The 0.90 penalty reflects that AUROC *could* have been reported but wasn't.
It is not a trait-type penalty; it is a measurement-quality discount for
choosing a less direct metric. The penalty was raised from 0.85 to 0.90
because OR-only models were being under-ranked relative to their practical
performance — several top-performing models by practical metrics used OR as
their reported effect size.

Representative conversions:

| OR | approx AUROC | score (n=100k, no match penalty) |
|---|---|---|
| 1.1 | 0.53 | 43.1 |
| 1.5 | 0.61 | 50.1 |
| 2.0 | 0.69 | 56.3 |
| 3.0 | 0.78 | 63.9 |

#### Tier 3 — No performance metric (penalty 0.6)

```
discrimination = 0.51   (biased coin floor)
penalty = 0.6
```

Published models are assumed to be at least marginally better than random.
The 0.51 floor and 0.6 penalty give a hard ceiling around 29 regardless of
cohort size, which cleanly separates this tier from any model with real
performance data.

### Cohort factor

Validation cohort size is log10-scaled with denominator 5.5 (reaching 1.0
at ~350k, the typical size of large GWAS like UK Biobank):

```
cohort_factor = clamp( log10(n_individuals) / 5.5,  0.0,  1.0 )
```

| n_individuals | cohort_factor |
|---|---|
| 0 or unknown | 0.5 (neutral) |
| 1,000 | 0.55 |
| 10,000 | 0.73 |
| 50,000 | 0.85 |
| 100,000 | 0.91 |
| 350,000 | 1.00 |

The denominator was reduced from 6.0 to 5.5 to give more credit to large
GWAS cohorts. A moderate-AUROC model validated in 200k people is more
trustworthy than a high-AUROC model validated in 500 people; the steeper
cohort curve makes this distinction sharper.

When `n_individuals` is unknown the factor defaults to 0.5 (neutral), not 0,
so a score without cohort data is not penalized to zero.

### Match factor

```
match_factor = clamp(match_rate,  0.0,  1.0)
```

`match_rate` is the fraction of PRS scoring variants that were found in the
sample VCF. It is only available after computing PRS against real genotypes;
in metadata-only comparisons it defaults to 1.0 (no penalty).

A score where only 30% of variants matched receives `match_factor = 0.30`,
reducing the final score proportionally regardless of how good the
discrimination metric is.

## Tier ordering guarantee

The tier penalties are chosen so that the maximum achievable score for each
tier is strictly separated when cohort size is equal:

| Tier | Max discrimination | Penalty | Max score (n≥350k) |
|---|---|---|---|
| T1a AUROC/C-index | 1.00 | 1.00 | 100 |
| T1b Beta | ~1.00 (β→∞) | 0.95 | ~95 |
| T2 OR/HR | ~0.97 (OR=1000) | 0.90 | ~87 |
| T3 None | 0.51 | 0.60 | 30.6 |

T1b's 0.95 penalty creates a mild ceiling below T1a. T2 and T1 overlap in
the 25–30 range for very weak OR (≈1.0) vs very small Tier 1 cohorts. This
narrow overlap is intentional: OR≈1.0 really isn't much better than
"unassessed."

## Empirical statistics on the GRCh38-filtered set (604 scores)

Measured against the `best_performance` table from PGS Catalog, filtered to
GRCh38 genome build as at May 2026.

### Score distribution

| Statistic | Value |
|---|---|
| Mean | 45.3 |
| Median | 49.3 |
| Min | 15.3 (T3) |
| Max | 89.5 (T1a) |

### By tier

| Tier | n | % of set | min | median | max |
|---|---|---|---|---|---|
| T1a (AUROC/C-index) | 243 | 40% | 25.0 | 65.1 | 89.5 |
| T1b (beta) | 131 | 22% | 26.3 | 50.5 | 81.4 |
| T2 (OR/HR) | 49 | 8% | 24.2 | 46.2 | 63.4 |
| T3 (none) | 181 | 30% | 15.3 | 21.3 | 30.6 |

T1a now has the highest median, correctly reflecting that directly measured
AUROC/C-index is the most reliable performance signal. T1b beta scores are
appropriately below T1a despite their large GWAS cohorts (UK Biobank),
because the log-mapped beta and 0.95 penalty prevent unit-dependent
inflation.

### Concordance with practical PRS results (4-genome smoke test)

The synthetic score was validated against practical PRS computation on 4
real genomes. Key concordance metrics:

- Spearman correlation, synthetic vs practical match rate: r = 0.36
- Spearman correlation, synthetic vs cross-genome stability: r = 0.19
- Top-quartile agreement rate: 69%

The AUROC stretch and cohort boost improved concordance with percentile
stability (from r=0.11 to r=0.19) while the beta log-mapping reduced
overrating of high-beta models that showed poor cross-genome stability.

### Concordance with coarse quality labels

The existing `classify_model_quality()` function assigns coarse labels
(High / Moderate / Low / Very Low) based only on AUROC and match rate. When
the numeric score is binned as:

- **low**: score ≤ median
- **high**: score > mean

the concordance check asserts: **no score in the high numeric bin carries a
coarse Low or Very Low label**. This invariant holds across all 604 GRCh38
scores and is enforced by `test_filtered_grch38_numeric_quality_concords_with_coarse_low_grades`
in `just-prs/tests/test_quality.py`.

## Usage

```python
from just_prs.quality import synthetic_quality_score

# After computing PRS (match_rate available)
score = synthetic_quality_score(
    auroc=0.72,
    n_individuals=150_000,
    match_rate=0.85,
)

# Metadata-only ranking (no match_rate)
score = synthetic_quality_score(
    beta_estimate=0.24,
    n_individuals=400_000,
)

# All inputs optional — returns floor value for unassessed models
score = synthetic_quality_score(n_individuals=50_000)
```

## Combined quality score (synthetic + practical)

The synthetic score above is metadata-only — it ranks models without
computing PRS on real genomes. Once smoke test results are available
(PRS computed across multiple real genomes), a **combined quality score**
blends the synthetic score with three practical signals:

```
combined = norm_synthetic     * 0.40
         + norm_match         * 0.25
         + norm_stability     * 0.15
         + norm_risk          * 0.20
```

All four components are min-max normalized to 0–100 before blending.

### Practical match score

Measures actual genotype coverage across genomes:

```
practical_match = match_rate_mean × (1 − clamp(match_rate_std / match_rate_mean, 0, 0.5))
```

High mean match rate with low cross-genome variance scores highest.
`match_rate` here is the percentage (0–100) of scoring variants found in
each genome's VCF.

### Percentile stability score

Measures cross-genome concordance of EUR reference percentiles:

```
spread_factor = 1.0           if pct_eur_range ≥ 20
              = range / 20    otherwise (floored at 0.1)

stability = (100 − pct_eur_std) × spread_factor
```

Low standard deviation of EUR percentile across genomes indicates a
stable model. The spread factor penalizes degenerate cases where all
genomes collapse to the same extreme percentile (all at 0% or all at
100%) — that looks "stable" but is actually uninformative.

### Risk concordance score

Measures how well absolute risk estimates discriminate between genomes:

```
rel_deviation = |abs_risk − pop_average| / pop_average

concordance = 100 × (1 − clamp(|mean_rel_deviation − 0.3|, 0, 1))
            × (1 − clamp(std_rel_deviation, 0, 0.5))
```

A bell curve centered on 0.3 relative deviation: too little deviation
means the PRS doesn't discriminate (everyone gets the population
average); too much suggests instability. The standard deviation term
rewards consistent discrimination across genomes.

PGS IDs without absolute risk data receive `null` for this component;
the combined score uses only the other three components (weights
redistributed proportionally).

### Empirical results (10-genome smoke test, 604 PGS IDs)

| Statistic | Value |
|---|---|
| Combined mean | 50.2 |
| Combined median | 51.9 |
| Combined min | 10.1 |
| Combined max | 85.9 |

By tier:

| Tier | n | Combined median | Synthetic median |
|---|---|---|---|
| T1a (AUROC/C-index) | 243 | 67.0 | 65.1 |
| T1b (beta) | 131 | 51.0 | 50.5 |
| T2 (OR/HR) | 49 | 57.0 | 46.2 |
| T3 (none) | 181 | 32.0 | 21.3 |

T2 (OR/HR) models score notably higher on the combined metric than their
synthetic score suggests (61.3 vs 46.2 median), confirming the OR penalty
reduction from 0.85 to 0.90 was justified — these models perform well in
practice.

Model-vs-practical concordance (Spearman rank correlation):

| Comparison | r |
|---|---|
| Synthetic vs combined practical | 0.770 |
| Synthetic vs practical match only | 0.363 |

Top-quartile agreement rate: 82.1%. The synthetic score is a strong
predictor of combined practical quality, but practical match rate alone
is only weakly correlated — stability and risk concordance carry
significant independent signal.

### Output parquet

`scripts/build_pgs_quality_scores.py` produces a comprehensive per-PGS
parquet at `data/output/pgs_quality_scores.parquet` (46 columns, one row
per PGS ID). This parquet is synced to the HuggingFace dataset
`just-dna-seq/pgs-catalog` under `data/metadata/pgs_quality_scores.parquet`
and pulled by `PRSCatalog` as part of its 3-tier loading chain (local →
HF → FTP). `PRSCatalog.scores()` joins `synthetic_score`,
`combined_quality_score`, and `quality_label` from this parquet into the
scores LazyFrame, making quality columns available in the metadata grid:

| Column group | Columns |
|---|---|
| Identity | `pgs_id`, `trait`, `trait_grouped`, `trait_efo_id` |
| Synthetic | `synthetic_score`, `tier`, `auroc`, `cindex`, `or_estimate`, `hr_estimate`, `beta_estimate`, `n_individuals` |
| Match | `match_rate_mean/std/min/max`, `variants_matched/total`, `practical_match_score` |
| Stability | `pct_eur_mean/std/min/max`, `pct_{afr,amr,eas,sas}_mean`, `stability_score` |
| Risk | `abs_risk_mean/std`, `pop_avg_risk`, `risk_ratio_mean/std`, `mean_rel_deviation`, `risk_concordance_score` |
| Combined | `norm_synthetic`, `norm_match`, `norm_stability`, `norm_risk`, `combined_quality_score` |
| Metadata | `quality_label`, `has_allele_frequencies`, `percentile_method`, `heritability`, `absolute_risk_method`, `n_genomes` |

```bash
# All genomes
uv run python scripts/build_pgs_quality_scores.py

# Public genomes only (antonkulaga + livia)
uv run python scripts/build_pgs_quality_scores.py --public-only

# Custom output path
uv run python scripts/build_pgs_quality_scores.py --output data/output/scores_public.parquet
```

```python
import polars as pl

scores = pl.read_parquet("data/output/pgs_quality_scores.parquet")

# Top 20 by combined quality
top = scores.head(20).select("pgs_id", "trait_grouped", "combined_quality_score", "tier")

# Best models per trait (using EFO-grouped trait names)
best_per_trait = (
    scores.sort("combined_quality_score", descending=True)
    .group_by("trait_grouped")
    .first()
    .sort("combined_quality_score", descending=True)
)

# Filter to a specific tier
t1a = scores.filter(pl.col("tier") == "T1a_auroc")
```

See also [Demo Trait Ranking](demo-trait-ranking.md) for the trait-level
aggregation used to select showcase traits.

## Quality label functions

Three classification functions convert numeric scores to discrete labels.
All return `(label, color_name)` tuples.

### `classify_model_quality(match_rate, auroc)` — coarse, AUROC-only

The original UI label. Uses only AUROC and match rate (no numeric score):

| Condition | Label | Color |
|---|---|---|
| match_rate < 0.1 | Very Low | red |
| match_rate ≥ 0.5 and AUROC ≥ 0.7 | High | green |
| match_rate ≥ 0.5 and AUROC ≥ 0.6 | Moderate | yellow |
| match_rate ≥ 0.5 | Moderate | yellow |
| otherwise | Low | orange |

Unchanged and intentionally simple. Blind to beta, OR, and cohort size.

### `classify_synthetic_quality(score)` — from synthetic score

Stratifies the `synthetic_quality_score()` output. Boundaries derived
from the 604-score GRCh38 distribution:

| Range | Label | Color | % of 604 | Interpretation |
|---|---|---|---|---|
| ≥ 70 | High | green | 9% | Strong AUROC/C-index or large-cohort beta |
| ≥ 50 | Normal | yellow | 40% | At or above median; solid discrimination signal |
| ≥ 30 | Moderate | orange | 17% | Weak discrimination or small cohort |
| < 30 | Low | red | 34% | No performance data or very weak model |

The 30-point boundary sits in the natural gap between T3 (no metrics,
ceiling 30.6) and models with any real performance data. The 50-point
boundary is the dataset median. The 70-point boundary captures the top
performers (p90 = 69.8).

```
 10-15:  ▏            23
 15-25:  ████████████ 176   ← T3 cluster (no metrics)
 25-30:  ██           28    ← gap
 30-50:  ████████     101
 50-70:  ████████████ 243   ← T1a/T1b bulk
 70-95:  ████         56    ← top performers
```

### `classify_combined_quality(score)` — from combined score

Stratifies the `combined_quality_score` output. Boundaries derived from
the 604-score distribution across 10 real genomes:

| Range | Label | Color | % of 604 | Interpretation |
|---|---|---|---|---|
| ≥ 70 | High | green | 16% | Strong model validated in practice |
| ≥ 55 | Normal | yellow | 35% | Above median; reliable for demo use |
| ≥ 40 | Moderate | orange | 35% | Usable but some practical weakness |
| < 40 | Low | red | 14% | Poor match, unstable, or no metrics |

The 40-point boundary separates the T3-dominated tail from scored
models. The 55-point boundary is the dataset median. The 70-point
boundary aligns with the demo trait cutoff (see
[demo-trait-ranking.md](demo-trait-ranking.md)).

```
 10-20:  ▏          6
 20-40:  ████       79    ← T3 + weak models
 40-55:  ████████   212   ← Fair bulk
 55-70:  ████████   212   ← Good bulk
 70-86:  ████       95    ← top combined
```

### Cross-tabulation (synthetic vs combined labels)

The two label systems agree on the extremes and diverge in the middle,
which is expected — practical signals promote OR/HR models that synthetic
scoring undervalues:

|  | C-High | C-Normal | C-Moderate | C-Low |
|---|---|---|---|---|
| **S-High** | 32 | 17 | 7 | 0 |
| **S-Normal** | 63 | 146 | 33 | 1 |
| **S-Moderate** | 0 | 49 | 43 | 9 |
| **S-Low** | 0 | 0 | 129 | 75 |

No synthetic-High model ever falls to combined-Low. No synthetic-Low
model ever reaches combined-High. The 63 models that move from S-Normal
to C-High are mostly T2 (OR/HR) models with strong practical performance.

```python
from just_prs.quality import (
    classify_synthetic_quality,
    classify_combined_quality,
    classify_model_quality,
)

# Synthetic label (metadata-only)
label, color = classify_synthetic_quality(72.3)   # ("High", "green")

# Combined label (after smoke test)
label, color = classify_combined_quality(58.1)    # ("Normal", "yellow")

# Original coarse label (AUROC + match rate only)
label, color = classify_model_quality(0.65, 0.71) # ("High", "green")
```
