# What to look at

This file is the analysis hint for
[`just-dna-seq/prs-sample-scores`](https://huggingface.co/datasets/just-dna-seq/prs-sample-scores).
Schemas and ownership live in `README.md`. Consumer rules live in `AGENTS.md`.

Published at: `$published_at`
Runtime/evidence parent: `$parent_revision`

As of this snapshot: **$n_samples public genomes**, **$n_pgs PGS IDs**,
**2 scoring profiles**, **$n_runtime_rows** runtime/analysis rows,
**$n_eligible** analysis-eligible, **$n_summaries** trait summaries.
Matrix shape is `$n_samples × $n_pgs × 2`.

Pin a commit SHA when you download. Do not follow floating `main` if you want
these numbers to stay still. This is not a diagnosis, prescription, or clinical
recommendation. A PRS is a genetic predisposition score, not a measurement of
the trait.

## What is in the repo

Three jobs own three layers. Later evidence or runtime reruns must not overwrite
another layer's files.

| Layer | Files | Role |
|---|---|---|
| Runtime | `data/samples.parquet`, `data/sample_ancestry.parquet`, `data/runtime_results.parquet`, `data/runtime_manifest.json` | Who was scored, raw PRS, ancestry |
| Evidence | nine `data/*.parquet` tables plus `data/evidence_manifest.json` | Traits, papers, guidelines, actionability, extra-clinical context. No sample scores |
| Integration | `data/model_analysis.parquet`, `data/trait_summaries.parquet`, `data/manifest.json`, root `README.md`, `AGENTS.md`, `ANALYSIS.md` | Hydrated percentiles/risk/quality plus usable-scope trait medians, plus this hint |

Start here, in this order:

1. `data/manifest.json` — pinned revisions, hashes, grains
2. `data/trait_summaries.parquet` — one median per sample × trait × profile
3. `data/actionability.parquet` and `data/trait_contexts.parquet` — evidence language
4. `data/model_analysis.parquet` — drill-down for one sample × PGS × profile

Raw scores stay in `data/runtime_results.parquet`. Never compare raw PRS values
across PGS IDs. The scale is model-specific.

## Samples

| `sample_id` | Family | Closest 1000G cohort | Notes |
|---|---|---|---|
| `anton` | — | CEU (Northern/Western European) | CC0 / public domain |
| `livia` | — | IBS (Iberian/Spanish) | CC-BY-4.0 |
| `o-mom`, `o-dad`, `o-son1`, `o-son2`, `o-daughter` | `o-family` | CEU | Publication-allowed under recorded consent. Not independent people |

Every validated genome selects **EUR** at confidence 1.0. That label drives
percentiles and risk. CEU/IBS are nearest reference cohorts, not nationality.
Private ingest aliases are local-only and are not in this dataset.

## Two scoring profiles

| `score_profile_id` | Meaning |
|---|---|
| `$unrestored_profile` | Observed variants plus only safely resolvable hom-ref absences |
| `$restored_profile` | Also fills absent WGS loci as homozygous reference from a pinned reference-allele universe |

Restoration is not genotype imputation and does not infer alternate alleles
from LD. Aggregate each profile on its own, then compare `delta_median_pct`.

A large restoration swing usually means the unrestored run had poor coverage
(many unscorable absent sites), not that the person “changed.” Prefer the
unrestored profile unless you have a genome-wide variant-only WGS and want the
fill.

## Eligibility

`model_analysis` keeps **every** runtime row for audit. Only
`analysis_eligible=true` rows carry lookup-like z-scores and percentiles.

A row is ineligible when it is quarantined, fails numeric invariants, has an
inconsistent scoring fingerprint, lacks a trustworthy selected-population
distribution, or is otherwise excluded. Those rows stay in the table with null
metrics plus `exclusion_reasons`. They never enter `trait_summaries`.

Always filter `analysis_eligible` before treating a percentile or risk value as
a result. `not_assessed` on evidence tables means “not established,” not
“not actionable.”

Usable trait models require ≥50% variant match. Quality scopes are a hierarchy:
all ⊇ usable ⊇ high_moderate ⊇ high_quality.

## Load a pinned snapshot

Keep the Hugging Face tree (`data/` prefix, docs at repo root):

```python
from huggingface_hub import snapshot_download

root = snapshot_download(
    "just-dna-seq/prs-sample-scores",
    repo_type="dataset",
    revision="<pin a commit SHA>",
)
```

Then:

```python
import json
from pathlib import Path

import polars as pl

data = Path(root) / "data"

manifest = json.loads((data / "manifest.json").read_text())
summaries = pl.read_parquet(data / "trait_summaries.parquet")
analysis = pl.read_parquet(data / "model_analysis.parquet")
samples = pl.read_parquet(data / "samples.parquet")
ancestry = pl.read_parquet(data / "sample_ancestry.parquet")
traits = pl.read_parquet(data / "traits.parquet")
contexts = pl.read_parquet(data / "trait_contexts.parquet")
actionability = pl.read_parquet(data / "actionability.parquet")

eligible = analysis.filter(pl.col("analysis_eligible"))
```

`population_metrics` is a list of five structs (`AFR`, `AMR`, `EAS`, `EUR`,
`SAS`). Unnesting collides with the row-level `source_revision` column — drop
it first:

```python
eur = (
    eligible.drop("source_revision")
    .explode("population_metrics", empty_as_null=True)
    .unnest("population_metrics")
    .filter(pl.col("superpopulation") == "EUR")
)
```

DuckDB, same files:

```sql
SELECT sample_id, trait_id, trait_label, score_profile_id,
       median_pct, n_usable, absolute_risk, risk_vs_average
FROM read_parquet('data/trait_summaries.parquet')
WHERE percentile_population = 'EUR'
  AND n_usable > 0
  AND score_profile_id = '$unrestored_profile'
ORDER BY sample_id, trait_id;
```

You do not need `just-prs` installed to analyze these tables. The library is
only required to rescore a genome or to call `resolve_official_prs`.

## Analyses worth running

Numbers in the cards below are computed from **this snapshot's**
`trait_summaries.parquet`, unrestored profile, EUR. Re-run the queries after
any republish.

### 1. Trait dashboard (start here)

One row per person for a canonical trait. Use the ontology ID, never a fuzzy
label join.

```python
UNRESTORED = "$unrestored_profile"

def trait_card(trait_id: str) -> pl.DataFrame:
    return (
        summaries.filter(
            (pl.col("trait_id") == trait_id)
            & (pl.col("score_profile_id") == UNRESTORED)
            & (pl.col("n_usable") > 0)
        )
        .select(
            "sample_id", "trait_label", "median_pct", "n_usable",
            "most_reliable_pgs_id", "absolute_risk", "risk_vs_average",
            "heritability_text", "spread", "outliers",
        )
        .sort("median_pct", descending=True)
    )

trait_card("MONDO_0005148")  # type 2 diabetes mellitus
trait_card("EFO_0004337")    # intelligence
trait_card("EFO_0004340")    # body mass index
```

| Trait | Anton | Livia | o-mom | o-dad | o-son1 | o-son2 | o-daughter |
|---|---:|---:|---:|---:|---:|---:|---:|
| Type 2 diabetes (`MONDO_0005148`) median % | $t2d_pct_row |
| T2D risk vs population average | $t2d_risk_row |
| Intelligence (`EFO_0004337`) median % | $iq_pct_row |
| BMI (`EFO_0004340`) median % | $bmi_pct_row |

Direction matters. For disease traits a **higher** percentile is more genetic
risk (worse). For ability traits such as intelligence a **higher** percentile
is more of the trait. Do not rank a “family winner.”

T2D typically has tens of usable models and a mapped EUR h². Intelligence has
few usable models and no mapped risk or h² — that is `N/A`, not zero.

Cite every PGS ID you quote: `https://www.pgscatalog.org/score/{pgs_id}/`.

### 2. Model disagreement on one trait

A median hides spread. `spread` and `outliers` are already on the summary.

```python
t2d = trait_card("MONDO_0005148")
print(t2d.select("sample_id", "median_pct", "spread", "outliers", "n_usable"))
```

To see the underlying models, explode `model_analysis.traits` and keep the
selected EUR percentile:

```python
def trait_models(sample_id: str, trait_id: str) -> pl.DataFrame:
    rows = eligible.filter(
        (pl.col("sample_id") == sample_id)
        & (pl.col("score_profile_id") == UNRESTORED)
    )
    eur_pct = (
        rows.drop("source_revision")
        .explode("population_metrics", empty_as_null=True)
        .unnest("population_metrics")
        .filter(pl.col("superpopulation") == "EUR")
        .select("pgs_id", "percentile", "match_rate", "quality_label", "or_per_sd", "auroc")
    )
    exploded = (
        rows.explode("traits", empty_as_null=True)
        .unnest("traits")
        .filter(pl.col("trait_id") == trait_id)
        .select("pgs_id", "label", "trait_reported")
    )
    return exploded.join(eur_pct, on="pgs_id", how="inner").sort("percentile")

trait_models("anton", "MONDO_0005148")
```

Look at agreement (cluster of percentiles), not only the quality-best model.
`most_reliable_pgs_id` is the most reliable **model**, not the best outcome.

### 3. Absolute risk vs percentile

Risk is the median across in-scope models that have prevalence + effect size,
recomputed from the selected-population z-score. Missing science is null/`N/A`.
A risk ratio of 1 is the population average.

```python
summaries.filter(
    (pl.col("n_risk_models") > 0)
    & (pl.col("score_profile_id") == UNRESTORED)
    & (pl.col("sample_id") == "anton")
).select(
    "trait_id", "trait_label", "median_pct",
    "absolute_risk", "risk_vs_average", "n_risk_models", "n_usable",
).sort("n_usable", descending=True)
```

Do not treat a lone High-quality 99th-percentile model as the trait risk if
dozens of other usable models sit near 50. The summary median is the headline.

h² on the same row is **population** SNP heritability (fraction of trait
variation associated with genetic differences in a studied population). It is
not an individual's “percent genetic.”

### 4. Extra-clinical contexts (not medical advice)

`trait_contexts.parquet` tags traits with `context_class` in
`aging`, `appearance`, `behavioral`, `career`, `checkup_hint`, `curiosity`,
`environment`, `life_optimization`, `lifestyle`, `pharmacology`, `sports`.

```python
lifestyle = (
    contexts.filter(pl.col("context_class").is_in(["lifestyle", "sports", "aging"]))
    .join(traits, on="trait_id", how="left")
    .join(
        summaries.filter(
            (pl.col("sample_id") == "anton")
            & (pl.col("score_profile_id") == UNRESTORED)
            & (pl.col("n_usable") > 0)
        ),
        on="trait_id",
        how="inner",
    )
    .select("context_class", "trait_id", "label", "median_pct", "n_usable")
    .sort(["context_class", "median_pct"], descending=[False, True])
)
```

Pharmacology context is not prescribing advice. `curiosity` is the large
residual class — interesting, not actionable.

### 5. Guideline-supported conditions

PRS actionability is almost entirely `not_assessed` in this snapshot. A small
set of **conditions** are `supported` because a cited public guideline exists —
still not a personal recommendation.

```python
supported = actionability.filter(
    pl.col("condition_actionability_status") == "supported"
).join(traits, on="trait_id", how="left")

cards = supported.join(
    summaries.filter(
        (pl.col("score_profile_id") == UNRESTORED) & (pl.col("n_usable") > 0)
    ),
    on="trait_id",
    how="inner",
).select(
    "sample_id", "trait_id", "label", "median_pct",
    "absolute_risk", "prs_actionability_status",
    "condition_actionability_status",
)
```

### 6. Did WGS restoration move the trait?

Compare profiles only after each side is summarized.

```python
RESTORED = "$restored_profile"

delta = summaries.filter(
    (pl.col("sample_id") == "anton")
    & (pl.col("score_profile_id") == UNRESTORED)
    & (pl.col("n_usable") >= 3)
    & pl.col("delta_median_pct").is_not_null()
).with_columns(
    pl.col("delta_median_pct").abs().alias("abs_delta")
).sort("abs_delta", descending=True).select(
    "trait_id", "trait_label", "median_pct",
    "paired_median_pct", "delta_median_pct", "n_usable", "caveats",
)
```

On Anton, some autoimmune / CLL-adjacent traits jump tens of percentile points
once absent sites are filled. Treat that as a coverage story first: inspect
`match_rate` and `variants_unscorable_absent` on the matching
`model_analysis` / runtime rows before believing either number.

### 7. The same model in all five populations

Percentiles are ancestry-relative. EUR is selected for these seven genomes;
the other four populations are still on every eligible row.

```python
def five_pops(sample_id: str, pgs_id: str) -> pl.DataFrame:
    row = eligible.filter(
        (pl.col("sample_id") == sample_id)
        & (pl.col("pgs_id") == pgs_id)
        & (pl.col("score_profile_id") == UNRESTORED)
    )
    return (
        row.drop("source_revision")
        .explode("population_metrics", empty_as_null=True)
        .unnest("population_metrics")
        .select("pgs_id", "trait_reported", "superpopulation",
                "z_score", "percentile", "available", "exclusion_reason")
    )

five_pops("anton", "PGS005245")
```

A person can sit at the 90th EUR percentile and a very different AFR
percentile for the same raw score. That is expected. Do not pick the
population that makes a nicer story.

### 8. Family concordance (descriptive only)

Family concordance is descriptive only. o-family rows share inheritance.
Parent/child similarity is expected. This seven-person set cannot estimate
heritability, penetrance, segregation, or clinical transmission.

```python
family = (
    samples.filter(pl.col("family_id") == "o-family")
    .select("sample_id", "relationship_role")
    .join(
        summaries.filter(
            (pl.col("trait_id") == "MONDO_0005148")
            & (pl.col("score_profile_id") == UNRESTORED)
        ),
        on="sample_id",
    )
    .select("sample_id", "relationship_role", "median_pct", "absolute_risk")
)
```

$family_t2d_note

### 9. Official lookup for a genome you already have

If a local VCF hashes to a published sample (or you pass a published alias),
`just-prs` `resolve_official_prs` returns the runtime row without rescoring.
It refuses quarantined PGS IDs and invalid rows. This path needs the library
and a local VCF; it is not required to analyze the tables above.

```python
from pathlib import Path
from just_prs.sample_scores import PrecomputedPolicy, resolve_official_prs

result = resolve_official_prs(
    policy=PrecomputedPolicy.REQUIRE,
    pgs_id="PGS000001",
    alias="anton",
    vcf_path=Path("/absolute/existing/anton.vcf"),
    compute=lambda: None,  # unused on a hit under REQUIRE
    reference_restoration=False,
)
print(result.score, result.computation_source)
```

A miss (unknown genome, quarantine, fingerprint mismatch) raises
`PrecomputedMiss` under `REQUIRE`, or falls through to `compute()` under
`AUTO`. Integration `analysis_eligible=false` rows must never be treated as
official hits.

## Rebuild (only if sources change)

Do not rescore genomes to refresh analysis. From the just-prs workspace, stage
the pinned parents and run:

```bash
uv run pipeline sample-score-integration
```

Dagster UI is `http://<host>:3010` (default `0.0.0.0:3010`). Use `--headless`
only for non-interactive runs. `--offline` fails closed if a pin is not already
staged.

## Related docs

- `README.md` — live schemas and file map
- `AGENTS.md` — consumer contract (do not violate these semantics)
- [just-prs reference restoration](https://github.com/dna-seq/just-prs/blob/main/docs/reference-restoration.md)
- [Absolute risk methodology](https://github.com/dna-seq/just-prs/blob/main/docs/absolute-risk-methodology.md)
- [Sample ancestry methodology](https://github.com/dna-seq/just-prs/blob/main/docs/sample-ancestry-methodology.md)
