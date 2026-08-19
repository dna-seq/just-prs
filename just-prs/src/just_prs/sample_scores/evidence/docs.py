"""Generate HF-root README.md and AGENTS.md from live evidence schemas."""

from __future__ import annotations

from just_prs.sample_scores.evidence.models import (
    EVIDENCE_MODELS,
    EVIDENCE_PRIMARY_KEYS,
    schema_field_docs,
)

_RUNTIME_PENDING = (
    "Sample runtime scores (`samples.parquet`, `runtime_results.parquet`) are a "
    "later upload. Empty or absent runtime files are expected. Do not invent "
    "PRS scores, percentiles, or coverage counters."
)


def _schema_markdown() -> str:
    chunks: list[str] = []
    for name, model in EVIDENCE_MODELS.items():
        keys = ", ".join(f"`{key}`" for key in EVIDENCE_PRIMARY_KEYS[name])
        chunks.append(f"### `{name}.parquet`\n")
        chunks.append(f"Primary key: {keys}\n")
        chunks.append("| Column | Type | Notes |")
        chunks.append("|---|---|---|")
        for field_name, type_name, description in schema_field_docs(model):
            chunks.append(f"| `{field_name}` | `{type_name}` | {description} |")
        chunks.append("")
    return "\n".join(chunks)


def render_readme(*, published_at: str, n_traits: int, n_papers: int, n_guidelines: int) -> str:
    schemas = _schema_markdown()
    return f"""# prs-sample-scores

Public evidence tables for [just-prs](https://github.com/dna-seq/just-prs) and the
Hugging Face dataset `just-dna-seq/prs-sample-scores`.

{_RUNTIME_PENDING}

Published at: `{published_at}`

Current evidence coverage: **{n_traits} traits**, **{n_papers} papers**, **{n_guidelines} guidelines**.

## What this dataset is

Catalog-level evidence that joins a PGS Catalog score (`pgs_id`) or a canonical
trait (`trait_id`). Every evidence row is about a published score or a trait —
never a private genome.

The combined dataset license is **CC-BY-4.0**. Per-sample source licenses will
appear on `samples.parquet` when runtime scores are uploaded.

## File map

Evidence tables live under `data/`:

- `data/traits.parquet`
- `data/score_trait_links.parquet`
- `data/papers.parquet`
- `data/score_paper_links.parquet`
- `data/guidelines.parquet`
- `data/guideline_trait_links.parquet`
- `data/actionability.parquet`
- `data/trait_contexts.parquet`
- `data/record_search_terms.parquet`
- `data/evidence_manifest.json`

Pending later upload (do not invent):

- `data/samples.parquet`
- `data/runtime_results.parquet`
- `data/trait_summaries.parquet`
- `data/model_analysis.parquet`

This repository root also ships `README.md` (this file) and `AGENTS.md`.

## Join contract

```text
score_trait_links.pgs_id      → catalog PGS ID (later: runtime_results.pgs_id)
score_trait_links.trait_id    → traits.trait_id
score_paper_links.pgs_id      → catalog PGS ID
score_paper_links.paper_id    → papers.paper_id
actionability.trait_id        → traits.trait_id
guideline_trait_links         → guidelines.guideline_id + traits.trait_id
trait_contexts.trait_id       → traits.trait_id
record_search_terms.trait_id  → traits.trait_id
```

Family concordance is a **query** over future `trait_summaries` + `samples.family_id`,
not a table in this publish.

## Load with Polars

```python
import polars as pl

bonds = pl.read_parquet("data/score_trait_links.parquet")
traits = pl.read_parquet("data/traits.parquet")
action = pl.read_parquet("data/actionability.parquet")
contexts = pl.read_parquet("data/trait_contexts.parquet")

panel = (
    bonds.join(traits, on="trait_id", how="left")
    .join(action, on="trait_id", how="left")
    .join(contexts, on="trait_id", how="left")
)
```

## Actionability

Three columns, never a boolean:

1. `prs_actionability_status` — does a source explicitly support using a PRS to change management?
2. `condition_actionability_status` — does the condition have a guideline-supported pathway?
3. `context_resolution_status` — would additional clinical context decide whether a pathway applies?

Allowed values: `supported`, `against`, `insufficient_evidence`, `not_assessed`, `not_applicable`.
**`not_assessed` means unknown, not negative.** Every non-`not_assessed` row cites a guideline.

ClinGen rows are condition/genetic context. They are never automatic PRS actionability.

## Extra-clinical contexts

`trait_contexts.context_class` is a second track. It must not be mixed into the
three clinical statuses. Headline class: **`aging`** (longevity / healthspan).
`pharmacology` is present only when a published drug-response PGS exists.

## Abstracts

`papers.abstract_text` is filled only when Europe PMC reports a redistribution
license compatible with this dataset's CC-BY-4.0 (public domain, CC0, or CC-BY
without NC/SA/ND). Otherwise `abstract_status` is `link_only`.

## Limitations

- These tables are research information, not a diagnosis or a care plan.
- Population / ancestry caveats of the underlying PGS models still apply.
- Guideline adapters record publicly obtainable pages and structured fields.
  They skip login walls and paywalls. They do not fabricate recommendations.
- Consumers decide whether they read the files themselves or with an LLM, and
  they follow any local rules that apply to them.

## Schemas

Generated from the live Pydantic models in `just_prs.sample_scores.evidence.models`.

{schemas}
"""


def render_agents() -> str:
    return f"""# AGENTS.md — prs-sample-scores analysis contract

This file is the machine/LLM contract for a downloaded copy of
`just-dna-seq/prs-sample-scores`. It is self-contained.

{_RUNTIME_PENDING}

## Required reading order

1. Read `data/evidence_manifest.json` (and later `data/manifest.json` when plan 3 publishes it).
2. If `runtime_results.parquet` / `trait_summaries.parquet` are absent, stay on
   evidence tables. Do not invent sample scores.
3. When runtime files exist: read `trait_summaries.parquet`, then join
   `actionability.parquet`. Use `model_analysis.parquet` for drill-down.
4. Join `traits.parquet` via `score_trait_links.parquet`. Join papers via
   `score_paper_links.parquet`.

## Rules

1. Filter to successful, reliable, non-quarantined rows when runtime scores exist.
   Never compare raw scores across different PGS models.
2. Do not count multiple correlated models as independent evidence. Use canonical
   trait summaries when they exist.
3. Compare both scoring profiles when both are present. Downgrade confidence when
   they materially disagree.
4. Distinguish model validity, condition actionability, and PRS-specific actionability.
5. Never infer an action from a percentile alone. Check guideline direction, grade,
   jurisdiction, eligibility, freshness, and `prs_specific`.
6. Treat `not_assessed` as unknown, not negative. Missing evidence is never
   “not actionable.”
7. When suggesting records or extra data, use only `record_search_terms` and state
   that record review may contextualize risk rather than confirm disease.
8. Extra-clinical findings (`trait_contexts`) are a separate track. They are useful
   (longevity, career, appearance, sports, lifestyle, personal stacks). Never promote
   them as screening, treatment, or PRS-guided management. A checkup hint is
   “this existing measurement would contextualize the score,” not “get this test
   because of the PRS.”
9. Longevity / `aging` is a headline extra-clinical result: report relative standing,
   related healthspan traits, and family concordance when samples exist. It is not
   a predicted age at death. `life_optimization` may stack already-tracked levers.
   `career` and `appearance` are allowed uses. `pharmacology` is useful in the
   context of a drug already taken, not a prescription. `behavioral` may be used
   for self-reflection, not as a diagnosis. `environment` suggests an exposure
   history, not a clinic visit.
10. Family comparison uses `samples.family_id` + `trait_summaries` only, and only
    after those files exist. Prefer longevity when present. Report
    concordance/discordance and published h². Do not invent household heritability,
    and do not treat family agreement as a clinical pathway.
11. Cite PGS ID, ontology ID, PMID/DOI, and guideline URL/version for every
    important claim.
12. State uncertainty, ancestry/coverage limitations, snapshot dates, and that
    results are research information rather than diagnosis.
13. Do not convert `canary_scores.parquet` into runtime hits. Do not write
    extra-clinical classes into the three actionability statuses.

## Worked queries (evidence-only)

```python
import polars as pl

traits = pl.read_parquet("data/traits.parquet")
contexts = pl.read_parquet("data/trait_contexts.parquet")
action = pl.read_parquet("data/actionability.parquet")
terms = pl.read_parquet("data/record_search_terms.parquet")

aging = contexts.filter(pl.col("context_class") == "aging")
clinical_pathways = action.filter(
    pl.col("condition_actionability_status") == "supported"
)
extra = terms.filter(pl.col("track") == "extra_clinical")
```

## Family concordance recipe (after runtime scores exist)

```python
samples = pl.read_parquet("data/samples.parquet")
summaries = pl.read_parquet("data/trait_summaries.parquet")
family = samples.filter(pl.col("family_id") == "o-family")
joined = summaries.join(family.select("sample_id", "relationship_role"), on="sample_id")
# Compare percentiles within family_id. Use published h² from the catalog.
# Do not compute household heritability from these few relatives.
```

Until `trait_summaries.parquet` and `samples.parquet` are published, stop after
the evidence-only queries.
"""
