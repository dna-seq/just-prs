"""Generate the final manifest, README, AGENTS, and ANALYSIS hint from live schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import polars as pl

from just_prs.sample_scores.evidence.models import EVIDENCE_MODELS, EVIDENCE_PRIMARY_KEYS, schema_field_docs
from just_prs.sample_scores.integration.models import (
    INTEGRATION_MANIFEST_SCHEMA_VERSION,
    MODEL_ANALYSIS_PRIMARY_KEY,
    MODEL_ANALYSIS_SCHEMA_VERSION,
    TRAIT_SUMMARY_PRIMARY_KEY,
    TRAIT_SUMMARY_SCHEMA_VERSION,
    IntegrationManifest,
    ModelAnalysisRow,
    TraitSummaryRow,
)
from just_prs.sample_scores.integration.pins import (
    CATALOG_REPO,
    CATALOG_REVISION,
    FINE_COHORT_DISCLAIMER,
    PERCENTILES_REPO,
    PERCENTILES_REVISION,
    SAMPLE_SCORES_REPO,
    SAMPLE_SCORES_REVISION,
    StagingIndexRow,
)
from just_prs.sample_scores.integration.staging import file_sha256
from just_prs.sample_scores.models import RESTORED_PROFILE_ID, UNRESTORED_PROFILE_ID

_ANALYSIS_GUIDE_TEMPLATE = Path(__file__).with_name("analysis_guide.md")
_EXAMPLE_SAMPLES: tuple[str, ...] = (
    "anton",
    "livia",
    "o-mom",
    "o-dad",
    "o-son1",
    "o-son2",
    "o-daughter",
)


def _schema_table(model: type[object], keys: tuple[str, ...]) -> str:
    lines = [
        f"Primary key: {', '.join(f'`{key}`' for key in keys)}",
        "",
        "| Column | Type | Notes |",
        "|---|---|---|",
    ]
    for name, type_name, description in schema_field_docs(model):
        lines.append(f"| `{name}` | `{type_name}` | {description} |")
    return "\n".join(lines)


def _summary_cell(
    summaries: pl.DataFrame,
    *,
    trait_id: str,
    sample_id: str,
    profile_id: str,
    column: str,
) -> str:
    if summaries.is_empty() or column not in summaries.columns:
        return "—"
    rows = summaries.filter(
        (pl.col("trait_id") == trait_id)
        & (pl.col("sample_id") == sample_id)
        & (pl.col("score_profile_id") == profile_id)
    )
    if rows.is_empty():
        return "—"
    value = rows[column][0]
    if value is None:
        return "N/A"
    if column == "median_pct":
        return f"{float(value):.1f}"
    return str(value)


def _example_row(
    summaries: pl.DataFrame,
    *,
    trait_id: str,
    profile_id: str,
    column: str,
) -> str:
    return " | ".join(
        _summary_cell(
            summaries,
            trait_id=trait_id,
            sample_id=sample_id,
            profile_id=profile_id,
            column=column,
        )
        for sample_id in _EXAMPLE_SAMPLES
    )


def _family_t2d_note(summaries: pl.DataFrame, *, profile_id: str) -> str:
    cells = {
        sample_id: (
            _summary_cell(
                summaries,
                trait_id="MONDO_0005148",
                sample_id=sample_id,
                profile_id=profile_id,
                column="median_pct",
            ),
            _summary_cell(
                summaries,
                trait_id="MONDO_0005148",
                sample_id=sample_id,
                profile_id=profile_id,
                column="risk_vs_average",
            ),
        )
        for sample_id in ("o-mom", "o-dad", "o-son1", "o-son2", "o-daughter")
    }
    if any(pct == "—" for pct, _ in cells.values()):
        return (
            "On unrestored T2D, describe parent and child medians from the query "
            "above. That is a description of these scores, not a pedigree analysis."
        )
    mom_pct, mom_risk = cells["o-mom"]
    dad_pct, dad_risk = cells["o-dad"]
    son1_pct, son1_risk = cells["o-son1"]
    return (
        f"On unrestored T2D, o-mom is ~{mom_pct}th percentile ({mom_risk}) and "
        f"o-dad ~{dad_pct}th ({dad_risk}); o-son1 is higher (~{son1_pct}th, "
        f"{son1_risk}) and o-son2 / o-daughter sit near the mother. That is a "
        "description of these scores, not a pedigree analysis."
    )


def render_analysis(
    *,
    published_at: str,
    n_samples: int,
    n_pgs: int,
    n_runtime_rows: int,
    n_eligible: int,
    n_summaries: int,
    parent_revision: str,
    summaries: pl.DataFrame,
) -> str:
    template = Template(_ANALYSIS_GUIDE_TEMPLATE.read_text(encoding="utf-8"))
    return template.substitute(
        published_at=published_at,
        parent_revision=parent_revision,
        n_samples=str(n_samples),
        n_pgs=str(n_pgs),
        n_runtime_rows=str(n_runtime_rows),
        n_eligible=str(n_eligible),
        n_summaries=str(n_summaries),
        unrestored_profile=UNRESTORED_PROFILE_ID,
        restored_profile=RESTORED_PROFILE_ID,
        t2d_pct_row=_example_row(
            summaries,
            trait_id="MONDO_0005148",
            profile_id=UNRESTORED_PROFILE_ID,
            column="median_pct",
        ),
        t2d_risk_row=_example_row(
            summaries,
            trait_id="MONDO_0005148",
            profile_id=UNRESTORED_PROFILE_ID,
            column="risk_vs_average",
        ),
        iq_pct_row=_example_row(
            summaries,
            trait_id="EFO_0004337",
            profile_id=UNRESTORED_PROFILE_ID,
            column="median_pct",
        ),
        bmi_pct_row=_example_row(
            summaries,
            trait_id="EFO_0004340",
            profile_id=UNRESTORED_PROFILE_ID,
            column="median_pct",
        ),
        family_t2d_note=_family_t2d_note(summaries, profile_id=UNRESTORED_PROFILE_ID),
    )


def render_readme(
    *,
    published_at: str,
    n_samples: int,
    n_pgs: int,
    n_eligible: int,
    n_summaries: int,
    parent_revision: str,
    pgs_without_distribution: list[str],
) -> str:
    analysis_schema = _schema_table(ModelAnalysisRow, MODEL_ANALYSIS_PRIMARY_KEY)
    summary_schema = _schema_table(TraitSummaryRow, TRAIT_SUMMARY_PRIMARY_KEY)
    evidence_chunks = []
    for name, model in EVIDENCE_MODELS.items():
        evidence_chunks.append(f"### `{name}.parquet`\n")
        evidence_chunks.append(_schema_table(model, EVIDENCE_PRIMARY_KEYS[name]))
        evidence_chunks.append("")
    missing = ", ".join(f"`{item}`" for item in pgs_without_distribution[:12]) or "none"
    return f"""# prs-sample-scores

Published individual PRS and catalog evidence for public genomes, used by
[just-prs](https://github.com/dna-seq/just-prs). Dataset:
`just-dna-seq/prs-sample-scores`.

Published at: `{published_at}`
Parent runtime/evidence revision: `{parent_revision}`

Current coverage: **{n_samples} public genomes**, **{n_pgs} PGS IDs**,
**{n_eligible} analysis-eligible runtime rows**, **{n_summaries} trait summaries**.

## How to read this dataset

**What to look at:** [`ANALYSIS.md`](ANALYSIS.md) is the analysis hint —
trait cards, model disagreement, restoration deltas, five-population
percentiles, and family concordance. Start there if you want queries, not
schemas.

Then:

1. `data/manifest.json` — pinned revisions, hashes, grains, and exclusion rules
2. `data/trait_summaries.parquet` — ancestry-selected usable-scope medians
3. `data/actionability.parquet` and `data/trait_contexts.parquet` — evidence, not scores
4. `data/model_analysis.parquet` — drill-down for one sample × PGS × profile

Raw runtime scores remain in `data/runtime_results.parquet`. Do not compare raw
scores across PGS IDs. A percentile is relative to one 1000G superpopulation, not
a disease probability.

## File map

Runtime-owned (do not overwrite from evidence or integration jobs):

- `data/samples.parquet`
- `data/sample_ancestry.parquet`
- `data/runtime_results.parquet`
- `data/runtime_manifest.json`

Evidence-owned:

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

Integration-owned:

- `data/model_analysis.parquet` (schema {MODEL_ANALYSIS_SCHEMA_VERSION})
- `data/trait_summaries.parquet` (schema {TRAIT_SUMMARY_SCHEMA_VERSION})
- `data/manifest.json` (schema {INTEGRATION_MANIFEST_SCHEMA_VERSION})
- `README.md` (this file)
- `AGENTS.md`
- `ANALYSIS.md` (what to look at)

## Scoring profiles

Two published profiles, aggregated independently:

- `{UNRESTORED_PROFILE_ID}` — observed variants plus only safely resolvable hom-ref absences
- `{RESTORED_PROFILE_ID}` — additionally classifies absent WGS loci as homozygous reference
  using a pinned reference-allele universe

Restoration is not genotype imputation and does not infer alternate alleles
from LD. Compare the two profiles only for the same sample and PGS ID after each
side has been summarized on its own.

## Ancestry and percentiles

Selected percentile population is the sample's broad 1000G superpopulation from
`sample_ancestry.parquet`. In this release every validated public genome selects
`EUR` at confidence 1.0; that is derived per sample, not hardcoded.

{FINE_COHORT_DISCLAIMER}
Display CEU as “Northern/Western European” and IBS as “Iberian/Spanish” with the
IGSR cohort definition. Fine-cohort labels never drive percentiles, coherence, or
risk.

`model_analysis.population_metrics` always has exactly five entries
(`AFR`, `AMR`, `EAS`, `EUR`, `SAS`). Unavailable entries are null plus a reason,
never a fabricated zero.

PGS IDs without a trustworthy selected-population distribution stay in
`model_analysis` with `analysis_eligible=false` and never enter summaries.
Examples: {missing}.

## Quality, risk, and heritability

- Match rate is the fraction of scoring variants used. Weight-mass coverage is
  the fraction of absolute model weight. Neither is ancestry.
- Usable models require ≥50% match (`just_prs.trait_summary.is_usable_model`).
- Absolute risk uses `just_prs.absolute_risk.estimate_absolute_risk` from a
  selected-population z-score plus pinned prevalence and OR/AUROC. Missing
  science is `N/A` / null, not zero. A risk ratio of 1 is the population average.
- h² is population-level SNP heritability: the fraction of trait variation
  statistically associated with genetic differences in a studied population. It
  is not an individual's “percent genetic,” causal fraction, or disease
  probability.
- Trait joins use ontology IDs (`just_prs.ontology.normalize_trait_id` plus
  pinned aliases). Never join prevalence or h² by label.
- `not_assessed` means evidence was not established, not “not actionable.”

## Licenses and privacy

The combined dataset license is **CC-BY-4.0**. Named public genomes (Anton,
Livia) have their own source licenses on `samples.parquet`. Anonymized `o-*`
family samples are publication-allowed under the recorded consent basis only.
Private ingest aliases are local-only and must not appear here.
Unknown genomes are never uploaded.

Family rows are not independent. Parent/child concordance is expected from
shared inheritance. This seven-person dataset cannot estimate heritability,
penetrance, segregation, or clinical transmission. Do not rank a family winner.

## Load with Polars

```python
import polars as pl

manifest = pl.read_json("data/manifest.json")  # or json.load
analysis = pl.read_parquet("data/model_analysis.parquet")
summaries = pl.read_parquet("data/trait_summaries.parquet")
eligible = analysis.filter(pl.col("analysis_eligible"))
eur = (
    eligible.drop("source_revision")
    .explode("population_metrics")
    .unnest("population_metrics")
    .filter(pl.col("superpopulation") == "EUR")
)
```

## Load with DuckDB

```sql
SELECT sample_id, trait_id, score_profile_id, median_pct, n_usable
FROM read_parquet('data/trait_summaries.parquet')
WHERE percentile_population = 'EUR'
  AND n_usable > 0
ORDER BY sample_id, trait_id;
```

Always filter `analysis_eligible` before treating a percentile or risk value as
a lookup result. Quarantined PGS IDs remain in `model_analysis` for audit.

## Glossary

- **PRS** — polygenic score for one PGS model; scale is model-specific
- **PGS ID** — PGS Catalog score identifier
- **z-score** — `(score - reference mean) / std` for one superpopulation
- **percentile** — `Phi(z) × 100`; relative standing, not probability
- **match rate** — fraction of scoring variants used
- **weight coverage** — fraction of absolute model weight used
- **restoration** — hom-ref fill of absent WGS loci, not imputation
- **broad ancestry** — AFR/AMR/EAS/EUR/SAS reference similarity
- **nearest 1000G cohort** — CEU/IBS-style fine label inside that panel
- **absolute risk** — lifetime/population probability from z + prevalence + effect
- **risk ratio** — absolute risk / population prevalence; 1 = average
- **h²** — population SNP heritability, not an individual genetic percent
- **ontology mapping** — EFO/MONDO/OBA/HP identifiers for the same or related trait
- **not_assessed** — evidence not established

## Medical and privacy limits

These tables are research/education artifacts for citizen scientists. They are
not a diagnosis, prescription, or clinical recommendation. Pharmacogenomic
context is not prescribing advice. Cite the PGS Catalog page for each PGS ID
(https://www.pgscatalog.org/score/{{pgs_id}}/).

## `model_analysis.parquet`

{analysis_schema}

## `trait_summaries.parquet`

{summary_schema}

## Evidence tables

{"".join(evidence_chunks)}
"""


def render_agents(
    *,
    published_at: str,
    parent_revision: str,
) -> str:
    return f"""# Agent notes for prs-sample-scores

Published at: `{published_at}`
Pinned parent: `{parent_revision}`

## Ownership

- Runtime job owns samples, ancestry, runtime_results, runtime_manifest.
- Evidence job owns the nine evidence parquets plus evidence_manifest.json.
- Integration job owns model_analysis, trait_summaries, final manifest.json,
  README.md, AGENTS.md, and ANALYSIS.md.

Do not let an evidence rerun overwrite final docs. Do not upload
`identity_cache.json`, checkpoint `parts/`, worker reports, or normalized genomes.

## Consumer order

1. Read `ANALYSIS.md` for what to look at, then `data/manifest.json` for revisions and hashes.
2. Use `trait_summaries` for ancestry-selected usable medians.
3. Use actionability/context tables for evidence language.
4. Use `model_analysis` only for drill-down. Filter `analysis_eligible`.

## Semantics agents must not violate

- Raw PRS values are not comparable across PGS IDs.
- Percentile is not disease probability.
- CEU/IBS are nearest 1000G cohorts, never percentile populations.
- Restoration is not imputation.
- Higher percentile is not always worse; preserve trait direction.
- Missing risk/h² is null/`N/A`, never 0.
- h² is not an individual “percent genetic.”
- `not_assessed` is not “not actionable.”
- Family concordance is descriptive only.

## Lookup

`resolve_official_prs` still reads runtime rows. It refuses quarantined PGS IDs
and rows that fail numeric invariants. Integration `analysis_eligible=false`
rows must never be treated as official hits.
"""


def write_final_docs(
    output_dir: Path,
    *,
    analysis: pl.DataFrame,
    summaries: pl.DataFrame,
    sources: list[StagingIndexRow],
    scoring_set_fingerprint: str | None,
    sample_set_fingerprint: str | None,
    reference_universe_fingerprint: str | None,
    n_pgs_ids: int,
    pgs_without_distribution: list[str],
    omitted_catalog_tables: list[str],
    parent_revision: str = SAMPLE_SCORES_REVISION,
    published_at: str | None = None,
    final_revision: str | None = None,
) -> IntegrationManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    published_at = published_at or datetime.now(timezone.utc).isoformat()
    analysis_path = output_dir / "model_analysis.parquet"
    summaries_path = output_dir / "trait_summaries.parquet"
    analysis.write_parquet(analysis_path)
    summaries.write_parquet(summaries_path)
    readme = render_readme(
        published_at=published_at,
        n_samples=int(analysis["sample_id"].n_unique()),
        n_pgs=n_pgs_ids,
        n_eligible=int(analysis.filter(pl.col("analysis_eligible")).height),
        n_summaries=summaries.height,
        parent_revision=parent_revision,
        pgs_without_distribution=pgs_without_distribution,
    )
    agents = render_agents(published_at=published_at, parent_revision=parent_revision)
    analysis_guide = render_analysis(
        published_at=published_at,
        n_samples=int(analysis["sample_id"].n_unique()),
        n_pgs=n_pgs_ids,
        n_runtime_rows=analysis.height,
        n_eligible=int(analysis.filter(pl.col("analysis_eligible")).height),
        n_summaries=summaries.height,
        parent_revision=parent_revision,
        summaries=summaries,
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "AGENTS.md").write_text(agents, encoding="utf-8")
    (output_dir / "ANALYSIS.md").write_text(analysis_guide, encoding="utf-8")
    outputs = {
        "data/model_analysis.parquet": {
            "sha256": file_sha256(analysis_path),
            "bytes": analysis_path.stat().st_size,
            "rows": analysis.height,
            "schema_version": MODEL_ANALYSIS_SCHEMA_VERSION,
            "primary_key": list(MODEL_ANALYSIS_PRIMARY_KEY),
        },
        "data/trait_summaries.parquet": {
            "sha256": file_sha256(summaries_path),
            "bytes": summaries_path.stat().st_size,
            "rows": summaries.height,
            "schema_version": TRAIT_SUMMARY_SCHEMA_VERSION,
            "primary_key": list(TRAIT_SUMMARY_PRIMARY_KEY),
        },
        "README.md": {"sha256": file_sha256(output_dir / "README.md"), "bytes": (output_dir / "README.md").stat().st_size},
        "AGENTS.md": {"sha256": file_sha256(output_dir / "AGENTS.md"), "bytes": (output_dir / "AGENTS.md").stat().st_size},
        "ANALYSIS.md": {
            "sha256": file_sha256(output_dir / "ANALYSIS.md"),
            "bytes": (output_dir / "ANALYSIS.md").stat().st_size,
        },
    }
    manifest = IntegrationManifest(
        published_at=published_at,
        parent_revision=parent_revision,
        final_revision=final_revision,
        sample_scores_repo=SAMPLE_SCORES_REPO,
        catalog_repo=CATALOG_REPO,
        percentiles_repo=PERCENTILES_REPO,
        sample_scores_revision=SAMPLE_SCORES_REVISION,
        catalog_revision=CATALOG_REVISION,
        percentiles_revision=PERCENTILES_REVISION,
        scoring_set_fingerprint=scoring_set_fingerprint,
        sample_set_fingerprint=sample_set_fingerprint,
        reference_universe_fingerprint=reference_universe_fingerprint,
        n_samples=int(analysis["sample_id"].n_unique()),
        n_pgs_ids=n_pgs_ids,
        n_profiles=2,
        n_runtime_rows=analysis.height,
        n_analysis_eligible=int(analysis.filter(pl.col("analysis_eligible")).height),
        n_trait_summaries=summaries.height,
        pgs_without_distribution=pgs_without_distribution,
        omitted_catalog_tables=omitted_catalog_tables,
        sources=sources,
        outputs=outputs,
        notes=[
            FINE_COHORT_DISCLAIMER,
            "Final manifest.json is hashed externally; it does not embed its own SHA256.",
            "score_development_ancestry.parquet was not in the pinned catalog tree and was omitted.",
        ],
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
