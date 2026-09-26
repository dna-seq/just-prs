# Score Annotation & Trait Clustering — Plan

**Status (2026-09-26): tooling in place, Phase 0 not started.** Decisions below come from a design
interview. Two Claude Code skills drive the work — `curate-papers` (queue of publications) and
`curate-trait` (start from a trait such as longevity or diabetes) — sharing one helper
(`prs curation` (`just_prs.curation`)), one lookup agent (`plugins/just-prs/agents/score-lookup.md`) and one rulebook
([curation-rules.md](curation-rules.md)). Phase 0 exists to revise the data model before scale-up.

## Problem

The catalog has 5,337 scores from 594 publications, 696 mapped ontology labels, and 1,732
distinct reported phenotypes. Two gaps make "By Trait" misleading:

1. **Direction is not annotated.** Nothing says what a *higher* score predicts.
   `trait_contexts.high_percentile_means` is hardcoded to `"higher_trait_value"` for every trait
   (`sample_scores/evidence/contexts.py`), and contexts are assigned by regex.
2. **Ontology terms do not group phenotypes by meaning.** One concept is split across terms,
   and one term mixes opposite polarities.

Examples from the cached catalog:

| PGS ID | Reported phenotype | Mapped term | Catalog metric | Higher score means |
|---|---|---|---|---|
| PGS000318 / 319 | All-cause mortality (female / male) | `age at death` | HR 1.10 / 1.15 | shorter life (unfavorable) |
| PGS000906 | Longevity | `life span determination trait` | HR 0.89 | longer life (favorable) |
| PGS001072 | Facial aging, looking older than you are | `aging rate` | — | looks older |
| PGS001141 | Facial ageing, looking younger than you are | `skin aging` | — | looks younger |
| PGS001127 | Never Smoker | `smoking status measurement` | — | *less* likely to smoke |
| PGS001129 | Smoking status (ever vs never) | `smoking status measurement` | — | *more* likely to smoke |

Averaging those rows under one ontology label averages opposite phenotypes. The
reported-phenotype grouping (commit `b1a6d60`) stops the averaging, but it cannot say which
direction is which, and it fragments one concept into many groups.

## Decisions

| Topic | Decision |
|---|---|
| Direction model | **Two axes.** A per-score fact from the paper ("higher score → higher risk of X / lower risk of X / higher value of X"), plus a separate, optional valence (desirable / undesirable / neutral / context-dependent), labeled as a judgment. |
| Clustering | **Curated hierarchy**: domain → concept → phenotype. Each concept declares a canonical axis and each score declares its polarity relative to that axis. |
| Unit of work | **Per publication** (594). The biobank-wide papers get a template pass from their supplementary tables. |
| Runtime | **Claude Code sessions only.** No API loop in Dagster. Two entry points: by paper (`curate-papers`) and by trait (`curate-trait`), same files and checks. |
| Lookups | **Low-effort lookup agents** (Opus, `effort: low`) fetch and quote; the orchestrating session makes every judgment call. |
| Progress | **Tracked and resumable**: a queue file, an append-only progress log, a status command, and a retry cap. |
| Evidence depth | Abstract always; **full text + supplementary tables when open access** (Europe PMC / PMC). |
| Storage | **Repo text files are the source of truth**, compiled into parquets under `just-dna-seq/pgs-catalog` `data/metadata/`, joined by `PRSCatalog.scores()`. |
| Review gate | **`agent_proposed` → `human_verified`.** Each claim carries a verbatim provenance quote, a locator, and a confidence. The UI marks proposed annotations as unreviewed. |
| Aggregation | **Align to the cluster axis.** Opposite-polarity percentiles are flipped (`p → 100 − p`) before the trait median, and the UI says so. Each model's native value stays visible. |
| Empirical checks | All three: **1000G score correlation**, **effect-metric sign**, **lead-variant sign**. |
| Exploration | Nine types × 5–6 scores (~50). "Visual" means **appearance**. |

## Data model v0 (Phase 0 revises this)

Small, reviewable text files under `curation/` at the repo root. They are hand-curated source
(like the guideline records in `evidence/guidelines.py`), never generated data or parquet.

`curation/clusters.yaml` is the hierarchy. The axis and valence live on the concept, so
valence is decided once per concept, not once per score:

```yaml
aging:
  label: Aging
  concepts:
    lifespan:
      label: Lifespan and survival
      axis: longer life              # what "up" means after alignment
      valence: desirable             # desirable | undesirable | neutral | context_dependent
      valence_basis: "Longer survival is the outcome people want."
      phenotypes: [longevity, all_cause_mortality, parental_age_at_death]
```

`curation/publications/<PGP>.yaml` holds the per-score facts:

```yaml
pgp_id: PGP000095
paper: {pmid: "...", doi: "...", title: "..."}
evidence_read: {abstract: true, full_text: pmc, supplement: ["Table S2"]}
template: null          # biobank papers: defaults applied to every score, overridable
scores:
  PGS000318:
    phenotype: all_cause_mortality
    cluster: aging.lifespan
    measurement_kind: time_to_event   # binary | continuous | ordinal | onehot_category | time_to_event | composite | response
    score_effect:
      direction: increases            # increases | decreases — what a higher score does to `target`
      target: hazard of death from any cause
      coding: null                    # e.g. the UK Biobank field coding for ordinal traits
    polarity: -1                      # phenotype sign × direction sign (checked)
    strata: {sex: female}
    provenance:
      - {source: full_text, locator: "Methods, Statistical analysis", quote: "...", confidence: high}
    status: agent_proposed            # agent_proposed | human_verified | rejected
    notes: null
```

Rules:

- A `score_effect` without a verbatim `quote` is invalid. Missing evidence stays `null` and is shown
  as "direction not annotated". It is never guessed.
- Each phenotype in `clusters.yaml` carries `sign: ±1` relative to the concept axis, so `polarity`
  = phenotype sign × direction sign is mechanical. A check enforces it.
- Valence is never inferred from a score. It comes only from the cluster node.

## Phase 0 — manual exploration (~50 scores)

The goal is to learn which fields are actually needed. Each type pairs clean cases with known traps.

| Type | Candidate scores | What it tests |
|---|---|---|
| Diseases | PGS000001, PGS000011, PGS000014, PGS000021, PGS000025, PGS000033 | Baseline "higher = more risk". PGS000033 (T2D, insulin-resistance SNPs) has HR 0.98, so a partitioned score may not follow the disease sign |
| Appearance | PGS001098, PGS001093, PGS001896, PGS001897, PGS001937, PGS001987, PGS002314 | One-hot hair colors vs a single ordinal hair-color score, skin color, balding patterns. PGS001937 (ease of tanning) may follow a UK Biobank coding where a higher value means *less* tanning |
| Aging | PGS000906, PGS002795, PGS000318, PGS001393, PGS005228, PGS001072, PGS001141 | Longevity vs mortality vs parental age at death vs frailty vs visible aging: one domain with several axes and opposite polarities |
| Numeric | PGS000297, PGS000060, PGS000065, PGS000303, PGS003289, PGS000127 | Direction is simply "higher value", but valence differs (HDL, LDL, eGFR). PGS003289 (eGFR from cystatin C) has beta −0.05, so the label and sign may disagree |
| Categorical / ordinal | PGS001127, PGS001129, PGS000336, PGS001055, PGS001087, PGS001001 | Never vs ever smoker. Chronotype and alcohol frequency, where the UK Biobank coding may run morning → evening and daily → never |
| Drug response | PGS000718, PGS000769, PGS002730, PGS001885, PGS000688, PGS000834 | Benefit vs toxicity. PGS002730 (statin LDL lowering) has beta −0.05. PGS000688 ("statin adjusted") and PGS000834 (insulin response) are *not* drug response, which tests the regex trap |
| Biobank mass papers | PGP000244 (Tanigawa, 779 scores): PGS001092, PGS001244, PGS001247, PGS001007, PGS000991, PGS001111 | Whether one template per paper holds up; one-hot vs ordinal encodings of the same field |
| Composite / factor | PGS005221, PGS005222, PGS005226, PGS000205, PGS000848, PGS000852 | Factor and principal-component axes, where direction comes from the factor loading, not the name; pathway-partitioned T2D scores |
| Stratified | PGS000318/319, PGS000829/830, PGS000900/901, PGS000322/323 | Same concept, separate strata. Testosterone in females vs males may carry a different valence |

For each score, the explorer answers the same questions:

1. What exactly was the outcome or measurement, including coding and units?
2. Which direction do the effect alleles and weights point? Where does the paper say so (quote + locator)?
3. What does the catalog's HR/OR/beta refer to, and does its sign agree?
4. Which concept does it belong to, and what is that concept's natural axis?
5. Is there a valence, and is it context-dependent (U-shaped, sex-specific, age-specific)?
6. What did *not* fit the schema? Every miss becomes a candidate field.

Deliverables: `curation/exploration/NOTES.md` (per-score findings and schema misses), the first
`clusters.yaml`, ~50 filled publication entries, and a **schema v1** that replaces v0 above.

Exploration uses the same lookup agent as the loop, so its cost and quality are measured before scale-up.

## Phase 1 — the annotation loop (Claude Code)

### Roles

| Role | Runs as | Effort / model | Does | Never does |
|---|---|---|---|---|
| Orchestrator | The main Claude Code session, via the `curate-papers` or `curate-trait` skill (`plugins/just-prs/skills/`) | Normal effort, strong model | Picks queue items, dispatches lookups, decides direction / cluster / polarity, writes YAML, runs checks, updates the queue | Fetch papers itself when a lookup agent can |
| Lookup agent | `score-lookup` subagent (`plugins/just-prs/agents/score-lookup.md`), several in parallel | **Opus at low effort** (set in the agent frontmatter) | Fetches the catalog rows, abstract, OA full text, supplement tables, and GWAS Catalog associations for top-weight variants. Returns structured excerpts with verbatim quotes and locators | Decide direction, valence, or cluster |
| Second reader | Same lookup agent, or a medium-effort variant | Medium | Re-reads only the `low`-confidence and check-failed items | Overwrite a `human_verified` entry |
| Checks | Deterministic helper (`uv run prs curation check`) | — | Schema, provenance, polarity consistency, the three empirical checks | Call an LLM |

The lookup agent's output contract is a fixed JSON shape (`excerpts[]` with
`source, locator, quote`; `metrics[]`; `variants[]`), so a low-effort model has little room to
improvise and the orchestrator can verify that each quote exists in the fetched text.

Lookup results are cached under `<cache>/annotation_lookups/<pgp_id>/`, never in the repo, so a
resumed or repeated run does not re-fetch papers.

### Per-publication flow

1. Take the next `pending` (or retryable `checks_failed`) row from the queue and mark it `in_progress`.
2. Dispatch a lookup agent for that paper. For biobank papers, dispatch one per supplement table or trait block.
3. Draft `publications/<PGP>.yaml`. Reuse existing cluster nodes; propose a new node only with a
   written reason, and add it to `clusters.yaml` as `agent_proposed`.
4. Run `uv run prs curation check <PGP>`.
5. On a failure, send one targeted second-reader pass. If it still fails, record `checks_failed` with the reason.
6. Update the queue and append a progress line.

Biobank papers (the 8 largest cover ~3,000 scores) get a `template` block: one direction rule for the
whole paper (e.g. "binary traits: higher = case; continuous: higher = higher value"), plus explicit
per-score overrides for one-hot, ordinal, and reverse-coded fields. The checks still run on every score.

### Progress tracking

- **`curation/queue.csv`**: `pgp_id, n_scores, priority, status, attempts, last_error, updated_at`.
  Statuses: `pending → in_progress → drafted → checks_passed | checks_failed → human_verified`,
  plus `blocked` (no usable text) and `rejected`.
- **`curation/progress.txt`** (not `.log`, which `.gitignore` drops): an append-only line per
  publication or trait step, in the same shape as the pipeline progress lines:
  `Annotation: 42/594 PGP (7.1%), scores 612/5337 (11.5%), PGP000095 → 2 scores, checks 2 ok / 0 warn, lookups 3, 4.1 min`.
- **`uv run prs curation status`** (and `trait-status <slug>`): done/total by status, scores covered (weighted by score count,
  because a few papers dominate), check pass rate, and the review backlog (`agent_proposed` awaiting review).
- **Resume and retry**: an `in_progress` row older than the session is treated as interrupted and
  reclaimed. After 3 attempts a row becomes `blocked` with its last error, and is not retried forever.
- **Batches**: a session processes N publications (default ~10) and stops at a clean boundary, so
  review diffs stay small. Driving it with `/loop` is optional.

Priority order after Phase 0:

1. Papers behind trait groups that already mix polarities (reported phenotype contains
   never / ever / low / high / absence / age at).
2. Traits on the demo and trait-ranking lists.
3. Single-score papers (372).
4. Biobank template papers.

## Checks

| Check | Input | Flags |
|---|---|---|
| Schema + provenance | YAML | Missing quote or locator; a quote that is not found verbatim in the cached lookup text |
| Polarity consistency | YAML + `clusters.yaml` | `polarity` ≠ phenotype sign × direction sign |
| Effect-metric sign | `performance.parquet` (HR/OR/beta and the evaluated outcome) | Sign contradicts the annotated direction. Compared against the *evaluation's* outcome, which can differ from the score's trait |
| 1000G correlation | Cached per-individual `reference_scores/1000g/<pgs_id>/scores.parquet` (6,982 cached locally) | After polarity alignment, a clearly negative correlation between two scores in one concept. Computed within one super-population so ancestry structure does not drive it. The threshold is set during Phase 0 |
| Lead-variant sign | Top-weight variants with rsIDs, GWAS Catalog associations | Effect-allele direction disagrees with published associations for the named trait (e.g. APOE ε4 should point against longevity) |

These checks can flag a claim but never write one. A failing check routes the item to the second
reader, then to human review.

## Build, publish, consume

- **Build**: a compile step turns `curation/` into `score_annotations.parquet` (grain `pgs_id`) and
  `trait_clusters.parquet` (grain: cluster node), validated by Pydantic models and a blocking
  Dagster asset check, published to `just-dna-seq/pgs-catalog` `data/metadata/` alongside
  `trait_prevalence.parquet`. It is listed in `hf.CLEANED_PARQUET_FILES`.
- **`PRSCatalog`**: `scores()` left-joins the lean columns (`cluster`, `polarity`,
  `direction`, `annotation_status`); `score_info_row()` merges the full record with provenance.
- **Trait summary** (`just_prs.trait_summary`, the single home for aggregation): a new grouping mode
  by cluster; percentiles are aligned to the cluster axis before the median, and each flipped model
  is labeled (e.g. "mortality score, shown as longevity").
- **UI / reports / prompts**: "higher percentile means …" text on every row, green/red from the
  cluster valence only (neutral traits stay uncolored), and an "unreviewed" marker on `agent_proposed` rows.
- **Retire the regex default**: `trait_contexts.high_percentile_means` is derived from the
  annotations, or left `null` when a score is not annotated, never hardcoded.

## Open questions (settle during Phase 0)

- Correlation threshold, and the minimum number of informative 1000G variants for the correlation check to count.
- How to represent U-shaped traits (BMI, blood pressure), where the axis is fine but the valence is not monotonic.
- Whether one-hot category scores (hair color black / blonde / red) form one concept with several
  "categories", or several binary phenotypes.
- Whether clusters may nest deeper than domain → concept → phenotype.
- How the review UX works: diff review in git only, or also a short review checklist per batch.
