# Curation Rules — score direction and trait clusters

Shared rules for the `curate-papers` and `curate-trait` skills (`plugins/just-prs/skills/`; the lookup agent is
`plugins/just-prs/agents/score-lookup.md`), and the
checklist a human reviewer uses when flipping an entry to `human_verified`. The background and
design decisions are in [score-annotation-plan.md](score-annotation-plan.md).

## Files

| Path | What | Written by |
|---|---|---|
| `curation/queue.csv` | One row per publication: phase, priority, status, attempts, counts | `prs curation` (`just_prs.curation`) only |
| `curation/progress.txt` | Append-only progress lines (papers and traits) | `prs curation` (`just_prs.curation`) only |
| `curation/clusters.yaml` | Domain → concept → phenotype hierarchy with axis, valence, phenotype signs | Agent (as `agent_proposed`), reviewer |
| `curation/publications/<PGP>.yaml` | Per-score facts with verbatim provenance | Agent, reviewer |
| `curation/traits/<slug>.yaml` | One trait question: search terms, per-score in/out-of-scope decisions, concepts | `trait-scan` + agent |
| `curation/exploration/set.yaml` | Phase 0 score IDs by type | Fixed |
| `curation/exploration/NOTES.md` | Phase 0 findings and schema misses | Agent |
| `<cache>/annotation_lookups/<PGP>/` | Raw downloaded sources + `lookup*.json` (agent output, never evidence) | `score-lookup` agent |
| `curation/staging/<slug>/` | One trait agent's private copy during a `curate-traits` batch (gitignored) | `stage` / `trait-curator` |
| `data/output/curation/trait_priorities.csv` | Review priority of every ontology trait group (regenerable, not committed) | `prioritize` |
| `<cache>/annotation_lookups/_traits/<slug>/` | `triage.csv`, `corr.parquet` | `trait-scan` |

Never hand-edit `queue.csv` or `progress.txt`. Only a human sets `status: human_verified`. A
`human_verified` entry is frozen: if new evidence contradicts it, report that instead of editing it.

## Helper commands

`uv run prs curation <command>` (from the repo root):

| Command | Does |
|---|---|
| `init` | Build or refresh the paper queue from the catalog. Idempotent; keeps statuses |
| `next --n N [--phase explore\|main]` | Next papers: pending, retryable `checks_failed`, or interrupted `in_progress` |
| `claim <PGP>` | Mark `in_progress`, count the attempt |
| `context <PGP> [--pgs-ids …] [--top-variants N]` | JSON: paper, scores, evaluations, lead variants, cluster index, lookup dir, existing YAML |
| `check <PGP>` | Deterministic checks; exit 1 on any ERROR |
| `finish <PGP> [--lookups K] [--blocked REASON]` | Checks + coverage, set status, append progress line |
| `log <PGP> --message …` | Progress line without a status change (biobank blocks) |
| `status` | Paper progress, review backlog, exploration coverage |
| `trait-scan <slug> [--query …] --term … [--efo …] [--exclude …] [--anchor PGS]` | Find a trait's scores, merge the worklist, print triage |
| `trait-status <slug>` | Sorted view: concept → phenotype → polarity, conflicts, gaps |
| `trait-log <slug> --message …` | Progress line for a trait |
| `prioritize [--top N]` | Rank all ontology trait groups by review need (score count, reported-phenotype mix, flip wording, 1000G negative pairs, popularity) → `data/output/curation/trait_priorities.csv`; no LLM |
| `stage <slug>` | Private copy `curation/staging/<slug>/` (+ `.base/` snapshot) for one parallel trait agent |
| `merge <slug> [--dry-run] [--keep]` | Three-way merge of a staging copy back into `curation/`; conflicts (changed on both sides, or main `human_verified`) are reported and skipped; validates with a build |
| `build [--out DIR] [--verified-only]` | Compile `curation/` into `score_annotations.parquet` (resolved rows, published), `curation_review.parquet` (the review queue), `trait_clusters.parquet`, `concept_search_terms.parquet` and `curation_manifest.json` (default `data/output/curation/`); fails on any inconsistent file |
| `push [--verified-only] [--replace] [--repo …]` | Rebuild, **merge with the published release** (published rows missing here are kept; published `human_verified` rows are never replaced by unreviewed ones; conflicting cluster definitions abort), then upload all curation files to `just-dna-seq/pgs-catalog` `data/metadata/` in one commit. Needs `HF_TOKEN`; refuses empty tables; `--replace` overwrites instead |

## Memory safety

Curation never scores a genome or a reference panel: direction comes from quotes, metric signs and cached
1000G correlations. The only scoring-file read is `curation context --top-variants`, which

- returns at once on a cached parquet and reads only the projected columns with a streaming `top_k`;
- on a cache miss takes the shared lock `<cache>/annotation_lookups/.scoring_fetch.lock`, so parallel
  agents fetch one score at a time (an eager `.txt.gz` parse of a multi-million-variant score costs
  several GB);
- inside the lock pulls the parquet mirror from `just-dna-seq/pgs-catalog` first and falls back to EBI
  plus the local parse only for IDs the mirror lacks.

Every score still gets its lead-variant check; large ones just wait their turn.

## What the checks do

| Check | ERROR / WARN when | Never |
|---|---|---|
| schema | Unknown key, bad enum, annotated score without cluster/phenotype/polarity/provenance, null `score_effect` without a reason | — |
| membership | A score is not in that publication | — |
| cluster | Unknown concept or phenotype | — |
| polarity | Declared polarity ≠ phenotype sign × direction sign | — |
| quote | Quote not found verbatim in the raw sources (or in catalog fields for `source: catalog`) | Reads `lookup.json` (that is agent output) |
| metric_sign (WARN) | HR/OR/beta sign disagrees with the expected sign for the evaluated outcome | Treats AUROC / C-index as directional |
| corr_1000g (WARN) | Two scores in one concept correlate below −0.2 in 1000G EUR after polarity alignment | Counts a missing file as a pass |
| lead_variant (WARN) | Recorded `disagree` / `mixed` | Counts `not_run` as a pass |
| coverage (`finish` only) | A target score has no entry | — |

`NOT_RUN` means the check could not run, not that it passed.

## Decision rules

- **Direction is a fact about the score.** `score_effect.direction` is `increases` or `decreases`
  of `target`, taken from a verbatim quote. The catalog metric sign is supporting evidence only
  once you know which outcome it was measured against.
- **Binary traits:** "higher = more likely case" is the default, but check which group is the case.
  "Never smoker" makes never-smokers the cases.
- **Ordinal / questionnaire fields:** record `coding` (field ID and value order, e.g.
  `UKB 1727: 1 very tanned … 4 never tan`) with a quote. Reverse-coded fields are the most common trap.
- **One-hot categories** (hair color black / blonde / red): one phenotype per category inside one concept.
- **Composite / factor / PC scores:** direction comes from the loadings. Quote the passage that says
  what a high value means; the factor's name is not enough.
- **Pathway-partitioned scores** (e.g. T2D via insulin resistance): phenotype is the disease; note the partition.
- **Stratified scores:** same phenotype, with `strata: {sex: female}` etc.
- **Drug response:** the phenotype is the response (benefit, toxicity, efficacy), not the
  drug-adjusted biomarker. "LDL (statin adjusted)" is a lipid measurement.
- **Polarity** = phenotype `sign` × (`+1` increases / `−1` decreases), relative to the concept axis.
- **Metric outcome:** when the catalog metric targets another phenotype of the same concept
  (a longevity score with a hazard ratio for death), set `evaluation_outcome: {phenotype: …}`.
  For an outcome in another concept, set `{expected_metric_sign: ±1, basis: "…"}`.
- **Valence lives only on the concept** (`desirable` / `undesirable` / `neutral` /
  `context_dependent`) with a one-line `valence_basis`. Use `context_dependent` for U-shaped,
  sex-specific or age-specific cases. Appearance traits are usually `neutral`.
- **Concept boundaries:** one concept per axis a person would read as "the same question". Lifespan
  and frailty are different concepts in the same domain. Reuse concepts before creating new ones;
  a new concept or phenotype needs a one-line reason.
- **Never guess.** No quote, no direction: `score_effect: null` plus `unannotated_reason`.
- **Resolved vs review.** A recorded direction is published only when nothing about it is
  uncertain. Set `review_reason` (one line) when the direction rests on an inferred coding, a
  correlation instead of a statement, a suspected inverted deposit, or a metric that contradicts
  it. Low-confidence-only evidence and `disagree`/`mixed` lead variants go to review
  automatically. Review rows land in `curation_review.parquet`, with source links, for a human
  or reviewer agent to settle; a human settles one by fixing the entry or setting `human_verified`.
- **Inverted deposits and broken files: record, never edit.** When the paper states a
  direction but the deposited file's lead loci all point the other way (and it anti-correlates
  with the consensus), the file behaves as the negated score. Record `score_effect` and
  `polarity` **as distributed** (so users' percentiles read correctly) and add a `corrections`
  entry: `kind: deposit_inverted`, `action: polarity_as_distributed`, `paper_direction` (what the
  paper says higher means), plus a second `metric_not_from_file` / `discount_metrics` entry
  because the paper's OR/HR/AUROC describe the authors' score, not the file. Only a whole-file
  swap is a pure sign flip; if only some loci are opposite, use `partial_allele_inconsistency`
  (no sign repairs it). Degenerate weights are `broken_weights` / `quarantine` (proposed).
  Require a stated paper direction **and** ≥2 opposite canonical loci **and** strong
  anti-correlation before calling a deposit inverted; correlation alone is not enough. Scoring
  weights are never modified. An entry with corrections stays in review until `human_verified`;
  fill `upstream_report` once the problem is reported to PGS Catalog.
- **Search aliases.** Give each new concept 2–5 `aliases` people would type (plain words,
  opposite phrasings: "bad cholesterol", "night owl", "evening person"). Phenotypes take
  `aliases` too. Catalog names of member scores are added automatically at build.

## Standard lookup questions

Ask these for every target score, plus any score-specific ones:

1. What exactly is the outcome or measurement (definition, case/control coding, units, questionnaire field and its value order)?
2. In which direction do the weights point: does a higher score mean more or less of that outcome? Quote where the paper says so.
3. Which outcome and comparison do the reported HR/OR/beta refer to?
4. Is the score stratified (sex, age, ancestry) or derived from a factor, principal component, or pathway partition?

## YAML formats

A concept in `curation/clusters.yaml`:

```yaml
version: 0
domains:
  aging:
    label: Aging
    concepts:
      lifespan:
        label: Lifespan and survival
        axis: longer life
        valence: desirable
        valence_basis: Longer survival is the outcome people want.
        status: agent_proposed
        aliases: [longevity, life expectancy, how long will I live]
        phenotypes:
          longevity: {label: Survival to old age, sign: 1}
          all_cause_mortality: {label: Death from any cause, sign: -1}
```

`curation/publications/<PGP>.yaml` (unknown keys are rejected on purpose):

```yaml
pgp_id: PGP000000
paper: {pmid: "00000000", doi: 10.0000/example, title: "…"}
evidence_read: {abstract: true, full_text: europepmc, supplement: [Table S1]}
template: null          # biobank papers: {rule: "…", provenance: [ … ]}
scores:
  PGS000000:
    cluster: aging.lifespan
    phenotype: longevity
    measurement_kind: binary        # binary | continuous | ordinal | onehot_category | time_to_event | composite | response
    score_effect:
      direction: increases          # increases | decreases
      target: likelihood of reaching extreme old age
      coding: null
    polarity: 1
    evaluation_outcome: {phenotype: all_cause_mortality}   # or {expected_metric_sign: ±1, basis: "…"} or null
    strata: {}
    provenance:
      - source: abstract            # abstract | full_text | supplement | catalog
        locator: Abstract
        quote: "<copied verbatim from a file in lookup_dir>"
        confidence: high            # high | medium | low
    lead_variant_check: {status: agree, note: "top 5 variants: 4 agree, 1 not in GWAS Catalog"}
    review_reason: null     # one line when a human should check this direction before it is published
    corrections: []         # deliberate deviations from the paper, e.g.
    #  - kind: deposit_inverted          # deposit_inverted | partial_allele_inconsistency | broken_weights | metric_not_from_file
    #    action: polarity_as_distributed # polarity_as_distributed | quarantine | discount_metrics | none
    #    effect: "Direction describes the file as distributed; the paper says higher = increases."
    #    paper_direction: increases      # required for deposit_inverted
    #    evidence: "3/3 lead loci opposite; r = -0.95 vs consensus"
    #    recorded: 2026-09-27
    #    by: agent                       # agent | human:<name>
    #    upstream_report: null           # issue URL / e-mail date once reported to PGS Catalog
    status: agent_proposed
    notes: null
  PGS000001:
    score_effect: null
    unannotated_reason: No open-access text; the abstract does not state the case definition.
exploration_findings: |
  What did not fit the schema, and which field would have helped.
```

`curation/traits/<slug>.yaml` (created by `trait-scan`; the agent edits `decision`, `note`,
`concepts`, `status`, `summary`):

```yaml
slug: longevity
query: sort out longevity and lifespan scores
match: {terms: [age at death, longevity, lifespan], efo_ids: [], exclude: []}
anchor: PGS000318
status: in_progress                 # in_progress | needs_review | resolved
concepts: [aging.lifespan]
scores:
  PGS000906: {decision: in_scope, note: null}
  PGS000718: {decision: out_of_scope, note: drug response (beta-blocker survival benefit)}
summary: null
```

## Reviewer checklist (before `human_verified`)

1. Open the quote's locator in the paper: does it say what the entry claims?
2. Is `target` the phenotype the score was actually trained on, not the one it was evaluated on?
3. Do the concept's axis and valence read naturally for a non-expert?
4. Are the WARNs in `check` explained in `notes`?
5. For each `corrections` entry: is the evidence still true, and has it been reported upstream?

## Published tables (`just-dna-seq/pgs-catalog`, `data/metadata/`)

| File | Grain | Key columns |
|---|---|---|
| `score_annotations.parquet` | `pgs_id` | `annotated`, `annotation_status`, `cluster`, `axis`, `valence`, `phenotype`, `phenotype_sign`, `direction`, `target`, `coding`, `polarity`, `higher_percentile_means`, `evaluation_outcome_phenotype`, `strata_json`, `confidence`, `provenance_json`, `lead_variant_status`, `corrections_json`, `correction_kinds`, `unannotated_reason` |
| `trait_clusters.parquet` | `(cluster, phenotype)` | `domain`, `concept_label`, `axis`, `valence`, `valence_basis`, `concept_status`, `phenotype_label`, `phenotype_sign`, `n_scores` |
| `curation_manifest.json` | one per build | `built_at`, `source_commit`, counts, `sha256` of both parquets |

Unannotated entries are exported with `annotated = false` and their reason, so "not read" and
"no direction found" stay distinguishable. `rejected` entries are never exported. Consumers
show `agent_proposed` rows as unreviewed; `--verified-only` builds a reviewed-only release.
The Curated tab in the app compiles this directory on open (Reload picks up edits) and
browses the same resolved, review, and cluster tables; outside a clone it shows the
published tables instead.

## How the apps use what you publish

A push changes what users see, so curate with the consumer in mind (full policy: AGENTS.md,
"Consuming curated directions"; code: `just_prs.score_direction`):

- `PRSCatalog` (just-prs, prs-ui, just-dna-lite) pulls the published tables and re-pulls a copy
  older than `PRS_CURATION_MAX_AGE_HOURS` (default 24). `prs curation pull` refreshes now;
  `PRS_CURATION_DIR=data/output/curation` previews a local `prs curation build` without pushing.
- Every result row shows `higher_percentile_means`. Only **resolved** rows (published, not in
  `curation_review.parquet`) get valence colour: green for the favourable end, red only for the
  alarming end, none for `neutral` / `context_dependent` concepts.
- Trait summaries group by concept and put every resolved score on the concept axis
  (percentile × `polarity`). Review-queue rows (including every unconfirmed `deposit_inverted`),
  rows without a direction, uncurated scores and other concepts' scores are listed but not counted.
  A wrong `polarity` therefore flips a user's number; a wrong `valence` turns red into green.
- A `deposit_inverted` correction stays in review until a human verifies it; once verified the
  score is counted with its as-distributed direction and always shows an "inverted deposit" flag.
- Concept `aliases` feed the By Trait search and `PRSCatalog.search()`: add the words people type
  ("night owl", "bad cholesterol").
Push only when the user asks. Curation is incremental: curate any slice, push, and later pushes
add to it. Uncurated scores simply have no row.
