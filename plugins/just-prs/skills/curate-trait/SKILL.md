---
name: curate-trait
description: (Contributors) Sort out one trait end to end — e.g. diabetes, aging, longevity, intelligence — when its PGS scores point in different directions or are split across ontology terms. Finds every matching score, triages them (reported phenotype, metric signs, 1000G correlation), decides what is in scope, reads only the papers behind those scores with low-effort lookup agents, and produces a sorted view (concept → phenotype → polarity) with conflicts. Reuses the curate-papers files and checks. Use when the user starts from a trait rather than a paper.
argument-hint: "<trait, e.g. longevity | type 2 diabetes | intelligence> [status]"
allowed-tools: Bash(uv run prs curation *) Bash(uvx just-prs curation *) Bash(prs curation *)
---

# Curate by trait

The user starts from a question ("sort out longevity"). You find every score that belongs to it,
separate the directions, and leave behind annotations that the rest of the system reuses. Rules,
formats, and checks are shared with `curate-papers` and live in `docs/curation-rules.md`; read it
at the start of the session.

For contributors: run from a clone of the just-prs repo, where `curation/` lives. The helper is
`uv run prs curation <command>` (repo root; `uvx just-prs curation` works too once released), written below as
`curation <command>`. Write the full command every time.

Arguments (`$ARGUMENTS`): a trait in plain words, optionally followed by `status`. Derive a short
slug (`longevity`, `type-2-diabetes`, `intelligence`). With `status`, run only
`curation trait-status <slug>` and report it.

## What gets written where

- `curation/traits/<slug>.yaml`: the worklist, with search terms, one in/out-of-scope decision per
  score, the concepts this trait maps to, and a closing summary.
- Per-score facts go into the **same** `curation/publications/<PGP>.yaml` files that
  `curate-papers` uses, adding only this trait's scores. Never remove or rewrite another score's entry.
- New concepts and phenotypes go into `curation/clusters.yaml` as `agent_proposed`.
- Don't `claim` or `finish` papers in the paper queue: that queue tracks whole-paper coverage.
  Progress for this work goes through `trait-log`.

## Steps

1. **Scan.** Choose search terms that cover the trait's synonyms and opposite phrasings (for
   longevity: longevity, lifespan, life span, age at death, all-cause mortality, survival). Run
   `curation trait-scan <slug> --query "<the user's words>" --term … [--efo …]`. Read the triage:
   - the distinct reported phenotypes. Rerun with more `--term` / `--efo` if obvious members are
     missing (terms accumulate), or with `--exclude <regex>` to drop false hits;
   - `grp` / `r_anc`: sign of the 1000G EUR correlation with the anchor score. A `-` group next to a
     `+` group is a direction split. It is a hint to verify, not evidence;
   - `min_r(partner)`: the most negatively correlated peer;
   - `metric → evaluated as`: HR/OR/beta sign and the outcome it was measured on. Metrics evaluated
     on another outcome (a longevity score evaluated on CAD) are common traps;
   - `annotation`: scores that are already curated. Reuse them; never re-decide `human_verified` ones.
2. **Scope.** Edit the worklist: set `decision: in_scope` or `out_of_scope` (with a `note`) for
   every score, and set `concepts`. Use the reported phenotype and metadata for this, not the
   papers. Split into separate concepts when a person would read them as different questions: for
   "aging", lifespan, frailty, and visible aging are three concepts in one domain. Drug-response
   scores are usually out of scope for the underlying trait. Draft the concepts and phenotype signs
   in `clusters.yaml`.
3. **Checkpoint with the user when the scope is large.** If more than 40 scores are in scope, or
   the split into concepts is not obvious, show the proposed concepts with counts per phenotype,
   and ask which groups to resolve first. Otherwise continue.
4. **Resolve.** Group the in-scope, not-yet-annotated scores by paper (`trait-status` lists them).
   For each paper: run `curation context <PGP> --pgs-ids <this trait's scores> --top-variants 5`,
   dispatch a `score-lookup` agent (`subagent_type: "score-lookup"`, up to 4 in parallel) with the
   standard questions plus trait-specific ones, write the entries, then run `curation check <PGP>`.
   Fix ERRORs; for a WARN send one narrower follow-up lookup. After each paper, run
   `curation trait-log <slug> --message "<PGP>: <n> scores" --lookups <K>`.
5. **Consolidate.** Run `curation trait-status <slug>`. Look at:
   - aligned-correlation conflicts: two scores in one concept that disagree after alignment. Re-read
     both with a targeted lookup, since a phenotype or polarity is probably wrong;
   - concepts used but not listed in the worklist;
   - in-scope scores still unread, or unannotated with a reason.
   Then set the worklist `status`: `resolved` when every in-scope score is annotated or unannotated
   with a reason and there are no unexplained conflicts, otherwise `needs_review`. Write a
   three-to-six-line `summary` in plain words.
6. **Report** to the user:
   - the sorted view from `trait-status` (concept → `+`/`−` phenotype → scores);
   - what a high percentile means for each group, for example "higher PGS000318 = shorter expected life";
   - conflicts and open questions, and the out-of-scope scores with the reason;
   - the latest `trait-log` line.

## Large traits

Diabetes-sized traits (hundreds of scores from biobank papers) go in rounds: the first round covers
the papers behind the most distinct reported phenotypes, and biobank papers get a `template` entry
(see curate-papers) limited to this trait's scores. Stop after each round of about 10 papers with a
`trait-log` line and a short report, so the user can redirect.

## Publishing

`curation build` compiles everything into parquet (`data/output/curation/`) and fails on any
inconsistent file, so run it at the end of a batch as a final check. `curation push` uploads the
tables to Hugging Face (`just-dna-seq/pgs-catalog`, `data/metadata/`). Run it only when the user
asks.
