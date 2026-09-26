---
name: curate-traits
description: (Contributors) Review many traits at once, map-reduce style — one fresh trait-curator agent per trait, each in its own staging copy, then a serial merge. Use when asked to review several traits, "the next N priority traits", or a list like "height, LDL, smoking". Keeps the main conversation short; for a single trait discussed interactively use curate-trait.
argument-hint: "[top N | trait, trait, … ] [--max-lookups K] [--parallel P]"
allowed-tools: Bash(uv run prs curation *) Bash(prs curation *)
---

# Curate many traits (map-reduce)

You are the **orchestrator**. Don't curate anything yourself. Each trait gets a fresh `trait-curator`
agent with its own context. You stage, dispatch, merge, and report, and you keep only each agent's
short JSON report in this conversation. Rules and formats: `docs/curation-rules.md`.

Every command below is `uv run prs curation <command>` from the repo root.

## Inputs (`$ARGUMENTS`)

- `top N`: the N highest-priority uncurated traits. Run `prioritize` if
  `data/output/curation/trait_priorities.csv` is missing or older than the catalog, then take the first N rows
  with `n_curated = 0`.
- A comma-separated list of traits in plain words.
- `--max-lookups K` (default 4): lookup budget **per trait**.
- `--parallel P` (default 4): how many trait agents run at once.

Derive a short slug per trait (`ldl-cholesterol`, `body-height`, `smoking-status`) and 2–6 search
terms: synonyms plus opposite phrasings, for example never/ever and current/previous for smoking.

## 1. Plan (one message to the user, then go)

Show a table: slug, catalog label, n_scores, negative pairs from the priority file, search terms.
Warn when a trait has more than 100 scores; it will be split into rounds. Then start unless the
user objects. Don't wait for a reply when the user already asked for "top N" or gave a list.

## 2. Map

1. `stage <slug>` for every trait. Staging is cheap: it copies the small YAML files into
   `curation/staging/<slug>/`, and `.base/` keeps the snapshot the merge compares against.
2. Dispatch `trait-curator` agents with the Agent tool (`subagent_type: "trait-curator"`,
   `run_in_background: true`), at most P at a time. Each brief contains: `slug`, the user's words,
   the search terms, `root: curation/staging/<slug>`, `max_lookups`, and the priority-row signals
   (n_neg_pairs, min_r). Start the next trait whenever one finishes.
3. Don't read agent transcripts. Wait for completion notifications; each one returns a JSON report.

## 3. Reduce (serial, as each report arrives)

1. `merge <slug> --dry-run`, then `merge <slug>`. The merge is three-way per score and per
   phenotype: it takes what the agent changed and reports anything also changed in main, or already
   `human_verified`, as a CONFLICT. Conflicted staging dirs are kept.
2. After the merge, the merged tables are validated automatically. If `merge` exits non-zero
   (inconsistent clusters, for example two agents defining one phenotype differently), fix
   `curation/clusters.yaml` by hand, merging the definitions, then rerun `build`.
3. Record the report's `status`, counts, `lookups_used`, `suspected_sign_flips`, and `open_questions`
   in a running table. Nothing else from the report goes into this conversation.

Two agents may both create the same new concept (for example `lipids.ldl_cholesterol` from both an
"LDL" and a "total cholesterol" run). Identical definitions merge silently; different ones are a
conflict for you to settle, keeping one axis and valence and moving phenotypes under it.

## 4. Report (end of the batch)

- A table per trait: status, annotated/in-scope, catalog-only vs read, lookups used.
- **Suspected sign flips** (pairs with strong negative correlation inside one phenotype), listed
  first: they can mean inverted percentiles for users, not just a missing label.
- Open questions and conflicts that need a human.
- The command to push when the user is ready: `uv run prs curation push`. Never push unasked.
- A suggestion for the next batch (the next rows in `trait_priorities.csv`).

## Cost notes

- A trait agent starts with an empty context, so a batch costs about the sum of the traits, not a
  growing conversation. Run batches in a **fresh session**.
- Most spend is `score-lookup` calls. The trait agent is told to annotate catalog-only where the
  evidence allows and to look up only ambiguous papers. Lower `--max-lookups` to spend less; scores
  left unread are marked `not read: lookup budget reached`, and later batches pick them up.
- Check `/usage` before and after the first batch to calibrate how many traits fit your weekly budget.
