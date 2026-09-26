---
name: curate-papers
description: (Contributors) Curate PGS scores paper by paper — work through the publication queue (curation/queue.csv), read each PGS Catalog paper through low-effort lookup agents, record what a higher score means (direction) and where it sits in the trait hierarchy (curation/clusters.yaml), and track progress. Use for Phase 0 exploration, batch annotation of the catalog, a specific PGP publication, or annotation status. For "sort out <trait>" questions use curate-trait instead.
argument-hint: "[explore | next N | PGP000123 | status]"
allowed-tools: Bash(uv run prs curation *) Bash(uvx just-prs curation *) Bash(prs curation *)
---

# Curate by paper

You are the **orchestrator**: you pick work, brief lookup agents, make every judgment call, and
write the YAML. The rules, file formats, and check semantics are in `docs/curation-rules.md`. Read it
at the start of the session; it is shared with `curate-trait`. The design rationale is in
`docs/score-annotation-plan.md`.

For contributors: run from a clone of the just-prs repo, where `curation/` lives. The helper is
`uv run prs curation <command>` (repo root; `uvx just-prs curation` works too once released), written below as
`curation <command>`. Write the full command every time; shell state does not persist.

Arguments (`$ARGUMENTS`):

| Argument | Do |
|---|---|
| `explore` | Next batch from the Phase 0 exploration set (`--phase explore`) |
| `next N` | Next N publications in queue order (default 10) |
| `PGP000123` | That one publication |
| `status` | Print status and the review backlog only |
| _(empty)_ | `status`, then ask whether to run `explore` or `next` |

## Roles and effort

- **You:** decisions and YAML.
- **`score-lookup` agent** (Opus, `effort: low`): fetches and quotes. Dispatch it with the Agent
  tool, `subagent_type: "score-lookup"`. Run up to 4 in parallel, one per publication. Don't fetch
  papers yourself unless the lookup failed twice.
- **`curation check`:** deterministic, no LLM. Its verdict beats your impression.

## Loop

1. `curation status`. With no queue, run `curation init` first (idempotent).
2. `curation next --n <N> [--phase explore]`. `target_pgs_ids` limits scope in the explore phase;
   empty means all of the paper's scores. The context's `existing_annotation` may already hold
   entries written by `curate-trait`: keep them, and fill in only what is missing.
3. For each publication:
   1. `curation claim <PGP>`, then `curation context <PGP>`. Add `--top-variants 5` when there are
      at most 20 target scores; otherwise pass `--pgs-ids` in blocks of ≤ 20.
   2. Dispatch `score-lookup` with `pgp_id`, `pmid`/`doi`, `lookup_dir`, the target scores with
      their reported traits, the lead-variant rsIDs with effect allele and weight sign, the
      standard questions from `docs/curation-rules.md`, and any score-specific questions (for
      example, "the catalog says HR 0.89 against 'Survival': hazard of what?").
   3. Write `curation/publications/<PGP>.yaml` from the returned excerpts. Reuse concepts from the
      context's `clusters` list; add a concept or phenotype to `curation/clusters.yaml` (as
      `agent_proposed`) only when nothing fits. Fill `lead_variant_check` from the lookup's `variants`.
   4. `curation check <PGP>`. Fix every ERROR. For a WARN, send **one** narrower follow-up lookup.
      If it is still unresolved, keep the entry with `confidence: low` and explain it in `notes`,
      or set `score_effect: null` with `unannotated_reason`.
   5. `curation finish <PGP> --lookups <K>`, or `--blocked "<reason>"` when there is no usable text.
      `finish` also requires an entry for every target score. Echo its progress line.
4. End of batch: `curation status`, then report the latest progress line, the WARNs that need a
   human, and any concepts you proposed. Stop at the batch size.

## Biobank template papers (priority 4, more than 100 scores)

Put the paper-wide rule in `template` with its own quote, then still write every score entry.
Entries may reuse the template quote, but reverse-coded fields, one-hot categories, and composites
need their own. Claim once, then process blocks of ≤ 20 scores. After each block, run
`curation check <PGP>` and then `curation log <PGP> --message "block 3/39 (PGS001100..PGS001119)"`.
Run `finish` only after the last block. A session that stops mid-paper leaves it `in_progress`;
`next` reports it as interrupted after 6 h, and the blocks already written stay in the YAML.

## Phase 0 (explore)

The set spans nine types (diseases, appearance, aging, numeric, categorical/ordinal, drug response,
biobank mass papers, composite/factor, stratified). Its purpose is to find out **what the schema is
missing**. After each publication, add a short section to `curation/exploration/NOTES.md`: scores
covered, what was easy, what did not fit, and which field would have helped. When the set is done,
propose a schema v1 to the user. Don't change the models in `prs curation` (`just_prs.curation`) or the docs
before the user agrees.

## Publishing

`curation build` compiles everything into parquet (`data/output/curation/`) and fails on any
inconsistent file, so run it at the end of a batch as a final check. `curation push` uploads the
tables to Hugging Face (`just-dna-seq/pgs-catalog`, `data/metadata/`). Run it only when the user
asks.
