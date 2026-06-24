# RefCall / Reference-Allele Resolution — Plan (deferred coverage-recovery branch)

> **Status (A + B implemented).** Items **A (reference-allele resolution)** and **B
> (reference-FASTA resource)** are done, with a key design change from the original
> sketch: instead of resolving REF on the fly per run, the catalog's scoring positions
> are resolved **once** into a small `reference_allele_universe.parquet`
> `(genome_build, chrom, pos, ref, ref_source)` and published to HF — the 3 GB genome
> FASTA is a *precompute-only* input, never a runtime resource (no whole-genome parquet).
> Verified upstream that polars-bio cannot do vcf→gvcf and that faidx via **pysam** (not
> a bespoke reader) is the right FASTA tool. Implementation: `just_prs/reference_allele.py`
> (`resolve_reference_alleles`), `reference.py` (`download_reference_fasta` /
> `reference_fasta_path` / `REFERENCE_FASTA`), `prs.py` (`resolve_reference` flag +
> `_apply_reference_resolution` + `variants_ref_resolved_panel/_fasta` counters on both
> engines), `PRSCatalog.compute_prs(..., resolve_reference=True)` with HF pull-on-miss,
> `hf.py` (`push_/pull_reference_allele_universe`), and the Dagster
> `reference_allele_pipeline` (`ensembl_grch38_fasta` → `reference_fasta` →
> `reference_allele_universe` → `hf_reference_allele_universe` + asset check). `pysam` is
> in the `[reference]` extra with the `sys_platform != 'win32'` marker (precompute is
> Linux/WSL; runtime reads the small parquet and never needs pysam). The flag defaults
> **off** until the empirical WGS coverage gate passes. **Items C (gVCF END-block
> expansion) and D (array ALL_SITES + maf_fill) remain TODO.**

Branch: `refcall-resolution` (stacked on `scoring-foundations`). This is the deferred
**full P1 / F15 / F22** work the demo-safe foundations round intentionally skipped. It is
the lever that turns the ~50% genome-wide WGS coverage into near-complete coverage, and
it is where `variants_unscorable_absent` is meant to go to ~0.

**Do not start until `scoring-foundations` is merged/stable.** This plan assumes the
foundations are present (C_wt accounting, `variants_unscorable_absent`, the explicit
`genotype_input_mode` flag, the true z-score chain).

## Context — why coverage is stuck at ~50%

A plain WGS VCF records a row only where the sample differs from reference; a PGS scoring
variant **absent** from that callset means the sample is hom-ref there (callability
permitting). The engine already imputes hom-ref for an absent variant **when the scoring
file carries `reference_allele`** (`prs.py` VARIANT_ONLY → `variants_assumed_hom_ref`). The
residual gap is `variants_unscorable_absent`: absent variants whose reference allele is
unknown, so dosage can't be oriented. Genome-wide scoring files frequently omit
`reference_allele`/`other_allele`, so ~half of every genome-wide score lands here.

Two independent sources can supply the missing REF base:

1. **Reference panel `.pvar`** — already in-repo and parsed (`parse_pvar`, preserves
   `REF`/`ALT` and the `CHROM:POS:REF:ALT` `ID`). Covers all panel positions for free; no
   new download. This is the cheap first tier.
2. **GRCh38 reference FASTA** (faidx) — the universal fallback for positions not in the
   panel. ~3 GB new resource, fetched/cached like the existing reference panels.

Plus the gVCF case (F22): when the input *is* a gVCF, `END`-spanned hom-ref reference
blocks carry true dose-0 information that `normalize_vcf` currently discards.

## Work items

### A. Reference-allele resolution for absent score variants (F15 — the main lever)

- **New module** `just_prs/reference_allele.py` (or extend `reference.py`):
  `resolve_reference_alleles(scoring_df, genome_build, *, panel_pvar=..., fasta=...)`.
  For each scoring variant lacking `reference_allele`:
  1. **Panel tier:** left-join against the panel `.pvar` parquet on `(chrom, pos)` (DuckDB,
     same pattern as `_ResolvedRefPanel.match_scoring`) and take its `REF`. Verify one of
     {effect_allele, other_allele} equals the panel REF or ALT before trusting it; on a
     multiallelic/mismatch, leave unresolved (do not guess).
  2. **FASTA tier:** faidx lookup of the single REF base at `(chrom, pos)` for the
     remainder. Verify the resolved base is consistent with the score's alleles.
  - Returns the scoring frame with a filled `reference_allele` plus a per-variant
    `ref_allele_source` ∈ {`scoring_file`, `panel`, `fasta`, `unresolved`} for accounting.
- **Wire into the engine** (`prs.py`): when `genotype_input_mode == VARIANT_ONLY` and
  `reference_allele` is null, consult the resolved value. An absent score position then
  scores as hom-ref (dose 2 if effect==REF, else 0), dissolving `variants_unscorable_absent`.
  Keep it **off by default behind a flag** (e.g. `resolve_reference=False`) until validated,
  then flip the default once coverage recovery is confirmed.
- **Accounting:** add `variants_ref_resolved_panel` / `variants_ref_resolved_fasta` counters;
  `variants_unscorable_absent` should fall toward 0 on a WGS input where REF resolves.

### B. Reference-FASTA resource (mirrors the panel download)

- `download_reference_fasta(genome_build, cache_dir)` + `reference_fasta_path(...)` alongside
  `download_reference_panel`. Cache under `<cache>/reference_fasta/`. Reject zero-byte/corrupt
  downloads; verify the `.fai` index (build it if missing).
- **Build/contig verification (correctness-critical):** before trusting any FASTA REF base,
  confirm contig naming (chr-prefix vs not) and build (GRCh37 vs 38) match the normalized
  genotypes and the scoring build. A silent contig/build mismatch would inject wrong REF
  bases — fail loudly / mark `unresolved` rather than guess.

### C. gVCF `END`-block expansion (F22 — the rigorous form for gVCF inputs)

- In `normalize.py` (or a span-aware reader): when the input is a gVCF, index reference
  blocks `(chrom, start, end)` and, at score time, resolve a score position falling inside a
  confident hom-ref block to dose-0/2 *matched* (Genomi's `spans`-join pattern). Gate on the
  block GT `0/0` and `MIN_DP`/`GQ` clearing a threshold (`absence_allowed = has_reference_blocks
  and has_depth`). For array/sparse VCF, absence stays `missing` — never silently hom-ref.
- This is lower priority than A (99% of inputs are plain VCF), but it is the *rigorous* answer
  for the gVCF inputs F22 documents.

### D. Array absence-semantics (moved here from the foundations round)

- Forcing arrays to `ALL_SITES` is correct (absent = off-chip = missing, never hom-ref) but
  the engine currently **ignores `maf_fill` in the ALL_SITES branch** — so a naive switch
  silently disables the array path's intentional MAF-fill recovery. Fix here by teaching the
  ALL_SITES branch to honor `maf_fill` (port the `do_maf_fill` dosage/​counter logic from the
  VARIANT_ONLY branch in both engines), then set `compute_array_prs` to pass
  `genotype_input_mode=ALL_SITES`. Verify `variants_maf_filled` / `effective_coverage` are
  unchanged for arrays while phantom hom-ref is eliminated.

## Critical files

- `just-prs/src/just_prs/reference.py` (panel `.pvar` REF tier, FASTA fetch) and/or new
  `reference_allele.py`.
- `just-prs/src/just_prs/prs.py` (consult resolved REF in the VARIANT_ONLY dosage chain;
  ALL_SITES + maf_fill; new counters), `models.py` (`PRSResult` counters).
- `just-prs/src/just_prs/normalize.py` (gVCF span index/expansion).
- `just-prs/src/just_prs/array_scoring.py` (flip to ALL_SITES once D lands).

## Verification

- **Primary empirical gate:** on a plain WGS VCF, with resolution on, genome-wide scores
  recover coverage (track ~50% → recovered) and `variants_unscorable_absent` → ~0 where REF
  resolves; an array input gains **no** phantom hom-ref (absent → `missing`).
- **Engine parity:** the resolved-REF path produces identical scores/counters in polars and
  DuckDB.
- **Build/contig guard:** a deliberately mismatched FASTA (GRCh37 vs GRCh38, chr-prefix) marks
  variants `unresolved` rather than injecting wrong REF.
- **gVCF:** expanding blocks on a real gVCF jumps coverage toward ~100% for genome-wide scores
  (the F22 definitive test).
- **Regression:** `scripts/smoke_test_all_prs.py` over the 10-genome set; reduced same-sample
  same-trait percentile spread across genome-wide scores after recovery (the F15/F20 symptom).
- `uv run python -m pytest just-prs/tests/ -v` stays green; add deterministic network-free
  tests for `resolve_reference_alleles` (panel + a tiny synthetic FASTA) and the ALL_SITES
  maf_fill parity.

## Sequencing

`A (panel tier)` first — it needs no new download and likely closes most of the gap. Then
`B (FASTA tier)` for the long tail, behind the build/contig guard. `D (array + ALL_SITES
maf_fill)` is independent and small. `C (gVCF spans)` last — lowest input frequency, highest
parsing effort. Flip `resolve_reference` to default-on only after the empirical coverage gate
passes on real WGS data.
