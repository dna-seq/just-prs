# Batch 2 — Metadata Surfacing & Convenience — Plan

Branch: `metadata-surfacing` (off `main`, which already has the scoring-foundations
round: `C_wt`, `PercentileResult`, `percentile_full`, `attach_performance`). This is the
**next easiest batch** after the foundations round — all additive, self-contained, no new
heavy resources and no research. It is independent of, and complementary to, the deferred
coverage work in `docs/refcall-resolution-plan.md` (F15/F22).

## Why these, why now

After `just-prs` 0.4.8 shipped the foundations (F9/F10/F11/F12 resolved, F20 partial), the
genuinely-pending items split into: **heavy/research** (F15/F22 coverage recovery →
already planned; F19 sample-ancestry inference + veto; F20 composite-`Q` → roadmap P4) and
a cluster of **cheap, additive** fixes that need no engine-internal changes. This batch is
that cheap cluster. Each item is independently revertible.

Selection rationale (verified against the code on `main`):

- **F23** is a one-parameter addition — `PRSCatalog.compute_prs` is missing the
  `genotypes_lf` that `compute_prs_batch` already has, and the free `just_prs.prs.compute_prs`
  already accepts it. Trivial.
- **F4** is pure wiring — `just_prs.vcf.detect_genome_build()` already implements the
  contig-length + `##reference` detection; nothing surfaces or acts on it.
- **F19 (echo slice only)** — evaluation `ancestry_broad` already flows via
  `PerformanceInfo`; only the percentile **panel** ancestry needs echoing. Development
  ancestry sourcing, sample-ancestry inference, and the coherence/veto stay **deferred**.

Explicitly **NOT** in this batch: F15/F22 (refcall branch), F19 inference/veto + dev
ancestry (research, roadmap P3), F20 composite `Q` (roadmap P4), F2 fuzzy trait search
(different subsystem — REST client; can be its own small batch later).

## Work items

### 1. F23 — `genotypes_lf` on `PRSCatalog.compute_prs` (trivial, unblocks the wrapper)

- Add `genotypes_lf: pl.LazyFrame | None = None` to `PRSCatalog.compute_prs`
  (`prs_catalog.py`), forward it to the free `compute_prs(..., genotypes_lf=...)`, then run
  the existing `_attach_performance` when requested — exactly mirroring `compute_prs_batch`.
  Optionally also forward `genotypes_parquet` for the DuckDB-direct path / engine selection,
  matching the batch method's options (keep minimal: `genotypes_lf` is what F23 asks for).
- **Why it matters:** gives a single-score one-call path that both reuses a normalized
  Parquet **and** attaches best performance. The MCP wrapper then drops its free-function
  branch where `attach_performance=True` is currently silently a no-op.
- **Files:** `prs_catalog.py` (`compute_prs`).
- **Acceptance:** `catalog.compute_prs(vcf_path, pgs_id, genotypes_lf=lf, attach_performance=True)`
  returns a `PRSResult` with `performance` populated and without re-reading the VCF; result
  equals the `vcf_path`-only path on the same genotypes.

### 2. F4 — surface detected genome build + mismatch warning (wiring only)

- `just_prs.vcf.detect_genome_build(vcf_path)` already returns `"GRCh37"`/`"GRCh38"`/`None`.
  Wire it into the compute path:
  - In `PRSCatalog.compute_prs` / `compute_prs_batch` (and/or the free `compute_prs`),
    when a real `vcf_path` is available, call `detect_genome_build(vcf_path)` and record it.
  - Add to `PRSResult`: `detected_genome_build: str | None` and `build_mismatch: bool`
    (True when a build was detected and differs from the `genome_build` used for scoring).
    Emit an Eliot warning on mismatch.
  - When genotypes are supplied as a pre-normalized LazyFrame/Parquet (no VCF header),
    detection returns `None` → `detected_genome_build=None`, `build_mismatch=False` (cannot
    prove a mismatch; do not guess).
- Optionally record the detected build in `normalize_vcf` output metadata too, but the
  result-surfacing is the F4 deliverable.
- **Files:** `vcf.py` (reuse `detect_genome_build`), `prs.py` / `prs_catalog.py`, `models.py`
  (`PRSResult` + `EnrichedPRSResult` fields), `enrich.py` (pass-through).
- **Acceptance:** a GRCh37-contig VCF scored against a GRCh38 scoring file yields
  `detected_genome_build="GRCh37"`, `build_mismatch=True`, and a logged warning; a matching
  build yields `build_mismatch=False`; a header-less normalized-Parquet input yields
  `detected_genome_build=None` (no false mismatch).

### 3. F19 (echo slice) — surface the percentile reference-panel ancestry

- Add `ancestry: str | None` (the superpopulation actually used) and `panel: str | None`
  to `PercentileResult`; `percentile_full` already receives `ancestry`/`panel`, so set them
  on the returned object (only meaningful for the `reference_panel` method; `None`/`""` for
  theoretical/AUROC).
- Surface the percentile **panel ancestry** on `EnrichedPRSResult` (a
  `reference_panel_ancestry` field) so the wrapper can show "scored vs <panel ancestry>
  reference." Evaluation `ancestry_broad` is already available via `PerformanceInfo`
  (foundations `attach_performance`) and `EnrichedPRSResult.ancestry` — keep using it.
- **Explicitly deferred (NOT this batch):** per-score **development** ancestry from the PGS
  Catalog "Ancestry Distribution" metadata (needs a cleanup-pipeline column that may not
  exist yet), **sample**-ancestry inference (plink2 projection / peddy / somalier), and the
  3-way coherence **veto** (`K_anc`). Those are roadmap P3 and remain research.
- **Files:** `models.py` (`PercentileResult`, `EnrichedPRSResult`), `prs_catalog.py`
  (`percentile_full` sets `ancestry`/`panel`), `enrich.py` (echo).
- **Acceptance:** a reference-panel percentile reports which superpopulation/panel it used;
  the wrapper can render "percentile relative to <panel> <ancestry>" without guessing.

### 4. F16 — reconcile trait associated-ID count (investigate-first, low priority)

- `TraitInfo.associated_pgs_ids` (REST, `catalog.py:search_traits`) and the by-trait scoring
  denominator come from different retrieval paths and disagree (~195 vs 220 for
  MONDO_0005148). **Investigate both paths first**; only fold in if the reconciliation is a
  small, clear change (e.g. both should read the same cleaned-metadata association set).
  If it turns out to need REST-vs-bulk reconciliation logic, split it out — do not let it
  expand this batch.
- **Files:** `catalog.py`, `prs_catalog.py` (association lookups).
- **Acceptance:** the directly-associated PGS-ID count for a trait is consistent across the
  metadata and compute paths, or a clear note on why they legitimately differ.

## Verification

- `uv run python -m pytest just-prs/tests/ -v` stays green.
- **F23:** new network-light test — `compute_prs(..., genotypes_lf=lf, attach_performance=True)`
  equals the `vcf_path` path and has `performance` populated (reuse the existing test VCF +
  a pre-scanned normalized parquet).
- **F4:** deterministic test — a synthetic VCF with GRCh37 contig lengths is detected as
  GRCh37 and flagged against a GRCh38 scoring build; a normalized-Parquet input yields
  `detected_genome_build=None` and no mismatch.
- **F19:** a reference-panel percentile result carries the panel ancestry it used (mock/skip
  if reference distributions are unavailable in the test env, like the existing percentile
  tests).
- Real-data smoke (`scripts/smoke_test_all_prs.py`) unchanged in score values — this batch
  is metadata-only.

## Sequencing

`F23 (1)` and `F4 (2)` are independent and trivial — do them first. `F19 echo (3)` needs
the `PercentileResult`/`enrich` touch. `F16 (4)` is investigate-first and optional; drop it
from the batch if it isn't a small clear change. Commit on `metadata-surfacing`; merge to
`main` after the suite is green, same flow as the foundations round.
