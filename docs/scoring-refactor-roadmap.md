# just-prs Scoring Refactor — Roadmap

A dependency-ordered plan to move `just-prs` from a raw-score engine to a
**posterior, per-genome quality-gated** risk engine. Strategic altitude — phases,
dependencies, file pointers, acceptance criteria. No code here.

**Scope:** the `just-prs` library only. MCP-wrapper (`just-prs-mcp`) surface changes
are downstream and out of scope. All paths are under
`just-prs/src/just_prs/` unless noted.

## Why

The field gates PGS models on **priors** (variant-count caps, declared ancestry,
declared performance). just-prs can afford to gate on **posteriors** — compute every
applicable score against *this* genome, judge each result's quality vs *this*
callset, drop noise, and combine survivors weighted by quality. The design thesis is
in [`posterior-quality-gating.md`](posterior-quality-gating.md); the competitor
evidence and the lettered Alt Ideas (A1–A9) are in
`just-prs-mcp/docs/competitor-research.md`; the bug/UX findings (F#) are in
`just-prs-mcp/docs/just-prs-pending-fixes.md`.

**The key fact that shapes this work:** most building blocks already exist. This is
mostly a *connect-and-re-point* refactor, not a rebuild.

## 0. Inventory — what already exists

| Capability | Status | Where |
|---|---|---|
| HWE analytic null (A2) — `E=Σ2pw`, `Var=Σ2p(1−p)w²` | **Exists** | `_compute_theoretical_stats` `prs.py:189-243` → `theoretical_mean/std` on `PRSResult` |
| Hom-ref imputation (A3 / "effect==ref→dose 2") | **Exists** | VARIANT_ONLY mode → `variants_assumed_hom_ref`, `variants_maf_filled` (`prs.py`, `models.py:187-250`) |
| 3-tier percentile (ref-panel → theoretical → AUROC-approx) | **Exists** | `prs_catalog.py:775-841`; `ancestry_percentile` `reference.py:2493-2529` |
| 3-method absolute risk incl. **h²-liability `Φ((z·√r²−t)/√(1−r²))`** (= our A1 √R²) | **Exists** | `absolute_risk.py` (OR-per-SD, AUC-bivariate, h²-liability ~`285-371`) |
| Synthetic (prior) + combined (populational posterior) quality + classifiers | **Exists** | `quality.py:27-272`; build in `scripts/build_pgs_quality_scores.py` |
| Reference-panel scoring (pgenlib/plink2); HGDP+1kGP panel | **Exists** | `reference.py` (`compute_reference_prs_polars` `1291-1527`) |
| Cross-genome smoke harness (per-superpop pct + stability) | **Exists** | `scripts/smoke_test_all_prs.py`, `rate_prs_quality.py` |
| Score development/evaluation ancestry | **Buried** | `ancestry_broad` (evaluation cohort) in `best_performance` — not on `ScoreInfo` |
| WGS VCF **and** consumer-array (23andMe/AncestryDNA) ingest | **Exists** | `normalize.py`; `normalize_array` (`arrays.py`, CLI `normalize-array`) |
| Input-type / absence-mode flag | **Exists, inference-only** | `GenotypeInputMode` `prs.py:20`; `_infer_genotype_input_mode` `prs.py:147` — add an authoritative caller flag, keep inference as the omitted-flag fallback |
| Reference-allele lookup for absent score variants (→ orient hom-ref dose) | **Missing** | absent + unknown REF ⇒ `variants_unscorable_absent`; no Ensembl FASTA/dbSNP lookup |
| gVCF `END` block expansion (rare input only) | **Missing / low-priority** | `normalize.py` keeps one row per record; 99% of inputs are plain VCF |
| z-score / reference mean-std exposed on result | **Missing** | computed inside `percentile()`, never returned |
| Weight-mass coverage `C_wt` | **Missing** | only count-based `match_rate` exists |
| Sample-ancestry inference; ancestry coherence/veto | **Missing** | — |
| Per-(score×genome) posterior `Q`; within-genome stability | **Missing** | only the cross-genome populational `combined_quality_score` |

## Progress

- **Round 1 — demo-safe foundations (landed 2026-06-22, `scoring-foundations` branch).**
  Additive, no change to the core score value:
  - **P1 (partial):** weight-mass coverage `C_wt` (`weight_mass_coverage` +
    `weight_mass_matched/total` on `PRSResult`, both engines; GenoBoost uses
    max|dosage_k_weight| as the mass surrogate). FASTA reference-allele resolution
    still deferred (see `refcall-resolution` branch).
  - **P2 (F12):** `PercentileResult` + `PRSCatalog.percentile_full()` expose the true
    `z_score`/`reference_mean`/`reference_std`; `PRSResult` carries them on the
    theoretical path; `absolute_risk_from_score()` chains raw score → z → risk;
    `enrich.py` feeds the true z into absolute risk instead of inverting the percentile.
  - **F9/F20 (partial):** `percentile_full` attaches a `C_wt`-driven `reliable`/`caveat`
    verdict (floor `MIN_RELIABLE_WEIGHT_MASS_COVERAGE = 0.20`).
  - **F10/F11:** `interpret_prs_result` is method-aware; `format_effect_size`/
    `format_classification` return `None` not `""`; `compute_prs(_batch)` gain
    `attach_performance`.
  - **Deferred to `refcall-resolution`:** FASTA/`.pvar` REF resolution + gVCF `END`
    expansion (full P1/F15/F22), array ALL_SITES semantics (entangled with MAF-fill).
    Still design-only: sample-ancestry/`K_anc` (P3), runtime `Q`/`R_tech` (P4),
    consensus (P5), cost-gate (P6).

## Phases

Format per phase: intent · build · files · acceptance · risk.

### Phase 1 — Coverage truth (F15; F22 only for the rare gVCF) — *root cause, do first*

Reality check: **~99% of personal genomes arrive as a plain, variant-only VCF**;
gVCF is a rare artifact, so gVCF `END`-block expansion is the wrong thing to
optimize. A plain VCF records only sites where the sample differs from reference, so
a PGS score variant **absent** from a whole-genome callset means the sample is
**hom-ref** there (callability permitting). The ~50% coverage gap (F15) is dominated
by these absent-but-hom-ref score positions, not by gVCF blocks.

The engine already imputes hom-ref for an absent variant **when the scoring file
carries the reference/other allele** (VARIANT_ONLY → `variants_assumed_hom_ref`,
`prs.py:402-414`). The residual gap is `variants_unscorable_absent` — absent
variants whose reference allele is unknown, so dosage can't be oriented; genome-wide
scoring files frequently omit it.

- **Build:**
  - **Resolve the reference allele at each score position from an Ensembl GRCh38
    reference FASTA** (faidx lookup; optionally dbSNP / Ensembl Variation for
    rsID-only files). Whichever of {effect, other} matches the reference base is REF;
    an absent score position then scores as hom-ref (dose 2 if effect==REF, else 0),
    dissolving `variants_unscorable_absent`. The reference-panel `.pvar` is an in-repo
    REF source for panel positions; the FASTA is the universal fallback.
  - Add **weight-mass coverage `C_wt`** = Σ|β|(matched) / Σ|β|(total) beside the
    count-based `match_rate`.
  - Keep gVCF `END`-block / depth handling as an **opportunistic enhancement** for
    the rare gVCF input (it gives true per-site callability), not the primary lever.
- **Input type is an explicit caller flag — never inferred (correctness-critical):**
  - Both inputs are first-class and **already ingested**: **WGS/WES VCF**
    (`normalize.py`) and **consumer arrays** (23andMe / AncestryDNA) via
    `normalize_array` (`arrays.py`, CLI `normalize-array`). When the caller declares
    the source, that flag is **authoritative — never overridden by a guess**. When it
    is **omitted** (the default: assume a non-expert user who declares nothing), fall
    back to `_infer_genotype_input_mode` (`prs.py:147`) and **surface the resolved
    type** in the result so it's transparent. When inference is genuinely ambiguous,
    bias to the **conservative** semantics (absent → `missing`) — never fabricate
    hom-ref on a guess (mis-calling a chip as WGS would invent phantom coverage).
  - The declared type sets absence-semantics, which are **opposite**:
    - **WGS/WES whole-genome callset** (`VARIANT_ONLY`): a record exists only where
      the sample differs from reference ⇒ absent score positions are **hom-ref**
      (oriented via the FASTA REF lookup above).
    - **Array / chip** (`ALL_SITES`): the file already lists genotypes (incl. hom-ref)
      at every assayed site ⇒ absent positions are **off-chip = `missing`, never
      hom-ref**; the FASTA hom-ref trick does **not** apply. Low `C_wt` for a
      genome-wide score on a chip is the *honest* result (per-chip coverage is already
      modeled — `{chip}_coverage`, `prs_catalog.py:162`).
  - Even on a declared WGS callset, absence ⇒ hom-ref still assumes the site was
    **callable**, which a bare VCF can't prove. Keep it the default but **flag it**;
    when callability evidence exists (callable-regions BED, per-site DP, or a gVCF)
    downgrade non-callable absent sites to `missing`. Feeds `C_wt`/reliability (P2)
    and the `H_hwe` artifact check (P4).
- **Files:** `prs.py` (absence handling + accounting `402-414`; keep inference
  `147-175` as the omitted-flag fallback, bias it conservative), `arrays.py` (array
  path stamps its mode), `reference.py` (FASTA /
  `.pvar` REF lookup), `normalize.py`, `models.py` (`PRSResult`); a reference-FASTA
  resource fetch alongside the existing EBI panel download.
- **Acceptance:** an explicit input-type flag is **honored when given** and the
  resolved type (declared or inferred) is **surfaced** in the result; with the flag
  omitted, inference runs and stays conservative (no fabricated hom-ref on an
  ambiguous guess). On a WGS input, genome-wide scores recover coverage (track
  ~50% → recovered) and `variants_unscorable_absent` → ~0 where the FASTA resolves
  REF; an array input gains **no** phantom hom-ref (absent → `missing`); `C_wt`
  present; hom-ref-by-absence is flagged; engine-parity tests pass.
- **Risk:** (a) over-optimistic hom-ref on non-callable regions — mitigate with the
  callability flag; (b) FASTA↔VCF contig/build mismatch (chr prefix, GRCh37 vs 38)
  must be verified before trusting a REF base; (c) the reference FASTA is a new
  ~3 GB resource — fetch/cache like the panels.

### Phase 2 — Close the raw→risk chain (F12 / F11 / F9) — *needs P1*

Today the z-score and the reference mean/std are computed *inside* `percentile()` and
thrown away; callers must re-derive z by hand, and absolute risk takes z as an input.

- **Build:** expose `z_score`, `reference_mean`, `reference_std`, and the
  percentile-method-actually-used on `PRSResult`. Add an `absolute_risk_from_score`
  convenience that chains compute → percentile → risk. Make `percentile()`
  coverage-aware: attach a reliability verdict driven by `C_wt` (F9), so a deflated
  low-coverage score can't emit an authoritative extreme percentile.
- **Files:** `models.py`, `prs_catalog.py` (`percentile` `775-841`, `absolute_risk`
  `1163-1263`), `enrich.py`.
- **Acceptance:** a single call takes a raw score to z → percentile → absolute risk;
  the percentile output carries a `C_wt`-tied reliability flag; the h²-liability path
  (A1) is reachable from a raw score without manual z wiring.

### Phase 3 — Ancestry surfacing + K_anc (F19) — *parallel to P2*

- **Build:** surface score **development/evaluation ancestry** on `ScoreInfo` (lift
  `ancestry_broad`, the evaluation-cohort ancestry, out of `best_performance`; add
  development ancestry if/where the catalog exposes it); echo the **reference-panel ancestry**
  actually used in the percentile output; add **sample-ancestry inference** — plink2
  projection onto the HGDP+1kGP panel + k-NN on PCs (peddy/somalier as low-effort
  fallback). Emit a 3-way coherence verdict (development vs sample vs panel) and an
  independent mismatch flag.
- **Files:** `models.py` (`ScoreInfo`), `prs_catalog.py`, `reference.py`, a new
  `ancestry.py`. Reference panel = HGDP+1kGP (`pgsc_HGDP+1kGP_v1`, already supported).
- **Acceptance:** every enriched result reports dev/sample/panel ancestry and a
  coherence verdict; a EUR-developed score applied to a non-EUR sample is flagged.

### Phase 4 — Posterior quality `Q` per (score × genome) (F20 / F9 + thesis) — *needs P1–P3*

Today only the precomputed, cross-genome (populational) `combined_quality_score`
exists. Personal prediction needs a runtime per-genome posterior.

- **Build:** `Q = C_wt · R_tech · H_hwe · K_anc · V_prior · A_size` (product, so any
  near-zero factor kills the score):
  - `C_wt` — weight-mass coverage (Phase 1); the backbone, makes the threshold scale-free.
  - `R_tech` — **within-genome** jackknife/bootstrap of matched variants (new). Keep
    the existing cross-genome `stability_score` as a *catalog prior*, don't conflate.
  - `H_hwe` — Z_HWE (theoretical null) vs Z_panel (reference panel) coherence; reuses
    existing pieces. Extreme |Z| at low coverage ⇒ artifact (A5).
  - `K_anc` — ancestry coherence from Phase 3, able to **veto** independently (§ Circularity).
  - `V_prior` — `synthetic_score/100` (carries AUROC/β/OR tiering + cohort; A6).
  - `A_size` — soft U-curve: mild oversimplification cap on tiny N, mild skepticism on
    genome-wide; mostly emergent from `C_wt`.
- **Threshold is emergent** from `C_wt · A_size` — no hardcoded 50%. A 3-SNP miss
  craters `C_wt`; a diffuse genome-wide score self-penalizes; the golden middle passes.
- **Files:** `quality.py` (new Q function alongside the existing ones), `prs.py` /
  `enrich.py`, a new jackknife util, `models.py`.
- **Acceptance:** `Q` present per result; both tiny toy scores and diffuse
  genome-wide models are down-weighted; an ancestry-mismatched score is vetoed even
  when it agrees with the rest.

### Phase 5 — Q-weighted consensus per trait (thesis, A7) — *needs P4*

- **Build:** noise-gate (drop `Q < floor`) → **Q-weighted robust median** of z across
  surviving scores + Q-weighted IQR as a confidence band + a concordance flag. Apply
  the A1 √R² calibration to the consensus **separately** (the ensemble buys a robust
  percentile/direction; A1 turns it into absolute risk).
- **Files:** `prs_catalog.py` (by-trait path), `enrich.py`, `models.py` (a trait
  consensus model).
- **Acceptance:** a trait returns one consensus percentile/direction + band + a
  per-score table, robust to the low-quality artifacts the gate didn't fully kill.

### Phase 6 — Cost prior-gate (the one retained prior-gate) — *independent*

- **Build:** classify the ~6M-variant genome-wide tier (the slow, weight-diffuse,
  "expected to fail" models) and expose a `variants_number` threshold/flag so the
  default compute-all path can skip them while an opt-in includes them. The threshold
  lives in the library; the MCP `mode` wiring that consumes it is the wrapper's job
  (out of scope here).
- **Files:** `scoring.py` / `prs_catalog.py`, `models.py`.

## Cross-cutting

- **Re-point stability:** cross-genome `stability_score` stays a *catalog prior*
  (search/ranking/demo selection — see [`demo-trait-ranking.md`](demo-trait-ranking.md));
  the new within-genome `R_tech` is the *personal* signal. Don't let one masquerade
  as the other.
- **Reference panel:** standardize on HGDP+1kGP (`pgsc_HGDP+1kGP_v1`) — it doubles as
  the K_anc reference *and* the reference-panel percentile source (scoring its EUR
  subset yields the empirical mean/SD Phase 2 needs).
- **Circularity guard:** a posterior gate that rewards agreement can manufacture
  confident consensus on a *shared* bias (e.g. many EUR-trained scores agreeing on a
  non-EUR genome). `K_anc` must veto independently — not just contribute a fractional
  term — so the Q-weighted median can't launder a population mismatch into false
  precision.
- **Safe defaults (assume a non-expert user):** every knob — input type, mode,
  reference panel, quality floor — must have a sensible default that works with zero
  configuration. When a default is derived by inference, surface what was chosen and
  bias toward the conservative/honest outcome, never the optimistic one.
- **Epistemic limit:** the 10-genome set has no phenotype labels. We can improve `Q`'s
  *construct* validity (measures the right thing in principle) but cannot measure its
  *criterion* validity (predicts real phenotype) on current data. State this in any
  Q-bearing output; acquiring genotype-phenotype pairs is the highest-value way to
  close it.

## Sequencing

```
P1 ──► P2 ─┐
       P3 ─┴─► P4 ──► P5
P6 (independent)
```

P1 first — it corrupts every downstream percentile and quality number. P2 and P3 are
independent of each other once P1 lands. P4 needs P1–P3; P5 needs P4. P6 can land
any time.

## Verification

- **Unit / wiring (network-free):** in `just-prs/tests/` — engine parity
  (`test_cross_engine.py`), new field presence on `PRSResult`/`ScoreInfo`, Q math,
  jackknife determinism.
- **Real-data regression:** `scripts/smoke_test_all_prs.py` over the 10-genome set
  before/after each phase.
- **Primary empirical gates:** (a) Phase-1 coverage recovery on a plain WGS VCF via
  reference-allele resolution (`variants_unscorable_absent` → ~0), with array VCFs
  gaining no phantom hom-ref; (b) reduced spread of same-sample/same-trait
  percentiles across genome-wide scores after P1–P4 (the F15/F20 symptom).

## Sources

- `posterior-quality-gating.md` — the design thesis and the `Q` factors.
- `prs-quality-score.md` / `demo-trait-ranking.md` — the existing quality foundation
  this drives from.
- `absolute-risk-methodology.md` — the OR / AUC-bivariate / h²-liability models.
- `just-prs-mcp/docs/competitor-research.md` — competitor survey + Alt Ideas A1–A9.
- `just-prs-mcp/docs/just-prs-pending-fixes.md` — F-findings (F9/F11/F12/F15/F19/F20/F22).
