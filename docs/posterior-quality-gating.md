# Posterior Quality Gating (design)

A design direction for how `just-prs` decides which PGS models to trust for a
**single person**, and how to combine them. It evolves the existing quality
machinery — [`prs-quality-score.md`](prs-quality-score.md) (`synthetic_quality_score`
/ `combined_quality_score`) and [`demo-trait-ranking.md`](demo-trait-ranking.md) —
rather than replacing it. Those are where we **drive from**.

Status: design / not yet implemented. Cross-references the competitor survey and
the lettered Alt Ideas (A1–A9) in the sibling repo's
`just-prs-mcp/docs/competitor-research.md`, and the F-findings in
`just-prs-mcp/docs/just-prs-pending-fixes.md`.

## 1. Premise — prior-gate vs posterior-gate

Competitors **prior-gate**: they shrink the candidate set before touching the
genome (variant-count caps, declared ancestry, declared performance). They do this
because they implicitly assume computation is expensive.

For `just-prs` it isn't. So the default posture is **posterior-gate**: compute
*every* applicable score against *this* genome, derive a quality of each **result
vs this specific callset**, drop the noise, and combine survivors weighted by that
quality. The gate is a property of the **(score × genome) pair**, not of the
score's metadata sheet.

We already have the scaffold: `combined_quality_score` blends a metadata *prior*
(`synthetic_score`, 0.40) with three *posterior* signals (match 0.25, stability
0.15, risk 0.20). This doc fixes what those posterior signals measure and moves the
personal gate from a precomputed catalog column to a runtime per-genome quantity.

## 2. The one prior-gate we keep — and it's a cost gate

Do **not** drop prior-gating entirely. Keep exactly one, and make it about compute
budget, not quality:

- **~6M-variant genome-wide models → `extended` mode only.** These are the slow
  tier (they dominate the wall-clock of a full trait panel) and the tier we expect
  to fail on weight-mass coverage anyway. Essentials mode computes the entire
  catalog *except* these; extended mode includes them for callers who insist.
- This is a budget decision, not a verdict on the score. It happens to align: the
  expensive tier is also the suspect tier, so we lose little by deferring it.

Everything else: compute all, judge after.

## 3. The reframe — populational reliability ≠ phenotypic reliability

PGS models are **populational instruments by design** — fit to rank a cohort, not
to predict one person. Using them for personal prediction is an assumption shift,
and our quality signals must shift with it.

The existing `combined_quality_score` posterior signals are populational:

- `stability_score` = cross-**genome** percentile std over the 10-genome batch.
  It even *rewards low* between-genome spread (guarded by a spread-factor against
  total collapse). For a population that reads as "robust"; for personal prediction
  low between-person spread can equally mean "does not discriminate."
- `risk_concordance` = a bell curve on how risk estimates spread *across genomes*.
  Again a statement about the population, not about the reliability of one estimate.

Two stabilities are being conflated:

| | Measures | Good for | Current metric |
|---|---|---|---|
| **Within-individual reliability** | does *my* Z survive caller/coverage/imputation noise; does the matched portion carry the score's signal | personal prediction | **not measured** |
| **Between-individual discrimination** | does the model rank a crowd | population studies / catalog priors | `stability_score`, `risk_concordance` |

Decision: **keep the populational scores as catalog priors** (search, ranking, demo
selection — they're good at that; leave them). **Add a per-genome posterior `Q`**
built from within-individual signals, computed at runtime against the user's genome.

The question Q must answer: *what holds value for this person's phenotypic
prediction*, not *how reproducible is this model over a population*.

## 4. Posterior quality `Q(score, genome) ∈ [0,1]`

A product of near-independent factors, so any one near zero kills the score:

```
Q = C_wt · R_tech · H_hwe · K_anc · V_prior · A_size
```

- **C_wt — weight-mass coverage.** Fraction of Σ|βᵢ| carried by *matched* variants,
  not fraction of variant count. This is the backbone and replaces the current
  `match_factor` (raw match-rate) for the *personal* gate. It makes the threshold
  scale-free (§5).
- **R_tech — within-genome technical reliability.** How much *this person's* Z moves
  under perturbation. Cross-caller Z spread when multiple callsets exist (A4); else
  a single-callset jackknife/bootstrap over matched variants. This is the *within*-
  individual reading of A4's "large scores are reproducible" — a personal signal,
  not a populational one.
- **H_hwe — HWE-null coherence.** Is the raw score sane against the analytic null
  `E=Σ2pβ`, `Var=Σ2p(1−p)β²` (A2)? Extreme |Z| at low coverage = artifact, down-
  weight hard (A5).
- **K_anc — ancestry coherence.** Agreement among development ancestry, sample
  ancestry, and reference-panel ancestry (F19). **Independent veto** — see §7.
- **V_prior — validity prior.** The metadata discrimination/cohort signal we already
  compute: reuse `synthetic_score`/100 (or its `discrimination · penalty` part) so
  AUROC/C-index/β/OR tiering and cohort size still count (A1/A6). This is the only
  *prior* term inside Q; it carries predictive-validity information the genome alone
  can't supply.
- **A_size — size adequacy (U-curve).** Mild oversimplification cap on tiny N (even a
  fully-matched 3-SNP score is a weak instrument) and a mild skepticism prior on
  genome-wide N. Largely *emergent* from C_wt (§5), so keep it light — a soft
  multiplier, not a hard cut.

## 5. The floating threshold is emergent, not hardcoded

We do **not** set "50% for big scores, less for genome-wide." The threshold falls
out of `C_wt · A_size`:

- **Tiny scores (≤ ~5–20 SNPs):** each variant carries large weight mass, so a
  single miss craters C_wt. "Non-100% match for a 3-SNP score is a flop" is
  automatic — no special case. A_size adds a mild cap so even a perfect tiny score
  isn't over-trusted.
- **Golden middle (clumped, ~hundreds–tens of thousands):** full Q reachable at
  moderate weight-mass coverage; this is where models concentrated on real signal
  live.
- **Genome-wide (≫100k):** not banned (when in extended mode), but must *earn* it.
  A diffuse score where matched variants carry little of Σ|β| self-penalizes via
  C_wt. This reconciles the "genome-wide is fishy" instinct with A4's finding that
  big scores are numerically *stable*: a stable-but-diffuse model is technically
  reproducible (high R_tech) yet low-value (low C_wt) → down-weighted, not excluded.
  Reproducibility and meaningfulness are different axes; Q separates them.

## 6. Aggregation — Q-weighted robust median

Per trait (EFO/MONDO-grouped, as in demo-trait-ranking):

1. **Noise gate:** drop scores with `Q < floor`.
2. **Combine survivors** by **Q-weighted median** of Z (robust to the artifacts the
   gate didn't fully kill). Report the Q-weighted IQR as a confidence band and a
   concordance flag across scores (A7).
3. **Calibrate to absolute risk separately**, not inside the ensemble: apply the
   √R² scaling `E[trait] = mean + SD·√R²·Z` (A1) per score or to the consensus with
   a representative R². The ensemble buys a robust *percentile/direction*; A1 turns
   it into risk. Mixing the two would average incommensurable scales.

This is the line the doc's opening claims: differ from the landscape in **quality**
— a per-genome, posterior-weighted consensus, not a single prior-filtered score.

## 7. The circularity trap (important)

A posterior gate that rewards *agreement* can manufacture confident consensus on a
*shared bias* — e.g. a dozen EUR-trained scores all agreeing on a non-EUR genome.
Agreement ≠ truth. Therefore **K_anc must be able to veto the whole ensemble
independently**, not just contribute a fractional term to a weighted mean.
Otherwise the Q-weighted median launders a population mismatch into false
precision. This is the one place where a multiplicative/veto structure is
non-negotiable.

## 8. Honest epistemic limit

We can improve Q's **construct validity** (it measures the right thing in
principle) but cannot, on current data, measure its **criterion validity** (does it
predict actual phenotype). The 10-genome set has no phenotype labels, so the only
empirical anchor available is cross-genome behavior — precisely the populational
signal this reframe moves past. Our within-individual factors (C_wt, R_tech, H_hwe)
are therefore *principled proxies*, validated by coherence and ablation, not by
held-out phenotype. State this in any output that leans on Q. Acquiring even a
handful of genotype-phenotype pairs would be the highest-value way to close this.

## 9. Concrete next steps (driving from existing code)

- **Re-point stability:** add a within-genome jackknife/cross-caller `R_tech`
  alongside (not replacing) the cross-genome `stability_score`. Keep the latter as a
  catalog prior; surface the former in the per-genome result.
- **Add weight-mass coverage** `C_wt` next to `match_rate` in the PRS result and in
  `pgs_quality_scores.parquet` (`scripts/build_pgs_quality_scores.py`). Requires
  per-variant |β| at match time — already available in the scoring file.
- **Compute Q at runtime** in the compute path (per score × genome), distinct from
  the precomputed catalog `combined_quality_score`. Expose it on the trait report.
- **Implement the cost prior-gate:** route ~6M-variant scores to `extended` mode
  (ties into the MCP `mode` axis; threshold on `variants_number`).
- **Empirically shape A_size** from the 10-genome set: where does C_wt-vs-N bend?
- **Wire the K_anc veto** once development/panel ancestry is surfaced (F19).

## 10. To research (birdflight — not yet scoped)

### 10.1 K_anc — sample-ancestry resolution, open options

K_anc needs *three* ancestry reads to compare: (a) the score's **development**
ancestry, (b) the percentile **reference-panel** ancestry — both metadata, surfaced
under F19 — and (c) the **sample's** inferred genetic ancestry, which we don't yet
compute. (c) is the missing capability. Landscape scan:

- **impute.me** had an `ethnicity` module but the project went **proprietary**
  (moved to mynucleus.com; repo README stripped, license unclear). Treat as **not
  reusable** — do not depend on it. Its method (PCA projection onto a reference
  panel) is standard and fully reproduced by the live open tools below, so nothing
  is actually lost.
- **peddy** (MIT, pure-Python) — samples ~25k sites from a VCF, PCA on the 1000G
  2504, SVM → continental superpopulation. Lightweight, easy to vendor/integrate.
- **somalier** (MIT, single binary) — genome "sketches" projected onto 1000G PCA +
  labels; very fast on WGS VCF. Strong fit for our compute path.
- **FRAPOSA** (open Python) — online-PCA / out-of-sample projection onto a chosen
  reference (1000G / HGDP); the rigorous option if we ever want to own the PCA.
- **Self-hosted path (likely the right one for us):** project the sample onto a
  reference panel's eigenvectors with **plink2** (`--score` + `--read-freq` against
  precomputed PCA weights) and assign continental superpop by **k-NN** on the PC
  coordinates. We already depend on plink2/pgenlib, so this adds little. Mature WGS
  practice favours this over an off-the-shelf classifier.
- **Reference panel — prefer HGDP+1kGP over plain 1000G.** PGS Catalog distributes
  it first-party as **`pgsc_HGDP+1kGP_v1`** (gnomAD HGDP+1kGP callset; ~667 unrelated
  EUR after a KING kinship cutoff). Same panel doubles as the **reference-panel
  percentile** source (scoring its genomes gives the empirical mean/SD that F12
  needs) and as the K_anc reference. Other open panels: 1000G phase 3, HGDP.
- Heavier/classic fallbacks: EIGENSOFT `smartpca` projection, supervised ADMIXTURE,
  NCBI GRAF-pop.

Strategic lean (un-clinched): **self-hosted plink2 projection + k-NN onto
HGDP+1kGP** (one panel serving K_anc *and* the F12 reference percentile), with
**peddy/somalier** as the low-effort fallback (somalier is fast and proven in the
QC/relatedness role; peddy embeds in pure Python). Output is a continental superpop
(± admixture fractions) feeding the 3-way coherence check and the independent veto
(§7). Continental resolution is almost certainly enough (PGS dev ancestry is itself
continental); sub-continental is likely over-engineering.

> Adjacent confirmations worth carrying into H_hwe / F12: scoring the panel's EUR
> genomes yields an empirical **Z_panel**, and cross-checking the analytic
> **Z_HWE** (A2) against it both validates the null and supplies the reference
> mean/SD F12 lacks. Per-sample **jackknife** (R_tech) and **cross-caller** spread
> (A4) against the same panel are established techniques, not speculative.

### 10.2 Fifth Q signal — personal absolute-risk discrimination

Candidate additional posterior term for Q (§4): how far *this* person's calibrated
absolute risk sits from the population average — a personal-discrimination signal,
as opposed to the populational `risk_concordance` catalog prior. **Blocked on a
solid z→absolute-risk path (F12 / A1 √R² calibration)**; until that lands this can't
be computed honestly. Park as a 7th factor in Q, behind F12. No detail this run.
