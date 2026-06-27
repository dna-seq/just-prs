# Sample (individual) genetic-ancestry inference — design (in `just-prs`)

**Status:** Stage 1 (Level-1 veto) **implemented** (2026-06-27, branch `sample-ancestry-l1`).
Module `just_prs.ancestry` (vendored FRAPOSA OADP, MIT), `PRSCatalog.infer_ancestry` /
`assess_ancestry_coherence`, `prs ancestry infer|check`, HF sync, and the Dagster
`ancestry_model_pipeline` (plink2 LD-prune + numpy SVD, `check_ancestry_model_valid`).
Validated: real 1000G/GRCh38 build → **leave-one-out super-pop accuracy 0.9965** over 53,526
pruned variants / 2,575 unrelated samples; real WGS samples (anton/livia/newton) → EUR conf
1.00 at ~99% coverage (after the variant-only hom-ref-absent fix). **Level-2 ancestry
*proportions* also shipped** via `estimate_proportions` — a pure-numpy simplex-constrained
least-squares (`mixture_method="pca_nnls"`) of the sample's PC vector onto our own 1000G
super-population centroids, so proportions share the EUR/EAS/AFR/AMR/SAS vocabulary of the
percentile panel (e.g. newton → EUR 0.976 / AMR 0.022 / EAS 0.002). This realises Level 2 on
the coherent in-house model rather than the Privé reference — the **Privé 21-group ingestion is
deferred** as a finer-resolution refinement (it needs a 1.7 GB GPL-data download, a
reverse-engineered shrinkage `correction` vector, a QP solver, and GRCh37→38 liftover, and its
groups roll up to the same continental level anyway).

**Decision: keep it coupled inside `just-prs`** (`just_prs.ancestry`). An earlier draft scoped a
standalone `just-ancestry` library with input/output interface contracts — that is **dropped for
now** to avoid premature abstraction. A larger extraction is envisioned *later* and separately:
a single I/O + annotation library covering *all input types (VCF/gVCF/array/TSV) → polars-bio
frame → liftover → ref-call padding → normalization → meta-annotation (incl. global ancestry)*.
Until then, ancestry inference lives in this package with no interface ceremony. **Close the loop
first, grow later.**

**Origin:** the missing third leg of the ancestry-coherence story (tracked as F19 in
`just-prs-mcp`). The *development* ancestry per score (`dev_ancestry_broad` /
`dev_ancestry_distribution`, `PRSCatalog.development_ancestry()`) shipped on
`f19-f21-dev-ancestry`; the *reference-panel* ancestry already ships on `PercentileResult`. This
adds the **sample's own** genetic ancestry and the coherence check that needs all three.

---

## 1. Goal & scope

**Goal.** From a user's normalized genotypes (WGS VCF, gVCF, or consumer array, already in the
`just_prs` normalized parquet schema), infer **genetic ancestry** — a super-population label,
membership probabilities, and continuous PC coordinates — in the **same PCA space and reference
populations** our percentile distributions use, so it is directly comparable to the score's
development ancestry and the percentile panel.

**This loop (MVP, in scope):**
- A reference-PCA model, precomputed once, published to HF, pulled at runtime.
- A pure-Python, **numpy-2-native** runtime projector + classifier → `AncestryInference`
  (Level 1 of §6; permissive deps only).
- The **coherence verdict**: compare sample vs percentile-panel vs score-development ancestry and
  emit a plain-English reliability note.
- Real-data validation (leave-one-out on the reference panel).

**Near-term, still in-package (Level 2):** a PCA-derived fractional `mixture` + fuzzy reliability
(§6/§7). Cheap, permissive, no new heavy dep — folded in once Level 1 lands.

**Deferred to the backlog (`docs/features_backlog.md`), NOT this loop:**
- **True model-based admixture proportions** (Level 3 north-star) via **fastmixture** (GPL) /
  **sgkit** (Apache build-time toolbox). See the backlog entries — including the numpy-downgrade
  caveat sgkit introduces.
- **PGS PC-adjustment** (the `Z_norm` continuous re-normalization of the *score value*) — a
  separate, larger feature; we persist PC coords so it can be built later.
- Relatedness / sex / sample-swap checks — not our use case.

---

## 2. Method & tool choice (permissive, numpy-2)

The engine is a **reference PCA + shrinkage-corrected projection + a classifier**.

- **Backbone = FRAPOSA's method** (PGS Catalog's own tool; OADP = online-SVD + Procrustes
  shrinkage). Chosen for **vocabulary coherence**: its calls land in the same EUR/EAS/AFR/AMR/SAS
  vocabulary as our percentile panels and `dev_ancestry_broad`, so all three legs compare
  apples-to-apples. peddy (discontinued) and somalier (Nim binary, Windows-unportable) would each
  impose a second, incoherent ancestry vocabulary — rejected.
- **Vendor, don't depend.** `fraposa-pgsc` is **MIT (clean — no license asterisk)**; the only
  blocker is its `numpy<2.0` / `pandas<2.0` pins, which collide with the workspace's numpy 2.x.
  So port its ~150–200 lines of numpy projection math (`standardize`, `svd_online`,
  `procrustes_diffdim`, `oadp`, `pca_stu`) into `just_prs.ancestry` with attribution + the MIT
  notice. `pca_stu` takes a genotype matrix, so runtime never needs PLINK files or a subprocess.
- **Permissive runtime stack:** numpy 2.x + `scikit-learn` (BSD) for the classifier (a numpy-only
  k-NN keeps it to numpy alone). No GPL, no compiled binary, no PLINK at inference → installable
  everywhere including Windows.
- **PCA tools give a label + probabilities, not fractions.** PCA is a variance transform; admixed
  individuals sit *between* clusters, so the classifier posterior is *uncertainty*, not an
  admixture proportion. True fractions need model-based tools — deferred (§7, backlog).

---

## 3. Architecture — precompute → publish → pull

The established pattern (mirrors `reference_allele_universe`, `chip_coverage`, reference
distributions): a heavy precompute artifact published to HF, a light pure-Python runtime consumer.

**Precompute (Dagster, `prs-pipeline`)** — `ancestry_model_pipeline`:

| Asset | Group | What it does |
|---|---|---|
| `pgsc_reference_panel` (SourceAsset) | `external` | PGS Catalog processed panels at EBI FTP (`pgsc_1000G_v1`, `pgsc_HGDP+1kGP_v1`; both GRCh37+GRCh38); URL in metadata |
| `ancestry_pca_model` | `compute` | Build the PCA per panel (GRCh38); persist artifact (§4). Metadata logs panel, build, n_pruned_variants, n_pcs, **leave-one-out accuracy** |
| `hf_ancestry_model` | `upload` | Push artifact to `just-dna-seq/prs-percentiles` (`data/ancestry/`) |

- Build path for the MVP: **prefer ingesting the PGS Catalog published reference PCA** (identical
  coordinates to the calculator, no LD-prune step). If recomputing, LD-prune with the existing
  auto-downloaded **plink2** binary (build-time only). `sgkit.ld_prune()` is the Apache-licensed
  alternative — backlogged because it downgrades numpy (see backlog).
- Asset check `check_ancestry_model_valid` (ERROR): pruned sites non-empty & unique, loadings
  shape matches `n_pcs`, every reference IID labelled, LOO accuracy ≥ threshold, no NaN/inf.
- Robustness policy: mtime staleness vs source panel, coverage metadata, parquet-readability guard.

**Both panels, GRCh38 only:** 1000G (5 super-pops) **and** HGDP+1kGP (finer pops), each built in
**GRCh38 only** — a native GRCh37 model is redundant because the hom-ref-absent imputation that
makes projection work is inline at the model's GRCh38 pruned sites, so a GRCh37 sample is simply
**lifted to GRCh38** at inference (`infer_ancestry` canonicalizes to GRCh38 via `liftover`). This
matches the project-wide lift-to-GRCh38 direction and halves the build/maintenance. **Default =
1000G** (its 5 super-pops match the percentile + dev-ancestry vocabulary, so the verdict is
coherent by construction); HGDP+1kGP offered for finer resolution, rolled up to the broad set for
the comparison. Artifact filenames are panel- and
build-aware (e.g. `ancestry_model_1000g_GRCh38.parquet`), matching the build-aware naming used by
`reference_allele_universe[_<build>].parquet`.

**Runtime (`just-prs` core).** `just_prs.hf.pull_ancestry_model(panel, build)` (lazy pull + cache
under `<cache>/ancestry/`, same retry/dedup as `pull_reference_distributions`); `just_prs.ancestry`
(vendored projector + classifier); `PRSCatalog.infer_ancestry(...)`.

---

## 4. Reference model artifact

Per `(panel, build)`, a small self-describing bundle (≪ 50 MB; the reference genomes never ship):

- `ancestry_model_<panel>_<build>.parquet` — pruned sites: `chrom, pos, ref, alt, effect_allele,
  mean, std` + `loading_pc1…pc10`.
- `ancestry_refpcs_<panel>_<build>.parquet` — reference individuals in PC space:
  `iid, superpop, population, pc1…pc10` (classifier training set + plot backdrop).
- `…_meta.json` — `n_pcs`, singular values (for OADP shrinkage), classifier spec, n_reference,
  LOO accuracy, source panel version/URL, build, label vocabulary, fine→broad mapping.

The k-NN/RF classifier is reconstructed from `ancestry_refpcs` at load (not a pickled,
sklearn-version-frozen blob).

---

## 5. Runtime API

```python
class AncestryInference(BaseModel):       # just_prs.models
    panel: str
    genome_build: str                     # build assumed/projected in
    superpopulation: str                  # EUR/EAS/AFR/AMR/SAS or "UNKNOWN"   [Level 1, MVP]
    probabilities: dict[str, float]       # classifier posterior (NOT genome fractions)
    mixture: dict[str, float] | None = None    # PCA-derived fractions, sum ~1  [Level 2]
    pc_coords: list[float]
    n_variants_used: int
    n_variants_model: int
    coverage: float
    confidence: float
    fine_population: str | None = None

def PRSCatalog.infer_ancestry(
    self, genotypes_path=None, *, genotypes_lf=None,
    panel="1000g", sample_build=None,
) -> AncestryInference: ...
```

- **Genome build is an implicit input → GRCh38.** Genotypes from the `polars-bio` VCF reader carry
  no build tag, so absent a detected (`detect_genome_build`) or declared build, **assume GRCh38**
  (the modern default our models target) and record it on the result. If the resolved build ≠ the
  model's build, lift via `just_prs.liftover` or fail loudly — never silently match across builds.
- **Coverage honesty (the F15/F9 theme).** Below a coverage floor, return `"UNKNOWN"` + low
  confidence rather than a confident wrong call; surface `n_variants_used / n_variants_model` so
  "couldn't tell" ≠ "genuinely intermediate".

---

## 6. The coherence verdict (this loop's payoff)

With sample ancestry available, all three legs are in the broad-super-pop vocabulary:

| Leg | Source | Status |
|---|---|---|
| **sample** | `AncestryInference.superpopulation` | **this loop (new)** |
| **percentile panel** | `PercentileResult.reference_panel_ancestry` | ships |
| **score development** | `dev_ancestry_broad` / `dev_ancestry_distribution` | ships (`f19-f21-dev-ancestry`) |

```python
def assess_ancestry_coherence(
    sample_superpop, percentile_panel_ancestry, dev_ancestry_distribution,
) -> AncestryCoherence: ...   # level ∈ {coherent, panel_mismatch, dev_mismatch, both, unknown}
```

- Needs a small curated **PGS broad-label ↔ 1000G super-pop** map (`European→EUR`, `East Asian→EAS`,
  `Hispanic or Latin American→AMR`, `South Asian→SAS`, admixed/`NR`→ambiguous, …), kept beside the
  chip/ontology maps.
- `dev_mismatch` keys on the sample super-pop's *fraction* in `dev_ancestry_distribution` (not just
  the headline), so a multi-ancestry score isn't falsely flagged.
- **Plain-English, citizen-scientist output** (per the project's explain-in-plain-language rule):
  e.g. *"This score was developed mainly in East Asian samples; your inferred ancestry is European,
  so its percentile may be less accurate for you."* Advisory (warning + confidence downgrade), not
  a hard block — consistent with how coverage reliability is surfaced.

---

## 7. Capability ladder (what's in this loop vs the backlog)

The verdict is the floor; richer ancestry-aware behaviour climbs a ladder. The contract is
forward-compatible: the same `mixture` field carries the upgrade, only its provenance/accuracy
changes.

| Level | Capability | Where |
|---|---|---|
| **1 — Veto (crisp)** | Hard super-pop label + binary coherence verdict | **This loop (MVP).** Pure-Python, permissive. |
| **2 — Tiered + fuzzy** | PCA-derived fractional `mixture` + **fuzzy-logic** graded reliability | **Near-term, in-package.** Pure-numpy, permissive. |
| **3 — True admixture / local ancestry** | Genome-fraction proportions (fastmixture) / per-segment local ancestry (RFMix) | **Backlog** (`features_backlog.md`). GPL/heavy → offline or opt-in extra. |

### Mixed / admixed populations — what fractions buy (motivates Level 2)

PCA + classifier gives a hard label + posterior; **true fractions** (`EUR 0.63 / EAS 0.27 / SAS
0.10`) need model-based tools (backlog). But the "linear combination" property of PCA lets us
**approximate** fractions cheaply at Level 2: express the individual's PC vector as a non-negative,
sum-to-one combination of reference-population centroids (NNLS / barycentric) — pure-numpy, reusing
the same PCA. `AMR` (Admixed American) is itself an admixed super-pop, so a bare "AMR" label tells a
score-matcher almost nothing — the sharpest motivation for fractions.

What a fractional vector enables beyond a yes/no veto:
1. **Graded reliability** — concordance (cosine/overlap) of the individual's fractions vs the
   score's dev-ancestry distribution → a continuous down-weight, not a cliff.
2. **Per-individual score ranking** — order a trait's candidate scores by concordance with *this*
   person; ancestry becomes a personalized selector. (Highest-value use.)
3. **Interpolated reference distribution** — blend per-pop distributions by the person's fractions
   for percentile placement instead of snapping to one bucket.
4. **"No good reference" detection** — if no single fraction dominates, flag that even the blend is
   uncertain and widen the band (the case hard labels handle worst).
5. **Transferability direction** — PC distance / `Fst` from the dev population → a qualitative
   expected-attenuation note (direction + coarse bucket, never false precision).

### The fuzzy layer (Level 2), concretely

Turns continuous, uncertain fractions into a graded, explainable reliability verdict and *subsumes*
the Level-1 veto (the veto is a crisp slice). Fuzzy membership over fractions + confidence
(`predominantly-X` ≳0.8, `substantially-X` ≳0.5, `strongly-admixed`, `coverage∈{low,ok,high}`); a
Mamdani rule base combining sample profile × score-dev distribution × coverage; defuzzify to a
`[0,1]` reliability multiplier + a plain-English label. Triangular/trapezoidal membership + min/max
inference is a few dozen lines of numpy — **implement dependency-free**; `scikit-fuzzy` optional.

---

## 8. Validation (real data + ground truth)

- **Leave-one-out** on each reference panel: hold out each individual, project, classify, compare to
  its label → per-panel accuracy (expect ≈0.97+ broad). Asserted in `check_ancestry_model_valid` + a
  test.
- **Coverage stress:** down-sample to GSA-array density and to a sparse callset → graceful confidence
  decay and `UNKNOWN` below the floor.
- **Cross-build:** a GRCh37 sample is lifted to GRCh38 and classifies consistently with the same
  sample in native GRCh38 (no native GRCh37 model exists by design).

---

## 9. Dependencies & portability

- **Runtime:** numpy (2.x) + optional `scikit-learn` (BSD); numpy-only k-NN keeps it to numpy alone.
  No GPL, no binary, no PLINK → Windows-installable. Vendored fraposa math is **MIT**.
- **Build-time (offline):** the existing plink2 binary for LD-prune, *or* ingest the PGS Catalog
  published PCA (preferred). `sgkit` (Apache) is the cleaner LD-prune/PCA/`Fst` source but is
  **backlogged** because adding it **downgrades numpy** (see backlog).

---

## 10. Phasing

Per the branch-isolation rule, on its own branch after `f19-f21-dev-ancestry` commits:
- **A** — `just_prs.ancestry`: vendored projector + classifier + LOO test on a tiny fixture.
- **B** — precompute asset + HF sync + asset_check (1000G GRCh38 first; ingest published PCA).
- **C** — `PRSCatalog.infer_ancestry` + `AncestryInference` + GRCh38-implicit build + coverage honesty.
- **D** — `assess_ancestry_coherence` verdict + label map + plain-English text. **(Loop closed here.)**
- **E** — Level 2 PCA-NNLS fuzzy mixture (near-term, in-package).
- **F** — second panel (HGDP+1kGP) + second build; cross-build round-trip test.
- Surfacing (MCP `TraitScoreRow` / prs-ui badge) — separate, optional.

---

## 11. Open questions

1. **Ingest the published PCA vs rebuild** — strongly prefer ingesting the PGS Catalog reference PCA
   (identical coordinates, no PLINK build). Confirm the published bundle exposes loadings +
   standardization params parsably.
2. **Classifier default** — numpy-only k-NN (dependency-free) vs RandomForest (matches the calculator;
   needs scikit-learn). Lean k-NN-default + RF-optional.
3. **Coverage floor** for `UNKNOWN` — calibrate from the coverage-stress validation.
4. **HGDP+1kGP fine→broad rollup** — adopt the calculator's grouping for comparability.

---

## 12. Sources

- FRAPOSA — <https://github.com/PGScatalog/fraposa_pgsc>
- PGS Catalog calculator, genetic ancestry — <https://pgsc-calc.readthedocs.io/en/latest/explanation/geneticancestry.html>
- PGS Catalog reference resources — <https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/> · <https://pgsc-calc.readthedocs.io/en/latest/how-to/database.html>
- PCA vs admixture proportions — <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0040115>
- peddy — <https://github.com/brentp/peddy> · somalier (MIT) — <https://github.com/brentp/somalier>

> Backlogged engines (sgkit, fastmixture, ADMIXTURE/RFMix) and their licenses are catalogued in
> `docs/features_backlog.md`.
