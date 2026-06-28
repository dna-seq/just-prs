# Features backlog

Deferred UI/feature work that is scoped but intentionally not yet implemented.
Each entry records the motivation, what already exists, and the implementation
options so it can be picked up without re-discovery.

## Multi-ancestry overlay bell curves in the prs-ui results panel

**Status:** deferred (2026-06-25). Surfaced while wiring the under-the-hood PRS
signals (C_wt coverage, percentile reliability/caveat, z-score, build-mismatch)
into the web UI — those landed; this visualization did not.

**What exists today.** The Compute PRS results detail panel renders a **single**
reference bell curve (the `bell_curve` renderer in
`prs-ui/prs_ui/components/prs_section.py`, fed `population_percentiles_chart` built
in `prs_ui/mixin.py::_build_prs_results_grid`). Per-population percentiles are shown
as **metric cards** (`pct_AFR/AMR/EAS/EUR/SAS`), not as overlaid curves.

Meanwhile the core library already ships the overlay primitives in
`just_prs.viz` (Altair):

- `plot_prs_multi_ancestry(pgs_id, distributions_df, user_score=..., ancestries=...)`
  — overlaid per-ancestry bell curves with a user-score marker.
- `plot_trait_scores(..., ancestries=[...])` — trait-grouped reference curve(s) with
  per-model z-score dots, multiple color-coded population curves.

The per-population reference stats needed to draw overlays
(`reference_mean` / `reference_std` and the five `pct_*` values) are **already
available on the enriched row** after the recent wiring work — no new computation
is required, only a renderer that consumes more than one curve.

**Why deferred.** Wiring an overlay into the live UI needs one of:

1. **Extend the reflex-mui-datagrid `bell_curve` JS renderer** to accept and draw
   N curves (one per population) with independent color legends. This is the
   cleaner long-term path but is an **upstream change** to `reflex-mui-datagrid`
   (project preference: fix upstream rather than monkey-patch locally).
2. **Embed an Altair/Vega spec** produced by `just_prs.viz.plot_prs_multi_ancestry`
   directly in the detail panel (e.g. via a Vega-Lite component), bypassing the
   JS bell_curve renderer for the multi-curve case. Keeps the change inside prs-ui
   but introduces a second charting path to maintain.

**Recommended next step.** Option 1 (upstream renderer support), so the MCP
`plot_trait_panel` JSON and the prs-ui chart stay fed from the same shape. If that
stalls, fall back to Option 2 for the results panel only.

**Related:** the MCP wrapper tracks the empirical-cohort-histogram variant of this
(per-individual reference scores) as F27 in
`just-prs-mcp/docs/just-prs-pending-fixes.md`; that one additionally needs a new
`reference_individual_scores()` library accessor.

## Model-based admixture proportions (ancestry Level-3 north-star)

**Status:** deferred (2026-06-27). **Level 1 has now shipped** (branch
`sample-ancestry-l1`: `just_prs.ancestry` + `PRSCatalog.infer_ancestry` /
`assess_ancestry_coherence` + the `ancestry_model_pipeline`; validated at LOO 0.9965 on
1000G/GRCh38). **Level 2** (Privé flat-CSV ancestry *proportions* via projection + QP —
pure-numpy, permissive) is the immediate next step (Stage 2 in
`docs/sample-ancestry-methodology.md`). This backlog item is **Level 3**: *true*
model-based admixture proportions
(`EUR 0.63 / EAS 0.27 / SAS 0.10`) and per-segment local ancestry — a strictly richer
signal than a PCA approximation, but gated on non-permissive / heavy tooling.

**Why it's worth having (beyond the MVP veto).** True proportions sharpen every
ancestry-aware behaviour: graded reliability, per-individual score ranking,
interpolated reference distributions, and "no good reference" detection for deeply
admixed individuals (e.g. `AMR`, itself an admixed super-population, which a hard label
barely describes). See `sample-ancestry-methodology.md` §7.

**Engines (none is a clean permissive runtime dep):**

- **fastmixture** (`Rosemeis/fastmixture`, Santander/Refoyo-Martínez/Meisner 2024) —
  the Python-native, numpy-2, peer-reviewed engine; **~30× faster than ADMIXTURE**; has
  a **`--projection` mode** (feed a precomputed reference `P` → estimate only `Q` for a
  new genome) that exactly fits our precompute→project pattern. **Blockers:** (1)
  **GPL-3.0** — cannot be hard-imported by MIT code; (2) **no wheels** (Cython sdist →
  compiles from source, needs a C compiler, Windows-unfriendly like pgenlib); (3)
  single-sample (`n=1`) projection EM stability is unvalidated.
  *Integration if pursued:* (a) run offline in the pipeline and ship `Q` as data
  (program output is not a GPL derivative → runtime stays MIT) — best for
  reference-individual proportions; or (b) an **optional extra
  `just-prs[fastmixture]`** the user explicitly opts into (base stays MIT; the user
  assembles the GPL combination). Never `import fastmixture` from the core package.
- **ADMIXTURE** — historical reference standard but **avoid entirely**: closed-source
  binary under an **academic, non-commercial, non-redistributable** license (non-OSI;
  *worse* than GPL). Named only as fastmixture's benchmark baseline.
- **RFMix (v2)** — per-segment **local** ancestry (→ global fractions); most accurate on
  complex multi-way admixture; compiled C++, license to verify. The path to a future
  `local_ancestry` (chromosome-painting) field.
- **fastSTRUCTURE** (GPL, dated) / **Neural ADMIXTURE** (PyTorch; overkill for single
  genomes, less accurate on complex admixture per fastmixture's benchmarks) — not
  recommended.

**Recommended next step (when picked up):** start with fastmixture `--projection`
offline in the pipeline against a precomputed reference `P`, shipping `Q` as data;
validate single-sample stability before any per-genome runtime use.

## sgkit as the Apache build-time genetics toolbox (to investigate — numpy downgrade)

**Status:** deferred / to-investigate (2026-06-27). Surfaced while choosing permissive
tooling for the ancestry model build. **sgkit** (Apache-2.0; the actively-developed
successor to MIT-but-maintenance-only `scikit-allel`) ships exactly the build-time
primitives we'd otherwise reach for a PLINK binary or hand-rolled numpy:

- **`ld_prune()`** — pure-Python LD pruning → **removes the PLINK binary** from the
  reference-model build.
- **PCA** component/loading/**projection** specs, **allele frequencies**, and **`Fst()`**
  (genetic distance — a clean Apache source for the transferability signal in
  `sample-ancestry-methodology.md` §7).
- Does **not** provide: KNN classification (we add via scikit-learn/numpy), the
  shrinkage-corrected OADP projection (vendored fraposa keeps that), or admixture
  proportions (see the fastmixture entry above).

**Blocker to investigate — it downgrades numpy.** Adding sgkit to the workspace resolved
but **pinned numpy down**:

```
- numpy==2.5.0
+ numpy==2.1.3
```

The downgrade almost certainly comes from a **transitive** dependency (sgkit pulls
**dask + xarray**, and the numba/numbagg/bottleneck stack typically lags numpy support),
not sgkit's own code. **To investigate before adopting:**
- *Where* the `numpy<2.2`-ish ceiling originates (which transitive dep) and whether it's
  already lifted upstream.
- *What we lose / risk at numpy 2.1.3*: any numpy-2.2–2.5 APIs/perf we rely on, and —
  more importantly — whether the downgrade is **workspace-wide** (it would also pin
  `polars-bio`, `pgenlib`, and the rest), i.e. is this a quiet regression for the whole
  monorepo or isolatable to an optional extra / the offline pipeline env only.
- Whether sgkit can be confined to a **build-time-only / separate pipeline environment**
  (it pulls heavy dask+xarray regardless) so the numpy ceiling never touches the runtime
  package.

**Recommended next step:** don't add sgkit to the core/runtime deps. If adopted, scope it
to the offline pipeline env (or an extra) and first pin down the source + blast radius of
the numpy downgrade. For the ancestry MVP, build via the existing plink2 binary or ingest
the PGS Catalog published PCA instead.
