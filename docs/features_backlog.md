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
