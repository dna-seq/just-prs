# Array PRS against the HGDP+1kGP (`hgdp_1kg`) panel (code wired; run the builds to enable)

**Status: code 100% panel-parameterized, data not built yet.** Every layer of the
consumer-array scoring path already accepts a `panel` argument — there is **no feature/code
work** required to score arrays against `hgdp_1kg`. What's missing is two long-pole
**data builds** (both multi-hour reference-panel compute) plus their HF publish. Until they
run, `compute_array_prs(..., panel="hgdp_1kg")` has no reference distributions to produce a
percentile and no LD-proxy table to recover off-chip variants.

This mirrors the [GRCh37 universe runbook](grch37-universe-build.md): wired interface,
deferred data.

## Why it's deferred, not abandoned

The interface is already panel-agnostic and in place:

- `array_scoring.compute_array_prs(panel="1000g")` — `panel` threads through to LD-proxy
  resolution, restoration, and percentile lookup; nothing in the array path is hardcoded to
  `1000g`.
- `PRSCatalog.percentile(panel=)` / `reference_distributions(panel=)` /
  `reference_data_status(panel=)` — all route by panel to `{panel}_distributions.parquet`.
- `build_ld_proxy_batch(panel=)` and the `ld_proxy_table` asset accept any panel.
- `reference.compute_reference_prs_batch(panel=)` writes `{panel}_distributions.parquet`.
- Ancestry already canonicalizes the HGDP superpops to the 1000G-5 (`to_canonical_superpops`:
  HGDP `CSA→SAS`, `MID` dropped), so percentile/ancestry lookups work for `hgdp_1kg`.
- The HGDP+1kGP panel itself (`pgsc_HGDP+1kGP_v1`, GRCh37 + GRCh38 `.pgen/.pvar.zst/.psam`)
  is already cached / downloadable via `download_reference_panel(panel="hgdp_1kg")`.

Panel-independent layers (`chip_coverage`, `reference_allele_universe`) need nothing
hgdp-specific — they key on genome build, not reference panel.

So the only gap is **data**: the per-panel reference distributions and the per-panel LD-proxy
table. Both are produced by running existing pipelines with `--panel hgdp_1kg`.

## Work to enable (run existing pipelines — no code changes)

1. **Reference distributions for `hgdp_1kg`** (the percentile bell curves — the bigger run,
   full reference scoring of ~5,300 PGS against the HGDP panel, hours):
   ```
   prs reference score-batch --panel hgdp_1kg            # or: uv run pipeline run --panel hgdp_1kg
   ```
   Produces `percentiles/hgdp_1kg_distributions.parquet` (+ `_quality` / `_distribution_quality_issues`),
   then publish to `just-dna-seq/prs-percentiles` (the pipeline's upload asset, or
   `hf.push_reference_distributions(..., panel="hgdp_1kg")`).

2. **LD-proxy table for `hgdp_1kg`** (single deduplicated table, gsa_v3 × GRCh38):
   ```
   systemd-run --user --scope -p MemoryMax=30G \
     uv run pipeline ld-proxy --panel hgdp_1kg --full-catalog --headless --memory-limit-gb 24
   ```
   The CLI has **no** `--chip`/`--build` flags — `_LD_PROXY_CHIP_BUILD_COMBOS`
   (`prs-pipeline/src/prs_pipeline/assets.py`) is the policy constant and defaults to
   `[("gsa_v3", "GRCh38")]`. The build is the per-PGS resumable batch → streaming dedup-merge
   → single `hgdp_1kg_ld_proxy_gsa_v3_GRCh38.parquet`, auto-published to HF.

3. **Smoke-test** end-to-end:
   ```python
   from just_prs.array_scoring import compute_array_prs
   r = compute_array_prs(array_path, scoring_file, panel="hgdp_1kg", ld_proxy=True)
   # expect a percentile from hgdp_1kg distributions and n_proxied > 0 on off-chip-heavy scores
   ```

## Lower-priority follow-ons

- **GRCh37 LD-proxy** (any panel): blocked on build-matched GRCh37 typed positions; add
  `("gsa_v3", "GRCh37")` to `_LD_PROXY_CHIP_BUILD_COMBOS` only once those exist. See the
  [GRCh37 universe runbook](grch37-universe-build.md).
- Run (2) for additional chips as new chip manifests are added to `chip_coverage.CHIPS`.
