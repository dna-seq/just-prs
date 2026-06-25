# GRCh37 reference-allele universe (code wired; run the overnight build to publish)

**Status: code wired, data not built yet.** The universe lineage is now build-parameterized
end-to-end (filename, HF push/pull, pipeline assets, build script) and the GSA **A1
(GRCh37)** manifest is wired, so `chip_typed_positions(Chip.GSA_V3, build="GRCh37")` and the
two restoration gates (`prs._normalize_restoration_scope`, `array_scoring._resolve_array_restoration`)
**unlock automatically as soon as a GRCh37 universe parquet is published**. Until then they
still degrade cleanly to a no-op. The remaining step is the long-pole **data build**: run the
command below to download the ~5,385 `hmPOS_GRCh37` scoring files, build, validate, and
publish `reference_allele_universe_GRCh37.parquet`. It mirrors the GRCh38 build (a full
night) — run it the same way.

The published GRCh38 universe keeps its historical unsuffixed filename
(`reference_allele_universe.parquet`); GRCh37 (and any future build) is `_<build>`-suffixed
(`reference_allele_universe_GRCh37.parquet`) via `hf.reference_allele_universe_filename(build)`.
The pipeline build is selected by `PRS_PIPELINE_GENOME_BUILD` (set by the build script's
`--genome-build`).

## Goal

1. Build + publish `reference_allele_universe` for **GRCh37** (so the runtime can fill
   missing reference alleles for GRCh37 inputs).
2. Add the GSA **A1** (GRCh37) typed-position manifest so `Chip.GSA_V3` has a GRCh37
   typed-position set.
3. Activate array chip restoration for real GRCh37 arrays (the Part-C wiring stops
   no-opping once 1+2 exist).

## Why it's deferred, not abandoned

The interface is build-agnostic and already in place: `reference_restoration` scope,
`_apply_reference_resolution`, `chip_typed_positions(chip, build=...)`, the
`REFERENCE_FASTA`/`REFERENCE_FASTA_CHR1_LENGTH` registry (GRCh37 entry exists), and the
1000G panel (ships GRCh37 `.pgen/.pvar`). The only missing pieces are GRCh37 *data*
(scoring parquets + universe) and the A1 manifest — all long unattended compute.

## Work-through

1. **Build-parameterize the universe lineage** (drop hardcoded `"GRCh38"`):
   - `prs-pipeline/assets.py`: `reference_fasta` (build arg), `_ref_resolution_targets`
     (glob `*_hmPOS_{build}.parquet`), `reference_allele_universe` (genome_build), and
     the per-chromosome union. Use `download_reference_fasta(build)` /
     `reference_fasta_path(build)` (registry already has GRCh37).
   - Filename carries the build: `reference_allele_universe_{build}.parquet`. Make
     `hf.push_/pull_reference_allele_universe` and `PRSCatalog._reference_universe_path`
     **build-aware** (today they assume the unsuffixed GRCh38 file — keep that as the
     GRCh38 name or migrate both to suffixed names with a back-compat read).
2. **GRCh37 scoring parquets must exist first (the long pole):** download the
   `hmPOS_GRCh37` harmonized scoring files (~5,385) via
   `bulk_download_scoring_files(..., genome_build="GRCh37")` → `scoring_files_parquet`
   (build-parameterized). Expect another download+convert night (~the GRCh38 cost).
3. **Tiers:** GRCh37 panel `.pvar` resolves via `_find_reference_panel_file(ref_dir,
   "GRCh37", ".pvar.zst")` (panel ships both builds). FASTA tier: GRCh37 primary assembly
   via the registry; the `REFERENCE_FASTA_CHR1_LENGTH["GRCh37"]` guard already exists.
4. **Build + validate + push** the GRCh37 universe (same validation: `ref ∈ ACGTN`, no
   dup `(chrom,pos)`, panel/fasta/unresolved split, per-chrom streamed union, DuckDB cap).
5. **GSA A1 manifest:** add the A1 (GRCh37) manifest URL to the `gsa_v3` chip entry (or a
   build→url map) and have `chip_typed_positions(Chip.GSA_V3, build="GRCh37")` fetch A1.
   Drop the `NotImplementedError` for GRCh37 once wired.
6. **Activate + validate arrays:** with the GRCh37 universe + A1 positions present,
   `_resolve_array_restoration` stops no-opping for GRCh37 arrays. Validate on a real
   23andMe/AncestryDNA file (OFF vs ON, like the newton WGS test) and confirm hom-ref
   restoration is confined to chip-typed positions (off-chip absent stays unscorable).

## Overnight runnability

Extend `scripts/build_reference_allele_universe.py --genome-build GRCh37 --push` to also
trigger the GRCh37 scoring download/convert (or add a sibling `scripts/build_grch37_universe.py`).
One resumable command, reusing the hard-won guardrails:

- `PRS_CACHE_DIR` on a large `/data` volume (the `/`-partition fills otherwise — the
  universe + panel + scoring caches total ~40 GB per build).
- Per-chromosome **streamed** position union (memory-bounded; a single hash-aggregate OOMs),
  `_DONE`-marked resumable parts, `PRS_DUCKDB_MEMORY_LIMIT` cap (~20GB), `preserve_insertion_order=false`.
- Expect a full-night runtime dominated by the GRCh37 scoring-file download + parquet conversion.

Example:
```bash
PRS_CACHE_DIR=/data/just-dna-lite/just-prs PRS_DUCKDB_MEMORY_LIMIT=20GB \
  uv run python scripts/build_reference_allele_universe.py --genome-build GRCh37 --push
```

## Verification

- Universe validation (ref ∈ ACGTN, no dupes, panel/fasta/unresolved split logged).
- Real GRCh37 array OFF-vs-ON recovery; restoration confined to chip-typed positions.
- Engine parity (polars vs DuckDB) with the GRCh37 universe + `Chip.GSA_V3` scope.
- No Windows/`pysam` regression (FASTA tier precompute-only, Linux/WSL).
