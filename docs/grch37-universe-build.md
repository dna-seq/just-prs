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

## Build results (2026-06-26)

Both universes are built, validated, and published to HF `just-dna-seq/pgs-catalog`
(`data/reference/`), each from all 5,385 `hmPOS_{build}` scoring parquets (coverage 5385/5385):

| build | filename | rows | panel | fasta | unresolved | resolved |
|-------|----------|------|-------|-------|------------|----------|
| GRCh38 | `reference_allele_universe.parquet` (unsuffixed) | 34,929,041 | 26.01M | 7.93M | 0.99M | 97.16% |
| GRCh37 | `reference_allele_universe_GRCh37.parquet` | 34,922,878 | 31.83M | 2.39M | 0.70M | 97.99% |

Validation per build: `ref ∈ {A,C,G,T,N}`, no duplicate `(chrom,pos)`, no null in resolved rows.
A parser fix landed during the build — some harmonized files serialize integer positions in
scientific notation (e.g. `chr_position=7.2e+07`); `scoring._parse_gz_scoring_file` now reads
Int64 columns as Float64 then casts (positions < 2^53 → exact). See also the publish
**completeness gate** (refuses `--push` below `--min-coverage`) and download retry/logging.

## Restoration validation (2026-06-26) — real consumer files

OFF (no restoration) vs WGS (`True`, whole universe) vs CHIP (`Chip.GSA_V3`, GSA-typed ∩ universe),
GRCh37, against PGS000337 (75,028 variants). Data sources are all **CC0 / public**:

| file | vendor / format | markers | chip (auto-detect) | restoration path validated |
|------|-----------------|---------|--------------------|----------------------------|
| Corpas *father* (figshare 4491215, CC0) | 23andMe v3, 4-col TSV | 957,353 | `omniexpress` | forced GSA → mechanics + confinement |
| *Jessica* (PGP/Arvados, public) | AncestryDNA V2, 5-col TSV (GSA) | 675,396 | **`gsa_v3` (auto)** | **runtime auto-path engages** |
| *Dave* (PGP/Arvados, public) | MyHeritage 2018, comma-CSV | 704,479 | `omniexpress` (correct — pre-2019 MyHeritage *is* OmniExpress) | new CSV ingestion + forced GSA |

Findings (consistent across all files, both engines):
- **OFF** restores nothing; **WGS** restores every absent scoring locus (e.g. all 57,443 for
  Corpas/PGS000337); **CHIP** restores only the GSA-typed subset (3,774), leaving the off-chip
  absent unscorable — confinement is exact (`3,774 + 53,669 = 57,443`).
- **Score is invariant** across modes (restored loci are hom-ref → 0 effect-allele dosage); what
  improves is `variants_matched` / `match_rate` — the F15 lever disambiguates "low score" from
  "low coverage", it does not move the score.
- `_resolve_array_restoration` **auto-engages** for `gsa_v3`+GRCh37 (Jessica: matched 15,647 →
  19,582, 3,935 restored) and **gracefully no-ops** for `omniexpress` (no manifest) — degradation
  confirmed.

Vendor-format coverage: 4-col TSV (23andMe), 5-col TSV (AncestryDNA), and comma-CSV
(MyHeritage/FTDNA — ingestion added in `arrays.py`, commit `ea4fc63`). Re-encoding the real Corpas
genotypes into all three formats yields byte-identical normalized output and identical restoration
results, proving the parser is format-agnostic.

Data-access note: PGP raw files now redirect to Arvados `collections.ac2it.arvadosapi.com`
collections — some are public-readable but require a **browser User-Agent** (else they 401 as
anti-bot); others are genuinely non-public (401 even with a UA). 23andMe-`_v5_`-named PGP files sit
in non-public collections, but AncestryDNA V2 *is* the same Illumina GSA v3.0 platform as 23andMe
v5 (`detect_chip_generation` even labels it "23andMe v5"), so Jessica's file covers the v5/GSA path.

## Still open

- Engine parity spot-check (polars vs DuckDB) on the GRCh37 universe + `Chip.GSA_V3` (numbers
  above are DuckDB; polars path expected identical).
- The runtime **GRCh37→38 liftover bridge** (restore-in-37 → lift→38 → score-in-38) is a separate
  path under active development; confirm EBI-harmonize(38→37) vs pyliftover(37→38) round-trip.
- No Windows/`pysam` regression (FASTA tier is precompute-only, Linux/WSL).
- A **2019+** MyHeritage file would let the auto-path engage GSA (the 2018 sample is OmniExpress).
