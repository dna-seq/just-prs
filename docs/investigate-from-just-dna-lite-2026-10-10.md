# To investigate: findings from just-dna-lite's Paper 1 review (2026-10-10)

## Resolution (just-prs 0.10.2, 2026-10-11)

1. Allele-aware matching, both engines: `375c377`, refined in `09366b8` (a one-base-REF
   record with any SNV ALT, e.g. `REF=G, ALT=A|AA`, still scores its SNV allele).
   New `PRSResult.variants_allele_mismatch`. Rules: `docs/vcf_flow.md`.
2. Multi-allelic ALTs are split (`|` and `,`): same commits.
3. Haploid GT counts as homozygous, 0..2, in both engines and in `normalize_vcf`'s
   `genotype` column. This is deliberate: PLINK2's default `--xchr-model 2` codes males 0..2,
   and the 1000G reference distributions were scored on that scale. Diploid-coded male chrX
   `1/1` = 2 therefore stays. No sex inference.
4. Benchmark inputs: `592dfba` (every `hm_inferOtherAllele` candidate, PASS-only pgen,
   liftover duplicates summed, chromosomes 1-22 in every engine, PLINK2 logs, per-ID CSV,
   ID list, git commit in the metadata; `attribute_residual.py` for the per-cause residual).
5. Docs: `53e8128` (panel text, catalog counts, chunk docstring, HLA loss) and
   `docs/benchmarks.md` in `592dfba`.
6. `PRSResult.model_flags` (`31f68c3`): dominant/recessive/interaction/diplotype/haplotype,
   ratio-scale weights, ignored per-dosage weights. Scores stay additive.

Not done here: refreshing the HF metadata snapshot (5,337 of 6,991 scores) is a
`pipeline catalog` run; the published `prs-sample-scores` runtime tables were computed under
position-only matching and are stale until rescored. Section 4 items (strand, DuckDB spill
comments) were left as observations.

## Work order from the just-dna-lite session (2026-10-10, the author's instruction: "fix all bugs and remeasure")

Fix, each with a regression test on real data:
1. **Allele-aware position join** (section 6, 9.5% of the PLINK2 gap). {effect, other} must equal {REF, one ALT} of
   the record; an SNV row must not match an indel sharing its anchor base. Both engines.
2. **Split multi-allelic ALT** (section 6, 1.4%). `C,G` gives a dose for the matching ALT index. Both engines.
3. **Haploid GT** (section 1). The Polars engine crashes on `GT="1"`; decide hemizygous dosage semantics, and
   make both engines agree (chrX male, chrM). Also chrX 1/1 counted as 2 copies.
4. **benchmark_engines.py inputs** (section 6, 87% of the gap): use every `hm_inferOtherAllele` candidate,
   build the pgen with PASS-only filtering (`--vcf-filter`, or from the normalized parquet), deduplicate
   liftover-duplicate IDs (the cause of the 4 PLINK2 failures), and keep PLINK2 stderr, `.log` and a
   per-ID results CSV.
5. **Docs:** the stale docstring (section 2), the 2,504/phase-3 text (section 3), the stale counts (section 5),
   and docs/benchmarks.md (wrong failure cause, "1-3 variants", tracemalloc vs RSS).
6. Optional, if cheap: read `is_dominant`, `is_recessive`, `is_interaction` and `weight_type`, or at least flag
   such scores in the result instead of scoring them silently as additive (section 5).

Then remeasure: rerun the engine benchmark (PGS000006 to PGS000106 excluding PGS000085, on Zenodo 18370498
`antonkulaga.vcf`, md5 21598da4f77d8237577662ab89e8a55d), and commit the results CSV, the PLINK2 logs and the
exact ID list. Report r, max |Δ| and the per-cause residual. The old-run analysis and scripts for
the per-variant attribution are in `/data/just-dna-cache/r-gap/deliverables/` (`run_all.sh`). Prepare
a patch version bump locally. Publishing and pushing need the author's explicit approval. Paper 1's PRS
benchmark section waits for the released version.

Written by a just-dna-lite session, which treats this repo as read-only. Nothing in this repo was changed
except this note. The findings come from a validator that read just-prs 0.10.1, which is both the version
just-dna-lite locks and the HEAD of this checkout (54cd4b0); `diff -rq` showed the two are identical.

## 1. The Polars engine crashes on haploid genotypes (likely a live bug)

- **What happens:** `compute_prs` (the Polars engine) raises `ComputeError: get index is out of bounds`
  on a haploid GT such as `"1"`. Callers write haploid GTs on male chrX and on chrM, DRAGEN among them.
- **Where:** `vcf.py:238`, `allele_2 = gt_normalized.str.split("/").list.get(1)`, which has no `null_on_oob`.
  The no-call helper `_gt_no_call_expr` (`prs.py:438-446`) does pass `null_on_oob=True`, but the crash
  happens before it runs.
- **The DuckDB engine scores the same row.** `split_part` returns `''` for the missing second allele, so
  `prs.py:1009-1024` counts it as a called genotype with dosage 1. The engines therefore disagree on
  haploid input: one crashes, the other scores. Decide which behaviour is correct for a hemizygous call,
  likely dosage 0/1 on a 0..1 scale, or 0/2, and make the two engines agree.
- **Repro:** a 1-row genotype frame with `GT="1"` at a scoring-file position, run through both engines.
  Scratch script from the validator, outside this repo:
  `/tmp/claude-1000/-data-sources-just-dna-lite/1abf7834-73d5-4643-98b1-9f52eddf1aed/scratchpad/validate-c17/t.py`
  (session scratch; it may be gone).
- **Who it hits:** just-dna-lite's web UI computes single scores through the Polars path
  (`webui/src/webui/state.py`, around line 3977). prs-ui's mixin defaults to DuckDB
  (`prs_ui/mixin.py:1661`), so the exposure depends on the entry point.

## 2. Stale docstring: the DuckDB chunk size

`compute_prs_duckdb`'s docstring (`prs.py:1303`) still says "default 250_000 rows". `memory.py:200-232`
now sizes the chunk from available RAM and clamps it to 250,000 to 2,000,000 rows, unless
`PRS_SCORING_JOIN_CHUNK_SIZE` overrides it.

## 3. The methodology text names the wrong reference panel

`viz.py:1558`, `2247` and `2472` say "1000 Genomes Project phase 3 reference panel (2,504 individuals, 5
superpopulations)". The pipeline scores `pgsc_1000G_v1` with 3,202 individuals (`reference.py:70-75`;
`cli.py:2208` checks `n_samples != 3202`; `docs/validation.md:30`).

## 4. Smaller observations from the same read (no action implied)

- There is no strand flipping or strand-ambiguous SNP handling. An effect allele that is neither REF nor ALT
  is scored as dosage 0 with no flag (`vcf.py:212-220`, `prs.py:1009-1019`). If that is deliberate, the
  docs could say so.
- DuckDB spill-to-disk is DuckDB's default; `prs.py` never sets `temp_directory`. Code comments at
  `prs.py:1182`, `1642` and `1657` describe spilling as if just-prs set it up.

## 5. Catalog coverage, measured against the PGS Catalog release of 2026-09-17 (6,991 scores)

A full scan by a just-dna-lite session: every GRCh38 and GRCh37 harmonized file, and every GRCh38 row
(4.50 billion). Per-rule ID lists sat in that session's scratchpad (`prs-survey/categories.json`), which
is temporary. Rerun the scan before relying on those lists.

- **Nothing is refused.** No file lacks coordinates, an effect allele or a weight column.
- **5 give a meaningless score:** PGS000343, PGS000941, PGS000966 and PGS003757 have no position on any
  row (HLA or allelotype), and PGS005228 gives the same allele as effect and other on every row. These
  are exactly the 5 `ReferencePanelError`s in `1000g_quality.parquet`.
- **76 more are scored with a known approximation:**
  - `is_dominant` and `is_recessive` are typed but never read (`scoring.py:67-68`).
  - `is_interaction` and `is_diplotype` are never read.
  - `weight_type` "OR over expected risk" (PGS000152-162) is summed additively.
  - PGS002246, 002247 and 002253 carry both `effect_weight` and dosage weights; the engine uses
    `effect_weight`, but the authors scored with the dosage weights.
  - Effect alleles that are not nucleotides (HLA names, D/I, `<CN0>` and similar) silently get dose 0
    (`vcf.py:258-264`).
- **Stale numbers in docs:**
  - README.md:11, :63 and just-prs/README.md:7 say "~5,385 scores".
  - The HF metadata (`scores.parquet`, 2026-06-14) holds 5,337 scores, up to the 2026-05-07 release, so
    1,654 newer scores are missing from `scores()` and `search()`.
  - CLAUDE.md:159-161 gives a "worst score 0.091%" harmonization loss; it is 100% for the 4 HLA scores.
- **847 scores are quarantined** (`exclude_from_catalog`), which leaves 4,490 visible by default.

## 6. Why just-prs and PLINK2 disagree in benchmark_engines.py (measured, zero residual)

Reproduced exactly (r = 0.99985891, max |Δ| 21.41 on PGS000012). Every per-variant difference was
assigned a cause. The deliverables (parquet, scripts and `run_all.sh`) are in
`/data/just-dna-cache/r-gap/deliverables/` on the user's machine.

- **71.4%: a benchmark defect.** `_prepare_plink2_score_input` uses only the first `/`-separated
  `hm_inferOtherAllele` candidate. Where the sample's ALT is a later candidate, PLINK2 cannot match
  heterozygous effect=REF sites (2,024 variants across 20 scores).
- **15.7%: a benchmark defect.** The pgen is built from the unfiltered VCF, with no `--vcf-filter`, so
  PLINK2 counts DeepVariant RefCall 0/0 sites as 2 reference copies. just-prs reads PASS only.
- **9.5%: just-prs defect (false matches).** The position join never checks the other allele or the
  variant type. An SNV scoring row matches an indel record that shares its anchor base (VCF `REF=AT,
  ALT=A` against an SNV row with effect allele `A` counts a dose the sample does not carry), and indel
  rows match a different indel. just-dna-lite's annotation engine had the same defect and fixed it by
  requiring reference-allele agreement in the position join ("`G>A` matched `GTGTCT>A`"). Fix: an
  allele-aware join, where {effect, other} must equal {REF, one ALT} of the record.
- **1.4%: just-prs defect.** Multi-allelic ALT strings (e.g. `C|G`) are not split, so dose is 0.
- **1.9%:** chrX is excluded by `--chr 1-22`. Separately, chrX hemizygous calls coded 1/1 count as 2.
- **Why PLINK2 fails on PGS000013/014/016/017:** each file has pairs of GRCh37 variants that lift over
  to the same GRCh38 variant. Writing both allele orders duplicates the IDs, and PLINK2 aborts with
  "ALT1 allele … appears multiple times". docs/benchmarks.md gives the wrong cause ("4-part ID
  matching") and says the matched counts differ by "1–3 variants" (on PGS000012 they differ by 3,677).
- docs/benchmarks.md describes tracemalloc memory; the current script measures RSS.
