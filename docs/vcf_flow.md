# VCF / array input flow — from upload to score

How a genotype input travels from a file on disk to a PRS, where each decision is
made, and how reference restoration is scoped. This documents the *actual* code
paths (verified), including the compression edge case and the generalized
restoration interface.

## End-to-end path

```
file on disk
  │  (no extension trust — compression is detected by CONTENT)
  ├─ VCF  ─────────► normalize_vcf()            [normalize.py]
  │                    pb.scan_vcf(path)  ← polars-bio sniffs BGZF/gzip magic bytes
  │                    strip chr prefix, compute genotype, PASS/DP/QUAL filter
  │                    → normalized parquet (chrom,pos,ref,alt,GT,genotype)
  │
  └─ array ───────► normalize_array()           [arrays.py]
       (23andMe/      _read_array_text() ← open_maybe_compressed() (magic bytes) / .zip
        Ancestry)     observed-allele encoding → SAME normalized parquet schema

normalized parquet
  │
  ├─ detect_genome_build()        [vcf.py]  ← open_maybe_compressed() (magic bytes)
  │     ##reference= / ##contig= lengths → "GRCh37" | "GRCh38" | None
  │
  ├─ genotype_input_mode resolution  [prs.py _resolve_genotype_input_mode]
  │     "auto" → _infer_genotype_input_mode():
  │         alt has <NON_REF>  OR  filter has RefCall   → ALL_SITES
  │         otherwise                                   → VARIANT_ONLY
  │
  ├─ scoring join              [prs.py compute_prs / compute_prs_duckdb]
  │     VARIANT_ONLY : scoring LEFT JOIN genotypes   (absent loci retained)
  │     ALL_SITES    : genotypes INNER JOIN scoring  (absent loci dropped = missing)
  │
  └─ reference restoration (scope-gated, VARIANT_ONLY only)
        _normalize_restoration_scope() + _apply_reference_resolution()
        fills a missing reference_allele from the reference-allele universe so an
        absent locus scores as hom-ref — within the chosen scope.
```

## Compression: detect by content, not extension (the gz bug)

polars-bio's reader sniffs the BGZF/gzip magic bytes (`1f 8b`) when reading a VCF, so
a BGZF stream named `.vcf` (no `.gz`) normalizes fine. **But two code paths historically
chose their opener from the filename suffix**, which silently mis-handled that case:

- `detect_genome_build()` opened with `gzip.open if name.endswith((".gz",".bgz")) else open`.
  A BGZF-named-`.vcf` was read as plain text → garbage header → build returned `None`
  (the symptom: "build detection failed" on a file that scores fine). Commit `42ed0e8a`
  had band-aided this by adding `.bgz` to the suffix list.
- `arrays._read_array_text()` had the same `.gz`-suffix assumption.

**Fix:** `just_prs.io_utils.open_maybe_compressed(path)` decides by the magic bytes
(`is_gzip()` reads the first two bytes; BGZF is gzip-compatible so `gzip.open` reads it),
extension-independent. Both call sites now use it. `.zip` arrays stay special-cased.
Regression test: `tests/test_io_utils.py` (BGZF written as `.vcf` → correct build).

## genotype_input_mode semantics

| mode | absent scoring locus means | join |
|---|---|---|
| `VARIANT_ONLY` (default for plain VCFs) | sample is hom-ref there (callability assumed) | LEFT (scoring-driven) |
| `ALL_SITES` (gVCF / `<NON_REF>` / `RefCall`) | unavailable — stays missing | INNER (genotype-driven) |
| `PLINK_PRESENT_ONLY` | only present variants scored (PLINK parity) | INNER |

Note: a normalized **array auto-infers `VARIANT_ONLY`** (no `NON_REF`/`RefCall`
markers). Arrays actually report *all typed sites* (incl. hom-ref), so "absent" means
*untyped/off-chip*, which must **not** be blanket hom-ref — that's why restoration is
**scope-gated** (below) rather than applied to every absent locus.

## Which record is a scoring row's variant (allele-aware matching)

Both engines join on `(chrom, pos)` and then check that the record is the scoring
row's variant (`vcf.allele_compatible_expr`, mirrored in SQL by
`prs._duckdb_matched_sql`). Position alone is not identity: an indel shares its
anchor base with any SNV at the same position.

| scoring row | record | result |
|---|---|---|
| SNV (single-base effect and other alleles) | one-base REF with an SNV ALT (incl. `REF=G, ALT=A\|AA`) | dose = called alleles equal to the effect base |
| SNV | deletion / MNV (`REF=AT, ALT=A`) or pure insertion (`REF=A, ALT=AT`) | allele mismatch |
| indel | record whose alleles include the effect allele and a known other allele | dose counted |
| indel | a different indel (`CAAA>C` vs scored `CAAAA/C`) | allele mismatch |

- The other allele is `other_allele`, or else the `/`-separated
  `hm_inferOtherAllele` candidates. With none known, an indel row needs only its
  effect allele among the record's alleles.
- Multi-allelic ALTs (`G|T` from polars-bio, `G,T` in VCF text) are split, so a
  GT index selects its own allele. A third allele at an SNV (scoring `T/C`,
  sample `T/G`) still counts the REF copy.
- A position whose only records fail the check is scored as an **absent** locus
  (hom-ref when `reference_allele` is known, otherwise unscorable) and counted in
  `variants_allele_mismatch`. Several records at one position never score a row twice.
- A consumer array's homozygous encoding (`ref=alt`) skips the other-allele check.
- **Copies follow the karyotype.** A man has one X. Many callers do not write it
  that way: DeepVariant 1.1.0 writes anton's non-PAR X as 99,903 `1/1` and 4,818
  `0/1` and has no haploid GT, while DRAGEN writes it sex-aware and declares
  `##referenceSexKaryotype`. So the engines do not trust the GT's length.
  `just_prs.karyotype` infers XX / XY / XXY / X0 from the non-PAR X het rate
  (relative to the autosomes) and the called Y records, or takes
  `sample_karyotype=`. In a one-copy region (a man's non-PAR X and Y, MT in
  anyone), `1` and `1/1` both count once, a het (or any call on Y without a Y) is
  a no-call counted in `variants_ploidy_conflict`, and an absent hom-ref locus
  counts one copy. PAR is diploid. `unknown` scores every GT as written.
- **Coding.** `x_coding="copies"` (default) counts copies, 0..1 on a man's X.
  `"dosage_compensation"` counts 0..2, as PLINK2 `--xchr-model 2`, many GWAS and
  the 1000G pgen do (one active X in either sex). The 1000G reference
  distributions were built that way (a pgenlib read of 2,000 non-PAR X sites:
  males `{0, 2}`, never 1), so under `copies` a result that scored one-copy loci
  carries `reference_coding_mismatch=True` until the reference is rescored.
- **Provenance.** `normalize_vcf` stores the caller, its version, the declared
  karyotype, the caller's ploidy model (`sex_aware` / `diploid`) and the
  inferred karyotype as parquet metadata; read them with
  `normalize.read_genotype_provenance`.
- No strand flipping: harmonized scoring files are on the forward strand. An
  effect allele that is neither REF nor any ALT of an SNV record scores 0.
- Scores the additive sum only approximates (dominant/recessive/interaction/
  diplotype/haplotype rows, ratio-scale weights, ignored per-dosage weights) are
  still summed additively, and listed in `PRSResult.model_flags`.

## Reference restoration scope

`reference_restoration` (on `compute_prs` / `compute_prs_duckdb` / `compute_prs_batch` /
`PRSCatalog` / `prs compute --reference-restoration`) selects *which absent positions may
be hom-ref filled*. The reference-allele **universe** parquet
(`(genome_build, chrom, pos, ref, ref_source)`, published to `just-dna-seq/pgs-catalog
data/reference/`) is always the REF source of truth; the scope only gates eligibility.

Type: `RestorationScope = bool | Chip | Path | pl.DataFrame` (`Chip` is a `StrEnum`;
`Path`/`DataFrame` are the embedder escape hatch). Normalized by
`_normalize_restoration_scope()`:

| value | meaning | eligible set |
|---|---|---|
| `False` (default) | off — old behavior | none (absent + unknown ref → `variants_unscorable_absent`) |
| `True` | **WGS** | the whole universe (the universe index *is* the set — no duplication) |
| `Chip` (e.g. `Chip.GSA_V3`) | **array/chip** | chip-typed positions ∩ universe |
| `Path` / `DataFrame` | custom | that `(chrom,pos)` set ∩ universe |

Policy: **WGS → `True`; unknown → `False`; chip → `Chip`.** Default is still
`False` until the caller opts in — a `.vcf` is not proof of genome-wide WGS,
and gVCF / `RefCall` never restores. Why, with the intelligence report
gotcha: [reference-restoration.md](reference-restoration.md). Only positions whose
`reference_allele` was null/empty are filled; existing values always win. Counters
`variants_ref_resolved_panel` / `variants_ref_resolved_fasta` (subsets of
`variants_assumed_hom_ref`) record the source from the universe's `ref_source`
(`RefSource` StrEnum). Point-sets today; interval/range scopes (e.g. a callability BED)
are a planned overlap-join extension.

### Build-gating (important)

Restoration is correct only when the input build, the chip-manifest build, and the
universe build all agree. The published universe + GSA **A2** manifest are **GRCh38**,
while current consumer arrays are **GRCh37**. So array chip restoration
(`array_scoring.compute_array_prs` → `_resolve_array_restoration`) **no-ops on GRCh37
today** (logged) and activates once the GRCh37 universe + GSA **A1** manifest land
(see [grch37-universe-build.md](grch37-universe-build.md)). The chip-manifest gate in
`chip_typed_positions(chip, build=...)` raises rather than mixing builds.

## Array path specifics

`compute_array_prs` chains: normalize → `detect_chip_generation` → LD-proxy substitution
(impute untyped scoring positions from a typed proxy via the LD table) → reference
restoration (chip-typed hom-ref, build-gated) → `maf_fill` (2·MAF for remaining gaps) →
score. Restoration composes **before** maf_fill: a restored hom-ref locus is ref-known,
so the dosage chain takes it ahead of the MAF fill. LD-proxy and restoration are
complementary — LD-proxy estimates *untyped* positions; restoration recovers *typed*
positions absent from the file (no-calls).
