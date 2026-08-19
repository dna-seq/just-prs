# Reference restoration — why it is off by default

A plain WGS VCF usually lists sites where the sample **differs** from the
reference. A PGS scoring variant that is **absent** from that file is therefore
often homozygous-reference (hom-ref) — *if* the file is really a genome-wide
callset and the site was callable.

Most PGS Catalog scoring files do not name the reference allele. Without it the
engine cannot orient an absent locus, so it counts the site as
`variants_unscorable_absent` instead of guessing. **Reference restoration**
fills that missing REF from the published `reference_allele_universe.parquet`,
so those absent loci can score as hom-ref.

That fill is a **claim about the file**, not a better model. It is opt-in.

```bash
# default: only score what was observed + REF already in the scoring file
prs compute --vcf /abs/path/sample.vcf --pgs-id PGS000001

# WGS fill: treat catalog-wide absent sites as hom-ref
prs compute --vcf /abs/path/sample.vcf --pgs-id PGS000001 --reference-restoration wgs
prs plot trait intelligence --vcf /abs/path/sample.vcf --reference-restoration wgs -o out.html
```

Values: `off` (default) · `wgs` (whole universe) · a chip id (`gsa_v3`, typed
positions only). Same flag on `prs plot` and `prs prompt`. The plot result
cache keys on this choice, so an unrestored hit is never reused for a restored
run. The UI **Recover absent loci (WGS)** checkbox is the same `wgs` / `off`
choice; recovered and unrestored session results are stored separately and
the checkbox switches caches when both already exist.

Implementation and join semantics: [vcf_flow.md](vcf_flow.md).

## Policy

| Input | Default | Restoration |
|---|---|---|
| Unknown / undeclared VCF | `off` | Do not guess |
| Genome-wide variant-only WGS | still `off` until the caller says `wgs` | Whole universe |
| Consumer array | `off` unless `--reference-restoration gsa_v3` (or `compute_array_prs`, which uses chip scope) | Chip-typed positions only |
| gVCF / `<NON_REF>` / DeepVariant `RefCall` | n/a | **Never.** Mode is `all_sites`; restoration is skipped |

`auto` genotype mode infers `all_sites` from `<NON_REF>` or `RefCall`, else
`variant_only`. Restoration engages only in `variant_only`. A wrong-on is a
silent coverage lie; a low match rate is honest.

## Why not default-on for every VCF

A `.vcf` is not proof of genome-wide WGS.

- **Arrays look variant-only.** They have no `<NON_REF>` / `RefCall` markers.
  `wgs` on an array would treat every untyped off-chip PGS position as hom-ref
  and report ~100% coverage on scores that need imputation. Arrays need **chip
  scope**, not a blanket fill.
- **WES, a chromosome extract, or a hard-filtered subset** look the same.
  Filling the whole catalog universe there invents genotypes the file never
  assayed.
- **Even real WGS is only assumed callable** at missing sites. A `PASS`-only
  VCF dropped a lot of the genome. Unrestored ~50% match is “what we saw.”
  Restored ~99% is “treat the rest as hom-ref.” That is a modeling choice.

When the caller declares nothing, the engine stays conservative: never
fabricate hom-ref on a guess.

## Why gVCF is never restored

A gVCF already speaks to reference sites (per-site records or `END` ref-blocks).
An absent scoring locus means “not in this callset,” not “hom-ref.” Filling
those from the universe would ignore the file’s own callability.

DeepVariant `RefCall` in a normalized parquet is treated as `all_sites` for the
same reason — o-mom’s parquet has `RefCall`, so `auto` skips restoration even
when `--reference-restoration wgs` is set. Force
`--genotype-input-mode variant_only` only when you intend WGS-style absence
semantics on that file.

The right gVCF lever is expanding `END` blocks (not shipped). That is true
per-site callability, not a universe guess. gVCF is also a rare personal-genome
input; ~99% of files are plain variant-only VCFs.

## What changes when you turn it on

On genome-wide WGS, match rate typically goes from ~50% to ~99%. Percentiles
usually move a little. A few models move a lot, especially genome-wide scores
that were mostly `unscorable_absent`.

Intelligence (Anton / Livia / o-family, 2026-08-18):

- Five usable models (match ≥50% unrestored) are **byte-identical** between an
  older o-family report and a fresh unrestored recompute.
- **PGS003724** (IQ, 6.7M variants) is the type specimen: unrestored everyone
  lands at percentile 0 (~43% match). Restored match ~99% and people separate
  (Anton ~94, Livia ~90, o-son2 ~60, others low). That is coverage fill, not a
  better IQ model. It is also the canary-collapse example.

Restored ≠ better. It answers a different question: “score the catalog as if
this WGS was callable everywhere the universe has a REF.”

## Why two HTML reports can disagree

Compare reports by **(samples × PGS IDs × restoration × model scope)**, not by
the title “intelligence.”

The older [`data/examples/intel_o_family.html`](../data/examples/intel_o_family.html)
is unrestored o-family only, five usable models, dashboard scope `usable`
(match ≥50%). Headline: Mom median **22.4**, risk **0.77x**. Every shared
percentile matches a fresh unrestored recompute exactly.

A later `prs plot trait intelligence --fuzzy` report looks different when it:

1. Adds Anton / Livia.
2. Includes **PGS003724** (unrestored percentile 0, or restored onto a real
   scale). `--models usable` still drops it from the median because unrestored
   match is ~43%.
3. Turns **`--reference-restoration wgs`** on. Then several non-canary
   percentiles move (largest here: o-son1 PGS003510 33 → 16, o-dad PGS003510
   67 → 81).
4. Prints CLI “best model” risk (o-mom **1.42x** from PGS001232) instead of the
   HTML card, which uses the **median across usable models** (**0.77x**).

Same genomes, same trait, three different numbers — all consistent with their
scope. Open the HTML cards, not the CLI “best model” line, when comparing
family reports.
