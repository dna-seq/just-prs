# just-prs: The Polygenic Risk Score (PRS) Toolbox

[![PyPI version](https://badge.fury.io/py/just-prs.svg)](https://pypi.org/project/just-prs/)
[![PyPI version](https://badge.fury.io/py/prs-ui.svg)](https://pypi.org/project/prs-ui/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Research use only](https://img.shields.io/badge/use-research%20only-orange.svg)](#research-use-only-interpreting-prs-results)
[![Not medical advice](https://img.shields.io/badge/medical-not%20advice-red.svg)](#research-use-only-interpreting-prs-results)
[![MCP ready](https://img.shields.io/badge/MCP-Claude%20%7C%20Cursor%20%7C%20Codex-blueviolet.svg)](https://github.com/dna-seq/just-prs-mcp)
[![Web UI](https://img.shields.io/badge/UI-browser%20app-2ea44f.svg)](#web-ui)

`just-prs` is a Polygenic Risk Score (PRS) toolbox for working with the
[PGS Catalog](https://www.pgscatalog.org/): normalize VCFs, search scores and
traits, compute PRS values, compare them with reference populations, and estimate
absolute disease risk.

The goal is not to pretend PRS science is cleaner than it is. Published PRS can
be contradictory, unevenly validated, ancestry-biased, or simply low quality.
Instead of showing only a tiny hand-picked highlight reel, `just-prs` gives you
the broader research landscape with quality signals, match rates, reference
context, and warnings so you can inspect the evidence yourself.

Many human traits and common diseases, such as type 2 diabetes, coronary artery
disease, height, and longevity, are **polygenic**: they are influenced by
thousands of small genetic effects across the genome rather than one single
"faulty gene". A PRS adds those small effects together and places your result
relative to a reference population. It is not a diagnosis and does not guarantee
an outcome; it is a way to visualize inherited predisposition and, where enough
evidence is available, translate a percentile into an absolute-risk estimate.

You can use it three ways:

- **Open it in the browser** with the `prs-ui` web app: upload a VCF, browse
  traits, compute scores, and inspect bell curves, absolute-risk estimates, and
  plain-English explanations.
- **Use it with Claude, Cursor, Codex, Antigravity, or other AI agents** through
  [just-prs-mcp](https://github.com/dna-seq/just-prs-mcp): ask an agent to
  download a public genome, normalize it, search the catalog, compute PRS, and
  explain the result in chat.
- **Run it from the CLI or Python** for scripts, notebooks, and pipelines:
  Polars/DuckDB-backed VCF normalization, scoring-file parsing, variant matching,
  batch scoring, and reference-panel workflows.

## Contents

- [Web UI](#web-ui)
- [Use with Claude, Cursor, Codex, Antigravity, or other agents](#use-with-claude-cursor-codex-antigravity-or-other-agents)
- [CLI and Python](#cli-and-python)
- [Test Genomes (Quick Play)](#test-genomes-quick-play)
- [Research Use Only: Interpreting PRS Results](#research-use-only-interpreting-prs-results)
- [Installation](#installation)
- [Features](#features)
- [Why not PLINK2?](#why-not-plink2)
- [Project Structure](#project-structure)
- [Embedding PRS UI in Another Reflex App](#embedding-prs-ui-in-another-reflex-app)
- [Testing](#testing)
- [Documentation](#documentation)
- [Data sources](#data-sources)

## Web UI

Prefer a browser? The [Reflex](https://reflex.dev/) app lets you upload a VCF,
browse PGS Catalog traits and scores, compute PRS results, and inspect bell
curves, absolute-risk context, and plain-English interpretations.

![PRS Compute UI — upload VCF, select scores, compute PRS](images/PRS_screenshot.jpg)

### Setup

```bash
# From the workspace root — install all packages (including prs-ui)
uv sync --all-packages

# Launch the UI (shortcut defined in pyproject.toml)
uv run ui
```

The UI opens at http://localhost:3000 with three tabs: **Compute PRS**,
**Metadata Sheets**, and **Scoring File**.

### Compute PRS (default tab)

A single workbench has one shared genotype source feeding two selection modes:

1. **Upload a VCF once** — the app detects the genome build, normalizes the VCF,
   caches the normalized Parquet, and feeds both selection modes.
2. **Select by PRS or by Trait** — compute individual PGS IDs, or pick a whole
   trait such as type 2 diabetes and aggregate all associated PGS models into a
   consensus summary.
3. **Download CSV** — export computed results from the results table.

The metadata and scoring-file tabs are for browsing PGS Catalog sheets and
streaming harmonized scoring files by PGS ID.

## Use with Claude, Cursor, Codex, Antigravity, or other agents

The MCP server lives at
[github.com/dna-seq/just-prs-mcp](https://github.com/dna-seq/just-prs-mcp) and
is published as `just-prs-mcp`, so Claude Code, Cursor, Codex, Antigravity, and
other MCP-capable agents can launch it without cloning anything. This is a
first-class path for non-developers too: you can ask the assistant what you want
in plain language and let it call the PRS tools.

For Claude Code:

```bash
claude mcp add just-prs -- uvx just-prs-mcp@latest stdio
```

For Cursor, add this to `.cursor/mcp.json` in your project, or to your user MCP
configuration:

```json
{
  "mcpServers": {
    "just-prs": {
      "command": "uvx",
      "args": ["just-prs-mcp@latest", "stdio"],
      "env": {
        "PRS_MCP_MODE": "essentials"
      }
    }
  }
}
```

For Codex, use the equivalent MCP server configuration:

```toml
[mcp_servers.just-prs]
command = "uvx"
args = ["just-prs-mcp@latest", "stdio"]
```

For Antigravity or another MCP-capable assistant, add the same server command:
`uvx just-prs-mcp@latest stdio`.

Then ask your agent something like: "Download Anton's sample genome, normalize
it, and compute the PRS score for type 2 diabetes."

## CLI and Python

Run one-off analyses, scripts, notebooks, and batch jobs directly from the
terminal or Python.

```bash
# Compute PRS for a single score
prs compute --vcf sample.vcf.gz --pgs-id PGS000001

# Multiple scores at once
prs compute --vcf sample.vcf.gz --pgs-id PGS000001,PGS000002,PGS000003

# Normalize a VCF to Parquet (strip chr prefix, compute genotype, quality filter)
prs normalize --vcf sample.vcf.gz --pass-filters "PASS,." --min-depth 10

# Search the catalog
prs catalog scores search --term "breast cancer"
```

```python
import polars as pl
from pathlib import Path

from just_prs import PRSCatalog, VcfFilterConfig, normalize_vcf
from just_prs.prs import compute_prs

catalog = PRSCatalog()

config = VcfFilterConfig(pass_filters=["PASS", "."], min_depth=10)
parquet_path = normalize_vcf(Path("sample.vcf.gz"), Path("sample.parquet"), config=config)
genotypes_lf = pl.scan_parquet(parquet_path)

results = catalog.search("type 2 diabetes", genome_build="GRCh38").collect()
result = compute_prs(
    vcf_path="sample.vcf.gz",
    scoring_file="PGS000001",
    genome_build="GRCh38",
    genotypes_lf=genotypes_lf,
)
print(f"Score: {result.score:.6f}, Match rate: {result.match_rate:.1%}")
```

## Test Genomes (Quick Play)

You can try the toolbox without using your own genome. Two public WGS VCFs from
the `just-dna-lite` authors are documented and ready for demos, testing, and
agent workflows:

1. **Anton Kulaga's Genome** (CC0 / Public Domain)
   - **Zenodo Record**: [18370498](https://zenodo.org/records/18370498)
   - **VCF File**: `antonkulaga.vcf` (~482 MB)
   - **Direct URL**: `https://zenodo.org/api/records/18370498/files/antonkulaga.vcf/content`

2. **Livia Zaharia's Genome** (CC-BY-4.0)
   - **Zenodo Record**: [19487816](https://zenodo.org/records/19487816)
   - **VCF File**: `SIMHIFQTILQ.hard-filtered.vcf.gz` (~349 MB)
   - **Direct URL**: `https://zenodo.org/api/records/19487816/files/SIMHIFQTILQ.hard-filtered.vcf.gz/content`

An MCP-enabled agent can fetch either genome with `download_sample_genome` using
`sample="anton"` or `sample="livia"`. You can also download a VCF and upload it
to the browser UI, or run the CLI directly:

```bash
curl -L -o anton.vcf "https://zenodo.org/api/records/18370498/files/antonkulaga.vcf/content"
prs compute --vcf anton.vcf --pgs-id PGS000001
```

## Research Use Only: Interpreting PRS Results

The UI is designed to make the messiness visible instead of hiding it. In this
example, several PGS models for the same intelligence-related trait are shown
together: their percentile positions on the bell curve, variant match rates,
quality breakdown, outliers, and consensus summary are all visible at once.

![Trait-first PRS interpretation example — multiple PGS models, match rates, quality summary, and consensus bell curve](images/intelligence.jpg)

When reading a result like this, look at the whole panel, not only the largest
percentile number:

- **Bell curve and markers** show where each model places the genome relative to
  a reference population; disagreement between markers is information, not a UI
  bug.
- **Variant match rate** shows whether each score had enough overlapping variants
  in the genome file to be interpretable.
- **Quality breakdown** separates high, moderate, low, and very-low-quality
  models, so weak models do not silently count the same as better-supported ones.
- **Outlier and consensus summaries** help you see whether a trait-level signal is
  stable across models or dominated by one unusual score.
- **Source links** let you inspect the underlying PGS Catalog entries instead of
  trusting a single opaque number.

### Why not show only a few "best" scores?

Because that can create a false sense of certainty. Many people arrive with
intuitions from medical-grade genetic testing: a narrow question, a validated
gene or variant, and a relatively clear interpretation. PRS research is not like
that. It is a broad, statistical, still-evolving literature where scores for the
same trait can be inconsistent, weakly validated, ancestry-biased, or sensitive
to how the phenotype was defined.

`just-prs` is intentionally a toolbox, not a black-box verdict engine. It exposes
many available PGS Catalog models, highlights quality and match-rate problems,
groups scores by trait, and lets you see disagreement instead of hiding it. The
point is to help you inspect the evidence, not to pretend the field has one clean
answer for every trait.

### Why do several PRS for the same trait give different answers?

This is normal, and it is one of the main reasons the UI supports **trait-first**
analysis instead of forcing you to pick a single PGS ID. The PGS Catalog often
has many scores for the same broad trait, but they may have been trained on
different cohorts, ancestries, phenotype definitions, genome builds, variant
sets, and statistical methods. A "type 2 diabetes" score from one study is not
necessarily the same model as a "type 2 diabetes" score from another study.

So if four or five PRS models disagree, it usually means one or more of these is
true: the models are measuring slightly different definitions of the trait; some
models are lower quality; your VCF did not match enough variants for one score;
the score was developed in an ancestry group unlike the reference population you
are comparing against; or the published effect sizes simply do not generalize
well to every person.

### What does "research use only" actually mean here?

It means you should not treat a PRS result as medical-grade evidence. Many people
are used to genetic tests that look at a narrow, high-confidence question, such
as a known pathogenic variant in a clinically validated gene. PRS are different:
they are statistical models built from many small associations, often with modest
predictive power and uneven validation.

The [PGS Catalog](https://www.pgscatalog.org/) is an excellent research resource,
but being listed there does **not** mean every score is clinically ready,
high-quality, ancestry-portable, or useful for an individual decision. Some
scores are exploratory, some are trained on small or narrow cohorts, some perform
poorly outside the original study population, and some may match too few variants
in your genome file to be interpretable.

### Which PRS should I trust more?

Prefer scores with better published evaluation metrics, higher variant match
rates, relevant ancestry information, and agreement with other high-quality
models for the same trait. Treat a single PRS as one research signal, not as a
verdict. The trait summary view is designed to help you see consensus and
outliers rather than overreacting to one score.

### Does a high PRS mean I will get a disease?

No. A PRS is not a diagnosis. It is a statistical estimate of inherited
predisposition relative to a reference population. Environment, age, sex,
family history, lifestyle, clinical biomarkers, and chance can matter as much as
or more than common genetic variants.

### What should I do with a worrying result?

Do not panic, and do not make medical decisions from research-grade PRS output
alone. If a result concerns you, especially if it matches your family history or
clinical context, discuss it with a clinician or genetic counselor and confirm
important findings through appropriate clinical testing.

### Why does ancestry matter?

PRS models are often strongest in populations similar to the people used to train
and validate them. Many published PGS Catalog scores still come from cohorts with
heavy European ancestry bias. `just-prs` can show reference percentiles across
available population panels, but that does not make every score equally reliable
for every ancestry.

### What is linkage disequilibrium, and why does it matter?

Many GWAS variants are not proven causal variants. They are often **tagging**
nearby genomic regions because variants close together can be inherited together;
this correlation pattern is called **linkage disequilibrium** (LD). LD patterns
vary between populations, so a variant that tags risk well in one ancestry group
may tag it poorly in another. This is one reason PRS can lose accuracy when moved
outside the cohort where they were trained.

### What is penetrance, and is PRS the same thing?

**Penetrance** usually describes how often people with a specific high-impact
variant develop a related disease. PRS are different: they combine thousands of
common variants, each usually with a tiny effect. A high PRS does not mean a
disease is inevitable, and a low PRS does not mean protection is guaranteed. It
is a probabilistic signal, not a deterministic mutation report.

### Why does coverage or match rate matter?

A PRS scoring file may contain hundreds, thousands, or millions of variants, but
your genome file may not contain all of them. Variants can be missing because the
input is exome-only, array-derived, low coverage, filtered, in a different genome
build, or lacks the alleles needed to score confidently. Low match rate means the
score is being computed from an incomplete subset of the model, so the result may
be weak or misleading. Always inspect matched variants, total variants, and
assumed/missing loci before interpreting a score.

### Why can population reference curves disagree?

Reference percentiles answer "where does this score sit compared with this
reference panel?" They do not prove that the original PGS model works equally
well in that population. A score can have a percentile in several 1000 Genomes
superpopulations while still being trained mostly in Europeans, calibrated on a
different cohort, or affected by ancestry-specific LD and allele-frequency
patterns.

### What does absolute risk mean?

Absolute risk tries to convert a relative PRS percentile into a real-world
probability using trait prevalence and published performance data. This is useful
for context, but it is only as good as the underlying prevalence estimate, model
quality, and study population. When the evidence is weak or missing, the app
should show that rather than pretending the number is precise.

## Installation

Requires Python >= 3.13. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

**From PyPI:**

```bash
pip install just-prs
```

**From source (development):**

```bash
git clone https://github.com/antonkulaga/just-prs
cd just-prs
uv sync --all-packages   # installs all three subprojects + dev deps
```

To install only the core library without UI or pipeline: `cd just-prs/just-prs && uv sync`.

The CLI is available as both `just-prs` and `prs`.

### Windows

The web UI and VCF-based PRS computation (the main use case) work on Windows with **no C compiler required**. The reference-panel dependency `pgenlib` is automatically excluded on Windows (via a `sys_platform != 'win32'` marker) because it has no Windows wheels and its bundled C fails to compile with MSVC. So a plain checkout works out of the box:

```bash
cd just-prs
uv sync --all-packages
uv run ui
```

Only the reference-panel / `.pgen` scoring features (`prs reference …`, `prs pgen …`, and the Dagster pipeline) are unavailable on native Windows. If you need those, run them under **WSL** or **Linux**, where `pgenlib` installs from a prebuilt wheel.

## Features

- **PRS computation from VCF** — normalize VCFs to Parquet, compute one or many
  PGS IDs, and inspect match rates, effect sizes, quality labels, percentiles,
  and absolute-risk context.
- **Trait-first analysis** — select a trait such as type 2 diabetes instead of a
  single score; compute all associated PGS models and summarize agreement,
  outliers, and quality.
- **PGS Catalog metadata** — search cleaned score, trait, performance,
  publication, prevalence, and scoring-file metadata without hand-parsing the
  catalog sheets.
- **Fast data engine** — Polars and DuckDB-backed scoring, zstd-compressed
  Parquet caches, and HuggingFace sync for cleaned metadata and reference
  distributions.
- **Reference and pgen workflows** — optional Linux/WSL support for `.pgen`,
  `.pvar.zst`, `.psam`, 1000G / HGDP+1kGP reference scoring, and PLINK2
  cross-validation.
- **Reusable UI components** — embed the PRS workbench or individual Reflex
  components in another app via `PRSComputeStateMixin` and `load_genotypes(path)`.

## Why not PLINK2?

[PLINK2](https://www.cog-genomics.org/plink/2.0/) is the gold-standard tool for whole-genome association analysis, and its `--score` command is widely used for PRS computation. `just-prs` provides a pure Python alternative that produces **identical results** (validated with Pearson r = 1.0 across 3,202 samples, relative per-sample differences < 5e-7 — see [validation details](docs/validation.md)) while offering several practical advantages:

| | PLINK2 | just-prs |
|---|---|---|
| **Installation** | Platform-specific binary; manual download or conda | `pip install just-prs` — pure Python, works everywhere |
| **Integration** | Shell subprocess with text file I/O | Native Python API — returns polars DataFrames directly |
| **Composability** | Fixed CLI pipeline; parse .sscore/.log outputs | Modular functions: parse variants, read genotypes, match alleles, compute scores — mix and match |
| **Intermediate formats** | Must write temporary score input files | Operates on in-memory DataFrames and numpy arrays |
| **Dependencies** | External binary + system libraries | Only Python packages (pgenlib, polars, numpy) |
| **Debugging** | Parse log files for match stats | Structured Eliot logging with full variant-level visibility |
| **Batch scoring** | One subprocess per PGS ID | Reuses parsed `.pvar` and genotype caches across scores |

The core building blocks — `parse_pvar()`, `parse_psam()`, `read_pgen_genotypes()`, `match_scoring_to_pvar()`, and `compute_reference_prs_polars()` — are all public API and can be used independently for any analysis involving PLINK2 binary format files.

### Quick example: score a PGS against any .pgen dataset

```python
from just_prs import compute_reference_prs_polars
from pathlib import Path

scores_df = compute_reference_prs_polars(
    pgs_id="PGS000001",
    scoring_file=Path("PGS000001_hmPOS_GRCh38.txt.gz"),
    ref_dir=Path("/path/to/pgen_dir"),  # any dir with .pgen/.pvar.zst/.psam
    out_dir=Path("/tmp/output"),
    genome_build="GRCh38",
)
# Returns a polars DataFrame: iid, superpop, population, score, pgs_id
```

```bash
# Or from the CLI:
prs pgen score PGS000001 /path/to/pgen_dir/
prs reference score PGS000001  # single PGS ID against 1000G panel

# Batch score all PGS IDs to build population distributions:
prs reference score-batch                              # all PGS IDs
prs reference score-batch --pgs-ids PGS000001,PGS000002
prs reference score-batch --limit 50 --panel hgdp_1kg  # HGDP+1kGP panel
```

For cross-validation against PLINK2, use `prs reference compare PGS000001` which runs both engines and reports per-sample correlation and timing.

## Project Structure

This is a **uv workspace** with three subprojects:

| Package | Directory | Description |
|---|---|---|
| **just-prs** | `just-prs/` | Core library: PRS computation, PGS Catalog client, VCF normalization, scoring files. Published to PyPI. |
| **prs-ui** | `prs-ui/` | Reflex web UI for interactive PRS computation. Published to PyPI. |
| **prs-pipeline** | `prs-pipeline/` | Dagster pipeline for computing reference distributions from population panels (1000G, HGDP+1kGP). |

The workspace root is a non-published wrapper that depends on all three
subprojects and provides convenience scripts such as `uv run ui` and
`uv run pipeline`.

## Embedding PRS UI in Another Reflex App

The PRS computation UI is packaged as reusable [Reflex](https://reflex.dev/) components. Install `prs-ui` (which pulls in `just-prs` automatically), mix `PRSComputeStateMixin` into your state, and feed normalized genotypes through the loose-coupling `load_genotypes(path)` hook. The genotype **source** is detachable — your app supplies its own (a public-genome selector, a consumer-array file, a pre-normalized parquet) and never has to use the bundled VCF upload:

```python
import reflex as rx
from reflex_mui_datagrid import LazyFrameGridMixin
from prs_ui import PRSComputeStateMixin, prs_section


class MyAppState(rx.State):
    genome_build: str = "GRCh38"
    cache_dir: str = ""
    status_message: str = ""


class PRSState(PRSComputeStateMixin, LazyFrameGridMixin, MyAppState):
    """Consumer state — no override needed; load_genotypes is built in."""


def prs_page() -> rx.Component:
    return prs_section(PRSState)


# From your own source handler, push genotypes into the consumer:
#   async def on_genome_ready(self, parquet_path: str):
#       prs = await self.get_state(PRSState)
#       prs.load_genotypes(parquet_path)          # built-in loose-coupling hook
#       for event in prs.set_genome_build("GRCh38"):
#           yield event
```

`load_genotypes(path)` is the loose-coupling contract (it sets `prs_genotypes_path`, rescans the LazyFrame, and clears stale results); you can also call `set_prs_genotypes_lf()` directly with a `pl.scan_parquet()` LazyFrame for memory-efficient, no-re-read computation. For the full single-tab **By PRS / By Trait** experience with your own source, render `prs_workbench(source_section=..., prs_state=..., trait_state=..., mode_state=..., trait_selector=...)`. Individual sub-components (`prs_scores_selector`, `prs_results_table`, `trait_summary_table`, `prs_compute_button`, `prs_progress_section`, `prs_build_selector`, `prs_shared_build_bar`, `vcf_source_section`) can be used independently for custom layouts. `trait_summary_table(state)` groups PRS results by trait and shows consensus bell curves, outlier detection, and quality breakdown — call `state.build_trait_summary()` after computation to populate it.

## Testing

The project includes an extensive integration test suite that runs against real genomic data and external tools -- no mocked data or synthetic fixtures. All tests are reproducible on any Linux, macOS, or Windows machine.

```bash
uv run pytest just-prs/tests/ -v
```

| Test suite | What it validates | Data source |
|---|---|---|
| `test_plink.py` | PRS scores match [PLINK2](https://www.cog-genomics.org/plink/2.0/) `--score` within floating-point precision for 5 GRCh38 scores | Real whole-genome VCF from Zenodo; PLINK2 auto-downloaded |
| `test_percentile.py` | Theoretical mean/SD from allele frequencies, percentile computation, and cross-validation against PLINK2 for 5 scores with allele frequency data | Real PGS scoring files with `allelefrequency_effect` |
| `test_reference_plink2.py` | Reference panel PLINK2 scoring: variant ID construction, allele matching, end-to-end scoring of 4 PGS IDs across 3,202 samples, superpopulation coverage, distribution aggregation | 1000G reference panel + PLINK2 binary (both auto-downloaded) |
| `test_prs.py` | End-to-end PRS computation (single and batch) on a real VCF | Zenodo test VCF |
| `test_cleanup.py` | Full cleanup pipeline: column renaming, genome build normalization, metric string parsing, performance flattening, `PRSCatalog` search/percentile on live catalog data | Real PGS Catalog bulk metadata (~5,000+ scores) via EBI FTP |
| `test_scoring.py` | Scoring file download, parsing, and caching | Real PGS000001 harmonized scoring file |
| `test_scoring_parquet_cache.py` | Parquet cache roundtrip: schema/value fidelity, header metadata preservation, skip-download when cached, PRS equivalence between `.txt.gz` and parquet | 4 real PGS scoring files (PGS000001/2/10/13) + test VCF |
| `test_catalog.py` | REST API client: score lookup, trait search, download URL resolution | Live PGS Catalog REST API |

Key properties of the test suite:

- **PLINK2 cross-validation** — our pgenlib + polars engine produces identical results to PLINK2 `--score` (Pearson r = 1.0 across 3,202 samples, relative per-sample differences < 5e-7). Both VCF-level PRS and reference panel scoring are validated ([details](docs/validation.md))
- **Real data throughout** — test VCF auto-downloaded from Zenodo, PLINK2 binary auto-downloaded for the host platform, scoring files fetched from EBI FTP
- **Percentile verification** — theoretical statistics computed from allele frequencies are validated against manual row-by-row computation, and percentiles are checked for mathematical consistency (CDF symmetry, known quantiles)
- **No mocking** — all tests run real pipelines against real data to catch integration issues

## Documentation

- [CLI Reference](docs/cli.md) — full command-line usage for `prs compute`, `prs normalize`, `prs pgen`, `prs reference`, `prs catalog`, and bulk downloads
- [Python API](docs/python-api.md) — `PRSCatalog`, pgen operations, VCF normalization, reference panel scoring, FTP downloads, REST client, cleanup pipeline, HuggingFace sync
- [Absolute Risk Methodology](docs/absolute-risk-methodology.md) — mathematical models, prevalence data sourcing, confidence tiers, and caveats for converting PRS percentiles to absolute disease risk
- [Dagster Pipelines](docs/dagster.md) — architecture and orchestration of the reference panel and metadata pipelines
- [Validation](docs/validation.md) — accuracy benchmarks against PLINK2 `--score` (individual VCF and reference panel)
- [Cleanup Pipeline](docs/cleanup-pipeline.md) — genome build normalization, column renaming, metric parsing

## Data sources

- PGS Catalog REST API: <https://www.pgscatalog.org/rest/>
- EBI FTP bulk downloads: <https://ftp.ebi.ac.uk/pub/databases/spot/pgs/>
- PGS Catalog download documentation: <https://www.pgscatalog.org/downloads/>
- Cleaned metadata and scoring parquets on HuggingFace: <https://huggingface.co/datasets/just-dna-seq/pgs-catalog>
