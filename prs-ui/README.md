# prs-ui

[![PyPI version](https://badge.fury.io/py/prs-ui.svg)](https://badge.fury.io/py/prs-ui)

Reusable [Reflex](https://reflex.dev/) UI components for **Polygenic Risk Score (PRS)** computation using the [PGS Catalog](https://www.pgscatalog.org/).

Built on top of [`just-prs`](https://pypi.org/project/just-prs/) for the computation engine and [`reflex-mui-datagrid`](https://pypi.org/project/reflex-mui-datagrid/) for data grid display.

## Installation

```bash
pip install prs-ui
```

## Running the Standalone App

```bash
uv run start
```

The launcher reads `.env`, resolves both the Reflex frontend port (`PRS_UI_PORT`, default `3000`) and backend port (`PRS_UI_BACKEND_PORT`, default `8000`), and passes the resolved pair explicitly to Reflex so another running Reflex app cannot shift only one side.

## Quick Start

```python
import polars as pl
import reflex as rx
from reflex_mui_datagrid import LazyFrameGridMixin
from prs_ui import PRSComputeStateMixin, prs_section


class MyAppState(rx.State):
    genome_build: str = "GRCh38"
    cache_dir: str = ""
    status_message: str = ""


class PRSState(PRSComputeStateMixin, LazyFrameGridMixin, MyAppState):
    def load_genotypes(self, parquet_path: str) -> None:
        lf = pl.scan_parquet(parquet_path)
        self.set_prs_genotypes_lf(lf)  # preferred: provide a LazyFrame
        self.prs_genotypes_path = parquet_path


def prs_page() -> rx.Component:
    return prs_section(PRSState)
```

## Components

| Component | Description |
|-----------|-------------|
| `prs_workbench(source_section, prs_state, trait_state, mode_state, trait_selector, ...)` | Unified single-tab layout: one shared genotype source + `Select by PRS` / `Select by Trait` sub-tabs, per-mode controls, compute button, and per-mode results |
| `vcf_source_section(source_state)` | Reference **detachable** genotype source: compact VCF upload + build detection + collapsed normalized preview. Swap it for your own source in a host app |
| `prs_shared_build_bar(source_state)` | Single genome-build selector that fans the build out to all consumer states |
| `prs_section(state)` | Older single-state section: build selector + score grid + compute button + progress + results |
| `prs_build_selector(state)` | Genome build dropdown (GRCh37/GRCh38) |
| `prs_scores_selector(state)` | MUI DataGrid for score selection. Selection (checkboxes + Select/Clear) is **read-only and dimmed until genotypes are loaded**, with an "upload a VCF" callout |
| `prs_compute_button(state)` | Compute button with disclaimer callout |
| `prs_progress_section(state)` | Progress bar and status text during computation |
| `prs_results_table(state)` | Results table with quality badges, interpretation cards, and CSV download |

The genotype source is **loosely coupled** to the PRS logic: a source pushes a normalized
genotypes parquet into each consumer via the additive `load_genotypes(path)` hook (and
optionally `set_genome_build(build)`), so a host app can replace `vcf_source_section` /
`GenomicGridState` with its own source (public genome, consumer-array file, pre-normalized
parquet) without touching `PRSComputeStateMixin`. VCF **normalization** (not upload) is the
slow step; it is content-aware cached (a fresh normalized parquet is reused) and shown via an
indeterminate progress bar.

## State Mixin

`PRSComputeStateMixin` provides all PRS computation logic as a Reflex state mixin. Mix it into your concrete state class alongside `LazyFrameGridMixin` to get the full PRS workflow.

The preferred input method is a polars LazyFrame via `set_prs_genotypes_lf()` -- memory-efficient and avoids re-reading VCF files on each computation. A parquet path (`prs_genotypes_path`) is supported as a fallback.

## Documentation

See the [just-prs documentation](https://github.com/antonkulaga/just-prs) for the full Python API, CLI reference, and integration guide.
