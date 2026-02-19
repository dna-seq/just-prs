# prs-ui

[![PyPI version](https://badge.fury.io/py/prs-ui.svg)](https://badge.fury.io/py/prs-ui)

Reusable [Reflex](https://reflex.dev/) UI components for **Polygenic Risk Score (PRS)** computation using the [PGS Catalog](https://www.pgscatalog.org/).

Built on top of [`just-prs`](https://pypi.org/project/just-prs/) for the computation engine and [`reflex-mui-datagrid`](https://pypi.org/project/reflex-mui-datagrid/) for data grid display.

## Installation

```bash
pip install prs-ui
```

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
| `prs_section(state)` | Complete PRS section: build selector + score grid + compute button + progress + results |
| `prs_build_selector(state)` | Genome build dropdown (GRCh37/GRCh38) |
| `prs_scores_selector(state)` | MUI DataGrid for score selection with checkboxes and filtering |
| `prs_compute_button(state)` | Compute button with disclaimer callout |
| `prs_progress_section(state)` | Progress bar and status text during computation |
| `prs_results_table(state)` | Results table with quality badges, interpretation cards, and CSV download |

## State Mixin

`PRSComputeStateMixin` provides all PRS computation logic as a Reflex state mixin. Mix it into your concrete state class alongside `LazyFrameGridMixin` to get the full PRS workflow.

The preferred input method is a polars LazyFrame via `set_prs_genotypes_lf()` -- memory-efficient and avoids re-reading VCF files on each computation. A parquet path (`prs_genotypes_path`) is supported as a fallback.

## Documentation

See the [just-prs documentation](https://github.com/antonkulaga/just-prs) for the full Python API, CLI reference, and integration guide.
