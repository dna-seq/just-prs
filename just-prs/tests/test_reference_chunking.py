"""Bounded scoring-side chunking in the reference-panel variant match.

``_ResolvedRefPanel.match_scoring`` registers the scoring file with DuckDB in
slices sized by ``scoring_join_chunk_size`` instead of handing it the whole
frame as one Arrow buffer. The catalog holds a few multi-million-variant scores
(PGS005172-PGS005197 are ~9.5M each); registering one of those whole put three
simultaneous copies of it in RAM -- the polars frame, its allele-filtered copy,
and the Arrow copy -- plus the DuckDB hash table built over that, with no
memory-pressure check anywhere in the path.

The property that has to hold is that chunking changes nothing about the
answer: same matched rows, whatever the chunk size. These tests pin that by
running the *same* scoring file through the *same* function at several chunk
sizes and comparing the results to the unchunked run.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from just_prs.memory import scoring_join_chunk_size
from just_prs.prs import _normalize_scoring_columns
from just_prs.reference import _ResolvedRefPanel, reference_panel_dir
from just_prs.scoring import ensure_scoring_file, parse_scoring_file, resolve_cache_dir

REF_DIR = reference_panel_dir()
REF_PANEL_AVAILABLE = (REF_DIR / "GRCh38_1000G_ALL.pgen").exists()

# 3,820 variants: enough rows to split many ways, small enough to score in seconds.
PGS_ID = "PGS000007"
GENOME_BUILD = "GRCh38"

pytestmark = pytest.mark.skipif(
    not REF_PANEL_AVAILABLE, reason="1000G reference panel not available"
)


@pytest.fixture(scope="module")
def panel() -> _ResolvedRefPanel:
    return _ResolvedRefPanel(REF_DIR, genome_build=GENOME_BUILD)


@pytest.fixture(scope="module")
def scoring_lf() -> pl.LazyFrame:
    cache = resolve_cache_dir() / "scores"
    cache.mkdir(parents=True, exist_ok=True)
    scoring_file = ensure_scoring_file(PGS_ID, cache, GENOME_BUILD)
    return _normalize_scoring_columns(parse_scoring_file(scoring_file))


@pytest.fixture(scope="module")
def variants_total(scoring_lf: pl.LazyFrame) -> int:
    return int(scoring_lf.select(pl.len()).collect().item())


def _canonical(df: pl.DataFrame) -> pl.DataFrame:
    """Sort rows and columns so two matched frames compare independent of order.

    Chunked runs emit matched rows grouped by scoring-file slice; the single-join
    run emits them in whatever order DuckDB's hash join produced. Neither order is
    part of the contract -- ``compute_reference_prs_polars`` argsorts on
    ``variant_idx`` immediately afterwards.
    """
    return df.select(sorted(df.columns)).sort(by=sorted(df.columns))


def _match_at_chunk_size(
    panel: _ResolvedRefPanel,
    scoring_lf: pl.LazyFrame,
    chunk_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> pl.DataFrame:
    """Run the match with ``PRS_SCORING_JOIN_CHUNK_SIZE`` pinned, or unset."""
    if chunk_size is None:
        monkeypatch.delenv("PRS_SCORING_JOIN_CHUNK_SIZE", raising=False)
    else:
        monkeypatch.setenv("PRS_SCORING_JOIN_CHUNK_SIZE", str(chunk_size))
    return panel.match_scoring(scoring_lf, pgs_id=PGS_ID)


class TestChunkingIsResultPreserving:
    """Chunk size must not change which variants match."""

    @pytest.mark.parametrize("chunk_size", [500, 1000, 3819, 3820, 3821, 100_000])
    def test_chunked_match_equals_unchunked(
        self,
        panel: _ResolvedRefPanel,
        scoring_lf: pl.LazyFrame,
        variants_total: int,
        chunk_size: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Baseline: one chunk larger than the file, i.e. the pre-fix single join.
        baseline = _match_at_chunk_size(
            panel, scoring_lf, variants_total * 10, monkeypatch
        )
        assert baseline.height > 0, "fixture must actually match something"

        chunked = _match_at_chunk_size(panel, scoring_lf, chunk_size, monkeypatch)

        assert chunked.height == baseline.height
        assert chunked.schema == baseline.schema
        assert _canonical(chunked).equals(_canonical(baseline))

    def test_chunk_boundaries_around_exact_file_size(
        self,
        panel: _ResolvedRefPanel,
        scoring_lf: pl.LazyFrame,
        variants_total: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A chunk one row under the file size takes the loop; one over does not.

        This is the off-by-one that would silently drop the last row, and the
        ``variants_total > chunk_size > 0`` guard is what decides it.
        """
        under = _match_at_chunk_size(panel, scoring_lf, variants_total - 1, monkeypatch)
        exact = _match_at_chunk_size(panel, scoring_lf, variants_total, monkeypatch)
        over = _match_at_chunk_size(panel, scoring_lf, variants_total + 1, monkeypatch)

        assert _canonical(under).equals(_canonical(exact))
        assert _canonical(over).equals(_canonical(exact))

    def test_lazyframe_and_dataframe_inputs_agree(
        self,
        panel: _ResolvedRefPanel,
        scoring_lf: pl.LazyFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The batch path passes a LazyFrame; older callers pass a DataFrame."""
        monkeypatch.setenv("PRS_SCORING_JOIN_CHUNK_SIZE", "500")
        from_lazy = panel.match_scoring(scoring_lf, pgs_id=PGS_ID)
        from_eager = panel.match_scoring(scoring_lf.collect(), pgs_id=PGS_ID)

        assert _canonical(from_lazy).equals(_canonical(from_eager))


class TestChunkSizing:
    """``scoring_join_chunk_size`` is what decides whether the loop runs."""

    def test_env_override_is_exact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRS_SCORING_JOIN_CHUNK_SIZE", "1234")
        assert scoring_join_chunk_size(10_000) == 1234

    def test_chunk_never_exceeds_remaining(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRS_SCORING_JOIN_CHUNK_SIZE", "1000")
        assert scoring_join_chunk_size(10) == 10

    def test_empty_scoring_file_is_not_chunked(self) -> None:
        """A zero-row file must fall to the single-join path, not loop forever."""
        assert scoring_join_chunk_size(0) == 0

    def test_auto_size_engages_loop_for_a_nine_million_variant_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The catalog's largest scores must actually take the chunked path.

        If auto-sizing ever returned something >= 9.5M this fix would be inert on
        exactly the files it exists for.
        """
        monkeypatch.delenv("PRS_SCORING_JOIN_CHUNK_SIZE", raising=False)
        nine_and_a_half_million = 9_502_208
        chunk = scoring_join_chunk_size(nine_and_a_half_million)
        assert 0 < chunk < nine_and_a_half_million
