"""User-built scoring files: cache layout, reference distributions, percentiles.

A custom score is any scoring file that is not a PGS Catalog ID — typically
one built from GWAS summary statistics by :mod:`just_prs.prscs`. It has no
published reference distribution, so :func:`build_custom_reference_distribution`
scores it against the local 1000G panel once and stores
``<score_dir>/<panel>_distributions.parquet`` next to the scoring file, where
:func:`custom_percentiles` finds it.

Layout (``<cache>/custom_scores/<score_id>/``)::

    <score_id>_hmPOS_GRCh38.parquet    scoring file (header metadata embedded)
    1000g_distributions.parquet        per-superpopulation mean/std/quantiles
    prscs/                             PRS-CS inputs, posteriors and logs
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from eliot import start_action
from pydantic import BaseModel

from just_prs.scoring import read_scoring_header, resolve_cache_dir

SUPERPOPULATIONS: tuple[str, ...] = ("AFR", "AMR", "EAS", "EUR", "SAS")


class CustomScore(BaseModel):
    """A scoring file found under ``<cache>/custom_scores``."""

    score_id: str
    scoring_file: Path
    header: dict[str, str]
    distributions: dict[str, Path] = {}


def custom_scores_dir(cache_dir: Path | None = None) -> Path:
    """Root for user-built scores: ``<cache>/custom_scores``."""
    return (cache_dir or resolve_cache_dir()) / "custom_scores"


def custom_score_path(score_id: str, genome_build: str = "GRCh38", cache_dir: Path | None = None) -> Path:
    """Canonical scoring parquet of a user-built score."""
    return custom_scores_dir(cache_dir) / score_id / f"{score_id}_hmPOS_{genome_build}.parquet"


def distributions_path(scoring_file: Path, panel: str = "1000g") -> Path:
    """Where the reference distribution of ``scoring_file`` lives (its sibling)."""
    return scoring_file.parent / f"{panel}_distributions.parquet"


def resolve_custom_score(value: str | Path, genome_build: str = "GRCh38", cache_dir: Path | None = None) -> Path:
    """Accept either a path to a scoring file or the id of a built custom score."""
    path = Path(value).expanduser()
    if path.exists():
        return path
    candidate = custom_score_path(str(value), genome_build, cache_dir)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"{value!r} is neither an existing scoring file nor a built custom score ({candidate})")


def list_custom_scores(cache_dir: Path | None = None) -> list[CustomScore]:
    """Every built custom score, with its header and any reference distributions."""
    root = custom_scores_dir(cache_dir)
    scores: list[CustomScore] = []
    for scoring_file in sorted(root.glob("*/*_hmPOS_*.parquet")):
        scores.append(
            CustomScore(
                score_id=scoring_file.parent.name,
                scoring_file=scoring_file,
                header=read_scoring_header(scoring_file),
                distributions={
                    p.name.removesuffix("_distributions.parquet"): p
                    for p in scoring_file.parent.glob("*_distributions.parquet")
                },
            )
        )
    return scores


def build_custom_reference_distribution(
    scoring_file: Path,
    score_id: str | None = None,
    panel: str = "1000g",
    genome_build: str = "GRCh38",
    force: bool = False,
) -> Path:
    """Score a custom file on the reference panel and store its distribution.

    Needs the local panel (``prs reference download``) and the ``reference``
    extra (pgenlib). Reuses the stored distribution when it is newer than the
    scoring file. Per-individual scores are kept under
    ``<score_dir>/reference_<panel>/`` for inspection.
    """
    from just_prs.reference import aggregate_distributions, compute_reference_prs_polars, reference_panel_dir

    score_id = score_id or read_scoring_header(scoring_file).get("pgs_id") or scoring_file.parent.name
    out = distributions_path(scoring_file, panel)
    if out.exists() and not force and out.stat().st_mtime >= scoring_file.stat().st_mtime:
        return out
    with start_action(action_type="custom_scores:reference_distribution", score_id=score_id, panel=panel):
        scores = compute_reference_prs_polars(
            pgs_id=score_id,
            scoring_file=scoring_file,
            ref_dir=reference_panel_dir(panel=panel),
            out_dir=scoring_file.parent / f"reference_{panel}",
            genome_build=genome_build,
        )
        scores.write_parquet(scoring_file.parent / f"reference_{panel}" / "scores.parquet")
        aggregate_distributions(scores).write_parquet(out)
    return out


def custom_percentiles(
    scoring_file: Path,
    prs_score: float,
    panel: str = "1000g",
) -> dict[str, float] | None:
    """Percentile of ``prs_score`` in each superpopulation, or ``None`` without a distribution."""
    from just_prs.reference import ancestry_percentile

    dist = distributions_path(scoring_file, panel)
    if not dist.exists():
        return None
    lf = pl.scan_parquet(dist)
    score_id = lf.select(pl.col("pgs_id").first()).collect().item()
    percentiles: dict[str, float] = {}
    for superpop in SUPERPOPULATIONS:
        value = ancestry_percentile(prs_score, score_id, superpop, lf)
        if value is not None:
            percentiles[superpop] = value
    return percentiles
