"""Deterministic PGS-ID selection for the real-data public-sample pilot."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from just_prs.prs import is_dosage_weight_format
from just_prs.scoring import scoring_parquet_path


def select_pilot_pgs_ids(
    scores_dir: Path,
    *,
    genome_build: str = "GRCh38",
    catalog_ids: list[str] | None = None,
) -> list[str]:
    """Pick small/median/large, one GenoBoost, and one permanently failing ID.

    Uses cached scoring-parquet byte sizes. Does not download. The failing ID
    is a catalog-shaped name that is not on disk so the engine records ``failed``.
    """
    sized: list[tuple[int, str]] = []
    genoboost: list[tuple[int, str]] = []
    candidates = catalog_ids
    if candidates is None:
        candidates = []
        for path in scores_dir.glob(f"*_hmPOS_{genome_build}.parquet"):
            pgs_id = path.name.split("_hmPOS_")[0]
            if pgs_id.startswith("PGS"):
                candidates.append(pgs_id)
        candidates = sorted(set(candidates))
    for pgs_id in candidates:
        path = scoring_parquet_path(pgs_id, scores_dir, genome_build)
        if not path.exists():
            continue
        size = path.stat().st_size
        sized.append((size, pgs_id))
        try:
            cols = pl.scan_parquet(path).collect_schema().names()
        except (pl.exceptions.ComputeError, OSError):
            continue
        if is_dosage_weight_format(cols):
            genoboost.append((size, pgs_id))
    if len(sized) < 3:
        raise FileNotFoundError(
            f"Need at least 3 cached {genome_build} scoring parquets in {scores_dir}"
        )
    sized.sort()
    picked = [
        sized[0][1],
        sized[len(sized) // 2][1],
        sized[-1][1],
    ]
    if genoboost:
        genoboost.sort()
        mid = genoboost[len(genoboost) // 2][1]
        if mid not in picked:
            picked.append(mid)
    failing = "PGS999999"
    if failing not in picked:
        picked.append(failing)
    return picked
