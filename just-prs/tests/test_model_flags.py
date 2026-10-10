"""Scores the additive engines only approximate are flagged, not silently summed.

Real PGS Catalog files: PGS000021 (diplotype rows), PGS000024 (interaction
rows), PGS000152 ("Odds Ratio over expected risk" weights), PGS002246
(effect_weight next to per-dosage weights), PGS000006 (plain additive).
"""

from __future__ import annotations

import pytest

from just_prs.prs import ScoringModelFlag, _model_flags, _resolve_scoring
from just_prs.scoring import resolve_cache_dir

CASES = {
    "PGS000006": set(),
    "PGS000021": {ScoringModelFlag.DIPLOTYPE, ScoringModelFlag.HAPLOTYPE},
    "PGS000024": {ScoringModelFlag.INTERACTION, ScoringModelFlag.HAPLOTYPE},
    "PGS000152": {ScoringModelFlag.RATIO_WEIGHTS},
    "PGS002246": {ScoringModelFlag.DOSAGE_WEIGHTS_IGNORED},
}


@pytest.mark.parametrize(("pgs_id", "expected"), CASES.items())
def test_model_flags_match_the_scoring_file(pgs_id: str, expected: set[ScoringModelFlag]) -> None:
    cache = resolve_cache_dir() / "scores"
    try:
        scoring_lf = _resolve_scoring(pgs_id, "GRCh38", cache)
    except Exception as exc:  # offline and not cached
        pytest.skip(f"{pgs_id} unavailable: {exc}")
    flags = _model_flags(pgs_id, scoring_lf, "GRCh38", cache, pgs_id)
    assert set(flags) == {str(f) for f in expected}
