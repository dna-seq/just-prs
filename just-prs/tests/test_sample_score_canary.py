"""Public canary rows come from unrestored public-wgs-pass-v1 runtime scores."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from just_prs.canary_audit import CANARY_RESULT_SCHEMA
from just_prs.normalize import VcfFilterConfig
from just_prs.sample_scores.canary import (
    attach_percentiles,
    canary_rows_from_unrestored_runtime,
    private_canary_samples,
    public_canary_labels,
)
from just_prs.sample_scores.models import (
    PUBLIC_WGS_PASS_V1,
    RESTORED_PROFILE_ID,
    UNRESTORED_PROFILE_ID,
)
from just_prs.sample_scores.publish import is_publication_allowed
from just_prs.sample_scores.store import write_runtime_results
from just_prs.canary_audit import CanarySample


def test_public_wgs_pass_v1_is_pass_dot_only() -> None:
    config = PUBLIC_WGS_PASS_V1.to_filter_config()
    assert isinstance(config, VcfFilterConfig)
    assert config.pass_filters == ["PASS", "."]
    assert config.min_depth is None
    old_canary = VcfFilterConfig()
    assert old_canary.pass_filters != config.pass_filters or old_canary.min_depth != config.min_depth


def test_canary_rows_use_unrestored_only(tmp_path: Path) -> None:
    write_runtime_results(
        pl.DataFrame([
            {
                "sample_id": "anton",
                "pgs_id": "PGS000001",
                "scoring_build": "GRCh38",
                "score_profile_id": UNRESTORED_PROFILE_ID,
                "scoring_fingerprint": "fp",
                "status": "ok",
                "score": 1.5,
                "match_rate": 0.9,
                "sample_genotype_sha256": "b" * 64,
            },
            {
                "sample_id": "anton",
                "pgs_id": "PGS000001",
                "scoring_build": "GRCh38",
                "score_profile_id": RESTORED_PROFILE_ID,
                "scoring_fingerprint": "fp",
                "status": "ok",
                "score": 9.9,
                "match_rate": 0.99,
                "sample_genotype_sha256": "b" * 64,
            },
        ]),
        tmp_path,
    )
    rows = canary_rows_from_unrestored_runtime(tmp_path, sample_ids=["anton"])
    assert rows.height == 1
    assert rows["score"][0] == 1.5
    assert rows["percentile"][0] is None


def test_attach_percentiles_joins_distributions_once() -> None:
    rows = pl.DataFrame(
        {
            "pgs_id": ["PGS000001"],
            "sample_id": ["anton"],
            "percentile": [None],
            "z_score": [None],
            "match_rate": [0.9],
            "score": [10.0],
        },
        schema=CANARY_RESULT_SCHEMA,
    )
    distributions = pl.DataFrame({
        "pgs_id": ["PGS000001", "PGS000001"],
        "superpopulation": ["EUR", "AFR"],
        "mean": [0.0, 5.0],
        "std": [5.0, 5.0],
    })
    annotated = attach_percentiles(rows, distributions, ancestry="EUR")
    assert annotated["z_score"][0] == 2.0
    assert annotated["percentile"][0] is not None
    assert annotated["percentile"][0] > 95.0


def test_private_labels_stay_off_the_runtime_path(tmp_path: Path) -> None:
    samples = [
        CanarySample(label="anton", vcf_path=tmp_path / "anton.vcf"),
        CanarySample(label="stranger", vcf_path=tmp_path / "x.vcf"),
    ]
    assert public_canary_labels(samples) == ["anton"]
    assert [s.label for s in private_canary_samples(samples)] == ["stranger"]
    assert is_publication_allowed("stranger") is False
