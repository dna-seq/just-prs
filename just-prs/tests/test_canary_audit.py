"""Canary-collapse flags: PGS003724-style 0th-percentile on every genome."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from just_prs.canary_audit import (
    CANARY_COLLAPSE_ISSUE,
    canary_collapse_issues,
    encode_canary_samples_env,
    excluded_pgs_ids,
    flag_canary_collapses,
    load_canary_result_rows,
    load_canary_scores,
    merge_audit_issues,
    parse_canary_samples_env,
    parse_canary_vcf_spec,
    parse_canary_vcf_specs,
    upsert_canary_score_rows,
)


def _row(
    pgs_id: str,
    sample_id: str,
    *,
    percentile: float,
    z_score: float,
    match_rate: float,
) -> dict[str, object]:
    return {
        "pgs_id": pgs_id,
        "sample_id": sample_id,
        "percentile": percentile,
        "z_score": z_score,
        "match_rate": match_rate,
        "score": 50.0,
    }


def test_flag_pgs003724_style_collapse() -> None:
    rows = pl.DataFrame([
        _row("PGS003724", "anton", percentile=0.0, z_score=-9.2, match_rate=0.425),
        _row("PGS003724", "livia", percentile=0.0, z_score=-9.5, match_rate=0.428),
        _row("PGS003724", "mom", percentile=0.0, z_score=-8.0, match_rate=0.428),
        _row("PGS004427", "anton", percentile=93.8, z_score=1.5, match_rate=0.53),
        _row("PGS004427", "livia", percentile=99.3, z_score=2.4, match_rate=0.53),
        _row("PGS004427", "mom", percentile=22.4, z_score=-0.8, match_rate=0.53),
    ])
    flags = flag_canary_collapses(rows)
    assert flags["pgs_id"].to_list() == ["PGS003724"]
    assert flags["exclude_from_catalog"][0] is True
    assert flags["flag"][0] == CANARY_COLLAPSE_ISSUE
    assert excluded_pgs_ids(flags) == ["PGS003724"]


def test_real_low_tail_with_good_coverage_is_not_flagged() -> None:
    rows = pl.DataFrame([
        _row("PGS_REAL", "anton", percentile=2.0, z_score=-2.1, match_rate=0.91),
        _row("PGS_REAL", "livia", percentile=1.5, z_score=-2.2, match_rate=0.90),
        _row("PGS_REAL", "mom", percentile=3.0, z_score=-1.9, match_rate=0.92),
    ])
    flags = flag_canary_collapses(rows)
    assert flags.height == 0


def test_single_sample_zero_is_not_enough() -> None:
    rows = pl.DataFrame([
        _row("PGS003724", "anton", percentile=0.0, z_score=-9.2, match_rate=0.42),
    ])
    flags = flag_canary_collapses(rows, n_samples=3)
    assert flags.height == 0


def test_majority_of_three_flags_two_zeros() -> None:
    rows = pl.DataFrame([
        _row("PGS003724", "anton", percentile=0.0, z_score=-9.2, match_rate=0.42),
        _row("PGS003724", "livia", percentile=0.2, z_score=-8.1, match_rate=0.43),
        _row("PGS003724", "mom", percentile=48.0, z_score=-0.1, match_rate=0.90),
    ])
    flags = flag_canary_collapses(rows, n_samples=3)
    assert flags["pgs_id"].to_list() == ["PGS003724"]
    assert flags["n_extreme"][0] == 2


def test_canary_issues_expand_per_superpopulation() -> None:
    flags = flag_canary_collapses(pl.DataFrame([
        _row("PGS003724", "anton", percentile=0.0, z_score=-9.2, match_rate=0.42),
        _row("PGS003724", "livia", percentile=0.0, z_score=-9.5, match_rate=0.43),
    ]))
    distributions = pl.DataFrame({
        "pgs_id": ["PGS003724", "PGS003724", "PGS004427"],
        "superpopulation": ["EUR", "AFR", "EUR"],
        "mean": [767.0, 728.0, -0.15],
        "std": [77.0, 52.0, 0.19],
        "n": [633, 893, 633],
        "median": [764.0, 720.0, -0.16],
        "p5": [649.0, 640.0, -0.46],
        "p25": [712.0, 690.0, -0.28],
        "p75": [818.0, 760.0, -0.02],
        "p95": [901.0, 810.0, 0.18],
    })
    issues = canary_collapse_issues(distributions, flags)
    assert set(issues["pgs_id"].to_list()) == {"PGS003724"}
    assert set(issues["superpopulation"].to_list()) == {"EUR", "AFR"}
    assert set(issues["issue"].to_list()) == {CANARY_COLLAPSE_ISSUE}
    assert set(issues["severity"].to_list()) == {"ERROR"}


def test_merge_audit_issues_dedupes() -> None:
    base = pl.DataFrame({
        "pgs_id": ["PGS_LOW"],
        "superpopulation": ["EUR"],
        "severity": ["ERROR"],
        "issue": ["quality_low_match_rate"],
        "recommended_action": ["exclude"],
        "mean": [0.0],
        "std": [1.0],
        "n": [633],
        "median": [0.0],
        "p5": [-1.0],
        "p25": [-0.5],
        "p75": [0.5],
        "p95": [1.0],
    })
    extra = canary_collapse_issues(
        pl.DataFrame({
            "pgs_id": ["PGS_LOW", "PGS_LOW"],
            "superpopulation": ["EUR", "AFR"],
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "n": [633, 893],
            "median": [0.0, 0.0],
            "p5": [-1.0, -1.0],
            "p25": [-0.5, -0.5],
            "p75": [0.5, 0.5],
            "p95": [1.0, 1.0],
        }),
        pl.DataFrame({
            "pgs_id": ["PGS_LOW"],
            "exclude_from_catalog": [True],
        }),
    )
    merged = merge_audit_issues(base, extra)
    assert merged.height == 3
    assert CANARY_COLLAPSE_ISSUE in merged["issue"].to_list()


def test_scores_hides_excluded_ids(tmp_path: Path) -> None:
    from just_prs import PRSCatalog
    from just_prs.canary_audit import write_catalog_flags

    meta = tmp_path / "metadata"
    meta.mkdir()
    pl.DataFrame({
        "pgs_id": ["PGS003724", "PGS004427"],
        "name": ["IQ", "fluid"],
        "trait_reported": ["Intelligence quotient", "Fluid intelligence score"],
        "trait_efo": ["intelligence", "intelligence"],
        "genome_build": ["GRCh37", "GRCh37"],
    }).write_parquet(meta / "scores.parquet")
    pl.DataFrame({"pgs_id": ["PGS003724"], "ppm_id": ["x"]}).write_parquet(meta / "performance.parquet")
    pl.DataFrame({"pgs_id": ["PGS003724"], "ppm_id": ["x"]}).write_parquet(meta / "best_performance.parquet")
    flags = flag_canary_collapses(pl.DataFrame([
        _row("PGS003724", "anton", percentile=0.0, z_score=-9.2, match_rate=0.42),
        _row("PGS003724", "livia", percentile=0.0, z_score=-9.5, match_rate=0.43),
    ]))
    write_catalog_flags(flags, tmp_path)
    catalog = PRSCatalog(cache_dir=tmp_path)
    visible = catalog.scores().collect()["pgs_id"].to_list()
    assert "PGS003724" not in visible
    assert "PGS004427" in visible
    all_ids = catalog.scores(include_excluded=True).collect()["pgs_id"].to_list()
    assert "PGS003724" in all_ids


def test_load_canary_result_rows_from_scores_parquet(tmp_path: Path) -> None:
    upsert_canary_score_rows(tmp_path, pl.DataFrame([
        _row("PGS003724", "anton", percentile=0.0, z_score=-9.2, match_rate=0.425),
        _row("PGS003724", "livia", percentile=0.0, z_score=-8.0, match_rate=0.428),
        _row("PGS003724", "mom", percentile=0.0, z_score=-8.5, match_rate=0.427),
    ]))
    rows = load_canary_result_rows(tmp_path)
    assert rows.height == 3
    assert set(rows["sample_id"].to_list()) == {"anton", "livia", "mom"}
    flags = flag_canary_collapses(rows, n_samples=3)
    assert flags["pgs_id"].to_list() == ["PGS003724"]


def test_parse_canary_vcf_spec_label_and_bare_path() -> None:
    assert parse_canary_vcf_spec("mom=/data/mom.vcf.gz") == ("mom", "/data/mom.vcf.gz")
    assert parse_canary_vcf_spec("anton") == ("anton", "anton")
    assert parse_canary_vcf_spec("SIMHIFQTILQ.hard-filtered.vcf.gz")[0] == "SIMHIFQTILQ"


def test_parse_canary_vcf_specs_resolves_paths(tmp_path: Path) -> None:
    anton = tmp_path / "antonkulaga.vcf"
    livia = tmp_path / "livia.vcf.gz"
    mom = tmp_path / "mom.vcf.gz"
    for path in (anton, livia, mom):
        path.write_bytes(b"##fileformat=VCFv4.2\n")
    samples = parse_canary_vcf_specs(
        [str(anton), f"livia={livia}", f"mom={mom}"],
        tmp_path,
    )
    assert [sample.label for sample in samples] == ["antonkulaga", "livia", "mom"]
    assert samples[2].vcf_path == mom.resolve()
    encoded = encode_canary_samples_env(samples)
    roundtrip = parse_canary_samples_env(encoded, tmp_path)
    assert [sample.label for sample in roundtrip] == ["antonkulaga", "livia", "mom"]


def test_upsert_canary_scores_replaces_same_sample(tmp_path: Path) -> None:
    first = pl.DataFrame([
        _row("PGS003724", "anton", percentile=0.0, z_score=-9.2, match_rate=0.42),
        _row("PGS004427", "anton", percentile=90.0, z_score=1.2, match_rate=0.80),
    ])
    upsert_canary_score_rows(tmp_path, first)
    upsert_canary_score_rows(tmp_path, pl.DataFrame([
        _row("PGS003724", "anton", percentile=0.0, z_score=-8.1, match_rate=0.43),
        _row("PGS003724", "livia", percentile=0.0, z_score=-8.0, match_rate=0.43),
    ]))
    stored = load_canary_scores(tmp_path)
    anton_iq = stored.filter((pl.col("pgs_id") == "PGS003724") & (pl.col("sample_id") == "anton"))
    assert anton_iq.height == 1
    assert anton_iq["z_score"][0] == -8.1
    assert stored.height == 3
