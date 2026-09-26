"""Checks behind the curate-papers / curate-trait skills, run on the real cleaned catalog.

PGS000318 (All-cause mortality, female; HR 1.10) is the fixture: its catalog text and
evaluation metric are real, so the quote and metric-sign checks run against source data.
"""

import json
from pathlib import Path

import polars as pl
import pytest
import yaml
from just_prs.curation import Level, run_checks

CLUSTERS = {
    "version": 0,
    "domains": {
        "aging": {
            "label": "Aging",
            "concepts": {
                "lifespan": {
                    "label": "Lifespan and survival",
                    "axis": "longer life",
                    "valence": "desirable",
                    "valence_basis": "Longer survival is the outcome people want.",
                    "phenotypes": {
                        "longevity": {"label": "Survival to old age", "sign": 1},
                        "all_cause_mortality": {"label": "Death from any cause", "sign": -1},
                    },
                }
            },
        }
    },
}


def _entry(polarity: int, quote: str) -> dict:
    return {
        "cluster": "aging.lifespan",
        "phenotype": "all_cause_mortality",
        "measurement_kind": "time_to_event",
        "score_effect": {"direction": "increases", "target": "hazard of death from any cause"},
        "polarity": polarity,
        "strata": {"sex": "female"},
        "provenance": [
            {"source": "catalog", "locator": "performance.trait_reported", "quote": quote, "confidence": "low"}
        ],
    }


def _write(root: Path, scores: dict) -> None:
    (root / "publications").mkdir(parents=True)
    (root / "clusters.yaml").write_text(yaml.safe_dump(CLUSTERS))
    (root / "publications" / "PGP000095.yaml").write_text(
        yaml.safe_dump({"pgp_id": "PGP000095", "scores": scores})
    )


def _by_check(findings: list) -> dict[str, set[Level]]:
    grouped: dict[str, set[Level]] = {}
    for f in findings:
        grouped.setdefault(f.check, set()).add(f.level)
    return grouped


def test_correct_annotation_passes(tmp_path: Path) -> None:
    _write(tmp_path, {"PGS000318": _entry(-1, "All-cause mortality (age at death in females)")})
    checks = _by_check(run_checks(tmp_path, "PGP000095", threshold=0.2))
    assert checks["polarity"] == {Level.OK}
    assert checks["quote"] == {Level.OK}
    # HR 1.10 against mortality: expected sign (-1) * (-1) = +1, observed > 1.
    assert checks["metric_sign"] == {Level.OK}
    assert Level.ERROR not in set().union(*checks.values())


def test_wrong_polarity_and_invented_quote_are_errors(tmp_path: Path) -> None:
    _write(tmp_path, {"PGS000318": _entry(1, "Mortality was lower among carriers of the score")})
    checks = _by_check(run_checks(tmp_path, "PGP000095", threshold=0.2))
    assert checks["polarity"] == {Level.ERROR}
    assert checks["quote"] == {Level.ERROR}


def test_score_from_another_paper_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, {"PGS000906": _entry(-1, "All-cause mortality (age at death in females)")})
    checks = _by_check(run_checks(tmp_path, "PGP000095", threshold=0.2))
    assert checks["membership"] == {Level.ERROR}


def test_null_direction_needs_a_reason(tmp_path: Path) -> None:
    _write(tmp_path, {"PGS000318": {"score_effect": None}})
    findings = run_checks(tmp_path, "PGP000095", threshold=0.2)
    assert [f.check for f in findings] == ["schema"]
    assert findings[0].level == Level.ERROR
    assert "unannotated_reason" in findings[0].message


@pytest.mark.parametrize("unknown_key", ["higher_is_better", "polarty"])
def test_unknown_keys_fail_instead_of_being_dropped(tmp_path: Path, unknown_key: str) -> None:
    entry = _entry(-1, "All-cause mortality (age at death in females)")
    entry[unknown_key] = 1
    _write(tmp_path, {"PGS000318": entry})
    findings = run_checks(tmp_path, "PGP000095", threshold=0.2)
    assert findings[0].check == "schema" and findings[0].level == Level.ERROR


def test_build_exports_annotations_and_clusters(tmp_path: Path) -> None:
    from just_prs.curation import write_tables

    _write(
        tmp_path,
        {
            "PGS000318": _entry(-1, "All-cause mortality (age at death in females)"),
            "PGS000319": {"score_effect": None, "unannotated_reason": "not read yet"},
        },
    )
    out = tmp_path / "out"
    manifest = write_tables(tmp_path, out)
    annotations = pl.read_parquet(out / "score_annotations.parquet")
    clusters = pl.read_parquet(out / "trait_clusters.parquet")
    assert set(annotations["pgs_id"]) == {"PGS000318", "PGS000319"}
    row = annotations.filter(pl.col("pgs_id") == "PGS000318").row(0, named=True)
    assert (row["polarity"], row["axis"], row["valence"]) == (-1, "longer life", "desirable")
    assert row["higher_percentile_means"] == "higher score = more hazard of death from any cause"
    assert annotations.filter(pl.col("pgs_id") == "PGS000319")["annotated"].to_list() == [False]
    usage = dict(zip(clusters["phenotype"], clusters["n_scores"], strict=True))
    assert usage == {"all_cause_mortality": 1, "longevity": 0}
    assert (manifest["n_scores"], manifest["n_annotated"], manifest["n_verified"]) == (2, 1, 0)
    assert set(json.loads((out / "curation_manifest.json").read_text())["sha256"]) == {
        "score_annotations.parquet",
        "trait_clusters.parquet",
    }


def test_build_refuses_inconsistent_files(tmp_path: Path) -> None:
    from just_prs.curation import CurationBuildError, write_tables

    _write(tmp_path, {"PGS000318": _entry(1, "All-cause mortality (age at death in females)")})
    with pytest.raises(CurationBuildError, match="polarity"):
        write_tables(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_merge_keeps_published_rows_and_protects_verified(tmp_path: Path) -> None:
    from just_prs.curation import merge_with_published, write_tables

    published_root, local_root = tmp_path / "published", tmp_path / "local"
    verified = {**_entry(-1, "All-cause mortality (age at death in females)"), "status": "human_verified"}
    _write(published_root, {"PGS000318": verified, "PGS000319": {"score_effect": None, "unannotated_reason": "old"}})
    write_tables(published_root, tmp_path / "published_out")
    # This clone has only an unreviewed PGS000318 and nothing about PGS000319.
    _write(local_root, {"PGS000318": {**verified, "status": "agent_proposed", "notes": "local edit"}})
    out = tmp_path / "out"
    write_tables(local_root, out)
    manifest = merge_with_published(out, tmp_path / "published_out")
    merged = pl.read_parquet(out / "score_annotations.parquet")
    assert set(merged["pgs_id"]) == {"PGS000318", "PGS000319"}
    assert merged.filter(pl.col("pgs_id") == "PGS000318")["annotation_status"].to_list() == ["human_verified"]
    assert (manifest["n_kept_from_published"], manifest["n_protected_verified"]) == (1, 1)


def test_merge_rejects_conflicting_cluster_definitions(tmp_path: Path) -> None:
    from just_prs.curation import CurationBuildError, merge_with_published, write_tables

    _write(tmp_path / "published", {"PGS000318": _entry(-1, "All-cause mortality (age at death in females)")})
    write_tables(tmp_path / "published", tmp_path / "published_out")
    _write(tmp_path / "local", {})
    clusters = yaml.safe_load((tmp_path / "local" / "clusters.yaml").read_text())
    clusters["domains"]["aging"]["concepts"]["lifespan"]["valence"] = "neutral"
    (tmp_path / "local" / "clusters.yaml").write_text(yaml.safe_dump(clusters))
    write_tables(tmp_path / "local", tmp_path / "out")
    with pytest.raises(CurationBuildError, match="valence"):
        merge_with_published(tmp_path / "out", tmp_path / "published_out")


def _cli(*args: str) -> str:
    from just_prs.curation import app
    from typer.testing import CliRunner

    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_stage_and_merge_combine_parallel_trait_work(tmp_path: Path) -> None:
    _write(tmp_path, {"PGS000318": _entry(-1, "All-cause mortality (age at death in females)")})
    root = str(tmp_path)
    _cli("stage", "longevity", "--root", root)
    _cli("stage", "mortality-men", "--root", root)
    # Two trait agents add different entries to the same publication in their own copies.
    for slug, pgs_id, sex in (("longevity", "PGS000319", "male"), ("mortality-men", "PGS000319", "male")):
        path = tmp_path / "staging" / slug / "publications" / "PGP000095.yaml"
        doc = yaml.safe_load(path.read_text())
        doc["scores"][pgs_id] = {**_entry(-1, "All-cause mortality (age at death in males)"), "strata": {"sex": sex}}
        path.write_text(yaml.safe_dump(doc))
    _cli("merge", "longevity", "--root", root)
    merged = yaml.safe_load((tmp_path / "publications" / "PGP000095.yaml").read_text())
    assert set(merged["scores"]) == {"PGS000318", "PGS000319"}
    assert not (tmp_path / "staging" / "longevity").exists()
    # The second agent wrote an identical entry: no conflict, nothing lost.
    output = _cli("merge", "mortality-men", "--root", root)
    assert "0 conflicts" in output


def test_merge_reports_conflicting_edits_and_keeps_staging(tmp_path: Path) -> None:
    _write(tmp_path, {"PGS000318": _entry(-1, "All-cause mortality (age at death in females)")})
    root = str(tmp_path)
    _cli("stage", "longevity", "--root", root)
    main_path = tmp_path / "publications" / "PGP000095.yaml"
    main = yaml.safe_load(main_path.read_text())
    main["scores"]["PGS000318"]["notes"] = "edited in main"
    main_path.write_text(yaml.safe_dump(main))
    stage_path = tmp_path / "staging" / "longevity" / "publications" / "PGP000095.yaml"
    staged = yaml.safe_load(stage_path.read_text())
    staged["scores"]["PGS000318"]["notes"] = "edited by trait agent"
    stage_path.write_text(yaml.safe_dump(staged))
    output = _cli("merge", "longevity", "--root", root)
    assert "CONFLICT PGP000095/PGS000318" in output
    assert yaml.safe_load(main_path.read_text())["scores"]["PGS000318"]["notes"] == "edited in main"
    assert (tmp_path / "staging" / "longevity").exists()
