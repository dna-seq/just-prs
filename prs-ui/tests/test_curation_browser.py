"""Display frames for the Curated tab, compiled from a tiny curation tree."""

import json
from pathlib import Path

import pytest
import yaml
from just_prs.curation import CurationBuildError

from prs_ui.curation_browser import (
    compile_curation_browser,
    find_curation_root,
    prepare_cluster_display,
    prepare_score_display,
)

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

QUOTE = "All-cause mortality (age at death in females)"


def _entry(status: str = "agent_proposed", confidence: str = "medium") -> dict:
    return {
        "cluster": "aging.lifespan",
        "phenotype": "all_cause_mortality",
        "measurement_kind": "time_to_event",
        "score_effect": {"direction": "increases", "target": "hazard of death from any cause"},
        "polarity": -1,
        "status": status,
        "provenance": [
            {
                "source": "catalog",
                "locator": "performance.trait_reported",
                "quote": QUOTE,
                "confidence": confidence,
            }
        ],
    }


def _write(root: Path) -> None:
    (root / "publications").mkdir(parents=True)
    (root / "clusters.yaml").write_text(yaml.safe_dump(CLUSTERS))
    (root / "publications" / "PGP000095.yaml").write_text(
        yaml.safe_dump(
            {
                "pgp_id": "PGP000095",
                "paper": {"pmid": "12345678", "doi": "10.1000/example"},
                "scores": {
                    "PGS000318": _entry(),
                    "PGS000319": _entry(status="human_verified"),
                    "PGS000320": {
                        "score_effect": None,
                        "unannotated_reason": "The abstract does not state the case definition.",
                    },
                },
            }
        )
    )


def test_find_curation_root_walks_to_clusters_yaml(tmp_path: Path) -> None:
    curation = tmp_path / "curation"
    curation.mkdir()
    (curation / "clusters.yaml").write_text("version: 0\n")
    nested = tmp_path / "prs-ui" / "prs_ui"
    nested.mkdir(parents=True)

    assert find_curation_root(nested) == curation
    assert find_curation_root(curation) == curation


def test_score_display_splits_verified_unreviewed_and_review(tmp_path: Path) -> None:
    _write(tmp_path)
    data = compile_curation_browser(tmp_path)

    assert (data.n_verified, data.n_unreviewed, data.n_review) == (1, 1, 1)
    scores = prepare_score_display(data.tables.annotations, "all")
    by_id = {row["pgs_id"]: row for row in scores.iter_rows(named=True)}
    assert set(by_id) == {"PGS000318", "PGS000319"}
    assert by_id["PGS000318"]["review_status"] == "Unreviewed"
    assert by_id["PGS000319"]["review_status"] == "Human verified"
    assert by_id["PGS000318"]["higher_score_means"] == (
        "higher score = more hazard of death from any cause"
    )
    assert by_id["PGS000318"]["valence"] == "Desirable"
    assert by_id["PGS000318"]["alignment"] == "Opposite of the phenotype"
    assert by_id["PGS000318"]["direction"] == "Increases"
    assert QUOTE in by_id["PGS000318"]["evidence"]
    links = json.loads(by_id["PGS000318"]["sources"])
    assert {"label": "PGS Catalog", "url": "https://www.pgscatalog.org/score/PGS000318/"} in links
    assert {"label": "DOI", "url": "https://doi.org/10.1000/example"} in links

    verified = prepare_score_display(data.tables.annotations, "human_verified")
    assert verified["pgs_id"].to_list() == ["PGS000319"]

    review = prepare_score_display(data.tables.review, "all")
    assert review["pgs_id"].to_list() == ["PGS000320"]
    assert "not state the case definition" in review["review_reason"][0]


def test_cluster_display_counts_resolved_scores_and_axis_sign(tmp_path: Path) -> None:
    _write(tmp_path)
    data = compile_curation_browser(tmp_path)
    clusters = prepare_cluster_display(data.tables.clusters)
    by_phenotype = {row["phenotype"]: row for row in clusters.iter_rows(named=True)}

    mortality = by_phenotype["Death from any cause"]
    assert mortality["n_scores"] == 2
    assert mortality["versus_axis"] == "Opposite of the axis"
    assert mortality["valence"] == "Desirable"
    assert mortality["axis"] == "longer life"
    assert clusters["n_scores"][0] == 2


def test_inconsistent_curation_refuses_to_compile(tmp_path: Path) -> None:
    _write(tmp_path)
    path = tmp_path / "publications" / "PGP000095.yaml"
    payload = yaml.safe_load(path.read_text())
    payload["scores"]["PGS000318"]["polarity"] = 1
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(CurationBuildError, match="polarity"):
        compile_curation_browser(tmp_path)
