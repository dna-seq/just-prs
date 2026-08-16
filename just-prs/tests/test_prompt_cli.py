"""CLI tests for ``prs prompt`` — the agent-facing Ask-AI prompt emitter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from just_prs.cli import app, _load_prompt_results
from just_prs.viz import ai_prefill_url, assistant_char_limit


def test_load_prompt_results_groups_sample_mapping(tmp_path: Path) -> None:
    path = tmp_path / "family.json"
    path.write_text(
        json.dumps(
            {
                "Anton": [{"pgs_id": "PGS000001", "percentile": 72.0, "match_rate": 0.9}],
                "Livia": [{"pgs_id": "PGS000001", "percentile": 41.0, "match_rate": 0.88}],
            }
        )
    )
    first, multi = _load_prompt_results(path)
    assert multi is not None
    assert set(multi) == {"Anton", "Livia"}
    assert first[0]["pgs_id"] == "PGS000001"


def test_load_prompt_results_groups_sample_field(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(
        json.dumps(
            [
                {"pgs_id": "PGS000001", "percentile": 72.0, "sample": "Anton"},
                {"pgs_id": "PGS000001", "percentile": 41.0, "sample_name": "Livia"},
            ]
        )
    )
    _, multi = _load_prompt_results(path)
    assert multi is not None
    assert set(multi) == {"Anton", "Livia"}


def test_prompt_cli_from_results_json_stdout(tmp_path: Path) -> None:
    path = tmp_path / "family.json"
    path.write_text(
        json.dumps(
            {
                "Anton": [
                    {
                        "pgs_id": "PGS000001",
                        "percentile": 72.0,
                        "match_rate": 0.9,
                        "quality_label": "High",
                    }
                ],
                "Livia": [
                    {
                        "pgs_id": "PGS000001",
                        "percentile": 41.0,
                        "match_rate": 0.88,
                        "quality_label": "High",
                    }
                ],
            }
        )
    )
    result = CliRunner().invoke(app, ["prompt", "intelligence", "--results", str(path)])
    assert result.exit_code == 0, result.output
    assert "across 2 samples" in result.stdout
    assert "Anton=72.0" in result.stdout
    assert "Livia=41.0" in result.stdout
    assert "Interpret these combined" in result.stdout
    # Progress belongs on stderr so agents can pipe stdout into another model.
    assert "Interpret these combined" not in result.stderr


def test_prompt_cli_url_mode(tmp_path: Path) -> None:
    path = tmp_path / "one.json"
    path.write_text(
        json.dumps(
            [{"pgs_id": "PGS000001", "percentile": 58.0, "match_rate": 0.8, "quality_label": "High"}]
        )
    )
    result = CliRunner().invoke(
        app,
        ["prompt", "BMI", "--results", str(path), "--assistant", "claude", "--url"],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("https://claude.ai/new?q=")
    assert "Interpret these combined" not in result.stdout


def test_prompt_cli_rejects_url_for_other() -> None:
    result = CliRunner().invoke(app, ["prompt", "BMI", "--results", "x.json", "--url"])
    assert result.exit_code == 1
    assert "paste-only" in result.output


def test_prompt_cli_sequential_invokes_do_not_close_console(tmp_path: Path) -> None:
    """Progress redirect must not pin Rich's Console to a closed CliRunner stream."""
    path = tmp_path / "one.json"
    path.write_text(json.dumps([{"pgs_id": "PGS000001", "percentile": 58.0, "match_rate": 0.8}]))
    runner = CliRunner()
    first = runner.invoke(app, ["prompt", "BMI", "--results", str(path)])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, ["prompt", "BMI", "--results", "x.json", "--url"])
    assert second.exit_code == 1
    assert "paste-only" in second.output


def test_assistant_char_limit_and_prefill_url() -> None:
    assert assistant_char_limit("claude") == 6000
    assert assistant_char_limit("chatgpt") == 3000
    url = ai_prefill_url("claude", "hello world")
    assert url.startswith("https://claude.ai/new?q=")
    with pytest.raises(ValueError, match="no prefill URL"):
        ai_prefill_url("other", "hello")
