"""CLI tests for prs-pipeline commands that must not launch heavy jobs."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from prs_pipeline import cli


runner = CliRunner()


def test_ld_proxy_refuses_without_scope() -> None:
    result = runner.invoke(cli.app, ["ld-proxy"])

    assert result.exit_code == 2
    assert "Refusing to launch" in result.output


def test_ld_proxy_limit_sets_pgs_batch_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_execvp(_file: str, _args: list[str]) -> None:
        raise RuntimeError("launch stopped")

    monkeypatch.setattr(cli, "_setup_dagster_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "_set_pipeline_env", lambda panel, no_cache=False: False)
    monkeypatch.setattr(cli, "_kill_port", lambda port: None)
    monkeypatch.setattr(cli, "_cancel_orphaned_runs", lambda: None)
    monkeypatch.setattr(cli.os, "execvp", _fake_execvp)
    monkeypatch.delenv("PRS_LD_FULL_CATALOG", raising=False)
    monkeypatch.delenv("PRS_LD_PGS_IDS", raising=False)
    monkeypatch.delenv("PRS_LD_LIMIT_TARGETS", raising=False)

    result = runner.invoke(cli.app, ["ld-proxy", "--limit", "5"])

    assert result.exit_code == 1
    assert "launch stopped" in str(result.exception)
    assert cli.os.environ["PRS_LD_LIMIT_TARGETS"] == "5"
    assert "PRS_LD_FULL_CATALOG" not in cli.os.environ
    assert "PRS_LD_PGS_IDS" not in cli.os.environ
