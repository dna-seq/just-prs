import os
from pathlib import Path
from typing import Any

import pytest
import reflex.reflex as reflex_runner

from prs_ui import cli


def test_env_port_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRS_UI_PORT", raising=False)

    assert cli._env_port("PRS_UI_PORT") is None


def test_env_port_reads_valid_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRS_UI_PORT", "3000")

    assert cli._env_port("PRS_UI_PORT") == 3000


def test_launch_reflex_passes_explicit_frontend_and_backend_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_run_kwargs: list[dict[str, Any]] = []
    captured_chdir: list[Path] = []

    def fake_run_reflex(**kwargs: Any) -> None:
        captured_run_kwargs.append(kwargs)

    def fake_chdir(path: str | os.PathLike[str]) -> None:
        captured_chdir.append(Path(path))

    monkeypatch.setenv("PRS_UI_HOST", "127.0.0.1")
    monkeypatch.setenv("PRS_UI_PORT", "3000")
    monkeypatch.setenv("PRS_UI_BACKEND_PORT", "8000")
    monkeypatch.delenv("PRS_UI_DATA_DIR", raising=False)
    monkeypatch.setattr(cli, "_check_system_dependencies", lambda: None)
    monkeypatch.setattr(reflex_runner, "_run", fake_run_reflex)
    monkeypatch.setattr(os, "chdir", fake_chdir)

    cli._launch_reflex()

    assert captured_chdir == [Path(cli.__file__).resolve().parent.parent]
    assert captured_run_kwargs[0]["frontend_port"] == 3000
    assert captured_run_kwargs[0]["backend_port"] == 8000
    assert captured_run_kwargs[0]["backend_host"] == "127.0.0.1"
    assert os.environ["PRS_UI_DATA_DIR"] == str(Path.cwd() / "data")


def test_launch_reflex_leaves_unset_ports_for_reflex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_run_kwargs: list[dict[str, Any]] = []

    def fake_run_reflex(**kwargs: Any) -> None:
        captured_run_kwargs.append(kwargs)

    monkeypatch.delenv("PRS_UI_PORT", raising=False)
    monkeypatch.delenv("PRS_UI_BACKEND_PORT", raising=False)
    monkeypatch.setattr(cli, "_check_system_dependencies", lambda: None)
    monkeypatch.setattr(reflex_runner, "_run", fake_run_reflex)
    monkeypatch.setattr(os, "chdir", lambda path: None)

    cli._launch_reflex()

    assert captured_run_kwargs[0]["frontend_port"] is None
    assert captured_run_kwargs[0]["backend_port"] is None


def test_launch_reflex_preserves_explicit_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_run_kwargs: list[dict[str, Any]] = []
    explicit = Path("/tmp/prs-ui-data")

    def fake_run_reflex(**kwargs: Any) -> None:
        captured_run_kwargs.append(kwargs)

    monkeypatch.setenv("PRS_UI_DATA_DIR", str(explicit))
    monkeypatch.setattr(cli, "_check_system_dependencies", lambda: None)
    monkeypatch.setattr(reflex_runner, "_run", fake_run_reflex)
    monkeypatch.setattr(os, "chdir", lambda path: None)

    cli._launch_reflex()

    assert captured_run_kwargs
    assert os.environ["PRS_UI_DATA_DIR"] == str(explicit)


def test_default_data_dir_falls_back_to_invocation_dir(tmp_path: Path) -> None:
    assert cli._default_data_dir(tmp_path) == tmp_path / "data"


def test_launch_preselect_ui_uses_shared_reflex_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[bool] = []

    monkeypatch.delenv("PRS_UI_PRESELECT_ENABLED", raising=False)
    monkeypatch.delenv("PRS_UI_PRESELECT_QUERY", raising=False)
    monkeypatch.delenv("PRS_UI_PRESELECT_VCF", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "_launch_reflex", lambda: launched.append(True))

    cli.launch_preselect_ui()

    assert launched == [True]
    assert os.environ["PRS_UI_PRESELECT_ENABLED"] == "1"
    assert os.environ["PRS_UI_PRESELECT_QUERY"] == "Type 1 diabetes (T1D)"


def test_resolve_port_rejects_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRS_UI_PORT", "abc")

    with pytest.raises(ValueError, match="PRS_UI_PORT must be an integer port"):
        cli._env_port("PRS_UI_PORT")
