import os
from pathlib import Path

import pytest
from reflex.utils import processes

from prs_ui import cli


def test_resolve_reflex_config_lets_reflex_increment_frontend_and_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handled: list[tuple[str, int, bool]] = []

    def fake_handle_port(port_type: str, port: int, auto_increment: bool) -> int:
        handled.append((port_type, port, auto_increment))
        return port + (1 if port_type == "frontend" else 2)

    monkeypatch.setenv("PRS_UI_HOST", "0.0.0.0")
    monkeypatch.setenv("PRS_UI_PORT", "3000")
    monkeypatch.setenv("PRS_UI_BACKEND_PORT", "8000")
    monkeypatch.setattr(processes, "handle_port", fake_handle_port)

    config = cli._resolve_reflex_config()

    assert config.host == "0.0.0.0"
    assert config.frontend_port == 3001
    assert config.backend_port == 8002
    assert handled == [
        ("frontend", 3000, True),
        ("backend", 8000, True),
    ]


def test_launch_reflex_passes_explicit_frontend_and_backend_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_config: list[cli.ReflexServerConfig] = []
    captured_chdir: list[Path] = []

    def fake_run_reflex(config: cli.ReflexServerConfig) -> None:
        captured_config.append(config)

    def fake_chdir(path: str | os.PathLike[str]) -> None:
        captured_chdir.append(Path(path))

    config = cli.ReflexServerConfig(host="0.0.0.0", frontend_port=3000, backend_port=8000)
    monkeypatch.setattr(cli, "_resolve_reflex_config", lambda: config)
    monkeypatch.setattr(cli, "_run_reflex", fake_run_reflex)
    monkeypatch.setattr(os, "chdir", fake_chdir)

    cli._launch_reflex()

    assert captured_chdir == [Path(cli.__file__).resolve().parent.parent]
    assert captured_config == [config]


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


def test_resolve_reflex_config_does_not_reuse_frontend_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handled: list[tuple[str, int, bool]] = []

    def fake_handle_port(port_type: str, port: int, auto_increment: bool) -> int:
        handled.append((port_type, port, auto_increment))
        return port

    monkeypatch.setenv("PRS_UI_PORT", "3000")
    monkeypatch.setenv("PRS_UI_BACKEND_PORT", "3000")
    monkeypatch.setattr(processes, "handle_port", fake_handle_port)

    config = cli._resolve_reflex_config()

    assert config.frontend_port == 3000
    assert config.backend_port == 3001
    assert handled == [
        ("frontend", 3000, True),
        ("backend", 3001, True),
    ]


def test_resolve_port_rejects_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRS_UI_PORT", "abc")

    with pytest.raises(ValueError, match="PRS_UI_PORT must be an integer port"):
        cli._resolve_reflex_config()
