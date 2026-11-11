"""Path resolution."""
from __future__ import annotations

from engine.core.paths import PROJECT_ROOT, ensure_arqvio_home, get_arqvio_home


def test_get_arqvio_home_from_env(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / "arqvio"
    monkeypatch.setenv("ARQVIO_HOME", str(target))
    home = get_arqvio_home()
    assert home == target.resolve()


def test_ensure_arqvio_home_creates_dir(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / "arqvio"
    monkeypatch.setenv("ARQVIO_HOME", str(target))
    home = ensure_arqvio_home()
    assert home.exists()
    assert home.is_dir()


def test_project_root_is_repo() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").exists()
