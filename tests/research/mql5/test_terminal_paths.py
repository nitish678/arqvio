"""MT5-side paths resolve at call time from the resolved terminal, or fail by name."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.research.mql5 import terminal as term


def _fake_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, origins: list[str]) -> Path:
    install = tmp_path / "mt5" / "1"
    install.mkdir(parents=True)
    (install / "terminal64.exe").write_bytes(b"")
    (install / "MetaEditor64.exe").write_bytes(b"")
    appdata = tmp_path / "AppData"
    root = appdata / "MetaQuotes" / "Terminal"
    for i, origin in enumerate(origins):
        d = root / f"HASH{i}"
        (d / "MQL5").mkdir(parents=True)
        (d / "origin.txt").write_bytes(origin.encode("utf-16"))
    (root / "Common" / "Files").mkdir(parents=True)
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("ARQVIO_MT5_TERMINAL", str(install / "terminal64.exe"))
    return install


def test_the_data_dir_is_the_one_whose_origin_names_the_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 isolated_arqvio_home: Path) -> None:
    install = _fake_install(tmp_path, monkeypatch, origins=[str(tmp_path / "elsewhere"), str(tmp_path / "mt5" / "1")])
    assert term.terminal_exe() == install / "terminal64.exe"
    assert term.metaeditor_exe() == install / "MetaEditor64.exe"
    assert term.data_dir().name == "HASH1"
    assert term.mql5_dir() == term.data_dir() / "MQL5"
    d = term.arqvio_files_dir()
    assert d.is_dir() and d.parent == term.common_files_dir() and d.name == "ARQVIO_S6"


def test_no_matching_or_two_matching_data_dirs_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           isolated_arqvio_home: Path) -> None:
    _fake_install(tmp_path, monkeypatch, origins=[str(tmp_path / "elsewhere")])
    with pytest.raises(term.TerminalPathError, match="0 data directories"):
        term.data_dir()


def test_two_data_dirs_naming_the_same_terminal_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            isolated_arqvio_home: Path) -> None:
    same = str(tmp_path / "mt5" / "1")
    _fake_install(tmp_path, monkeypatch, origins=[same, same])
    with pytest.raises(term.TerminalPathError, match="2 data directories"):
        term.data_dir()


def test_a_missing_metaeditor_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       isolated_arqvio_home: Path) -> None:
    install = _fake_install(tmp_path, monkeypatch, origins=[str(tmp_path / "mt5" / "1")])
    (install / "MetaEditor64.exe").unlink()
    with pytest.raises(term.TerminalPathError, match="MetaEditor64.exe not found"):
        term.metaeditor_exe()
