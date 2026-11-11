"""Where the MT5-side files are, resolved at CALL time from the one resolved terminal.

The terminal executable comes from ``mt5_terminal.require_terminal`` (env ->
config -> discovery). Everything else hangs off it:

* MetaEditor is the sibling ``MetaEditor64.exe`` of that terminal;
* the terminal's data directory is the ``%APPDATA%\\MetaQuotes\\Terminal\\<hash>``
  whose ``origin.txt`` (UTF-16) names the terminal's install directory;
* the Common files directory (``FILE_COMMON`` in MQL5) is
  ``%APPDATA%\\MetaQuotes\\Terminal\\Common\\Files``; ARQVIO's subtree there is
  ``ARQVIO_S6``.

Nothing is cached and nothing is a module constant: a path that does not exist
raises ``TerminalPathError`` naming what was looked for.
"""
from __future__ import annotations

import os
from pathlib import Path

from engine.research.data.mt5_terminal import require_terminal

ARQVIO_SUBDIR = "ARQVIO_S6"


class TerminalPathError(FileNotFoundError):
    """An MT5-side path that should exist and does not."""


def terminal_exe() -> Path:
    return Path(require_terminal())


def metaeditor_exe() -> Path:
    exe = terminal_exe().parent / "MetaEditor64.exe"
    if not exe.is_file():
        raise TerminalPathError(f"MetaEditor64.exe not found next to {terminal_exe()}")
    return exe


def _terminals_root() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise TerminalPathError("APPDATA is not set; cannot locate the MetaQuotes terminal directories")
    return Path(appdata) / "MetaQuotes" / "Terminal"


def data_dir() -> Path:
    """The data directory whose origin.txt names the resolved terminal's install directory."""
    install = terminal_exe().parent.resolve()
    root = _terminals_root()
    matches = []
    for child in sorted(root.iterdir()) if root.is_dir() else ():
        origin = child / "origin.txt"
        if not origin.is_file():
            continue
        named = origin.read_bytes().decode("utf-16").strip().lstrip("﻿")
        if Path(named).resolve() == install:
            matches.append(child)
    if len(matches) != 1:
        raise TerminalPathError(f"{len(matches)} data directories under {root} name {install} in origin.txt")
    return matches[0]


def mql5_dir() -> Path:
    d = data_dir() / "MQL5"
    if not d.is_dir():
        raise TerminalPathError(f"no MQL5 directory in {data_dir()}")
    return d


def tester_data_dir() -> Path:
    """The Strategy Tester's own data directory for this terminal (its synced history copy lives here)."""
    return _terminals_root().parent / "Tester" / data_dir().name


def common_files_dir() -> Path:
    d = _terminals_root() / "Common" / "Files"
    if not d.is_dir():
        raise TerminalPathError(f"no Common\\Files directory at {d}")
    return d


def arqvio_files_dir() -> Path:
    """Common\\Files\\ARQVIO_S6, created on first use (MQL5 FILE_COMMON reads it)."""
    d = common_files_dir() / ARQVIO_SUBDIR
    d.mkdir(exist_ok=True)
    return d


__all__ = [
    "ARQVIO_SUBDIR",
    "TerminalPathError",
    "arqvio_files_dir",
    "common_files_dir",
    "data_dir",
    "metaeditor_exe",
    "mql5_dir",
    "terminal_exe",
    "tester_data_dir",
]
