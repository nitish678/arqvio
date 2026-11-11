"""Where the MT5 terminal actually is.

``load_cell_bars`` defaulted to a hardcoded ``C:\\mt5\\1\\terminal64.exe``,
and ``server.py``'s health probe had the same string written a second time
with different escaping. On any machine that installed MetaTrader anywhere
else, both silently reported "MT5 closed" - the terminal was running and the
path was wrong, and nothing distinguished those two facts.

Resolution order, first hit wins:

1. ``ARQVIO_MT5_TERMINAL`` in the environment. An operator override, and the
   only thing that can point at a terminal in an unusual place.
2. ``configs/mt5.json`` -> ``{"terminal_path": "..."}``. Written by the UI
   when the operator picks one.
3. Discovery: the portable layout this project uses (``C:\\mt5\\<n>\\``),
   then the standard per-user MetaQuotes install, then Program Files.

A path that does not exist on disk is never returned. When nothing is found
the answer is ``None`` - "not found" is reported as itself, never as a
plausible default that then fails to connect for an unexplained reason.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from engine.core.paths import get_configs_dir

ENV_VAR = "ARQVIO_MT5_TERMINAL"
CONFIG_PATH = get_configs_dir() / "mt5.json"
EXE_NAME = "terminal64.exe"


@dataclass(frozen=True)
class TerminalResolution:
    """Where the terminal was found, and how. The 'how' is shown in the UI."""

    path: str | None
    source: str  # ENV | CONFIG | DISCOVERED | NOT_FOUND
    candidates_checked: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return self.path is not None

    def display(self) -> str:
        if self.path is None:
            return "MT5 TERMINAL NOT FOUND"
        return f"{self.path} ({self.source})"


def _portable_candidates() -> list[Path]:
    """The portable layout this project uses: C:\\mt5\\1, C:\\mt5\\2, ..."""
    root = Path("C:/mt5")
    if not root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            out.append(child / EXE_NAME)
    return out


def _installed_candidates() -> list[Path]:
    """Standard installs, per-user first."""
    out: list[Path] = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        terminals = Path(appdata) / "MetaQuotes" / "Terminal"
        if terminals.is_dir():
            for child in sorted(terminals.iterdir()):
                if child.is_dir():
                    out.append(child / EXE_NAME)
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if not base:
            continue
        program_files = Path(base)
        if not program_files.is_dir():
            continue
        for child in sorted(program_files.glob("*MetaTrader*")):
            out.append(child / EXE_NAME)
    return out


def resolve_terminal(*, config_path: Path | None = None) -> TerminalResolution:
    """Find the MT5 terminal, or report that there isn't one."""
    checked: list[str] = []

    env_value = os.environ.get(ENV_VAR)
    if env_value:
        checked.append(env_value)
        if Path(env_value).is_file():
            return TerminalResolution(env_value, "ENV", tuple(checked))

    cfg = CONFIG_PATH if config_path is None else config_path
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            configured = data.get("terminal_path")
        except (json.JSONDecodeError, OSError):
            configured = None
        if configured:
            checked.append(str(configured))
            if Path(configured).is_file():
                return TerminalResolution(str(configured), "CONFIG", tuple(checked))

    for candidate in _portable_candidates() + _installed_candidates():
        checked.append(str(candidate))
        if candidate.is_file():
            return TerminalResolution(str(candidate), "DISCOVERED", tuple(checked))

    return TerminalResolution(None, "NOT_FOUND", tuple(checked))


def save_terminal_path(path: str, *, config_path: Path | None = None) -> None:
    """Persist an operator's choice. Refuses a path that is not there."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} is not a file")
    cfg = CONFIG_PATH if config_path is None else config_path
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"terminal_path": path}, indent=2), encoding="utf-8")


def require_terminal(*, config_path: Path | None = None) -> str:
    """The resolved path, or an error naming everywhere that was looked."""
    res = resolve_terminal(config_path=config_path)
    if res.path is None:
        looked = "\n  ".join(res.candidates_checked) or "(nowhere - no candidates)"
        raise FileNotFoundError(
            f"MT5 terminal not found. Set {ENV_VAR} or write "
            f"{CONFIG_PATH} with a terminal_path. Checked:\n  {looked}"
        )
    return res.path


__all__ = [
    "CONFIG_PATH",
    "ENV_VAR",
    "TerminalResolution",
    "require_terminal",
    "resolve_terminal",
    "save_terminal_path",
]
