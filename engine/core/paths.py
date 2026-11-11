"""Path resolution for ARQVIO_HOME and project root.

WHAT: Single source of truth for where ARQVIO_HOME is.
WHY: Per project rules, ARQVIO_HOME is profile-aware and overridable via env.
HOW: ARQVIO_HOME env var > default (~/<ARQVIO>). All file I/O uses these helpers.
TRADE-OFF: One path resolver; tests can monkeypatch env without touching os.path.
TEST: tests/core/test_paths.py asserts resolution precedence.
FAILURE: Misconfigured ARQVIO_HOME raises ConfigError on access, not silently.
"""
from __future__ import annotations

import os
from pathlib import Path

from engine.core.errors import ConfigError

# Default user-data directory name (under $HOME / %USERPROFILE%).
_HOME_DIR_NAME = "ARQVIO"

# Repo root: ARQVIO/  (parent of engine/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def get_arqvio_home() -> Path:
    """Resolve the ARQVIO user-data home directory.

    Priority:
        1. ARQVIO_HOME env var (must exist or be creatable).
        2. $HOME / %USERPROFILE% / ARQVIO.
    """
    override = os.environ.get("ARQVIO_HOME")
    if override:
        return Path(override).expanduser().resolve()

    home_root = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home_root:
        raise ConfigError(
            "Neither ARQVIO_HOME nor USERPROFILE/HOME is set. Cannot resolve ARQVIO home."
        )
    return Path(home_root).expanduser().resolve() / _HOME_DIR_NAME


def app_mode() -> bool:
    """True inside the installed ARQVIO app (its launcher sets ARQVIO_APP=1).

    The app starts empty and works on the owner's own markets only: the repo's registry cells, whose pinned data
    lives on the developer's machine, are not part of it."""
    return os.environ.get("ARQVIO_APP") == "1"


def ensure_arqvio_home() -> Path:
    """Return ARQVIO_HOME, creating it if it does not exist."""
    home = get_arqvio_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


def get_artifacts_dir(*, create: bool = False) -> Path:
    """Return the writable ARQVIO artefacts directory."""
    path = get_arqvio_home() / "artifacts"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_configs_dir(*, create: bool = False) -> Path:
    """Return the writable ARQVIO configuration directory."""
    path = get_arqvio_home() / "configs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir(*, create: bool = False) -> Path:
    """Return the writable ARQVIO logs directory."""
    path = get_arqvio_home() / "logs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir(*, create: bool = False) -> Path:
    """Return the writable ARQVIO data directory."""
    path = get_arqvio_home() / "data"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "PROJECT_ROOT",
    "ensure_arqvio_home",
    "get_artifacts_dir",
    "get_arqvio_home",
    "get_configs_dir",
    "get_data_dir",
    "get_logs_dir",
]
