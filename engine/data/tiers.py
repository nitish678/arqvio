"""Data tier classification.

Three tiers, each with its own provenance and storage path:

  TIER_1   bundled inside the installer; D1 + H4 + spread matrices +
           symbol specs + time offsets. Read-only.
  TIER_2   optional signed .zip (H1 + H2) extracted into
           ``%APPDATA%/ARQVIO/data/tier2/``. Lazily downloaded.
  TIER_3   incremental MT5 sync into
           ``%APPDATA%/ARQVIO/data/tier3/{symbol}/{timeframe}/{year}.parquet``.
           Pulled from ``last_stored_bar_time`` forward; never a full
           re-pull.

M15 / M5 are NEVER bundled. They are pulled per-symbol per-user-request
only, into the tier3 location with a `user_requested=true` flag.

Ticks are NEVER bundled. They live in a temporary calibration workspace
and are deleted after the hourly spread matrix is written.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class DataTier(str, Enum):
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"
    UNAVAILABLE = "UNAVAILABLE"

    @property
    def description(self) -> str:
        return {
            DataTier.TIER_1: "Bundled in the installer (D1 + H4 + spread matrices + specs).",
            DataTier.TIER_2: "Optional signed data pack (H1 + H2).",
            DataTier.TIER_3: "Incremental MT5 sync (last_stored_bar_time forward).",
            DataTier.UNAVAILABLE: "No data available for this cell.",
        }[self]


# Timeframes that the installer ships with. Any other timeframe requires
# an explicit per-user pull.
TIER_1_TIMEFRAMES: frozenset[str] = frozenset({"D1", "H4"})
TIER_2_TIMEFRAMES: frozenset[str] = frozenset({"H1", "H2"})
TIER_3_TIMEFRAMES: frozenset[str] = frozenset({"M5", "M15", "M30", "H1", "H2", "H4", "D1"})

# Cells the campaign builder will suggest *by default*. Anything outside
# this list requires an opt-in.
DEFAULT_CAMPAIGN_TIMEFRAMES: tuple[str, ...] = ("D1", "H4", "H2", "H1", "M30", "M15")


def _appdata() -> Path:
    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / "ARQVIO"
    return Path.home() / ".arqvio"


def app_data_dir() -> Path:
    """The user-mutable ARQVIO data directory. Created on first access."""
    path = _appdata() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tier1_path() -> Path:
    """Read-only bundled tier 1 root (next to the engine binary)."""
    return Path(__file__).resolve().parents[2] / "data_tier1"


def tier2_path() -> Path:
    return app_data_dir() / "tier2"


def tier3_path(symbol: str, timeframe: str, year: int) -> Path:
    safe_symbol = symbol.replace("/", "_").replace("\\", "_")
    return (
        app_data_dir() / "tier3" / safe_symbol / timeframe / f"{int(year)}.parquet"
    )


def tier3_root(symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace("\\", "_")
    return app_data_dir() / "tier3" / safe_symbol / timeframe


def tick_calibration_workspace(symbol: str) -> Path:
    """Ephemeral directory for a tick pull. Delete after the spread matrix
    is written - ticks are never bundled."""
    safe_symbol = symbol.replace("/", "_").replace("\\", "_")
    path = app_data_dir() / "calibration_workspace" / safe_symbol
    path.mkdir(parents=True, exist_ok=True)
    return path
