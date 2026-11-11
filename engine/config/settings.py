"""Centralized configuration via Pydantic Settings.

WHAT: Typed settings with env override + ARQVIO_HOME / .env file support.
WHY: One source of truth for engine config. Secrets stay in .env (gitignored) or OS env.
HOW: pydantic-settings BaseSettings; prefix 'ARQVIO_'. Defaults are safe (demo, no live).
TRADE-OFF: All Stage-1 settings are simple types. Future stages will add nested models.
TEST: tests/config/test_settings.py asserts precedence and defaults.
FAILURE: Invalid values raise ValidationError on app start, not silently.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from engine.core.paths import get_arqvio_home

# Load .env from ARQVIO_HOME if present, then from project root.
# Order matters: ARQVIO_HOME/.env takes priority (so installs override the repo).
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent


def _bootstrap_dotenv() -> None:
    """Load .env files in priority order. Does not overwrite existing env vars."""
    home = os.environ.get("ARQVIO_HOME")
    if home:
        load_dotenv(Path(home) / ".env", override=False)
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    """ARQVIO engine settings.

    Env override: prefix `ARQVIO_` (e.g. ARQVIO_API_PORT=9000).
    """

    model_config = SettingsConfigDict(
        env_prefix="ARQVIO_",
        env_file=None,  # we handle .env manually so order is explicit
        case_sensitive=False,
        extra="ignore",
    )

    # --- Paths ---
    arqvio_home: Path = Field(default_factory=get_arqvio_home)

    # --- Logging ---
    log_level: str = Field(default="INFO", description="DEBUG | INFO | WARNING | ERROR")

    # --- API ---
    api_host: str = Field(default="127.0.0.1", description="Loopback only by default")
    api_port: int = Field(default=8741, ge=1, le=65535)

    # --- Live safety defaults (Stage 1: read-only, not yet enforced) ---
    demo_mode: bool = Field(default=True)
    live_trading_enabled: bool = Field(default=False)

    # --- Engine ---
    engine_version: str = Field(default="4.0.0")

    # --- Execution cost model (A4/A5) ---
    # Commission per LOT per SIDE, in account currency. Default 0.0 and
    # it stays 0.0 until the operator confirms their account type: an
    # Exness Standard account is spread-only, a Raw account is not, and
    # assuming either would put a made-up number into every trade.
    # Per-symbol overrides go in commission_per_lot_per_side_by_symbol.
    commission_per_lot_per_side: float = Field(default=0.0, ge=0.0)
    commission_per_lot_per_side_by_symbol: dict[str, float] = Field(
        default_factory=dict
    )

    # Entry slippage cap, in POINTS, by asset class. The half-normal
    # draw has no upper bound, so without a cap one tail sample can
    # charge an entry hundreds of points. These two defaults are the
    # sanctioned values from the spec text; nothing else is invented.
    slippage_cap_points_fx: float = Field(default=10.0, gt=0.0)
    slippage_cap_points_metals: float = Field(default=30.0, gt=0.0)

    # Max-spread entry gate. The threshold is DATA-DRIVEN -- the p95 of
    # the symbol's own calibrated spread distribution -- so it adapts to
    # the instrument instead of hardcoding a pip count. Set a value here
    # to override for a symbol.
    max_spread_gate_percentile: float = Field(default=95.0, gt=0.0, le=100.0)
    max_spread_points_by_symbol: dict[str, float] = Field(default_factory=dict)
    # v4 (ARQVIO_V4.md 10.1): "regime" judges each bar against the median spread of its own previous
    # spread_regime_days (engine/research/costs/spread_regime.py); "fixed" is the p95 gate above. An operator
    # override in max_spread_points_by_symbol is a fixed cap and wins.
    spread_gate_mode: str = Field(default="regime", pattern="^(regime|fixed)$")
    spread_regime_days: int = Field(default=20, ge=1, le=365)
    spread_regime_multiple: float = Field(default=3.0, gt=0.0)
    # v4 (ARQVIO_V4.md 10.1 cost): "bar" charges each fill the signal candle's own historical spread and the
    # calibrated hour bucket only where a bar has none; "hourly" is the W0 hour-bucket cost everywhere.
    spread_cost_mode: str = Field(default="bar", pattern="^(bar|hourly)$")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance. .env is loaded on first call only."""
    _bootstrap_dotenv()
    return Settings()
