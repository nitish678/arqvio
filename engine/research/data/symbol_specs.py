"""Persistent store of MT5 symbol contract specs.

Until S1 the backtester knew two symbols, hardcoded in
``real_backtest._DEFAULT_SPECS``. Everything else raised
``DataError: no SymbolSpec known for ...`` -- which is why XAUUSDm, the
one calibrated symbol with a zero-spread hour, could never reach the
engine and F1's no-trade window had no production cell to fire on.

The values here are a SNAPSHOT of what the broker reported at a moment
in time, not constants. Contract size, tick value and swap rates change,
and a backtest costed from the wrong ones is wrong in a way no
statistical gate can see. So each file records when it was taken, from
which account and server, and nothing in this module invents a value it
could not read.

Specs live in ``get_artifacts_dir()/specs/{SYMBOL}.json`` -- ARQVIO_HOME,
not the repo. They are data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.core.errors import DataError
from engine.core.paths import get_artifacts_dir
from engine.research.data.symbol_spec import SymbolSpec

SPECS_DIRNAME = "specs"

# MT5 reports trade_stops_level in POINTS. A broker that reports 0 means
# "no minimum distance enforced", which is not the same as "stops may sit
# arbitrarily close to price". Documented policy: fall back to a 2-pip
# floor and WARN, so a stop that could never be placed live cannot be
# filled in simulation.
STOPS_LEVEL_FALLBACK_PIPS = 2.0


def mt5_dow_to_python_weekday(mt5_dow: int) -> int:
    """MT5 ENUM_DAY_OF_WEEK -> Python ``date.weekday()``.

    These are NOT the same numbering and the difference is silent:

        MT5     SUNDAY=0 MONDAY=1 TUESDAY=2 WEDNESDAY=3 ... SATURDAY=6
        Python  MONDAY=0 TUESDAY=1 ...                      SUNDAY=6

    costs/swap.py:91 compares the triple-swap day against
    ``d.weekday()``, so storing MT5's raw value would charge the 3x roll
    one day late -- Thursday's triple on Friday. The hardcoded specs
    this store replaces were already in Python's convention (their
    comment read "0=Mon ... 6=Sun"), so converting here keeps the engine
    reading one numbering throughout.
    """
    return (int(mt5_dow) - 1) % 7


def specs_dir() -> Path:
    return get_artifacts_dir() / SPECS_DIRNAME


@dataclass(frozen=True)
class SpecValidation:
    """What was wrong with a snapshot, if anything."""

    symbol: str
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def validate_snapshot(data: dict[str, Any]) -> SpecValidation:
    """Refuse a spec that cannot produce correct P&L.

    The three multiplicative inputs must be positive: a zero tick size
    divides, and a zero tick value silently zeroes every trade's P&L
    rather than failing.
    """
    symbol = str(data.get("name", "?"))
    errors: list[str] = []
    warnings: list[str] = []

    for field in ("trade_tick_size", "trade_contract_size"):
        value = data.get(field)
        if value is None or float(value) <= 0.0:
            errors.append(f"{field}={value!r} must be > 0")

    for field in ("trade_tick_value_profit", "trade_tick_value_loss"):
        value = data.get(field)
        if value is None or float(value) <= 0.0:
            errors.append(f"{field}={value!r} must be > 0")

    stops = data.get("trade_stops_level")
    if stops is not None and int(stops) == 0:
        warnings.append(
            f"trade_stops_level=0; applying the documented "
            f"{STOPS_LEVEL_FALLBACK_PIPS:g}-pip fallback floor"
        )

    return SpecValidation(
        symbol=symbol,
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def effective_stops_level_points(data: dict[str, Any]) -> int:
    """Minimum stop distance in POINTS, with the documented fallback."""
    stops = int(data.get("trade_stops_level") or 0)
    if stops > 0:
        return stops
    digits = int(data.get("digits") or 0)
    # A pip is 10 points on a 3- or 5-digit quote, 1 point otherwise.
    points_per_pip = 10 if digits in (3, 5) else 1
    return int(STOPS_LEVEL_FALLBACK_PIPS * points_per_pip)


def save_snapshot(data: dict[str, Any], target_dir: Path | None = None) -> Path:
    """Write one symbol's snapshot. Refuses an invalid one."""
    check = validate_snapshot(data)
    if not check.ok:
        raise DataError(
            f"refusing to persist {check.symbol}: {'; '.join(check.errors)}"
        )
    directory = specs_dir() if target_dir is None else target_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data['name']}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_snapshot(
    symbol: str, target_dir: Path | None = None,
) -> dict[str, Any] | None:
    """The raw stored snapshot, or None when the symbol has none."""
    directory = specs_dir() if target_dir is None else target_dir
    path = directory / f"{symbol}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def to_symbol_spec(data: dict[str, Any]) -> SymbolSpec:
    """Build the engine's SymbolSpec from a stored snapshot.

    trade_tick_value is carried through for compatibility; the engine
    uses the profit/loss pair per leg.
    """
    check = validate_snapshot(data)
    if not check.ok:
        raise DataError(
            f"{check.symbol} snapshot is unusable: {'; '.join(check.errors)}"
        )
    return SymbolSpec(
        name=str(data["name"]),
        path=str(data.get("path", "")),
        visible=bool(data.get("visible", True)),
        digits=int(data["digits"]),
        point=float(data["point"]),
        contract_size=float(data["trade_contract_size"]),
        trade_tick_size=float(data["trade_tick_size"]),
        trade_tick_value=float(
            data.get("trade_tick_value", data["trade_tick_value_profit"])
        ),
        trade_tick_value_profit=float(data["trade_tick_value_profit"]),
        trade_tick_value_loss=float(data["trade_tick_value_loss"]),
        spread_points=int(data.get("spread", 0) or 0),
        currency_base=str(data.get("currency_base", "")),
        currency_profit=str(data.get("currency_profit", "")),
        volume_min=float(data["volume_min"]),
        volume_step=float(data["volume_step"]),
        volume_max=float(data["volume_max"]),
        swap_long=float(data.get("swap_long", 0.0)),
        swap_short=float(data.get("swap_short", 0.0)),
        # MT5 enum -> Python weekday; see mt5_dow_to_python_weekday.
        swap_rollover_triple_day=mt5_dow_to_python_weekday(
            data.get("swap_rollover3days", 3)
        ),
        freeze_level=int(data.get("trade_freeze_level", 0) or 0),
        stop_level=effective_stops_level_points(data),
        snapshot_at_utc=str(data.get("snapshot_utc", "")),
        source=str(data.get("source", "MT5")),
    )


def load_spec(symbol: str, target_dir: Path | None = None) -> SymbolSpec | None:
    """The engine's spec for a symbol, or None when not ingested."""
    data = load_snapshot(symbol, target_dir)
    if data is None:
        return None
    return to_symbol_spec(data)


def available_symbols(target_dir: Path | None = None) -> tuple[str, ...]:
    directory = specs_dir() if target_dir is None else target_dir
    if not directory.is_dir():
        return ()
    return tuple(sorted(p.stem for p in directory.glob("*.json")))


def snapshot_from_mt5_symbol_info(
    info: Any, *, account_login: int | None, server: str | None,
) -> dict[str, Any]:
    """Convert an ``mt5.symbol_info()`` result into a storable snapshot.

    Every field is read from the terminal. Nothing is defaulted except
    the provenance stamps, which describe the read itself.
    """
    return {
        "name": str(info.name),
        "path": str(getattr(info, "path", "")),
        "visible": bool(getattr(info, "visible", True)),
        "digits": int(info.digits),
        "point": float(info.point),
        "spread": int(getattr(info, "spread", 0) or 0),
        "currency_base": str(getattr(info, "currency_base", "")),
        "currency_profit": str(getattr(info, "currency_profit", "")),
        "trade_contract_size": float(info.trade_contract_size),
        "trade_tick_size": float(info.trade_tick_size),
        "trade_tick_value": float(getattr(info, "trade_tick_value", 0.0)),
        "trade_tick_value_profit": float(info.trade_tick_value_profit),
        "trade_tick_value_loss": float(info.trade_tick_value_loss),
        "trade_stops_level": int(getattr(info, "trade_stops_level", 0) or 0),
        "trade_freeze_level": int(getattr(info, "trade_freeze_level", 0) or 0),
        "volume_min": float(info.volume_min),
        "volume_max": float(info.volume_max),
        "volume_step": float(info.volume_step),
        "swap_long": float(getattr(info, "swap_long", 0.0)),
        "swap_short": float(getattr(info, "swap_short", 0.0)),
        "swap_mode": int(getattr(info, "swap_mode", 0) or 0),
        "swap_rollover3days": int(getattr(info, "swap_rollover3days", 3)),
        "snapshot_utc": datetime.now(tz=UTC).isoformat(),
        "account_login": account_login,
        "server": server,
        "source": "MT5:exness",
    }


__all__ = [
    "STOPS_LEVEL_FALLBACK_PIPS",
    "mt5_dow_to_python_weekday",
    "SpecValidation",
    "available_symbols",
    "effective_stops_level_points",
    "load_snapshot",
    "load_spec",
    "save_snapshot",
    "snapshot_from_mt5_symbol_info",
    "specs_dir",
    "to_symbol_spec",
    "validate_snapshot",
]
