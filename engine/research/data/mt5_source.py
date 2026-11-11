"""MT5 data source: symbol catalog detection + per-symbol spec snapshot.

Two main entry points:

    detect_symbols(terminal_path) -> list[SymbolInfo]
        Returns the broker's complete symbol catalog with category metadata.
        No data is fetched; this is a fast read of the static catalog.

    snapshot_symbol_spec(terminal_path, name) -> SymbolSpec
        Returns a frozen, JSON-serialisable spec for one symbol. Includes
        profit/loss tick values, lot bounds, swap rates, etc. - everything
        the cost model needs.

    fetch_bars(terminal_path, name, timeframe, date_from, date_to) -> BarData
        Pulls bars from the broker. No data is cached; the warehouse
        module handles persistence.

Why a per-call terminal_path: ARQVIO supports multiple MT5 terminals
(Spec 4.2). Each terminal has its own symbol catalog. A module-level
singleton would couple callers to a single path; a per-call argument makes
the multi-account path explicit and testable.
"""
# ruff: noqa: N813  # `import MetaTrader5 as mt5` is the upstream convention
from __future__ import annotations

import math
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import MetaTrader5 as mt5

from engine.core.errors import DataError
from engine.research.data.symbol_spec import SymbolSpec
from engine.research.data.time_basis import convert_times, probe_basis

if TYPE_CHECKING:
    from numpy import ndarray


class Timeframe(str, Enum):
    """The 9 MT5 timeframes the spec requires."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H2 = "H2"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"

    @property
    def mt5_constant(self) -> int:
        return _MT5_TF[self.value]


_MT5_TF: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}


@dataclass(frozen=True)
class SymbolInfo:
    """Lightweight catalog entry; the per-symbol spec is much heavier."""

    name: str
    path: str
    category: str             # e.g. "Forex", "Crypto", "Indices", "Stocks"
    visible: bool
    digits: int
    point: float
    spread_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "category": self.category,
            "visible": self.visible,
            "digits": self.digits,
            "point": self.point,
            "spread_points": self.spread_points,
        }


@dataclass(frozen=True)
class BarData:
    """Bar data from MT5. Bar ``time`` is a UNIX-second stamp expressed in
    **UTC**: the data layer PROBES the broker's timestamp basis at fetch time
    (R-C0) and converts only if the probe shows a non-UTC server clock. On
    this server the S2 audit (2c458fc) proved ticks encode UTC directly, so
    the probe returns ``server_equals_utc`` and timestamps pass through
    UNCHANGED -- no EET/EEST subtraction. The ``time_basis`` field records
    which basis was measured.

    The ``time_basis`` field records the basis the timestamps were stamped
    in, so consumers that concatenate BarData from different sources can
    refuse a mixed-basis cell rather than silently mis-bucket bars.

    Fields (all numpy arrays, one element per bar):
        time: bar open time (UNIX seconds, **UTC**)
        open, high, low, close: OHLC
        tick_volume: number of ticks that formed the bar
        spread: spread at bar open, in points
        real_volume: trade volume if available (0 for FX)
        time_basis: "server_equals_utc" | "broker_offset" | "unresolved";
            measured at fetch time (R-C0). "server_equals_utc" means the
            broker encodes UTC directly (identity pass-through).
            "broker_offset" means EET/EEST offsets were subtracted.
            "unresolved" is refused. Mixed-basis concatenation is refused
            by the sync layer (G12).
    """

    time: ndarray
    open: ndarray
    high: ndarray
    low: ndarray
    close: ndarray
    tick_volume: ndarray
    spread: ndarray
    real_volume: ndarray
    symbol: str
    timeframe: Timeframe
    source: str = "MT5"
    time_basis: str = "utc"

    @property
    def n_bars(self) -> int:
        if self.time is None:
            return 0
        return int(self.time.size)


_MT5_INIT_TIMEOUT_S = 30.0


def _initialise(terminal_path: str) -> None:
    """Initialise MT5 pointed at a specific terminal. Idempotent.

    mt5.initialize() can block indefinitely if the terminal cannot start;
    this wrapper enforces a timeout so callers get a deterministic error.
    """
    box: list[Any] = [None]

    def worker() -> None:
        try:
            box[0] = mt5.initialize(path=terminal_path)
        except Exception:
            box[0] = None

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=_MT5_INIT_TIMEOUT_S)
    if t.is_alive() or not box[0]:
        err = mt5.last_error()
        raise DataError(
            f"MT5 initialize failed or timed out after {_MT5_INIT_TIMEOUT_S}s at {terminal_path!r}: {err}"
        )


def _shutdown() -> None:
    with suppress(Exception):
        mt5.shutdown()


def detect_symbols(terminal_path: str) -> list[SymbolInfo]:
    """Return the broker's complete symbol catalog.

    Categories are derived from the symbol's MT5 `path` field
    (e.g. "Standard\\\\Forex\\\\EURUSD" -> category "Forex").
    """
    _initialise(terminal_path)
    try:
        syms = mt5.symbols_get()
        if syms is None:
            raise DataError("mt5.symbols_get() returned None")
        out: list[SymbolInfo] = []
        for s in syms:
            parts = (s.path or "").split("\\")
            category = parts[1] if len(parts) >= 2 else "Other"
            out.append(
                SymbolInfo(
                    name=s.name,
                    path=s.path,
                    category=category,
                    visible=bool(s.visible),
                    digits=int(s.digits),
                    point=float(s.point),
                    spread_points=int(s.spread),
                )
            )
        return out
    finally:
        _shutdown()


def detect_symbols_by_category(
    terminal_path: str, category: str
) -> list[SymbolInfo]:
    """Convenience wrapper to filter the catalog by category."""
    return [s for s in detect_symbols(terminal_path) if s.category == category]


def snapshot_symbol_spec(terminal_path: str, name: str) -> SymbolSpec:
    """Take a frozen snapshot of one symbol's full contract parameters.

    Reads BOTH trade_tick_value_profit and trade_tick_value_loss separately
    (Exness-specific). Falls back to trade_tick_value if either is missing
    or non-finite, with a log-worthy comment in the docstring of the caller
    if that happens.
    """
    _initialise(terminal_path)
    try:
        info = mt5.symbol_info(name)
        if info is None:
            raise DataError(f"symbol {name!r} not found in catalog at {terminal_path!r}")
        # Some broker attributes are 0.0 on non-Forex (e.g. crypto tick_size).
        # Validate them; do not fabricate values.
        if not math.isfinite(info.trade_tick_size) or info.trade_tick_size <= 0:
            raise DataError(
                f"symbol {name!r}: trade_tick_size is invalid "
                f"({info.trade_tick_size}); cannot snapshot"
            )
        if not math.isfinite(info.trade_contract_size) or info.trade_contract_size <= 0:
            raise DataError(
                f"symbol {name!r}: trade_contract_size is invalid "
                f"({info.trade_contract_size}); cannot snapshot"
            )
        # Profit/loss tick values - may be equal (most brokers) or different
        # (Exness). Prefer the dedicated fields; fall back to trade_tick_value.
        tv_profit = float(
            getattr(info, "trade_tick_value_profit", info.trade_tick_value) or 0.0
        )
        tv_loss = float(
            getattr(info, "trade_tick_value_loss", info.trade_tick_value) or 0.0
        )
        if not math.isfinite(tv_profit) or tv_profit == 0:
            tv_profit = float(info.trade_tick_value)
        if not math.isfinite(tv_loss) or tv_loss == 0:
            tv_loss = float(info.trade_tick_value)
        return SymbolSpec(
            name=info.name,
            path=info.path,
            visible=bool(info.visible),
            digits=int(info.digits),
            point=float(info.point),
            contract_size=float(info.trade_contract_size),
            trade_tick_size=float(info.trade_tick_size),
            trade_tick_value=float(info.trade_tick_value),
            trade_tick_value_profit=tv_profit,
            trade_tick_value_loss=tv_loss,
            spread_points=int(info.spread),
            currency_base=info.currency_base,
            currency_profit=info.currency_profit,
            volume_min=float(info.volume_min) if info.volume_min > 0 else 0.01,
            volume_step=float(info.volume_step) if info.volume_step > 0 else 0.01,
            volume_max=float(info.volume_max) if info.volume_max > 0 else 100.0,
            swap_long=float(info.swap_long),
            swap_short=float(info.swap_short),
            # Exness charges weekend roll on Wednesday; most brokers do too.
            # MT5 does not expose this directly, so the convention is hard-coded
            # here. When MT5 eventually exposes it, read it from the broker.
            swap_rollover_triple_day=2,
            freeze_level=int(info.freeze_level) if hasattr(info, "freeze_level") else 0,
            stop_level=int(info.stop_level) if hasattr(info, "stop_level") else 0,
            snapshot_at_utc=datetime.now(tz=UTC).isoformat(),
            source=f"MT5:{terminal_path}",
        )
    finally:
        _shutdown()


def fetch_ticks(
    terminal_path: str,
    name: str,
    date_from: datetime,
    date_to: datetime,
) -> Any:
    """Read UTC ticks for research; never sends orders or changes terminal settings."""
    if date_from >= date_to:
        raise DataError("date_from must be < date_to")
    _initialise(terminal_path)
    try:
        ticks = mt5.copy_ticks_range(name, date_from, date_to, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            raise DataError(f"no ticks returned for {name}: {mt5.last_error()}")
        return ticks
    finally:
        _shutdown()


def fetch_bars(
    terminal_path: str,
    name: str,
    timeframe: Timeframe,
    date_from: datetime,
    date_to: datetime,
) -> BarData:
    """Pull a contiguous bar range from MT5. Naive datetimes are treated as UTC."""
    if date_from >= date_to:
        raise DataError(
            f"date_from must be < date_to; got {date_from} >= {date_to}"
        )
    if timeframe not in Timeframe:
        raise DataError(f"unsupported timeframe: {timeframe!r}")
    _initialise(terminal_path)
    try:
        probe = probe_basis(mt5, name)
        rates = mt5.copy_rates_range(name, timeframe.mt5_constant, date_from, date_to)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            raise DataError(
                f"no bars returned for {name} {timeframe.value} "
                f"from {date_from} to {date_to}: {err}"
            )
        # mt5.copy_rates_range returns a structured numpy array; unpack it.
        # The probe (R-C0) classifies the timestamp basis at fetch time:
        # server_equals_utc passes timestamps through unchanged; broker_offset
        # subtracts the measured EET/EEST delta. On this server the S2 audit
        # proved ticks are UTC directly, so convert_times is identity here.
        raw_time = rates["time"].astype("int64")
        utc_time = convert_times(raw_time, probe)
        return BarData(
            time=utc_time,
            open=rates["open"].astype("float64"),
            high=rates["high"].astype("float64"),
            low=rates["low"].astype("float64"),
            close=rates["close"].astype("float64"),
            tick_volume=rates["tick_volume"].astype("int64"),
            spread=rates["spread"].astype("int32"),
            real_volume=rates["real_volume"].astype("int64"),
            symbol=name,
            timeframe=timeframe,
            source=f"MT5:{terminal_path}",
            time_basis=probe.basis,
        )
    finally:
        _shutdown()
