"""The exact per-cell inputs an oracle run uses, for the oracle reference and the generated EA alike.

Two configurations are pre-registered in the S6 design DR:

* ``campaign_cell(symbol, dataset_id)``: the GA campaign's own slow setup
  (``real_backtest.slow_cell_setup``: calibrated hourly mean-bucket spreads,
  matrix no-trade hours, percentile max-spread gate) on the FULL pinned window,
  with ``run_slow_trial``'s BacktestConfig;
* ``battery_cell()``: the 60-DNA parity battery's own config
  (``tests/research/backtest/test_family_kernel_parity.py``): the LAST 800 bars of
  USDCADm ``aed3bf314539ec5a``, no-trade hour 21, the slice's p90 spread gate,
  hourly spreads ``h % 20 + 1``.

``engine_config`` returns the BacktestConfig the oracle runs with, and
``CellConfig`` carries every constant the EA is stamped with. Both come from the
same objects, so the two sides cannot be configured apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.research.backtest.engine import BacktestConfig
from engine.research.costs.slippage import slippage_cap_for_symbol
from engine.research.data.mt5_source import BarData
from engine.research.data.symbol_spec import SymbolSpec
from engine.research.search.real_backtest import (
    load_cell_bars_from_snapshot,
    slow_cell_setup,
    spec_for,
)

BATTERY_SYMBOL = "USDCADm"
BATTERY_DATASET = "aed3bf314539ec5a"
BATTERY_SLICE = 800
BATTERY_NO_TRADE_HOURS = (21,)
BATTERY_HOURLY = tuple(float(h % 20 + 1) for h in range(24))
INITIAL_EQUITY = 10_000.0
SLIPPAGE_MEAN, SLIPPAGE_STD = 0.3, 0.2


@dataclass(frozen=True)
class CellConfig:
    name: str
    kind: str  # "campaign" | "battery"
    symbol: str
    dataset_id: str
    bars: BarData = field(repr=False, compare=False)
    spec: SymbolSpec = field(repr=False, compare=False)
    hourly_spread_points: tuple[float, ...] | None
    no_trade_hours: tuple[int, ...]
    max_spread_points: float | None
    commission_per_lot_per_side: float
    slippage_cap_points: float
    # the campaign's v4 per-bar spread gate and fill cost (SlowCellSetup), so the harness runs what the campaign ran
    # (2026-11-01: missing here after v4 1.1/1.2, the oracle run took 567 trades against the campaign's 936)
    spread_cap_by_bar: np.ndarray | None = field(default=None, repr=False, compare=False)
    fill_spread_points: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def n_bars(self) -> int:
        return int(np.asarray(self.bars.time).size)

    @property
    def first_time(self) -> int:
        return int(self.bars.time[0])

    @property
    def last_time(self) -> int:
        return int(self.bars.time[-1])

    @property
    def replica_names(self) -> tuple[str, str]:
        """The tester replica symbols: low-first (longs) and high-first (shorts); <= 31 chars (MT5)."""
        window = "full" if self.kind == "campaign" else f"last{self.n_bars}"
        stem = f"AS6_{self.symbol}_{self.dataset_id[:6]}_{window}"
        return f"{stem}_L", f"{stem}_H"

    def engine_config(self, strategy_id: str, *, risk_pct: float, notes: str = "") -> BacktestConfig:
        common = dict(
            strategy_id=strategy_id, symbol=self.symbol, timeframe="H1", initial_equity=INITIAL_EQUITY,
            commission_per_lot_per_side=self.commission_per_lot_per_side,
            slippage_mean_points=SLIPPAGE_MEAN, slippage_std_points=SLIPPAGE_STD,
            charge_swap=True, charge_commission=True, no_trade_hours=self.no_trade_hours,
            max_spread_points=self.max_spread_points, hourly_spread_points=self.hourly_spread_points,
        )
        if self.kind == "campaign":  # run_slow_trial, verbatim
            return BacktestConfig(risk_per_trade_pct=float(risk_pct), seed=0, notes=notes,
                                  spread_cap_by_bar=self.spread_cap_by_bar, fill_spread_points=self.fill_spread_points,
                                  **common)
        return BacktestConfig(**common)  # the battery passes only these


def campaign_cell(symbol: str, dataset_id: str) -> CellConfig:
    cell = load_cell_bars_from_snapshot(symbol, "H1", dataset_id)
    setup = slow_cell_setup(cell)
    return CellConfig(
        name=f"{symbol.lower()}_h1_{dataset_id}_full", kind="campaign", symbol=symbol, dataset_id=dataset_id,
        bars=cell.bars, spec=setup.spec, hourly_spread_points=setup.hourly_spread_points,
        no_trade_hours=tuple(setup.no_trade_hours), max_spread_points=setup.max_spread_points,
        commission_per_lot_per_side=float(setup.commission), slippage_cap_points=slippage_cap_for_symbol(symbol),
        spread_cap_by_bar=setup.spread_cap_by_bar, fill_spread_points=setup.fill_spread_points,
    )


def battery_cell() -> CellConfig:
    full = load_cell_bars_from_snapshot(BATTERY_SYMBOL, "H1", BATTERY_DATASET).bars
    s = slice(-BATTERY_SLICE, None)
    bars = BarData(
        time=np.asarray(full.time)[s], open=np.asarray(full.open)[s], high=np.asarray(full.high)[s],
        low=np.asarray(full.low)[s], close=np.asarray(full.close)[s], tick_volume=np.asarray(full.tick_volume)[s],
        spread=np.asarray(full.spread)[s], real_volume=np.asarray(full.real_volume)[s],
        symbol=full.symbol, timeframe=full.timeframe,
    )
    gate = float(np.percentile(np.asarray(bars.spread, dtype=np.float64), 90))
    return CellConfig(
        name=f"usdcadm_h1_{BATTERY_DATASET}_last{BATTERY_SLICE}", kind="battery", symbol=BATTERY_SYMBOL,
        dataset_id=BATTERY_DATASET, bars=bars, spec=spec_for(BATTERY_SYMBOL), hourly_spread_points=BATTERY_HOURLY,
        no_trade_hours=BATTERY_NO_TRADE_HOURS, max_spread_points=gate, commission_per_lot_per_side=0.0,
        slippage_cap_points=slippage_cap_for_symbol(BATTERY_SYMBOL),
    )


__all__ = ["CellConfig", "INITIAL_EQUITY", "SLIPPAGE_MEAN", "SLIPPAGE_STD", "battery_cell", "campaign_cell"]
