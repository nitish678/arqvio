"""Per-cell precompute for the fast path.

Everything the kernel needs that depends only on (cell, config) and not
on the candidate, computed once instead of 250 times. The slow engine
recomputes several of these per bar per candidate; the profile put 88%
of a trial inside on_bar, and half of that in four np.mean slices that
a rolling sum makes O(1).

THE SLOW ENGINE IS THE SPECIFICATION. Every array here is lifted from
what engine.py actually does, with the line cited. Where the S2 brief
described something the oracle does not do, the oracle wins and the
difference is recorded below rather than silently reconciled.

TWO PLACES THE ORACLE DIFFERS FROM THE BRIEF
--------------------------------------------

1. NO DST, ANYWHERE -- hour label is now asserted UTC.
   engine.py still derives everything from the raw bar timestamp:

       bar_hour_index = (bar_time // 3600) % 24   engine.py:341
       day_index      = bar_time // 86400         engine.py:389, 491
       friday         = day_index % 7 == 1        engine.py:217

   engine.py does NOT import time_offset (asserted by
   test_prep_matches_engine_inputs.test_no_dst_offset_is_applied_anywhere);
   it buckets on whatever stamp it is handed. W0.2 closes the G12 gap at
   the DATA layer: mt5_source.fetch_bars and real_backtest.load_cell_bars
   now run rates["time"] through broker_to_utc before constructing BarData,
   so bar_time handed to the engine is genuinely UTC, not accidentally-UTC.

   The G12 audit finding (Exness server clock == UTC) is preserved as the
   REASON the conversion is a no-op for Exness today, not as an assumption
   the code relies on: a non-UTC broker now shifts correctly. Fields here
   remain bar_hour_index / day_index (not utc_hour) because the bucket is
   derived from the stamp, and the name records that the derivation is
   the contract, not the broker's clock.

2. SPREAD COST IS HOURLY, NOT FLAT (W0). The S2 brief asks for
   "spread_pts per bar from the calibrated artifact's hour buckets".
   W0 closes that gap: when a SpreadMatrix is available, the engine and
   kernel both charge spread from the hour's bucket (the kernel reads
   fill_spread_price[i], the bucket of bar i's hour) instead of the flat
   spec.spread_in_price(). v4 (ARQVIO_V4.md 10.1): with fill_spread_points
   a fill charges the signal candle's own historical spread, and the hour's
   bucket only where that spread is unknown (NaN).
   The per-bar spread column
   is still used ONLY for the gate comparison (engine.py:459), while the
   hourly cost vector feeds the fill cost.

   When no matrix is present (or hourly_spread_points is None), the
   flat spec.spread_in_price() is used at every hour, repeated across
   all 24 buckets -- the pre-W0 behaviour, so cells without a matrix
   are unaffected.

   G13 -- retained as a MODELLING NOTE, not a parity concern. The
   calibrated artifact holds a 24-bucket hourly profile per symbol.
   When hourly_spread_points is set (W0 active), each fill charges the
   bucket for that bar's UTC hour; when it is None the flat snapshot
   spread applies everywhere.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from engine.research.backtest.kernel import (
    FAM_DONCHIAN_BREAKOUT,
    FAM_MA_CROSS,
    FAM_MACD_CROSS,
    FAM_RSI_REVERSION,
    FAM_SIGNALS,
    FAM_STOCH_CROSS,
)
from engine.research.costs.spread_regime import gate_excess
from engine.research.data.symbol_spec import SymbolSpec
from engine.research.features.atr import (
    DEFAULT_ATR_PERIOD,
    sma_atr_series,
    true_range_series,
)

# 1970-01-01 was a Thursday, so day index 1 is a Friday.
# Mirrors engine.py:217 _is_friday.
_FRIDAY_DAY_MOD = 1
_EPOCH = date(1970, 1, 1)


@dataclass(frozen=True)
class PreppedCell:
    """Candidate-independent inputs to the kernel.

    Arrays are all length n_bars and aligned to the bar index the slow
    engine loops over.
    """

    symbol: str
    timeframe: str
    n_bars: int

    # --- price, straight from BarData ------------------------------------
    open: np.ndarray            # float64
    high: np.ndarray            # float64
    low: np.ndarray             # float64
    close: np.ndarray           # float64

    # --- time, matching engine.py:341 / 389 / 217 ------------------------
    # NOT called utc_hour on purpose (G12). See the module docstring.
    bar_hour_index: np.ndarray  # int64, 0..23 -- see G12 note
    day_index: np.ndarray       # int64, bar_time // 86400
    is_friday: np.ndarray       # bool

    # --- windows and gates ------------------------------------------------
    no_trade_mask: np.ndarray   # bool, hour in no_trade_hours (F1)
    bar_spread_points: np.ndarray   # float64, BarData.spread -- GATE ONLY (regime gate: spread - cap)
    max_spread_points: float    # p95 threshold, 0.0 for the regime gate, or NaN when the gate is off

    # --- indicators -------------------------------------------------------
    atr14: np.ndarray           # float64, NaN for the first 13 bars (G9)

    # --- swap schedule ----------------------------------------------------
    # Multiplier per day index, 3 on the triple day and 1 otherwise,
    # matching swap.py:91 (d.weekday() == triple_dow). Indexed by
    # day_index - day_index_min.
    swap_multiplier: np.ndarray     # int64
    day_index_min: int

    # --- scalars the kernel needs ----------------------------------------
    point: float
    digits: int
    spread_cost_one_way: float      # account ccy per lot, FLAT (see note 2)
    # W0: per-hour one-way spread cost per lot. When the caller supplies
    # hourly_spread_points (W0), this is the hourly vector; otherwise it
    # is the flat cost repeated 24 times. The kernel indexes by
    # bar_hour_index[i] so the slow and fast paths agree per-bar.
    hourly_spread_cost_per_lot: np.ndarray  # float64, shape (24,)
    # The same 24 buckets as a PRICE (points * point, or the flat
    # spec.spread_in_price()). The kernel charges price * volume *
    # per-unit in the oracle's multiplication order (engine.py:808,
    # costs/spread.py); scaling a per-lot cost by volume afterwards
    # differs from it in the last bit on about a third of fills.
    # Per bar: the price charged on a fill at that bar -- the bucket of its hour, or (v4) the signal candle's own
    # spread * point where fill_spread_points has one. Precomputed so the kernel never branches on the source.
    fill_spread_price: np.ndarray  # float64, shape (n_bars,)
    # W0: where the hourly cost vector came from -- "mean_buckets" (a
    # calibrated SpreadMatrix), "flat" (no matrix; spec.spread_in_price
    # repeated), or "none" (no spread cost at all); v4 "bar_spread+<that>"
    # when fills charge the bar's own spread first. Recorded so the
    # report can state the source rather than letting a reader infer it.
    spread_cost_source: str
    # W0: 16-hex digest of hourly_spread_cost_per_lot, so a reader can
    # confirm the cost model has not drifted between runs without
    # comparing 24 floats by eye.
    cost_vector_hash: str
    per_unit_profit: float          # tick_value_profit / trade_tick_size
    per_unit_loss: float            # tick_value_loss  / trade_tick_size
    swap_long_points: float
    swap_short_points: float
    slippage_cap_points: float
    volume_min: float
    volume_max: float
    volume_step: float

    # --- v4 switches: the bar label and tick volume (signal_switches / switch_gates read them) ---------
    time: np.ndarray            # int64, the bars' own time
    tick_volume: np.ndarray     # float64, BarData.tick_volume

    def bar_count(self) -> int:
        return int(self.n_bars)


def _swap_multiplier_by_day(
    day_min: int, day_max: int, triple_dow: int,
) -> np.ndarray:
    """3 on the triple-swap weekday, 1 otherwise, per day index.

    swap.py:91 compares ``d.weekday() == triple_dow`` where d is the
    roll date. The roll dates are open_d+1 .. close_d inclusive
    (swap.py _nights_between), so a multiplier keyed on the day index
    reproduces the per-night charge exactly.
    """
    n = int(day_max - day_min) + 1
    out = np.ones(max(n, 0), dtype=np.int64)
    for k in range(out.size):
        d = _EPOCH + timedelta(days=int(day_min + k))
        if d.weekday() == int(triple_dow):
            out[k] = 3
    return out


def prep_cell(
    bars,
    spec: SymbolSpec,
    *,
    no_trade_hours: tuple[int, ...] = (),
    max_spread_points: float | None = None,
    hourly_spread_points: tuple[float, ...] | None = None,
    slippage_cap_points: float | None = None,
    atr_period: int = DEFAULT_ATR_PERIOD,
    spread_cap_by_bar: np.ndarray | None = None,
    fill_spread_points: np.ndarray | None = None,
) -> PreppedCell:
    """Build the kernel's candidate-independent inputs for one cell.

    ``spread_cap_by_bar`` (the v4 regime gate) makes the gate input ``spread - cap`` with threshold 0.0 -- the
    oracle's ``spread[i] > cap[i]`` exactly; callers then pass ``max_spread_points=0.0`` to run_fast."""
    from engine.research.costs.slippage import slippage_cap_for_symbol
    from engine.research.costs.spread import spread_cost

    time = np.asarray(bars.time, dtype=np.int64)
    n = int(time.size)

    # engine.py:341 -- raw bar-label hour, no broker offset (G12).
    bar_hour_index = (time // 3600) % 24
    # engine.py:389 / 491 -- raw bar-label day index.
    day_index = time // 86400
    # engine.py:217 -- Friday is day index % 7 == 1.
    is_friday = (day_index % 7) == _FRIDAY_DAY_MOD

    blocked_hours = frozenset(int(h) for h in no_trade_hours)
    no_trade_mask = np.zeros(n, dtype=np.bool_)
    if blocked_hours:
        for h in blocked_hours:
            no_trade_mask |= bar_hour_index == h

    # G9: SMA of true range, NaN across the 13-bar warmup.
    close = np.asarray(bars.close, dtype=np.float64)
    high = np.asarray(bars.high, dtype=np.float64)
    low = np.asarray(bars.low, dtype=np.float64)
    atr14 = sma_atr_series(
        true_range_series(high, low, close), period=atr_period
    )

    day_min = int(day_index.min()) if n else 0
    day_max = int(day_index.max()) if n else 0
    swap_multiplier = _swap_multiplier_by_day(
        day_min, day_max, spec.swap_rollover_triple_day
    )

    # Note 2: FLAT one-way spread cost per lot (spread_cost at 1.0 lot).
    # Reported and hashed; the kernel charges from fill_spread_price.
    spread_one_way_per_lot = float(spread_cost(spec, 1.0).one_way)

    # W0: per-hour spread, as a cost per lot (hashed into the manifest) and
    # as a PRICE (charged). The kernel computes price * volume * per_unit in
    # engine.py:808 / costs/spread.py:63's order; scaling a per-lot cost by
    # volume instead differs in the last bit on about a third of fills.
    # Without hourly_spread_points the flat value fills all 24 buckets so
    # the kernel can always index by bar_hour_index[i].
    per_unit_per_lot = (
        max(spec.trade_tick_value_profit, spec.trade_tick_value_loss) / spec.trade_tick_size
    )
    if hourly_spread_points is not None:
        hourly = np.array(
            [h * spec.point * per_unit_per_lot for h in hourly_spread_points],
            dtype=np.float64,
        )
        hourly_price = np.array(
            [h * spec.point for h in hourly_spread_points], dtype=np.float64,
        )
        spread_cost_source = "mean_buckets"
    else:
        hourly = np.full(24, spread_one_way_per_lot, dtype=np.float64)
        hourly_price = np.full(24, spec.spread_in_price(), dtype=np.float64)
        spread_cost_source = "flat"
    # engine.py _simulate_fill: the bar's own spread * point when it has one, else the hour's bucket price.
    fill_price = hourly_price[bar_hour_index]
    digest = np.round(hourly, decimals=12).tobytes()
    if fill_spread_points is not None:
        pts = np.asarray(fill_spread_points, dtype=np.float64)
        if pts.size != n:
            raise ValueError(f"fill_spread_points has {pts.size} values for {n} bars")
        own = ~np.isnan(pts)
        fill_price = np.where(own, pts * spec.point, fill_price)
        spread_cost_source = f"bar_spread+{spread_cost_source}"
        digest += np.round(fill_price, decimals=12).tobytes()
    cost_vector_hash = hashlib.sha256(digest).hexdigest()[:16]

    return PreppedCell(
        symbol=spec.name,
        timeframe=str(getattr(bars, "timeframe", "")),
        n_bars=n,
        open=np.asarray(bars.open, dtype=np.float64),
        high=high,
        low=low,
        close=close,
        bar_hour_index=bar_hour_index.astype(np.int64),
        day_index=day_index.astype(np.int64),
        is_friday=is_friday,
        no_trade_mask=no_trade_mask,
        bar_spread_points=(
            np.asarray(bars.spread, dtype=np.float64) if spread_cap_by_bar is None
            else gate_excess(bars.spread, spread_cap_by_bar)
        ),
        max_spread_points=(
            0.0 if spread_cap_by_bar is not None
            else float("nan") if max_spread_points is None
            else float(max_spread_points)
        ),
        atr14=atr14,
        swap_multiplier=swap_multiplier,
        day_index_min=day_min,
        point=float(spec.point),
        digits=int(spec.digits),
        spread_cost_one_way=spread_one_way_per_lot,
        hourly_spread_cost_per_lot=hourly,
        fill_spread_price=np.ascontiguousarray(fill_price, dtype=np.float64),
        spread_cost_source=spread_cost_source,
        cost_vector_hash=cost_vector_hash,
        # engine.py:215-217 -- per-leg tick values (W3).
        per_unit_profit=spec.trade_tick_value_profit / spec.trade_tick_size,
        per_unit_loss=spec.trade_tick_value_loss / spec.trade_tick_size,
        swap_long_points=float(spec.swap_long),
        swap_short_points=float(spec.swap_short),
        slippage_cap_points=(
            float(slippage_cap_points) if slippage_cap_points is not None
            else slippage_cap_for_symbol(spec.name)
        ),
        volume_min=float(spec.volume_min),
        volume_max=float(spec.volume_max),
        volume_step=float(spec.volume_step),
        time=time,
        tick_volume=np.asarray(bars.tick_volume, dtype=np.float64),
    )


@dataclass(frozen=True)
class FamilyArrays:
    """Per-candidate indicator input to the kernel's family branch.

    ``ind_a``/``ind_b`` are the ORACLE's own indicator series on the full
    cell -- RSI (a only), %K/%D, MACD/signal, Donchian upper/lower --
    computed by the very functions the strategy calls. Each is causal, so
    its value at t equals the strategy's value on the visible window [0, t].
    ``warmup`` is the bar before which the strategy's on_bar is silent.
    """

    family: int
    ind_a: np.ndarray
    ind_b: np.ndarray
    warmup: int
    thr_lo: float
    thr_hi: float
    midline: bool


_EMPTY = np.zeros(0, dtype=np.float64)

# MA_CROSS computes its means in the kernel; it carries no series.
MA_CROSS_ARRAYS = FamilyArrays(FAM_MA_CROSS, _EMPTY, _EMPTY, 0, np.nan, np.nan, False)


def family_key(cfg) -> tuple:
    """Memo key for family_arrays: the config type plus every field the
    indicator and gates read. Never an object identity."""
    from engine.research.backtest.strategies.donchian_breakout import DonchianBreakoutConfig
    from engine.research.backtest.strategies.macd_cross import MACDCrossConfig
    from engine.research.backtest.strategies.rsi import RSIConfig
    from engine.research.backtest.strategies.stoch_cross import StochCrossConfig

    if isinstance(cfg, RSIConfig):
        return ("RSI", cfg.period, cfg.oversold, cfg.overbought, bool(cfg.exit_midline))
    if isinstance(cfg, StochCrossConfig):
        return ("STOCH", cfg.k_period, cfg.d_period, cfg.smoothing,
                cfg.k_oversold, cfg.k_overbought)
    if isinstance(cfg, MACDCrossConfig):
        return ("MACD", cfg.fast_period, cfg.slow_period, cfg.signal_period)
    if isinstance(cfg, DonchianBreakoutConfig):
        return ("DONCHIAN", cfg.period)
    from engine.research.backtest.strategies.signals import SignalsConfig

    if isinstance(cfg, SignalsConfig):
        return ("SIGNALS", tuple(cfg.switches))
    raise TypeError(f"no kernel family for config {type(cfg).__name__}")


def family_arrays(prepped: PreppedCell, cfg) -> FamilyArrays:
    """Compute the family series for one config, once, on the full cell.

    Sequential O(n) indicator passes (Wilder RMA, EMA, rolling windows),
    strictly causal; NaN warmup is carried through and the kernel's
    finiteness checks mirror the strategy's, so warmup never signals.
    """
    from engine.research.backtest.strategies.donchian_breakout import (
        DonchianBreakoutConfig,
        donchian_channels,
    )
    from engine.research.backtest.strategies.macd_cross import MACDCrossConfig, macd_series
    from engine.research.backtest.strategies.rsi import RSIConfig
    from engine.research.backtest.strategies.stoch_cross import StochCrossConfig, stoch_series
    from engine.research.features.rsi import rsi_series

    if isinstance(cfg, RSIConfig):
        return FamilyArrays(
            FAM_RSI_REVERSION, rsi_series(prepped.close, period=cfg.period), _EMPTY,
            int(cfg.period), float(cfg.oversold), float(cfg.overbought),
            bool(cfg.exit_midline),
        )
    if isinstance(cfg, StochCrossConfig):
        k, d = stoch_series(
            prepped.high, prepped.low, prepped.close,
            k_period=cfg.k_period, d_period=cfg.d_period, smoothing=cfg.smoothing,
        )
        return FamilyArrays(
            FAM_STOCH_CROSS, k, d, int(cfg.k_period + cfg.d_period + cfg.smoothing - 2),
            float(cfg.k_oversold), float(cfg.k_overbought), False,
        )
    if isinstance(cfg, MACDCrossConfig):
        line, signal, _ = macd_series(
            prepped.close, fast_period=cfg.fast_period,
            slow_period=cfg.slow_period, signal_period=cfg.signal_period,
        )
        return FamilyArrays(
            FAM_MACD_CROSS, line, signal, int(cfg.slow_period + cfg.signal_period - 1),
            np.nan, np.nan, False,
        )
    if isinstance(cfg, DonchianBreakoutConfig):
        upper, lower = donchian_channels(prepped.high, prepped.low, cfg.period)
        return FamilyArrays(
            FAM_DONCHIAN_BREAKOUT, upper, lower, int(cfg.period), np.nan, np.nan, False,
        )
    from engine.research.backtest.signal_switches import signal_arrays
    from engine.research.backtest.strategies.signals import WARMUP, SignalsConfig

    if isinstance(cfg, SignalsConfig):
        # the same classes the oracle steps, run over the whole cell
        long_s, short_s = signal_arrays(prepped.time, prepped.open, prepped.high, prepped.low, prepped.close,
                                        tuple(cfg.switches), volume=prepped.tick_volume)
        return FamilyArrays(FAM_SIGNALS, long_s, short_s, int(WARMUP), np.nan, np.nan, False)
    raise TypeError(f"no kernel family for config {type(cfg).__name__}")


__all__ = [
    "MA_CROSS_ARRAYS",
    "FamilyArrays",
    "PreppedCell",
    "family_arrays",
    "family_key",
    "prep_cell",
]
