"""Real backtest wiring for the search runner.

This module is the bridge between the DNA space and the S3 event-driven
backtester. It:

  1. Loads the per-cell bar data (real MT5 data) into a BarData object.
  2. Builds a SymbolSpec + BacktestConfig from the cell + DNA.
  3. Maps the DNA to an MA-crossover strategy (the only strategy family
     in S3; expanding to more gene families is post-S5 work).
  4. Runs the BacktestEngine and extracts:
       - per-period returns (from equity_curve)  -- used by PBO/CSCV
       - realised Sharpe                         -- used by DSR
       - n_trades, total_return, max_dd          -- used by the funnel
  5. Fails LOUDLY if the backtest function is called while the bar
     data is the synthetic placeholder (zero-edge generator), per
     Maalik's 2026-09-17 instruction.

The wrapper is also import-safe: if MT5 is not reachable it raises
DataError so the caller can fall back to a synthetic test, never
silently mixing the two.
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import numpy as np

from engine.config.settings import get_settings
from engine.core.errors import DataError
from engine.core.paths import get_artifacts_dir
from engine.research.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
)
from engine.research.backtest.strategies import Family
from engine.research.backtest.strategies.donchian_breakout import (
    DonchianBreakoutConfig,
    DonchianBreakoutStrategy,
)
from engine.research.backtest.strategies.ma_crossover import (
    MACrossConfig,
    MACrossStrategy,
)
from engine.research.backtest.strategies.macd_cross import (
    MACDCrossConfig,
    MACDCrossStrategy,
)
from engine.research.backtest.strategies.rsi import (
    RSIConfig,
    RSIStrategy,
)
from engine.research.backtest.strategies.signals import SignalsConfig, SignalsStrategy
from engine.research.backtest.strategies.switch import ENGINE_MGMT, close_rule_kwargs, stop_rule_kwargs
from engine.research.backtest.strategies.stoch_cross import (
    StochCrossConfig,
    StochCrossStrategy,
)
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.data.mt5_terminal import require_terminal
from engine.research.data.symbol_spec import SymbolSpec
from engine.research.data.time_basis import convert_times, probe_basis
from engine.research.registry.annualisation import bars_per_year
from engine.research.search.dna import DNA, FailureTaxonomy
from engine.research.search.runner import TrialResult

# The terminal path is resolved at call time, never baked in. This was
# MT5_TERMINAL = r"C:\mt5\1\terminal64.exe", so every machine with
# MetaTrader installed anywhere else reported "MT5 closed" while the
# terminal was running -- a wrong path and a closed terminal were the
# same observation. See engine/research/data/mt5_terminal.py.

_MT5_INIT_TIMEOUT_S = 30.0


def _initialize_mt5(mt5: Any, terminal_path: str) -> bool:
    """Initialize MT5 with a hard timeout ceiling.

    The MetaTrader5 module's initialize() can block indefinitely if the
    terminal cannot start or reach a login. This wrapper enforces a
    timeout so callers get a deterministic DataError instead of a hang.
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
    if t.is_alive():
        return False
    return bool(box[0])


# ---------------------------------------------------------------------------
# SymbolSpec construction
# ---------------------------------------------------------------------------

_DEFAULT_SPECS: dict[str, SymbolSpec] = {
    "USDCADm": SymbolSpec(
        name="USDCADm",
        path="Standard\\Forex\\USDCADm",
        visible=True,
        digits=5,
        point=1e-5,
        contract_size=100_000,
        trade_tick_size=1e-5,
        trade_tick_value=0.717891,
        trade_tick_value_profit=0.717891,
        trade_tick_value_loss=0.717963,
        spread_points=14,
        currency_base="USD",
        currency_profit="CAD",
        volume_min=0.01,
        volume_step=0.01,
        volume_max=100.0,
        swap_long=-0.5,
        swap_short=-0.4,
        swap_rollover_triple_day=2,
        freeze_level=0,
        stop_level=0,
        snapshot_at_utc="2026-09-17T00:00:00+00:00",
        source="MT5:exness",
    ),
    "GBPJPYm": SymbolSpec(
        name="GBPJPYm",
        path="Standard\\Forex\\GBPJPYm",
        visible=True,
        digits=3,
        point=0.001,
        contract_size=100_000,
        trade_tick_size=0.001,
        trade_tick_value=0.667,
        trade_tick_value_profit=0.667,
        trade_tick_value_loss=0.667,
        spread_points=20,
        currency_base="GBP",
        currency_profit="JPY",
        volume_min=0.01,
        volume_step=0.01,
        volume_max=100.0,
        swap_long=-0.6,
        swap_short=-0.5,
        swap_rollover_triple_day=2,
        freeze_level=0,
        stop_level=0,
        snapshot_at_utc="2026-09-17T00:00:00+00:00",
        source="MT5:exness",
    ),
}


def spec_for(symbol: str) -> SymbolSpec:
    """The symbol's contract spec, from the persistent store.

    W3. This used to read _DEFAULT_SPECS, a two-symbol hardcoded table,
    and raise for everything else -- which is why XAUUSDm, the one
    calibrated symbol with a zero-spread hour, could never reach the
    engine and F1's window had no production cell to fire on.

    The store wins over the table wherever both have a symbol. The live
    snapshot disagreed with the hardcoded values on tick value, volume
    max and BOTH swap fields for USDCADm and GBPJPYm, so preferring the
    table would mean costing real trades from numbers the broker
    stopped reporting. The table stays only as a fallback for a machine
    with no ingest yet, and says so when it is used.
    """
    from engine.research.data.symbol_specs import load_spec

    spec = load_spec(symbol)
    if spec is not None:
        return spec
    if symbol in _DEFAULT_SPECS:
        return _DEFAULT_SPECS[symbol]
    raise DataError(
        f"no SymbolSpec for {symbol!r}: nothing in "
        f"{get_artifacts_dir() / 'specs'} and no hardcoded fallback. "
        "Run scripts/ingest_symbol_specs.py with the MT5 terminal open."
    )


# ---------------------------------------------------------------------------
# DNA -> strategy config
# ---------------------------------------------------------------------------

def dna_to_ma_config(dna: DNA) -> MACrossConfig:
    """Map DNA fields to MACrossConfig. The only S3 family is MA-crossover.

    fast = max(5, int(dna.max_feature_lookback / 4))
    slow = max(fast + 5, int(dna.max_feature_lookback))
    volume = float(dna.risk_pct)  # risk_pct interpreted as lot fraction
    """
    fast = max(5, int(dna.max_feature_lookback / 4))
    slow = max(fast + 5, int(dna.max_feature_lookback))
    volume = max(0.01, min(1.0, float(dna.risk_pct) * 10.0))
    # W1: sl_atr / tp_atr reach the engine. They were dropped here,
    # which is why the C5 fixture found five distinct dna_hashes all
    # producing byte-identical results: the genes were sampled, hashed
    # and stored, and then discarded one function before they could
    # change anything. The strategy_id carries them too, so two DNAs
    # with different stops no longer collide on strategy identity.
    sl_atr = max(0.0, float(dna.sl_atr))
    tp_atr = max(0.0, float(dna.tp_atr))
    return MACrossConfig(
        fast=fast,
        slow=slow,
        volume=volume,
        strategy_id=f"ma_cross_{fast}_{slow}_sl{sl_atr:.4f}_tp{tp_atr:.4f}",
        sl_atr=sl_atr,
        tp_atr=tp_atr,
    )


def dna_to_rsi_config(dna: DNA) -> RSIConfig:
    """Map DNA fields to RSIConfig for the RSI_REVERSION family.

    B2: the period gene (rsi_period, sampled in [5, 21]) drives both the
    strategy and the rsi_14 feature's lookback. The feature registry's
    rsi_14 is registered with lookback = period + 1, so the gene-varying
    period is resolved to the RSIConfig at dispatch time and the slow engine
    computes RSI directly from rsi_series with that period.
    """
    period = int(dna.rsi_period)
    sl_atr = max(0.0, float(dna.sl_atr))
    tp_atr = max(0.0, float(dna.tp_atr))
    volume = max(0.01, min(1.0, float(dna.risk_pct) * 10.0))
    oversold = float(dna.rsi_oversold)
    overbought = float(dna.rsi_overbought)
    exit_midline = bool(dna.rsi_exit_midline)
    return RSIConfig(
        period=period,
        oversold=oversold,
        overbought=overbought,
        exit_midline=exit_midline,
        volume=volume,
        strategy_id=(
            f"rsi_{period}_os{oversold:.0f}_ob{overbought:.0f}"
            f"{'_midline' if exit_midline else ''}"
            f"_sl{sl_atr:.4f}_tp{tp_atr:.4f}"
        ),
        sl_atr=sl_atr,
        tp_atr=tp_atr,
    )


def dna_to_stoch_config(dna: DNA) -> StochCrossConfig:
    """Map DNA fields to StochCrossConfig for the STOCH_CROSS family."""
    sl_atr = max(0.0, float(dna.sl_atr))
    tp_atr = max(0.0, float(dna.tp_atr))
    volume = max(0.01, min(1.0, float(dna.risk_pct) * 10.0))
    stoch_oversold = float(dna.stoch_oversold)
    stoch_overbought = float(dna.stoch_overbought)
    return StochCrossConfig(
        k_period=int(dna.stoch_k_period),
        d_period=int(dna.stoch_d_period),
        smoothing=int(dna.stoch_smoothing),
        k_oversold=stoch_oversold,
        k_overbought=stoch_overbought,
        volume=volume,
        strategy_id=(
            f"stoch_{dna.stoch_k_period}_{dna.stoch_d_period}_"
            f"sm{dna.stoch_smoothing}_os{stoch_oversold:.0f}_"
            f"ob{stoch_overbought:.0f}_sl{sl_atr:.4f}_tp{tp_atr:.4f}"
        ),
        sl_atr=sl_atr,
        tp_atr=tp_atr,
    )


def dna_to_macd_config(dna: DNA) -> MACDCrossConfig:
    """Map DNA fields to MACDCrossConfig for the MACD_CROSS family."""
    sl_atr = max(0.0, float(dna.sl_atr))
    tp_atr = max(0.0, float(dna.tp_atr))
    volume = max(0.01, min(1.0, float(dna.risk_pct) * 10.0))
    return MACDCrossConfig(
        fast_period=int(dna.macd_fast),
        slow_period=int(dna.macd_slow),
        signal_period=int(dna.macd_signal_period),
        volume=volume,
        strategy_id=(
            f"macd_{dna.macd_fast}_{dna.macd_slow}_"
            f"sig{dna.macd_signal_period}_sl{sl_atr:.4f}_tp{tp_atr:.4f}"
        ),
        sl_atr=sl_atr,
        tp_atr=tp_atr,
    )


def dna_to_signals_config(dna: DNA) -> SignalsConfig:
    """Map DNA fields to SignalsConfig for the SIGNALS family (v4 batch 2.6)."""
    sl_atr = max(0.0, float(dna.sl_atr))
    tp_atr = max(0.0, float(dna.tp_atr))
    volume = max(0.01, min(1.0, float(dna.risk_pct) * 10.0))
    return SignalsConfig(switches=tuple(dna.signal_switches), volume=volume,
                         strategy_id=f"signals_{'+'.join(dna.signal_switches)}_sl{sl_atr:.4f}_tp{tp_atr:.4f}",
                         sl_atr=sl_atr, tp_atr=tp_atr)


def dna_to_donchian_config(dna: DNA) -> DonchianBreakoutConfig:
    """Map DNA fields to DonchianBreakoutConfig for the DONCHIAN_BREAKOUT family."""
    sl_atr = max(0.0, float(dna.sl_atr))
    tp_atr = max(0.0, float(dna.tp_atr))
    volume = max(0.01, min(1.0, float(dna.risk_pct) * 10.0))
    return DonchianBreakoutConfig(
        period=int(dna.donchian_period),
        volume=volume,
        strategy_id=(
            f"donchian_{dna.donchian_period}_sl{sl_atr:.4f}_tp{tp_atr:.4f}"
        ),
        sl_atr=sl_atr,
        tp_atr=tp_atr,
    )


_FAMILY_REGISTRY: dict[str, Family] = {
    "RSI_REVERSION": Family(
        name="RSI_REVERSION",
        config_cls=RSIConfig,
        strategy_cls=RSIStrategy,
        dna_to_config=dna_to_rsi_config,
    ),
    "STOCH_CROSS": Family(
        name="STOCH_CROSS",
        config_cls=StochCrossConfig,
        strategy_cls=StochCrossStrategy,
        dna_to_config=dna_to_stoch_config,
    ),
    "MACD_CROSS": Family(
        name="MACD_CROSS",
        config_cls=MACDCrossConfig,
        strategy_cls=MACDCrossStrategy,
        dna_to_config=dna_to_macd_config,
    ),
    "DONCHIAN_BREAKOUT": Family(
        name="DONCHIAN_BREAKOUT",
        config_cls=DonchianBreakoutConfig,
        strategy_cls=DonchianBreakoutStrategy,
        dna_to_config=dna_to_donchian_config,
    ),
    "SIGNALS": Family(
        name="SIGNALS",
        config_cls=SignalsConfig,
        strategy_cls=SignalsStrategy,
        dna_to_config=lambda dna: dna_to_signals_config(dna),
    ),
}


def _gated(dna: DNA, strategy: object) -> object:
    """The family strategy, wrapped in the ARQVIO entry gates when the DNA carries any.

    W4-2: when the V4 switch genome has any bias/filter/management/risk switch ON, wrap the
    family strategy in a ``SwitchStrategy`` instead (the switch gates supersede / layer on top
    of the session+trend EMA gates). When switches are off, the legacy GatedStrategy path is
    used unchanged.
    """
    # v4 batch B: bias and filter switches are entry gates (backtest/switch_gates.py), carried by GatedStrategy
    # and the kernel's gate arrays; only management/risk switches still need the SwitchStrategy.
    # breakeven / sl_reduce / sl_lock / atr_trail are the engine loop's (BacktestConfig): no strategy-side gate.
    switches_on = bool([m for m in dna.mgmt_switches if m not in ENGINE_MGMT] or dna.risk_switches)
    if switches_on:
        # The entry gates (session, trend EMA, bias/filter switches) stay GatedStrategy's -- the kernel's gate arrays
        # -- and SwitchStrategy layers the management close rules on top; its own bias/filter evaluators are not
        # used (2026-11-01: they raised for every bias switch, so a gated DNA with a close rule could not run).
        from engine.research.backtest.strategies.gated import GatedStrategy
        from engine.research.backtest.strategies.switch import SwitchConfig, SwitchStrategy

        inner = strategy
        if dna.gated:
            inner = GatedStrategy(strategy, session=dna.session, trend_ema=dna.trend_ema,
                                  bias_switches=tuple(dna.bias_switches), filter_switches=tuple(dna.filter_switches))
        cfg = SwitchConfig(inner_strategy_id=inner.strategy_id, mgmt_switches=dna.mgmt_switches,
                           risk_switches=dna.risk_switches, params={"sl_atr": dna.sl_atr})
        return SwitchStrategy(inner, cfg)

    # legacy path: no V4 switches -> ARQVIO entry gates (session + trend EMA) only.
    if not dna.gated:
        return strategy
    from engine.research.backtest.strategies.gated import GatedStrategy
    return GatedStrategy(strategy, session=dna.session, trend_ema=dna.trend_ema,
                         bias_switches=tuple(dna.bias_switches), filter_switches=tuple(dna.filter_switches))


def dna_to_strategy(dna: DNA):
    """Dispatch a DNA to its strategy family.

    Returns (config, strategy). The slow engine accepts any object with
    an ``on_bar`` method and ``strategy_id``.

    Families are enumerated in SIGNAL_VALUES (dna.py:B4). Each family is
    either:
      - a slow-engine-only family in ``_FAMILY_REGISTRY``, or
      - the MA-cross gate (kernel expressible).

    A signal not in either set raises at the call site — there is no silent
    fallback to MA-cross, so an unbacked family is loud, not a phantom.
    """
    family = _FAMILY_REGISTRY.get(dna.signal)
    if family is not None:
        cfg = family.dna_to_config(dna)
        return cfg, _gated(dna, family.strategy_cls(cfg))
    if dna.signal == "MA_CROSS":
        return dna_to_ma_config(dna), _gated(dna, MACrossStrategy(dna_to_ma_config(dna)))
    raise DataError(
        f"signal={dna.signal!r} has no strategy dispatcher; "
        f"add it to SIGNAL_VALUES only when backed by a registered "
        f"feature and a dna_to_*_config + strategy class."
    )


# ---------------------------------------------------------------------------
# Bar loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellBars:
    """Loaded bar data for one (symbol, timeframe) cell."""

    symbol: str
    timeframe: str
    bars: BarData
    n_bars: int
    n_returns: int
    returns: np.ndarray  # log-returns, length n_bars-1
    source: str  # "MT5_LIVE", "BUNDLED", "CACHED"
    dataset_hash: str


def slice_cell_bars(cell: CellBars, n_bars: int) -> CellBars:
    """Return a CellBars restricted to the first ``n_bars`` bars.

    Used by the fast planted-edge canary so it exercises the full pipeline on
    a reduced dataset without a second MT5 fetch. Source is preserved so the
    real_backtest_factory still accepts it.
    """
    n = int(min(n_bars, cell.n_bars))
    if n < 30:
        raise DataError(f"slice_cell_bars: n_bars={n} too small (need >= 30)")
    b = cell.bars
    sliced = BarData(
        time=np.asarray(b.time)[:n],
        open=np.asarray(b.open)[:n],
        high=np.asarray(b.high)[:n],
        low=np.asarray(b.low)[:n],
        close=np.asarray(b.close)[:n],
        tick_volume=np.asarray(b.tick_volume)[:n],
        spread=np.asarray(b.spread)[:n],
        real_volume=np.asarray(b.real_volume)[:n],
        symbol=b.symbol,
        timeframe=b.timeframe,
    )
    log_ret = np.diff(np.log(np.asarray(b.close)[:n]))
    return CellBars(
        symbol=cell.symbol,
        timeframe=cell.timeframe,
        bars=sliced,
        n_bars=n,
        n_returns=int(log_ret.size),
        returns=log_ret,
        source=cell.source,
        dataset_hash=f"{cell.dataset_hash}-sliced{n}",
    )


def load_cell_bars_from_snapshot(symbol: str, timeframe: str, dataset_id: str, *,
                                 target_dir: Path | None = None) -> CellBars:
    """Build a CellBars from a PINNED snapshot instead of the live feed.

    load_cell_bars asks MT5 for the most recent N bars, so its window
    slides with wall-clock time and the same call returns different
    data tomorrow (G14). Nothing computed that way is reproducible, and
    no fast-vs-slow parity claim can be made against it.

    The dataset_hash here is the snapshot's CONTENT digest, not the old
    ``real-mt5-{symbol}-{timeframe}-{n}`` label -- that one was composed
    of the bar COUNT, so two different windows of equal length hashed
    identically and the substitution was invisible.

    ``target_dir`` reads a snapshot kept outside the artifacts store (the
    canary's, which ships in the repository).
    """
    from engine.research.data.dataset_snapshot import load_meta, load_snapshot

    meta = load_meta(dataset_id, target_dir)
    if meta is None:
        raise DataError(
            f"no dataset snapshot {dataset_id!r}; take one with "
            "scripts/snapshot_dataset.py"
        )
    if meta.symbol != symbol or meta.timeframe != timeframe:
        raise DataError(
            f"snapshot {dataset_id!r} is {meta.symbol} {meta.timeframe}, "
            f"not {symbol} {timeframe}"
        )
    bars = load_snapshot(dataset_id, target_dir)
    closes = np.asarray(bars.close, dtype=np.float64)
    log_ret = np.diff(np.log(closes))
    return CellBars(
        symbol=symbol,
        timeframe=timeframe,
        bars=bars,
        n_bars=int(closes.size),
        n_returns=int(log_ret.size),
        returns=log_ret,
        source="MT5_SNAPSHOT",
        dataset_hash=dataset_id,
    )


def load_cell_bars(
    *,
    symbol: str,
    timeframe: str,
    expected_bars: int | None = None,
    terminal_path: str | None = None,
) -> CellBars:
    """Fetch bars from MT5 and return a CellBars. Defect-1 check: if
    ``expected_bars`` is given, the bar count must match within 1% slack.

    ``terminal_path`` defaults to the RESOLVED terminal (env override,
    then configs/mt5.json, then discovery). Passing one explicitly still
    wins, so a caller driving two terminals keeps doing so.
    """
    if terminal_path is None:
        terminal_path = require_terminal()
    if not _initialize_mt5(mt5, terminal_path):
        raise DataError(
            f"MT5 initialize failed or timed out after {_MT5_INIT_TIMEOUT_S}s: {mt5.last_error()}"
        )
    try:
        for _ in range(15):
            info = mt5.terminal_info()
            if info is not None and info.connected:
                break
            time.sleep(1.0)
        else:
            raise DataError(
                f"MT5 terminal did not report connected within 15s of initialize(): {mt5.last_error()}"
            )
        probe = probe_basis(mt5, symbol)
        tf_const = getattr(mt5, f"TIMEFRAME_{timeframe}")
        # A symbol outside Market Watch can return nothing on a cold request
        # while the terminal syncs its history. GBPJPYm (visible=False) did
        # exactly that mid-campaign on 2026-09-18: the same request that had
        # served 10,122 bars an hour earlier came back empty, and the cell
        # was lost 85 minutes into a run. Select first, then retry once.
        if not mt5.symbol_select(symbol, True):
            raise DataError(
                f"MT5 symbol_select failed for {symbol}: {mt5.last_error()}"
            )
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, expected_bars or 100_000)
        if rates is None or len(rates) < 30:
            time.sleep(2.0)  # let the terminal finish pulling history
            rates = mt5.copy_rates_from_pos(
                symbol, tf_const, 0, expected_bars or 100_000
            )
        if rates is None or len(rates) < 30:
            raise DataError(
                f"MT5 returned no bars for {symbol} {timeframe} after select "
                f"and one retry: {mt5.last_error()}"
            )
        if expected_bars is not None:
            if abs(len(rates) - expected_bars) > max(1, expected_bars * 0.01):
                raise DataError(
                    f"bars={len(rates)} vs expected={expected_bars} for {symbol} {timeframe}"
                )
        # Build BarData
        n = int(len(rates))
        times = np.asarray(rates["time"], dtype=np.int64)
        opens = np.asarray(rates["open"], dtype=np.float64)
        highs = np.asarray(rates["high"], dtype=np.float64)
        lows = np.asarray(rates["low"], dtype=np.float64)
        closes = np.asarray(rates["close"], dtype=np.float64)
        # MT5 struct names: tick_volume, spread, real_volume
        def _col(name: str, default: np.ndarray) -> np.ndarray:
            try:
                return np.asarray(rates[name])
            except (ValueError, KeyError):
                return default
        tick_vol = _col("tick_volume", np.zeros(n, dtype=np.float64))
        spread = _col("spread", np.zeros(n, dtype=np.int32))
        real_vol = _col("real_volume", np.zeros(n, dtype=np.float64))
        # BarData's exact attribute names
        try:
            tf_enum = Timeframe.from_string(timeframe)
        except Exception:
            tf_enum = Timeframe.D1 if timeframe == "D1" else Timeframe.H1
        # The probe (R-C0) classifies the timestamp basis at fetch time.
        # convert_times is identity when the server is UTC (Exness audit);
        # for a broker_offset server it subtracts the measured EET/EEST delta.
        utc_times = convert_times(times, probe)
        bars = BarData(
            time=utc_times,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            tick_volume=tick_vol,
            spread=spread,
            real_volume=real_vol,
            symbol=symbol,
            timeframe=tf_enum,
            time_basis=probe.basis,
        )
        log_ret = np.diff(np.log(closes))
        return CellBars(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
            n_bars=n,
            n_returns=int(log_ret.size),
            returns=log_ret,
            source="MT5_LIVE",
            dataset_hash=f"real-mt5-{symbol}-{timeframe}-{n}",
        )
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Real backtest factory
# ---------------------------------------------------------------------------

def _cost_vector_hash(
    hourly_spread_points: tuple[float, ...] | None, spec: SymbolSpec
) -> str:
    """16-hex digest of the per-hour spread cost vector.

    Mirrors the cost-vector computation in prep_cell (prep.py:232-238):
    each hour's spread_points * point * per_unit_per_lot, or the flat
    spec.spread_in_price() when hourly_spread_points is None. The digest
    lets the report flag a cost-model change without a reader comparing
    24 floats by eye.
    """
    import hashlib

    import numpy as np

    from engine.research.costs.spread import spread_cost

    per_unit_per_lot = (
        max(spec.trade_tick_value_profit, spec.trade_tick_value_loss)
        / spec.trade_tick_size
    )
    if hourly_spread_points is not None:
        hourly = np.array(
            [h * spec.point * per_unit_per_lot for h in hourly_spread_points],
            dtype=np.float64,
        )
    else:
        flat = float(spread_cost(spec, 1.0).one_way)
        hourly = np.full(24, flat, dtype=np.float64)
    return hashlib.sha256(np.round(hourly, decimals=12).tobytes()).hexdigest()[:16]


def _load_spread_matrix(symbol: str):
    """The symbol's calibrated 24h spread row, or None if it has none.

    Read from ARQVIO_HOME/artifacts/spread resolved at CALL time (CLAUDE.md s7):
    the directory run_cell's launch gate checks and hashes, so the matrix that
    prices trades is the one the report's spread_artifact_sha256 describes."""
    from engine.data.spread_calibration import load, spread_dir

    try:
        return load(symbol, spread_dir())
    except Exception:
        # A matrix that will not load is not a matrix. F2 refuses the launch
        # for this symbol; here it simply contributes no blocked hours.
        return None


_ANNOUNCED: set[str] = set()


def log_no_trade_hours(symbol: str, hours: tuple[int, ...]) -> None:
    """Say it once per symbol, so the window is visible in the run log."""
    if symbol in _ANNOUNCED:
        return
    _ANNOUNCED.add(symbol)
    print(f"  {symbol}: zero-spread hours UTC {list(hours)} are untradeable")


@dataclass(frozen=True)
class SlowCellSetup:
    """Everything the slow engine is configured with on one cell, resolved
    once. Real candidates and their shams both read it, so a control can
    never run under different costs or gates from the candidate it judges."""

    spec: SymbolSpec
    no_trade_hours: tuple[int, ...]
    hourly_spread_points: tuple[float, ...] | None
    max_spread_points: float | None
    commission: float
    # v4 regime gate (costs/spread_regime.py): per-bar cap; None = the fixed max_spread_points gate.
    spread_cap_by_bar: np.ndarray | None = None
    # v4 cost: the spread charged on a fill at each bar (the signal candle's); None = the hourly rule only.
    fill_spread_points: np.ndarray | None = None


_COST_MODE: ContextVar[str | None] = ContextVar("spread_cost_mode", default=None)


@contextmanager
def spread_cost_mode(mode: str | None) -> Iterator[None]:
    """Factories built inside use ``mode`` ("bar" | "hourly") instead of settings.spread_cost_mode: a vault
    strategy re-runs under the cost model it was bred with. Per thread (a ContextVar); None = settings."""
    if mode not in (None, "bar", "hourly"):
        raise ValueError(f"spread cost mode {mode!r}; one of bar, hourly")
    token = _COST_MODE.set(mode)
    try:
        yield
    finally:
        _COST_MODE.reset(token)


def fill_spread_for(cell: CellBars, *, flat_cost: bool) -> np.ndarray | None:
    """Per-bar fill spread in points when the cost mode is "bar" (the v4 default), else None.

    flat_cost isolates the spread cost as one flat number, so it keeps the bar's own spread out as well."""
    from engine.research.costs.spread_regime import fill_spread_points

    if flat_cost or (_COST_MODE.get() or get_settings().spread_cost_mode) != "bar":
        return None
    return fill_spread_points(np.asarray(cell.bars.spread))


def spread_gate_for(cell: CellBars, fixed_cap: float | None, *, operator_override: bool) -> np.ndarray | None:
    """The per-bar regime cap for ``cell``, or None when the fixed gate applies.

    Fixed when the operator set a cap for the symbol, when settings.spread_gate_mode is "fixed", or when there is
    no fixed cap at all (no calibration matrix: no gate, as before -- the regime gate does not invent one)."""
    from engine.research.costs.spread_regime import regime_cap, window_bars

    settings = get_settings()
    if operator_override or settings.spread_gate_mode != "regime" or fixed_cap is None:
        return None
    return regime_cap(np.asarray(cell.bars.spread), window=window_bars(cell.timeframe, settings.spread_regime_days),
                      multiple=settings.spread_regime_multiple)


def slow_cell_setup(
    cell: CellBars, *, flat_cost: bool = False, caller: str = "real_backtest_factory",
) -> SlowCellSetup:
    """Resolve the slow engine's per-cell configuration.

    Defect-2 check: refuses any source that is not real bars. Real bars +
    synthetic backtester is forbidden and vice versa.

    MT5_SNAPSHOT is real MT5 bars that were written to disk under their
    own content hash instead of being re-fetched (G14). The guard exists
    to keep SYNTHETIC data out of the real backtester, and a snapshot is
    not synthetic -- it is the same bars, pinned so the run can be
    reproduced. Excluding it would mean no parity or acceptance work
    could ever run on reproducible data.
    """
    if cell.source not in {"MT5_LIVE", "MT5_SNAPSHOT", "BUNDLED"}:
        raise DataError(
            f"{caller} called with source={cell.source!r}; "
            f"only MT5_LIVE, MT5_SNAPSHOT or BUNDLED is allowed."
        )
    spec = spec_for(cell.symbol)
    # Hours whose calibrated median spread is zero. A zero spread is a free
    # trade, so those hours are untradeable rather than cheap. Resolved once
    # per cell, not per candidate. A symbol with no matrix contributes no
    # blocked hours here -- refusing that case is F2's job, at launch, not
    # something to decide silently in the backtester.
    no_trade_hours: tuple[int, ...] = ()
    hourly_spread_points: tuple[float, ...] | None = None
    # A5: the max-spread entry gate threshold, from this symbol's own
    # calibrated distribution rather than a hardcoded pip count. An
    # operator override wins; no matrix means no gate, not a guess.
    settings = get_settings()
    max_spread_points: float | None = settings.max_spread_points_by_symbol.get(
        cell.symbol
    )
    _matrix = _load_spread_matrix(cell.symbol)
    if _matrix is not None:
        no_trade_hours = tuple(sorted(_matrix.no_trade_hours))
        if no_trade_hours:
            log_no_trade_hours(cell.symbol, no_trade_hours)
        # flat_cost isolates the spread cost as the single variable: the
        # no-trade gate and max-spread gate stay live (from the matrix),
        # but the per-hour cost vector is forced to None so prep_cell /
        # the engine charge the flat spec spread instead of mean_buckets.
        if not flat_cost:
            hourly_spread_points = tuple(float(b) for b in _matrix.mean_buckets)
        if max_spread_points is None:
            max_spread_points = _matrix.spread_percentile_points(
                settings.max_spread_gate_percentile
            )
    commission = settings.commission_per_lot_per_side_by_symbol.get(
        cell.symbol, settings.commission_per_lot_per_side
    )
    return SlowCellSetup(
        spec=spec,
        no_trade_hours=no_trade_hours,
        hourly_spread_points=hourly_spread_points,
        max_spread_points=max_spread_points,
        commission=commission,
        spread_cap_by_bar=spread_gate_for(cell, max_spread_points,
                                          operator_override=cell.symbol in settings.max_spread_points_by_symbol),
        fill_spread_points=fill_spread_for(cell, flat_cost=flat_cost),
    )


def run_slow_backtest(
    cell: CellBars,
    setup: SlowCellSetup,
    dna: DNA,
    cfg: Any,
    strategy: Any,
    *,
    notes: str,
) -> Any:
    """Run one strategy on the slow engine: the full BacktestResult (its trades and equity curve included).

    ``run_slow_trial`` scores this result; the app's strategy inspector reads its trades.
    """
    config = BacktestConfig(
        strategy_id=cfg.strategy_id,
        symbol=cell.symbol,
        timeframe=cell.timeframe,
        initial_equity=10_000.0,
        risk_per_trade_pct=float(dna.risk_pct),
        commission_per_lot_per_side=setup.commission,
        slippage_mean_points=0.3,
        slippage_std_points=0.2,
        charge_swap=True,
        charge_commission=True,
        seed=0,
        notes=notes,
        no_trade_hours=setup.no_trade_hours,
        max_spread_points=setup.max_spread_points,
        hourly_spread_points=setup.hourly_spread_points,
        spread_cap_by_bar=setup.spread_cap_by_bar,
        fill_spread_points=setup.fill_spread_points,
        **stop_rule_kwargs(dna.mgmt_switches),
    )
    engine = BacktestEngine(spec=setup.spec, config=config, rng_seed=0)
    return engine.run(cell.bars, strategy)


def run_slow_trial(
    cell: CellBars,
    setup: SlowCellSetup,
    dna: DNA,
    cfg: Any,
    strategy: Any,
    *,
    notes: str,
) -> TrialResult:
    """Run one strategy on the slow engine and score it as a TrialResult.

    ``cfg`` supplies strategy_id and the stop genes; ``dna`` supplies the
    risk genes and the identity the result is filed under.
    """
    result = run_slow_backtest(cell, setup, dna, cfg, strategy, notes=notes)

    # Per-period returns from the equity curve.
    eq = np.asarray(result.equity_curve, dtype=np.float64)
    if eq.size >= 2 and np.all(eq[:-1] > 0):
        per_period = np.diff(eq) / eq[:-1]
    else:
        per_period = np.zeros(0, dtype=np.float64)

    # Realised Sharpe on the per-period returns.
    if per_period.size >= 2 and np.std(per_period) > 0:
        sharpe = float(np.mean(per_period) / np.std(per_period, ddof=1))
        # Annualise by the annualisation law (252 x 24 x 60 / tf minutes): every timeframe, never a default.
        ppy = bars_per_year(cell.timeframe)
        sharpe_annual = sharpe * math.sqrt(ppy)
    else:
        sharpe_annual = 0.0
        per_period = np.zeros(0, dtype=np.float64)

    # Max drawdown from the equity curve.
    if eq.size >= 2:
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / np.maximum(running_max, 1e-9)
        max_dd = float(-drawdowns.min()) if drawdowns.size > 0 else 0.0
    else:
        max_dd = 0.0

    total_return = float((eq[-1] / eq[0]) - 1.0) if eq.size >= 2 else 0.0

    return TrialResult(
        dna=dna,
        outcome=FailureTaxonomy.EVALUATED.value,
        sharpe=sharpe_annual,
        n_trades=int(result.n_trades),
        total_return=total_return,
        max_dd=max_dd,
        spread_uncalibrated=False,
        per_period_returns=per_period,
        blocked_zero_spread_events=int(result.blocked_zero_spread_events),
        deferred_zero_spread_fills=int(result.deferred_zero_spread_fills),
        sltp_deferred_zero_spread=int(result.sltp_deferred_zero_spread),
        blocked_max_spread=int(result.blocked_max_spread),
        costs={
            "spread": float(result.cost_spread),
            "commission": float(result.cost_commission),
            "slippage": float(result.cost_slippage),
            "swap": float(result.cost_swap),
        },
        exit_reasons=dict(result.exit_reasons),
        # A candidate with either protective level can exit INTRABAR,
        # so its fill prices depend on within-bar path that OHLC bars
        # do not record. Those need tick-resolution parity in S6.
        parity_class=(
            "TICK" if (cfg.sl_atr > 0.0 or cfg.tp_atr > 0.0)
            else "M1_OHLC"
        ),
    )


def real_backtest_factory(
    cell: CellBars,
    *,
    flat_cost: bool = False,
) -> Callable[[DNA], TrialResult]:
    """Build a backtest_fn that runs the S3 engine on the loaded cell."""
    setup = slow_cell_setup(cell, flat_cost=flat_cost)

    def _bt(dna: DNA) -> TrialResult:
        cfg, strategy = dna_to_strategy(dna)
        return run_slow_trial(
            cell, setup, dna, cfg, strategy, notes=f"dna_hash={dna.dna_hash}",
        )
    _bt.spread_cost_source = (
        "mean_buckets" if setup.hourly_spread_points is not None else "flat"
    )
    _bt.cost_vector_hash = _cost_vector_hash(
        setup.hourly_spread_points, setup.spec
    )
    if setup.fill_spread_points is not None:  # the fast factory's own label and digest (prep.prep_cell)
        from engine.research.backtest.prep import prep_cell

        prepped = prep_cell(cell.bars, setup.spec, hourly_spread_points=setup.hourly_spread_points,
                            fill_spread_points=setup.fill_spread_points)
        _bt.spread_cost_source, _bt.cost_vector_hash = prepped.spread_cost_source, prepped.cost_vector_hash
    return _bt


GAUSSIAN_CONTROL_KIND = "gaussian_iid_N(0,0.5)"


def gaussian_control_factory(
    rng: np.random.Generator,
) -> Callable[[list[TrialResult]], list[float]]:
    """A vs-random control of iid N(0, 0.5) Sharpes, one per scored trial
    (Lopez de Prado's "unskilled" benchmark).

    It ignores the data, the costs and each trial's trade count, so it is
    not a control for a real campaign -- that is the sham in
    engine/research/search/sham_control.py. It stays as the planted-edge
    canary's control, where the question is whether the pipeline surfaces
    a planted edge, not whether a real one exists.
    """
    def _ctrl(trials: list[TrialResult]) -> list[float]:
        return list(rng.normal(0.0, 0.5, size=len(trials)))
    _ctrl.kind = GAUSSIAN_CONTROL_KIND
    return _ctrl


def real_scores(trials: list[TrialResult]) -> list[float]:
    out: list[float] = []
    for t in trials:
        if t.sharpe is not None and np.isfinite(t.sharpe):
            out.append(float(t.sharpe))
    return out


def fast_backtest_factory(cell, *, rng_seed: int = 0, flat_cost: bool = False):
    """A backtest_fn that fills the SAME TrialResult the slow path does.

    Drop-in for real_backtest_factory. Every derived quantity -- Sharpe,
    max drawdown, total return, per-period returns -- is computed here
    by the SAME arithmetic the slow factory uses, on the equity curve
    the kernel produced, so a field cannot silently differ between
    modes because two call sites computed it two ways.

    A candidate the kernel cannot express EXACTLY is not approximated:
    it runs on the slow engine and is counted. ``fallbacks`` on the
    returned function records how many, and why.
    """
    import math

    import numpy as np

    from engine.config.settings import get_settings
    from engine.research.backtest.fast_path import (
        expressible,
        record_exit_reasons,
        record_to_dict,
        run_fast,
    )
    from engine.research.backtest.gates import gate_arrays
    from engine.research.backtest.prep import (
        MA_CROSS_ARRAYS,
        family_arrays,
        family_key,
        prep_cell,
    )
    from engine.research.features.atr import DEFAULT_ATR_PERIOD

    if cell.source not in {"MT5_LIVE", "MT5_SNAPSHOT", "BUNDLED"}:
        raise DataError(
            f"fast_backtest_factory called with source={cell.source!r}; "
            f"only MT5_LIVE, MT5_SNAPSHOT or BUNDLED is allowed."
        )
    spec = spec_for(cell.symbol)

    settings = get_settings()
    max_spread_points = settings.max_spread_points_by_symbol.get(cell.symbol)
    no_trade_hours: tuple[int, ...] = ()
    hourly_spread_points: tuple[float, ...] | None = None
    matrix = _load_spread_matrix(cell.symbol)
    if matrix is not None:
        no_trade_hours = tuple(sorted(matrix.no_trade_hours))
        if no_trade_hours:
            log_no_trade_hours(cell.symbol, no_trade_hours)
        if not flat_cost:
            hourly_spread_points = tuple(float(b) for b in matrix.mean_buckets)
        if max_spread_points is None:
            max_spread_points = matrix.spread_percentile_points(
                settings.max_spread_gate_percentile
            )
    commission = settings.commission_per_lot_per_side_by_symbol.get(
        cell.symbol, settings.commission_per_lot_per_side
    )

    spread_cap = spread_gate_for(cell, max_spread_points,
                                 operator_override=cell.symbol in settings.max_spread_points_by_symbol)
    fill_spread = fill_spread_for(cell, flat_cost=flat_cost)
    prepped = prep_cell(
        cell.bars, spec,
        no_trade_hours=no_trade_hours,
        max_spread_points=max_spread_points,
        hourly_spread_points=hourly_spread_points,
        spread_cap_by_bar=spread_cap,
        fill_spread_points=fill_spread,
    )
    kernel_gate = prepped.max_spread_points if spread_cap is not None else max_spread_points  # 0.0 = spread - cap
    slow_fn = real_backtest_factory(cell, flat_cost=flat_cost)
    ppy = bars_per_year(cell.timeframe)

    state = {"fallbacks": 0, "reasons": []}
    atr_period = DEFAULT_ATR_PERIOD  # prep_cell's default: the series in prepped.atr14
    family_cache: dict[tuple, object] = {}
    # The strategy inspector sets capture["on"]: the next kernel run also keeps its trade log and equity curve
    # (the kernel's own trades_out rows; the scores are unchanged).
    capture: dict[str, Any] = {}

    def _bt(dna: DNA) -> TrialResult:
        cfg, strategy = dna_to_strategy(dna)
        config_probe = BacktestConfig(
            strategy_id=cfg.strategy_id,
            symbol=cell.symbol,
            timeframe=cell.timeframe,
            initial_equity=10_000.0,
            charge_swap=True,
            charge_commission=True,
            commission_per_lot_per_side=commission,
            slippage_mean_points=0.3,
            slippage_std_points=0.2,
            no_trade_hours=no_trade_hours,
            max_spread_points=max_spread_points,
            hourly_spread_points=hourly_spread_points,
            spread_cap_by_bar=spread_cap,
            fill_spread_points=fill_spread,
        )
        verdict = expressible(dna, config_probe)
        if not verdict.ok:
            state["fallbacks"] += 1
            if verdict.reason not in state["reasons"]:
                state["reasons"].append(verdict.reason)
            return slow_fn(dna)

        if cfg.atr_period != atr_period:
            # prep carries ONE ATR series; a different period would stop
            # at different levels than the oracle.
            state["fallbacks"] += 1
            reason = f"atr_period={cfg.atr_period} != prepped {atr_period}"
            if reason not in state["reasons"]:
                state["reasons"].append(reason)
            return slow_fn(dna)
        if dna.signal == "MA_CROSS":
            fast_genes = {"fast": cfg.fast, "slow": cfg.slow}
            family = MA_CROSS_ARRAYS
        else:
            # Indicator series once per distinct (family, genes) on this
            # cell -- keyed on the gene values, never on an object id.
            key = family_key(cfg)
            family = family_cache.get(key)
            if family is None:
                family = family_arrays(prepped, cfg)
                family_cache[key] = family
            fast_genes = {}
        gates = None
        if dna.gated:
            gkey = ("GATES", dna.session, dna.trend_ema, tuple(dna.bias_switches), tuple(dna.filter_switches))
            gates = family_cache.get(gkey)
            if gates is None:
                gl, gs = gate_arrays(prepped.bar_hour_index, prepped.close, dna.session, dna.trend_ema)
                if dna.bias_switches or dna.filter_switches:
                    from engine.research.backtest.switch_gates import switch_arrays

                    sl, ss = switch_arrays(cell.bars.time, prepped.high, prepped.low, prepped.close,
                                           tuple(dna.bias_switches), tuple(dna.filter_switches),
                                           open_=prepped.open, volume=prepped.tick_volume)
                    gl, gs = gl & sl, gs & ss
                gates = (gl, gs)
                family_cache[gkey] = gates
        from engine.research.backtest.kernel import TR_SIZE

        trades_out = np.zeros((prepped.n_bars + 1, TR_SIZE), dtype=np.float64) if capture.get("on") else None
        record, eq, status = run_fast(
            prepped,
            **fast_genes, volume=cfg.volume,
            sl_atr=cfg.sl_atr, tp_atr=cfg.tp_atr, family=family, gates=gates,
            initial_equity=10_000.0,
            max_spread_points=kernel_gate,
            **stop_rule_kwargs(dna.mgmt_switches),
            **close_rule_kwargs(dna.mgmt_switches),
            charge_swap=True,
            charge_commission=True,
            commission_per_lot_per_side=commission,
            slippage_mean_points=0.3,
            slippage_std_points=0.2,
            rng_seed=rng_seed,
            want_equity=True,
            trades_out=trades_out,
        )
        if status != 0:
            # The kernel refused mid-run. Same rule: fall back, count it.
            state["fallbacks"] += 1
            reason = f"kernel status {status}"
            if reason not in state["reasons"]:
                state["reasons"].append(reason)
            return slow_fn(dna)

        agg = record_to_dict(record)
        if trades_out is not None:
            capture.update(trades=trades_out[: int(agg["n_trades"])].copy(), equity=eq.copy())

        # Identical arithmetic to real_backtest_factory, on the kernel's
        # equity curve.
        if eq.size >= 2 and np.all(eq[:-1] > 0):
            per_period = np.diff(eq) / eq[:-1]
        else:
            per_period = np.zeros(0, dtype=np.float64)
        if per_period.size >= 2 and np.std(per_period) > 0:
            sharpe = float(np.mean(per_period) / np.std(per_period, ddof=1))
            sharpe_annual = sharpe * math.sqrt(ppy)
        else:
            sharpe_annual = 0.0
            per_period = np.zeros(0, dtype=np.float64)
        if eq.size >= 2:
            running_max = np.maximum.accumulate(eq)
            drawdowns = (eq - running_max) / np.maximum(running_max, 1e-9)
            max_dd = float(-drawdowns.min()) if drawdowns.size > 0 else 0.0
        else:
            max_dd = 0.0
        total_return = float((eq[-1] / eq[0]) - 1.0) if eq.size >= 2 else 0.0

        return TrialResult(
            dna=dna,
            outcome=FailureTaxonomy.EVALUATED.value,
            sharpe=sharpe_annual,
            n_trades=int(agg["n_trades"]),
            total_return=total_return,
            max_dd=max_dd,
            spread_uncalibrated=False,
            per_period_returns=per_period,
            blocked_zero_spread_events=int(agg["blocked_zero_spread_events"]),
            deferred_zero_spread_fills=int(agg["deferred_zero_spread_fills"]),
            sltp_deferred_zero_spread=int(agg["sltp_deferred_zero_spread"]),
            blocked_max_spread=int(agg["blocked_max_spread"]),
            costs={
                "spread": float(agg["cost_spread"]),
                "commission": float(agg["cost_commission"]),
                "slippage": float(agg["cost_slippage"]),
                "swap": float(agg["cost_swap"]),
            },
            exit_reasons=record_exit_reasons(record),
            parity_class=(
                "TICK" if (cfg.sl_atr > 0.0 or cfg.tp_atr > 0.0)
                else "M1_OHLC"
            ),
        )

    _bt.fallback_state = state
    _bt.capture = capture
    _bt.spread_cost_source = prepped.spread_cost_source
    _bt.cost_vector_hash = prepped.cost_vector_hash
    return _bt
