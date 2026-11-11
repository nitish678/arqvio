"""W1.F1-F4: the 10-step FIRED fixtures, per kernel family.

Each step runs the same synthetic bars through engine.py (the oracle) and
through the kernel's family branch, asserts they agree EXACTLY -- every
counter, the exit partition, the full trade list, final equity and each
cost line -- and then asserts the feature under test actually FIRED. A parity fixture
that passes because nothing happened proves nothing.

  1 fires                  5 gap fill               9 family exit per spec
  2 entry from signal      6 costs                  10 warmup silent
  3 SL/TP attached         7 swap                   11 +inf ATR suppresses
  4 deferral               8 guardrails (gate, daily loss, Friday close)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine.research.backtest.engine import BacktestConfig, BacktestEngine
from engine.research.backtest.fast_path import (
    record_exit_reasons,
    record_to_dict,
    run_fast,
)
from engine.research.backtest.kernel import (
    TR_DIR,
    TR_ENTRY_BAR,
    TR_EXIT_BAR,
    TR_EXIT_PRICE,
    TR_EXIT_REASON,
    TR_FILL_BAR,
    TR_REASON_SIGNAL,
    TR_REASON_SL,
    TR_REASON_TP,
    TR_SIZE,
    TR_SL,
    TR_TP,
)
from engine.research.backtest.prep import family_arrays, prep_cell
from engine.research.backtest.strategies.donchian_breakout import (
    DonchianBreakoutConfig, DonchianBreakoutStrategy,
)
from engine.research.backtest.strategies.macd_cross import (
    MACDCrossConfig, MACDCrossStrategy,
)
from engine.research.backtest.strategies.rsi import RSIConfig, RSIStrategy
from engine.research.backtest.strategies.stoch_cross import (
    StochCrossConfig, StochCrossStrategy,
)
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.real_backtest import spec_for

from tests.research.backtest.test_kernel_oracle_arithmetic import (
    assert_inf_atr_suppressed_identically,
)
from tests.research.backtest.test_kernel_parity_steps import (
    slow_trade_rows,
    trade_list_diffs,
)

SYMBOL = "USDCADm"


def _rsi(**kw):
    return RSIConfig(**{"period": 9, "oversold": 35.0, "overbought": 65.0, **kw}), RSIStrategy


def _stoch(**kw):
    return StochCrossConfig(**{
        "k_period": 9, "d_period": 3, "smoothing": 2,
        "k_oversold": 25.0, "k_overbought": 75.0, **kw,
    }), StochCrossStrategy


def _macd(**kw):
    return MACDCrossConfig(**{
        "fast_period": 5, "slow_period": 13, "signal_period": 4, **kw,
    }), MACDCrossStrategy


def _donchian(**kw):
    return DonchianBreakoutConfig(**{"period": 12, **kw}), DonchianBreakoutStrategy


FAMILIES = {
    "RSI_REVERSION": _rsi,
    "STOCH_CROSS": _stoch,
    "MACD_CROSS": _macd,
    "DONCHIAN_BREAKOUT": _donchian,
}


def _bars(
    n: int = 600, seed: int = 7, vol: float = 0.0016, spread_points=14,
    gaps: bool = False,
) -> BarData:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, vol, size=n)) + 1.10
    if gaps:
        walk[200:] += 0.02
        walk[350:] -= 0.03
    spread = (
        np.full(n, spread_points, dtype=np.int32) if np.isscalar(spread_points)
        else np.asarray(spread_points, dtype=np.int32)
    )
    return BarData(
        time=np.array([86400 * 3 + i * 3600 for i in range(n)], dtype=np.int64),
        open=walk.copy(), high=walk + 0.0012, low=walk - 0.0012,
        close=walk.copy(), tick_volume=np.ones(n), spread=spread,
        real_volume=np.zeros(n), symbol=SYMBOL, timeframe=Timeframe.H1,
    )


def _compare(bars: BarData, family: str, *, strategy_kw=None, **engine_kw):
    """Both engines, same bars; assert exact agreement; return (slow, rec, trades)."""
    cfg, strategy_cls = FAMILIES[family](
        volume=0.1, strategy_id="steps",
        sl_atr=engine_kw.pop("sl_atr", 0.0), tp_atr=engine_kw.pop("tp_atr", 0.0),
        **(strategy_kw or {}),
    )
    opts = dict(
        no_trade_hours=(), max_spread_points=None, hourly_spread_points=None,
        daily_loss_limit_pct=None, friday_close_hour=None, charge_swap=False,
        charge_commission=False, commission_per_lot_per_side=0.0,
        slippage_mean_points=0.0, slippage_std_points=0.0,
    )
    opts.update(engine_kw)
    spec = spec_for(SYMBOL)
    slow = BacktestEngine(
        spec=spec,
        config=BacktestConfig(
            strategy_id="steps", symbol=SYMBOL, timeframe="H1",
            initial_equity=10_000.0, **opts,
        ),
        rng_seed=0,
    ).run(bars, strategy_cls(cfg))

    prepped = prep_cell(
        bars, spec, no_trade_hours=opts["no_trade_hours"],
        max_spread_points=opts["max_spread_points"],
        hourly_spread_points=opts["hourly_spread_points"],
    )
    trades = np.zeros((prepped.n_bars, TR_SIZE), dtype=np.float64)
    record, _, status = run_fast(
        prepped, volume=cfg.volume, sl_atr=cfg.sl_atr, tp_atr=cfg.tp_atr,
        initial_equity=10_000.0,
        max_spread_points=opts["max_spread_points"],
        daily_loss_limit_pct=opts["daily_loss_limit_pct"],
        friday_close_hour=opts["friday_close_hour"],
        charge_swap=opts["charge_swap"],
        charge_commission=opts["charge_commission"],
        commission_per_lot_per_side=opts["commission_per_lot_per_side"],
        slippage_mean_points=opts["slippage_mean_points"],
        slippage_std_points=opts["slippage_std_points"],
        rng_seed=0, family=family_arrays(prepped, cfg), trades_out=trades,
    )
    assert status == 0, f"{family}: kernel status {status}"
    got = record_to_dict(record)
    for name in (
        "n_trades", "n_round_trips", "n_open", "blocked_zero_spread_events",
        "deferred_zero_spread_fills", "sltp_deferred_zero_spread",
        "blocked_max_spread", "blocked_daily_loss", "friday_flattens",
    ):
        assert got[name] == getattr(slow, name), (
            f"{family} {name}: fast={got[name]} slow={getattr(slow, name)}"
        )
    assert record_exit_reasons(record) == dict(slow.exit_reasons)
    rows = trades[: got["n_trades"]]
    diffs = trade_list_diffs([tuple(r) for r in rows], slow_trade_rows(slow))
    assert not diffs, f"{family} trade list: {diffs[:5]}"
    assert got["final_equity"] == slow.final_equity, (
        f"{family} final_equity fast={got['final_equity']!r} slow={slow.final_equity!r}"
    )
    for name in ("cost_commission", "cost_spread", "cost_slippage", "cost_swap"):
        assert got[name] == getattr(slow, name), (
            f"{family} {name} fast={got[name]!r} slow={getattr(slow, name)!r}"
        )
    return slow, record, rows


fam = pytest.mark.parametrize("family", list(FAMILIES))


@fam
def test_step1_fires(family: str) -> None:
    slow, _, _ = _compare(_bars(), family)
    assert slow.n_trades > 3, f"{family}: only {slow.n_trades} trades"


@fam
def test_step2_entry_from_signal(family: str) -> None:
    """MARKET2: every entry fills at the open of the bar after its signal."""
    _, _, rows = _compare(_bars(), family)
    assert rows.shape[0] > 3
    assert np.all(rows[:, TR_FILL_BAR] == rows[:, TR_ENTRY_BAR] + 1)


@fam
def test_step3_sl_tp_attached(family: str) -> None:
    slow, record, rows = _compare(_bars(), family, sl_atr=1.0, tp_atr=1.5)
    assert np.all(np.isfinite(rows[:, TR_SL])) and np.all(np.isfinite(rows[:, TR_TP]))
    long_ = rows[:, TR_DIR] > 0
    assert np.all(np.where(long_, rows[:, TR_SL] < rows[:, TR_TP], rows[:, TR_SL] > rows[:, TR_TP]))
    reasons = record_exit_reasons(record)
    assert reasons.get("SL", 0) > 0 and reasons.get("TP", 0) > 0, reasons


@fam
def test_step4_deferral(family: str) -> None:
    slow, _, rows = _compare(
        _bars(n=900, vol=0.0030), family, sl_atr=1.0, tp_atr=1.5,
        no_trade_hours=(3, 4, 5, 6, 7, 8),
    )
    assert slow.blocked_zero_spread_events > 0
    assert slow.deferred_zero_spread_fills > 0
    assert np.any(rows[:, TR_FILL_BAR] > rows[:, TR_ENTRY_BAR] + 1), "no entry was deferred"


@fam
def test_step5_gap_fill(family: str) -> None:
    """A level jumped over by the open exits AT the open, not at the level."""
    bars = _bars(n=500, seed=3, vol=0.0015, gaps=True)
    _, _, rows = _compare(bars, family, sl_atr=1.0, tp_atr=1.5)
    opens = np.asarray(bars.open)
    protective = np.isin(rows[:, TR_EXIT_REASON], (TR_REASON_SL, TR_REASON_TP))
    gapped = [
        r for r in rows[protective]
        if r[TR_EXIT_PRICE] == opens[int(r[TR_EXIT_BAR])]
        and r[TR_EXIT_PRICE] != (r[TR_SL] if r[TR_EXIT_REASON] == TR_REASON_SL else r[TR_TP])
    ]
    assert gapped, f"{family}: no protective exit was gap-filled"


@fam
def test_step6_costs(family: str) -> None:
    slow, _, _ = _compare(
        _bars(), family, sl_atr=1.0, tp_atr=2.0, charge_commission=True,
        commission_per_lot_per_side=3.5, slippage_mean_points=0.3,
        slippage_std_points=0.2,
        hourly_spread_points=tuple(float(h % 20 + 1) for h in range(24)),
    )
    assert slow.cost_commission > 0 and slow.cost_slippage > 0 and slow.cost_spread > 0


@fam
def test_step7_swap(family: str) -> None:
    slow, _, _ = _compare(_bars(n=800), family, sl_atr=2.0, tp_atr=4.0, charge_swap=True)
    assert slow.cost_swap != 0.0


@fam
def test_step8a_guardrail_max_spread_gate(family: str) -> None:
    spread = np.where(np.arange(600) % 5 == 0, 40, 14)
    slow, _, _ = _compare(_bars(spread_points=spread), family, max_spread_points=20.0)
    assert slow.blocked_max_spread > 0


@fam
def test_step8b_guardrail_daily_loss(family: str) -> None:
    slow, _, _ = _compare(_bars(n=900, vol=0.0030), family, sl_atr=1.0, daily_loss_limit_pct=0.05)
    assert slow.blocked_daily_loss > 0


@fam
def test_step8c_guardrail_friday_close(family: str) -> None:
    slow, _, _ = _compare(_bars(n=800), family, sl_atr=3.0, friday_close_hour=12)
    assert slow.friday_flattens > 0


@fam
def test_step9_family_exit_per_spec(family: str) -> None:
    """RSI: the midline exit closes FLAT (no re-entry on the next bar).
    STOCH / MACD / DONCHIAN: the opposite signal reverses -- the close is
    followed by an entry the other way, signalled one bar later."""
    kw = {"exit_midline": True} if family == "RSI_REVERSION" else {}
    _, _, rows = _compare(_bars(n=900), family, strategy_kw=kw)
    closes = [k for k in range(rows.shape[0]) if rows[k, TR_EXIT_REASON] == TR_REASON_SIGNAL]
    assert closes

    def reversed_(k: int) -> bool:
        return (
            k + 1 < rows.shape[0]
            and rows[k + 1, TR_ENTRY_BAR] == rows[k, TR_EXIT_BAR] + 1
            and rows[k + 1, TR_DIR] == -rows[k, TR_DIR]
        )

    if family == "RSI_REVERSION":
        assert any(not reversed_(k) for k in closes), "midline exit never fired flat"
    else:
        assert all(reversed_(k) for k in closes), "an opposite signal did not reverse"


@fam
def test_step10_warmup_silent(family: str) -> None:
    """No signal inside the warmup; a series shorter than the warmup trades nothing."""
    bars = _bars()
    cfg, _ = FAMILIES[family](volume=0.1, strategy_id="steps")
    warmup = family_arrays(prep_cell(bars, spec_for(SYMBOL)), cfg).warmup
    _, _, rows = _compare(bars, family)
    assert rows.shape[0] > 0 and rows[:, TR_ENTRY_BAR].min() >= warmup
    short = _bars(n=warmup)
    slow, _, _ = _compare(short, family)
    assert slow.n_trades == 0
    assert math.isfinite(slow.final_equity)


@fam
def test_step11_nonfinite_atr_suppresses(family: str) -> None:
    """A +inf high makes ATR +inf for 14 bars; the oracle's atr_at returns
    None there, so a signal wanting stops emits nothing -- in both engines
    (review finding R2)."""
    def entries(bars: BarData, **kw) -> list[int]:
        _, _, rows = _compare(bars, family, **kw)
        return [int(r[TR_ENTRY_BAR]) for r in rows]

    assert_inf_atr_suppressed_identically(entries, family)
