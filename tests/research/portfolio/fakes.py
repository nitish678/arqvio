"""Synthetic member runs for the replay's fixtures: real Trade / Fill objects, no engine, exact dyadic values.

``fake_run`` builds a MemberRun whose ``equity_curve`` is computed from its own events by the engine's equity
identity (init + FEE + SWAP + REALIZE so far + MARK), so the member-sum reconciliation has a consistent member side.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from engine.research.backtest.order import Fill, Trade
from engine.research.portfolio.sim import MemberRun


@dataclass
class T:
    tid: str
    entry_bar: int
    fee: float
    exit_bar: int | None = None
    gross: float = 0.0
    direction: str = "long"
    entry: float = 1.0
    sl: float | None = 0.99
    volume: float = 1.0


def _fill(tid: str, kind: str, direction: str, volume: float, price: float, cost: float, bar: int, t: int) -> Fill:
    return Fill(fill_id=f"{tid}-{kind}", order_id=f"{tid}-{kind}", direction=direction, volume=volume,
                requested_price=price, filled_price=price, spread_cost=cost, slippage_cost=0.0, commission_cost=0.0,
                bar_index=bar, bar_time=t)


def fake_run(mid: str, symbol: str, times: list[int], trades: list[T], marks: list[float],
             swaps: list[tuple[int, str, float]] = (), *, init: float = 500.0, per_unit_loss: float = 1.0,
             contract_size: float = 1.0, usd_base: bool = True) -> MemberRun:
    objs = []
    for x in trades:
        tr = Trade(trade_id=x.tid, strategy_id="fake", symbol=symbol, direction=x.direction,
                   entry_fill=_fill(x.tid, "in", x.direction, x.volume, x.entry, x.fee, x.entry_bar, times[x.entry_bar]),
                   exit_fill=None if x.exit_bar is None else _fill(x.tid, "out", x.direction, x.volume, x.entry, 0.0,
                                                                    x.exit_bar, times[x.exit_bar]),
                   entry_bar_index=x.entry_bar, exit_bar_index=x.exit_bar, volume=x.volume, sl_price=x.sl,
                   gross_price_pnl=x.gross)
        objs.append(tr)
    swap_at: dict[int, float] = {}
    last: dict[str, float] = {}
    for bar, tid, cum in swaps:
        swap_at[bar] = swap_at.get(bar, 0.0) + (cum - last.get(tid, 0.0))
        last[tid] = cum
    eq = [init]
    acc = init
    for i in range(len(times)):
        acc += sum(-x.fee for x in trades if x.entry_bar == i) + sum(x.gross for x in trades if x.exit_bar == i)
        acc -= swap_at.get(i, 0.0)
        eq.append(acc + marks[i])
    result = SimpleNamespace(trades=tuple(objs), equity_curve=np.asarray(eq, dtype=np.float64), n_trades=len(objs),
                             final_equity=eq[-1])
    return MemberRun(mid, symbol, np.asarray(times, dtype=np.int64), result, tuple(marks), tuple(swaps),
                     per_unit_loss, contract_size, usd_base, init)
