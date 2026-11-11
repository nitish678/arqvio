"""The shared-account replay (DR §2): members through the FROZEN slow engine, one exact cash ledger, caps that REFUSE.

``run_member`` runs one DNA on one cell through ``_TapEngine``: a subclass of the frozen ``BacktestEngine`` that
OVERRIDES two private methods only to observe them and delegates to ``super()`` unchanged:

* ``_unrealised`` records each bar's GROSS mark, ``self._gross_pnl(position, close - entry)``, which is the
  engine's own expression (0.0 when flat);
* ``_charge_swap`` records the trade's cumulative ``swap_charged`` after each charge and the bar it lands on.

The engine's result is unchanged: ``tests/research/portfolio/test_sim.py`` asserts the tapped equity curve is
bit-identical to an untapped run. ``member_events`` turns a run into FEE / SWAP / REALIZE events and per-bar marks;
``replay`` merges every member on the union time axis into one ``Ledger`` (conservation checked at every close),
applies the heat / margin / floor caps at each entry against the state at the previous close, records CONFLICT
events for opposite same-symbol positions (hedging account: never netted), and runs the member-sum reconciliation
(DR §2 check (b)) with every refused trade's contribution itemized.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from engine.research.backtest.engine import BacktestEngine, BacktestResult
from engine.research.backtest.order import Trade
from engine.research.mql5.cell_config import CellConfig
from engine.research.portfolio.heat import HEAT_ALERT, HEAT_CAP, OpenRisk, account_heat, position_risk
from engine.research.portfolio.ledger import CashEvent, Ledger
from engine.research.search.dna import DNA
from engine.research.search.real_backtest import dna_to_strategy

MARGIN_LEVERAGE = 100.0
MARGIN_CAP = 0.50
FLOOR_SHARE = 0.20
RECON_TOL = 1e-8


class _TapEngine(BacktestEngine):
    """The frozen engine, observed: per-bar gross marks and every swap charge (bar, trade, cumulative swap)."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.marks: list[float] = []
        self.swaps: list[tuple[int, str, float]] = []

    def _unrealised(self, position, price):  # noqa: ANN001 -- the frozen engine's own signature
        value = super()._unrealised(position, price)
        gross = 0.0 if position is None else self._gross_pnl(
            position, price - float(position.trade.entry_fill.filled_price))
        self.marks.append(gross)
        return value

    def _charge_swap(self, position, from_day, to_day):  # noqa: ANN001
        out = super()._charge_swap(position, from_day, to_day)
        self.swaps.append((len(self.marks), out.trade.trade_id, float(out.trade.swap_charged)))
        return out


@dataclass(frozen=True)
class MemberRun:
    mid: str
    symbol: str
    times: np.ndarray
    result: BacktestResult
    marks: tuple[float, ...]
    swaps: tuple[tuple[int, str, float], ...]
    per_unit_loss: float
    contract_size: float
    usd_base: bool
    initial_equity: float


def run_member(mid: str, dna: DNA, cell: CellConfig, *, initial_equity: float, tap: bool = True) -> MemberRun:
    cfg, strategy = dna_to_strategy(dna)
    config = dataclasses.replace(cell.engine_config(cfg.strategy_id, risk_pct=dna.risk_pct,
                                                    notes=f"dna_hash={dna.dna_hash}"), initial_equity=initial_equity)
    eng = (_TapEngine if tap else BacktestEngine)(spec=cell.spec, config=config, rng_seed=0)
    result = eng.run(cell.bars, strategy)
    return MemberRun(mid, cell.symbol, np.asarray(cell.bars.time, dtype=np.int64), result,
                     tuple(getattr(eng, "marks", ())), tuple(getattr(eng, "swaps", ())), float(eng._per_unit_loss),
                     float(cell.spec.contract_size), cell.spec.currency_base == "USD", float(initial_equity))


@dataclass(frozen=True)
class BarEvents:
    fees: tuple[tuple[Trade, float], ...]
    swaps: tuple[tuple[str, Fraction], ...]
    exits: tuple[tuple[str, float], ...]


def exit_fill_bar(t: Trade) -> int | None:
    """The bar whose open (or range) filled the exit. NOT ``exit_bar_index``: for a SIGNAL close the engine records
    the SIGNAL bar there (engine.py:703-709) while the money moves at the next open, the exit fill's bar."""
    return None if t.exit_fill is None else int(t.exit_fill.bar_index)


def member_events(run: MemberRun) -> list[BarEvents]:
    """Per bar index: the FEE (entry), SWAP (exact delta of cumulative swap) and REALIZE (gross) events."""
    n = run.times.size
    fees: list[list] = [[] for _ in range(n)]
    swaps: list[list] = [[] for _ in range(n)]
    exits: list[list] = [[] for _ in range(n)]
    for t in run.result.trades:
        fees[t.entry_fill.bar_index].append((t, -float(t.entry_fill.total_cost)))
        if t.exit_fill is not None:
            exits[exit_fill_bar(t)].append((t.trade_id, float(t.gross_price_pnl)))
    last: dict[str, Fraction] = {}
    for bar, tid, cum in run.swaps:
        now = Fraction(cum)
        swaps[bar].append((tid, -(now - last.get(tid, Fraction(0)))))
        last[tid] = now
    return [BarEvents(tuple(fees[i]), tuple(swaps[i]), tuple(exits[i])) for i in range(n)]


def open_trade_at_close(run: MemberRun, i: int) -> Trade | None:
    for t in run.result.trades:
        if t.entry_fill.bar_index <= i and (t.exit_fill is None or exit_fill_bar(t) > i):
            return t
    return None


def member_reconstruction_max_abs(run: MemberRun) -> float:
    """max |engine equity - (init + cumulative events + mark)| over the member's bars (the tap is complete)."""
    evs = member_events(run)
    acc = Fraction(run.initial_equity)
    worst = 0.0
    for i, e in enumerate(evs):
        acc += sum((Fraction(x) for _, x in e.fees), Fraction(0)) + sum((x for _, x in e.swaps), Fraction(0)) \
            + sum((Fraction(x) for _, x in e.exits), Fraction(0))
        worst = max(worst, abs(float(acc + Fraction(run.marks[i])) - float(run.result.equity_curve[i + 1])))
    return worst


@dataclass
class Replay:
    e0: float
    axis: list[int]
    bars: list[dict]
    events: list[dict] = field(default_factory=list)
    refused: dict[str, dict] = field(default_factory=dict)
    recon_max_abs: float = 0.0
    recon_bars: int = 0
    conservation_bars: int = 0


def _margin(run: MemberRun, t: Trade) -> float:
    price = 1.0 if run.usd_base else float(t.entry_fill.filled_price)
    return float(t.volume) * run.contract_size * price / MARGIN_LEVERAGE


def replay(runs: list[MemberRun], *, e0: float) -> Replay:
    axis = sorted({int(x) for r in runs for x in r.times})
    index = [{int(x): i for i, x in enumerate(r.times)} for r in runs]
    events = [member_events(r) for r in runs]
    led = Ledger(e0)
    rep = Replay(e0, axis, [])
    accepted_open: dict[str, Trade] = {}      # member -> accepted trade open at the previous close
    refused: set[str] = set()
    refused_contrib: dict[str, Fraction] = {}  # trade_id -> contribution to equity so far (events + mark)
    last_bar: list[int | None] = [None] * len(runs)
    prev_equity = float(e0)
    above = False
    for t in axis:
        # Caps see the positions open at the previous close plus the entries already accepted at this bar, in
        # member order (no look-ahead; entries at one bar never pass the caps without seeing each other).
        exposure: list[tuple[str, Trade]] = list(accepted_open.items())
        for k, r in enumerate(runs):
            i = index[k].get(t)
            if i is None:
                continue
            last_bar[k] = i
            ev = events[k][i]
            for trade, fee in ev.fees:
                why = _cap_breach(runs, exposure, r, trade, prev_equity, e0)
                if why is not None:
                    refused.add(trade.trade_id)
                    rep.refused[trade.trade_id] = {"member": r.mid, "t": t, "cap": why[0], "values": why[1]}
                    rep.events.append({"t": t, "kind": "REFUSED", "member": r.mid, "trade": trade.trade_id,
                                       "cap": why[0], "values": why[1]})
                    refused_contrib[trade.trade_id] = Fraction(fee)
                    continue
                led.apply(CashEvent(t, r.mid, "FEE", trade.trade_id, fee))
                exposure.append((r.mid, trade))
            for tid, amt in ev.swaps:
                if tid in refused:
                    refused_contrib[tid] += amt
                else:
                    led.apply(CashEvent(t, r.mid, "SWAP", tid, amt))
            for tid, gross in ev.exits:
                if tid in refused:
                    refused_contrib[tid] += Fraction(gross)
                else:
                    led.apply(CashEvent(t, r.mid, "REALIZE", tid, gross))
            open_t = open_trade_at_close(r, i)
            if open_t is not None and open_t.trade_id not in refused:
                accepted_open[r.mid] = open_t
                led.set_mark(r.mid, r.marks[i])
            else:
                accepted_open.pop(r.mid, None)
                led.set_mark(r.mid, 0.0)
        state = led.close_bar(t)
        rep.conservation_bars += 1
        risks = [OpenRisk(m, _run(runs, m).symbol, position_risk(tr.entry_fill.filled_price, tr.sl_price, tr.volume,
                                                                    _run(runs, m).per_unit_loss))
                 for m, tr in accepted_open.items()]
        hp = account_heat(risks, state.equity)
        now_above = hp.unbounded or (hp.heat is not None and hp.heat > HEAT_ALERT)
        if now_above and not above:
            rep.events.append({"t": t, "kind": "HEAT_HIGH", "heat": hp.heat, "unbounded": hp.unbounded,
                               "per_symbol": hp.per_symbol})
        above = now_above
        for sym in sorted({rk.symbol for rk in risks}):
            dirs = {accepted_open[m].direction for m in accepted_open if _run(runs, m).symbol == sym}
            if len(dirs) > 1:
                rep.events.append({"t": t, "kind": "CONFLICT", "symbol": sym,
                                   "members": sorted(m for m in accepted_open if _run(runs, m).symbol == sym)})
        recon = _member_sum(runs, last_bar, e0, refused_contrib, refused)
        rep.recon_max_abs = max(rep.recon_max_abs, abs(float(recon) - state.equity))
        rep.recon_bars += 1
        rep.bars.append({"t": t, "cash": state.cash, "marks": state.marks, "equity": state.equity,
                         "heat": hp.heat, "heat_unbounded": hp.unbounded, "heat_per_symbol": hp.per_symbol})
        prev_equity = state.equity
    rep.refused = {tid: rep.refused[tid] | {"contribution_total": float(refused_contrib[tid])} for tid in rep.refused}
    return rep


def _run(runs: list[MemberRun], mid: str) -> MemberRun:
    return next(r for r in runs if r.mid == mid)


def _cap_breach(runs: list[MemberRun], exposure: list[tuple[str, Trade]], r: MemberRun, trade: Trade,
                prev_equity: float, e0: float) -> tuple[str, dict] | None:
    if not prev_equity > FLOOR_SHARE * e0:
        return "FLOOR", {"equity": prev_equity, "floor": FLOOR_SHARE * e0}
    risks = [position_risk(t.entry_fill.filled_price, t.sl_price, t.volume, _run(runs, m).per_unit_loss)
             for m, t in exposure]
    risks.append(position_risk(trade.entry_fill.filled_price, trade.sl_price, trade.volume, r.per_unit_loss))
    if any(x is None for x in risks):
        return "HEAT", {"heat": None, "unbounded": True, "cap": HEAT_CAP}
    heat = sum(risks) / prev_equity
    if heat > HEAT_CAP:
        return "HEAT", {"heat": heat, "cap": HEAT_CAP}
    margin = sum(_margin(_run(runs, m), t) for m, t in exposure) + _margin(r, trade)
    if margin > MARGIN_CAP * prev_equity:
        return "MARGIN", {"margin": margin, "cap": MARGIN_CAP * prev_equity}
    return None


def _member_sum(runs: list[MemberRun], last_bar: list[int | None], e0: float,
                refused_contrib: dict[str, Fraction], refused: set[str]) -> Fraction:
    """E0 + sum of member equity deltas - refused trades' contributions (events so far + current mark)."""
    total = Fraction(e0)
    for k, r in enumerate(runs):
        i = last_bar[k]
        if i is None:
            continue
        total += Fraction(float(r.result.equity_curve[i + 1])) - Fraction(r.initial_equity)
        open_t = open_trade_at_close(r, i)
        if open_t is not None and open_t.trade_id in refused:
            total -= Fraction(r.marks[i])
    return total - sum(refused_contrib.values(), Fraction(0))


def equity_on_axis(run: MemberRun, axis: list[int]) -> np.ndarray:
    """The member's engine equity at each axis time: its latest bar close at or before t (initial before bar 0)."""
    idx = np.searchsorted(run.times, np.asarray(axis, dtype=np.int64), side="right") - 1
    curve = np.asarray(run.result.equity_curve, dtype=np.float64)
    return np.where(idx >= 0, curve[np.clip(idx, 0, None) + 1], run.initial_equity)


def return_panel(runs: list[MemberRun], axis: list[int]) -> np.ndarray:
    """(len(axis) - 1) x members simple returns of each member's forward-filled engine equity (DR §3 input)."""
    eq = np.column_stack([equity_on_axis(r, axis) for r in runs])
    return eq[1:] / eq[:-1] - 1.0


__all__ = ["FLOOR_SHARE", "MARGIN_CAP", "MARGIN_LEVERAGE", "MemberRun", "RECON_TOL", "Replay", "equity_on_axis",
           "exit_fill_bar", "member_events", "member_reconstruction_max_abs", "open_trade_at_close", "replay",
           "return_panel", "run_member"]
