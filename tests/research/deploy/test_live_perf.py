"""Live performance: closing deals by magic, the backtest band, the verdict, orphans."""
from __future__ import annotations

from types import SimpleNamespace as D

import numpy as np

from engine.research.deploy import live_perf as lp


def test_only_closing_deals_of_the_magic_count_with_swap_and_commission() -> None:
    deals = [D(magic=7, entry=0, profit=0, swap=0, commission=-1, time=1, volume=0.1),     # opening: not a trade
             D(magic=7, entry=1, profit=12.0, swap=-0.5, commission=-1.0, time=2, volume=0.1),
             D(magic=8, entry=1, profit=99.0, swap=0, commission=0, time=3, volume=0.1)]
    t = lp.deal_trades(deals, 7)
    assert len(t) == 1 and t[0]["net"] == 10.5


def test_the_verdict_needs_fifteen_trades_and_reads_the_band() -> None:
    bt = np.array([10.0, -5.0, 8.0, -4.0, 6.0])
    bot = {"bot": "b", "magic": 7, "dna_hash": "abcd1234"}
    few = lp.judge_bot(bot, [{"time": 1, "net": 1.0, "volume": 0.1}] * 5, bt, 0.1, 0.1)
    assert few["verdict"] == "TOO FEW"
    good = lp.judge_bot(bot, [{"time": i, "net": 3.0, "volume": 0.1} for i in range(20)], bt, 0.1, 0.1)
    assert good["verdict"] in ("HOLDING", "AHEAD") and good["band"]["p5"] < good["band"]["p95"]
    bad = lp.judge_bot(bot, [{"time": i, "net": -10.0, "volume": 0.1} for i in range(20)], bt, 0.1, 0.1)
    assert bad["verdict"] == "WEAKENING"
    half = lp.judge_bot(bot, [{"time": i, "net": 1.0, "volume": 0.05} for i in range(20)], bt, 0.1, 0.05)
    assert half["lot_scale_vs_backtest"] == 0.5


def test_orphans_are_arqvio_magics_without_a_bot() -> None:
    deals = [D(magic=610500009, symbol="EURUSD"), D(magic=610500000, symbol="XAUUSD"), D(magic=12, symbol="X")]
    pos = [D(magic=610500009, symbol="EURUSD")]
    out = lp.orphans(deals, pos, {610500000}, 610500000, 610599999)
    assert out == [{"magic": 610500009, "deals": 1, "open_positions": 1, "symbol": "EURUSD"}]


def test_incubation_advice_is_advice_on_live_trades_only():
    from engine.research.deploy.live_perf import incubation_advice

    stages = (0.25, 0.5, 1.0)
    good = {"trades": 12, "net": 40.0, "band": {"p5": -30.0, "p50": 10.0, "p95": 60.0}, "verdict": "TOO FEW"}
    a = incubation_advice(0, good, stages)
    assert a["ready"] and a["next_lot_scale"] == 0.5 and "never changes" in a["note"]
    assert not incubation_advice(1, good, stages)["ready"]            # 50% -> 100% needs 30 live trades
    assert not incubation_advice(0, {**good, "net": -50.0}, stages)["ready"]
    assert not incubation_advice(0, {**good, "verdict": "WEAKENING"}, stages)["ready"]
    assert incubation_advice(2, good, stages)["note"] == "full size"
