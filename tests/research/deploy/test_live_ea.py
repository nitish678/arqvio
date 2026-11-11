"""The live EA: every family renders, chart-locked, deterministic, guardrails validated, the tester EA untouched."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.deploy import full_deploy as fd
from engine.research.deploy.live_ea import (
    INCUBATION_STAGES,
    PERIOD_ENUMS,
    Guardrails,
    attach_script_source,
    generate_live,
)
from engine.research.mql5 import codegen
from engine.research.mql5.codegen import CodegenError
from engine.research.search import sampler
from engine.research.search.dna import SIGNAL_VALUES

MAGIC = 610_500_000


def one_per_family() -> dict:
    rng = np.random.default_rng(7)
    seen: dict = {}
    while len(seen) < len(SIGNAL_VALUES):
        d = sampler.random_dna(rng)
        seen.setdefault(d.signal, d)
    return seen


@pytest.mark.parametrize("family", SIGNAL_VALUES)
def test_every_family_renders_a_chart_locked_deterministic_ea(family: str) -> None:
    dna = one_per_family()[family]
    g = Guardrails(lots=0.02, daily_loss_pct=2.5, max_account_positions=3, news_block_minutes=15)
    a = generate_live(dna, symbol="XAUUSDm", timeframe="M15", magic=MAGIC, guardrails=g, no_trade_hours=(21,))
    b = generate_live(dna, symbol="XAUUSDm", timeframe="M15", magic=MAGIC, guardrails=g, no_trade_hours=(21,))
    assert a.source == b.source and "\r" not in a.source
    src = a.source
    assert 'const string LOCK_SYMBOL = "XAUUSDm";' in src and "const ENUM_TIMEFRAMES LOCK_PERIOD = PERIOD_M15;" in src
    assert f"const long MAGIC = {MAGIC};" in src and "InpAllowReal" not in src  # trades the account it is on
    assert "g_tester = MQLInfoInteger(MQL_TESTER) != 0;" in src  # the Strategy Tester runs it on any chart
    assert f"//--- family {family}" in src  # the tester EA's own family partial
    assert "input double InpLots = 0.02;" in src and "input double InpDailyLossPct = 2.5;" in src
    assert "C_NO_TRADE[21] = true;" in src and "C_NO_TRADE[20] = false;" in src
    assert a.inputs["InpLotScale"] == f"{INCUBATION_STAGES[0]:.8f}" and "InpAllowReal" not in a.inputs
    assert a.name == f"arqvio_{family.lower()}_{dna.dna_hash}_xauusdm_m15"


def test_the_chart_and_the_dna_change_the_source_and_the_name() -> None:
    dna = one_per_family()["RSI_REVERSION"]
    base = generate_live(dna, symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=Guardrails())
    other = generate_live(dna, symbol="EURUSD", timeframe="H4", magic=MAGIC, guardrails=Guardrails())
    nudged = generate_live(dataclasses.replace(dna, rsi_period=dna.rsi_period + 1 if dna.rsi_period < 21 else 20),
                           symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=Guardrails())
    assert len({base.source, other.source, nudged.source}) == 3
    assert len({base.name, other.name, nudged.name}) == 3


@pytest.mark.parametrize(("bad", "why"), [
    ({"lots": 0.0}, "lots"), ({"lot_scale": 0.3}, "incubation"), ({"daily_loss_pct": 101.0}, "daily_loss_pct"),
    ({"max_account_positions": 0}, "max_account_positions"), ({"friday_close_hour": 24}, "friday_close_hour"),
    ({"news_block_minutes": -1}, "news_block_minutes"), ({"history_bars": 50}, "history_bars"),
])
def test_guardrails_outside_their_range_are_refused(bad: dict, why: str) -> None:
    g = dataclasses.replace(Guardrails(), **bad)
    with pytest.raises(CodegenError, match=why):
        generate_live(one_per_family()["MA_CROSS"], symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=g)


def test_the_chart_magic_and_unknown_guardrails_are_refused() -> None:
    dna = one_per_family()["MA_CROSS"]
    with pytest.raises(CodegenError, match="timeframe"):
        generate_live(dna, symbol="EURUSD", timeframe="W1", magic=MAGIC, guardrails=Guardrails())
    with pytest.raises(CodegenError, match="outside the arqvio-deploy range"):
        generate_live(dna, symbol="EURUSD", timeframe="H1", magic=610_900_000, guardrails=Guardrails())
    with pytest.raises(CodegenError, match="symbol"):
        generate_live(dna, symbol='EUR"USD', timeframe="H1", magic=MAGIC, guardrails=Guardrails())
    with pytest.raises(CodegenError, match="unknown guardrail"):
        Guardrails.from_dict({"lots": 0.1, "martingale": True})


def test_the_tester_ea_digest_does_not_cover_the_live_templates() -> None:
    # a live template in the tester directory would change every parity EA's source
    assert not list(codegen._templates_dir().glob("ea_live*")) and "attach" not in str(codegen._templates_dir())
    assert set(PERIOD_ENUMS) == {"M1", "M5", "M15", "M30", "H1", "H2", "H4", "D1"}


def test_templates_encode_the_period_as_mt5_saves_it_and_presets_carry_no_cr() -> None:
    dna = one_per_family()["MACD_CROSS"]
    for tf, (ptype, psize) in {"M15": (0, 15), "H1": (1, 1), "H4": (1, 4), "D1": (2, 1), "M5": (0, 5)}.items():
        ea = generate_live(dna, symbol="EURUSD", timeframe=tf, magic=MAGIC, guardrails=Guardrails())
        text = fd.template_text(ea)
        assert f"\r\nperiod_type={ptype}\r\nperiod_size={psize}\r\n" in text
        assert f"\r\npath=Experts\\ARQVIO\\{ea.name}.ex5\r\nexpertmode=33\r\n" in text
        assert "\r\nInpLots=0.01000000\r\n" in text
        assert b"\r" not in fd.preset_text(ea)
    assert "ChartApplyTemplate" in attach_script_source()


def test_the_live_ea_carries_the_entry_gates_and_the_tester_ea_refuses_them() -> None:
    gated = dataclasses.replace(one_per_family()["MACD_CROSS"], session="LONDON", trend_ema=200)
    src = generate_live(gated, symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=Guardrails()).source
    assert 'const string GATE_SESSION = "LONDON";' in src and "const int GATE_LO = 7;" in src
    assert "const int GATE_HI = 16;" in src and "const int TREND_EMA = 200;" in src
    plain = generate_live(gated.ungated(), symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=Guardrails()).source
    assert 'const string GATE_SESSION = "";' in plain and "const int TREND_EMA = 0;" in plain
    # the refusal comes before the cell is read: no pinned data is needed to see it
    with pytest.raises(CodegenError, match="entry gates"):
        codegen.generate(gated, None, magic=610_000_000)  # type: ignore[arg-type]


def test_the_heartbeat_reports_the_dashboard_status_and_every_deal_goes_to_the_ledger() -> None:
    """Spec 2.4: one BotStatus() feeds the chart and the heartbeat file; the ledger is the terminal's own deal record,
    never written from a Strategy Tester run."""
    dna = next(iter(one_per_family().values()))
    src = generate_live(dna, symbol="XAUUSDm", timeframe="H1", magic=MAGIC, guardrails=Guardrails()).source
    assert src.count("BotStatus(") == 3  # defined once, called by Dash() and Heartbeat()
    assert 'J("status") + ":" + J(status)' in src and 'J("reason") + ":" + J(reason)' in src
    assert '"arqvio.bot_ledger"' in src and ".ledger.jsonl" in src
    assert "if(g_tester || !HistoryDealSelect(deal)) return;" in src and "Ledger(trans.deal);" in src


def test_a_breakeven_strategy_deploys_with_its_stop_rule_and_other_management_is_refused() -> None:
    dna = dataclasses.replace(next(iter(one_per_family().values())), mgmt_switches=("breakeven",))
    src = generate_live(dna, symbol="XAUUSDm", timeframe="H1", magic=MAGIC, guardrails=Guardrails()).source
    assert "input double InpBreakevenAtR = 1.0;" in src and "if(k == n - 1) ManageStops(rr[k].close);" in src
    plain = generate_live(dataclasses.replace(dna, mgmt_switches=()), symbol="XAUUSDm", timeframe="H1", magic=MAGIC,
                          guardrails=Guardrails()).source
    assert "input double InpBreakevenAtR = 0.0;" in plain
    with pytest.raises(CodegenError, match="no live-EA implementation"):
        generate_live(dataclasses.replace(dna, mgmt_switches=("partial_tp",)), symbol="XAUUSDm", timeframe="H1",
                      magic=MAGIC, guardrails=Guardrails())
    with pytest.raises(CodegenError):  # the Strategy Tester parity EA still refuses management switches
        codegen._validated(dna)


def test_signals_family_renders_its_switches_and_refuses_unready_ones():
    """v4 batch 2.6: a SIGNALS DNA renders the family_signals partial with exactly its switches."""
    import dataclasses as dc

    import numpy as np

    from engine.research.search.sampler import random_dna

    d = dc.replace(random_dna(np.random.default_rng(5)), signal="SIGNALS", signal2="",
                   signal_switches=("engulfing", "williams"))
    ea = generate_live(d, symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=Guardrails())
    assert ea.family == "SIGNALS"
    assert "SgVote" in ea.source and "g_open[i - 1]" in ea.source and "SgWilliams" in ea.source
    assert "SgCci(i - 1)" not in ea.source  # a switch that is off is not compiled in
    with pytest.raises(CodegenError, match="no live-EA implementation"):
        generate_live(dc.replace(d, signal_switches=("macd",)), symbol="EURUSD", timeframe="H1", magic=MAGIC,
                      guardrails=Guardrails())


def test_the_last_signals_bias_and_filters_render_and_an_unready_gate_is_refused():
    """2026-11-01: fib, atr_run, vwap, orb, kalman; bias trailing, market_struct, vwap, psar; filters sma, receding,
    doji, regime, event -- each compiles in only when on (MetaEditor 0/0 was checked on these renders)."""
    import dataclasses as dc

    import numpy as np

    from engine.research.backtest.switch_gates import CHOP_RATIO
    from engine.research.mql5.codegen import bits
    from engine.research.search.sampler import random_dna

    d = dc.replace(random_dna(np.random.default_rng(5)), signal="SIGNALS", signal2="",
                   signal_switches=("fib", "atr_run", "vwap", "orb", "kalman"),
                   bias_switches=("trailing", "market_struct", "vwap", "psar"),
                   filter_switches=("sma", "receding", "doji", "regime", "event"))
    src = generate_live(d, symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=Guardrails()).source
    for marker in ("g_orb_fired = true", "g_kd_now > 0.0", "g_vw_pv +=", "0.618 * (hh - ll)", "1.5 * SgMean(g_tr, i, 14)",
                   "g_trl_dir = -1", "g_ps_af = MathMin", "SwVwap(i, w)", "h2 > h1 && l2 > l1",
                   "0.25 * a", "nw / 5.0 >= old / 5.0", "0.1 * rng", f"FromBits({bits(CHOP_RATIO)})", "gap <= 1.0 * a",
                   "g_vol[i] = (double)r.tick_volume;"):
        assert marker in src, marker
    plain = generate_live(dc.replace(d, signal_switches=("cci",), bias_switches=(), filter_switches=()),
                          symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=Guardrails()).source
    assert "g_orb_fired = true" not in plain and "g_trl_dir = -1" not in plain and "0.1 * rng" not in plain
    with pytest.raises(CodegenError, match="bias ema"):  # the EMA bias is the trend_ema gate, never a bias switch
        generate_live(dc.replace(d, bias_switches=("ema",)), symbol="EURUSD", timeframe="H1", magic=MAGIC,
                      guardrails=Guardrails())


def test_the_stop_moving_management_switches_become_inputs():
    import dataclasses as dc

    import numpy as np

    from engine.research.search.sampler import random_dna

    d = dc.replace(random_dna(np.random.default_rng(5)), mgmt_switches=("sl_reduce", "sl_lock", "atr_trail"))
    src = generate_live(d, symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=Guardrails()).source
    for line in ("input double InpBreakevenAtR = 0.0;", "input double InpSlReduceAtR = 0.5;",
                 "input double InpSlLockAtR = 2.0;", "input double InpAtrTrailK = 2.0;",
                 "input double InpAtrTrailAtR = 1.0;", "HistoryOrderGetDouble(id, ORDER_SL)"):
        assert line in src, line
    off = generate_live(dc.replace(d, mgmt_switches=()), symbol="EURUSD", timeframe="H1", magic=MAGIC,
                        guardrails=Guardrails()).source
    assert "input double InpAtrTrailK = 0.0;" in off and "input double InpSlLockAtR = 0.0;" in off


def test_the_close_rules_become_inputs_and_run_before_the_family():
    import dataclasses as dc

    import numpy as np

    from engine.research.search.sampler import random_dna

    d = dc.replace(random_dna(np.random.default_rng(5)), mgmt_switches=("friday_close_profit", "eod_close"))
    src = generate_live(d, symbol="EURUSD", timeframe="H1", magic=MAGIC, guardrails=Guardrails()).source
    assert "input double InpFridayCloseProfitPct = 100.0;" in src and "input int    InpEodCloseHour = 20;" in src
    assert "if(!MgmtClose(g_n - 1, x) && FamilyOnBar(g_n - 1, x)) Gate(x);" in src
    assert "if(MgmtClose(g_n - 1, x)) { if(k == n - 1) Act(x); }" in src
    off = generate_live(dc.replace(d, mgmt_switches=()), symbol="EURUSD", timeframe="H1", magic=MAGIC,
                        guardrails=Guardrails()).source
    assert "input double InpFridayCloseProfitPct = 0.0;" in off and "input int    InpEodCloseHour = -1;" in off
