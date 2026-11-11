"""DECK portfolio view: correlation on shared days only, the worst axis is the grade, honest gaps."""
from __future__ import annotations

import numpy as np

from engine.research.vault import portfolio_view as pv


def _m(i, sym="XAUUSD", fam="MA_CROSS", tf="H1"):
    return {"id": i, "symbol": sym, "family": fam, "timeframe": tf, "dna": {"window_start_hour_utc": 0, "window_end_hour_utc": 24}}


def _series(seed, days=60, start=0):
    r = np.random.default_rng(seed).normal(0, 0.01, days)
    return {str(np.datetime64(20000 + start + k, "D")): float(r[k]) for k in range(days)}


def test_twins_are_one_bet_and_the_worst_axis_is_the_grade() -> None:
    a = _series(1)
    out = pv.judge([_m("a"), _m("b"), _m("c", sym="EURUSD", fam="RSI_REVERSION")], {"a": a, "b": dict(a), "c": _series(2)})
    assert out["corr"][0][1] == 1.0 and abs(out["corr"][0][2]) < 0.5
    assert out["grades"]["correlation"] == "F" and out["overall"] == "F"
    assert out["axes"]["market"]["largest"] == "XAUUSD" and abs(out["axes"]["market"]["share"] - 2 / 3) < 1e-12
    assert any("move together" in t for t in out["tips"]) and any("grow to 5+" in t for t in out["tips"])
    assert len(out["combined"]) == 60


def test_pairs_with_too_little_shared_history_get_no_number() -> None:
    out = pv.judge([_m("a"), _m("b", sym="EURUSD")], {"a": _series(1, days=40), "b": _series(2, days=40, start=30)})
    assert out["corr"][0][1] is None and out["grades"]["correlation"] is None and out["combined"] == []


def test_daily_returns_use_each_days_last_bar() -> None:
    t = np.array([0, 3600, 86400, 90000, 172800], dtype=np.int64)
    eq = np.array([100.0, 100.0, 101.0, 102.0, 102.0, 99.96])
    r = pv.daily_returns(t, eq)
    assert list(r.values()) == [102.0 / 101.0 - 1.0, 99.96 / 102.0 - 1.0]
