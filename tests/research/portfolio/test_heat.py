"""Account heat (DR §4): a known-answer fixture, the per-symbol rollup, unbounded positions, threshold crossings."""
from __future__ import annotations

import pytest

from engine.research.portfolio.heat import (HEAT_ALERT, HEAT_CAP, HeatError, OpenRisk, account_heat, position_risk,
                                            upward_crossings)


def test_the_known_answer_fixture() -> None:
    # USDCADm-like long: |1.36000 - 1.35500| = 0.005 * per-unit 100000 * 0.5 lot = 250
    # XAUUSDm-like short: |2000.0 - 2010.0| = 10.0 * per-unit 100 * 0.25 lot = 250
    a = position_risk(1.36, 1.355, 0.5, 100000.0)
    b = position_risk(2000.0, 2010.0, 0.25, 100.0)
    assert a == pytest.approx(250.0, rel=1e-12) and b == 250.0
    hp = account_heat([OpenRisk("A", "USDCADm", a), OpenRisk("B", "XAUUSDm", b), OpenRisk("C", "XAUUSDm", 125.0)],
                      10000.0)
    assert hp.heat == pytest.approx(625.0 / 10000.0, rel=1e-12) and not hp.unbounded
    assert hp.per_symbol == {"USDCADm": pytest.approx(0.025, rel=1e-12), "XAUUSDm": 0.0375}


def test_a_position_without_a_stop_makes_heat_unbounded() -> None:
    hp = account_heat([OpenRisk("A", "S", 10.0), OpenRisk("B", "T", position_risk(1.0, None, 1.0, 1.0))], 100.0)
    assert hp.heat is None and hp.unbounded and hp.per_symbol == {"S": 0.1, "T": None}


def test_heat_refuses_bad_inputs() -> None:
    with pytest.raises(HeatError, match="not positive"):
        account_heat([], 0.0)
    with pytest.raises(HeatError, match="negative"):
        position_risk(1.0, 0.9, -1.0, 1.0)


def test_upward_crossings_of_the_alert_threshold() -> None:
    assert HEAT_ALERT == 0.06 and HEAT_CAP == 0.10
    assert upward_crossings([0.01, 0.07, 0.08, 0.05, 0.061, None, 0.02, 0.06]) == [1, 4]
    assert upward_crossings([0.07]) == [0] and upward_crossings([]) == []
