"""The shared ledger: exact conservation, and the DR's hand-audited 3-bar x 2-member fixture (DR §2 check (d))."""
from __future__ import annotations

from fractions import Fraction

import pytest

from engine.research.portfolio.ledger import CashEvent, ConservationError, Ledger
from engine.research.portfolio.sim import replay
from tests.research.portfolio.fakes import T, fake_run

TIMES = [3600, 7200, 10800]
# DR §2 (d), verbatim: (cash, sum of marks, equity) per bar.
EXPECTED = [(999.0, 4.0, 1003.0), (998.0, 4.0, 1002.0), (1003.0, -3.0, 1000.0)]


def _fixture_runs():
    a = fake_run("A", "AAA", TIMES, [T("a1", entry_bar=0, fee=1.0, exit_bar=2, gross=5.0)], [4.0, 6.0, 0.0])
    b = fake_run("B", "BBB", TIMES, [T("b1", entry_bar=1, fee=0.5)], [0.0, -2.0, -3.0], [(1, "b1", 0.5)])
    return [a, b]


def test_the_hand_audited_fixture_is_the_doc_ledger_exactly() -> None:
    rep = replay(_fixture_runs(), e0=1000.0)
    assert [(b["cash"], b["marks"], b["equity"]) for b in rep.bars] == EXPECTED
    assert rep.conservation_bars == 3 and rep.recon_max_abs == 0.0 and not rep.refused


def test_the_member_sum_of_the_fixture_is_the_portfolio_equity() -> None:
    a, b = _fixture_runs()
    deltas = [(float(a.result.equity_curve[i + 1]) - 500.0) + (float(b.result.equity_curve[i + 1]) - 500.0)
              for i in range(3)]
    assert [1000.0 + d for d in deltas] == [e for _, _, e in EXPECTED]
    assert [float(a.result.equity_curve[i + 1]) - 500.0 for i in range(3)] == [3.0, 5.0, 4.0]
    assert [float(b.result.equity_curve[i + 1]) - 500.0 for i in range(3)] == [0.0, -3.0, -4.0]


def test_conservation_is_exact_in_rationals() -> None:
    led = Ledger(1000.0)
    led.apply(CashEvent(1, "A", "FEE", "x", -0.1))       # 0.1 is not dyadic: exact Fraction of its double
    led.set_mark("A", 0.2)
    s = led.close_bar(1)
    assert led.equity == Fraction(1000.0) + Fraction(-0.1) + Fraction(0.2)
    assert s.equity == float(led.equity) and s.cash == float(Fraction(1000.0) + Fraction(-0.1))


def test_a_broken_invariant_names_the_bar() -> None:
    led = Ledger(1000.0)
    led.set_mark("A", 1.0)
    led.cash += 1  # a bug that credits cash without an event
    with pytest.raises(ConservationError, match="bar 7"):
        led.close_bar(7)


def test_only_cash_kinds_reach_the_cash_ledger() -> None:
    with pytest.raises(ValueError, match="not a cash event"):
        Ledger(1.0).apply(CashEvent(1, "A", "MARK", "x", 1.0))
