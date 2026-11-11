"""Pin Bailey MinBTL formula and reject the deleted linear fit."""
from __future__ import annotations

from engine.validation.minbtl import min_backtest_length


def _f(n_trials: int, target_sharpe: float) -> float:
    return min_backtest_length(n_trials, target_sharpe).min_years


def test_minbtl_500_at_sr_1_5() -> None:
    assert 4.13 <= _f(500, 1.5) <= 4.15


def test_minbtl_5000_at_sr_1_5() -> None:
    assert 6.03 <= _f(5000, 1.5) <= 6.05


def test_minbtl_50k_at_sr_1_5() -> None:
    assert 7.97 <= _f(50000, 1.5) <= 7.99


def test_minbtl_500k_at_sr_1_5() -> None:
    assert 9.93 <= _f(500000, 1.5) <= 9.95


def test_monotonic_growth() -> None:
    for n in [100, 1_000, 10_000, 100_000]:
        assert _f(10 * n, 1.0) / _f(n, 1.0) < 2
