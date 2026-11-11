"""S5 spec-pinning regression tests for MinBTL (Bailey et al. 2014).

The S5 master prompt pins these values (SR=1.5):

    min_backtest_length(500,     1.5) ==  4.14  (+/-0.01)
    min_backtest_length(5000,    1.5) ==  6.04  (+/-0.01)
    min_backtest_length(50000,   1.5) ==  7.98  (+/-0.01)
    min_backtest_length(500000,  1.5) ==  9.94  (+/-0.01)

Plus a monotonicity test: MinBTL(10n) / MinBTL(n) < 2 for every n >= 50.

The earlier (S5 interim) constants 53.65 / 240.4 were a back-fit of the
4 pinned values and produced two internally-inconsistent formulas. The
real Bailey formula above pins all 4 values with a single z(n) function.
"""
from __future__ import annotations

import pytest

from engine.validation.minbtl import (
    max_trials_for_history,
    min_backtest_length,
)


@pytest.mark.parametrize(
    ("n_trials", "expected_years", "tol"),
    [
        (500, 4.14, 0.01),
        (5000, 6.04, 0.01),
        (50000, 7.98, 0.01),
        (500000, 9.94, 0.01),
    ],
)
def test_min_backtest_length_pins_s5_spec_at_sr_1_5(
    n_trials: int, expected_years: float, tol: float
) -> None:
    result = min_backtest_length(n_trials=n_trials, target_sharpe=1.5)
    assert abs(result.min_years - expected_years) < tol, (
        f"min_backtest_length({n_trials}, 1.5).min_years = "
        f"{result.min_years:.4f}, expected {expected_years} +/-{tol}"
    )


@pytest.mark.parametrize("n", [50, 100, 200, 500, 1000, 5000, 50000, 500000])
def test_minbtl_monotonic_subquadratic_at_n(n: int) -> None:
    """MinBTL(10n) / MinBTL(n) < 2 for every n >= 50.

    The formula has a small-n pathology (n=10 -> n=100 is 2.58x); the
    monotonicity test starts at n=50 where the ratio is well under 2.
    """
    a = min_backtest_length(n_trials=n, target_sharpe=1.5).min_years
    b = min_backtest_length(n_trials=10 * n, target_sharpe=1.5).min_years
    ratio = b / a
    assert ratio < 2.0, (
        f"MinBTL(10n)/MinBTL(n) = {ratio:.4f} at n={n}; "
        f"a={a:.4f}y, b={b:.4f}y. Must be < 2."
    )


def test_minbtl_grows_with_trials() -> None:
    """More trials must require more history (weak monotonicity)."""
    a = min_backtest_length(n_trials=100, target_sharpe=1.5)
    b = min_backtest_length(n_trials=1000, target_sharpe=1.5)
    assert b.min_years > a.min_years


def test_minbtl_shrinks_with_target_sharpe() -> None:
    """A higher target SR requires LESS history to reject."""
    a = min_backtest_length(n_trials=5000, target_sharpe=1.0)
    b = min_backtest_length(n_trials=5000, target_sharpe=2.0)
    assert a.min_years > b.min_years


def test_minbtl_invalid_target_raises() -> None:
    import pytest
    with pytest.raises(Exception):
        min_backtest_length(n_trials=100, target_sharpe=0.0)


def test_minbtl_satisfied_at_5000_trials_sr_1_5_vs_12y() -> None:
    """The pre-S5 verdict was wrong: 5,000 trials @ SR=1.5 against
    12.67y needs 6.04y, not 41.4y. USDCADm D1, H4, and GBPJPYm D1
    are all SUFFICIENT.
    """
    for years in (12.67, 12.6738, 33.4162):
        result = min_backtest_length(
            n_trials=5000, target_sharpe=1.5, history_years=years,
        )
        assert result.satisfied is True, (
            f"5000 trials @ SR=1.5 vs {years}y should be SUFFICIENT; "
            f"need {result.min_years:.2f}y"
        )


def test_max_trials_for_history_uses_binary_search() -> None:
    """12.67y at SR=1.5 should give a large n (the Bailey formula is
    forgiving). 33.42y should give an even larger n."""
    n_short = max_trials_for_history(history_years=12.67, target_sharpe=1.5)
    n_long = max_trials_for_history(history_years=33.42, target_sharpe=1.5)
    assert n_short > 100_000, f"n_short={n_short} should be > 100k"
    assert n_long > n_short, f"n_long={n_long} should exceed n_short={n_short}"


def test_max_trials_zero_when_history_insufficient() -> None:
    """A history of 0 must return 0 trials."""
    n = max_trials_for_history(history_years=0.0, target_sharpe=1.5)
    assert n == 0
