"""The free caps: base-token market matching, timeframes, power, switches and check_launch reasons.

WHY: the caps are defined once (engine/licence/limits.py); these tests pin that one definition.
"""
from __future__ import annotations

import pytest

from engine.licence.limits import (
    FREE_LIMITS,
    NO_LIMITS,
    PAID_LIMITS,
    Limits,
    base_token,
    check_launch,
    market_allowed,
    power_allowed,
    timeframe_allowed,
)


@pytest.mark.parametrize(
    ("symbol", "base"),
    [("XAUUSD", "XAUUSD"), ("XAUUSDm", "XAUUSD"), ("XAUUSD.pro", "XAUUSD"), ("#GOLD", "GOLD"),
     ("gold.ecn", "GOLD"), ("USDJPYc", "USDJPY"), ("EURUSD", "EURUSD"), ("", "")],
)
def test_base_token(symbol: str, base: str) -> None:
    assert base_token(symbol) == base


@pytest.mark.parametrize("symbol", ["XAUUSD", "XAUUSDm", "XAUUSD.pro", "GOLD", "USDCAD.r", "usdjpy"])
def test_free_markets_allowed(symbol: str) -> None:
    assert market_allowed(symbol, FREE_LIMITS)


@pytest.mark.parametrize("symbol", ["EURUSD", "EURUSDm", "XAGUSD", "US500", "XAUUSDEXTRALONG"])
def test_other_markets_refused(symbol: str) -> None:
    assert not market_allowed(symbol, FREE_LIMITS)


def test_timeframes_and_power() -> None:
    assert timeframe_allowed("H1", FREE_LIMITS) and timeframe_allowed("h2", FREE_LIMITS)
    assert not timeframe_allowed("M5", FREE_LIMITS) and not timeframe_allowed("H4", FREE_LIMITS)
    assert power_allowed("eco", FREE_LIMITS) and power_allowed("Medium", FREE_LIMITS)
    assert not power_allowed("high", FREE_LIMITS)


def test_free_switches() -> None:
    assert (FREE_LIMITS.smart_mode, FREE_LIMITS.deploy, FREE_LIMITS.deep_history) == (False, False, False)


def test_paid_and_no_limits_are_unlimited() -> None:
    big = {"markets": ["EURUSD", "GBPUSD", "US500", "XAGUSD", "BTCUSD"], "timeframes": ["M5", "H1", "H4"],
           "power": "max", "smart_mode": True, "deploy": True, "deep_history": True}
    for limits in (PAID_LIMITS, NO_LIMITS):
        assert market_allowed("EURUSD", limits) and timeframe_allowed("M1", limits) and power_allowed("max", limits)
        assert check_launch(big, limits) == []


def test_check_launch_allowed_free_request() -> None:
    request = {"markets": ["XAUUSDm", "USDJPY"], "timeframes": ["H1", "H2"], "power": "eco"}
    assert check_launch(request, FREE_LIMITS) == []
    assert check_launch({}, FREE_LIMITS) == []


def test_check_launch_reasons() -> None:
    request = {"markets": ["XAUUSD", "EURUSD"], "timeframes": ["H1", "M15"], "power": "high",
               "smart_mode": True, "deploy": True, "deep_history": True}
    assert check_launch(request, FREE_LIMITS) == [
        "market EURUSD is not in your plan",
        "timeframe M15 is not in your plan",
        "power mode high is not in your plan",
        "smart mode is not in your plan",
        "automation (deploy) is locked on your plan",
        "deep history is not in your plan",
    ]


def test_check_launch_counts_by_base_token() -> None:
    """XAUUSD and XAUUSDm are one market, so they do not use up two of the allowance."""
    capped = Limits(max_assets=1, max_tfs=1, free_markets=("XAUUSD",), free_tfs=("H1", "H2"))
    assert check_launch({"markets": ["XAUUSD", "XAUUSDm"], "timeframes": ["H1", "h1"]}, capped) == []
    reasons = check_launch({"markets": ["XAUUSD"], "timeframes": ["H1", "H2"]}, capped)
    assert reasons == ["2 timeframes chosen; your plan allows 1"]


def test_limits_dict_round_trip() -> None:
    assert Limits.from_dict(FREE_LIMITS.to_dict()) == FREE_LIMITS
    assert Limits.from_dict(PAID_LIMITS.to_dict()) == PAID_LIMITS
    assert Limits.from_dict({}).deploy is False  # a token that omits a switch never grants it
