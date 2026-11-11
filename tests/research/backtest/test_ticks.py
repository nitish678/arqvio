import numpy as np
import pytest

from engine.research.backtest.ticks import TickData, TickExecutionConfig, fill_market


def ticks():
    return TickData(np.array([1000, 1001, 1002], dtype=np.int64),
                    np.array([1.1000, 1.1001, 1.1002]),
                    np.array([1.1002, 1.1003, 1.1004]), "EURUSDm", "fixture")


def test_long_uses_ask_short_uses_bid_and_latency():
    t = ticks()
    long = fill_market(t, 0, "long", TickExecutionConfig(latency_ticks=1, point=0.0001, slippage_points=2))
    short = fill_market(t, 0, "short", TickExecutionConfig(latency_ticks=1))
    assert long.tick_index == short.tick_index == 1
    assert long.requested_price == 1.1003 and long.fill_price == pytest.approx(1.1005)
    assert short.requested_price == 1.1001 and short.fill_price == short.requested_price
    assert long.spread_price == pytest.approx(0.0002)


def test_bad_ticks_and_latency_refuse():
    with pytest.raises(ValueError, match="ask >= bid"):
        TickData(np.array([1]), np.array([2.0]), np.array([1.0]), "x", "fixture")
    with pytest.raises(ValueError, match="beyond"):
        fill_market(ticks(), 2, "long", TickExecutionConfig(latency_ticks=1))
