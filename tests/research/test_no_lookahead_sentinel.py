"""C1: the causal sentinel, with both controls.

A planted look-ahead oracle posting Sharpe 35 survived both DSR and
PBO. Statistical correction cannot catch leakage, because a leaking
strategy is not overfit -- it is right, for a reason it will not have
live.

The existing lookahead_guard checks a DNA FLAG (uses_future_bar). That
catches a genome that declares itself and nothing else. This checks the
implementation.

Both controls are required. A detector that never fires is
indistinguishable from a clean codebase and reads as proof, so the
negative control is not optional decoration -- if it stops failing, the
positive result below means nothing.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.research.backtest.strategies.ma_crossover import MACrossStrategy
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.search.lookahead import (
    LeakyStrategy,
    check_causality,
)
from engine.research.search.real_backtest import dna_to_ma_config
from engine.research.search.sampler import random_dna

N_CANDIDATES = 200
N_BARS = 400


def _bars(seed: int = 3) -> BarData:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, 0.0015, size=N_BARS)) + 1.10
    return BarData(
        time=np.array([86400 + i * 3600 for i in range(N_BARS)], dtype=np.int64),
        open=walk.copy(),
        high=walk + 0.0010,
        low=walk - 0.0010,
        close=walk.copy(),
        tick_volume=np.ones(N_BARS),
        spread=np.zeros(N_BARS, dtype=np.int32),
        real_volume=np.zeros(N_BARS),
        symbol="USDCADm",
        timeframe=Timeframe.H1,
    )


def test_no_lookahead_sentinel() -> None:
    """POSITIVE CONTROL: 200 real candidates are causal.

    The same strategy is run twice over the same bars -- once seeing
    the whole series, once seeing only bars 0..t -- and every decision
    must match. A causal strategy cannot tell the two apart.
    """
    bars = _bars()
    rng = np.random.default_rng(17)
    leaks = []

    for _ in range(N_CANDIDATES):
        dna = random_dna(rng)
        # Keep the MA windows inside the fixture so most candidates
        # actually emit decisions rather than sitting in warmup.
        dna = dataclasses.replace(dna, max_feature_lookback=int(rng.integers(20, 60)))
        cfg = dna_to_ma_config(dna)
        result = check_causality(lambda c=cfg: MACrossStrategy(c), bars)
        if result.leaked:
            leaks.append((dna.dna_hash, result.summary()))

    assert not leaks, (
        f"{len(leaks)} of {N_CANDIDATES} candidates read a future bar; "
        f"first: {leaks[0]}"
    )


def test_sentinel_sees_enough_decisions_to_be_meaningful() -> None:
    """A sentinel over a population that never trades proves nothing."""
    bars = _bars()
    cfg = dna_to_ma_config(
        dataclasses.replace(
            random_dna(np.random.default_rng(5)), max_feature_lookback=30
        )
    )
    result = check_causality(lambda: MACrossStrategy(cfg), bars)
    assert result.causal
    assert result.n_decisions > 0, (
        "the fixture produced no decisions at all, so a leak could not "
        "have been detected either"
    )


def test_sentinel_negative_control_detects_a_planted_leak() -> None:
    """NEGATIVE CONTROL: a strategy reading close[t+1] MUST be caught.

    If this stops failing the detector is broken, and every passing
    run above is worthless.
    """
    bars = _bars()
    result = check_causality(LeakyStrategy, bars)

    assert result.leaked, (
        "the sentinel did not catch a strategy that reads close[t+1]; "
        "it cannot be trusted to catch a real one"
    )
    assert result.divergences, "a leak was reported with no divergence cited"
    first = result.divergences[0]
    assert first.vectorized != first.stepwise
    assert "LEAK" in result.summary()


def test_sentinel_compares_decisions_not_pnl() -> None:
    """Two runs can make the same money from different trades.

    The divergence must be reported on the DECISION, so a leak that
    changes one decision in ten thousand is still caught.
    """
    bars = _bars()
    result = check_causality(LeakyStrategy, bars)
    # The leaky strategy differs in DIRECTION, which is a decision-level
    # difference and would be invisible in an equity comparison on a
    # symmetric random walk.
    assert any(
        d.vectorized.split("|")[0] != d.stepwise.split("|")[0]
        for d in result.divergences
    )


def test_each_run_gets_a_fresh_strategy_and_state() -> None:
    """Sharing state between the two views would mask a real divergence."""
    seen: list[int] = []

    class _Counting:
        strategy_id = "counting"

        def __init__(self) -> None:
            seen.append(1)

        def on_bar(self, bar_index, bars, state):
            state["calls"] = int(state.get("calls", 0)) + 1
            return None

    check_causality(_Counting, _bars())
    assert len(seen) == 2, "the factory must be called once per view"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_negative_control_fires_on_several_series(seed: int) -> None:
    """The detector is not an artefact of one fixture."""
    assert check_causality(LeakyStrategy, _bars(seed)).leaked
