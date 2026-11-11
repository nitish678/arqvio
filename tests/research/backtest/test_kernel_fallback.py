"""S2-C: the fallback gate. No silent approximation, ever.

The S2 golden rule. A candidate whose genes or config the kernel cannot
express EXACTLY runs on the slow engine and is counted -- it is never
approximated. One approximated candidate in a thousand is
indistinguishable from a real edge at the population level, which is
the failure this stage exists not to cause.

Zero fallbacks is the goal for every allowlisted family. The mechanism
exists so the goal is checkable rather than assumed.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np

from engine.research.backtest.engine import BacktestConfig
from engine.research.backtest.fast_path import (
    SUPPORTED_EXEC_MODES,
    SUPPORTED_MGMT,
    SUPPORTED_SIGNALS,
    expressible,
)
from engine.research.search.sampler import random_dna


# The families the kernel expresses (W1.F1-F4 each added one after its
# oracle parity went green) and the ones still refused.
_ALLOWLISTED = ("MA_CROSS", "RSI_REVERSION", "STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT")
_NOT_YET_EXPRESSIBLE: tuple[str, ...] = ()


def _config(**kw) -> BacktestConfig:
    base = {
        "strategy_id": "fb", "symbol": "USDCADm", "timeframe": "H1",
        "initial_equity": 10_000.0,
    }
    base.update(kw)
    return BacktestConfig(**base)


def test_every_allowlisted_family_is_fully_expressible() -> None:
    """500 DNAs per allowlisted family, zero fallbacks.

    Renamed from test_the_current_family_is_fully_expressible (MA_CROSS
    only): each W1.F<n> commit adds its family here once its oracle parity
    is green.
    """
    rng = np.random.default_rng(0)
    refused = []
    for family in _ALLOWLISTED:
        for _ in range(500):
            dna = dataclasses.replace(random_dna(rng), signal=family)
            verdict = expressible(dna, _config())
            if not verdict.ok:
                refused.append((family, dna.dna_hash, verdict.reason))
    assert not refused, f"{len(refused)} candidates fell back; first: {refused[0]}"


def test_a_non_none_mgmt_routes_to_the_slow_path() -> None:
    """The brief's own example: poke mgmt past the sampler pin.

    mgmt is pinned to NONE because nothing implements the others
    (test_mgmt_pinned_none_until_implemented). If one ever reaches the
    engine, the kernel must refuse it rather than quietly ignore it --
    quietly ignoring is exactly what dna_to_ma_config used to do, and
    it is what the C5 audit caught.
    """
    dna = random_dna(np.random.default_rng(1))
    for mgmt in ("BREAKEVEN", "TRAIL_ATR", "PARTIAL"):
        poked = dataclasses.replace(dna, mgmt=mgmt)
        verdict = expressible(poked, _config())
        assert not verdict.ok, f"mgmt={mgmt} was accepted by the kernel"
        assert "mgmt" in verdict.reason


def test_an_unsupported_exec_mode_routes_to_the_slow_path() -> None:
    """LIMIT/LIMIT2/STOP are S6. The kernel must not guess at them."""
    dna = random_dna(np.random.default_rng(2))
    for mode in ("LIMIT", "LIMIT2", "STOP"):
        poked = dataclasses.replace(dna, exec_mode=mode)
        verdict = expressible(poked, _config())
        assert not verdict.ok, f"exec_mode={mode} was accepted"
        assert "exec_mode" in verdict.reason


def test_a_lookahead_dna_is_refused() -> None:
    dna = random_dna(np.random.default_rng(3))
    poked = dataclasses.replace(dna, uses_future_bar=True)
    verdict = expressible(poked, _config())
    assert not verdict.ok
    assert "uses_future_bar" in verdict.reason


def test_the_gate_names_what_it_supports_rather_than_what_it_rejects() -> None:
    """Allowlist, not denylist.

    A denylist accepts anything nobody thought to forbid, which is how
    a future gene family would get silently approximated instead of
    falling back. The kernel allowlists exactly the families with a
    parity-proven branch; everything else falls back.
    """
    assert frozenset({"MARKET2"}) == SUPPORTED_EXEC_MODES
    assert frozenset({"NONE"}) == SUPPORTED_MGMT
    assert frozenset(_ALLOWLISTED) == SUPPORTED_SIGNALS

    dna = random_dna(np.random.default_rng(4))
    invented = dataclasses.replace(dna, exec_mode="SOME_FUTURE_MODE")
    assert not expressible(invented, _config()).ok, (
        "an exec mode nobody has heard of was accepted; the gate is "
        "behaving as a denylist"
    )

    # Every family not yet in the allowlist must fall back...
    for family in _NOT_YET_EXPRESSIBLE:
        f_dna = dataclasses.replace(dna, signal=family)
        verdict = expressible(f_dna, _config())
        assert not verdict.ok, f"{family} was accepted by the kernel"
        assert "signal" in verdict.reason
    # ...and so must a name nobody registered (DNA construction would refuse
    # it too, so the gate is probed with a bare attribute bag).
    phantom = SimpleNamespace(
        exec_mode="MARKET2", mgmt="NONE", signal="PHANTOM_FAMILY", uses_future_bar=False,
    )
    verdict = expressible(phantom, _config())
    assert not verdict.ok and "signal" in verdict.reason


def test_the_gate_is_cheap_enough_to_run_per_candidate() -> None:
    """It gates every candidate, so it must not need the kernel."""
    import time

    rng = np.random.default_rng(5)
    dnas = [random_dna(rng) for _ in range(2000)]
    cfg = _config()
    started = time.perf_counter()
    for dna in dnas:
        expressible(dna, cfg)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"2000 gate calls took {elapsed:.3f}s"
