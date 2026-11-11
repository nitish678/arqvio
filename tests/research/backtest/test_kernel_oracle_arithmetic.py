"""The kernel's arithmetic is the oracle's, bit for bit.

Lot size: engine.py sizes every fill with SymbolSpec.normalise_volume
(``round(v / step) * step``, then clamp) and the cost functions
re-normalise with it. The kernel used ``vmin + round((v - vmin)/step)*step``
-- the same lot count, a different last bit for about a third of
production volumes -- so every money figure the kernel produced differed
from the oracle's by an ulp. Trade lists cannot be compared exactly
(W1.F1-F4) while volume differs.

The S3 kernel review confirmed three more (R1-R3, report section 7):

  R1 spread cost multiplied per-lot then by volume, not price * volume *
     per-unit -- pinned end to end by the step fixtures' exact money
     assertions (test_kernel_parity_steps / test_family_kernel_steps);
  R2 a +inf ATR passed _stop_distances where atr_at returns None;
  R3 SL/TP levels rounded with np.round, not Python's round.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.backtest.kernel import (
    _normalise_volume,
    _round_to_digits,
    _stop_distances,
)
from engine.research.backtest.prep import prep_cell
from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.features.atr import atr_at
from engine.research.search.real_backtest import spec_for

from tests.research.backtest import test_kernel_parity_steps as ma_steps

SYMBOL = "USDCADm"


def test_kernel_volume_equals_symbol_spec_normalise_volume_bitwise() -> None:
    rng = np.random.default_rng(20260929)
    for symbol in ("USDCADm", "XAUUSDm"):
        spec = spec_for(symbol)
        # Production sizing (dna_to_*_config): clamp(risk_pct * 10, 0.01, 1.0),
        # plus the grid itself and out-of-range requests the clamp must catch.
        requested = np.concatenate([
            np.clip(rng.uniform(0.005, 0.02, 20_000) * 10.0, 0.01, 1.0),
            np.round(np.arange(1, 101) * 0.01, 2),
            np.array([0.0, 1e-9, spec.volume_min / 3, spec.volume_max * 7]),
        ])
        differ = []
        for v in requested:
            req = max(float(v), spec.volume_min)  # engine.py:797
            want = spec.normalise_volume(req)
            got = _normalise_volume(req, spec.volume_min, spec.volume_max, spec.volume_step)
            if got != want:
                differ.append((symbol, float(v), got, want))
        assert not differ, f"{len(differ)} volumes differ; first: {differ[:3]}"


# --- R3: SL/TP levels are Python's round(), not np.round ------------------

# The review's case: a USDCAD short at 1.36 with sl = 1.0 * ATR(7 points / 14).
_REVIEW_RAW = 1.36 + 0.00007 / 14


def _near_tie_levels() -> list[tuple[float, int]]:
    """Raw stop levels within an ulp of a half-quantum, at each quote
    precision in use (5: FX, 3 and 2: metals). These are the only inputs on
    which rounding the product price * 10**d disagrees with rounding the
    exact binary value."""
    rng = np.random.default_rng(20260930)
    out = [(_REVIEW_RAW, 5)]
    for digits, lo, hi in ((5, 0.5, 2.0), (3, 50.0, 200.0), (2, 1000.0, 3000.0)):
        quantum = 10.0 ** -digits
        for base in np.round(rng.uniform(lo, hi, 300), digits):
            for j in range(1, 6):
                for side in (1.0, -1.0):
                    x = float(base + side * (j + 0.5) * quantum)
                    out += [
                        (x, digits),
                        (float(np.nextafter(x, np.inf)), digits),
                        (float(np.nextafter(x, -np.inf)), digits),
                    ]
    return out


def test_stop_rounding_equals_python_round_on_near_ties() -> None:
    cases = _near_tie_levels()
    wrong = [
        (x, d, _round_to_digits(x, d), round(x, d))
        for x, d in cases if _round_to_digits(x, d) != round(x, d)
    ]
    assert not wrong, f"{len(wrong)}/{len(cases)} levels differ; first: {wrong[:3]}"
    # The set exercises the defect: np.round (the kernel's former rounding,
    # the algorithm numba compiles) decides some of these ties the other way.
    np_wrong = sum(1 for x, d in cases if float(np.round(x, d)) != round(x, d))
    assert np_wrong > 0, "no near-tie case separates np.round from round()"
    assert round(_REVIEW_RAW, 5) == 1.36001
    assert float(np.round(_REVIEW_RAW, 5)) == 1.36
    assert _round_to_digits(_REVIEW_RAW, 5) == 1.36001


def test_stop_rounding_equals_python_round_on_random_levels() -> None:
    rng = np.random.default_rng(20261001)
    for digits, lo, hi in ((5, 0.5, 2.0), (3, 50.0, 200.0), (2, 1000.0, 3000.0), (0, 1.0, 1e5)):
        for x in rng.uniform(lo, hi, 20_000):
            x = float(x)
            assert _round_to_digits(x, digits) == round(x, digits), (x, digits)


# --- R2: a +inf ATR suppresses, as atr_at's None does ---------------------

INF_BAR = 300
# Where the +inf high goes, tried in order until a signal lands inside the
# window it opens (a window only blocks the side the inf channel feeds).
INF_CANDIDATES = (INF_BAR, 180, 420)


def inf_window(inf_at: int) -> frozenset[int]:
    """Bars whose ATR is +inf: [inf_at, inf_at + 13]."""
    return frozenset(range(inf_at, inf_at + 14))


def inf_atr_bars(n: int = 600, *, inf_at: int | None = INF_BAR, flat: range = range(60, 90)) -> BarData:
    """Random-walk H1 bars with one +inf high (true range +inf, so ATR is
    +inf on [inf_at, inf_at+13] and NaN after: inf - inf) and a flat
    stretch (true range 0, so ATR 0) -- every ATR class atr_at rejects."""
    rng = np.random.default_rng(11)
    walk = np.cumsum(rng.normal(0.0, 0.0016, size=n)) + 1.10
    walk[flat.start:flat.stop] = walk[flat.start]
    high = walk + 0.0012
    low = walk - 0.0012
    high[flat.start:flat.stop] = walk[flat.start]
    low[flat.start:flat.stop] = walk[flat.start]
    if inf_at is not None:
        high[inf_at] = np.inf
    return BarData(
        time=np.array([86400 * 3 + i * 3600 for i in range(n)], dtype=np.int64),
        open=walk.copy(), high=high, low=low, close=walk.copy(),
        tick_volume=np.ones(n), spread=np.full(n, 14, dtype=np.int32),
        real_volume=np.zeros(n), symbol=SYMBOL, timeframe=Timeframe.H1,
    )


def test_stop_distances_suppress_exactly_where_atr_at_is_none() -> None:
    bars = inf_atr_bars()
    high, low, close = (np.asarray(a, dtype=np.float64) for a in (bars.high, bars.low, bars.close))
    atr = prep_cell(bars, spec_for(SYMBOL)).atr14
    seen = {"nan": 0, "+inf": 0, "zero": 0, "positive": 0}
    for i in range(atr.size):
        want = atr_at(high, low, close, i)
        suppress, sl_d, tp_d = _stop_distances(atr, i, 1.0, 1.5)
        assert suppress == (want is None), f"bar {i}: atr14={atr[i]!r} atr_at={want!r}"
        if want is not None:
            assert (sl_d, tp_d) == (1.0 * want, 1.5 * want), i
        # No stop asked for: nothing to suppress, whatever the ATR.
        assert _stop_distances(atr, i, 0.0, 0.0) == (False, 0.0, 0.0), i
        a = atr[i]
        seen["nan" if np.isnan(a) else "+inf" if np.isinf(a) else "zero" if a == 0.0 else "positive"] += 1
    assert seen["+inf"] == 14, seen  # the review's case, [INF_BAR, INF_BAR + 13]
    assert all(seen.values()), f"an ATR class was not exercised: {seen}"


def assert_inf_atr_suppressed_identically(entry_signal_bars, label: str) -> None:
    """``entry_signal_bars(bars, **stops)`` runs both engines on ``bars``
    through a harness that asserts EXACT agreement and returns the signal
    bar of every entry. Proves the +inf-ATR case fires (without stops a
    signal lands in the window) and that, with stops, both engines refuse
    every signal there."""
    for inf_at in INF_CANDIDATES:
        bars = inf_atr_bars(inf_at=inf_at, flat=range(0))
        window = inf_window(inf_at)
        atr = prep_cell(bars, spec_for(SYMBOL)).atr14
        assert frozenset(np.flatnonzero(np.isposinf(atr))) == window
        if window & set(entry_signal_bars(bars)):
            gated = entry_signal_bars(bars, sl_atr=1.0, tp_atr=1.5)
            assert not window & set(gated), (label, inf_at, sorted(window & set(gated)))
            return
    pytest.fail(f"{label}: no signal inside any +inf-ATR window {INF_CANDIDATES}; case not exercised")


def test_ma_cross_inf_atr_signals_are_suppressed_identically() -> None:
    def entries(bars: BarData, **kw) -> list[int]:
        slow, _, _ = ma_steps._compare(bars, fast=5, slow=13, label="inf-atr", **kw)
        return [int(t.entry_bar_index) for t in slow.trades]

    assert_inf_atr_suppressed_identically(entries, "MA_CROSS")
