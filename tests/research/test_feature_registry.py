"""Stage C: structural look-ahead safety of the feature registry.

A planted look-ahead oracle (Sharpe 35) survived DSR and PBO (arXiv 2608.27734,
Aug 2026). Statistical correction does not catch leakage, so leakage safety
must be structural — enforced at the data-access layer, not by convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.data.mt5_source import BarData, Timeframe
from engine.research.features import (
    DEFAULT_REGISTRY,
    FeatureRegistrationError,
    FeatureRegistry,
    LookaheadError,
)


def _bars(n: int = 100) -> BarData:
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return BarData(
        time=np.arange(n, dtype=np.int64) * 86400,
        open=close,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        tick_volume=np.ones(n),
        spread=np.zeros(n, dtype=np.int32),
        real_volume=np.zeros(n),
        symbol="TEST",
        timeframe=Timeframe.D1,
    )


# --- registration rejection of malformed features -----------------------------


def test_register_negative_lookback_raises() -> None:
    reg = FeatureRegistry()
    with pytest.raises(FeatureRegistrationError, match="lookback"):
        reg.register("bad", lookback=-1, fn=lambda w: 0.0)


def test_register_future_indexing_feature_is_caught_at_use() -> None:
    """A feature that tries to read t+1 must fail when run, not silently pass."""
    reg = FeatureRegistry()

    def oracle(window):
        return window.close(1)  # look-ahead: bar t+1

    reg.register("oracle", lookback=2, fn=oracle)
    bars = _bars()
    with pytest.raises(LookaheadError):
        reg.compute(bars, "oracle", t=50)


def test_register_duplicate_name_raises() -> None:
    reg = FeatureRegistry()
    reg.register("dup", lookback=2, fn=lambda w: 0.0)
    with pytest.raises(FeatureRegistrationError, match="already registered"):
        reg.register("dup", lookback=2, fn=lambda w: 0.0)


# --- structural guarantee: future is physically absent ------------------------


def test_window_cannot_see_future_bars() -> None:
    """The window slices arrays to end at t, so t+1 is not in the array."""
    reg = FeatureRegistry()
    reg.register("fwd", lookback=3, fn=lambda w: w.close(1))
    bars = _bars()
    # Even offset=1 (the immediate future) must raise because the guard rejects
    # any positive offset before indexing.
    with pytest.raises(LookaheadError):
        reg.compute(bars, "fwd", t=50)


def test_window_only_contains_bars_up_to_t() -> None:
    reg = FeatureRegistry()
    reg.register("v", lookback=5, fn=lambda w: w.close(0))
    bars = _bars()
    window = reg.window_at(bars, "v", t=50)
    # The window's closes must end exactly at bar 50 and contain nothing after.
    visible = window.closes()
    assert visible[-1] == pytest.approx(float(bars.close[50]))
    assert visible.size <= 5
    # The underlying close at bar 51 must NOT be reachable.
    with pytest.raises(LookaheadError):
        window.close(1)


def test_resolve_features_rejects_unregistered_name() -> None:
    reg = FeatureRegistry()
    reg.register("sma_20", lookback=20, fn=lambda w: 0.0)
    # A strategy/DNA naming an unregistered feature is rejected at construction.
    with pytest.raises(FeatureRegistrationError, match="not registered"):
        reg.resolve_features(("sma_20", "nonexistent"))


# --- planted-leak sentinel still fires ----------------------------------------


def test_planted_leak_sentinel_fires() -> None:
    """A feature sneaking a forward read must be caught — the whole point."""
    reg = FeatureRegistry()

    def leaky(window):
        # Build a 'winning' signal by peeking one bar ahead, exactly what an
        # oracle would do. The registry must refuse to run it.
        return window.high(1) - window.low(1)

    reg.register("leak", lookback=3, fn=leaky)
    bars = _bars()
    with pytest.raises(LookaheadError):
        reg.compute(bars, "leak", t=42)


def test_default_registry_has_reference_features() -> None:
    bars = _bars()
    val = DEFAULT_REGISTRY.compute(bars, "sma_20", t=50)
    assert np.isfinite(val)
    # last_close at t == close[t]
    assert DEFAULT_REGISTRY.compute(bars, "last_close", t=50) == pytest.approx(
        float(bars.close[50])
    )


# --- a DNA naming an unregistered feature dies at CONSTRUCTION -------------
def test_dna_naming_an_unregistered_feature_is_rejected_at_construction(
    monkeypatch,
) -> None:
    """Not at validation. By validation the genome has been scored and stored.

    The gene vocabulary names FAMILIES ("RSI_REVERSION"), not features.
    FAMILY_FEATURES maps a family to the registry entries it needs, and
    DNA.__post_init__ resolves them, so an unbacked family cannot become a
    trial.
    """
    from engine.research.features import registry as reg
    from engine.research.search.dna import DNA, DNA_SCHEMA_VERSION

    def _dna(signal: str) -> DNA:
        return DNA(
            bias="NONE", filter="NONE", signal=signal, signal2="",
            exec_mode="MARKET2", risk="FIXED_ATR", mgmt="NONE",
            sl_atr=1.0, tp_atr=2.0, risk_pct=0.01,
            friday_close=False, news_filter=False,
            daily_dd_limit=0.03, max_agg_risk=0.02,
            max_feature_lookback=40, max_hold_time=20,
            window_start_hour_utc=0, window_end_hour_utc=23,
            schema_version=DNA_SCHEMA_VERSION,
        )

    # A family that claims a feature nobody registered.
    monkeypatch.setattr(reg, "FAMILY_FEATURES", {"RSI_REVERSION": ("nonexistent_feature",)})
    with pytest.raises(FeatureRegistrationError, match="nonexistent_feature"):
        _dna("RSI_REVERSION")

    # Point the same family at a feature that IS registered, and it builds.
    monkeypatch.setattr(reg, "FAMILY_FEATURES", {"RSI_REVERSION": ("sma_20",)})
    assert _dna("RSI_REVERSION").signal == "RSI_REVERSION"


def test_every_gene_family_is_currently_unbacked() -> None:
    """Pins the gap so implementing a family forces it to be registered.

    This is the same finding the campaign's search-space audit reached from
    the other direction: real_backtest_factory reads two DNA fields and the
    only implemented family is MA-crossover. If someone wires up RSI without
    registering its features, this test changes and they have to look.
    """
    from engine.research.features.registry import unimplemented_families
    from engine.research.search.dna import (
        BIAS_VALUES,
        FILTER_VALUES,
        MGMT_VALUES,
        SIGNAL_VALUES,
    )

    families = list(BIAS_VALUES) + list(FILTER_VALUES) + list(SIGNAL_VALUES) + list(
        MGMT_VALUES
    )
    unbacked = set(unimplemented_families(families))
    # B4: phantom families (STOCH, MACD, DONCHIAN, M_BREAK, ENGULFING, BB_REV)
    # are removed from the gene space. Only the genuinely unbacked signal
    # families that remain in SIGNAL_VALUES are reported here.
    # W1: STOCH_CROSS, MACD_CROSS, DONCHIAN_BREAKOUT are now backed by
    # registered features + strategy classes + dna_to_*_config dispatchers.
    # MA_CROSS is NOT in FAMILY_FEATURES — it is the kernel gate strategy,
    # expressible directly without a feature lookup (see DECISIONS.md).
    assert unbacked == {
        "DAILY_TREND", "H4_TREND", "WEEKLY_TREND", "NONE",
        "CCI", "ADX", "ATR_PCT", "RANGE_COMP",
        "BREAKEVEN", "TRAIL_ATR", "PARTIAL",
        "MA_CROSS",
    }, sorted(unbacked)
    # The three W1 families are registered in FAMILY_FEATURES, so they
    # must NOT appear as unbacked.
    for family in ("STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT"):
        assert family not in unbacked, f"{family} should be backed now"


def test_require_registered_features_resolves_a_backed_family() -> None:
    from engine.research.features.registry import require_registered_features

    resolved = require_registered_features(
        ("MA",), mapping={"MA": ("sma_20", "last_close")}
    )
    assert resolved == ("sma_20", "last_close")

    with pytest.raises(FeatureRegistrationError):
        require_registered_features(("MA",), mapping={"MA": ("nope",)})


def test_register_zero_lookback_raises() -> None:
    """Zero is refused as well as negative.

    A zero-lookback feature sees an empty window, so whatever it returns did
    not come from the bars: it is a constant, or a closure over something
    outside the window. Both are leaks waiting to happen.
    """
    reg = FeatureRegistry()
    with pytest.raises(FeatureRegistrationError, match="must be >= 1"):
        reg.register("zero", lookback=0, fn=lambda w: 0.0)
    with pytest.raises(FeatureRegistrationError, match="must be >= 1"):
        reg.register("neg", lookback=-1, fn=lambda w: 0.0)
    assert reg.names() == ()


def test_lookahead_error_spellings_are_the_same_class() -> None:
    """Catching one spelling must catch the other."""
    from engine.research.features.registry import LookAheadError, LookaheadError

    assert LookAheadError is LookaheadError
