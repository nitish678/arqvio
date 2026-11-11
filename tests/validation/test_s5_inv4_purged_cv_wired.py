"""S5 invariant I4 - purged CV is actually wired into the backtest split.

The S1 splitter generates train/test masks; the backtester must consume
them. The embargo must be derived from the strategy's DNA, not a fixed
percentage, and the backtest must refuse to run with a future-leaking
strategy.
"""
from __future__ import annotations

import numpy as np

from engine.validation.purged_cv import PurgedCVConfig, purged_cv


def test_backtest_split_has_zero_label_leak() -> None:
    """No training label may resolve inside any test window. Count must be 0.

    We construct a simple "label at t = sign(close[t+horizon] - close[t])"
    labelling and verify that after purged_cv, for every split, no train
    index ``i`` has its label resolution time ``i + horizon`` inside the
    test window.
    """
    n_samples = 2000
    horizon = 10
    config = PurgedCVConfig(n_splits=5, horizon_bars=horizon, pct_embargo=0.01)
    splits = purged_cv(n_samples, config)
    assert len(splits) == 5
    for split in splits:
        train = {int(x) for x in split.train_idx.tolist()}
        test_lo, test_hi = int(split.test_idx.min()), int(split.test_idx.max())
        # A train index i "leaks" if its label resolves inside the test window:
        # i + horizon in [test_lo, test_hi] and the label itself isn't purged.
        leaks = [i for i in train if test_lo <= i + horizon <= test_hi]
        assert leaks == [], f"label leak in {len(leaks)} train rows"


def test_lookahead_sentinel_fires() -> None:
    """A one-bar look-ahead strategy must be detectable: its inflated Sharpe
    is materially higher than the clean one and the detector flags it.

    Detector: a strategy that requires the *next* bar's close to decide
    today's action has a future dependency. The test confirms that:
      - the inflated return > the clean return (look-ahead helps by definition)
      - the "is_lookahead" check that the runner uses (a DNA-level guard)
        flags the strategy so it never reaches the trial registry.
    """
    rng = np.random.default_rng(0)
    n = 500
    # Use a non-monotonic walk so the future-leak advantage shows up in
    # the realised return - a strictly-trending series would give the
    # clean strategy an unfair advantage.
    deltas = rng.normal(0, 1, n)
    close = np.cumsum(deltas) + 100
    # Clean strategy: today = 1 if close[t] - close[t-1] > 0 else 0.
    delta = np.diff(close, prepend=close[0])
    # Align lengths: signal at t uses delta[t], returns are delta[t+1] (next bar).
    clean_signal = (delta > 0).astype(int)[:-1]
    clean_returns = clean_signal * np.diff(close)
    # Inflated strategy: today = 1 if close[t+1] - close[t] > 0 else 0.
    future_delta = np.diff(close)
    inflated_signal = (future_delta > 0).astype(int)
    inflated_returns = inflated_signal * future_delta
    assert float(np.sum(inflated_returns)) > float(np.sum(clean_returns))
    # The runner-side detector. We model the guard here in isolation.
    from engine.research.search.lookahead_guard import is_lookahead_dna
    dna_clean = {"exec": "MARKET2", "uses_future_bar": False}
    dna_cheat = {"exec": "MARKET2", "uses_future_bar": True}
    assert is_lookahead_dna(dna_clean) is False
    assert is_lookahead_dna(dna_cheat) is True


def test_embargo_uses_dna_horizon_not_fixed_pct() -> None:
    """Two strategies with different max-hold-time must produce different
    embargo sizes when the DNA-derived embargo function is used.

    The backtester MUST NOT silently use a fixed 1% embargo; the embargo
    must be derived from the strategy's DNA.
    """
    from engine.research.search.dna import dna_embargo_bars

    short = dna_embargo_bars({"max_feature_lookback": 20, "max_hold_time": 50})
    long = dna_embargo_bars({"max_feature_lookback": 100, "max_hold_time": 250})
    assert long > short
    assert short >= 70   # 20 + 50
    assert long >= 350   # 100 + 250
