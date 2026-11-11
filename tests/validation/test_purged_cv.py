"""Purged + embargoed CV tests."""
from __future__ import annotations

import numpy as np
import pytest

from engine.core.errors import DataError
from engine.validation.purged_cv import PurgedCVConfig, purged_cv


def test_purged_cv_config_validates() -> None:
    with pytest.raises(DataError):
        PurgedCVConfig(n_splits=1, horizon_bars=0, pct_embargo=0.0)
    with pytest.raises(DataError):
        PurgedCVConfig(n_splits=2, horizon_bars=-1, pct_embargo=0.0)
    with pytest.raises(DataError):
        PurgedCVConfig(n_splits=2, horizon_bars=0, pct_embargo=1.5)


def test_purged_cv_basic_shapes() -> None:
    n = 1000
    cfg = PurgedCVConfig(n_splits=3, horizon_bars=0, pct_embargo=0.0)
    splits = purged_cv(n, cfg)
    assert len(splits) == 3
    for s in splits:
        assert s.train_idx.size + s.test_idx.size <= n
        # Train comes strictly before test.
        assert s.train_idx.max() < s.test_idx.min()
        # No overlap.
        assert np.intersect1d(s.train_idx, s.test_idx).size == 0


def test_purged_cv_horizon_drops_train_rows() -> None:
    n = 1000
    cfg_no_horizon = PurgedCVConfig(n_splits=3, horizon_bars=0, pct_embargo=0.0)
    cfg_with_horizon = PurgedCVConfig(n_splits=3, horizon_bars=50, pct_embargo=0.0)
    splits_a = purged_cv(n, cfg_no_horizon)
    splits_b = purged_cv(n, cfg_with_horizon)
    # The first split in B should have fewer train rows than A (purge).
    assert splits_b[0].train_idx.size < splits_a[0].train_idx.size
    assert splits_b[0].purge_count == 50


def test_purged_cv_embargo_drops_test_rows() -> None:
    n = 1000
    cfg_no_embargo = PurgedCVConfig(n_splits=3, horizon_bars=0, pct_embargo=0.0)
    cfg_with_embargo = PurgedCVConfig(n_splits=3, horizon_bars=0, pct_embargo=0.10)
    splits_a = purged_cv(n, cfg_no_embargo)
    splits_b = purged_cv(n, cfg_with_embargo)
    # The first test block should be smaller in B (embargo trims from the front).
    assert splits_b[0].test_idx.size < splits_a[0].test_idx.size


def test_purged_cv_no_leak_under_label_horizon() -> None:
    """Conservative property: train and test blocks are separated by at least
    ``horizon_bars`` rows when pct_embargo is also >= horizon/size."""
    n = 500
    horizon = 30
    cfg = PurgedCVConfig(
        n_splits=2,
        horizon_bars=horizon,
        pct_embargo=0.0,  # embargo off; rely on purging only
    )
    splits = purged_cv(n, cfg)
    for s in splits:
        if s.train_idx.size == 0 or s.test_idx.size == 0:
            continue
        # After purging, the first train row is at least `horizon` before test_start.
        test_start = int(s.test_idx.min())
        first_train_after_purge = int(s.train_idx.max()) if s.train_idx.size else -1
        # The last surviving train row must be at least `horizon` before the test block.
        assert (test_start - first_train_after_purge) >= horizon


def test_purged_cv_too_small_raises() -> None:
    with pytest.raises(DataError):
        purged_cv(2, PurgedCVConfig(n_splits=2, horizon_bars=0, pct_embargo=0.0))
