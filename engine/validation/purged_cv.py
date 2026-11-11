"""Purged + embargoed cross-validation splitter.

WHAT: Generate train/test index masks for time-series cross-validation that
      respect two leakage sources: overlapping labels (purging) and
      post-test information bleed (embargo).
WHY: A naive ``train_idx < test_idx`` split leaks information when:
      (a) a feature uses past ``lookback`` bars (the train set's last rows
          see the test set's first rows), and
      (b) the label at time ``t`` depends on bars up to time ``t + horizon``
          (the train set contains the label-generating region for the test
          set's first rows).
HOW: For a chronological split at index ``t``, drop train rows whose label
      depends on a bar >= t (purging) and drop test rows whose label
      depends on a bar <= t (also purging). Then add an embargo of
      ``pct_embargo`` of the total samples after each test block.

      This module produces **splits** (masks + metadata), not a fitted
      model. The caller runs the strategy on each split.

TRADE-OFF: We do not know the model's label horizon in general. The caller
      passes ``horizon_bars`` and the function uses it. Purging is
      conservative; this can reduce effective sample size, which is the
      intended behavior.
TEST: ``tests/validation/test_purged_cv.py`` — synthetic data with a known
      leak; assert the splitter prevents the leak.
FAILURE: Invalid parameters raise ``DataError``.

References: López de Prado (2018), ch. 7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.core.errors import DataError


@dataclass(frozen=True)
class PurgedSplit:
    """A single purged + embargoed train/test split.

    Attributes:
        train_idx: Indices into the returns array used for training.
        test_idx: Indices used for testing.
        purge_count: Number of train rows purged at the boundary.
        embargo_count: Number of test rows embargoed at the boundary.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    purge_count: int
    embargo_count: int


@dataclass(frozen=True)
class PurgedCVConfig:
    """Configuration for purged + embargoed cross-validation.

    Attributes:
        n_splits: Number of chronological train/test splits (>= 2).
        horizon_bars: How many bars after index ``t`` the label at ``t``
            depends on (the conservative upper bound). Rows in
            ``[t - horizon_bars + 1, t + horizon_bars - 1]`` are purged
            across the train/test boundary.
        pct_embargo: Fraction of the total sample size to embargo AFTER
            each test block (in [0, 1)).
    """

    n_splits: int
    horizon_bars: int
    pct_embargo: float

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise DataError(f"n_splits must be >= 2; got {self.n_splits}.")
        if self.horizon_bars < 0:
            raise DataError(f"horizon_bars must be >= 0; got {self.horizon_bars}.")
        if not 0.0 <= self.pct_embargo < 1.0:
            raise DataError(f"pct_embargo must be in [0, 1); got {self.pct_embargo}.")


def purged_cv(n_samples: int, config: PurgedCVConfig) -> list[PurgedSplit]:
    """Generate purged + embargoed chronological train/test splits.

    Args:
        n_samples: Total length of the time series (>= 2).
        config: :class:`PurgedCVConfig` with split count, horizon, embargo.

    Returns:
        List of :class:`PurgedSplit` of length ``config.n_splits``.

    Raises:
        DataError: If ``n_samples`` is too small for the requested config.
    """
    if n_samples < 4:
        raise DataError(f"n_samples must be >= 4; got {n_samples}.")
    if config.n_splits >= n_samples:
        raise DataError(
            f"n_splits ({config.n_splits}) must be < n_samples ({n_samples})."
        )

    # Equally sized chronological test blocks.
    test_size = n_samples // (config.n_splits + 1)
    if test_size < 1:
        raise DataError("Computed test block size < 1; reduce n_splits.")

    splits: list[PurgedSplit] = []
    embargo_count = int(round(n_samples * config.pct_embargo))
    horizon = config.horizon_bars

    for k in range(config.n_splits):
        test_start = (k + 1) * test_size
        test_end = test_start + test_size
        train_idx = np.arange(0, test_start, dtype=np.int64)

        # Purge: drop train rows within `horizon` of test_start (their labels
        # depend on bars in the test region).
        if horizon > 0:
            purge_lo = max(0, test_start - horizon)
            purge_mask = (train_idx >= purge_lo) & (train_idx < test_start)
            train_idx = train_idx[~purge_mask]
            purge_count = int(purge_mask.sum())
        else:
            purge_count = 0

        # Embargo: drop the first `embargo_count` test rows.
        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        if embargo_count > 0 and test_idx.size > 0:
            keep = max(0, test_idx.size - embargo_count)
            test_idx = test_idx[keep:]
            applied_embargo = min(embargo_count, max(0, (test_end - test_start) - keep))
        else:
            applied_embargo = 0

        splits.append(
            PurgedSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                purge_count=purge_count,
                embargo_count=applied_embargo,
            )
        )
    return splits
