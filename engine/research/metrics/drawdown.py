"""Drawdown metrics from the equity curve.

WHAT: ``max_drawdown`` (peak-to-trough as a negative fraction) and
      ``time_to_recovery_bars`` (the bar index at which the equity
      curve first returns to the level of the prior peak after the
      drawdown begins; ``None`` if the curve never recovers).

WHY: Max DD is the headline risk number; time-to-recovery distinguishes
     "drew down and bounced" from "drew down and never came back" - the
     latter is much worse even when the magnitude is similar.

HOW: ``np.maximum.accumulate`` for the running peak; the trough index
     of the global max drawdown is the local minimum. Recovery is
     then the first index after the trough where equity >= the
     peak-at-trough. We measure in BARS (the engine's natural unit),
     not days, so the value is comparable across instruments.

TRADE-OFF: We define recovery on a CLOSE-TO-CLOSE basis. A strategy
           that uses intrabar equity to recover and then loses it
           again by the bar's close would show "recovered" on the
           same bar - this is the conservative (faster) choice and
           matches what an external observer would record.
"""
from __future__ import annotations

import numpy as np


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a NEGATIVE fraction.

    Returns ``-0.18`` if the worst peak-to-trough drop was 18%.
    Returns ``0.0`` for a monotonically non-decreasing series and
    ``0.0`` for an empty / constant series. ``NaN`` / ``Inf`` entries
    are treated as missing (the value at that index is skipped).
    """
    e = np.asarray(equity, dtype=np.float64).ravel()
    e = e[np.isfinite(e)]
    if e.size < 2:
        return 0.0
    peak = np.maximum.accumulate(e)
    # Guard against a zero peak (would blow up division).
    safe_peak = np.where(peak == 0, 1.0, peak)
    dd = e / safe_peak - 1.0
    return float(np.min(dd))


def time_to_recovery_bars(equity: np.ndarray) -> int | None:
    """Bar index of first full recovery from the global max-drawdown
    trough, measured from the start of the series (0-indexed).

    Returns ``None`` if the series never recovers. Returns ``0`` if
    the max-drawdown trough is the very first bar (no recovery was
    ever needed). Returns ``None`` for empty / too-short / fully-flat
    series (max DD = 0).
    """
    e = np.asarray(equity, dtype=np.float64).ravel()
    e = e[np.isfinite(e)]
    if e.size < 2:
        return None
    peak = np.maximum.accumulate(e)
    dd = e - peak  # always <= 0
    min_dd = float(np.min(dd))
    if min_dd == 0.0:
        return None  # no drawdown; recovery is not defined
    trough_idx = int(np.argmin(dd))
    peak_at_trough = float(peak[trough_idx])
    # First index after the trough where equity >= peak_at_trough.
    after = e[trough_idx + 1 :]
    if after.size == 0:
        return None
    recovered = np.where(after >= peak_at_trough)[0]
    if recovered.size == 0:
        return None
    return int(trough_idx + 1 + int(recovered[0]))
