"""Hourly spread matrix calibrated from real tick data.

Spec requirement (S2 hard gate): a strategy that works only with a single
flat spread across all hours is suspicious - real FX spreads have a
rollover spike at the NY close (when liquidity drops) and a London-open
contraction. If the matrix is flat across all 24 hours, the calibration
is broken and downstream cost analysis is meaningless.

The matrix is shape (24, 7): mean spread in points for each
(hour_of_day_broker, day_of_week_broker). Values come from real ticks
fetched via MT5; we bucket by local hour and take the median (robust to
outlier spikes) plus a few other stats for inspection.

Functions:
    calibrate_from_ticks(ticks) -> SpreadMatrix
        Bucket real ticks into a 24x7 matrix.
    synthetic_usdcadm_matrix() -> SpreadMatrix
        Deterministic, gate-passable fallback for tests when no live
        ticks are available. Uses the same shape and shows a real
        rollover spike (NY 22:00 UTC) and a London contraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray


@dataclass(frozen=True)
class SpreadMatrix:
    """24x7 spread matrix (mean spread in points per hour of day x day of week).

    Attributes:
        median: ndarray of shape (24, 7), median spread in points per (h, d)
        p25: ndarray (24, 7), 25th percentile of spread
        p75: ndarray (24, 7), 75th percentile
        count: ndarray (24, 7), number of ticks in each bucket
        point: float, broker's smallest price increment (e.g. 1e-5 for USDCADm)
        source: str, where the data came from
    """

    median: ndarray   # (24, 7)
    p25: ndarray
    p75: ndarray
    count: ndarray
    point: float
    source: str

    def is_flat(self) -> bool:
        """True if the median matrix is constant across all (h, d) cells.

        A flat matrix means the calibration is broken - the S2 gate
        treats this as a failure.
        """
        if self.count.sum() == 0:
            return True
        m = self.median
        non_empty = m[self.count > 0]
        if non_empty.size == 0:
            return True
        return bool(np.allclose(non_empty, non_empty[0]))

    def has_rollover_spike(self, *, min_ratio: float = 1.5) -> bool:
        """True if any (h, d) cell is at least min_ratio times the daily minimum.

        A real broker has a rollover spike at NY close (around 22:00 UTC)
        and possibly at Asian session open. A perfectly flat matrix has
        ratio = 1.0 everywhere - this is the calibration failure.
        """
        if self.count.sum() == 0:
            return False
        m = np.where(self.count > 0, self.median, np.nan)
        finite = m[np.isfinite(m)]
        if finite.size == 0:
            return False
        return bool((finite.max() / max(finite.min(), 1e-12)) >= min_ratio)

    def to_dict(self) -> dict[str, object]:
        return {
            "median": self.median.tolist(),
            "p25": self.p25.tolist(),
            "p75": self.p75.tolist(),
            "count": self.count.tolist(),
            "point": self.point,
            "source": self.source,
            "flat": self.is_flat(),
            "rollover_spike": self.has_rollover_spike(),
        }


def calibrate_from_ticks(
    ticks_time: ndarray,
    ticks_spread: ndarray,
    *,
    point: float,
) -> SpreadMatrix:
    """Bucket ticks into a 24x7 matrix by (hour_of_day, day_of_week).

    Args:
        ticks_time: int64 UNIX seconds (broker-server time).
        ticks_spread: int32 spread in points at each tick.
        point: broker's smallest price increment (e.g. 1e-5).

    Returns:
        SpreadMatrix with median, p25, p75, count per cell.
    """
    if ticks_time.shape != ticks_spread.shape:
        raise ValueError("ticks_time and ticks_spread must have the same shape")
    if ticks_time.size == 0:
        # Empty input: return a zero matrix with the correct shape.
        return SpreadMatrix(
            median=np.zeros((24, 7)),
            p25=np.zeros((24, 7)),
            p75=np.zeros((24, 7)),
            count=np.zeros((24, 7), dtype=np.int64),
            point=point,
            source="empty",
        )
    # Convert UNIX seconds to broker-local (EET/EEST) components.
    # We extract hour and dayofweek via integer arithmetic on ticks_time
    # directly (no datetime64 conversion needed).
    # hour: (t % 86400) // 3600 in broker-local; but our offset_to_utc is
    # applied in the per-date table; here we only need (h, d) bucketing.
    h = (ticks_time // 3600) % 24
    d = ((ticks_time // 86400) + 4) % 7  # 1970-01-01 was Thursday (4)
    median = np.zeros((24, 7))
    p25 = np.zeros((24, 7))
    p75 = np.zeros((24, 7))
    count = np.zeros((24, 7), dtype=np.int64)
    for hour in range(24):
        for dow in range(7):
            mask = (h == hour) & (d == dow)
            n = int(mask.sum())
            count[hour, dow] = n
            if n > 0:
                vals = np.sort(ticks_spread[mask].astype(np.float64))
                median[hour, dow] = float(np.median(vals))
                p25[hour, dow] = float(np.percentile(vals, 25))
                p75[hour, dow] = float(np.percentile(vals, 75))
    return SpreadMatrix(
        median=median,
        p25=p25,
        p75=p75,
        count=count,
        point=point,
        source="ticks",
    )


def synthetic_usdcadm_matrix() -> SpreadMatrix:
    """Deterministic gate-passable spread matrix for USDCADm.

    The shape follows the Exness USD/CAD pair's known pattern:
      - Quietest during London (08:00-11:00 broker local) ~ 12 points
      - Spikes around NY close (22:00 broker local) up to ~ 25 points
      - Slightly elevated in Asian morning (00:00-03:00)
      - Wed (dow=2) and Sun (dow=6) have wider spreads
        (Sun is market open with low liquidity, Wed is rollover day)

    Median profile by hour (broker-local), 0-indexed hours:
      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     16 15 15 14 14 13 13 13 12 12 12 12 13 14 15 15 16 16 17 17 18 19 25 20

    dow multipliers (applied to hourly median):
      Mon=1.0, Tue=1.0, Wed=1.1, Thu=1.0, Fri=1.0, Sat=0.9 (closed), Sun=1.3
    """
    hour_median = np.array(
        [16, 15, 15, 14, 14, 13, 13, 13, 12, 12, 12, 12,
         13, 14, 15, 15, 16, 16, 17, 17, 18, 19, 25, 20],
        dtype=np.float64,
    )
    dow_mult = np.array(
        [1.0, 1.0, 1.1, 1.0, 1.0, 0.9, 1.3],
        dtype=np.float64,
    )
    median = np.outer(hour_median, dow_mult)
    # Sat (dow=5) has very few ticks; set to NaN-equivalent (0) and count=0
    # so the is_flat / has_rollover_spike methods treat it correctly.
    median[:, 5] = 0.0
    return SpreadMatrix(
        median=median,
        p25=median * 0.85,
        p75=median * 1.20,
        count=np.where(median > 0, 1000, 0).astype(np.int64),
        point=1e-5,
        source="synthetic:usdcadm-v1",
    )
