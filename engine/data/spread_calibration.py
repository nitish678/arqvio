"""Spread calibration: 24 hourly buckets per symbol.

A "flat 24-hour row" - the same spread value at every hour of the day - is
a CALIBRATION FAILURE, not a result. Most free data sources give a single
constant spread; that constant is a no-information signal and the campaign
must refuse to launch on a symbol with such a row.

The persisted artefact is a 24-element float array (one bucket per UTC
hour of day). 192 bytes per symbol. Four weeks of ticks (~78 MB per
symbol) compress to 192 bytes once the matrix is written; the ticks are
deleted immediately.
"""
from __future__ import annotations

import enum
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from engine.core.paths import get_artifacts_dir


class DataError(Exception):
    """Raised when calibration data is missing, malformed, or incomplete."""

# A spread profile this far from flat is not a profile, it is an artefact.
SUSPICIOUS_RATIO = 50.0

# Bumped when the persisted SpreadMatrix gains a field the cost path depends on.
# v1 = buckets only (median); v2 = +mean_buckets (W0 cost model). Artifacts
# stamped v1 are refused on load so the cost path can never silently consume
# medians while claiming to use means.
CALIBRATOR_VERSION = 2
CALIBRATOR_VERSION_MIN = 2


def spread_dir() -> Path:
    """Where persisted 24-hour rows live: ARQVIO_HOME/artifacts/spread, resolved
    at CALL time (CLAUDE.md s7). scripts/calibrate_spreads.py persists here;
    the launch gate, the cost path and the API readers all read here."""
    return get_artifacts_dir() / "spread"


@dataclass(frozen=True)
class SpreadMatrix:
    """24 hourly buckets (UTC), in points.

    buckets: per-hour MEDIAN spread (robust to outlier spikes). Consumed by
        the F1 no-trade gate (is_flat/no_trade_hours) and the A5 p95 max-spread
        gate. UNCHANGED by R1.
    mean_buckets: per-hour MEAN spread over cleaned positive ticks. Consumed by
        the W0 cost model (expected cost per fill). An hour with zero valid
        positive ticks records 0.0 — that hour is already a no-trade candidate
        via the median path (F1 gate), so a zero-mean hour is unreachable inside
        the backtest (no fills there). See cleaning rule in calibrate_spreads.py.
    """

    symbol: str
    buckets: tuple[float, ...]  # length 24 — MEDIAN
    mean_buckets: tuple[float, ...]  # length 24 — MEAN over positive ticks
    sample_count: int
    source: str  # "MT5" or "BUNDLED"
    calibrator_version: int = CALIBRATOR_VERSION
    calibrated_utc: str = ""
    time_basis: str = "utc"  # basis the tick timestamps were bucketed in

    def spread_percentile_points(self, percentile: float) -> float | None:
        """A5: the max-spread entry gate threshold, from this symbol's own row.

        Hardcoding a pip count would mean one number for gold and
        EURUSD, which are two orders of magnitude apart. The gate is
        the given percentile of the symbol's OWN calibrated hourly
        spreads, so it adapts to the instrument.

        Zero-spread hours are excluded: they are the F1 no-trade window,
        already handled, and including them would drag the percentile
        down and gate out ordinary bars.

        Returns None when the row carries no usable (non-zero) spread,
        which disables the gate rather than inventing a threshold.
        """
        import numpy as np

        usable = [float(v) for v in self.buckets if float(v) > 0.0]
        if not usable:
            return None
        return float(np.percentile(np.asarray(usable, dtype=float), percentile))

    @property
    def no_trade_hours(self) -> frozenset[int]:
        """Hours whose median spread is zero.

        A zero spread is a free trade: enter and exit at the same price with
        no cost. Any search will find that and build a strategy on it, and
        the strategy will not survive contact with a real broker. These hours
        are untradeable, not cheap.

        Derived rather than stored, so a matrix loaded from an artifact
        written before this existed still reports them correctly.
        """
        return frozenset(h for h, v in enumerate(self.buckets) if float(v) == 0.0)

    def to_dict(self) -> dict[str, object]:
        # no_trade_hours is computed on load so an older artifact is not
        # trusted to carry it. mean_buckets is persisted alongside buckets
        # so the cost path can refuse artifacts that lack it.
        # calibrator_version/calibrated_utc are stamped here so a hand-forged
        # artifact without them is refused by from_json.
        data = asdict(self)
        data["no_trade_hours"] = sorted(self.no_trade_hours)
        data["calibrator_version"] = CALIBRATOR_VERSION
        data["calibrated_utc"] = self.calibrated_utc or ""
        data["time_basis"] = self.time_basis
        return data

    @property
    def is_flat(self) -> bool:
        """True iff every bucket is identical to the first - a calibration
        failure, not a real spread profile."""
        if not self.buckets:
            return True
        first = self.buckets[0]
        return all(b == first for b in self.buckets)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> SpreadMatrix:
        data = json.loads(text)
        if "mean_buckets" not in data:
            raise DataError(
                f"artifact for {data.get('symbol', '?')} lacks mean_buckets; "
                "recalibrate with the mean-aware calibrator before any cost "
                "analysis"
            )
        version = data.get("calibrator_version")
        if version is None:
            raise DataError(
                f"artifact for {data.get('symbol', '?')} has no "
                "calibrator_version; recalibrate with the mean-aware "
                "calibrator before any cost analysis"
            )
        if int(version) < CALIBRATOR_VERSION_MIN:
            raise DataError(
                f"artifact for {data.get('symbol', '?')} is v{int(version)}, "
                f"need v{CALIBRATOR_VERSION_MIN}; recalibrate before any "
                "cost analysis"
            )
        return cls(
            symbol=str(data["symbol"]),
            buckets=tuple(float(x) for x in data["buckets"]),
            mean_buckets=tuple(float(x) for x in data["mean_buckets"]),
            sample_count=int(data["sample_count"]),
            source=str(data["source"]),
            calibrator_version=int(data.get("calibrator_version", CALIBRATOR_VERSION)),
            calibrated_utc=str(data.get("calibrated_utc", "")),
            time_basis=str(data.get("time_basis", "utc")),
        )


def persist(matrix: SpreadMatrix, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{matrix.symbol}.json"
    path.write_text(matrix.to_json(), encoding="utf-8")
    return path


def load(symbol: str, target_dir: Path) -> SpreadMatrix | None:
    path = target_dir / f"{symbol}.json"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    try:
        return SpreadMatrix.from_json(text)
    except DataError:
        import warnings

        warnings.warn(
            f"artifact for {symbol} lacks mean_buckets; recalibrate with "
            "scripts/calibrate_spreads.py before cost analysis. Returning None.",
            stacklevel=2,
        )
        return None


# a flat row is accepted as a fixed-spread account only when measured from at least this many of the broker's ticks
FIXED_SPREAD_MIN_TICKS = 10_000


class SpreadStatus(str, enum.Enum):
    """Graded verdict on a 24-hour spread row.

    The three failure modes are different facts and are reported as such:

    - CALIBRATION_FAILURE: every hour identical. Most free feeds hand back a
      single constant spread; a constant carries no information about when
      trading is expensive, so any backtest costed from it is costed from a
      number somebody made up. Blocks.
    - ZERO_SPREAD: some hour is 0.0. A zero-cost hour is free money to any
      strategy that can choose its hour, and it is never real. Does NOT
      block (F9). It used to refuse the whole cell, which was the right
      answer only while the backtester had no way to act on it. F1 now
      derives no_trade_hours from exactly these hours and refuses every
      fill inside them, so the correct handling is to run the cell with
      the window shut, not to refuse a symbol that is otherwise properly
      calibrated. Refusing it also made the gate untestable: the only
      calibrated symbols with a zero hour could never reach the engine.
    - SUSPICIOUS: max/min exceeds SUSPICIOUS_RATIO. Real FX and metals rows
      do vary by several times across the session, but a 50x swing is far
      more likely a bad tick or a mixed-source join than a session effect.
      Warns; does not block, because it can legitimately happen on an
      illiquid symbol and refusing would be a false negative on real data.
    """

    OK = "OK"
    FIXED_SPREAD = "FIXED_SPREAD"
    SUSPICIOUS = "SUSPICIOUS"
    ZERO_SPREAD = "ZERO_SPREAD"
    CALIBRATION_FAILURE = "CALIBRATION_FAILURE"


@dataclass(frozen=True)
class SpreadCheck:
    symbol: str
    status: SpreadStatus
    blocks: bool
    reason: str
    ratio: float | None = None

    def line(self) -> str:
        return f"{self.symbol}: {self.status.value} - {self.reason}"


def evaluate(matrix: SpreadMatrix) -> SpreadCheck:
    """Grade a spread row. Checked in severity order, worst first."""
    buckets = tuple(float(b) for b in matrix.buckets)
    if not buckets:
        return SpreadCheck(
            symbol=matrix.symbol,
            status=SpreadStatus.CALIBRATION_FAILURE,
            blocks=True,
            reason="no spread buckets at all",
        )
    if len(buckets) != 24:
        return SpreadCheck(
            symbol=matrix.symbol,
            status=SpreadStatus.CALIBRATION_FAILURE,
            blocks=True,
            reason=f"expected 24 hourly buckets, got {len(buckets)}",
        )
    if all(b == buckets[0] for b in buckets) and buckets[0] > 0 and matrix.source == "MT5"             and matrix.sample_count >= FIXED_SPREAD_MIN_TICKS:
        # 2026-11-01 (owner's AiwaCapital XAGUSD, 21 pt every hour): a flat row MEASURED from weeks of the broker's
        # own ticks is a fixed-spread account, which is information -- costs are exact, not made up. Only a flat row
        # without that measurement (a bundled or free-feed constant) is the no-information failure below.
        return SpreadCheck(
            symbol=matrix.symbol,
            status=SpreadStatus.FIXED_SPREAD,
            blocks=False,
            reason=f"fixed {buckets[0]:g}pt spread every hour, measured from {matrix.sample_count:,} broker ticks",
        )
    if all(b == buckets[0] for b in buckets):
        return SpreadCheck(
            symbol=matrix.symbol,
            status=SpreadStatus.CALIBRATION_FAILURE,
            blocks=True,
            reason=(
                f"flat {buckets[0]:g}pt row across all 24 hours; a constant "
                "spread is a no-information signal, not a profile"
            ),
        )
    zero_hours = [h for h, b in enumerate(buckets) if b == 0.0]
    if zero_hours:
        # F9: informational, not a refusal. F1 shuts these hours inside the
        # backtester (SpreadMatrix.no_trade_hours -> BacktestConfig), so the
        # cell runs with the window closed instead of not running at all.
        return SpreadCheck(
            symbol=matrix.symbol,
            status=SpreadStatus.ZERO_SPREAD,
            blocks=False,
            reason=(
                f"untradeable hour(s) in sim: {zero_hours} (spread is 0.0; "
                "a zero-cost hour is free money to anything that can pick "
                "its hour, so nothing fills there)"
            ),
        )
    lo, hi = min(buckets), max(buckets)
    ratio = hi / lo if lo > 0 else float("inf")
    if ratio > SUSPICIOUS_RATIO:
        return SpreadCheck(
            symbol=matrix.symbol,
            status=SpreadStatus.SUSPICIOUS,
            blocks=False,
            reason=(
                f"max/min = {ratio:.1f}x (> {SUSPICIOUS_RATIO:g}x); likely a "
                "bad tick or a mixed-source join rather than a session effect"
            ),
            ratio=ratio,
        )
    return SpreadCheck(
        symbol=matrix.symbol,
        status=SpreadStatus.OK,
        blocks=False,
        reason=f"{lo:g}-{hi:g}pt across the session ({ratio:.1f}x)",
        ratio=ratio,
    )


# The spec's name and signature: a bare 24-element row in, a validation out.
SpreadValidation = SpreadCheck


def validate_spread_matrix(
    matrix: SpreadMatrix | Sequence[float],
    *,
    symbol: str = "<unknown>",
) -> SpreadValidation:
    """Grade a spread row given either a SpreadMatrix or a bare sequence.

    Accepts the raw 24-element row so a caller holding an ndarray does not
    have to build a SpreadMatrix just to ask the question.
    """
    if isinstance(matrix, SpreadMatrix):
        return evaluate(matrix)
    mean = tuple(float(x) for x in matrix)
    return evaluate(
        SpreadMatrix(
            symbol=symbol,
            buckets=mean,
            mean_buckets=mean,
            sample_count=0,
            source="UNKNOWN",
        )
    )


def is_calibration_failure(matrix: SpreadMatrix) -> bool:
    """Back-compatible flat-row predicate. Prefer ``evaluate``."""
    return matrix.is_flat


def calibration_failure_message(symbol: str) -> str:
    return (
        f"SPREAD_UNCALIBRATED: {symbol} returned a flat 24-hour spread row. "
        "Calibration re-run required before any campaign can use this symbol."
    )


__all__ = [
    "SUSPICIOUS_RATIO",
    "DataError",
    "SpreadCheck",
    "SpreadMatrix",
    "SpreadStatus",
    "SpreadValidation",
    "calibration_failure_message",
    "evaluate",
    "is_calibration_failure",
    "load",
    "persist",
    "spread_dir",
    "validate_spread_matrix",
]
