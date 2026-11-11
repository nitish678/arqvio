"""Feature registry — structural look-ahead safety.

Motivation (arXiv 2608.27734, Aug 2026): a deliberately planted look-ahead
oracle posting Sharpe 35 survived BOTH Deflated Sharpe and PBO completely.
Statistical correction does not catch leakage. Leakage safety must be
STRUCTURAL — enforced at the data-access layer, not by convention.

The contract:

1. Every feature is a registered function with a DECLARED integer lookback
   (>= 0). Registration with a negative lookback raises at registration time.

2. A feature receives a ``BarWindow`` positioned at bar ``t``. The window
   exposes only bars in ``[t - lookback + 1, t]``. Any attempt to read at a
   relative offset > 0 (i.e. bar t+1 or beyond) raises ``LookaheadError`` —
   the future is physically absent from the array the feature can see, and the
   index guard catches even a computed forward index.

3. Strategies reference features BY NAME through the registry. A DNA naming an
   unregistered feature is rejected at construction (``resolve_features``).

The guarantee is physical, not advisory: the window slices the underlying
arrays so ``window.close(0)`` is bar t, ``window.close(-1)`` is t-1, and
``window.close(1)`` raises. A feature cannot see bar t+1 because it is not in
the array and the guard rejects the index.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from engine.core.errors import DataError

from engine.research.features.donchian import donchian_high_series, donchian_low_series
from engine.research.features.macd import macd_line_series, macd_signal_series
from engine.research.features.rsi import DEFAULT_RSI_PERIOD, rsi_series
from engine.research.features.stoch import stoch_d_series, stoch_k_series

if TYPE_CHECKING:
    from engine.research.data.mt5_source import BarData


class LookaheadError(DataError):
    """Raised when a feature attempts to read a bar at or beyond t+1."""


# The spec spells it LookAheadError; both names are the same class so
# neither spelling silently fails to catch the other.
LookAheadError = LookaheadError


class FeatureRegistrationError(DataError):
    """Raised when a feature declaration is malformed."""


@dataclass(frozen=True)
class BarWindow:
    """A read-only view of bars up to and including bar ``t``.

    Relative indexing: offset 0 is bar t (the newest visible bar), -1 is t-1,
    and so on down to ``-(lookback-1)``. A positive offset is a look-ahead and
    raises immediately. The underlying arrays are sliced to end at t, so even a
    guard bypass cannot reach future data.
    """

    _open: np.ndarray
    _high: np.ndarray
    _low: np.ndarray
    _close: np.ndarray
    _volume: np.ndarray
    t: int
    lookback: int

    def _idx(self, offset: int) -> int:
        if offset > 0:
            raise LookaheadError(
                f"feature read at offset={offset} (bar t+{offset}); the future "
                "is not visible at bar t."
            )
        if offset <= -self.lookback:
            raise DataError(
                f"feature read at offset={offset} exceeds declared "
                f"lookback={self.lookback}."
            )
        # Arrays are sliced to end at t, so t is the last element.
        idx = len(self._close) - 1 + offset
        if idx < 0:
            raise DataError(
                f"feature read at offset={offset} before start of window."
            )
        return idx

    def open(self, offset: int = 0) -> float:
        return float(self._open[self._idx(offset)])

    def high(self, offset: int = 0) -> float:
        return float(self._high[self._idx(offset)])

    def low(self, offset: int = 0) -> float:
        return float(self._low[self._idx(offset)])

    def close(self, offset: int = 0) -> float:
        return float(self._close[self._idx(offset)])

    def volume(self, offset: int = 0) -> float:
        return float(self._volume[self._idx(offset)])

    def closes(self) -> np.ndarray:
        """All visible closes, oldest-first, length <= lookback. Read-only."""
        arr = self._close.copy()
        arr.setflags(write=False)
        return arr

    def highs(self) -> np.ndarray:
        """All visible highs, oldest-first, length <= lookback. Read-only."""
        arr = self._high.copy()
        arr.setflags(write=False)
        return arr

    def lows(self) -> np.ndarray:
        """All visible lows, oldest-first, length <= lookback. Read-only."""
        arr = self._low.copy()
        arr.setflags(write=False)
        return arr


FeatureFn = Callable[[BarWindow], float]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    lookback: int
    fn: FeatureFn
    description: str = ""


class FeatureRegistry:
    """Process-local registry of look-ahead-safe features."""

    def __init__(self) -> None:
        self._features: dict[str, FeatureSpec] = {}

    def register(
        self,
        name: str,
        *,
        lookback: int,
        fn: FeatureFn,
        description: str = "",
    ) -> FeatureSpec:
        if not name or not isinstance(name, str):
            raise FeatureRegistrationError(f"feature name must be non-empty; got {name!r}")
        if not isinstance(lookback, int) or isinstance(lookback, bool):
            raise FeatureRegistrationError(
                f"lookback for {name!r} must be an int; got {type(lookback).__name__}"
            )
        if lookback <= 0:
            # Zero is refused as well as negative. A zero-lookback feature
            # sees an empty window, so whatever it returns cannot have come
            # from the bars - it is a constant or a closure over something
            # outside the window, and both are leaks waiting to happen.
            raise FeatureRegistrationError(
                f"lookback for {name!r} must be >= 1; got {lookback}"
            )
        if name in self._features:
            raise FeatureRegistrationError(f"feature {name!r} already registered")
        spec = FeatureSpec(name=name, lookback=lookback, fn=fn, description=description)
        self._features[name] = spec
        return spec

    def get(self, name: str) -> FeatureSpec:
        if name not in self._features:
            raise FeatureRegistrationError(
                f"feature {name!r} is not registered; strategies may only "
                f"reference registered features. Known: {sorted(self._features)}"
            )
        return self._features[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._features))

    def resolve_features(self, names: tuple[str, ...]) -> tuple[FeatureSpec, ...]:
        """Reject a strategy referencing any unregistered feature."""
        return tuple(self.get(n) for n in names)

    def window_at(self, bars: BarData, name: str, t: int) -> BarWindow:
        """Build a look-ahead-safe window for feature ``name`` at bar ``t``.

        The window slices each OHLCV array to ``[max(0, t-lookback+1) : t+1]``
        so it PHYSICALLY cannot contain any bar after t.
        """
        spec = self.get(name)
        n = bars.n_bars
        if t < 0 or t >= n:
            raise DataError(f"t={t} out of range [0,{n})")
        start = max(0, t - spec.lookback + 1)
        stop = t + 1  # exclusive; last element is bar t
        return BarWindow(
            _open=np.asarray(bars.open[start:stop], dtype=np.float64),
            _high=np.asarray(bars.high[start:stop], dtype=np.float64),
            _low=np.asarray(bars.low[start:stop], dtype=np.float64),
            _close=np.asarray(bars.close[start:stop], dtype=np.float64),
            _volume=np.asarray(bars.tick_volume[start:stop], dtype=np.float64),
            t=t,
            lookback=spec.lookback,
        )

    def compute(self, bars: BarData, name: str, t: int) -> float:
        spec = self.get(name)
        return float(spec.fn(self.window_at(bars, name, t)))


# A module-level default registry with a few reference features. Strategies
# and DNA validation resolve against this unless a custom registry is passed.
DEFAULT_REGISTRY = FeatureRegistry()


def _sma(window: BarWindow) -> float:
    closes = window.closes()
    return float(np.mean(closes)) if closes.size else float("nan")


def _last_close(window: BarWindow) -> float:
    return window.close(0)


def _true_range(window: BarWindow) -> float:
    high = window.high(0)
    low = window.low(0)
    prev_close = window.close(-1) if window.lookback >= 2 else window.close(0)
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


DEFAULT_REGISTRY.register(
    "sma_20", lookback=20, fn=_sma, description="20-bar simple moving average of close"
)
DEFAULT_REGISTRY.register(
    "last_close", lookback=1, fn=_last_close, description="close at bar t"
)
DEFAULT_REGISTRY.register(
    "true_range", lookback=2, fn=_true_range, description="true range at bar t"
)


def _rsi_14(window: BarWindow) -> float:
    """RSI(14) at bar t, Wilder smoothing, NaN during 14-bar warmup."""
    closes = window.closes()
    full = rsi_series(closes, period=DEFAULT_RSI_PERIOD)
    val = float(full[-1])
    if not np.isfinite(val):
        return float("nan")
    return val


DEFAULT_REGISTRY.register(
    "rsi_14", lookback=DEFAULT_RSI_PERIOD + 1, fn=_rsi_14,
    description="14-period Wilder RSI, NaN during 14-bar warmup",
)


def _stoch_k(window: BarWindow) -> float:
    closes = window.closes()
    highs = window.highs()
    lows = window.lows()
    if closes.size == 0:
        return float("nan")
    val = float(stoch_k_series(
        highs, lows, closes,
        period=DEFAULT_K_PERIOD, smoothing=DEFAULT_SMOOTHING,
    )[-1])
    return val if np.isfinite(val) else float("nan")


def _stoch_d(window: BarWindow) -> float:
    closes = window.closes()
    highs = window.highs()
    lows = window.lows()
    if closes.size == 0:
        return float("nan")
    _, slow_d = stoch_d_series(
        highs, lows, closes,
        period=DEFAULT_K_PERIOD, d_period=DEFAULT_D_PERIOD,
        smoothing=DEFAULT_SMOOTHING,
    )
    val = float(slow_d[-1])
    return val if np.isfinite(val) else float("nan")


def _macd_line(window: BarWindow) -> float:
    closes = window.closes()
    if closes.size == 0:
        return float("nan")
    val = float(macd_line_series(
        closes, fast_period=DEFAULT_FAST_PERIOD, slow_period=DEFAULT_SLOW_PERIOD,
    )[-1])
    return val if np.isfinite(val) else float("nan")


def _macd_signal(window: BarWindow) -> float:
    closes = window.closes()
    if closes.size == 0:
        return float("nan")
    _, sig = macd_signal_series(
        closes, fast_period=DEFAULT_FAST_PERIOD, slow_period=DEFAULT_SLOW_PERIOD,
        signal_period=DEFAULT_SIGNAL_PERIOD,
    )
    val = float(sig[-1])
    return val if np.isfinite(val) else float("nan")


def _donchian_upper(window: BarWindow) -> float:
    highs = window.highs()
    if highs.size == 0:
        return float("nan")
    return float(np.max(highs))


def _donchian_lower(window: BarWindow) -> float:
    lows = window.lows()
    if lows.size == 0:
        return float("nan")
    return float(np.min(lows))


DEFAULT_K_PERIOD = 14
DEFAULT_D_PERIOD = 3
DEFAULT_SMOOTHING = 3
DEFAULT_FAST_PERIOD = 12
DEFAULT_SLOW_PERIOD = 26
DEFAULT_SIGNAL_PERIOD = 9
DEFAULT_DONCHIAN_PERIOD = 20

DEFAULT_REGISTRY.register(
    "stoch_k", lookback=DEFAULT_K_PERIOD + DEFAULT_SMOOTHING, fn=_stoch_k,
    description="Smoothed stochastic %K (14/3), NaN during warmup",
)
DEFAULT_REGISTRY.register(
    "stoch_d", lookback=DEFAULT_K_PERIOD + DEFAULT_SMOOTHING + DEFAULT_D_PERIOD, fn=_stoch_d,
    description="Stochastic %D (SMA of %K), NaN during warmup",
)
DEFAULT_REGISTRY.register(
    "macd_line", lookback=DEFAULT_SLOW_PERIOD + DEFAULT_SIGNAL_PERIOD, fn=_macd_line,
    description="MACD line (EMA 12 - EMA 26), NaN during warmup",
)
DEFAULT_REGISTRY.register(
    "macd_signal", lookback=DEFAULT_SLOW_PERIOD + DEFAULT_SIGNAL_PERIOD * 2, fn=_macd_signal,
    description="MACD signal line (EMA of MACD line), NaN during warmup",
)
DEFAULT_REGISTRY.register(
    "donchian_upper", lookback=DEFAULT_DONCHIAN_PERIOD, fn=_donchian_upper,
    description="Donchian upper channel (n-bar high)",
)
DEFAULT_REGISTRY.register(
    "donchian_lower", lookback=DEFAULT_DONCHIAN_PERIOD, fn=_donchian_lower,
    description="Donchian lower channel (n-bar low)",
)


# ---------------------------------------------------------------------------
# Gene family -> the registered features that family needs
# ---------------------------------------------------------------------------
# A DNA names strategy FAMILIES ("RSI_REVERSION", "ADX", "DAILY_TREND", ...),
# not features. This maps each family to the registry entries it requires, so a
# DNA can be rejected at CONSTRUCTION when it names a family whose features
# are not registered.
#
# "RSI_REVERSION" is backed by rsi_14 (W0.3, renamed here in B2).
# "STOCH_CROSS" is backed by stoch_k + stoch_d (W1).
# "MACD_CROSS" is backed by macd_line + macd_signal (W1).
# "DONCHIAN_BREAKOUT" is backed by donchian_upper + donchian_lower (W1).
# The MA_CROSS family is not a DNA signal family in this table — it is the kernel
# gate strategy, expressible directly without a feature lookup, so it needs
# no entry here. unimplemented_families() reports any remaining gap so
# implementing a family forces its features to be registered.
FAMILY_FEATURES: dict[str, tuple[str, ...]] = {
    "RSI_REVERSION": ("rsi_14",),
    "STOCH_CROSS": ("stoch_k", "stoch_d"),
    "MACD_CROSS": ("macd_line", "macd_signal"),
    "DONCHIAN_BREAKOUT": ("donchian_upper", "donchian_lower"),
}


def unimplemented_families(families: Iterable[str]) -> tuple[str, ...]:
    """Families named by a DNA that have no feature backing at all."""
    return tuple(
        f for f in dict.fromkeys(families) if f and f not in FAMILY_FEATURES
    )


def require_registered_features(
    families: Iterable[str],
    *,
    registry: FeatureRegistry | None = None,
    mapping: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Resolve every feature the named families require, or raise.

    Raises ``FeatureRegistrationError`` naming the first unregistered
    feature. Called from ``DNA.__post_init__`` so the rejection happens when
    the genome is built, not later during validation when a bad genome has
    already been scored, stored and reported.
    """
    reg = DEFAULT_REGISTRY if registry is None else registry
    table = FAMILY_FEATURES if mapping is None else mapping
    resolved: list[str] = []
    for family in families:
        for feature in table.get(family, ()):
            reg.get(feature)  # raises if unregistered
            resolved.append(feature)
    return tuple(resolved)


__all__ = [
    "FAMILY_FEATURES",
    "BarWindow",
    "LookAheadError",
    "DEFAULT_REGISTRY",
    "FeatureFn",
    "FeatureRegistrationError",
    "FeatureRegistry",
    "FeatureSpec",
    "LookaheadError",
    "require_registered_features",
    "unimplemented_families",
]
