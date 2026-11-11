"""Structural forward-access blocking for strategy bar access.

Motivation (arXiv 2608.27734, Aug 2026): a deliberately planted look-ahead
oracle posting Sharpe 35 survived BOTH Deflated Sharpe and PBO completely.
Statistical correction does not catch leakage. Purged CV and the
``uses_future_bar`` sentinel are necessary and not sufficient — the sentinel
only catches a DNA that *declares* it cheats, and purging only fixes the
split boundary. Neither stops a strategy from reading ``close[t + 1]``.

``FeatureRegistry`` already makes features structurally safe: a ``BarWindow``
is sliced to end at bar t, so the future is physically absent. But strategies
do not go through the registry — ``Strategy.on_bar(bar_index, bars, state)``
receives the whole ``BarData``, and ``bars.close`` is the complete array
including every future bar. ``close[-1]`` is the last bar of the series.

This module closes that hole at the call boundary rather than by rewriting
every strategy: the engine hands ``on_bar`` a view whose arrays refuse to
return anything after bar t.

What is refused, and why:

- ``arr[i]`` for ``i > t``: the future, directly.
- ``arr[-1]`` and any negative index: negative indices count back from the
  end of the FULL series, so ``close[-1]`` is the final bar of the backtest.
  This is the most common accidental leak and it looks completely innocent.
- ``arr[a:b]`` with ``b > t + 1`` or ``b is None``: ``close[:]`` and
  ``close[t:]`` both reach the end of the series.
- fancy indexing with an array of positions, if any position exceeds t.

What is allowed: everything at or before bar t, which is all a strategy is
entitled to see when deciding what to do at bar t.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.research.features.registry import LookAheadError, LookaheadError


class ForwardAccessBlocker:
    """A read-only ndarray proxy bounded at bar ``t``.

    Wraps the full array rather than slicing it so that an out-of-range read
    is reported as the look-ahead it is, instead of silently returning a
    shorter array or an IndexError that a strategy might swallow.
    """

    __slots__ = ("_arr", "_t", "_name")

    def __init__(self, arr: np.ndarray, t: int, name: str = "bars") -> None:
        self._arr = arr
        self._t = int(t)
        self._name = name

    # -- the guard ----------------------------------------------------------
    def _refuse(self, what: str) -> LookaheadError:
        return LookaheadError(
            f"{self._name}[{what}] reaches past bar t={self._t}. A strategy "
            f"deciding at bar {self._t} may not read bar {self._t + 1} or "
            "later. Negative indices count from the END of the full series "
            "and are refused for the same reason."
        )

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int | np.integer):
            i = int(key)
            if i < 0 or i > self._t:
                raise self._refuse(str(key))
            return self._arr[i]

        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is not None and int(start) < 0:
                raise self._refuse(f"{start}:{stop}")
            if stop is None or int(stop) < 0 or int(stop) > self._t + 1:
                raise self._refuse(f"{'' if start is None else start}:"
                                   f"{'' if stop is None else stop}")
            return self._arr[key]

        # Fancy indexing: every requested position must be visible.
        idx = np.asarray(key)
        if idx.dtype == bool:
            if idx.size > self._t + 1:
                raise self._refuse("<bool mask longer than the visible window>")
            return self._arr[: idx.size][idx]
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) > self._t):
            raise self._refuse("<index array reaching past t>")
        return self._arr[key]

    def __len__(self) -> int:
        return self._t + 1

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """np.mean(view) and friends see only the visible window."""
        out = self._arr[: self._t + 1]
        return out if dtype is None else out.astype(dtype)

    def __iter__(self):
        return iter(self._arr[: self._t + 1])

    @property
    def size(self) -> int:
        return self._t + 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._t + 1,) + self._arr.shape[1:]

    @property
    def dtype(self) -> Any:
        return self._arr.dtype

    def __repr__(self) -> str:
        return f"ForwardAccessBlocker({self._name}, visible=0..{self._t})"


class BarsView:
    """``BarData`` with every array bounded at bar ``t``.

    Attribute access is proxied, so a field added to BarData later is
    blocked automatically rather than silently becoming a leak.
    """

    __slots__ = ("_bars", "_t", "_cache")

    def __init__(self, bars: Any, t: int) -> None:
        self._bars = bars
        self._t = int(t)
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        value = getattr(self._bars, name)
        if isinstance(value, np.ndarray):
            value = ForwardAccessBlocker(value, self._t, name=f"bars.{name}")
        elif name == "n_bars":
            # The length of the full series is itself future information.
            value = self._t + 1
        self._cache[name] = value
        return value

    def __repr__(self) -> str:
        return f"BarsView(t={self._t})"


def bars_up_to(bars: Any, t: int) -> BarsView:
    """The view a strategy is entitled to when deciding at bar ``t``."""
    return BarsView(bars, t)


# The spec names the proxy _ForwardAccessBlocker; exported under both.
_ForwardAccessBlocker = ForwardAccessBlocker

__all__ = [
    "BarsView",
    "ForwardAccessBlocker",
    "LookAheadError",
    "_ForwardAccessBlocker",
    "bars_up_to",
]
