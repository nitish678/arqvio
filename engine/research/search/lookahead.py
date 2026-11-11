"""Causal sentinel: does a strategy's decision depend on a future bar?

Motivation. A deliberately planted look-ahead oracle posting Sharpe 35
survived BOTH Deflated Sharpe and PBO. Statistical correction does not
catch leakage, because a leaking strategy is not overfit -- it is
right, for a reason it will not have live. The only defence is
structural, and the only proof that the structure holds is a test that
would catch it breaking.

``lookahead_guard`` already re-exports a DNA FLAG check
(``uses_future_bar``). That catches a genome that declares itself. It
cannot catch an implementation that reads ``close[t+1]`` while the flag
says False, which is the case that matters.

METHOD. Run the same strategy over the same bars two ways:

    vectorized  the strategy sees the WHOLE series on every call
    stepwise    the strategy sees only bars 0..t on the call for bar t

A causal strategy cannot tell the difference: everything it is allowed
to read is present in both. One that reaches forward gets a different
answer in the two runs, because in the stepwise run the future is
physically absent.

The comparison is over DECISIONS, not P&L. Two runs can produce the
same equity from different trades, and a leak that changes one decision
in ten thousand is still a leak.

A NEGATIVE CONTROL IS MANDATORY. A detector that never fires is
indistinguishable from a clean codebase, and far more dangerous,
because it reads as proof. ``LeakyStrategy`` here exists to be caught;
if it is not, the sentinel is broken and the positive result means
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from engine.research.data.mt5_source import BarData
from engine.research.features.forward_block import bars_up_to


class _Strategy(Protocol):
    def on_bar(
        self, bar_index: int, bars: BarData, state: dict[str, object],
    ) -> Any: ...


@dataclass(frozen=True)
class Divergence:
    """One bar where the two runs disagreed."""

    bar_index: int
    vectorized: str
    stepwise: str

    def __str__(self) -> str:
        return (
            f"bar {self.bar_index}: vectorized={self.vectorized} "
            f"stepwise={self.stepwise}"
        )


@dataclass(frozen=True)
class SentinelResult:
    """Whether a strategy's decisions depend on bars it may not see."""

    causal: bool
    n_bars_checked: int
    n_decisions: int
    divergences: tuple[Divergence, ...] = field(default=())

    @property
    def leaked(self) -> bool:
        return not self.causal

    def summary(self) -> str:
        if self.causal:
            return (
                f"CAUSAL: {self.n_decisions} decisions over "
                f"{self.n_bars_checked} bars identical under both views"
            )
        first = self.divergences[0] if self.divergences else None
        return (
            f"LEAK: {len(self.divergences)} divergence(s) over "
            f"{self.n_bars_checked} bars; first at {first}"
        )


def _decision_key(intent: Any) -> str:
    """A stable description of a decision, ignoring generated ids.

    intent_id is a fresh uuid per call and would make every bar look
    like a divergence.
    """
    if intent is None:
        return "-"
    parts = [
        str(getattr(intent, "direction", "?")),
        f"{float(getattr(intent, 'volume', 0.0)):.10g}",
        f"{int(getattr(intent, 'bar_index', -1))}",
    ]
    for name in ("sl_dist", "tp_dist", "sl_price", "tp_price"):
        value = getattr(intent, name, None)
        parts.append("-" if value is None else f"{float(value):.10g}")
    return "|".join(parts)


def check_causality(
    strategy_factory,
    bars: BarData,
    *,
    start_bar: int = 0,
    max_divergences: int = 10,
) -> SentinelResult:
    """Compare a strategy's decisions under the full series and a causal slice.

    ``strategy_factory`` is called twice with no arguments, so each run
    gets a fresh strategy and a fresh state dict. Sharing either would
    let the first run's state leak into the second and mask a real
    divergence.
    """
    n = int(bars.n_bars)
    vec_strategy = strategy_factory()
    step_strategy = strategy_factory()
    vec_state: dict[str, object] = {}
    step_state: dict[str, object] = {}

    divergences: list[Divergence] = []
    n_decisions = 0
    for i in range(start_bar, n):
        # The whole series, including bars after i.
        vec = vec_strategy.on_bar(i, bars, vec_state)
        # Only bars 0..i. bars_up_to slices, so i+1 is not in the array.
        step = step_strategy.on_bar(i, bars_up_to(bars, i), step_state)

        vk, sk = _decision_key(vec), _decision_key(step)
        if vk != "-" or sk != "-":
            n_decisions += 1
        if vk != sk:
            divergences.append(Divergence(i, vk, sk))
            if len(divergences) >= max_divergences:
                break

    return SentinelResult(
        causal=not divergences,
        n_bars_checked=n - start_bar,
        n_decisions=n_decisions,
        divergences=tuple(divergences),
    )


class LeakyStrategy:
    """NEGATIVE CONTROL. Reads the NEXT bar. Never use outside tests.

    This exists so the sentinel can be shown to fire. A detector that
    has never caught anything is not evidence of a clean codebase.
    """

    strategy_id = "leaky-oracle"

    def __init__(self, volume: float = 0.1) -> None:
        self.volume = volume

    def on_bar(self, bar_index, bars, state):
        from engine.research.backtest.order import OrderIntent

        close = np.asarray(bars.close)
        nxt = bar_index + 1
        if nxt >= close.size:
            # In the causal view the future is simply absent, which is
            # precisely what makes the two runs disagree.
            return None
        direction = "long" if close[nxt] > close[bar_index] else "short"
        return OrderIntent(
            direction=direction, volume=self.volume, bar_index=bar_index
        )


__all__ = [
    "Divergence",
    "LeakyStrategy",
    "SentinelResult",
    "check_causality",
]
