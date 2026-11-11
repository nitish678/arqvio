"""Built-in strategies (S3: MA crossover for the gate; S6+ adds more)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from engine.research.backtest.order import OrderIntent
from engine.research.data.mt5_source import BarData

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Family", "Strategy", "StrategyFamily"]


class Strategy(Protocol):
    """A strategy is callable: on_bar(bar_index, bars, state) -> OrderIntent | None.

    Shared by the slow-engine family (RSI_REVERSION, STOCH_CROSS,
    MACD_CROSS, DONCHIAN_BREAKOUT) and the gate (MA_CROSS). Each family
    ships its own concrete class; this Protocol is the contract they all
    satisfy, so the dispatcher need not know which one it holds.
    """

    strategy_id: str

    def on_bar(
        self,
        bar_index: int,
        bars: BarData,
        state: dict[str, object],
    ) -> OrderIntent | None: ...


class StrategyFamily(Protocol):
    """Universal per-family contract for F1-F4 slow-engine families.

    Every family provides:
      * ``Config`` -- a dataclass with at least ``volume``, ``strategy_id``,
        ``sl_atr``, ``tp_atr``, ``atr_period``.
      * ``StrategyCls`` -- the strategy class whose ``__init__`` accepts a
        single ``config`` keyword, exposing ``on_bar`` and ``strategy_id``.
      * ``dna_to_config`` -- a function (DNA) -> Config that maps a gene
        bundle to the family-specific config.

    The dispatcher (dna_to_strategy) iterates families by this Protocol
    rather than by hand-listed lambdas, so adding a family is one
    registration instead of three wiring points.
    """

    Config: type
    StrategyCls: type

    def dna_to_config(self, dna: object) -> object: ...


@dataclass(frozen=True)
class Family:
    """Concrete family descriptor satisfying :class:`StrategyFamily`.

    Bundles the three things every family exposes so the dispatcher can
    iterate them uniformly without per-family lambdas.
    """

    name: str
    config_cls: type
    strategy_cls: type
    dna_to_config: object  # Callable[[DNA], Any] -- kept minimal to avoid circular import
