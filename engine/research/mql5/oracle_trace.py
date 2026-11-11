"""The Python side of the parity harness: the ORACLE run, its intents, and the slippage stream.

``run_oracle(dna, cell)`` runs the unmodified slow engine (``BacktestEngine``)
with the cell's exact config. Two read-only taps, neither of which changes the
run:

* a strategy wrapper records every intent the strategy RETURNS, with its reason
  inferred from the strategy's own state transition: ``PENDING_OPEN`` (a pended
  reversal opened), ``ENTRY``, ``REVERSAL_CLOSE`` (a close that pends the
  reverse) and ``SIGNAL_EXIT`` (a close that pends nothing, e.g. the RSI
  midline);
* a proxy over the engine's RNG records every slippage draw in order.

``slippage_stream(cell, n)`` is the exogenous draw stream the EA reads: the
k-th ``realised_slippage_points`` value of ``default_rng(0)`` -- the engine's
own RNG, which it uses once per fill in fill order. ``run_oracle`` checks that
the run's recorded draws are a prefix of that stream, so the input the EA reads
IS the oracle's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.research.backtest.engine import BacktestEngine, BacktestResult
from engine.research.costs.slippage import realised_slippage_points
from engine.research.mql5.cell_config import SLIPPAGE_MEAN, SLIPPAGE_STD, CellConfig
from engine.research.search.dna import DNA
from engine.research.search.real_backtest import dna_to_strategy


class OracleTraceError(RuntimeError):
    """The oracle run's taps disagree with the inputs the EA is given."""


@dataclass(frozen=True)
class TracedIntent:
    bar_index: int
    bar_time: int
    direction: int
    volume: float
    sl_dist: float | None
    tp_dist: float | None
    reason: str


@dataclass(frozen=True)
class OracleRun:
    dna: DNA
    cell: CellConfig
    strategy_id: str
    result: BacktestResult
    intents: tuple[TracedIntent, ...]
    draws: tuple[float, ...]


class _RecordingStrategy:
    def __init__(self, inner: Any, times: np.ndarray) -> None:
        self.inner = inner
        self.strategy_id = inner.strategy_id
        self.times = times
        self.intents: list[TracedIntent] = []

    def on_bar(self, bar_index: int, bars: Any, state: dict[str, object]) -> Any:
        pending_before = state.get("pending_direction")
        intent = self.inner.on_bar(bar_index, bars, state)
        if intent is None:
            return None
        if intent.volume > 0:
            reason = "PENDING_OPEN" if pending_before is not None and "pending_direction" not in state else "ENTRY"
        else:
            reason = "REVERSAL_CLOSE" if state.get("pending_direction") is not None else "SIGNAL_EXIT"
        self.intents.append(TracedIntent(
            bar_index=int(bar_index), bar_time=int(self.times[bar_index]),
            direction=1 if intent.direction == "long" else -1, volume=float(intent.volume),
            sl_dist=None if intent.sl_dist is None else float(intent.sl_dist),
            tp_dist=None if intent.tp_dist is None else float(intent.tp_dist), reason=reason))
        return intent


class _RecordingRng:
    """Forwards to the engine's Generator; records each normal() draw's realised value."""

    def __init__(self, inner: np.random.Generator) -> None:
        self.inner = inner
        self.normals: list[float] = []

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any = None) -> Any:
        v = self.inner.normal(loc=loc, scale=scale, size=size)
        self.normals.append(float(v))
        return v


def slippage_stream(cell: CellConfig, n: int) -> list[float]:
    rng = np.random.default_rng(0)
    return [realised_slippage_points(mean_points=SLIPPAGE_MEAN, std_points=SLIPPAGE_STD, rng=rng,
                                     cap_points=cell.slippage_cap_points) for _ in range(n)]


def run_oracle(dna: DNA, cell: CellConfig) -> OracleRun:
    cfg, strategy = dna_to_strategy(dna)
    engine = BacktestEngine(spec=cell.spec, config=cell.engine_config(cfg.strategy_id, risk_pct=dna.risk_pct,
                                                                         notes=f"dna_hash={dna.dna_hash}"), rng_seed=0)
    rec_rng = _RecordingRng(engine._rng)
    engine._rng = rec_rng
    rec = _RecordingStrategy(strategy, np.asarray(cell.bars.time, dtype=np.int64))
    result = engine.run(cell.bars, rec)
    draws = [min(abs(v), cell.slippage_cap_points) for v in rec_rng.normals]
    stream = slippage_stream(cell, len(draws))
    if draws != stream:
        raise OracleTraceError("the oracle's slippage draws are not the exogenous stream the EA reads")
    return OracleRun(dna, cell, cfg.strategy_id, result, tuple(rec.intents), tuple(draws))


__all__ = ["OracleRun", "OracleTraceError", "TracedIntent", "run_oracle", "slippage_stream"]
