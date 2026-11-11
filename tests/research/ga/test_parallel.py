"""Power modes: a campaign on several worker processes is byte-identical to the same campaign on one."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine.research.ga import runner as ga
from engine.research.ga.config import CAMPAIGN_GATES, Selection, ga_config
from engine.research.ga.determinism import individuals_table, run_hash
from engine.research.ga.parallel import POWER_MODES, PoolEvaluator, workers_for
from engine.research.search import gene_pool as gp
from engine.research.search.dna import DNA
from engine.research.search.runner import TrialResult
from engine.research.search.sham_control import attempted_entries


class PureCell:
    """A deterministic, picklable stand-in for a pinned cell: every result is a pure function of the DNA.

    Some DNAs fall back to the slow path (recorded in fallback_state like the real fast factory), and one
    signal's sham raises, so both the fallback merge and the crash path are exercised."""

    name, symbol, timeframe, engine_mode_actual = "usdcadm_h1", "USDCADm", "H1", "fast"

    def __init__(self, sham_raises_for: str | None = "MACD_CROSS") -> None:
        self.shams: dict = {}
        self.backtest_fn = SimpleNamespace(fallback_state={"fallbacks": 0, "reasons": []})
        self.sham_raises_for = sham_raises_for
        self.context = {"dataset_hash": "synthetic", "cost_vector_hash": "synthetic"}

    def evaluate(self, dna: DNA) -> ga.Evaluation:
        state = self.backtest_fn.fallback_state
        fell_back = int(dna.dna_hash[-4:], 16) % 9 == 0
        if fell_back:
            # like the real fast factory: every fallback counted, each reason kept once, first seen first
            state["fallbacks"] += 1
            reason = f"synthetic fallback class {int(dna.dna_hash[:2], 16) % 3}"
            if reason not in state["reasons"]:
                state["reasons"].append(reason)
        rng = np.random.default_rng(int(dna.dna_hash, 16) % 2**32)
        edge = 0.0004 * (dna.sl_atr - 2.0) + 0.0002 * (dna.tp_atr - 3.0)
        returns = rng.normal(edge, 0.01, size=600)
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(6048))
        n_trades = 10 + int(dna.max_feature_lookback) % 80
        result = TrialResult(dna=dna, sharpe=sharpe, n_trades=n_trades, total_return=0.0, max_dd=0.1,
                             per_period_returns=returns)
        return ga.Evaluation(result, False, ga.ENGINE_SLOW_FALLBACK if fell_back else ga.ENGINE_KERNEL)

    def sham_hit(self, dna: DNA, entries: int) -> dict:
        if dna.signal == self.sham_raises_for:
            raise RuntimeError("synthetic sham edge case")
        sharpe = float(np.random.default_rng(int(dna.dna_hash[:8], 16)).normal(0.0, 0.5))
        return {"sharpe": sharpe, "requested": entries, "drawn": entries, "filled": entries, "truncated": False}

    def sham_sharpe(self, result: TrialResult) -> float:
        key = (result.dna.dna_hash, attempted_entries(result))
        if key not in self.shams:
            self.shams[key] = self.sham_hit(result.dna, key[1])
        return float(self.shams[key]["sharpe"])


def _cfg(**kw):
    base = dict(seed=11, gates=CAMPAIGN_GATES, gene_pool=gp.default_gene_pool(), population=40, generations=4,
                trials_budget=500, final_n=6, selection=Selection(elitism=4), kernel_share_min=0.5)
    return ga_config(**(base | kw))


def _comparable(generations: list[dict]) -> list[dict]:
    return [{k: v for k, v in g.items() if k != "wall_seconds"} for g in generations]


def test_power_modes_map_to_worker_counts() -> None:
    assert POWER_MODES == ("eco", "medium", "high", "extreme")
    assert [workers_for(p, cores=8) for p in POWER_MODES] == [1, 4, 7, 8]
    assert [workers_for(p, cores=1) for p in POWER_MODES] == [1, 1, 1, 1]
    with pytest.raises(ValueError):
        workers_for("turbo")


def test_a_pool_needs_two_workers() -> None:
    with pytest.raises(ValueError):
        PoolEvaluator(PureCell(), PureCell, 1)


def _both(cfg, **cell_kw):
    one = PureCell(**cell_kw)
    a = ga.search(cfg, one)
    pool = PoolEvaluator(PureCell(**cell_kw), partial(PureCell, **cell_kw), 2)
    try:
        b = ga.search(cfg, pool)
    finally:
        pool.close()
    return one, a, pool, b


def test_a_sham_that_raises_in_a_worker_is_the_same_recorded_crash(isolated_arqvio_home: Path) -> None:
    one, a, pool, b = _both(_cfg(), sham_raises_for="MACD_CROSS")
    errors_a = {h: e for g in a.generations for h, e in g["sham_errors"].items()}
    errors_b = {h: e for g in b.generations for h, e in g["sham_errors"].items()}
    assert errors_a and errors_a == errors_b
    assert (a.stop_reason, a.aborted) == (b.stop_reason, b.aborted)
    assert one.shams == pool.local.shams


def test_two_workers_reproduce_one_worker_byte_for_byte(isolated_arqvio_home: Path) -> None:
    cfg = _cfg()
    one = PureCell(sham_raises_for=None)
    a = ga.search(cfg, one)
    pool = PoolEvaluator(PureCell(sham_raises_for=None), partial(PureCell, sham_raises_for=None), 3,
                         affinity=lambda d: d.signal)  # coarse affinity: whole families share a worker
    try:
        b = ga.search(cfg, pool)
    finally:
        pool.close()
    assert run_hash(individuals_table(a.archive)) == run_hash(individuals_table(b.archive))
    assert _comparable(a.generations) == _comparable(b.generations)
    assert [i.effective_hash for i in a.finalists] == [i.effective_hash for i in b.finalists]
    assert (a.stop_reason, a.trials_used, a.collisions) == (b.stop_reason, b.trials_used, b.collisions)
    # exactly the shams the sequential loop asked for, and the same fallback record
    assert one.shams == pool.local.shams
    assert one.backtest_fn.fallback_state == pool.local.backtest_fn.fallback_state
    assert one.backtest_fn.fallback_state["fallbacks"] > 0
    assert len(a.generations) == cfg.generations and a.aborted is None  # the full loop, every generation bred


def test_with_power_is_the_evaluator_itself_on_one_worker() -> None:
    cell = PureCell()
    assert ga.with_power(cell, 1) is cell
