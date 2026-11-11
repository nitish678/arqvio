"""The V1 vault gates, run once per Final War finalist after breeding (research APP-2 "statistical gates",
BP03 §3-4; the inspector's proof block: "beat random luck, survive thousands of reshuffled test runs and stay stable
when its settings were nudged").

Each finalist is re-run ONCE over the market's whole window (the 70 % it was bred on + the 30 % held back), with the
position carried across the split, and judged:

* held-out trades     >= 15 on the held-back bars (research: fewer = reject)
* random              its held-out Sharpe against 100 random-entry twins' held-out Sharpe (research: "ROBUST = beat
                      the crowd of random strategies on held-out data"); twin k draws with its own salt
* reshuffled runs     1000 random orderings of its bar returns: the drawdown of the unluckiest 1 in 20 under 40 %
                      (research: "BAD-LUCK DRAWDOWN is the dip in the unluckiest 1 in 20 of its reshuffled runs")
* nudged settings     every engine-reaching gene one step either way (vault/robustness.neighbours), each re-run on
                      the bred window: the share that still makes money
* floors              whole-window max drawdown, profit factor (closed trades), trade count

A finalist that passes every gate enters the vault with the research score; none of these is ARQVIO's validation
core, so the vault marks it UNCERTIFIED (CERTIFY = engine V2).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from engine.research.search.dna import DNA
from engine.research.v1.config import V1Config
from engine.research.v1.score import max_drawdown, metrics, score, sharpe

RANDOM_SALT = "arqvio-v1-random"
SHUFFLE_SALT = "arqvio-v1-shuffle"


def _seed(dna_hash: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{dna_hash}".encode()).hexdigest()[:16], 16)


def shuffled_dd_p95(returns: np.ndarray, *, n: int, seed: int) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 10:
        return float("nan")
    rng = np.random.default_rng(seed)
    return float(np.percentile([max_drawdown(rng.permutation(r)) for _ in range(n)], 95))


def _returns(equity: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2 or not np.all(eq[:-1] > 0):
        return np.zeros(0)
    return np.diff(eq) / eq[:-1]


def judge(dna: DNA, *, cfg: V1Config, timeframe: str, split: int,
          full_run: Callable[[DNA], Any],
          twin_returns: Callable[[DNA, int, str], np.ndarray | None],
          evaluate_bred: Callable[[DNA], Any],
          neighbours: Callable[[DNA], list[tuple[str, DNA]]]) -> dict[str, Any]:
    """Every gate for one finalist. full_run returns GateResult or a legacy (kernel rows, equity) tuple.
    twin_returns(dna, entries, salt) returns a random twin's whole-window per-bar returns."""
    from engine.research.backtest.kernel import TR_SIZE

    from engine.research.search.evaluation_view import GateResult, kernel_view

    result = full_run(dna)
    if not isinstance(result, GateResult):
        rows, equity = result  # legacy gate fixtures/callers
        result = kernel_view(np.asarray(rows, dtype=np.float64).reshape(-1, TR_SIZE), equity)
    r = _returns(result.equity)
    if r.size <= split + 1:
        return {"passed": False, "failed": ["NO_HELD_OUT"], "reason": "the whole-window run is shorter than the split"}
    fills = result.fills
    closed = result.exits >= 0
    pnl_all = result.net[closed]
    closed_fills = fills[closed]
    ho_mask = fills >= split
    is_m = metrics(r[:split], timeframe, int(np.sum(~ho_mask)),
                   pnl_all[closed_fills < split] if np.any(closed) else None)
    ho_m = metrics(r[split:], timeframe, int(np.sum(ho_mask)),
                   pnl_all[closed_fills >= split] if np.any(closed) else None)
    full_m = metrics(r, timeframe, len(fills), pnl_all)

    entries = len(fills)
    twins = []
    for k in range(cfg.random_n):
        tr = twin_returns(dna, entries, f"{RANDOM_SALT}-{k}")
        if tr is not None and len(tr) > split + 1:
            twins.append(sharpe(np.asarray(tr)[split:], timeframe))
    finite = [x for x in twins if math.isfinite(x)]
    beaten = (sum(ho_m.sharpe > x for x in finite) / len(finite)) if finite else None

    dd95 = shuffled_dd_p95(r, n=cfg.shuffle_n, seed=_seed(dna.dna_hash, SHUFFLE_SALT))

    nb = []
    for _, cand in neighbours(dna):
        e = evaluate_bred(cand)
        s = None if e.crashed or e.result.sharpe is None else float(e.result.sharpe)
        nb.append(s)
    nb_ok = [s for s in nb if s is not None and math.isfinite(s)]
    hold = (sum(s > 0 for s in nb_ok) / len(nb_ok)) if nb_ok else 0.0

    checks = {
        "held_out_trades": (ho_m.n_trades, ho_m.n_trades >= cfg.held_out_min_trades),
        "beats_random": (beaten, beaten is not None and beaten >= cfg.random_beat_min),
        "reshuffled_dd_p95": (dd95, math.isfinite(dd95) and dd95 < cfg.shuffle_dd_max),
        "nudged_settings_hold": (hold, hold >= cfg.nudge_hold_min),
        "max_drawdown": (full_m.max_dd, full_m.max_dd <= cfg.floor_max_dd),
        "profit_factor": (full_m.profit_factor, full_m.profit_factor >= cfg.floor_pf),
        "trades": (full_m.n_trades, full_m.n_trades >= cfg.floor_trades),
    }
    failed = [k for k, (_, ok) in checks.items() if not ok]
    return {
        "passed": not failed, "failed": failed,
        "execution_path": result.execution_path, "fallback_reason": result.fallback_reason,
        "checks": {k: {"value": v, "passed": ok} for k, (v, ok) in checks.items()},
        "in_sample": is_m.to_dict(), "held_out": ho_m.to_dict(), "full": full_m.to_dict(),
        "random": {"n": len(finite), "beaten_share": beaten,
                   "p95": float(np.percentile(finite, 95)) if finite else None},
        "reshuffled": {"n": cfg.shuffle_n, "dd_p95": dd95},
        "nudged": {"n": len(nb), "positive_share": hold},
        **score(is_m, ho_m, full_m),
    }


__all__ = ["RANDOM_SALT", "SHUFFLE_SALT", "judge", "shuffled_dd_p95"]
