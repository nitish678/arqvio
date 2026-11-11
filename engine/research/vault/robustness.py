"""Robustness checks for one vault strategy, on the pinned cell it was bred on.

Three checks, each named for what it is -- all three are IN-SAMPLE (the GA saw the whole window), so none of
them is an out-of-sample result; the vault's out-of-sample evidence is the cross cell.

* reproduction    the strategy is re-run on its cell; its Sharpe must equal the campaign's record exactly
* neighbourhood   every engine-reaching gene nudged one step either way (an integer +-1, a threshold +-2, an
                  ATR multiple x0.9 / x1.1, a flag flipped), each neighbour re-run: the share that still makes
                  money, the plateau share (neighbours keeping at least half the strategy's Sharpe), and their
                  median Sharpe against the strategy's own. A real edge sits on a plateau, not a spike.
* random          N_RANDOM random-entry twins (the GA's sham: same attempted entries, sides by coin, same risk
                  and exits, slow engine), each with its own draw: the share whose Sharpe the strategy beats
* segments        the bred window cut into 3 equal time segments: the Sharpe of each (is the edge everywhere,
                  or in one lucky stretch?)
* bootstrap       2,000 circular block-bootstrap resamples of the bar returns (block = one day of bars): the
                  share whose Sharpe is <= 0, and the 95th percentile of the resampled max drawdown
* concentration   the profit of each calendar month (UTC) from the re-run's equity: the best month's share of
                  the whole profit, and the share of months with a position at all (v4, ARQVIO_V4.md 10.2 -- the
                  XAGUSD M5 strategy made 89% of its profit in January 2026 and sat out eight months)

Every random draw is seeded by the strategy's dna_hash, so a re-run gives the same numbers.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from engine.research.registry.annualisation import TF_MINUTES, bars_per_year
from engine.research.search.dna import DNA, validate_gene_space
from engine.research.search.gene_pool import FAMILY_GENES
from engine.research.search.sampler import SL_ATR_RANGE, TP_ATR_RANGE

N_SEGMENTS = 3
N_BOOTSTRAP = 2000
N_RANDOM = 100
RANDOM_SALT = "arqvio-v4-beat-random"  # twin k draws with f"{RANDOM_SALT}-{k}"
PLATEAU_FRACTION = 0.5
THRESHOLD_STEP = 2.0
ATR_FACTORS = (0.9, 1.1)


def annual_sharpe(returns: np.ndarray, timeframe: str) -> float:
    """real_backtest's Sharpe: mean / std(ddof=1) of the bar returns, annualised by the law; 0 when undefined."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2 or not np.std(r) > 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(bars_per_year(timeframe)))


def neighbours(dna: DNA) -> list[tuple[str, DNA]]:
    """(label, DNA) for every valid one-step nudge of the strategy's engine-reaching genes."""
    out: list[tuple[str, DNA]] = []

    def add(label: str, **change: Any) -> None:
        cand = dataclasses.replace(dna, **change)
        try:
            validate_gene_space(cand.to_dict())
        except ValueError:
            return
        if cand.dna_hash != dna.dna_hash:
            out.append((label, cand))

    for spec in FAMILY_GENES.get(dna.signal, ()):
        v = getattr(dna, spec.name)
        if spec.kind == "bool":
            add(f"{spec.name}={not v}", **{spec.name: not v})
        elif spec.kind == "int":
            step = max(1, round(abs(v) * 0.1)) if spec.name == "max_feature_lookback" else 1
            for s in (-step, step):
                if spec.lo <= v + s <= spec.hi:
                    add(f"{spec.name}{s:+d}", **{spec.name: int(v + s)})
        else:
            for s in (-THRESHOLD_STEP, THRESHOLD_STEP):
                if spec.lo <= v + s <= spec.hi:
                    add(f"{spec.name}{s:+.0f}", **{spec.name: float(v + s)})
    for gene, (lo, hi) in (("sl_atr", SL_ATR_RANGE), ("tp_atr", TP_ATR_RANGE)):
        v = getattr(dna, gene)
        for f in ATR_FACTORS:
            if lo <= v * f <= hi:
                add(f"{gene}x{f}", **{gene: float(v * f)})
    return out


def segments(returns: np.ndarray, timeframe: str, n: int = N_SEGMENTS) -> list[float]:
    r = np.asarray(returns, dtype=np.float64)
    return [annual_sharpe(part, timeframe) for part in np.array_split(r, n)]


def max_drawdown(returns: np.ndarray) -> float:
    eq = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate(([1.0], eq)))[1:]
    return float(-np.min((eq - peak) / peak))


def bootstrap(returns: np.ndarray, timeframe: str, *, seed: int, n: int = N_BOOTSTRAP) -> dict[str, float]:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 10:
        return {"p_sharpe_le_0": 1.0, "max_dd_p95": float("nan"), "block_bars": 0, "resamples": 0}
    block = max(1, 1440 // TF_MINUTES[timeframe])  # one day of bars
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(r.size / block)
    sharpes = np.empty(n)
    dds = np.empty(n)
    for k in range(n):
        starts = rng.integers(0, r.size, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[: r.size] % r.size
        sample = r[idx]
        sharpes[k] = annual_sharpe(sample, timeframe)
        dds[k] = max_drawdown(sample)
    return {"p_sharpe_le_0": float(np.mean(sharpes <= 0.0)), "max_dd_p95": float(np.percentile(dds, 95)),
            "block_bars": block, "resamples": n}


def concentration(returns: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    """Monthly profit (fractions of the starting equity) and how much of the profit one month carries."""
    r = np.asarray(returns, dtype=np.float64)
    t = np.asarray(times, dtype=np.int64)[-r.size:] if r.size else np.zeros(0, dtype=np.int64)
    if r.size == 0 or t.size != r.size:
        return {"months": 0, "largest_month_share": None, "largest_month": None, "months_in_market_share": None,
                "monthly": {}}
    eq = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    months = t.astype("datetime64[s]").astype("datetime64[M]").astype(str)
    monthly: dict[str, float] = {}
    active: dict[str, bool] = {}
    for k, m in enumerate(months):
        monthly[m] = monthly.get(m, 0.0) + float(eq[k + 1] - eq[k])
        active[m] = active.get(m, False) or bool(r[k] != 0.0)
    total = float(eq[-1] - eq[0])
    best = str(max(monthly, key=lambda m: monthly[m]))
    return {"months": len(monthly), "largest_month": best,  # plain Python values: the gates compare them
            "largest_month_share": (monthly[best] / total) if total > 0 else None,
            "months_in_market_share": sum(active.values()) / len(active),
            "monthly": {str(m): round(v, 6) for m, v in monthly.items()}}


def beat_random(sharpe: float, random_sharpes: list[float]) -> dict[str, Any]:
    """The share of random-entry twins whose Sharpe the strategy beats (strictly), and their p95."""
    finite = [x for x in random_sharpes if math.isfinite(x)]
    if not finite:
        return {"n": 0, "beaten_share": None, "random_p95": None}
    return {"n": len(finite), "beaten_share": sum(sharpe > x for x in finite) / len(finite),
            "random_p95": float(np.percentile(finite, 95))}


def run(dna: DNA, *, timeframe: str, recorded_sharpe: float | None,
        evaluate: Callable[[DNA], Any], bar_times: np.ndarray | None = None,
        random_sharpe: Callable[[DNA, int, int], float] | None = None) -> dict[str, Any]:
    """Every check for ``dna``; ``evaluate`` is the cell evaluator's (returns an Evaluation);
    ``random_sharpe(dna, attempted_entries, k)`` is the Sharpe of random-entry twin k."""
    base = evaluate(dna)
    if base.crashed or base.result.per_period_returns is None:
        return {"state": "FAILED", "reason": "the strategy did not re-run on its cell", "engine": base.engine}
    sharpe = float(base.result.sharpe or 0.0)
    returns = np.asarray(base.result.per_period_returns, dtype=np.float64)
    rows = []
    for label, cand in neighbours(dna):
        e = evaluate(cand)
        s = None if e.crashed or e.result.sharpe is None else float(e.result.sharpe)
        rows.append({"change": label, "dna_hash": cand.dna_hash, "sharpe": s,
                     "n_trades": int(e.result.n_trades or 0), "engine": e.engine})
    finite = [r["sharpe"] for r in rows if r["sharpe"] is not None and math.isfinite(r["sharpe"])]
    seed = int(dna.dna_hash, 16) % 2**32
    attempts = int(base.result.n_trades or 0) + int(getattr(base.result, "blocked_max_spread", 0) or 0)
    randoms = [] if random_sharpe is None else [random_sharpe(dna, attempts, k) for k in range(N_RANDOM)]
    return {
        "state": "DONE",
        "engine": base.engine,
        "reproduction": {"sharpe": sharpe, "recorded": recorded_sharpe,
                         "identical": recorded_sharpe is not None and sharpe == float(recorded_sharpe)},
        "neighbourhood": {"n": len(rows), "positive_share": (sum(s > 0 for s in finite) / len(finite)) if finite else 0.0,
                          "plateau_share": (sum(s >= PLATEAU_FRACTION * sharpe for s in finite) / len(finite))
                          if finite and sharpe > 0 else 0.0,
                          "median_sharpe": float(np.median(finite)) if finite else None,
                          "median_ratio": (float(np.median(finite)) / sharpe) if finite and sharpe > 0 else None,
                          "rows": rows},
        "segments": {"n": N_SEGMENTS, "sharpe": segments(returns, timeframe)},
        "bootstrap": bootstrap(returns, timeframe, seed=seed),
        "concentration": concentration(returns, bar_times) if bar_times is not None else None,
        "random": beat_random(sharpe, randoms) if random_sharpe is not None else None,
        "label": "in-sample checks on the bred window; the cross cell is the out-of-sample evidence",
    }


__all__ = ["annual_sharpe", "beat_random", "bootstrap", "concentration", "max_drawdown", "neighbours", "run", "segments"]
