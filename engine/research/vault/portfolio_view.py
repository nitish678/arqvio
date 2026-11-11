"""A portfolio of vault strategies, judged together (DECK PORTFOLIOS; ARQVIO_V4.md A17, A29).

Members are vault entries (the deployed bots' strategies, or any chosen set). Each is run once by the campaign's fast
kernel on its own pinned data; its equity is cut to UTC days and the daily returns of all members are lined up on
the days they share. From that panel:

* return correlation: every pair's Pearson correlation over their shared days; a pair sharing fewer than
  MIN_SHARED_DAYS days is "too little shared history" (no number is made up for it);
* diversity by axis -- market, currency, type, timeframe, session window -- the share of the largest label;
* grades A-F with published bars (CORR_BARS, SHARE_BARS); the portfolio grade is its WORST axis, because one
  concentrated axis is enough to make many strategies one bet;
* the combined equity of an equal-weight mix over the shared days, and plain suggestions.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

MIN_SHARED_DAYS = 30
CORR_BARS = ((0.30, "A"), (0.50, "B"), (0.70, "C"), (0.85, "D"))     # worst pairwise |corr| below the bar
SHARE_BARS = ((0.40, "A"), (0.55, "B"), (0.70, "C"), (0.85, "D"))    # largest label's share at or below the bar
ORDER = "ABCDF"


def grade(value: float | None, bars: tuple[tuple[float, str], ...], *, strict: bool) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    for bar, g in bars:
        if (value < bar) if strict else (value <= bar):
            return g
    return "F"


def daily_returns(times: np.ndarray, equity: np.ndarray) -> dict[str, float]:
    """UTC day -> that day's return, from the equity at each day's last bar (equity[k + 1] closes bar k)."""
    t = np.asarray(times, dtype=np.int64)
    eq = np.asarray(equity, dtype=np.float64)[1: t.size + 1]
    days = (t // 86400).astype(np.int64)
    last = {}
    for d, v in zip(days.tolist(), eq.tolist(), strict=False):
        last[d] = v
    ks = sorted(last)
    out = {}
    for a, b in zip(ks, ks[1:], strict=False):
        if last[a] > 0:
            out[str(np.datetime64(b, "D"))] = last[b] / last[a] - 1.0
    return out


def axes(entry: dict[str, Any]) -> dict[str, str]:
    sym = str(entry["symbol"])
    base = sym[:6].upper()
    d = entry.get("dna") or {}
    return {"market": sym, "currency": base[:3] if len(base) >= 6 else base, "type": str(entry["family"]),
            "timeframe": str(entry["timeframe"]),
            "session": f"{int(d.get('window_start_hour_utc', 0)):02d}-{int(d.get('window_end_hour_utc', 24)):02d} UTC"}


def judge(members: list[dict[str, Any]], series: dict[str, dict[str, float]]) -> dict[str, Any]:
    ids = [m["id"] for m in members]
    n = len(ids)
    corr: list[list[float | None]] = [[None] * n for _ in range(n)]
    worst = None
    worst_pair = None
    for i in range(n):
        corr[i][i] = 1.0
        for j in range(i + 1, n):
            common = sorted(set(series[ids[i]]) & set(series[ids[j]]))
            if len(common) < MIN_SHARED_DAYS:
                continue
            a = np.array([series[ids[i]][d] for d in common])
            b = np.array([series[ids[j]][d] for d in common])
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            corr[i][j] = corr[j][i] = round(c, 4)
            if worst is None or abs(c) > abs(worst):
                worst, worst_pair = c, (ids[i], ids[j])
    labels = [axes(m) for m in members]
    axis_rows = {}
    for ax in ("market", "currency", "type", "timeframe", "session"):
        counts: dict[str, int] = {}
        for lb in labels:
            counts[lb[ax]] = counts.get(lb[ax], 0) + 1
        top = max(counts, key=lambda k: counts[k]) if counts else None
        share = counts[top] / n if top else None
        axis_rows[ax] = {"counts": counts, "largest": top, "share": share,
                         "grade": grade(share, SHARE_BARS, strict=False) if n > 1 else None}
    grades = {"correlation": grade(abs(worst), CORR_BARS, strict=True) if worst is not None else None,
              **{ax: r["grade"] for ax, r in axis_rows.items()}}
    judged = [g for g in grades.values() if g]
    overall = max(judged, key=ORDER.index) if judged else None
    days = sorted(set.intersection(*(set(series[i]) for i in ids))) if ids else []
    combined: list[list[float]] = []
    if len(days) >= MIN_SHARED_DAYS:
        eq = 10_000.0
        for d in days:
            eq *= 1.0 + float(np.mean([series[i][d] for i in ids]))
            combined.append([int(np.datetime64(d, "s").astype(np.int64)), round(eq, 2)])
    tips = []
    if n < 5:
        tips.append(f"grow to 5+ strategies (now {n})")
    for ax in ("market", "currency", "type"):
        r = axis_rows[ax]
        if n > 1 and r["share"] is not None and r["share"] > 0.55:
            tips.append(f"diversify {ax}: {r['largest']} is {r['share']:.0%} of the members")
    if worst is not None and abs(worst) >= 0.7:
        tips.append(f"{worst_pair[0].split('@')[0][:8]} and {worst_pair[1].split('@')[0][:8]} move together "
                    f"(corr {worst:+.2f}): keep one")
    return {"members": [{"id": m["id"], "symbol": m["symbol"], "timeframe": m["timeframe"], "family": m["family"],
                         "grade": m.get("grade"), "days": len(series[m["id"]])} for m in members],
            "corr": corr, "worst_pair": {"ids": worst_pair, "corr": worst} if worst_pair else None,
            "axes": axis_rows, "grades": grades, "overall": overall, "shared_days": len(days),
            "combined": combined, "tips": tips,
            "bars": {"correlation": CORR_BARS, "share": SHARE_BARS, "min_shared_days": MIN_SHARED_DAYS}}


def build(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Run each member once (kernel; slow engine when it falls back) and judge the set."""
    series = build_series(entries)
    out = judge(entries, series)
    out["key"] = hashlib.sha256(json.dumps(sorted(e["id"] for e in entries)).encode()).hexdigest()[:16]
    return out


def build_series(entries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """id -> daily returns of each member, each run once on its own pinned bars under its own cost model."""
    from engine.research.search.dna import DNA
    from engine.research.search.real_backtest import (
        dna_to_strategy,
        fast_backtest_factory,
        load_cell_bars_from_snapshot,
        run_slow_backtest,
        slow_cell_setup,
        spread_cost_mode,
    )
    from engine.research.vault.library import recorded_cost_model

    series: dict[str, dict[str, float]] = {}
    for e in entries:
        cell = load_cell_bars_from_snapshot(e["symbol"], e["timeframe"], e["dataset_id"])
        dna = DNA(**e["dna"])
        costs = recorded_cost_model(e)  # each member under the cost model it was bred with
        with spread_cost_mode(costs):
            bt = fast_backtest_factory(cell)
        bt.capture["on"] = True
        bt(dna)
        if "equity" in bt.capture:
            equity = np.asarray(bt.capture["equity"])
        else:
            cfg, strat = dna_to_strategy(dna)
            with spread_cost_mode(costs):
                setup = slow_cell_setup(cell)
            equity = np.asarray(run_slow_backtest(cell, setup, dna, cfg, strat, notes="portfolio").equity_curve)
        series[e["id"]] = daily_returns(np.asarray(cell.bars.time), equity)
    return series


__all__ = ["CORR_BARS", "MIN_SHARED_DAYS", "SHARE_BARS", "axes", "build", "build_series", "daily_returns", "grade",
           "judge"]
