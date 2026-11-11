"""Live performance of the deployed bots, from the terminal's own deal history (ARQVIO_V4.md A13-A16).

For each bot (its magic number) the closed deals in MetaTrader 5 are its live trades: count, net, wins. They are set
against the strategy's own backtest: the backtest's closed-trade P&L, scaled from the backtest lot to the bot's live
lot, is bootstrapped (BOOT draws, seeded by the dna_hash) into the band of the cumulative P&L after the same number
of trades -- p5, p50, p95. Where the live sum sits in that band is the verdict:

    TOO FEW     fewer than MIN_TRADES live trades: nothing is judged yet
    HOLDING     the live sum is inside the band
    AHEAD       above p95
    WEAKENING   below p5 (the edge may be gone; worker4 plan: reassess at 15 trades)

Orphans are ARQVIO-range magics with deals or open positions but no bot in ARQVIO's list. Reads only: no order, no
setting, nothing written to the terminal.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

MIN_TRADES = 15
BOOT = 2000
OUT_ENTRIES = (1, 2, 3)  # DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY


def deal_trades(deals: list[Any], magic: int) -> list[dict[str, Any]]:
    """Closing deals of one magic, oldest first: their net P&L (profit + swap + commission) and time."""
    out = []
    for d in deals:
        if int(getattr(d, "magic", 0)) != int(magic) or int(getattr(d, "entry", 0)) not in OUT_ENTRIES:
            continue
        net = float(getattr(d, "profit", 0.0)) + float(getattr(d, "swap", 0.0)) + float(getattr(d, "commission", 0.0))
        out.append({"time": int(getattr(d, "time", 0)), "net": net, "volume": float(getattr(d, "volume", 0.0))})
    return sorted(out, key=lambda x: x["time"])


def band(backtest_pnl: np.ndarray, n: int, *, seed: int, boot: int = BOOT) -> dict[str, float] | None:
    """p5 / p50 / p95 of the sum of n trades drawn (with replacement) from the backtest's trade P&L."""
    p = np.asarray(backtest_pnl, dtype=np.float64)
    if n < 1 or p.size == 0:
        return None
    rng = np.random.default_rng(seed)
    sums = p[rng.integers(0, p.size, size=(boot, n))].sum(axis=1)
    return {"p5": float(np.percentile(sums, 5)), "p50": float(np.percentile(sums, 50)), "p95": float(np.percentile(sums, 95))}


def verdict(n: int, net: float, b: dict[str, float] | None) -> str:
    if n < MIN_TRADES or b is None:
        return "TOO FEW"
    if net < b["p5"]:
        return "WEAKENING"
    if net > b["p95"]:
        return "AHEAD"
    return "HOLDING"


def judge_bot(bot: dict[str, Any], trades: list[dict[str, Any]], backtest_pnl: np.ndarray, backtest_lot: float,
              live_lot: float) -> dict[str, Any]:
    n = len(trades)
    net = float(sum(t["net"] for t in trades))
    scale = (live_lot / backtest_lot) if backtest_lot > 0 else 0.0
    seed = int(str(bot.get("dna_hash", "0"))[:8] or "0", 16)
    b = band(np.asarray(backtest_pnl) * scale, n, seed=seed) if n else None
    wins = sum(1 for t in trades if t["net"] > 0)
    return {"bot": bot["bot"], "magic": bot["magic"], "trades": n, "net": round(net, 2),
            "win_rate": (wins / n) if n else None, "band": b, "verdict": verdict(n, net, b),
            "last_trade": datetime.fromtimestamp(trades[-1]["time"], tz=UTC).isoformat() if trades else None,
            "lot_scale_vs_backtest": round(scale, 4)}


# research incubation ladder (PART 4.10): 25 % -> (~10 live trades) -> 50 % -> (~30 live trades) -> 100 %, stepping
# only on live trades judged against the bot's own backtest band, never on time alone.
PROMOTE_AFTER: tuple[int, ...] = (10, 30)


def incubation_advice(stage: int, judged: dict[str, Any], stages: tuple[float, ...]) -> dict[str, Any]:
    """Whether a bot has earned its next size (worker4 W4-7 7.9). ADVICE ONLY: ARQVIO never changes a live bot's lot
    by itself (owner rule: a deployed bot's sizing is the owner's); the LIVE tab shows the checks and the owner
    presses Scale up."""
    if stage >= len(stages) - 1:
        return {"stage": stage, "lot_scale": stages[stage], "next_lot_scale": None, "ready": False,
                "checks": [], "note": "full size"}
    n = int(judged.get("trades") or 0)
    net = float(judged.get("net") or 0.0)
    b = judged.get("band")
    need = PROMOTE_AFTER[min(stage, len(PROMOTE_AFTER) - 1)]
    checks = [
        {"check": "live trades", "value": n, "need": f">= {need}", "passed": n >= need},
        {"check": "net profit", "value": round(net, 2), "need": "> 0", "passed": net > 0},
        {"check": "not below its backtest band (p5)", "value": None if b is None else round(b["p5"], 2),
         "need": "net >= p5", "passed": b is not None and net >= b["p5"]},
        {"check": "verdict", "value": judged.get("verdict"), "need": "not WEAKENING",
         "passed": judged.get("verdict") not in ("WEAKENING", "UNREAD", "NO BACKTEST")},
    ]
    return {"stage": stage, "lot_scale": stages[stage], "next_lot_scale": stages[stage + 1],
            "ready": all(c["passed"] for c in checks), "checks": checks,
            "note": "advice only: ARQVIO never changes a live bot's size itself"}


def orphans(deals: list[Any], positions: list[Any], known: set[int], lo: int, hi: int) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for d in deals:
        m = int(getattr(d, "magic", 0))
        if lo <= m <= hi and m not in known:
            r = seen.setdefault(m, {"magic": m, "deals": 0, "open_positions": 0, "symbol": str(getattr(d, "symbol", ""))})
            r["deals"] += 1
    for p in positions:
        m = int(getattr(p, "magic", 0))
        if lo <= m <= hi and m not in known:
            r = seen.setdefault(m, {"magic": m, "deals": 0, "open_positions": 0, "symbol": str(getattr(p, "symbol", ""))})
            r["open_positions"] += 1
    return sorted(seen.values(), key=lambda r: r["magic"])


__all__ = ["BOOT", "MIN_TRADES", "band", "deal_trades", "judge_bot", "orphans", "verdict"]
