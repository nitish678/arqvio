"""Fills-matched power check: could the GA's own gates have seen a planted edge?

Executes docs/decisions/DR-2026-10-01-fills-matched-power-check.md exactly:
planted (canary / W3 machinery) and no-plant candidates, each paired with a
real Stage-2 survivor as host, judged by the campaign's own gates -- kill
gates, DSR at a luck bar N, sham margin -- under two sham regimes:

  (a) v2      the GA's own sham (sham_backtest_factory, salt astra-s3-sham-v1);
  (b) matched the same v2 draw restricted to bars whose fill bar the gate
              admits: a near-full-fill REFERENCE for this check only, never a
              campaign control, never a regime.

It is not a campaign: it writes no trial registry and judges nothing but its
own planted candidates. It never re-judges the GA run.

    python -m engine.research.ga.power_check --cell <cell_id> --campaign <campaign dir> --out <dir>
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from engine.research.ga.config import CAMPAIGN_GATES, Gates
from engine.research.ga.determinism import canonical
from engine.research.ga.fitness import candidate_dsr, dsr_trials
from engine.research.power.context import PowerContext, resolve_power_context
from engine.research.search.canary import PLANTED_SCALE
from engine.research.search.dna import DNA
from engine.research.search.runner import TrialResult

POWER_SEED_PREFIX = "astra-s4-power-v1"  # frozen: seeds the power check (the pre-ARQVIO name)
STRENGTHS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
REPLICATES = 200
# The protocol's fixed screening luck bars. The primary bar is the evaluated campaign's own N (luck_bars).
SCREENING_LUCK_BARS: tuple[int, ...] = (20, 200, 1000)
POWERED_RATE = 0.95
MAX_FPR = 0.05
REGIMES = ("v2", "matched")


def plant_seed(strength: float, replicate: int) -> int:
    return int(hashlib.sha256(f"{POWER_SEED_PREFIX}:{strength}:{replicate}".encode()).hexdigest()[:16], 16)


def planted_series(strength: float, replicate: int, n: int, ppy: int) -> np.ndarray:
    """i.i.d. normal per-period returns at a target annual Sharpe (canary scale)."""
    mu = strength * PLANTED_SCALE / math.sqrt(ppy)
    return np.random.default_rng(plant_seed(strength, replicate)).normal(mu, PLANTED_SCALE, size=int(n))


def annual_sharpe(returns: np.ndarray, ppy: int) -> float:
    """The factories' arithmetic: mean / std(ddof=1) * sqrt(ppy); 0.0 if degenerate."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2 or np.std(r) <= 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(ppy))


def admitted_entry_mask(times: np.ndarray, spread: np.ndarray, *, max_spread_points: float | None,
                        no_trade_hours: Sequence[int], spread_cap_by_bar: np.ndarray | None = None) -> np.ndarray:
    """mask[i]: an entry emitted at bar i would FILL at bar i+1's open.

    The engine refuses a fill when spread[i+1] > max_spread_points and defers
    it when bar i+1's UTC hour is a no-trade hour (engine.py). The last bar has
    no next bar and is never admitted.
    """
    t = np.asarray(times, dtype=np.int64)
    s = np.asarray(spread, dtype=np.float64)
    ok = np.zeros(t.size, dtype=bool)
    nxt_hour = (t[1:] // 3600) % 24
    admitted = ~np.isin(nxt_hour, np.asarray(list(no_trade_hours), dtype=np.int64))
    if spread_cap_by_bar is not None:  # v4 regime gate: refused only where spread > cap (NaN cap: no gate)
        admitted &= ~(s[1:] > np.asarray(spread_cap_by_bar, dtype=np.float64)[1:])
    elif max_spread_points is not None:
        admitted &= s[1:] <= float(max_spread_points)
    ok[:-1] = admitted
    return ok


def draw_admitted_entries(dna_hash: str, salt: str, lo: int, hi: int, n_entries: int,
                          mask: np.ndarray) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """sham_control.draw_entries, over the admitted bars of [lo, hi] only."""
    from engine.research.search.sham_control import sham_seed_hex

    rng = np.random.default_rng(int(sham_seed_hex(dna_hash, salt), 16))
    idx = np.arange(lo, hi + 1, dtype=np.int64)
    eligible = idx[mask[lo:hi + 1]]
    k = min(max(0, int(n_entries)), int(eligible.size))
    if k == 0:
        return (), ()
    bars = np.sort(rng.choice(eligible, size=k, replace=False))
    longs = rng.random(k) < 0.5
    return tuple(int(b) for b in bars), tuple("long" if x else "short" for x in longs)


def matched_sham_factory(cell, *, salt: str) -> Callable[[DNA, int], TrialResult]:
    """Regime (b): the v2 sham drawn only on gate-admitted bars. Diagnostic only."""
    from engine.research.backtest.strategies.sham import ShamConfig, ShamStrategy
    from engine.research.search.real_backtest import dna_to_strategy, run_slow_trial, slow_cell_setup
    from engine.research.search.sham_control import eligible_entry_range, sham_seed_hex

    setup = slow_cell_setup(cell, caller="power_check.matched_sham_factory")
    mask = admitted_entry_mask(np.asarray(cell.bars.time), np.asarray(cell.bars.spread),
                               max_spread_points=setup.max_spread_points, no_trade_hours=setup.no_trade_hours,
                               spread_cap_by_bar=setup.spread_cap_by_bar)

    def _sham(dna: DNA, n_entries: int) -> TrialResult:
        cfg, _ = dna_to_strategy(dna)
        lo, hi = eligible_entry_range(cfg, cell.n_bars)
        bars, sides = draw_admitted_entries(dna.dna_hash, salt, lo, hi, n_entries, mask)
        sham_cfg = ShamConfig(entry_bars=bars, sides=sides, volume=cfg.volume,
                              strategy_id=f"shamfm_{sham_seed_hex(dna.dna_hash, salt)}",
                              sl_atr=cfg.sl_atr, tp_atr=cfg.tp_atr, atr_period=cfg.atr_period)
        return run_slow_trial(cell, setup, dna, sham_cfg, ShamStrategy(sham_cfg),
                              notes=f"fills-matched power-check sham of {dna.dna_hash}")

    _sham.admitted_share = float(mask.mean())
    return _sham


def load_hosts(individuals: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The first n survivors in admission order: DNA, trades, attempted entries, recorded sham."""
    hosts = []
    for row in individuals:
        f = row["fitness"]
        if f["killed_by"] is not None:
            continue
        m = row["metrics"]
        hosts.append({
            "dna": DNA(**row["dna"]),
            "dna_hash": row["dna_hash"],
            "n_trades": int(m["n_trades"]),
            "attempted": int(m["n_trades"]) + int(m["blocked_max_spread"]),
            "recorded_sham_sharpe": f["sham_sharpe"],
        })
        if len(hosts) == n:
            break
    if len(hosts) < n:
        raise ValueError(f"only {len(hosts)} survivors, {n} hosts pre-registered")
    return hosts


def archive_sharpes(individuals: list[dict[str, Any]]) -> list[float]:
    """Finite trial Sharpes of the run, admission order."""
    out = []
    for row in individuals:
        s = (row["metrics"] or {}).get("sharpe")
        if s is not None and isinstance(s, int | float) and math.isfinite(s):
            out.append(float(s))
    return out


def judge(returns: np.ndarray, n_trades: int, sham_sharpe: float, *, n_trials: int, prior: Sequence[float],
          ctx: PowerContext, gates: Gates = CAMPAIGN_GATES) -> dict[str, Any]:
    """One planted candidate through the campaign's own gates at luck bar n_trials, on ctx's cell."""
    sharpe = annual_sharpe(returns, ctx.ppy)
    trial = TrialResult(dna=None, sharpe=sharpe, n_trades=n_trades, per_period_returns=returns)  # type: ignore[arg-type]
    n, sharpes = dsr_trials([*prior[: n_trials - 1], sharpe], n_trials)
    reading = candidate_dsr(trial, n_trials=n, trial_sharpes=sharpes, symbol=ctx.symbol, timeframe=ctx.timeframe)
    trades_ok = n_trades >= gates.min_trades
    sharpe_ok = sharpe > gates.sharpe_min
    dsr_ok = reading.dsr is not None and reading.dsr >= gates.dsr_min
    margin = sharpe - float(sham_sharpe)
    sham_ok = margin >= gates.sham_margin_min
    return {"sharpe": sharpe, "dsr": reading.dsr, "dsr_status": reading.status, "n_trials": n,
            "sham_margin": margin, "trades_ok": trades_ok, "sharpe_ok": sharpe_ok, "dsr_ok": dsr_ok,
            "sham_ok": sham_ok, "detected": trades_ok and sharpe_ok and dsr_ok and sham_ok}


def luck_bars(ctx: PowerContext) -> tuple[int, ...]:
    """The screening bars plus the evaluated campaign's own N (primary), sorted, without duplicates."""
    return tuple(sorted({*SCREENING_LUCK_BARS, ctx.primary_n}))


def plant_rows(hosts: list[dict[str, Any]], shams: dict[str, dict[str, float]], prior: Sequence[float],
               n_returns: int, ctx: PowerContext) -> list[dict[str, Any]]:
    rows = []
    for s in STRENGTHS:
        for r in range(REPLICATES):
            host = hosts[r]
            returns = planted_series(s, r, n_returns, ctx.ppy)
            for regime in REGIMES:
                for n in luck_bars(ctx):
                    rows.append({"strength": s, "replicate": r, "host": host["dna_hash"], "regime": regime,
                                 "N": n, **judge(returns, host["n_trades"], shams[regime][host["dna_hash"]],
                                                 n_trials=n, prior=prior, ctx=ctx)})
    return rows


def power_table(rows: list[dict[str, Any]], ctx: PowerContext) -> list[dict[str, Any]]:
    out = []
    for s in STRENGTHS:
        for regime in REGIMES:
            for n in luck_bars(ctx):
                sel = [x for x in rows if x["strength"] == s and x["regime"] == regime and x["N"] == n]
                k = len(sel)
                out.append({
                    "strength": s, "regime": regime, "N": n, "replicates": k,
                    "detected": sum(x["detected"] for x in sel) / k,
                    "sharpe_gate": sum(x["sharpe_ok"] for x in sel) / k,
                    "dsr_gate": sum(x["dsr_ok"] for x in sel) / k,
                    "sham_gate": sum(x["sham_ok"] for x in sel) / k,
                    "median_sharpe": float(np.median([x["sharpe"] for x in sel])),
                    "median_dsr": float(np.median([x["dsr"] if x["dsr"] is not None else 0.0 for x in sel])),
                })
    return out


def verdict(table: list[dict[str, Any]], ctx: PowerContext) -> dict[str, Any]:
    """The pre-registered reading at the primary N, the evaluated campaign's own luck bar."""
    def rate(s, regime, n=ctx.primary_n):
        return next(t["detected"] for t in table if t["strength"] == s and t["regime"] == regime and t["N"] == n)

    powered = [s for s in STRENGTHS if s > 0 and all(rate(s, g) >= POWERED_RATE for g in REGIMES)]
    fpr = {f"{g}@N={n}": rate(0.0, g, n) for g in REGIMES for n in luck_bars(ctx)}
    return {
        "smallest_powered_strength": min(powered) if powered else None,
        "powered_at_1_5": 1.5 in powered,
        "rate_1_5": {g: rate(1.5, g) for g in REGIMES},
        "fpr": fpr,
        "controlled": all(v <= MAX_FPR for v in fpr.values()),
    }


def _shams(hosts, cell, *, log: Callable[[str], None]) -> dict[str, dict[str, Any]]:
    from engine.research.search.sham_control import SHAM_SALT, sham_backtest_factory

    factories = {"v2": sham_backtest_factory(cell), "matched": matched_sham_factory(cell, salt=SHAM_SALT)}
    out: dict[str, dict[str, Any]] = {g: {} for g in REGIMES}
    fills: dict[str, dict[str, int]] = {g: {"requested": 0, "filled": 0} for g in REGIMES}
    for i, host in enumerate(hosts):
        for g in REGIMES:
            res = factories[g](host["dna"], host["attempted"])
            out[g][host["dna_hash"]] = float(res.sharpe)
            fills[g]["requested"] += host["attempted"]
            fills[g]["filled"] += int(res.n_trades or 0)
        if (i + 1) % 20 == 0:
            log(f"  shams {i + 1}/{len(hosts)} hosts")
    return {"sharpe": out, "fills": fills, "admitted_share": factories["matched"].admitted_share}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from engine.core.paths import get_arqvio_home
    from engine.research.search.real_backtest import load_cell_bars_from_snapshot

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", required=True, help="registry cell_id of the evaluated campaign's search cell")
    p.add_argument("--campaign", required=True, type=Path, help="the campaign dir: manifest_v2.json + individuals.json")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    print(f"ARQVIO_HOME={get_arqvio_home()}")
    ctx = resolve_power_context(a.cell, a.campaign / "manifest_v2.json")
    print(f"context {json.dumps(ctx.as_dict(), sort_keys=True)}; luck bars {luck_bars(ctx)}")
    individuals_path = a.campaign / "individuals.json"
    individuals = json.loads(individuals_path.read_text(encoding="utf-8"))
    print(f"individuals: {individuals_path} sha256={hashlib.sha256(individuals_path.read_bytes()).hexdigest()} "
          f"rows={len(individuals)}")
    hosts = load_hosts(individuals, REPLICATES)
    prior = archive_sharpes(individuals)
    print(f"hosts={len(hosts)} prior_sharpes={len(prior)}")
    cell = load_cell_bars_from_snapshot(*ctx.cell)
    print(f"cell {ctx.cell}: n_bars={cell.n_bars} n_returns={cell.n_returns} source={cell.source}")
    t0 = time.perf_counter()
    sham = _shams(hosts, cell, log=print)
    t1 = time.perf_counter()
    mismatches = [h["dna_hash"] for h in hosts if sham["sharpe"]["v2"][h["dna_hash"]] != h["recorded_sham_sharpe"]]
    print(f"shams done in {t1 - t0:.1f}s; fills {sham['fills']}; admitted bar share {sham['admitted_share']:.4f}")
    print(f"sanity: regime-(a) sham == recorded GA sham for {len(hosts) - len(mismatches)}/{len(hosts)} hosts")
    if mismatches:
        print(f"STOP: regime-(a) shams do not reproduce the GA's recorded shams: {mismatches[:5]}")
        return 3
    rows = plant_rows(hosts, sham["sharpe"], prior, cell.n_returns, ctx)
    t2 = time.perf_counter()
    again = plant_rows(hosts, sham["sharpe"], prior, cell.n_returns, ctx)
    t3 = time.perf_counter()
    h1, h2 = (hashlib.sha256(canonical(x).encode()).hexdigest() for x in (rows, again))
    print(f"plants+DSR: {len(rows)} judgements in {t2 - t1:.1f}s; re-derived in {t3 - t2:.1f}s; "
          f"rows sha256 {h1} / {h2} {'IDENTICAL' if h1 == h2 else 'DIFFER'}")
    table = power_table(rows, ctx)
    v = verdict(table, ctx)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "power_rows.json").write_text(canonical(rows), encoding="utf-8")
    doc = {"table": table, "verdict": v, "shams": {"fills": sham["fills"], "admitted_share": sham["admitted_share"]},
           "context": ctx.as_dict(),
           "rows_sha256": h1, "rows_rederived_sha256": h2, "hosts": [h["dna_hash"] for h in hosts],
           "design": "docs/decisions/DR-2026-10-01-fills-matched-power-check.md",
           "wall_seconds": {"shams": t1 - t0, "plants": t2 - t1, "rederive": t3 - t2}}
    (a.out / "power_table.json").write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print("strength | regime  |    N | detected | sharpe>0 | dsr>=0.5 | margin>=0 | median sharpe | median dsr")
    for t in table:
        print(f"{t['strength']:8} | {t['regime']:7} | {t['N']:4} | {t['detected']:8.3f} | {t['sharpe_gate']:8.3f} | "
              f"{t['dsr_gate']:8.3f} | {t['sham_gate']:9.3f} | {t['median_sharpe']:13.4f} | {t['median_dsr']:.4g}")
    print(f"VERDICT {json.dumps(v, sort_keys=True)}")
    return 0 if h1 == h2 else 4


if __name__ == "__main__":
    sys.exit(main())
