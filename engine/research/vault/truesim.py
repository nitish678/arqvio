"""TRUE SIM for a vault portfolio (worker4 W4-8 8.6; idea: research "TRUE SIM portfolio simulation, the TOGETHER
line"): the members on ONE account through the shared-account replay (portfolio/sim.py -- the frozen slow engine,
one exact cash ledger, heat / margin / floor caps that refuse trades), instead of averaging separate curves.

Members must be on H1 pinned cells (the replay's cells, mql5/cell_config.campaign_cell); others are listed as skipped,
never approximated. The curve is down-sampled to at most MAX_POINTS for the screen; every number is the replay's.
"""
from __future__ import annotations

from typing import Any

E0 = 10_000.0
MAX_POINTS = 400


def run(entries: list[dict[str, Any]], *, e0: float = E0) -> dict[str, Any]:
    from engine.research.mql5.cell_config import campaign_cell
    from engine.research.portfolio.sim import replay, run_member
    from engine.research.search.dna import DNA

    runs, skipped, cells = [], [], {}
    for e in entries:
        if e.get("timeframe") != "H1":
            skipped.append({"id": e["id"], "reason": f"{e.get('timeframe')} cell: the replay runs H1 cells"})
            continue
        try:
            key = (e["symbol"], e["dataset_id"])
            cell = cells.get(key) or cells.setdefault(key, campaign_cell(*key))
            dna = DNA(**{k: tuple(v) if isinstance(v, list) else v for k, v in e["dna"].items()})
            runs.append(run_member(e["id"], dna, cell, initial_equity=e0))
        except Exception as exc:  # named per member: a damaged snapshot skips that member, never the whole sim
            skipped.append({"id": e["id"], "reason": f"{type(exc).__name__}: {exc}"})
    if not runs:
        return {"state": "EMPTY", "members": 0, "skipped": skipped}
    rep = replay(runs, e0=e0)
    bars = rep.bars
    step = max(1, len(bars) // MAX_POINTS)
    curve = [{"t": int(b["t"]), "equity": float(b["equity"])} for b in bars[::step]]
    if bars and curve[-1]["t"] != int(bars[-1]["t"]):
        curve.append({"t": int(bars[-1]["t"]), "equity": float(bars[-1]["equity"])})
    eq = [float(b["equity"]) for b in bars]
    peak, dd = e0, 0.0
    for v in eq:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak if peak > 0 else 0.0)
    conflicts = sum(1 for ev in rep.events if ev.get("kind") == "CONFLICT")
    return {"state": "DONE", "members": len(runs), "skipped": skipped, "e0": e0,
            "final_equity": eq[-1] if eq else e0, "return": (eq[-1] / e0 - 1.0) if eq else 0.0, "max_dd": dd,
            "refused_trades": len(rep.refused), "conflicts": conflicts, "curve": curve,
            "reconciliation_max_abs": float(rep.recon_max_abs)}


__all__ = ["E0", "MAX_POINTS", "run"]
