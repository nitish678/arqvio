"""The S7 portfolio analysis CLI: the replay plus every derived artifact the read-only API serves, and their alerts.

``python -m engine.research.portfolio.analyze [--out DIR] [--no-alerts]`` writes, under DIR (default
``<ARQVIO artifacts>/portfolio``, resolved at call time), canonical JSON:

* ``replay.json``: the shared-account replay and its verification (run.build);
* ``allocation.json``: HRP / ERC on the member return panel;
* ``diversity.json``: the census, with equal, HRP and ERC weightings;
* ``heat.json``: the per-bar heat, its per-symbol rollup, the HEAT_ALERT crossings;
* ``drift.json``: the drift monitor on the real member panel -- REPORTED, not judged (DR §5: flat bars give many
  zero returns, so the KS p-values are conservative).

Every artifact carries the replay's ``run_hash``. Unless ``--no-alerts`` is given, the replay's HEAT_HIGH / REFUSED
events and any drift rejection are emitted into the audited spool (alerts.dispatch.emit; the env-gated adapters
record SKIPPED when unconfigured).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engine.core.paths import get_artifacts_dir
from engine.research.alerts.adapters import DesktopAdapter, EmailAdapter, TelegramAdapter
from engine.research.alerts.dispatch import emit
from engine.research.alerts.producers import from_drift, from_replay
from engine.research.portfolio.allocation import allocate
from engine.research.portfolio.diversity import census
from engine.research.portfolio.drift import Q, RECENT, REF_END, REF_START, STEP, monitor
from engine.research.portfolio.heat import HEAT_ALERT, HEAT_CAP, upward_crossings
from engine.research.portfolio.members import E0, member_set
from engine.research.portfolio.run import build, canonical
from engine.research.portfolio.sim import replay, return_panel


def analyze(out_dir: Path, *, alerts: bool = True) -> dict[str, Path]:
    ms = member_set()
    rep = replay(list(ms.runs), e0=E0)
    art = build(ms, rep)
    rh = art["run_hash"]
    mids = [m.mid for m in ms.members]
    panel = return_panel(list(ms.runs), rep.axis)
    alloc = allocate(panel, mids) | {"run_hash": rh}
    weightings = {"equal": [1.0 / len(mids)] * len(mids),
                  "HRP": [alloc["HRP"]["weights"][m] for m in mids], "ERC": [alloc["ERC"]["weights"][m] for m in mids]}
    div = census(panel, list(ms.members), weightings) | {"run_hash": rh}
    series = [b["heat"] for b in rep.bars]
    heat = {"schema": "arqvio.portfolio.heat", "version": 1, "run_hash": rh, "alert_threshold": HEAT_ALERT,
            "cap": HEAT_CAP, "max_heat": max((h for h in series if h is not None), default=None),
            "crossings_t": [rep.bars[k]["t"] for k in upward_crossings(series)],
            "series": [{"t": b["t"], "heat": b["heat"], "unbounded": b["heat_unbounded"],
                        "per_symbol": b["heat_per_symbol"]} for b in rep.bars]}
    mon = monitor({m: panel[:, k] for k, m in enumerate(mids)})
    drift = {"schema": "arqvio.portfolio.drift", "version": 1, "run_hash": rh, "judged": False,
             "note": "the monitor on the real member panel is reported, not judged (DR §5)",
             "geometry": {"reference": [REF_START, REF_END], "recent": RECENT, "step": STEP}, "q": Q,
             "multiplicity": "Benjamini-Yekutieli over all members x checks", "by_cut": mon.cut,
             "checks": [{"member": c.member, "t": c.t, "d": c.d, "p": c.p} for c in mon.checks],
             "rejected": [{"member": c.member, "t": c.t, "p": c.p} for c in mon.rejected]}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, obj in (("replay", art), ("allocation", alloc), ("diversity", div), ("heat", heat), ("drift", drift)):
        p = out_dir / f"{name}.json"
        p.write_text(canonical(obj) + "\n", encoding="utf-8", newline="\n")
        paths[name] = p
    if alerts:
        adapters = [TelegramAdapter(), DesktopAdapter(), EmailAdapter()]
        for a in from_replay(rep.events, str(paths["replay"])) + from_drift(mon, str(paths["drift"])):
            emit(a, adapters)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-alerts", action="store_true")
    args = ap.parse_args(argv)
    paths = analyze(args.out or get_artifacts_dir(create=True) / "portfolio", alerts=not args.no_alerts)
    for name, p in paths.items():
        print(f"WROTE {name}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
