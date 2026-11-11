"""Build and write the replay artifact (DR §2 checks (a)-(c)); the run_hash; the CLI.

``python -m engine.research.portfolio.run [--out DIR]`` runs the declared member set through the replay and writes
``DIR/replay.json`` (default: ``<ARQVIO artifacts>/portfolio``, resolved at call time). The JSON is canonical
(sorted keys, fixed separators, no NaN), so two runs on the same inputs are byte-identical -- the determinism check
diffs two fresh-process runs. ``run_hash`` is the SHA-256 of the canonical JSON of the declared inputs only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from engine.core.paths import get_artifacts_dir
from engine.research.portfolio.heat import HEAT_ALERT, HEAT_CAP
from engine.research.portfolio.members import CELLS, E0, SIM_BARS, MemberSet, member_set, member_share
from engine.research.portfolio.sim import (FLOOR_SHARE, MARGIN_CAP, MARGIN_LEVERAGE, RECON_TOL, Replay,
                                           member_reconstruction_max_abs, replay)

CODE_VERSION = "astra-s7-portfolio-v1"  # frozen: part of every portfolio run hash (the pre-ARQVIO name)
SCHEMA = "arqvio.portfolio.replay"


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def declared_inputs(ms: MemberSet) -> dict:
    return {
        "code": CODE_VERSION, "e0": E0, "member_equity": member_share(), "window_bars": SIM_BARS,
        "cells": [{"symbol": s, "dataset_id": d} for s, d in CELLS],
        "caps": {"heat_cap": HEAT_CAP, "heat_alert": HEAT_ALERT, "margin_cap": MARGIN_CAP,
                 "margin_leverage": MARGIN_LEVERAGE, "floor_share": FLOOR_SHARE},
        "members": [{"mid": m.mid, "family": m.family, "symbol": m.symbol, "dataset_id": m.dataset_id,
                     "battery_index": m.battery_index, "dna_hash": m.dna.dna_hash} for m in ms.members],
    }


def run_hash(ms: MemberSet) -> str:
    return hashlib.sha256(canonical(declared_inputs(ms)).encode("utf-8")).hexdigest()


def build(ms: MemberSet, rep: Replay) -> dict:
    member_recon = {r.mid: member_reconstruction_max_abs(r) for r in ms.runs}
    return {
        "schema": SCHEMA, "version": 1, "run_hash": run_hash(ms), "inputs": declared_inputs(ms),
        "skipped_candidates": list(ms.skipped),
        "members": [{"mid": r.mid, "n_trades": r.result.n_trades, "final_equity": r.result.final_equity,
                     "reconstruction_max_abs": member_recon[r.mid]} for r in ms.runs],
        "verification": {
            "conservation_bars": rep.conservation_bars, "axis_bars": len(rep.axis),
            "conservation_exact": rep.conservation_bars == len(rep.axis),
            "recon_max_abs": rep.recon_max_abs, "recon_tol": RECON_TOL, "recon_ok": rep.recon_max_abs < RECON_TOL,
            "member_reconstruction_max_abs": max(member_recon.values()),
        },
        "refused": rep.refused,
        "events": rep.events,
        "bars": rep.bars,
    }


def write(artifact: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "replay.json"
    path.write_text(canonical(artifact) + "\n", encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    ms = member_set()
    art = build(ms, replay(list(ms.runs), e0=E0))
    path = write(art, args.out or get_artifacts_dir(create=True) / "portfolio")
    v = art["verification"]
    print(f"WROTE {path} run_hash {art['run_hash']} conservation {v['conservation_bars']}/{v['axis_bars']} "
          f"exact={v['conservation_exact']} recon_max_abs {v['recon_max_abs']:.3e} recon_ok={v['recon_ok']} "
          f"member_recon_max {v['member_reconstruction_max_abs']:.3e} refused {len(art['refused'])}")
    return 0 if v["conservation_exact"] and v["recon_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
