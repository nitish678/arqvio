"""A stand-in for engine.research.ga.runner in the campaign-verb tests, called by the app's real launcher through
``main(argv)``: the same CLI surface and output locations (the id from the app's own campaign_target, under the
ARQVIO_HOME it is given), a trials.sqlite registry trimmed on resume like run_pinned's, and a JSON "checkpoint" -- it
runs no backtest. Knobs: FAKE_GEN_SLEEP (seconds per generation), FAKE_HOLD_AT_GEN (after that generation's
checkpoint a fresh run waits to be stopped), FAKE_PROBE_FAIL (a probe exits 3), FAKE_FINALISTS (the manifest's count)."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from engine.api.routes_app_campaigns import campaign_target


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    for flag in ("--cell", "--engine", "--checkpoint", "--resume"):
        ap.add_argument(flag)
    for flag in ("--pop", "--gen", "--seed", "--budget", "--final-n", "--workers", "--islands"):
        ap.add_argument(flag, type=int)
    ap.add_argument("--monitor-snapshot-every", type=float)
    a = ap.parse_args(argv)
    probe = a.checkpoint is None
    if not probe and not (Path(a.checkpoint).parent / "preregistration.json").is_file():
        print("REFUSED: the runner started before the pre-registration record existed")
        return 4
    if not probe:  # the interpreter's own pid (the venv launcher is a separate parent process)
        (Path(a.checkpoint).parent / "fake_pid").write_text(str(os.getpid()), encoding="utf-8")
    if probe and os.environ.get("FAKE_PROBE_FAIL"):
        print("probe failure (FAKE_PROBE_FAIL)")
        return 3
    t = campaign_target(a.cell, {"pop": a.pop, "gen": a.gen, "budget": a.budget, "final_n": a.final_n}, a.seed,
                        a.islands)
    cid, out = t["campaign_id"], t["dir"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"ARQVIO_HOME={os.environ['ARQVIO_HOME']}")
    print("2026-10-10 08:00:00.000 | WARNING  | fake:main:1 | a loguru-shaped warning", flush=True)
    db = sqlite3.connect(out / "trials.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS trials (campaign_id TEXT, generation INTEGER, k INTEGER)")
    first = 0
    if a.resume:
        state = json.loads(Path(a.resume).read_text(encoding="utf-8"))
        first = state["generation"] + 1
        db.execute("DELETE FROM trials WHERE campaign_id = ? AND rowid NOT IN "
                   "(SELECT rowid FROM trials WHERE campaign_id = ? ORDER BY rowid LIMIT ?)",
                   (cid, cid, state["registry_rows"]))
        db.commit()
        print(f"RESUME from {a.resume} after generation {state['generation']}", flush=True)
    hold = int(os.environ.get("FAKE_HOLD_AT_GEN", "-1"))
    for g in range(first, a.gen):
        time.sleep(float(os.environ.get("FAKE_GEN_SLEEP", "0")))
        db.executemany("INSERT INTO trials VALUES (?, ?, ?)", [(cid, g, k) for k in range(a.pop)])
        db.commit()
        if a.checkpoint:
            rows = db.execute("SELECT COUNT(*) FROM trials WHERE campaign_id = ?", (cid,)).fetchone()[0]
            tmp = Path(a.checkpoint).with_suffix(".tmp")
            tmp.write_text(json.dumps({"generation": g, "registry_rows": rows, "config_hash": t["config_hash"]}),
                           encoding="utf-8")
            os.replace(tmp, a.checkpoint)
            (out / "monitor_snapshot.json").write_text(json.dumps({"generation": g}), encoding="utf-8")
        print(f"  gen {g:>2}: new={a.pop} (stand-in)", flush=True)
        if a.checkpoint:
            print(f"  CHECKPOINT after generation {g} -> {a.checkpoint}", flush=True)
        if g == hold and a.checkpoint and not a.resume:
            print("holding after the checkpoint (FAKE_HOLD_AT_GEN)", flush=True)
            time.sleep(300)
    db.close()
    (out / "manifest_v2.json").write_text(json.dumps({"run": {
        "stop_reason": "generations", "trials_used": a.budget // 2, "survivors": 3,
        "finalists": int(os.environ.get("FAKE_FINALISTS", a.final_n)),
        "wall_seconds": {"prepare": 2.0, "search": 8.0, "cross": 4.0, "total": 14.0}}}), encoding="utf-8")
    return 0
