"""``python -m engine.research.v1.cli`` -- one V1 (research engine) campaign on one market's cell.

    --cell C --preset research|quick --seed N --risk 0.01 --power eco|medium|high|max --out DIR [--resume]

Writes ``DIR/v1.log`` (the feed), ``DIR/v1_checkpoint.json`` (every generation) and ``DIR/v1_result.json`` (the run,
each finalist's vault gates, the strategies that passed). A file ``DIR/skip`` asks the run to finish its generation
and go to the Final War. ``engine/research/vault/library.import_v1`` adds the passing strategies to the vault.
Exit 0 = completed (with or without vault strategies), 2 = the cell refused, 1 = a crash.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from engine.research.v1.config import v1_config
from engine.research.v1.engine import V1Run, _json_default


def _logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} | {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def held_out_window(cell: str) -> dict[str, Any] | None:
    from engine.research.vault import held_out

    rec = held_out.record_for({"cell": cell}) or {}
    ho = rec.get("held_out") or {}
    return ho if ho.get("state") == "KEPT" else None


def gate_finalists(result: dict[str, Any], *, cell: str, ev: Any, cfg: Any, log) -> list[dict[str, Any]]:
    """The vault gates for every finalist on the market's whole window (gates.judge)."""
    from engine.research.search.dna import DNA
    from engine.research.search.real_backtest import fast_backtest_factory, load_cell_bars_from_snapshot
    from engine.research.search.sham_control import sham_backtest_factory
    from engine.research.v1.gates import judge
    from engine.research.vault.robustness import neighbours

    ho = held_out_window(cell)
    if ho is None:
        log("VAULT GATES · this market kept no held-out bars; V1 vaults nothing without them (research: 70/30)")
        return [{"dna_hash": f["dna_hash"], "passed": False, "failed": ["NO_HELD_OUT"]} for f in result["finalists"]]
    full = load_cell_bars_from_snapshot(ev.symbol, ev.timeframe, ho["dataset_hash16"])
    fbt = fast_backtest_factory(full)
    sham = sham_backtest_factory(full)

    def full_run(dna: DNA):
        from engine.research.search.evaluation_view import full_run as run_gate

        return run_gate(dna, full, fbt)

    def twin(dna: DNA, entries: int, salt: str) -> np.ndarray | None:
        res = sham(dna, max(int(entries), 1), salt=salt)
        return None if res.per_period_returns is None else np.asarray(res.per_period_returns)

    out = []
    for i, f in enumerate(result["finalists"], 1):
        dna = DNA(**{k: tuple(v) if isinstance(v, list) else v for k, v in f["dna"].items()})
        try:
            g = judge(dna, cfg=cfg, timeframe=ev.timeframe, split=int(ho["split_bar"]), full_run=full_run,
                      twin_returns=twin, evaluate_bred=ev.evaluate, neighbours=neighbours)
        except Exception as exc:  # recorded per finalist: the result says why it was not vaulted
            g = {"passed": False, "failed": ["ERROR"], "reason": f"{type(exc).__name__}: {exc}"}
        g.update(dna_hash=f["dna_hash"], split_time=ho.get("split_time"), held_out_dataset=ho["dataset_hash16"])
        out.append(g)
        log(f"VAULT GATES {i}/{len(result['finalists'])} · {f['dna']['signal']} {f['dna_hash'][:8]} · "
            + ("PASSED" if g["passed"] else "failed " + ",".join(g["failed"]))
            + (f" · SCORE {g['score']:.1f} {g['verdict']}" if "score" in g else ""))
    return out


def main(argv: list[str] | None = None) -> int:
    from engine.research.ga.parallel import workers_for
    from engine.research.ga.runner import GARunError, prepare_cell, with_power
    from engine.research.search.run_campaign_cli import SpreadBlockedError

    p = argparse.ArgumentParser(prog="engine.research.v1.cli")
    p.add_argument("--cell", required=True)
    p.add_argument("--preset", default="research")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--power", default="medium")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out)
    log = _logger(out / "v1.log")
    # SETTINGS (engine/core/app_settings.py): the owner's V1 numbers over the depth preset, and the OOS trades gate
    from engine.core import app_settings

    over = app_settings.v1_overrides()
    risk = over.pop("risk_pct", a.risk)
    over["held_out_min_trades"] = int(app_settings.load()["campaign"]["held_out_min_trades"])
    cfg = v1_config(a.preset, seed=a.seed, risk_pct=risk, **over)
    started = time.time()
    try:
        ev = prepare_cell(a.cell)
    except (GARunError, SpreadBlockedError) as exc:
        log(f"REFUSED · {exc}")
        return 2
    workers = workers_for(a.power)
    runner_ev = with_power(ev, workers)
    log(f"V1 · {ev.symbol} {ev.timeframe} · preset {cfg.preset} · proving grounds "
        f"{cfg.proving_grounds} · {cfg.generations} generations x 2 tribes · workers {workers}")
    try:
        run = V1Run(cfg, runner_ev, ev.timeframe, out_dir=out, log=log, skip_path=out / "skip")
        result = run.run(resume=a.resume)
        if hasattr(runner_ev, "close"):
            runner_ev.close()
        gates = gate_finalists(result, cell=a.cell, ev=ev, cfg=cfg, log=log)
    except Exception as exc:
        log(f"CRASH · {type(exc).__name__}: {exc}")
        (out / "v1_crash.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1
    vaulted = [g["dna_hash"] for g in gates if g.get("passed")]
    doc = {"campaign": {"cell": a.cell, "symbol": ev.symbol, "timeframe": ev.timeframe, "dataset_id": ev.dataset_id,
                        "seed": a.seed, "power": a.power, "workers": workers,
                        "spread_cost_source": ev.context.get("spread_cost_source"),
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                        "elapsed_s": round(time.time() - started, 1)},
           "run": result, "gates": gates, "vaulted": vaulted}
    tmp = out / "v1_result.json.tmp"
    tmp.write_text(json.dumps(doc, default=_json_default, indent=1), encoding="utf-8")
    tmp.replace(out / "v1_result.json")
    log(f"DONE · {result['tested']} strategies tested · {len(result['finalists'])} finalists · "
        f"VAULTED {len(vaulted)} · ELAPSED {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["gate_finalists", "held_out_window", "main"]
