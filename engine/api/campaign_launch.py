"""The app's GA launcher (build rulings 2026-10-10, the H-START LAW): the guard runs in the SAME process as the runner.

  python -m engine.api.campaign_launch --runner <module> --expect-campaign-id <id> --mode fresh|resume -- <runner argv>

The sidecar checks the target before spawning, but the runner re-imports the code on disk and re-reads the gene pool
in its own process, so the directory it would clear (engine/research/ga/runner.py:735-740) is re-derived here, in the
child, immediately before ``<module>.main(<runner argv>)`` runs with the same imports: an id that differs from the
pre-registered one, a fresh start into an existing directory, or a resume into a missing one exits 2 with a REFUSED
line and nothing is run. The runner itself is not modified (engine/research/ga/ is frozen).
"""
from __future__ import annotations

import argparse
import importlib
import sys


def main(argv: list[str]) -> int:
    if "--" not in argv:
        print("REFUSED: launcher argv has no '--' separating the runner argv")
        return 2
    cut = argv.index("--")
    ap = argparse.ArgumentParser(prog="campaign_launch")
    ap.add_argument("--runner", required=True)
    ap.add_argument("--expect-campaign-id", required=True)
    ap.add_argument("--mode", required=True, choices=("fresh", "resume"))
    own = ap.parse_args(argv[:cut])
    runner_argv = argv[cut + 1:]
    rp = argparse.ArgumentParser(add_help=False)
    for flag in ("--pop", "--gen", "--seed", "--budget", "--final-n"):
        rp.add_argument(flag, type=int, required=True)
    rp.add_argument("--cell", required=True)
    rp.add_argument("--islands", type=int, default=None)  # hashed into the config, as runner.main's IslandConfig
    r, _rest = rp.parse_known_args(runner_argv)
    runner = importlib.import_module(own.runner)
    from engine.api.routes_app_campaigns import H_START, campaign_target
    from engine.api.routes_app import display

    t = campaign_target(r.cell, {"pop": r.pop, "gen": r.gen, "budget": r.budget, "final_n": r.final_n}, r.seed,
                        r.islands)
    if t["campaign_id"] != own.expect_campaign_id:
        print(f"REFUSED: campaign id drift: pre-registered {own.expect_campaign_id}, computed now {t['campaign_id']}")
        return 2
    if own.mode == "fresh" and t["dir"].exists():
        print(f"REFUSED: {H_START.format(target=display(t['dir']))}")
        return 2
    if own.mode == "resume" and not t["dir"].is_dir():
        print(f"REFUSED: {display(t['dir'])} is missing; nothing of this run to continue")
        return 2
    print(f"LAUNCH {own.mode} {t['campaign_id']} via {own.runner}", flush=True)
    return int(runner.main(runner_argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
