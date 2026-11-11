"""python -m engine.research.reproof --cell <cell_id> --out <dir> --battery-log <path>

Re-prove one registered cell from the registry and disk only (DR-2026-10-08-u3-worker §3). Cell IDs come from the
loader (engine.research.registry.cells.validated_cells), never from prose. The tier comes from the entry's
campaign_status, never from a flag (DR-2026-10-09-u4-wizard §3). Exit 0 PROVEN@<sha> or PROVEN-PRECAMPAIGN@<sha>, 1
BLOCKED (proof written), 2 REFUSED (untrusted registry, unknown cell or a tier contradiction; an audit row, no proof).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engine.research.reproof.checks import ReproofRefused, reproof


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", required=True, help="a VALIDATED cell_id of configs/cells_registry.json")
    p.add_argument("--out", required=True, type=Path, help="where <cell_id>.proof.json and audit.jsonl are written")
    p.add_argument("--battery-log", required=True, type=Path, help="a closing parity-battery log (EXIT=0 run)")
    a = p.parse_args(argv)
    try:
        proof = reproof(a.cell, out_dir=a.out, battery_log=a.battery_log)
    except ReproofRefused as exc:
        print(f"REFUSED {a.cell}: {exc.reason}: {exc.detail}")
        return 2
    print(f"PROOF {proof['cell']} @ {proof['head'][:7]} (registry chain {proof['registry']['chain_hash'][:12]})")
    for name, c in proof["checks"].items():
        print(f"  {name:14s} {'PASS' if c['ok'] else 'FAIL ' + c['reason']}")
    for name, why in proof.get("not_applicable", {}).items():
        print(f"  {name:14s} NOT APPLICABLE ({proof['tier']}): {why}")
    print(f"VERDICT {proof['status']}  -> {a.out / (proof['cell'] + '.proof.json')}")
    return 0 if proof["verdict"] == "PROVEN" else 1


if __name__ == "__main__":
    sys.exit(main())
