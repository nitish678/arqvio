"""Run a parity subject end to end: the replica (built once per cell), the EA, the tester, the oracle, the verdict."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.research.mql5 import replica
from engine.research.mql5.cell_config import CellConfig, battery_cell, campaign_cell
from engine.research.mql5.codegen import generate
from engine.research.mql5.oracle_trace import OracleRun, run_oracle
from engine.research.mql5.parity import Verdict, judge
from engine.research.mql5.records import record_ea, record_parity
from engine.research.mql5.subjects import Subject
from engine.research.mql5.tester import TesterError, TesterRun, run_ea, run_terminal

CELL_SOURCES = {"usdcadm_full": ("USDCADm", "aed3bf314539ec5a"), "xauusdm_full": ("XAUUSDm", "447b843f87438a9a")}


def load_cells() -> dict[str, CellConfig]:
    cells = {k: campaign_cell(*v) for k, v in CELL_SOURCES.items()}
    cells["battery"] = battery_cell()
    return cells


BUILD_TIMEOUT_S = 180.0  # a build takes ~10 s; S6 saw one intermittent hang of the builder script


def ensure_replica(cell: CellConfig, log_dir: Path) -> str:
    """Write the cell inputs, build the replica symbols (always; the result file is the proof), then purge the
    tester's synced copies of them so the next run re-syncs this build (replica.purge_tester_copies).

    A build is plumbing, not a parity result: a build over its cap is retried ONCE, and the retry is part
    of the returned evidence text. A second failure raises."""
    replica.write_cell_inputs(cell)
    replica.compile_builder(cell, log_dir)
    ini = replica.startup_ini(cell, log_dir / f"build_{cell.name}.ini")
    notes = []
    for attempt in (1, 2):
        marker = replica.cell_dir(cell) / "build_result.txt"
        if marker.exists():
            marker.unlink()
        try:
            rc, wall = run_terminal(ini, timeout_s=BUILD_TIMEOUT_S)
            break
        except TesterError as exc:
            if attempt == 2:
                raise
            notes.append(f"build attempt 1 failed ({exc}); retried once")
    text = replica.verify_build(cell)
    purged = replica.purge_tester_copies(cell)
    return "\n".join(notes + [f"replica {cell.name}: builder terminal exit={rc} wall={wall:.1f}s (attempt {attempt})", text,
                              f"purged {len(purged)} stale tester copies: " + ", ".join(p.name for p in purged)])


@dataclass(frozen=True)
class SubjectRun:
    subject: Subject
    verdict: Verdict
    oracle: OracleRun
    tester: TesterRun


def run_subject(s: Subject, cell: CellConfig, log_dir: Path, *, timeout_s: float = 300.0) -> SubjectRun:
    """One subject end to end; its EA and verdict are recorded for the read-only API (records.py)."""
    oracle = run_oracle(s.dna, cell)
    ea = generate(s.dna, cell, magic=s.magic)
    record_ea(ea)
    run = run_ea(ea, cell, run_id=f"{s.sid}_{s.dna.dna_hash}", log_dir=log_dir, timeout_s=timeout_s)
    verdict = judge(f"{s.sid} {s.dna.signal} {s.dna.dna_hash} on {cell.name} ({s.note})", oracle, run, cell)
    record_parity(s.sid, s.note, cell.name, run.run_id, verdict)
    return SubjectRun(s, verdict, oracle, run)


__all__ = ["CELL_SOURCES", "SubjectRun", "ensure_replica", "load_cells", "run_subject"]
