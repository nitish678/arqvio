"""PARITY TEST for one vault strategy (worker4 W4-7 7.8; idea: research APP-4 "parity runner in the user's own MT5
Strategy Tester, own test-MetaTrader copy, the trading terminal never closed or restarted; grade A-F").

The strategy's EA runs in the Strategy Tester of a dedicated TEST terminal on a replica of its own pinned bars, and the
oracle runs the same bars; mql5/parity.judge compares them level by level (the S6 harness, mql5/harness.py).
Grade = the share of levels that pass, on the research bands (PART 4.4: A >= 85, B 70-84, C 55-69, D 40-54, F < 40).

Safety, before anything runs:
* the test terminal must be named explicitly (``ARQVIO_MT5_TERMINAL``) -- never discovered, so the owner's trading
  terminal is never picked by accident;
* it must not be the terminal of any deployed bot (artifacts/deploy/bots.json): the tester restarts its terminal;
* only what the tester parity EA can express runs: an H1 pinned cell and a DNA without entry gates or switches
  (mql5/codegen._validated says why otherwise -- those subjects are Phase 6).
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PARITY_GRADES: tuple[tuple[float, str], ...] = ((85.0, "A"), (70.0, "B"), (55.0, "C"), (40.0, "D"))
MAGIC_OFFSET = 90_000  # inside the s6-parity-fixtures range, clear of the pre-registered subjects


class ParityRefused(RuntimeError):
    """A parity test that must not or cannot run; the message says why."""


def grade(passed: int, total: int) -> tuple[float, str]:
    score = 100.0 * passed / total if total else 0.0
    for bar, letter in PARITY_GRADES:
        if score >= bar:
            return round(score, 1), letter
    return round(score, 1), "F"


def test_terminal() -> Path:
    from engine.research.data.mt5_terminal import ENV_VAR
    from engine.research.deploy import full_deploy as fd

    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        raise ParityRefused(f"no test terminal named: set {ENV_VAR} to a MetaTrader 5 kept for testing "
                            "(the Strategy Tester restarts it; your trading terminal is never used)")
    exe = Path(raw)
    if not exe.is_file():
        raise ParityRefused(f"the test terminal {exe} does not exist")
    for b in fd.load_bots():
        if b.get("status") != "REMOVED" and fd.same_terminal(b.get("terminal"), str(exe)):
            raise ParityRefused(f"{exe} runs deployed bot {b['bot']}: a parity test restarts its terminal, so it "
                                "runs only on a terminal without bots")
    return exe


def preflight(entry: dict[str, Any]) -> None:
    """Everything that can be refused without touching a terminal."""
    from engine.research.mql5.codegen import CodegenError, _validated
    from engine.research.search.dna import DNA

    if entry.get("timeframe") != "H1":
        raise ParityRefused(f"the tester replica is built for H1 cells; this strategy is {entry.get('timeframe')}")
    dna = DNA(**{k: tuple(v) if isinstance(v, list) else v for k, v in entry["dna"].items()})
    try:
        _validated(dna)
    except CodegenError as exc:
        raise ParityRefused(f"the tester parity EA cannot express it yet (Phase 6): {exc}") from exc
    test_terminal()


def run(entry: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    from engine.research.mql5.cell_config import campaign_cell
    from engine.research.mql5.harness import ensure_replica, run_subject
    from engine.research.mql5.magic import load_table
    from engine.research.mql5.subjects import Subject
    from engine.research.search.dna import DNA

    preflight(entry)
    dna = DNA(**{k: tuple(v) if isinstance(v, list) else v for k, v in entry["dna"].items()})
    cell = campaign_cell(entry["symbol"], entry["dataset_id"])
    ensure_replica(cell, log_dir)
    base = load_table().range("s6-parity-fixtures").base
    magic = base + MAGIC_OFFSET + int(dna.dna_hash[:4], 16) % 9_000
    sr = run_subject(Subject(sid="vault", group="vault", dna=dna, cell_key=cell.name, magic=magic,
                             note=f"vault parity test {entry['id']}"), cell, log_dir)
    levels = [{"level": lv.name, "passed": bool(lv.ok), "summary": lv.summary} for lv in sr.verdict.levels]
    score, letter = grade(sum(lv["passed"] for lv in levels), len(levels))
    return {"state": "DONE", "utc": datetime.now(UTC).isoformat(timespec="seconds"), "grade": letter, "score": score,
            "levels": levels, "run_id": sr.tester.run_id, "cell": cell.name}


__all__ = ["PARITY_GRADES", "ParityRefused", "grade", "preflight", "run", "test_terminal"]
