"""EA-side kill-switch cooperation: a generated EA, its kill file active mid-window, flattens its own magic
and stops trading (Strategy Tester run; probe-skipped loudly without a terminal)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from engine.research.mql5 import parity


def test_the_kill_file_flattens_own_magic_and_halts_the_ea(tmp_path: Path, pinned_home: Path) -> None:
    from engine.research.mql5.terminal import TerminalPathError, metaeditor_exe

    try:
        metaeditor_exe()
    except (FileNotFoundError, TerminalPathError) as exc:
        pytest.skip(f"LOUD SKIP: no MT5 terminal on this machine ({exc}); the S6 stage run is the evidence")
    from engine.research.mql5.cell_config import battery_cell
    from engine.research.mql5.codegen import generate
    from engine.research.mql5.harness import ensure_replica
    from engine.research.mql5.oracle_trace import run_oracle
    from engine.research.mql5.subjects import battery_dnas
    from engine.research.mql5.tester import run_ea

    cell = battery_cell()
    print(ensure_replica(cell, tmp_path))
    dna = battery_dnas("MACD_CROSS")[0]
    oracle = run_oracle(dna, cell)
    # The kill time: the middle of a trade the oracle holds, so the EA has a position to flatten.
    held = next(t for t in oracle.result.trades if t.exit_fill is not None
                and t.exit_fill.bar_time - t.entry_fill.bar_time >= 3 * 3600)
    kill_at = held.entry_fill.bar_time + 2 * 3600 + 59 * 60
    run = run_ea(generate(dna, cell, magic=610_000_012), cell, run_id=f"kill_{dna.dna_hash}", log_dir=tmp_path,
                 kill_after=kill_at)
    events = [r for r in csv.reader((run.run_dir / "events.csv").open(encoding="ascii")) if r]
    kills = [r for r in events if r[0] == "KILL"]
    print("\n".join(" ".join(r) for r in kills))
    assert any("close own ticket=" in r[3] and "retcode=10009" in r[3] for r in kills)
    assert any("halted at" in r[3] for r in kills)
    assert parity.read_diag(run)["halted"] == "1"
    deals = parity.parse_report_deals(run.report)
    after = [d for d in deals if parity._bar(d["Time"]) > kill_at]
    assert after == [] and all(d["Comment"] != "end of test" for d in deals)  # flat at the kill, nothing after
    assert deals and deals[-1]["Direction"] == "out"
