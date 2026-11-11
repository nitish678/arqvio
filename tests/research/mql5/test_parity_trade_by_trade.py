"""THE roadmap test: N-of-N trade match + PnL 8 decimals on the pre-registered S6 subject set.

Every registered subject (engine/research/mql5/subjects.py: the 20 GA finalists on both pinned cells,
the battery subset, the MA_CROSS seeds, the edge rules) runs end to end -- oracle, generated EA, the
Strategy Tester on the cell's replica, the verdict at levels A, H, B, C, D, E and DIAG -- and must pass every
level. Probe-skipped (loudly) on a machine without the MT5 terminal; in the S6 stage run it runs for real
and its -s output is the evidence (artifacts/reports/s6).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from engine.research.mql5.subjects import SUBJECT_IDS

PINNED = (("USDCADm", "aed3bf314539ec5a"), ("XAUUSDm", "447b843f87438a9a"))


def _terminal_or_skip() -> None:
    from engine.research.mql5.terminal import TerminalPathError, data_dir, metaeditor_exe

    try:
        metaeditor_exe()
        data_dir()
    except (FileNotFoundError, TerminalPathError) as exc:
        pytest.skip(f"LOUD SKIP: no MT5 terminal on this machine ({exc}); the S6 stage run is the evidence")


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory, real_artifacts_dir: Path):
    _terminal_or_skip()
    from engine.config import settings as settings_module
    from engine.research.mql5.ga_subjects import GA_CAMPAIGN
    from engine.research.mql5.harness import ensure_replica, load_cells
    from engine.research.mql5.subjects import subject_set

    need = [real_artifacts_dir / "datasets" / f"{s}_H1_{d}.{ext}" for s, d in PINNED for ext in ("json", "npz")]
    need += [real_artifacts_dir / sub / f"{s}.json" for sub in ("spread", "specs") for s, _ in PINNED]
    need += [real_artifacts_dir / "ga" / GA_CAMPAIGN / "manifest_v2.json"]
    missing = [p for p in need if not p.is_file()]
    if missing:
        pytest.skip(f"LOUD SKIP: pinned inputs not on this machine: {missing[0]}")
    root = tmp_path_factory.mktemp("s6_parity")
    home = root / "ARQVIO"
    for p in need:
        target = home / "artifacts" / p.relative_to(real_artifacts_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ARQVIO_HOME", str(home))
        settings_module.get_settings.cache_clear()
        cells = load_cells()
        log_dir = root / "logs"
        for cell in cells.values():
            print(ensure_replica(cell, log_dir))
        subjects = {s.sid: s for s in subject_set(cells["battery"])}
        yield {"cells": cells, "subjects": subjects, "log_dir": log_dir, "results": {}}
    settings_module.get_settings.cache_clear()


@pytest.mark.parametrize("sid", SUBJECT_IDS)
def test_parity_trade_by_trade(harness, sid: str) -> None:
    from engine.research.mql5.harness import run_subject

    s = harness["subjects"][sid]
    r = run_subject(s, harness["cells"][s.cell_key], harness["log_dir"])
    harness["results"][sid] = r
    text = r.verdict.render() + (f"\n  trades {r.oracle.result.n_trades}, intents {len(r.oracle.intents)}, "
                                 f"tester wall {r.tester.wall_s:.1f}s, magic {s.magic}")
    print(text)
    assert r.verdict.ok, text


def test_the_whole_registered_set_ran(harness) -> None:
    results = harness["results"]
    rows = []
    for sid in SUBJECT_IDS:
        r = results.get(sid)
        if r is None:
            rows.append(f"{sid:20s} NOT RUN")
            continue
        lv = {x.name: x for x in r.verdict.levels}
        rows.append(f"{sid:20s} {'PASS' if r.verdict.ok else 'FAIL'} trades {r.oracle.result.n_trades:4d} "
                    f"A {'ok' if lv['A'].ok else 'FAIL'} B {lv['B'].exact}/{lv['B'].compared} bars "
                    f"D {lv['D'].exact}/{lv['D'].compared} E {lv['E'].exact}/{lv['E'].compared} bit-exact "
                    f"inexact {len(lv['E'].inexact)} DIAG {lv['DIAG'].summary.split()[1]}")
    passed = sum(1 for r in results.values() if r.verdict.ok)
    trades = sum(r.oracle.result.n_trades for r in results.values())
    print("\nPARITY SUMMARY\n" + "\n".join(rows) +
          f"\nsubjects {len(results)}/{len(SUBJECT_IDS)} run, {passed} passed every level; trades matched N-of-N {trades}")
    assert len(results) == len(SUBJECT_IDS) and passed == len(SUBJECT_IDS)
