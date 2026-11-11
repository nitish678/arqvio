"""G9: the codegen's MQL5 ATR(14) is BITWISE the Python engine's ATR on identical USDCADm H1 bars.

The comparator is checked on dumps built from the pinned bars; the REAL run (the MQL5 script on the
terminal's own bars) is probe-skipped (loudly) when the terminal is absent -- in the S6 stage run it runs
for real and its log is the G9 evidence.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from engine.research.features.atr import sma_atr_series, true_range_series
from engine.research.mql5 import atr_verifier as av
from engine.research.mql5.cell_config import campaign_cell


def _u(x: float) -> str:
    return str(struct.unpack("<Q", struct.pack("<d", float(x)))[0])


def _dump(tmp_path: Path, cell, *, atr_shift: int = 0, bar_shift: bool = False) -> Path:
    i0, i1 = av.verifier_window(cell)
    t = np.asarray(cell.bars.time[i0:i1], dtype=np.int64)
    o, h, lo, c = (np.asarray(x[i0:i1], dtype=np.float64) for x in (cell.bars.open, cell.bars.high,
                                                                     cell.bars.low, cell.bars.close))
    if bar_shift:
        h = h.copy()
        h[7] = np.nextafter(h[7], np.inf)
    tr = true_range_series(h, lo, c)
    csum = np.cumsum(tr)
    atr = sma_atr_series(tr, 14)
    rows = []
    for k in range(t.size):
        a = "" if k < 13 or not atr[k] > 0 else _u(np.nextafter(atr[k], np.inf) if k == 500 and atr_shift else atr[k])
        rows.append(",".join([str(int(t[k])), _u(o[k]), _u(h[k]), _u(lo[k]), _u(c[k]), _u(tr[k]), _u(csum[k]), a,
                              _u(atr[k] if k >= 13 else 0.0), "x"]))
    p = tmp_path / "dump.csv"
    p.write_text("\n".join(rows) + "\n", encoding="ascii")
    return p


def test_the_window_is_the_pre_registered_seeded_one(pinned_home: Path) -> None:
    cell = campaign_cell("USDCADm", "aed3bf314539ec5a")
    i0, i1 = av.verifier_window(cell)
    assert i1 - i0 == 5000 and i0 == int(np.random.default_rng(20261009).integers(0, 59957 - 5000))


def test_a_faithful_dump_is_bitwise_equal(tmp_path: Path, pinned_home: Path) -> None:
    rep = av.compare(campaign_cell("USDCADm", "aed3bf314539ec5a"), _dump(tmp_path, campaign_cell("USDCADm", "aed3bf314539ec5a")))
    assert rep.ok and rep.bars_equal and rep.tr_bitwise == 5000 and rep.atr_bitwise == rep.atr_defined == 4987
    assert rep.iatr_compared == 4986 and rep.iatr_max_abs == 0.0


def test_one_ulp_of_atr_is_a_named_mismatch(tmp_path: Path, pinned_home: Path) -> None:
    cell = campaign_cell("USDCADm", "aed3bf314539ec5a")
    rep = av.compare(cell, _dump(tmp_path, cell, atr_shift=1))
    assert not rep.ok and rep.mismatches == [rep.mismatches[0]] and rep.mismatches[0].startswith("ATR[500]")


def test_different_bars_void_the_comparison(tmp_path: Path, pinned_home: Path) -> None:
    cell = campaign_cell("USDCADm", "aed3bf314539ec5a")
    rep = av.compare(cell, _dump(tmp_path, cell, bar_shift=True))
    assert not rep.ok and not rep.bars_equal and "bars differ at window indices [7]" in rep.mismatches[0]


def test_the_real_mql5_atr_is_bitwise_the_python_atr(tmp_path: Path, pinned_home: Path) -> None:
    from engine.research.mql5.terminal import TerminalPathError, metaeditor_exe

    try:
        metaeditor_exe()
    except (FileNotFoundError, TerminalPathError) as exc:
        pytest.skip(f"LOUD SKIP: no MT5 terminal on this machine ({exc}); the S6 stage run is the G9 evidence")
    cell = campaign_cell("USDCADm", "aed3bf314539ec5a")
    rep = av.compare(cell, av.run_verifier(cell, tmp_path))
    print(f"\nG9 ATR verifier {cell.symbol} H1 window {rep.window} {rep.first_time} .. {rep.last_time}: bars equal "
          f"{rep.bars_equal}; TR bitwise {rep.tr_bitwise}/5000; ported ATR bitwise {rep.atr_bitwise}/{rep.atr_defined}; "
          f"native iATR compared {rep.iatr_compared}, max |d| {rep.iatr_max_abs:.3e}, over 1e-12: {rep.iatr_over_tol}")
    if not rep.bars_equal:
        # The script reads the terminal's OWN USDCADm H1 history. The pinned bars came from the server of the deleted
        # C:\mt5\1 terminal; another server's feed differs (2026-10-26, Exness-MT5Trial6: 659 of 5000 bars), and an
        # ATR over different bars proves nothing either way. Same standing as "no terminal": the S6 run is the G9
        # evidence until the verifier reads a replica of the pinned bars (arqviov4.md, Phase 6).
        pytest.skip(f"LOUD SKIP: this terminal's {cell.symbol} H1 history is not the pinned bars "
                    f"({rep.mismatches[0] if rep.mismatches else 'bars differ'}); G9 cannot be judged here")
    assert rep.ok, rep.mismatches[:20]
