"""G9 closure: the codegen's MQL5 ATR vs the Python engine's ATR on the identical USDCADm H1 bars.

An MQL5 script (``templates/atr_verifier.mq5.j2``, which includes the SAME ``atr_core.j2`` the EAs
are generated with) reads the terminal's own USDCADm H1 bars over a fixed seeded window -- 5,000
bars from index ``default_rng(20261009).integers(0, n - 5000)`` of the pinned snapshot -- and dumps,
per bar, OHLC, TR, the cumulative sum, the ported ATR(14) (empty where undefined) and MT5's native
``iATR(14)``, all as exact bit patterns. ``compare`` holds them against the oracle's
``true_range_series`` / ``sma_atr_series`` on the pinned bars of the same window:

* the bars must be bit-identical (else the comparison is void);
* TR and the ported ATR must be BITWISE equal wherever ``atr_at`` defines the value (pre-registered);
* native iATR is reported informationally from index 14 (its first TR uses the bar before the
  window), as max |delta| and the count above 1e-12.
"""
from __future__ import annotations

import csv
import struct
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from engine.research.features.atr import DEFAULT_ATR_PERIOD, sma_atr_series, true_range_series
from engine.research.mql5.cell_config import CellConfig
from engine.research.mql5.codegen import TEMPLATE_VERSION, _environment
from engine.research.mql5.metaeditor import compile_mq5, require_clean
from engine.research.mql5.terminal import arqvio_files_dir, mql5_dir
from engine.research.mql5.tester import run_terminal

ATR_SEED = 20261009
WINDOW = 5000
IATR_TOLERANCE = 1e-12


class AtrVerifierError(RuntimeError):
    """The verifier could not produce a comparable dump."""


def verifier_window(cell: CellConfig) -> tuple[int, int]:
    i0 = int(np.random.default_rng(ATR_SEED).integers(0, cell.n_bars - WINDOW))
    return i0, i0 + WINDOW


def _name(cell: CellConfig) -> str:
    return f"atr_verifier_{cell.name}"


def render_script(cell: CellConfig) -> str:
    i0, i1 = verifier_window(cell)
    return _environment().get_template("atr_verifier.mq5.j2").render(
        template_version=TEMPLATE_VERSION, name=_name(cell), symbol=cell.symbol, period=DEFAULT_ATR_PERIOD,
        from_time=int(cell.bars.time[i0]), to_time=int(cell.bars.time[i1 - 1]), n=WINDOW,
        out_file=f"ARQVIO_S6\\\\atr\\\\{cell.name}.csv")  # an MQL5 string literal: each backslash escaped


def run_verifier(cell: CellConfig, log_dir: Path) -> Path:
    scripts = mql5_dir() / "Scripts" / "ARQVIO_S6"
    scripts.mkdir(parents=True, exist_ok=True)
    src = scripts / f"{_name(cell)}.mq5"
    src.write_text(render_script(cell), encoding="utf-8", newline="\n")
    require_clean(compile_mq5(src, log_path=log_dir / f"{_name(cell)}.compile.log"))
    dump = arqvio_files_dir() / "atr" / f"{cell.name}.csv"
    dump.parent.mkdir(parents=True, exist_ok=True)
    if dump.exists():
        dump.unlink()
    ini = log_dir / f"{_name(cell)}.ini"
    ini.write_text(f"[StartUp]\r\nScript=ARQVIO_S6\\{_name(cell)}\r\nSymbol={cell.symbol}\r\nPeriod=H1\r\n"
                   "ShutdownTerminal=1\r\n", encoding="ascii")
    run_terminal(ini, timeout_s=600)
    if not dump.is_file():
        raise AtrVerifierError(f"the verifier script wrote no {dump}")
    return dump


def _f(bits: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits)))[0]


@dataclass
class AtrReport:
    window: tuple[int, int]
    first_time: str = ""
    last_time: str = ""
    bars_equal: bool = False
    tr_bitwise: int = 0
    atr_defined: int = 0
    atr_bitwise: int = 0
    iatr_compared: int = 0
    iatr_max_abs: float = 0.0
    iatr_over_tol: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.bars_equal and not self.mismatches and self.atr_bitwise == self.atr_defined > 0


def compare(cell: CellConfig, dump: Path) -> AtrReport:
    i0, i1 = verifier_window(cell)
    rep = AtrReport((i0, i1))
    with dump.open("r", encoding="ascii", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if len(rows) != WINDOW or any(len(r) != 10 for r in rows):
        raise AtrVerifierError(f"{dump}: {len(rows)} rows (want {WINDOW}) or a row of the wrong width")
    t = np.asarray(cell.bars.time[i0:i1], dtype=np.int64)
    o, h, lo, c = (np.asarray(x[i0:i1], dtype=np.float64) for x in (cell.bars.open, cell.bars.high,
                                                                     cell.bars.low, cell.bars.close))
    rep.first_time = datetime.fromtimestamp(int(t[0]), tz=UTC).isoformat()
    rep.last_time = datetime.fromtimestamp(int(t[-1]), tz=UTC).isoformat()
    got = [(int(r[0]), _f(r[1]), _f(r[2]), _f(r[3]), _f(r[4])) for r in rows]
    want = list(zip(t.tolist(), o.tolist(), h.tolist(), lo.tolist(), c.tolist()))
    bad_bars = [k for k, (g, w) in enumerate(zip(got, want)) if g != w]
    rep.bars_equal = not bad_bars
    if bad_bars:
        rep.mismatches.append(f"bars differ at window indices {bad_bars[:10]} (of {len(bad_bars)})")
        return rep
    tr = true_range_series(h, lo, c)
    atr = sma_atr_series(tr, period=DEFAULT_ATR_PERIOD)
    for k, r in enumerate(rows):
        if _f(r[5]) == tr[k] and struct.pack("<d", _f(r[5])) == struct.pack("<d", float(tr[k])):
            rep.tr_bitwise += 1
        else:
            rep.mismatches.append(f"TR[{k}]: MQL5 {_f(r[5])!r} Python {float(tr[k])!r}")
        defined = k >= DEFAULT_ATR_PERIOD - 1 and np.isfinite(atr[k]) and atr[k] > 0.0
        if defined != (r[7] != ""):
            rep.mismatches.append(f"ATR[{k}]: defined MQL5={r[7] != ''} Python={defined}")
            continue
        if defined:
            rep.atr_defined += 1
            if struct.pack("<d", _f(r[7])) == struct.pack("<d", float(atr[k])):
                rep.atr_bitwise += 1
            else:
                rep.mismatches.append(f"ATR[{k}]: MQL5 {_f(r[7])!r} Python {float(atr[k])!r}")
        if k >= DEFAULT_ATR_PERIOD and r[8] != "" and defined:
            d = abs(_f(r[8]) - float(atr[k]))
            rep.iatr_compared += 1
            rep.iatr_max_abs = max(rep.iatr_max_abs, d)
            rep.iatr_over_tol += d > IATR_TOLERANCE
    return rep


__all__ = ["ATR_SEED", "AtrReport", "AtrVerifierError", "IATR_TOLERANCE", "WINDOW", "compare", "render_script",
           "run_verifier", "verifier_window"]
