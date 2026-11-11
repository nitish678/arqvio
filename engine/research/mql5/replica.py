"""The S6 tester replica of a pinned cell (the DR's cost-parity protocol, tester side).

``write_cell_inputs(cell)`` writes, under Common\\Files\\ARQVIO_S6\\cells\\<cell>:

* ``m1_L.csv`` / ``m1_H.csv`` -- the replica histories: per pinned H1 bar, flat
  M1 bars at :00 open, :01 open (the action tick), :02 the ADVERSE extreme
  (low on _L, high on _H), :03 open (the re-visit that resolves a gapped TP),
  :04 the other extreme, :05 close, :59 close (the control tick); before the
  window, 14 days of flat hourly bars at bar 0's open (the tester wants history
  before FromDate; the EA ignores them). The M1 spread field is 1 (one replica
  point, the tester minimum);
* ``bars.csv`` -- every pinned bar's time, open / high / low / close as exact
  IEEE bit patterns, and spread: what the EA computes from (the raw doubles),
  the max-spread gate input and the bar-time check;
* ``draws.csv`` -- the exogenous slippage stream as exact bit patterns;
* ``prims.csv`` -- the level-B primitive inputs (indicators.primitive_inputs): kind
  (p = stop price, v = volume) and exact bit pattern.

ORDER-PRESERVING ENCODING. The tester normalises tick prices to the replica's
digits, but 12% of the pinned USDCADm doubles sit one ulp off the 5-decimal
grid (e.g. 1.3189199999999999 next to the level 1.31892): the oracle compares
raw doubles, so ``raw >= level`` may be False where ``normalised >= level`` is
True (S6 parity run 2, GA finalist 1). A replica price is therefore the grid
value pushed ``ENC_OFFSET_RP`` replica points toward the raw double:
``q = round(p, d) + 2 * sign(p - round(p, d)) * 10^-(d+3)``. Every relation of a
price to a 5-decimal level -- including through the tester's +1-point ask -- is
then the oracle's, and a fill still rounds to the oracle's quote. The EA
computes from ``bars.csv`` (raw); the chart drives execution only.

``history_digest(cell)`` (the pinned raw bars) and ``replica_digest(cell)``
(the encoded chart bars) are the SHA-256s over (time, open, high, low, close)
little-endian bytes that the EA's history guard must reproduce.
"""
from __future__ import annotations

import hashlib
import shutil
import struct
from decimal import Decimal
from pathlib import Path

import numpy as np

from engine.research.mql5.cell_config import CellConfig
from engine.research.mql5.codegen import TEMPLATE_VERSION, _environment
from engine.research.mql5.indicators import primitive_inputs
from engine.research.mql5.metaeditor import compile_mq5, require_clean
from engine.research.mql5.oracle_trace import slippage_stream
from engine.research.mql5.terminal import arqvio_files_dir, data_dir, mql5_dir, tester_data_dir

PREFIX_DAYS = 14
REPLICA_PREFIX = "AS6_"
REPLICA_EXTRA_DIGITS = 3
DRAWS_PER_CELL = 200_000
ENC_OFFSET_RP = 2  # >= 2: a short's stop compares ASK = bid + 1 replica point


def encode(p: float, digits: int) -> str:
    """The replica price of raw double ``p``, as an exact decimal with digits + 3 places."""
    grid = f"{p:.{digits}f}"  # correctly rounded from the exact binary value
    r = float(grid)
    q = Decimal(grid)
    step = Decimal(1).scaleb(-(digits + REPLICA_EXTRA_DIGITS))
    if p > r:
        q += ENC_OFFSET_RP * step
    elif p < r:
        q -= ENC_OFFSET_RP * step
    return f"{q:.{digits + REPLICA_EXTRA_DIGITS}f}"


class ReplicaError(RuntimeError):
    """A replica that cannot be built or verified."""


def cell_dir(cell: CellConfig) -> Path:
    d = arqvio_files_dir() / "cells" / cell.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ohlc(cell: CellConfig) -> tuple[np.ndarray, ...]:
    b = cell.bars
    return (np.asarray(b.time, dtype=np.int64),) + tuple(np.asarray(x, dtype=np.float64)
                                                          for x in (b.open, b.high, b.low, b.close))


def m1_rows(cell: CellConfig, *, low_first: bool) -> list[str]:
    t, o, h, lo, c = _ohlc(cell)
    d = int(cell.spec.digits)
    first = int(t[0])
    rows = [f"{x},{encode(float(o[0]), d)},1" for x in range(first - PREFIX_DAYS * 86400, first, 3600)]
    for k in range(t.size):
        x1, x2 = (lo[k], h[k]) if low_first else (h[k], lo[k])
        base = int(t[k])
        for minute, p in ((0, o[k]), (1, o[k]), (2, x1), (3, o[k]), (4, x2), (5, c[k]), (59, c[k])):
            rows.append(f"{base + 60 * minute},{encode(float(p), d)},1")
    return rows


def _digest(rows: zip) -> str:
    h = hashlib.sha256()
    for t, o, hi, lo, c in rows:
        h.update(struct.pack("<q4d", int(t), o, hi, lo, c))
    return h.hexdigest()


def history_digest(cell: CellConfig) -> str:
    """The pinned raw bars: what the EA reads from bars.csv and computes with."""
    t, o, h, lo, c = _ohlc(cell)
    return _digest(zip(t, map(float, o), map(float, h), map(float, lo), map(float, c)))


def replica_digest(cell: CellConfig) -> str:
    """The encoded bars the tester serves on the replica chart."""
    t, o, h, lo, c = _ohlc(cell)
    d = int(cell.spec.digits)
    enc = [[float(encode(float(p), d)) for p in col] for col in (o, h, lo, c)]
    return _digest(zip(t, *enc))


def _u64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def write_cell_inputs(cell: CellConfig) -> Path:
    d = cell_dir(cell)
    (d / "m1_L.csv").write_text("\n".join(m1_rows(cell, low_first=True)) + "\n", encoding="ascii", newline="\n")
    (d / "m1_H.csv").write_text("\n".join(m1_rows(cell, low_first=False)) + "\n", encoding="ascii", newline="\n")
    t, o, h, lo, c = _ohlc(cell)
    sp = np.asarray(cell.bars.spread, dtype=np.int64)
    (d / "bars.csv").write_text(
        "\n".join(f"{int(t[k])},{_u64(o[k])},{_u64(h[k])},{_u64(lo[k])},{_u64(c[k])},{int(sp[k])}"
                  for k in range(t.size)) + "\n", encoding="ascii", newline="\n")
    (d / "prims.csv").write_text("\n".join(f"{kind},{_u64(x)}" for kind, x in primitive_inputs(cell)) + "\n",
                                 encoding="ascii", newline="\n")
    draws = slippage_stream(cell, DRAWS_PER_CELL)
    (d / "draws.csv").write_text("\n".join(str(struct.unpack("<Q", struct.pack("<d", v))[0]) for v in draws) + "\n",
                                 encoding="ascii", newline="\n")
    return d


def render_builder(cell: CellConfig) -> tuple[str, str]:
    sym_l, sym_h = cell.replica_names
    spec = cell.spec
    src = _environment().get_template("replica_builder.mq5.j2").render(
        template_version=TEMPLATE_VERSION, cell_name=cell.name, sym_l=sym_l, sym_h=sym_h,
        digits=int(spec.digits) + REPLICA_EXTRA_DIGITS, vmin=repr(float(spec.volume_min)),
        vstep=repr(float(spec.volume_step)), vmax=repr(float(spec.volume_max)))
    return f"replica_builder_{cell.name}", src


def compile_builder(cell: CellConfig, log_dir: Path) -> Path:
    name, src = render_builder(cell)
    scripts = mql5_dir() / "Scripts" / "ARQVIO_S6"
    scripts.mkdir(parents=True, exist_ok=True)
    path = scripts / f"{name}.mq5"
    path.write_text(src, encoding="utf-8", newline="\n")
    require_clean(compile_mq5(path, log_path=log_dir / f"{name}.compile.log"))
    return path


def purge_tester_copies(cell: CellConfig) -> list[Path]:
    """Remove the Strategy Tester's synced copies of this cell's replica symbols (history + tick cache).

    The tester serves its own copy of a custom symbol's history and does not re-sync it after the symbol is
    rebuilt under the same name (S6 DEV-12: the EA's chart digest was the previous build's). Only names with
    the replica prefix are ever touched."""
    names = cell.replica_names
    if not all(n.startswith(REPLICA_PREFIX) for n in names):
        raise ReplicaError(f"refusing to purge non-replica symbols {names}")
    removed: list[Path] = []
    bases = tester_data_dir() / "bases"
    for name in names:
        for d in sorted(bases.glob(f"*/history/{name}")) if bases.is_dir() else ():
            shutil.rmtree(d)
            removed.append(d)
    cache = data_dir() / "Tester" / "cache"
    for f in sorted(cache.glob("*.tst")) if cache.is_dir() else ():
        if any(f".{n}." in f.name for n in names):
            f.unlink()
            removed.append(f)
    return removed


def startup_ini(cell: CellConfig, path: Path) -> Path:
    name, _ = render_builder(cell)
    path.write_text(f"[StartUp]\r\nScript=ARQVIO_S6\\{name}\r\nSymbol={cell.symbol}\r\nPeriod=H1\r\nShutdownTerminal=1\r\n",
                    encoding="ascii")
    return path


def verify_build(cell: CellConfig) -> str:
    p = cell_dir(cell) / "build_result.txt"
    if not p.is_file():
        raise ReplicaError(f"the replica builder wrote no {p}")
    text = p.read_text(encoding="ascii", errors="replace")
    sym_l, sym_h = cell.replica_names
    n_rows = PREFIX_DAYS * 24 + 7 * cell.n_bars
    want = [f"OK {sym_l} m1={n_rows} ", f"OK {sym_h} m1={n_rows} ", "BUILD OK"]
    if not all(w in text for w in want):
        raise ReplicaError(f"replica build for {cell.name} not verified:\n{text}")
    return text


__all__ = [
    "DRAWS_PER_CELL",
    "ENC_OFFSET_RP",
    "PREFIX_DAYS",
    "REPLICA_PREFIX",
    "ReplicaError",
    "cell_dir",
    "compile_builder",
    "encode",
    "history_digest",
    "m1_rows",
    "purge_tester_copies",
    "render_builder",
    "replica_digest",
    "startup_ini",
    "verify_build",
    "write_cell_inputs",
]
