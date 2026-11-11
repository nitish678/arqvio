"""Parity comparators: the oracle run vs the EA's tester run, level by level, mismatches verbatim.

* A (compile): the EA's MetaEditor result, 0 errors / 0 warnings and a fresh .ex5.
* B (indicators / signals on the pinned bars): every bar's indicator row -- ATR, the family's values as bit
  patterns, its state-free conditions -- bitwise against indicators.expected_rows, the strategies' prefix check,
  and the stop-quantization / volume-normalise primitive vectors bitwise.
* H (history guard): the EA's SHA-256 over the bars it computed with == the
  pinned bars' digest, and over the chart bars the tester served == the
  replica's encoded digest; bar count, first / last time == the cell; the run
  complete; no HISTORY_MISMATCH / FATAL / EXEC_DIVERGENCE event. A failure here
  is ``S6 BLOCKED: tester history mismatch`` material, never a shrink.
* C (intents): every strategy intent -- bar, time, side, volume, sl / tp
  distance (bitwise), reason -- plus the engine's four gate counters.
* D (trade list, N-of-N): the TESTER REPORT's deals (cross-checked against the
  EA's own deal dump) paired in / out, vs the oracle's trades: count, then per
  row entry / exit bar time, side, volume, entry / exit price at the cell's quote
  precision (the replica artefact is <= 1 replica point, DR protocol).
* E (PnL to 8 decimals): every ledger component of every trade, the run totals,
  the final equity and EVERY equity point: |EA - oracle| < 1e-8, exact-bit
  agreement counted alongside.
* DIAG: the EA's early-gate label == the funnel rule on the oracle's equity.

Each mismatch line carries its row number; a level never fails summary-only.
"""
from __future__ import annotations

import csv
import html
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine.research.mql5.cell_config import CellConfig
from engine.research.mql5.codegen import early_gates
from engine.research.mql5.indicators import (FIELDS, expected_rows, hx, prefix_mismatches, primitive_expected,
                                             primitive_inputs)
from engine.research.mql5.oracle_trace import OracleRun
from engine.research.mql5.replica import history_digest, replica_digest
from engine.research.mql5.tester import TesterRun

TOL_E = 1e-8
COUNTERS = ("blocked_zero_spread_events", "deferred_zero_spread_fills", "sltp_deferred_zero_spread",
            "blocked_max_spread")
REPORT_HEADER = ["Time", "Deal", "Symbol", "Type", "Direction", "Volume", "Price", "Order", "Commission", "Swap",
                 "Profit", "Balance", "Comment"]
PROTECTIVE = ("SL", "TP", "SLTP_DEFERRED_ZERO_SPREAD", "FRIDAY_CLOSE")


class ParityParseError(RuntimeError):
    """An EA journal or tester report the comparator cannot read unambiguously."""


@dataclass
class Level:
    name: str
    lines: list[str] = field(default_factory=list)
    summary: str = ""
    exact: int = 0
    compared: int = 0
    inexact: list[str] = field(default_factory=list)  # within tolerance but not bit-exact: named, never hidden

    @property
    def ok(self) -> bool:
        return not self.lines

    def bad(self, msg: str) -> None:
        self.lines.append(msg)


def f64(bits: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits, 16)))[0]


def _rows(path: Path, width: int | None = None) -> list[list[str]]:
    with path.open("r", encoding="ascii", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if width is not None:
        for k, r in enumerate(rows):
            if len(r) != width:
                raise ParityParseError(f"{path.name} row {k}: {len(r)} fields, expected {width}: {r}")
    return rows


def read_diag(run: TesterRun) -> dict[str, str]:
    return {r[0]: r[1] for r in _rows(run.run_dir / "diag.csv", 2)}


def parse_report_deals(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S)
    cells = [[html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
             for r in rows]
    starts = [k for k, c in enumerate(cells) if c == REPORT_HEADER]
    if len(starts) != 1:
        raise ParityParseError(f"{path.name}: {len(starts)} deals header rows")
    out = []
    for c in cells[starts[0] + 1:]:
        if len(c) != len(REPORT_HEADER):
            break
        out.append(dict(zip(REPORT_HEADER, c)))
    if not out or out[0]["Type"] != "balance":
        raise ParityParseError(f"{path.name}: the deals table does not start with the balance deal")
    for d in out[1:]:
        if d["Type"] not in ("buy", "sell") or d["Direction"] not in ("in", "out"):
            raise ParityParseError(f"{path.name}: unexpected deal {d}")
    return out[1:]


def _bar(t: str) -> int:
    from datetime import UTC, datetime

    s = int(datetime.strptime(t, "%Y.%m.%d %H:%M:%S").replace(tzinfo=UTC).timestamp())
    return s - s % 3600


def tester_trades(run: TesterRun, cell: CellConfig) -> tuple[list[dict], Level]:
    """The report's deals paired in -> out (one ARQVIO position at a time), cross-checked with the EA dump."""
    lv = Level("D-report")
    deals = parse_report_deals(run.report)
    ea = _rows(run.run_dir / "deals.csv", 10)
    ea_trades = [r for r in ea if r[2] in cell.replica_names]
    if len(ea_trades) != len(deals):
        lv.bad(f"report lists {len(deals)} deals, the EA dump {len(ea_trades)}")
    for k, (d, e) in enumerate(zip(deals, ea_trades)):
        if round(float(d["Price"]), cell.spec.digits + 3) != round(f64(e[5]), cell.spec.digits + 3) or d["Symbol"] != e[2]:
            lv.bad(f"deal {k}: report {d['Symbol']} {d['Price']} vs EA dump {e[2]} {f64(e[5])!r}")
    trades: list[dict] = []
    open_t: dict | None = None
    for k, d in enumerate(deals):
        side = "long" if d["Type"] == "buy" else "short"
        if d["Direction"] == "in":
            if open_t is not None:
                lv.bad(f"deal {k}: an entry while position from deal {open_t['deal']} is open")
            open_t = {"deal": k, "side": side, "volume": float(d["Volume"]), "entry_time": _bar(d["Time"]),
                      "entry_price": round(float(d["Price"]), cell.spec.digits), "symbol": d["Symbol"]}
        else:
            if open_t is None:
                lv.bad(f"deal {k}: an exit with no open position")
                continue
            eod = d["Comment"] == "end of test"
            open_t.update(exit_time=None if eod else _bar(d["Time"]),
                          exit_price=None if eod else round(float(d["Price"]), cell.spec.digits),
                          exit_comment=d["Comment"])
            trades.append(open_t)
            open_t = None
    if open_t is not None:
        lv.bad(f"deal {open_t['deal']}: an entry never closed in the report")
    for k, t in enumerate(trades):
        want = cell.replica_names[0] if t["side"] == "long" else cell.replica_names[1]
        if t["symbol"] != want:
            lv.bad(f"trade {k}: {t['side']} on {t['symbol']}, the protocol routes it to {want}")
    lv.summary = f"{len(deals)} report deals -> {len(trades)} trades"
    return trades, lv


def oracle_trades(o: OracleRun) -> list[dict]:
    out = []
    for t in o.result.trades:
        row = {"side": t.direction, "volume": float(t.volume), "entry_time": int(t.entry_fill.bar_time),
               "entry_price": float(t.entry_fill.requested_price), "reason": t.exit_reason}
        if t.exit_fill is None:
            row.update(exit_time=None, exit_price=None)
        else:
            px = t.exit_fill.filled_price if t.exit_reason in PROTECTIVE else t.exit_fill.requested_price
            row.update(exit_time=int(t.exit_fill.bar_time), exit_price=float(px))
        out.append(row)
    return out


def compare_d(o: OracleRun, run: TesterRun, cell: CellConfig) -> tuple[Level, Level]:
    tt, rep = tester_trades(run, cell)
    ot = oracle_trades(o)
    lv = Level("D")
    lv.compared = len(ot)
    if len(tt) != len(ot):
        lv.bad(f"COUNT: tester {len(tt)} trades, oracle {len(ot)}")
    d = cell.spec.digits
    for k, (a, b) in enumerate(zip(tt, ot)):
        diffs = []
        for key in ("side", "entry_time", "exit_time"):
            if a[key] != b[key]:
                diffs.append(f"{key} tester={a[key]} oracle={b[key]}")
        if round(a["volume"], 2) != round(b["volume"], 2):
            diffs.append(f"volume tester={a['volume']} oracle={b['volume']}")
        if a["entry_price"] != round(b["entry_price"], d):
            diffs.append(f"entry_price tester={a['entry_price']} oracle={b['entry_price']!r}")
        if (a["exit_price"] is None) != (b["exit_price"] is None) or (
                a["exit_price"] is not None and a["exit_price"] != round(b["exit_price"], d)):
            diffs.append(f"exit_price tester={a['exit_price']} oracle={b['exit_price']!r} ({b['reason']})")
        if diffs:
            lv.bad(f"row {k}: " + "; ".join(diffs))
        else:
            lv.exact += 1
    lv.summary = f"{len(tt)} tester / {len(ot)} oracle trades, {lv.exact} rows equal"
    return lv, rep


def compare_c(o: OracleRun, run: TesterRun, diag: dict[str, str]) -> Level:
    lv = Level("C")
    ea = _rows(run.run_dir / "intents.csv", 8)
    lv.compared = len(o.intents)
    if len(ea) != len(o.intents):
        lv.bad(f"COUNT: EA {len(ea)} intents, oracle {len(o.intents)}")
    for k, (e, x) in enumerate(zip(ea, o.intents)):
        sl = None if e[5] == "" else f64(e[5])
        tp = None if e[6] == "" else f64(e[6])
        got = (int(e[1]), int(e[2]), int(e[3]), f64(e[4]), sl, tp, e[7])
        want = (x.bar_index, x.bar_time, x.direction, x.volume, x.sl_dist, x.tp_dist, x.reason)
        if got != want:
            lv.bad(f"intent {k}: EA {got} oracle {want}")
        else:
            lv.exact += 1
    for name in COUNTERS:
        if int(diag[name]) != int(getattr(o.result, name)):
            lv.bad(f"counter {name}: EA {diag[name]} oracle {getattr(o.result, name)}")
    lv.summary = f"{len(ea)} EA / {len(o.intents)} oracle intents, {lv.exact} equal; counters " + \
        ", ".join(f"{n}={diag[n]}" for n in COUNTERS)
    return lv


def compare_a(run: TesterRun) -> Level:
    lv = Level("A")
    c = run.compiled
    if c.errors or c.warnings or not c.ex5.is_file():
        lv.bad(f"{c.source.name}: {c.errors} errors, {c.warnings} warnings, ex5 present {c.ex5.is_file()}")
    lv.compared, lv.exact = 1, int(lv.ok)
    lv.summary = f"MetaEditor {c.errors} errors, {c.warnings} warnings, {c.elapsed_ms} ms ({c.log_path.name})"
    return lv


def compare_b(o: OracleRun, run: TesterRun, cell: CellConfig) -> Level:
    """Every bar's indicator row bitwise, the prefix check, and the primitive vectors bitwise."""
    lv = Level("B")
    width = 3 + FIELDS[o.dna.signal]
    ea = _rows(run.run_dir / "indicators.csv", width) if (run.run_dir / "indicators.csv").stat().st_size else []
    want = expected_rows(o.dna, cell)
    if len(ea) != len(want):
        lv.bad(f"COUNT: EA {len(ea)} indicator rows, oracle {len(want)}")
    defined = 0
    for e, w in zip(ea, want):
        lv.compared += 1
        if e == w:
            lv.exact += 1
            defined += sum(1 for x in w[2:] if x != "")
        else:
            cols = [k for k, (a, b) in enumerate(zip(e, w)) if a != b]
            lv.bad(f"bar {w[0]} ({w[1]}): columns {cols}: EA {e} oracle {w}")
    n_prefix, bad = prefix_mismatches(o.dna, cell)
    for line in bad:
        lv.bad(line)
    prims = _rows(run.run_dir / "prims_out.csv", 3)
    inputs = primitive_inputs(cell)
    if len(prims) != len(inputs):
        lv.bad(f"COUNT: EA {len(prims)} primitive rows, inputs {len(inputs)}")
    p_exact = 0
    for k, (r, (kind, x)) in enumerate(zip(prims, inputs)):
        exp = primitive_expected(cell, kind, x)
        if r[0] != kind or int(r[1]) != struct.unpack("<Q", struct.pack("<d", x))[0] or r[2] != hx(exp):
            lv.bad(f"primitive {k} {kind} {x!r}: EA {r} engine {hx(exp)} ({exp!r})")
        else:
            p_exact += 1
    lv.summary = (f"{lv.exact}/{len(want)} bars bitwise ({defined} defined values / conditions); prefix check "
                  f"{n_prefix} prefixes, {len(bad)} mismatches; primitives {p_exact}/{len(inputs)} bitwise")
    return lv


def _close(a: float, b: float, lv: Level, where: str) -> None:
    lv.compared += 1
    if struct.pack("<d", a) == struct.pack("<d", b):
        lv.exact += 1
    elif not abs(a - b) < TOL_E:
        lv.bad(f"{where}: EA {a!r} oracle {b!r} |d|={abs(a - b):.3e}")
    else:
        lv.inexact.append(f"{where}: EA {a!r} oracle {b!r} |d|={abs(a - b):.3e}")


def compare_e(o: OracleRun, run: TesterRun, diag: dict[str, str]) -> Level:
    lv = Level("E")
    rows = _rows(run.run_dir / "trades.csv", 24)
    trades = o.result.trades
    if len(rows) != len(trades):
        lv.bad(f"COUNT: EA ledger {len(rows)} trades, oracle {len(trades)}")
    for k, (r, t) in enumerate(zip(rows, trades)):
        w = f"trade {k}"
        ints = (int(r[1]), int(r[3]), int(r[4]), int(r[5]))
        want = (1 if t.direction == "long" else -1, t.entry_bar_index, t.entry_fill.bar_index, t.entry_fill.bar_time)
        if ints != want:
            lv.bad(f"{w} side/bars/time: EA {ints} oracle {want}")
        for idx, val, name in ((2, t.volume, "volume"), (6, t.entry_fill.requested_price, "entry_requested"),
                               (7, t.entry_fill.filled_price, "entry_filled"), (8, t.entry_fill.spread_cost, "spread"),
                               (9, t.entry_fill.slippage_cost, "slippage"), (10, t.entry_fill.commission_cost, "commission"),
                               (11, t.swap_charged, "swap")):
            _close(f64(r[idx]), float(val), lv, f"{w} {name}")
        for idx, val, name in ((12, t.sl_price, "sl"), (13, t.tp_price, "tp")):
            if (r[idx] == "") != (val is None):
                lv.bad(f"{w} {name}: EA {r[idx]!r} oracle {val!r}")
            elif val is not None:
                _close(f64(r[idx]), float(val), lv, f"{w} {name}")
        if r[23] != t.exit_reason:
            lv.bad(f"{w} reason: EA {r[23]} oracle {t.exit_reason}")
        if t.exit_fill is None:
            continue
        if (int(r[14]), int(r[15])) != (t.exit_bar_index, t.exit_fill.bar_time):
            lv.bad(f"{w} exit bar/time: EA {(r[14], r[15])} oracle {(t.exit_bar_index, t.exit_fill.bar_time)}")
        for idx, val, name in ((16, t.exit_fill.requested_price, "exit_requested"), (17, t.exit_fill.filled_price, "exit_filled"),
                               (18, t.exit_fill.spread_cost, "exit_spread"), (19, t.exit_fill.slippage_cost, "exit_slippage"),
                               (20, t.exit_fill.commission_cost, "exit_commission"), (21, t.gross_price_pnl, "gross"),
                               (22, t.realized_pnl, "realized")):
            _close(f64(r[idx]), float(val), lv, f"{w} {name}")
    res = o.result
    for key, val in (("cost_spread", res.cost_spread), ("cost_slippage", res.cost_slippage),
                     ("cost_commission", res.cost_commission), ("cost_swap", res.cost_swap),
                     ("final_equity", res.final_equity)):
        _close(f64(diag[key]), float(val), lv, f"total {key}")
    eq = _rows(run.run_dir / "equity.csv", 3)
    curve = np.asarray(res.equity_curve, dtype=np.float64)
    if len(eq) != curve.size:
        lv.bad(f"equity curve: EA {len(eq)} points, oracle {curve.size}")
    for k, (e, v) in enumerate(zip(eq, curve)):
        _close(f64(e[2]), float(v), lv, f"equity[{k}]")
    for key, val in (("n_trades", res.n_trades), ("n_round_trips", res.n_round_trips), ("n_open", res.n_open)):
        if int(diag[key]) != int(val):
            lv.bad(f"{key}: EA {diag[key]} oracle {val}")
    reasons: dict[str, int] = {}
    for r in diag["exit_reasons"].split(";"):
        if r:
            reasons[r] = reasons.get(r, 0) + 1
    if reasons != dict(res.exit_reasons):
        lv.bad(f"exit_reasons: EA {reasons} oracle {dict(res.exit_reasons)}")
    if int(diag["draws_used"]) != len(o.draws):
        lv.bad(f"slippage draws used: EA {diag['draws_used']} oracle {len(o.draws)}")
    lv.summary = f"{lv.compared} values, {lv.exact} bit-exact, tolerance {TOL_E:g}"
    return lv


def oracle_label(o: OracleRun) -> str:
    min_trades, max_dd_limit, ret = early_gates()
    eq = np.asarray(o.result.equity_curve, dtype=np.float64)
    running = np.maximum.accumulate(eq)
    max_dd = float(-((eq - running) / np.maximum(running, 1e-9)).min()) if eq.size >= 2 else 0.0
    total_return = float((eq[-1] / eq[0]) - 1.0) if eq.size >= 2 else 0.0
    if o.result.n_trades < min_trades:
        return "LowTrades"
    if max_dd > max_dd_limit:
        return "HighDD"
    if total_return < ret:
        return "NegReturn"
    return "PASSES_EARLY_GATES"


def compare_h(run: TesterRun, cell: CellConfig, diag: dict[str, str]) -> Level:
    lv = Level("H")
    want = {"history_sha256": history_digest(cell), "replica_sha256": replica_digest(cell),
            "bars_seen": str(cell.n_bars),
            "first_bar_time": str(cell.first_time), "last_bar_time": str(cell.last_time), "complete": "1",
            "halted": "0", "divergences": "0"}
    for k, v in want.items():
        if diag.get(k) != v:
            lv.bad(f"{k}: EA {diag.get(k)!r} pinned {v!r}")
    for r in _rows(run.run_dir / "events.csv", 4):
        if r[0] in ("HISTORY_MISMATCH", "FATAL", "EXEC_DIVERGENCE"):
            lv.bad(f"event {r}")
    lv.summary = f"sha256 {diag.get('history_sha256', '')[:16]}..., {diag.get('bars_seen')} bars"
    return lv


@dataclass
class Verdict:
    subject: str
    levels: list[Level]

    @property
    def ok(self) -> bool:
        return all(lv.ok for lv in self.levels)

    def render(self) -> str:
        out = [f"SUBJECT {self.subject}: {'PASS' if self.ok else 'FAIL'}"]
        for lv in self.levels:
            out.append(f"  {lv.name:9s} {'PASS' if lv.ok else 'FAIL'} {lv.summary}")
            out.extend(f"    MISMATCH {line}" for line in lv.lines[:200])
            out.extend(f"    WITHIN-TOLERANCE {line}" for line in lv.inexact[:50])
        return "\n".join(out)


def judge(subject: str, o: OracleRun, run: TesterRun, cell: CellConfig) -> Verdict:
    diag = read_diag(run)
    a = compare_a(run)
    h = compare_h(run, cell, diag)
    b = compare_b(o, run, cell)
    c = compare_c(o, run, diag)
    d, rep = compare_d(o, run, cell)
    e = compare_e(o, run, diag)
    g = Level("DIAG")
    want = oracle_label(o)
    if diag.get("DIAG early_gate_label") != want:
        g.bad(f"EA {diag.get('DIAG early_gate_label')!r} oracle {want!r}")
    g.summary = f"early_gate_label {diag.get('DIAG early_gate_label')} (oracle {want})"
    return Verdict(subject, [a, h, b, c, rep, d, e, g])


__all__ = ["Level", "ParityParseError", "TOL_E", "Verdict", "compare_a", "compare_b", "compare_c", "compare_d",
           "compare_e", "compare_h", "judge", "oracle_label", "oracle_trades", "parse_report_deals", "read_diag", "tester_trades"]
