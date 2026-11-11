"""Unit checks of the parity harness pieces: the oracle taps, the replica path, the tester INI, the comparators
(levels A, B, D, E)."""
from __future__ import annotations

import dataclasses
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine.research.backtest.order import OrderIntent
from engine.research.mql5 import oracle_trace as ot
from engine.research.mql5 import indicators as ind
from engine.research.mql5 import parity, replica
from engine.research.mql5.cell_config import battery_cell
from engine.research.mql5.codegen import generate
from engine.research.mql5.metaeditor import CompileResult
from engine.research.mql5.subjects import SUBJECT_IDS, battery_indices
from engine.research.mql5 import tester as st
from engine.research.search.sampler import random_dna


def _dna(family: str):
    return dataclasses.replace(random_dna(np.random.default_rng(20261001)), signal=family)


def test_the_oracle_draws_are_the_exogenous_stream(pinned_home: Path) -> None:
    run = ot.run_oracle(_dna("MACD_CROSS"), battery_cell())
    assert run.draws and list(run.draws) == ot.slippage_stream(battery_cell(), len(run.draws))
    n_fills = sum(1 for t in run.result.trades) + sum(
        1 for t in run.result.trades if t.exit_fill is not None and t.exit_reason == "SIGNAL")
    assert len(run.draws) >= n_fills  # close intents without a position also draw


def test_a_stream_the_oracle_did_not_draw_is_refused(pinned_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = ot.slippage_stream
    monkeypatch.setattr(ot, "slippage_stream", lambda cell, n: [v + 1e-12 for v in real(cell, n)])
    with pytest.raises(ot.OracleTraceError, match="not the exogenous stream"):
        ot.run_oracle(_dna("MACD_CROSS"), battery_cell())


class _Scripted:
    strategy_id = "scripted"

    def __init__(self):
        self.script = {1: ("long", 0.1), 2: ("long", 0.0, "pend"), 3: ("short", 0.1), 4: ("short", 0.0)}

    def on_bar(self, i, bars, state):
        step = self.script.get(i)
        if step is None:
            return None
        if len(step) == 3:
            state["pending_direction"] = "short"
        if i == 3:
            state.pop("pending_direction", None)
        return OrderIntent(direction=step[0], volume=step[1], bar_index=i)


def test_intent_reasons_follow_the_strategy_state() -> None:
    rec = ot._RecordingStrategy(_Scripted(), np.arange(10) * 3600)
    state = {}
    for i in range(6):
        if i == 3:
            state["pending_direction"] = "short"
        rec.on_bar(i, None, state)
    assert [x.reason for x in rec.intents] == ["ENTRY", "REVERSAL_CLOSE", "PENDING_OPEN", "SIGNAL_EXIT"]


def test_the_replica_paths_rebuild_the_encoded_pinned_bars_exactly(pinned_home: Path) -> None:
    """The chart path carries the ORDER-PRESERVING encoding of each pinned double (DEV-12), never a new price:
    its OHLC is the encoded OHLC exactly, and every encoded price rounds to the oracle's quote."""
    cell = battery_cell()
    d = int(cell.spec.digits)
    for low_first in (True, False):
        rows = [r.split(",") for r in replica.m1_rows(cell, low_first=low_first)]
        assert len(rows) == replica.PREFIX_DAYS * 24 + 7 * cell.n_bars
        win = rows[replica.PREFIX_DAYS * 24:]
        for k in range(cell.n_bars):
            seg = win[7 * k: 7 * k + 7]
            t0 = int(cell.bars.time[k])
            assert [int(r[0]) - t0 for r in seg] == [0, 60, 120, 180, 240, 300, 3540]
            px = [float(r[1]) for r in seg]
            raw = (cell.bars.open[k], cell.bars.high[k], cell.bars.low[k], cell.bars.close[k])
            assert (px[0], max(px), min(px), px[-1]) == tuple(float(replica.encode(float(p), d)) for p in raw)
            assert all(round(q, d) == round(float(p), d) for q, p in zip((px[0], max(px), min(px), px[-1]), raw))
            adverse = cell.bars.low[k] if low_first else cell.bars.high[k]
            assert px[2] == float(replica.encode(float(adverse), d)) and px[3] == px[0] and all(r[2] == "1" for r in seg)


def test_the_history_digest_is_over_the_pinned_bytes(pinned_home: Path) -> None:
    cell = battery_cell()
    d = replica.history_digest(cell)
    bars = dataclasses.replace(cell.bars, close=np.asarray(cell.bars.close).copy())
    bars.close[5] = np.nextafter(bars.close[5], np.inf)
    assert replica.history_digest(dataclasses.replace(cell, bars=bars)) != d and len(d) == 64


def test_the_tester_ini_names_the_replica_window_and_model(pinned_home: Path) -> None:
    cell = battery_cell()
    ea = generate(_dna("RSI_REVERSION"), cell, magic=610_000_000)
    ini = st.tester_ini(ea, cell, "run1").split("\r\n")
    assert f"Symbol={cell.replica_names[0]}" in ini and "Model=1" in ini and "ExpertParameters=run1.set" in ini
    assert "ShutdownTerminal=1" in ini and "Report=ARQVIO_S6\\reports\\run1" in ini


def test_the_registered_ids_and_battery_picks_are_fixed() -> None:
    assert len(SUBJECT_IDS) == 76 and SUBJECT_IDS[0] == "S-GA-usdcadm-00" and SUBJECT_IDS[-1] == "S-EDGE-e6"
    assert battery_indices()["RSI_REVERSION"] == [35, 13, 9, 2, 46, 24]


REPORT = """<table><tr><td>Deals</td></tr>
<tr><td>Time</td><td>Deal</td><td>Symbol</td><td>Type</td><td>Direction</td><td>Volume</td><td>Price</td><td>Order</td>
<td>Commission</td><td>Swap</td><td>Profit</td><td>Balance</td><td>Comment</td></tr>
<tr><td>2026.07.15 00:00:00</td><td>1</td><td></td><td>balance</td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>1</td><td>1</td><td></td></tr>
<tr><td>2026.07.15 18:01:30</td><td>2</td><td>{L}</td><td>buy</td><td>in</td><td>0.35</td><td>1.36000001</td><td>2</td><td>0</td><td>0</td><td>0</td><td>1</td><td>arqvio</td></tr>
<tr><td>2026.07.15 19:02:30</td><td>3</td><td>{L}</td><td>sell</td><td>out</td><td>0.35</td><td>1.3590000</td><td>3</td><td>0</td><td>0</td><td>0</td><td>1</td><td>sl 1.35900000</td></tr>
<tr><td>2026.07.15 20:01:30</td><td>4</td><td>{H}</td><td>sell</td><td>in</td><td>0.35</td><td>1.3580000</td><td>4</td><td>0</td><td>0</td><td>0</td><td>1</td><td>arqvio</td></tr>
<tr><td>2026.07.16 00:00:00</td><td>5</td><td>{H}</td><td>buy</td><td>out</td><td>0.35</td><td>1.35800001</td><td>5</td><td>0</td><td>0</td><td>0</td><td>1</td><td>end of test</td></tr>
</table>"""


def _bits(x: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', x))[0]:016X}"


def _run(tmp_path: Path, cell) -> st.TesterRun:
    L, H = cell.replica_names
    (tmp_path / "r.htm").write_text(REPORT.replace("{L}", L).replace("{H}", H), encoding="utf-8")
    deals = [(2, L, 1.36000001), (3, L, 1.359), (4, H, 1.358), (5, H, 1.35800001)]
    (tmp_path / "deals.csv").write_text("".join(f"{d},0,{s},0,0,{_bits(p)},{_bits(0.35)},0,0,610000000\n"
                                               for d, s, p in deals), encoding="ascii")
    (tmp_path / "ea.ex5").write_bytes(b"x")
    compiled = CompileResult(tmp_path / "ea.mq5", tmp_path / "ea.ex5", 0, 0, (), 900, tmp_path / "ea.compile.log")
    return st.TesterRun("r", "ea", cell.name, tmp_path, tmp_path / "r.htm", tmp_path / "a.log", 1.0, 0, compiled)


def test_report_deals_pair_into_trades_and_route_by_side(tmp_path: Path, pinned_home: Path) -> None:
    cell = battery_cell()
    trades, lv = parity.tester_trades(_run(tmp_path, cell), cell)
    assert lv.ok, lv.lines
    assert [(t["side"], t["entry_price"], t["exit_price"]) for t in trades] == [("long", 1.36, 1.359),
                                                                             ("short", 1.358, None)]
    assert trades[0]["entry_time"] == trades[0]["exit_time"] - 3600


def test_d_names_the_row_and_field_that_differ(tmp_path: Path, pinned_home: Path) -> None:
    cell = battery_cell()
    run = _run(tmp_path, cell)
    t0 = parity._bar("2026.07.15 18:01:30")

    def trade(direction, entry_t, price, exit_fill, reason):
        return SimpleNamespace(direction=direction, volume=0.35, exit_reason=reason, exit_fill=exit_fill,
                               entry_fill=SimpleNamespace(bar_time=entry_t, requested_price=price))

    good = [trade("long", t0, 1.36, SimpleNamespace(bar_time=t0 + 3600, filled_price=1.359, requested_price=1.3595), "SL"),
            trade("short", t0 + 7200, 1.358, None, "EOD")]
    lv, _ = parity.compare_d(SimpleNamespace(result=SimpleNamespace(trades=good)), run, cell)
    assert lv.ok and lv.exact == 2
    bad = [good[0], trade("short", t0 + 7200, 1.35801, None, "EOD")]
    lv, _ = parity.compare_d(SimpleNamespace(result=SimpleNamespace(trades=bad)), run, cell)
    assert lv.lines == ["row 1: entry_price tester=1.358 oracle=1.35801"]


def test_e_counts_exact_bits_names_inexact_and_fails_past_1e8() -> None:
    lv = parity.Level("E")
    parity._close(0.1, 0.1, lv, "a")
    parity._close(0.1 + 1e-15, 0.1, lv, "b")
    parity._close(0.1 + 2e-8, 0.1, lv, "c")
    assert (lv.compared, lv.exact) == (3, 1) and len(lv.inexact) == 1 and lv.inexact[0].startswith("b:")
    assert len(lv.lines) == 1 and lv.lines[0].startswith("c:")


def test_a_reports_the_compile_and_fails_on_a_warning(tmp_path: Path, pinned_home: Path) -> None:
    run = _run(tmp_path, battery_cell())
    lv = parity.compare_a(run)
    assert lv.ok and lv.summary.startswith("MetaEditor 0 errors, 0 warnings")
    warned = dataclasses.replace(run, compiled=dataclasses.replace(run.compiled, warnings=1))
    assert parity.compare_a(warned).lines == ["ea.mq5: 0 errors, 1 warnings, ex5 present True"]


def _b_run(tmp_path: Path, family: str, *, flip_bar: int | None = None, bit_bar: int | None = None,
           prim: int | None = None):
    cell = battery_cell()
    dna = _dna(family)
    rows = ind.expected_rows(dna, cell)
    if bit_bar is not None:  # one ulp in the first defined double of that bar
        k = next(j for j in range(3, len(rows[bit_bar])) if len(rows[bit_bar][j]) == 16)
        rows[bit_bar][k] = f"{int(rows[bit_bar][k], 16) + 1:016X}"
    if flip_bar is not None:  # one condition flag flipped
        k = len(rows[flip_bar]) - 1
        rows[flip_bar][k] = "0" if rows[flip_bar][k] == "1" else "1"
    (tmp_path / "indicators.csv").write_text("".join(",".join(r) + "\n" for r in rows), encoding="ascii")
    out = []
    for j, (kind, x) in enumerate(ind.primitive_inputs(cell)):
        y = ind.primitive_expected(cell, kind, x)
        if j == prim:
            y = float(np.nextafter(y, np.inf))
        out.append(f"{kind},{struct.unpack('<Q', struct.pack('<d', x))[0]},{ind.hx(y)}\n")
    (tmp_path / "prims_out.csv").write_text("".join(out), encoding="ascii")
    run = _run(tmp_path, cell)
    return parity.compare_b(SimpleNamespace(dna=dna), run, cell)


@pytest.mark.parametrize("family", ["MA_CROSS", "RSI_REVERSION", "STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT"])
def test_b_passes_a_faithful_dump_for_every_family(tmp_path: Path, pinned_home: Path, family: str) -> None:
    lv = _b_run(tmp_path, family)
    assert lv.ok, lv.lines[:5]
    assert lv.exact == lv.compared == 800 and "prefix check 800 prefixes, 0 mismatches" in lv.summary
    assert lv.summary.endswith("primitives 2500/2500 bitwise")


def test_b_names_one_ulp_one_flipped_condition_and_one_primitive(tmp_path: Path, pinned_home: Path) -> None:
    lv = _b_run(tmp_path, "RSI_REVERSION", bit_bar=500, flip_bar=600, prim=7)
    assert len(lv.lines) == 3
    assert lv.lines[0].startswith("bar 500 (") and "columns [3]" in lv.lines[0]
    assert lv.lines[1].startswith("bar 600 (") and "columns [7]" in lv.lines[1]
    assert lv.lines[2].startswith("primitive 7 p ")


class _Proc:
    pid = 4242

    def __init__(self, *a, **k) -> None:
        pass

    def wait(self, timeout=None) -> int:
        return 0


def _relaunch_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, running: list[dict], *, close_ok: bool = True):
    files = tmp_path / "files"
    files.mkdir()
    monkeypatch.setattr(st, "arqvio_files_dir", lambda: files)
    monkeypatch.setattr(st, "terminal_exe", lambda: tmp_path / "terminal64.exe")
    seen: dict = {"closed": []}

    def procs():
        return [r for r in running if r["ProcessId"] not in seen["closed"]]

    def close(pid: int) -> bool:
        if close_ok:
            seen["closed"].append(pid)
        return close_ok

    monkeypatch.setattr(st, "terminal_processes", procs)
    monkeypatch.setattr(st, "_close_gracefully", close)
    monkeypatch.setattr(st.subprocess, "Popen", _Proc)
    return files, seen


def test_a_plain_relaunch_of_the_own_exit_is_closed_and_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files, seen = _relaunch_env(tmp_path, monkeypatch, [])
    st.run_terminal(tmp_path / "a.ini", timeout_s=5)          # records the own exit
    exit_ms = json.loads((files / st.EXIT_STATE).read_text())["exit_ms"]
    running = [{"ProcessId": 77, "CommandLine": '"C:\\mt5\\1\\terminal64.exe"', "CreatedMs": exit_ms + 4000}]
    monkeypatch.setattr(st, "terminal_processes", lambda: [r for r in running if r["ProcessId"] not in seen["closed"]])
    st.run_terminal(tmp_path / "b.ini", timeout_s=5)
    assert seen["closed"] == [77]
    log = (files / st.RELAUNCH_LOG).read_text()
    assert "own plain terminal pid 77" in log and "created 4.0 s after own pid 4242 exited; closed gracefully=True" in log


def test_a_plain_terminal_this_process_started_is_closed_and_logged(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """DEV-20: MetaTrader5.initialize() in the same process (the suite's reachability checks) leaves one running."""
    files, seen = _relaunch_env(tmp_path, monkeypatch, [
        {"ProcessId": 79, "ParentProcessId": os.getpid(), "CommandLine": '"C:\\mt5\\1\\terminal64.exe"',
         "CreatedMs": 1}])
    st.run_terminal(tmp_path / "d.ini", timeout_s=5)
    assert seen["closed"] == [79]
    assert "own plain terminal pid 79" in (files / st.RELAUNCH_LOG).read_text()
    assert f"started by this process (pid {os.getpid()}, MetaTrader5 initialize); closed gracefully=True" in \
        (files / st.RELAUNCH_LOG).read_text()


@pytest.mark.parametrize("case", ["late", "config", "no_exit_recorded", "close_fails"])
def test_any_other_running_terminal_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    files, seen = _relaunch_env(tmp_path, monkeypatch, [], close_ok=case != "close_fails")
    if case != "no_exit_recorded":
        (files / st.EXIT_STATE).write_text('{"pid": 4242, "exit_ms": 1000000}')
    created = 1000000 + (120000 if case == "late" else 3000)
    cmd = '"C:\\mt5\\1\\terminal64.exe" /config:x.ini' if case == "config" else '"C:\\mt5\\1\\terminal64.exe"'
    running = [{"ProcessId": 78, "ParentProcessId": 1, "CommandLine": cmd, "CreatedMs": created}]
    monkeypatch.setattr(st, "terminal_processes", lambda: [r for r in running if r["ProcessId"] not in seen["closed"]])
    with pytest.raises(st.TesterError, match="did not close" if case == "close_fails" else "is running"):
        st.run_terminal(tmp_path / "c.ini", timeout_s=5)
    assert seen["closed"] == []


def test_the_report_parser_refuses_an_unknown_layout(tmp_path: Path) -> None:
    p = tmp_path / "x.htm"
    p.write_text("<table><tr><td>Deals</td></tr></table>", encoding="utf-8")
    with pytest.raises(parity.ParityParseError, match="0 deals header rows"):
        parity.parse_report_deals(p)
