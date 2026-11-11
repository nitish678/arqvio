"""Add market: a new (symbol, timeframe) cell from the owner's MetaTrader 5, in the app, as a QUICK cell.

The repo's new-cell wizard (engine/research/wizard) is a ceremony for the repo registry: it needs git, a sibling
cell of the same timeframe to explain gaps, and five CLI steps. This is the app's path, for any market the
terminal offers on M1-D1. One run, read-only against the terminal (market data only, never an order):

1. specs   the symbol's contract specs from symbol_info()          -> <ARQVIO_HOME>/artifacts/specs/<S>.json
           (kept when the file exists)
2. spread  a 24-hour spread matrix from the last 28 days of ticks   -> <ARQVIO_HOME>/artifacts/spread/<S>.json
           (kept when the file exists; the same rule as scripts/calibrate_spreads.py)
3. bars    copy_rates_range over the owner's window, pinned under its content hash
                                                                      -> <ARQVIO_HOME>/artifacts/datasets/
4. cell    a QUICK entry, sealed, appended to <ARQVIO_HOME>/configs/user_cells.json; its audit row is the pull's
           own measurement (bars, years, first and last bar), so the campaign preflight reads it like an audit row

Refusals (nothing written past the step that refused): no terminal; the symbol not offered; a server clock that is
not UTC (ARQVIO pins UTC bars: Exness and other GMT+0 servers work; a GMT+2/+3 server is named by its clock and
refused, the DST-aware conversion is a later stage); fewer than MIN_BARS bars; a cell that exists; an unknown cross
cell. A QUICK cell is labelled as such everywhere: its data was pinned in the app, not proven by
the repo's wizard ceremony.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from engine.core import atomic
from engine.core.paths import get_artifacts_dir
from engine.research.data.time_basis import SERVER_TIME, BasisProbe, server_time_probe
from engine.research.registry import cells as registry
from engine.research.registry.annualisation import TF_MINUTES, bars_per_year

MIN_BARS = 2000
HISTORY_TRIES = 4
# v4 held-out (ARQVIO_V4.md 10.3): breeding sees the first 70% of the window; the last 30% is kept back.
HELD_OUT_FRAC = 0.30
HISTORY_PAUSE_S = 2.0
TICK_WINDOW_DAYS = 28
AUDIT_FIELDS = ("symbol", "timeframe", "bar_cap", "years_available", "bars_returned", "first_bar_utc",
                "last_bar_utc", "source", "time_basis")
SOURCE = "MT5_LIVE (ARQVIO Add market)"


class AddMarketRefused(Exception):
    """A refused step: ``step`` names it, the message says why."""

    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"{step}: {detail}")
        self.step = step
        self.detail = detail


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _basis_refusal(probe: BasisProbe) -> str:
    """The refusal for a server clock ARQVIO cannot pin: the clock in broker terms, then what the owner can do.

    A GMT+2/+3 (EET/EEST) server is the common case and is not an owner mistake, so it says so and names the one
    working alternative instead of leaking the probe's internals.
    """
    offset = probe.measured_offset_s
    if probe.basis == "broker_offset" and abs(offset) in (7200, 10800):
        clock = f"GMT{'-' if offset < 0 else '+'}{abs(offset) // 3600}"
        return (f"this MT5 server's clock is {clock} (measured {offset:+d} s against UTC); ARQVIO pins UTC (GMT+0) "
                "bars today, so a GMT+0 server such as Exness/GMT+0 works -- a GMT+2/+3 server is refused by name, "
                "DST-aware conversion is a later stage")
    return (f"this MT5 server's clock is neither UTC nor GMT+2/+3 (measured {probe.delta_s:+.0f} s against UTC); "
            "ARQVIO pins UTC (GMT+0) bars, so a GMT+0 server such as Exness works")


def _closed_refusal(clock: Any) -> str:
    age = clock.newest_tick_age_s
    when = f"the newest tick is {age / 3600:.1f} h old" if age is not None and age == age else "no tick arrived"
    return (f"the market is closed ({when}) and no round-the-clock symbol (BTCUSD, ETHUSD, ...) answered, so the "
            "server clock cannot be measured; launch again while the market is open (Monday to Friday) -- after one "
            "measurement ARQVIO remembers a UTC server and works on weekends too")


def cell_dir(cid: str) -> Path:
    return get_artifacts_dir() / "cells" / cid


def ensure_specs(mt5: Any, symbol: str, account: Any) -> dict[str, Any]:
    from engine.research.data.symbol_specs import (
        save_snapshot,
        snapshot_from_mt5_symbol_info,
        specs_dir,
    )

    path = specs_dir() / f"{symbol}.json"
    if path.is_file():
        return {"path": str(path), "action": "kept", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    info = mt5.symbol_info(symbol)
    if info is None:
        raise AddMarketRefused("specs", f"symbol_info({symbol!r}) returned nothing: {mt5.last_error()}")
    data = snapshot_from_mt5_symbol_info(info, account_login=None, server=getattr(account, "server", None))
    try:
        written = save_snapshot(data)
    except Exception as exc:
        raise AddMarketRefused("specs", f"the terminal's specs cannot price a trade: {exc}") from exc
    return {"path": str(written), "action": "written", "sha256": hashlib.sha256(written.read_bytes()).hexdigest()}


def spread_matrix(mt5: Any, symbol: str, probe: Any, *, now: datetime) -> Any:
    """The 24-hour spread matrix from the last TICK_WINDOW_DAYS of ticks (scripts/calibrate_spreads.py's rule:
    hourly medians for the gates, hourly means of positive spreads for the cost model)."""
    from engine.data.spread_calibration import CALIBRATOR_VERSION, SpreadMatrix
    from engine.research.data.time_basis import convert_times

    info = mt5.symbol_info(symbol)
    point = float(getattr(info, "point", 0.0) or 0.0)
    if point <= 0:
        raise AddMarketRefused("spread", f"{symbol}: the terminal reports no point size")
    start = (now - timedelta(days=TICK_WINDOW_DAYS)).replace(tzinfo=None)
    ticks = mt5.copy_ticks_range(symbol, start, now.replace(tzinfo=None), mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        raise AddMarketRefused("spread", f"no ticks for {symbol} in the last {TICK_WINDOW_DAYS} days: "
                                         f"{mt5.last_error()}")
    spreads = (np.asarray(ticks["ask"], dtype=np.float64) - np.asarray(ticks["bid"], dtype=np.float64)) / point
    hours = (convert_times(np.asarray(ticks["time"], dtype=np.int64), probe) // 3600) % 24
    buckets, means, counts = np.zeros(24), np.zeros(24), np.zeros(24, dtype=np.int64)
    for h in range(24):
        vals = spreads[hours == h]
        counts[h] = vals.size
        if vals.size:
            buckets[h] = float(np.median(vals))
            positive = vals[vals > 0]
            if positive.size:
                means[h] = float(np.mean(positive))
    return SpreadMatrix(symbol=symbol, buckets=tuple(buckets), mean_buckets=tuple(means), sample_count=int(counts.sum()),
                        source="MT5", calibrator_version=CALIBRATOR_VERSION, calibrated_utc=now.isoformat(),
                        time_basis=probe.basis)


def _spread_basis(probe: Any) -> str:
    return "server_time" if getattr(probe, "basis", "") == SERVER_TIME else "utc"


def _stored_basis(path: Path) -> str:
    """The clock a stored spread matrix was bucketed in ("utc" for UTC servers and old files)."""
    try:
        basis = json.loads(path.read_text(encoding="utf-8")).get("time_basis") or "utc"
    except (OSError, ValueError):
        return "unreadable"
    return "server_time" if basis == SERVER_TIME else "utc"


def ensure_spread(mt5: Any, symbol: str, probe: Any, *, now: datetime) -> dict[str, Any]:
    from engine.data.spread_calibration import evaluate, persist, spread_dir

    path = spread_dir() / f"{symbol}.json"
    if path.is_file() and _stored_basis(path) == _spread_basis(probe):
        return {"path": str(path), "action": "kept", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    matrix = spread_matrix(mt5, symbol, probe, now=now)
    check = evaluate(matrix)
    if check.blocks:
        raise AddMarketRefused("spread", f"the measured spread matrix is unusable: {check.line()}")
    written = persist(matrix, spread_dir())
    return {"path": str(written), "action": "written", "sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
            "ticks": matrix.sample_count}


def pull_bars(mt5: Any, symbol: str, tf: str, start: datetime, end: datetime) -> dict[str, np.ndarray]:
    code = getattr(mt5, f"TIMEFRAME_{tf}")
    rates, n = None, 0
    for attempt in range(HISTORY_TRIES):
        # A symbol/timeframe whose history the terminal has not downloaded yet answers empty while it downloads
        # (6 and 10 years of M5 came back as 0 bars, then in full minutes later): ask again after a pause.
        rates = mt5.copy_rates_range(symbol, code, start.replace(tzinfo=None), end.replace(tzinfo=None))
        n = 0 if rates is None else len(rates)
        if n >= MIN_BARS or attempt == HISTORY_TRIES - 1:
            break
        time.sleep(HISTORY_PAUSE_S)
    if n < MIN_BARS:
        # Range pulls can return empty while MT5 builds history. The positional
        # fallback is capped by terminal_info.maxbars, and must page backwards
        # in chunks for M1 instead of asking for one oversized array.
        info = mt5.terminal_info()
        cap = int(getattr(info, "maxbars", 0) or 0) or 100_000
        chunk = min(cap, 10_000)
        newest_parts = []
        end_pos = 0
        use_pos = hasattr(mt5, "copy_rates_from_pos")
        while end_pos < cap:
            count = min(chunk, cap - end_pos)
            if use_pos:
                newest = mt5.copy_rates_from_pos(symbol, code, end_pos, count)
            else:
                # Compatibility for narrow test adapters and older wrappers;
                # the official MT5 package exposes copy_rates_from_pos.
                newest = mt5.copy_rates_from(symbol, code, end.replace(tzinfo=None), count)
            if newest is None or len(newest) == 0:
                break
            newest_parts.append(newest)
            end_pos += len(newest)
            if len(newest) < count or not use_pos:
                break
        if newest_parts:
            newest = np.concatenate(newest_parts)
            keep = newest[(newest["time"] >= int(start.timestamp())) & (newest["time"] < int(end.timestamp()))]
            if len(keep) > n:
                rates, n = keep, len(keep)
            # Do not count bars outside the requested window.
            if n < MIN_BARS and newest_parts and not use_pos:
                # Legacy adapters lack positional paging; preserve measured cap.
                n = min(cap, max(n, len(newest_parts[0])))
    if n < MIN_BARS:
        raise AddMarketRefused("bars", f"{n} {tf} bars of {symbol} in {start.date()}..{end.date()}; a cell needs "
                                       f"at least {MIN_BARS} (widen the window or pick a shorter timeframe)")
    return {k: np.asarray(rates[k]) for k in ("time", "open", "high", "low", "close", "spread", "tick_volume",
                                              "real_volume")}


def _snapshot_basis(probe: BasisProbe) -> str:
    return SERVER_TIME if probe.basis == SERVER_TIME else "utc"


def audit_row(symbol: str, tf: str, bars: dict[str, np.ndarray], basis: str = "utc") -> str:
    """The pull's own measurement as one target-audit CSV row (no header)."""
    t0, t1 = int(bars["time"][0]), int(bars["time"][-1])
    years = (t1 - t0) / (365.25 * 86400)
    row = [symbol, tf, str(len(bars["time"])), f"{years:.4f}", str(len(bars["time"])),
           datetime.fromtimestamp(t0, tz=UTC).isoformat(), datetime.fromtimestamp(t1, tz=UTC).isoformat(),
           SOURCE, basis]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(row)
    return buf.getvalue()


def quick_audit_rows() -> dict[tuple[str, str], dict[str, str]]:
    """(symbol, timeframe) -> the audit row each QUICK cell carries, parsed with the target audit's columns."""
    out = {}
    for e in registry.load_user():
        row = next(csv.reader([e["audit_row_ref"]["row"]]))
        if len(row) == len(AUDIT_FIELDS):
            out[(e["symbol"], e["timeframe"])] = {**dict(zip(AUDIT_FIELDS, row, strict=True)),
                                                  "as_of": e["audit_row_ref"]["as_of"]}
    return out


def append_cell(entry: dict[str, Any]) -> dict[str, Any]:
    """Seal ``entry`` (status QUICK) into the user cells file, after the existing ones; returns it sealed."""
    path = registry.user_cells_path()
    entries = registry.load_user()
    sealed = registry.seal([*entries, entry], schema=registry.USER_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sealed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    registry.load_user(tmp)  # the sealed document must verify before it replaces the file
    atomic.replace(tmp, path)
    return sealed["entries"][-1]


def remove_cell(cid: str) -> None:
    """Drop a QUICK cell from the user cells file (its pinned data stays on disk)."""
    entries = registry.load_user()
    keep = [e for e in entries if e["cell_id"] != cid]
    if len(keep) == len(entries):
        raise AddMarketRefused("remove", f"{cid} is not a QUICK cell")
    path = registry.user_cells_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registry.seal(keep, schema=registry.USER_SCHEMA), indent=2) + "\n", encoding="utf-8")
    atomic.replace(tmp, path)


def parse_day(day: str) -> datetime:
    try:
        return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise AddMarketRefused("window", f"{day!r} is not a YYYY-MM-DD day") from exc


def cell_id(symbol: str, tf: str) -> str:
    """eurusdm_h1; a broker suffix such as ".pro" or "#" becomes "_" (run and campaign ids allow [a-z0-9_] only)."""
    return re.sub(r"[^a-z0-9_]", "_", f"{symbol.lower()}_{tf.lower()}")


def _check_market(symbol: str, tf: str) -> None:
    if tf not in TF_MINUTES:
        raise AddMarketRefused("timeframe", f"{tf!r}; one of {sorted(TF_MINUTES, key=TF_MINUTES.get)}")
    if not symbol or not symbol.replace(".", "").replace("_", "").replace("#", "").isalnum():
        raise AddMarketRefused("symbol", f"{symbol!r} is not a symbol name")


def pull(symbol: str, tf: str, start_day: str, end_day: str, *, mt5: Any, terminal: str | None,
         now: datetime | None = None) -> dict[str, Any]:
    """Steps 1-3 for (symbol, tf): specs, spread matrix and the window's bars, pinned. Returns the pull's record
    (written to the cell's add_market.json); the cell itself is entered by ``register``. Raises AddMarketRefused."""
    from engine.research.data import server_clock
    from engine.research.data.dataset_snapshot import dataset_hash, find_snapshot, save_snapshot
    from engine.research.data.terminals import MT5_LOCK

    now = now or datetime.now(tz=UTC)
    _check_market(symbol, tf)
    cid = cell_id(symbol, tf)
    start, end = parse_day(start_day), parse_day(end_day)
    if not start < end <= now:
        raise AddMarketRefused("window", f"{start_day}..{end_day} must be a past window, start before end")
    with MT5_LOCK:
        if not mt5.initialize(**({"path": terminal} if terminal else {})):
            raise AddMarketRefused("terminal", f"MetaTrader 5 did not attach: {mt5.last_error()}")
        try:
            account = mt5.account_info()
            if account is None:
                raise AddMarketRefused("terminal", "the terminal is not logged in to an account")
            if not mt5.symbol_select(symbol, True):
                raise AddMarketRefused("symbol", f"{symbol!r} is not offered by this terminal: {mt5.last_error()}")
            server = str(getattr(account, "server", "") or "")
            try:
                clock = server_clock.resolve(mt5, symbol, server=server, now=now)
            except Exception as exc:
                raise AddMarketRefused("time basis", str(exc)) from exc
            probe = clock.probe
            if server_clock.closed_market(clock):
                raise AddMarketRefused("time basis", _closed_refusal(clock))
            if probe.basis == "broker_offset" and abs(probe.measured_offset_s) in (7200, 10800):
                probe = server_time_probe(probe)  # GMT+2/+3: pinned in the server's own clock (time_basis.py)
            elif probe.basis != "server_equals_utc":
                raise AddMarketRefused("time basis", _basis_refusal(probe))
            specs = ensure_specs(mt5, symbol, account)
            spread = ensure_spread(mt5, symbol, probe, now=now)
            bars = pull_bars(mt5, symbol, tf, start, end)
            mode = "REAL" if int(getattr(account, "trade_mode", 0)) == 2 else "DEMO"
        finally:
            mt5.shutdown()
    pinned = {"time": bars["time"], "open_": bars["open"], "high": bars["high"], "low": bars["low"],
              "close": bars["close"], "spread": bars["spread"]}
    h16 = dataset_hash(symbol=symbol, timeframe=tf, **pinned)
    if find_snapshot(h16) is None:
        save_snapshot(symbol=symbol, timeframe=tf, **pinned, tick_volume=bars["tick_volume"],
                      real_volume=bars["real_volume"], server=server, time_basis=_snapshot_basis(probe))
    first = datetime.fromtimestamp(int(bars["time"][0]), tz=UTC).isoformat()
    last = datetime.fromtimestamp(int(bars["time"][-1]), tz=UTC).isoformat()
    record = {"schema": "arqvio.add_market", "version": 1, "cell_id": cid, "symbol": symbol, "timeframe": tf,
              "window": {"requested_start": start.isoformat(), "requested_end": end.isoformat(), "first_bar": first,
                         "last_bar": last, "bars": int(len(bars["time"]))},
              "dataset_hash16": h16, "server": server, "account_mode": mode, "time_basis": probe.basis,
              "time_basis_source": clock.source, "server_offset_s": int(probe.measured_offset_s),
              "specs": specs, "spread": spread, "audit_row": audit_row(symbol, tf, bars, _snapshot_basis(probe)),
              "utc": utc_now(),
              "label": "QUICK: pinned in the app, not proven by the repo's wizard ceremony"}
    d = cell_dir(cid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "add_market.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def hold_out(record: dict[str, Any], frac: float = HELD_OUT_FRAC) -> dict[str, Any]:
    """Split a pull: the cell breeds on the first (1 - frac) of its bars (a pinned snapshot of its own); the full
    window stays pinned under ``held_out.dataset_hash16`` for the held-out test after breeding. A window too short
    to keep MIN_BARS for breeding is not split, and the record says so -- nothing is held out silently."""
    from engine.research.data.dataset_snapshot import dataset_hash, find_snapshot, load_meta, load_snapshot, save_snapshot

    full_id = record["dataset_hash16"]
    bars = load_snapshot(full_id)
    meta = load_meta(full_id)
    n = len(bars.time)
    split = int(n * (1.0 - frac))
    if split < MIN_BARS or n - split < 100:
        return {**record, "held_out": {"state": "NONE", "reason": f"{n:,} bars: too few to keep {frac:.0%} back "
                                       f"and breed on {MIN_BARS:,}", "frac": frac}}
    sl = {"time": bars.time[:split], "open_": bars.open[:split], "high": bars.high[:split], "low": bars.low[:split],
          "close": bars.close[:split], "spread": bars.spread[:split]}
    sym, tf = record["symbol"], record["timeframe"]
    train_id = dataset_hash(symbol=sym, timeframe=tf, **sl)
    if find_snapshot(train_id) is None:
        save_snapshot(symbol=sym, timeframe=tf, **sl, tick_volume=bars.tick_volume[:split],
                      real_volume=bars.real_volume[:split], server=meta.server if meta else None,
                      time_basis=meta.time_basis if meta else "utc")
    split_time = datetime.fromtimestamp(int(bars.time[split]), tz=UTC).isoformat()
    w = dict(record["window"])
    w.update(bars=split, last_bar=datetime.fromtimestamp(int(bars.time[split - 1]), tz=UTC).isoformat())
    out = {**record, "dataset_hash16": train_id, "window": w,
           "held_out": {"state": "KEPT", "dataset_hash16": full_id, "split_bar": split, "split_time": split_time,
                        "bars": n - split, "frac": frac}}
    d = cell_dir(record["cell_id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "add_market.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def entry_for(record: dict[str, Any], cross_cell: str) -> dict[str, Any]:
    """The QUICK registry entry for a pull, re-tested on ``cross_cell``."""
    cid, symbol, w = record["cell_id"], record["symbol"], record["window"]
    return {
        "schema_version": registry.SCHEMA_VERSION, "cell_id": cid, "status": registry.QUICK, "symbol": symbol,
        "timeframe": record["timeframe"], "dataset_hash16": record["dataset_hash16"],
        "dataset_window": {"start": w["first_bar"], "end": w["last_bar"], "bars": w["bars"]},
        "cross_cell": cross_cell, "annualisation_ppy": bars_per_year(record["timeframe"]),
        "spread_artifact_hash": {"sha256": record["spread"]["sha256"],
                                 "source": f"<ARQVIO_HOME>/artifacts/spread/{symbol}.json"},
        "specs_artifact_hash": {"sha256": record["specs"]["sha256"],
                                "source": f"<ARQVIO_HOME>/artifacts/specs/{symbol}.json"},
        "audit_row_ref": {"file": "<ARQVIO_HOME>/configs/user_cells.json", "line": 1, "as_of": record["utc"],
                          "row": record["audit_row"]},
        "power_check": {"status": "UNKNOWN", "ref": "not computed for a QUICK cell"},
        "worker_status": {"status": "UNKNOWN", "fills_in": "not part of a QUICK cell", "note": record["label"]},
        "campaign_status": "UNRUN", "entered_by_stage": "ARQVIO Add market", "entered_at": record["utc"],
        "evidence": [f"<ARQVIO_HOME>/artifacts/cells/{cid}/add_market.json"],
    }


def register(pulls: list[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
    """Enter pulled markets as QUICK cells in one sealed write: (record, cross_cell) each. A QUICK cell of the same
    id is replaced (fresh data for a market bred before); each cross cell must be another research cell after the
    write, so the markets of one launch can be each other's cross cell."""
    new_ids = {r["cell_id"] for r, _ in pulls}
    kept = [e for e in registry.load_user() if e["cell_id"] not in new_ids]
    known = ({e["cell_id"] for e in registry.research_entries()} - new_ids) | new_ids
    for r, cross in pulls:
        if cross == r["cell_id"] or cross not in known:
            raise AddMarketRefused("cross cell", f"{cross!r} is not another research cell; one of {sorted(known)}")
    entries = [entry_for(r, cross) for r, cross in pulls]
    path = registry.user_cells_path()
    sealed = registry.seal([*kept, *entries], schema=registry.USER_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sealed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    registry.load_user(tmp)  # the sealed document must verify before it replaces the file
    atomic.replace(tmp, path)
    return sealed["entries"][len(kept):]


def run(symbol: str, tf: str, start_day: str, end_day: str, cross_cell: str, *, mt5: Any,
        terminal: str | None, now: datetime | None = None) -> dict[str, Any]:
    """Add (symbol, tf) as a new QUICK cell; returns the run's record. Raises AddMarketRefused."""
    _check_market(symbol, tf)
    cid = cell_id(symbol, tf)
    known = {e["cell_id"] for e in registry.research_entries()}
    if cid in known or cid in {e["cell_id"] for e in registry.load()}:
        raise AddMarketRefused("cell", f"{cid} already exists")
    if cross_cell not in known:
        raise AddMarketRefused("cross cell", f"{cross_cell!r} is not a research cell; one of {sorted(known)}")
    record = pull(symbol, tf, start_day, end_day, mt5=mt5, terminal=terminal, now=now)
    (entry,) = register([(record, cross_cell)])
    record["cross_cell"] = cross_cell
    (cell_dir(cid) / "add_market.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["entry"] = entry
    return record


__all__ = ["AUDIT_FIELDS", "AddMarketRefused", "MIN_BARS", "append_cell", "audit_row", "cell_id", "entry_for",
           "pull", "quick_audit_rows", "register", "remove_cell", "run", "spread_matrix"]
