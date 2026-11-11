"""Full Deploy API (``/api/v2/app/deploy``): vault strategies -> live EAs on MT5 charts, and their heartbeats.

The steps are the owner's, one button each: add a bot from the vault, STAGE (generate + compile + place every
bot, MetaEditor must report 0 errors and 0 warnings), START (restart the terminal with algo trading on and the
attach script at start-up; ``confirm`` is required because it closes MetaTrader 5), then watch the heartbeats.
STOP ALL writes the STOP file every ARQVIO EA obeys. Each bot names the terminal -- and so the account -- it goes
to; ``/go`` does add + stage + start for one strategy in one call. The EA trades the account it is attached to.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.research.deploy import full_deploy as fd
from engine.research.deploy.live_ea import DEPLOY_RANGE, INCUBATION_STAGES, Guardrails
from engine.research.mql5.codegen import CodegenError

router = APIRouter(prefix="/api/v2/app/deploy", tags=["app-deploy"])
_lock = threading.Lock()


def _refuse(reason: str, status: int = 409) -> HTTPException:
    return HTTPException(status_code=status, detail=f"REFUSED: {reason}")


def _mt5(chosen: str | None = None) -> dict[str, Any]:
    """The paths of terminal ``chosen`` (an id or exe path from /api/v2/app/terminals), else of the resolved
    terminal, or why there are none (each part separately)."""
    from engine.research.mql5 import terminal

    if chosen:
        import os

        from engine.research.data import terminals as mt5_terminals

        try:
            t = mt5_terminals.find(chosen)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no MetaTrader 5 terminal {chosen!r}") from None
        if t.data_dir is None:
            raise _refuse(f"{t.path}: its data folder is unknown (start it once so MetaTrader 5 creates it)")
        common = Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files"
        return {"terminal": t.path, "data_dir": t.data_dir, "mql5_dir": str(Path(t.data_dir) / "MQL5"),
                "common_files": str(common), "metaeditor": str(Path(t.install_dir) / "MetaEditor64.exe")}
    out: dict[str, Any] = {}
    for key, fn in (("terminal", terminal.terminal_exe), ("data_dir", terminal.data_dir),
                    ("mql5_dir", terminal.mql5_dir), ("common_files", terminal.common_files_dir)):
        try:
            out[key] = str(fn())
        except (OSError, ValueError) as exc:
            out[key] = None
            out.setdefault("problems", []).append(f"{key}: {exc}")
    return out


def _need(mt5: dict[str, Any], *keys: str) -> list[Path]:
    missing = [k for k in keys if not mt5.get(k)]
    if missing:
        raise _refuse("MetaTrader 5 is not resolved: " + "; ".join(mt5.get("problems", missing)))
    return [Path(mt5[k]) for k in keys]


@router.get("")
def overview() -> dict[str, Any]:
    mt5 = _mt5()
    try:
        bots = fd.load_bots()
    except (fd.DeployError, ValueError) as exc:
        raise _refuse(str(exc)) from exc
    beats = fd.heartbeats(Path(mt5["common_files"])) if mt5.get("common_files") else {}
    attach = {r["bot"]: r for r in fd.attach_results(Path(mt5["mql5_dir"]))} if mt5.get("mql5_dir") else {}
    rows = []
    for b in bots:
        hb = beats.get(b["magic"])
        rows.append({**{k: b[k] for k in ("bot", "magic", "dna_hash", "symbol", "timeframe", "stage", "status",
                                          "guardrails", "source", "created_utc", "events")},
                     "lot_scale": INCUBATION_STAGES[b["stage"]], "heartbeat": hb, "attach": attach.get(b["bot"]),
                     "live": None if hb is None else hb["state"]})
    stop = fd.stop_file(Path(mt5["common_files"])).exists() if mt5.get("common_files") else None
    return {"bots": rows, "mt5": mt5, "stop_all": stop, "incubation_stages": list(INCUBATION_STAGES),
            "guardrail_defaults": Guardrails().to_dict(), "heartbeat_fresh_s": fd.HEARTBEAT_FRESH_S}


class AddBot(BaseModel):
    vault_id: str
    symbol: str | None = None  # the broker's name for the cell's symbol, when it differs (symbol mapping)
    guardrails: dict[str, Any] = {}
    terminal: str | None = None  # the terminal (and so the account) the bot goes to


def _cell_limits(symbol: str, timeframe: str, dataset_id: str) -> tuple[tuple[int, ...], dict[str, Any]]:
    """The cell's own no-trade hours and spread guardrails, as its backtest priced them.

    v4: a cell backtested under the spread-regime gate deploys the same rule (multiple x the median of the last
    20 days of bars) and no fixed cap -- a fixed cap from one account's spread refused 20 of 20 entries on an
    account type with wider spreads (live EA tester run, 2026-10-26)."""
    from engine.config.settings import get_settings
    from engine.research.costs.spread_regime import window_bars
    from engine.research.search.real_backtest import load_cell_bars_from_snapshot, slow_cell_setup

    setup = slow_cell_setup(load_cell_bars_from_snapshot(symbol, timeframe, dataset_id))
    no_trade = tuple(int(h) for h in setup.no_trade_hours)
    if setup.spread_cap_by_bar is not None:
        st = get_settings()
        return no_trade, {"max_spread_points": 0, "spread_regime_multiple": float(st.spread_regime_multiple),
                          "spread_regime_bars": window_bars(timeframe, st.spread_regime_days)}
    cap = setup.max_spread_points
    return no_trade, {"max_spread_points": int(round(cap)) if cap is not None else 0}


@router.post("/bots")
def add_bot(body: AddBot) -> dict[str, Any]:
    terminal = _mt5(body.terminal)["terminal"] if body.terminal else None
    from engine.research.search.dna import DNA
    from engine.research.vault import library

    try:
        e = library.get(body.vault_id)
    except library.VaultError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if e["timeframe"] is None:
        raise _refuse(f"the strategy's cell {e['cell']} is not a validated cell; its chart is unknown")
    try:
        no_trade, spread_guard = _cell_limits(e["symbol"], e["timeframe"], e["dataset_id"])
    except Exception as exc:
        raise _refuse(f"the cell's spread record could not be read: {type(exc).__name__}: {exc}") from exc
    try:
        g = Guardrails.from_dict({**spread_guard, **body.guardrails})
        with _lock:
            row = fd.add_bot(DNA(**e["dna"]), symbol=body.symbol or e["symbol"], timeframe=e["timeframe"],
                             guardrails=g, no_trade_hours=no_trade, terminal=terminal,
                             source={"vault_id": e["id"], "tier": e["tier"], "score": e["score"], "grade": e["grade"],
                                     "campaign_id": e["source"]["campaign_id"], "cell": e["cell"]})
    except (fd.DeployError, CodegenError, TypeError) as exc:
        raise _refuse(str(exc), 422) from exc
    return row


class Stage(BaseModel):
    stage: int


@router.post("/bots/{bot}/stage")
def set_stage(bot: str, body: Stage) -> dict[str, Any]:
    with _lock:
        try:
            return fd.set_stage(bot, body.stage)
        except fd.DeployError as exc:
            raise _refuse(str(exc), 422) from exc


@router.delete("/bots/{bot}")
def remove_bot(bot: str) -> dict[str, Any]:
    with _lock:
        try:
            return fd.remove_bot(bot)
        except fd.DeployError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


def _compiler(mt5: dict[str, Any]):
    from engine.research.mql5.metaeditor import compile_mq5, require_clean

    me = Path(mt5["metaeditor"]) if mt5.get("metaeditor") else None
    inc = Path(mt5["mql5_dir"]) if me else None
    return lambda source, log: require_clean(compile_mq5(source, log_path=log, metaeditor=me, include=inc)).ex5


class StageBody(BaseModel):
    terminal: str | None = None


@router.post("/stage")
def stage(body: StageBody | None = None) -> dict[str, Any]:
    chosen = body.terminal if body else None
    mt5 = _mt5(chosen)
    (mql5,) = _need(mt5, "mql5_dir")
    with _lock:
        try:
            return fd.stage(mql5, compile_fn=_compiler(mt5), terminal=mt5["terminal"] if chosen else None)
        except (fd.DeployError, CodegenError) as exc:
            raise _refuse(str(exc), 422) from exc
        except Exception as exc:  # MetaEditor's own failures: the message names the file and the diagnostics
            raise _refuse(f"compile failed: {type(exc).__name__}: {exc}", 422) from exc


class Start(BaseModel):
    confirm: bool
    terminal: str | None = None


@router.post("/start")
def start(body: Start) -> dict[str, Any]:
    if not body.confirm:
        raise _refuse("START restarts MetaTrader 5; send confirm=true to go ahead", 422)
    mt5 = _mt5(body.terminal)
    terminal, data = _need(mt5, "terminal", "data_dir")
    scope = str(terminal) if body.terminal else None
    with _lock:
        active = [b for b in fd.load_bots()
                  if b["status"] != "REMOVED" and fd.same_terminal(b.get("terminal"), scope)]
        unstaged = [b["bot"] for b in active if b["status"] != "STAGED"]
        if not active:
            raise _refuse("no bots to start")
        if unstaged:
            raise _refuse(f"stage first: {unstaged} changed since the last STAGE")
        ini = fd.start_config(data, active[0])
        try:
            return fd.restart_terminal(terminal, ini)
        except fd.DeployError as exc:
            raise _refuse(str(exc)) from exc


class Stop(BaseModel):
    stop: bool


@router.post("/stop")
def stop_all(body: Stop) -> dict[str, Any]:
    (common,) = _need(_mt5(), "common_files")
    return {"stop_all": fd.set_stop(common, body.stop)}


@router.get("/bots/{bot}/log")
def bot_log(bot: str, lines: int = 200) -> dict[str, Any]:
    (common,) = _need(_mt5(), "common_files")
    rows = [b for b in fd.load_bots() if b["bot"] == bot]
    if not rows:
        raise HTTPException(status_code=404, detail=f"no bot {bot!r}")
    p = common / "ARQVIO" / "bots" / f"{rows[0]['magic']}.log"
    if not p.is_file():
        return {"bot": bot, "lines": [], "path": str(p), "state": "NO LOG YET"}
    text = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"bot": bot, "lines": text[-max(1, min(int(lines), 2000)):], "path": str(p), "state": "OK"}


class Go(BaseModel):
    vault_id: str
    terminal: str
    guardrails: dict[str, Any] = {}
    attach: bool = True  # restart the terminal so the EA opens on its chart; False: install the EA only


def broker_symbol(symbol: str, offered: list[dict[str, Any]]) -> str | None:
    """The target terminal's name for ``symbol``: itself, or the same main pair under the broker's suffix."""
    from engine.research.data.terminals import MAIN_PAIRS, _is_alias, main_pairs

    if symbol in {r["name"] for r in offered}:
        return symbol
    base = next((b for b, aliases in MAIN_PAIRS if any(_is_alias(symbol.upper(), a) for a in aliases)), None)
    return next((r["symbol"] for r in main_pairs(offered) if r["pair"] == base), None) if base else None


@router.post("/go")
def go(body: Go) -> dict[str, Any]:
    """One strategy to one terminal: add the bot, stage that terminal's bots, then (attach) restart it with them."""
    from engine.core import applog
    from engine.research.data import terminals as mt5_terminals
    from engine.research.deploy.live_ea import bot_name
    from engine.research.search.dna import DNA
    from engine.research.vault import library

    mt5 = _mt5(body.terminal)
    try:
        e = library.get(body.vault_id)
    except library.VaultError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    t = mt5_terminals.find(body.terminal)
    # the strategy's hours were learned on one clock; the EA reads the target server's (deploy/clock_guard.py)
    from engine.research.deploy import clock_guard

    clock = clock_guard.check(e.get("dataset_id"), t.remembered_server)
    if not clock["ok"]:
        raise _refuse(clock["reason"])
    if clock["warn"]:
        applog.write("deploy", clock["warn"], "WARN")
    symbol = e["symbol"]
    if t.running:
        try:
            symbol = broker_symbol(e["symbol"], mt5_terminals.symbols(t)) or ""
        except RuntimeError:
            symbol = e["symbol"]
        if not symbol:
            raise _refuse(f"this terminal does not offer {e['symbol']} (nor the same pair under another name)")
    name = bot_name(DNA(**e["dna"]), symbol, e["timeframe"])
    if name not in {b["bot"] for b in fd.load_bots() if b["status"] != "REMOVED"}:
        add_bot(AddBot(vault_id=body.vault_id, symbol=symbol, guardrails=body.guardrails, terminal=body.terminal))
        applog.write("deploy", f"{name}: added for {t.path}")
    staged = stage(StageBody(terminal=body.terminal))
    applog.write("deploy", f"{len(staged['staged'])} EA(s) compiled into {mt5['mql5_dir']} (0 errors, 0 warnings)",
                 "OK")
    out: dict[str, Any] = {"bot": name, "symbol": symbol, "terminal": t.path, "staged": staged, "started": None}
    if body.attach:
        out["started"] = start(Start(confirm=True, terminal=body.terminal))
        applog.write("deploy", f"MetaTrader 5 restarted with algo trading on; {name} opens on its {symbol} "
                               f"{e['timeframe']} chart (it keeps running when ARQVIO is closed)", "OK")
    else:
        applog.write("deploy", f"{name} installed: drag it from Navigator > Expert Advisors > ARQVIO onto a "
                               f"{symbol} {e['timeframe']} chart")
    return out


_bt_cache: dict[str, tuple[Any, float]] = {}


def _backtest_trades(vault_id: str) -> tuple[Any, float] | None:
    """(closed-trade net P&L, mean lot) of a vault strategy's own backtest, by the kernel (cached per strategy)."""
    if vault_id in _bt_cache:
        return _bt_cache[vault_id]
    import numpy as np

    from engine.research.backtest.kernel import TR_EXIT_BAR, TR_NET, TR_VOLUME
    from engine.research.search.dna import DNA
    from engine.research.search.real_backtest import fast_backtest_factory, load_cell_bars_from_snapshot
    from engine.research.vault import library

    try:
        e = library.get(vault_id)
    except library.VaultError:
        return None
    cell = load_cell_bars_from_snapshot(e["symbol"], e["timeframe"], e["dataset_id"])
    bt = fast_backtest_factory(cell)
    bt.capture["on"] = True
    bt(DNA(**e["dna"]))
    if "trades" not in bt.capture:
        return None
    rows = bt.capture["trades"]
    closed = rows[rows[:, TR_EXIT_BAR] >= 0]
    out = (np.asarray(closed[:, TR_NET]), float(np.mean(closed[:, TR_VOLUME])) if len(closed) else 0.0)
    _bt_cache[vault_id] = out
    return out


@router.get("/live")
def live_performance() -> dict[str, Any]:
    """Each bot's live trades from its terminal's deal history against its backtest band, and orphan ARQVIO magics
    (engine/research/deploy/live_perf.py). Terminals that are not running are named, never started."""
    from datetime import UTC, datetime

    from engine.research.data import terminals as mt5_terminals
    from engine.research.deploy import live_perf as lp
    from engine.research.mql5.magic import load_table

    bots = [b for b in fd.load_bots() if b.get("status") != "REMOVED"]
    rng = load_table().range(DEPLOY_RANGE)
    by_term: dict[str, list[dict[str, Any]]] = {}
    for b in bots:
        by_term.setdefault(b.get("terminal") or "", []).append(b)
    rows, orphans, notes = [], [], []
    for path, group in by_term.items():
        try:
            t = mt5_terminals.find(path) if path else None
        except KeyError:
            t = None
        if t is None or not t.running:
            notes.append(f"{path or 'no terminal recorded'}: MetaTrader 5 is not running; its bots' live trades cannot be read")
            rows += [{"bot": b["bot"], "magic": b["magic"], "verdict": "UNREAD", "trades": None} for b in group]
            continue
        mt5 = mt5_terminals._mt5()
        with mt5_terminals.MT5_LOCK:
            ok = mt5_terminals._attach(t, mt5)
            if not ok:
                notes.append(f"{t.path}: the terminal did not answer")
                continue
            try:
                deals = list(mt5.history_deals_get(datetime(2020, 1, 1, tzinfo=UTC), datetime.now(tz=UTC)) or ())
                positions = list(mt5.positions_get() or ())
            finally:
                mt5.shutdown()
        for b in group:
            bt = _backtest_trades(b["source"]["vault_id"])
            lots = float(b["guardrails"].get("lots", 0.01)) * INCUBATION_STAGES[b["stage"]]
            if bt is None:
                rows.append({"bot": b["bot"], "magic": b["magic"], "verdict": "NO BACKTEST", "trades": None})
                continue
            judged = lp.judge_bot(b, lp.deal_trades(deals, b["magic"]), bt[0], bt[1], lots)
            judged["incubation"] = lp.incubation_advice(int(b["stage"]), judged, INCUBATION_STAGES)
            rows.append(judged)
        orphans += lp.orphans(deals, positions, {b["magic"] for b in bots}, rng.base, rng.last)
    return {"bots": rows, "orphans": orphans, "notes": notes, "min_trades": lp.MIN_TRADES}
