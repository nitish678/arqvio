"""Full Deploy: the owner's bots as live EAs on MT5 charts, and the heartbeat that says whether they run.

The bot list is ``<ARQVIO_HOME>/artifacts/deploy/bots.json``: one row per bot, each with its own magic from the
``arqvio-deploy`` range (allocated in order, never reused), its DNA, chart, guardrails and incubation stage.

``stage(...)`` writes, into the terminal's MQL5 directory:

    Experts/ARQVIO/<bot>.mq5 (+ .ex5)   the live EA, compiled by MetaEditor and required clean
    Files/ARQVIO/tpl/<bot>.tpl          a chart template whose <expert> block loads it with its inputs
    Presets/ARQVIO/<bot>.set            the same inputs, for attaching by hand from the Navigator
    Scripts/ARQVIO/arqvio_attach.mq5    the script that opens each chart and applies its template
    Files/ARQVIO/deploy_plan.csv        the bots that script attaches

``start_config(...)`` writes the terminal start file: algo trading on, the attach script at start-up. The
terminal is restarted only by an explicit ``restart_terminal`` call (the API asks the owner first).

Every live EA writes ``Common/Files/ARQVIO/bots/<magic>.json`` every 15 s; ``heartbeats`` reads them, so
"running" in the app means a heartbeat, never an assumption. ``Common/Files/ARQVIO/STOP`` makes every bot
flatten its own positions and stand by until the file is removed.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.core import atomic
from engine.core.paths import get_artifacts_dir
from engine.core.schema import same_schema
from engine.research.deploy.live_ea import (
    DEPLOY_RANGE,
    INCUBATION_STAGES,
    PERIOD_MINUTES,
    Guardrails,
    LiveEA,
    attach_script_source,
    bot_name,
    generate_live,
)
from engine.research.mql5.codegen import CodegenError
from engine.research.mql5.magic import load_table
from engine.research.search.dna import DNA

SCHEMA = "arqvio.deploy.bots"
VERSION = 1
HEARTBEAT_FRESH_S = 60  # a bot whose last heartbeat is older is reported SILENT
ATTACH_SCRIPT = "arqvio_attach"


class DeployError(ValueError):
    """A deploy step the engine refuses; the message says why and nothing was changed."""


def deploy_dir() -> Path:
    return get_artifacts_dir() / "deploy"


def bots_path() -> Path:
    return deploy_dir() / "bots.json"


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def load_bots(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or bots_path()
    if not p.is_file():
        return []
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not same_schema(doc.get("schema"), SCHEMA) or doc.get("version") != VERSION or not isinstance(doc.get("bots"), list):
        raise DeployError(f"{p} is not a {SCHEMA} v{VERSION} document")
    return doc["bots"]


def save_bots(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    p = path or bots_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": SCHEMA, "version": VERSION, "bots": rows}, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    atomic.replace(tmp, p)


def _row(rows: list[dict[str, Any]], bot: str) -> dict[str, Any]:
    for r in rows:
        if r["bot"] == bot:
            return r
    raise DeployError(f"no bot {bot!r}")


def render(row: dict[str, Any]) -> LiveEA:
    """The live EA of one bot row, at its current stage."""
    g = Guardrails.from_dict({**row["guardrails"], "lot_scale": INCUBATION_STAGES[row["stage"]]})
    return generate_live(DNA(**row["dna"]), symbol=row["symbol"], timeframe=row["timeframe"],
                         magic=row["magic"], guardrails=g, no_trade_hours=tuple(row.get("no_trade_hours", ())))


def add_bot(dna: DNA, *, symbol: str, timeframe: str, guardrails: Guardrails, source: dict[str, Any],
            no_trade_hours: tuple[int, ...] = (), path: Path | None = None,
            terminal: str | None = None) -> dict[str, Any]:
    """Register a bot at incubation stage 0 with the next unused magic; refuses a duplicate active bot."""
    rows = load_bots(path)
    name = bot_name(dna, symbol, timeframe)
    if any(r["bot"] == name and r["status"] != "REMOVED" for r in rows):
        raise DeployError(f"{name} is already deployed; change its stage or remove it first")
    r = load_table().range(DEPLOY_RANGE)
    slot = len(rows)  # every row ever added holds its slot: a magic is never reused
    if slot >= r.span:
        raise DeployError(f"the {DEPLOY_RANGE} magic range is exhausted ({r.span} bots)")
    row = {"bot": name, "magic": r.base + slot, "dna": dna.to_dict(), "dna_hash": dna.dna_hash, "symbol": symbol,
           "timeframe": timeframe, "guardrails": {**guardrails.to_dict(), "lot_scale": INCUBATION_STAGES[0]},
           "stage": 0, "no_trade_hours": sorted(no_trade_hours), "status": "PLANNED", "source": source,
           "created_utc": utc_now(), "events": [{"utc": utc_now(), "event": "ADDED", "stage": 0}],
           "terminal": terminal}
    try:
        ea = render(row)  # refuse now (bad DNA, chart or guardrails), not at stage time
    except CodegenError as exc:
        raise DeployError(str(exc)) from exc
    row["source_sha256"] = ea.source_sha256
    rows.append(row)
    save_bots(rows, path)
    return row


def set_stage(bot: str, stage: int, *, path: Path | None = None) -> dict[str, Any]:
    """Move a bot along the incubation ladder (any stage, either way: the owner decides). Needs a re-stage."""
    if stage not in range(len(INCUBATION_STAGES)):
        raise DeployError(f"stage must be one of {list(range(len(INCUBATION_STAGES)))} (lot scale {INCUBATION_STAGES})")
    rows = load_bots(path)
    row = _row(rows, bot)
    if row["status"] == "REMOVED":
        raise DeployError(f"{bot} was removed")
    row["stage"] = stage
    row["guardrails"]["lot_scale"] = INCUBATION_STAGES[stage]
    row["status"] = "PLANNED"
    row["source_sha256"] = render(row).source_sha256
    row["events"].append({"utc": utc_now(), "event": "STAGE", "stage": stage})
    save_bots(rows, path)
    return row


def remove_bot(bot: str, *, path: Path | None = None) -> dict[str, Any]:
    """Mark a bot removed: it leaves the next plan. Its EA keeps running until its chart is closed in MT5, and its open
    trades are KEPT to their own stop / target (research "KEEP ITS OPEN TRADES"; ARQVIO never closes a position
    itself -- closing is the owner's, in MetaTrader 5 or with Stop all bots)."""
    rows = load_bots(path)
    row = _row(rows, bot)
    row["status"] = "REMOVED"
    row["events"].append({"utc": utc_now(), "event": "REMOVED", "open_trades": "kept"})
    save_bots(rows, path)
    return row


def template_text(ea: LiveEA) -> str:
    """A one-chart template that loads ``ea`` with its inputs (expertmode 33: the EA may trade)."""
    tf = ea.timeframe
    minutes = PERIOD_MINUTES[tf]
    period_type, period_size = (2, minutes // 1440) if tf == "D1" else (1, minutes // 60) if minutes >= 60 \
        else (0, minutes)
    lines = ["<chart>", f"symbol={ea.symbol}", f"period_type={period_type}", f"period_size={period_size}",
             "<expert>", f"name={ea.name}", f"path=Experts\\ARQVIO\\{ea.name}.ex5", "expertmode=33", "<inputs>",
             *(f"{k}={v}" for k, v in ea.inputs.items()), "</inputs>", "</expert>",
             "<window>", "height=100.000000", "objects=0", "<indicator>", "name=Main", "path=", "apply=1",
             "show_data=1", "scale_inherit=0", "scale_line=0", "scale_line_percent=50", "scale_line_value=0.000000",
             "scale_fix_min=0", "scale_fix_min_val=0.000000", "scale_fix_max=0", "scale_fix_max_val=0.000000",
             "expertmode=0", "fixed_height=-1", "</indicator>", "</window>", "</chart>", ""]
    return "\r\n".join(lines)


def preset_text(ea: LiveEA) -> bytes:
    # LF only: MT5 keeps a CR as part of the value (S6 first parity run, engine/research/mql5/tester.py).
    return "".join(f"{k}={v}\n" for k, v in ea.inputs.items()).encode("ascii")


@dataclass(frozen=True)
class Staged:
    bot: str
    magic: int
    mq5: Path
    ex5: Path
    template: Path
    preset: Path
    source_sha256: str


Compile = Callable[[Path, Path], Path]  # (source, log) -> the clean .ex5, or raises


def same_terminal(a: str | None, b: str | None) -> bool:
    return a is None or b is None or a.lower() == b.lower()


def stage(mql5: Path, *, compile_fn: Compile, path: Path | None = None, terminal: str | None = None) -> dict[str, Any]:
    """Generate, compile and place every active bot (of ``terminal``, when given) and the attach script; write the
    plan. Returns a report."""
    rows = load_bots(path)
    active = [r for r in rows if r["status"] != "REMOVED" and same_terminal(r.get("terminal"), terminal)]
    if not active:
        raise DeployError("no bots to deploy: add one from the vault first")
    logs = deploy_dir() / "compile_logs"
    experts = mql5 / "Experts" / "ARQVIO"
    tpl_dir = mql5 / "Files" / "ARQVIO" / "tpl"
    presets = mql5 / "Presets" / "ARQVIO"
    for d in (experts, tpl_dir, presets, mql5 / "Scripts" / "ARQVIO", logs):
        d.mkdir(parents=True, exist_ok=True)
    staged: list[Staged] = []
    for row in active:
        ea = render(row)
        mq5 = experts / f"{ea.name}.mq5"
        mq5.write_bytes(ea.source.encode("utf-8"))
        ex5 = compile_fn(mq5, logs / f"{ea.name}.log")
        tpl = tpl_dir / f"{ea.name}.tpl"
        tpl.write_bytes(b"\xff\xfe" + template_text(ea).encode("utf-16-le"))  # templates need the BOM
        preset = presets / f"{ea.name}.set"
        preset.write_bytes(preset_text(ea))
        staged.append(Staged(ea.name, ea.magic, mq5, ex5, tpl, preset, ea.source_sha256))
    script = mql5 / "Scripts" / "ARQVIO" / f"{ATTACH_SCRIPT}.mq5"
    script.write_bytes(attach_script_source().encode("utf-8"))
    compile_fn(script, logs / f"{ATTACH_SCRIPT}.log")
    plan = mql5 / "Files" / "ARQVIO" / "deploy_plan.csv"
    plan.write_bytes("".join(f"{s.bot},{r['symbol']},{PERIOD_MINUTES[r['timeframe']]},{s.bot}.tpl\n"
                             for s, r in zip(staged, active, strict=True)).encode("ascii"))
    result = mql5 / "Files" / "ARQVIO" / "deploy_result.csv"
    if result.exists():
        result.unlink()  # a result left by an earlier deploy must not read as this one's
    for row, s in zip(active, staged, strict=True):
        row["status"] = "STAGED"
        row["source_sha256"] = s.source_sha256
        row["events"].append({"utc": utc_now(), "event": "STAGED", "stage": row["stage"]})
    save_bots(rows, path)
    return {"staged": [{"bot": s.bot, "magic": s.magic, "mq5": str(s.mq5), "ex5": str(s.ex5),
                        "template": str(s.template), "preset": str(s.preset), "source_sha256": s.source_sha256}
                       for s in staged], "plan": str(plan), "script": str(script), "utc": utc_now()}


def start_config(data_dir: Path, first: dict[str, Any]) -> Path:
    """The terminal start file: algo trading on and the attach script run at start-up on ``first``'s chart."""
    ini = data_dir / "arqvio_start.ini"
    tf = first["timeframe"]
    ini.write_text("\r\n".join(["[Experts]", "AllowLiveTrading=1", "Enabled=1", "Account=0", "Profile=0", "",
                                "[StartUp]", f"Script=ARQVIO\\{ATTACH_SCRIPT}", f"Symbol={first['symbol']}",
                                f"Period={tf}", "ShutdownTerminal=0", ""]), encoding="ascii")
    return ini


def attach_results(mql5: Path) -> list[dict[str, Any]]:
    """What the attach script reported per bot (empty until it has run)."""
    p = mql5 / "Files" / "ARQVIO" / "deploy_result.csv"
    if not p.is_file():
        return []
    raw = p.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
    out = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 4 and parts[0]:
            out.append({"bot": parts[0], "symbol": parts[1], "minutes": parts[2], "status": parts[3],
                        "detail": parts[4] if len(parts) > 4 else ""})
    return out


def heartbeats(common_files: Path, *, now: float | None = None) -> dict[int, dict[str, Any]]:
    """magic -> the bot's last heartbeat with its age in seconds and LIVE / SILENT."""
    t = time.time() if now is None else now
    out: dict[int, dict[str, Any]] = {}
    d = common_files / "ARQVIO" / "bots"
    for p in sorted(d.glob("*.json")) if d.is_dir() else ():
        try:
            hb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not same_schema(hb.get("schema"), "arqvio.bot_heartbeat") or not isinstance(hb.get("magic"), int):
            continue
        age = t - p.stat().st_mtime
        hb["age_s"] = round(age, 1)
        hb["state"] = "LIVE" if age <= HEARTBEAT_FRESH_S and not hb.get("standby", "").startswith("removed") \
            else "SILENT"
        out[hb["magic"]] = hb
    return out


def stop_file(common_files: Path) -> Path:
    return common_files / "ARQVIO" / "STOP"


def set_stop(common_files: Path, stop: bool) -> bool:
    """Create (stop all bots) or remove (let them trade) the STOP file; returns whether it now exists."""
    p = stop_file(common_files)
    if stop:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"stopped from ARQVIO at {utc_now()}\n", encoding="ascii")
    elif p.exists():
        p.unlink()
    return p.exists()


def terminal_pids(terminal: Path) -> list[int]:
    """PIDs of running processes whose executable is ``terminal`` (that install only, never another one)."""

    from engine.research.data.terminals import running_terminals

    return running_terminals().get(os.path.normcase(str(Path(terminal).resolve())), [])


def launch_detached(terminal: Path, ini: Path) -> None:
    """Start the terminal outside the engine's process tree (through ``cmd /c start``, which exits at once): closing
    ARQVIO ends its own processes with their children, and MetaTrader 5 -- with every EA on its charts -- must keep
    running when the app is closed."""
    subprocess.Popen(["cmd", "/c", "start", "", "/D", str(Path(terminal).parent), str(terminal), f"/config:{ini}"],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def restart_terminal(terminal: Path, ini: Path, *, wait_s: float = 45.0) -> dict[str, Any]:
    """Close the running terminal politely (WM_CLOSE: it saves its profile and charts), then start it with ``ini``.
    A terminal that does not close in ``wait_s`` is left running and the start is refused: never a forced kill."""
    before = terminal_pids(terminal)
    for pid in before:
        subprocess.run(["taskkill", "/PID", str(pid)], capture_output=True, timeout=30, check=False)
    deadline = time.monotonic() + wait_s
    while before and terminal_pids(terminal) and time.monotonic() < deadline:
        time.sleep(1.0)
    still = terminal_pids(terminal)
    if still:
        raise DeployError(f"the terminal (pid {still}) did not close within {wait_s:.0f} s; close MetaTrader 5 "
                          "yourself and press Start again")
    launch_detached(terminal, ini)
    deadline = time.monotonic() + 30
    started: list[int] = []
    while not started and time.monotonic() < deadline:
        time.sleep(1.0)
        started = terminal_pids(terminal)
    return {"closed": before, "started": started[0] if started else None, "config": str(ini), "utc": utc_now()}


__all__ = ["DeployError", "HEARTBEAT_FRESH_S", "Staged", "add_bot", "attach_results", "bots_path", "deploy_dir",
           "heartbeats", "load_bots", "preset_text", "remove_bot", "render", "save_bots", "set_stage", "set_stop",
           "launch_detached", "restart_terminal", "same_terminal", "stage", "start_config", "stop_file",
           "template_text", "terminal_pids"]
