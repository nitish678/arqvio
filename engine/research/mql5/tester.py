"""The Strategy Tester driver: one generated EA on its cell's replica, headless, fail-loud.

``run_ea`` places and compiles the EA (MetaEditor, 0 / 0 required), writes the
tester ``.set`` and INI at call-time paths, and runs
``terminal64.exe /config:<ini>`` (ShutdownTerminal=1). It refuses to start
while a terminal of this install is running (listed by full command line;
nothing is stopped) -- except a plain terminal ARQVIO itself caused (started by this
process, or the documented relaunch of its own last run), which
is closed gracefully and logged -- and a run past its wall cap has ONLY its own child process
stopped and is reported as a timeout, never as a result. Afterwards the run's
journal directory, the tester report and the agent-log lines of the run are
collected; a missing journal is an error naming the agent log tail.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from engine.research.mql5.cell_config import CellConfig
from engine.research.mql5.codegen import GeneratedEA
from engine.research.mql5.metaeditor import CompileResult, compile_mq5, require_clean
from engine.research.mql5.terminal import arqvio_files_dir, data_dir, mql5_dir, terminal_exe

DEPOSIT = 10_000_000
LEVERAGE = 500
JOURNAL_FILES = ("intents.csv", "fills.csv", "trades.csv", "equity.csv", "events.csv", "diag.csv", "deals.csv",
                 "indicators.csv", "prims_out.csv")


class TesterError(RuntimeError):
    """A tester run that did not produce a result that can be compared."""


@dataclass(frozen=True)
class TesterRun:
    run_id: str
    ea_name: str
    cell_name: str
    run_dir: Path
    report: Path
    agent_log: Path
    wall_s: float
    terminal_exit: int
    compiled: CompileResult


def terminal_processes() -> list[dict]:
    """terminal64.exe processes of THIS install: command line, parent pid, creation time (read-only)."""
    exe = str(terminal_exe())
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='terminal64.exe'\" | Select-Object ProcessId, "
           "ParentProcessId, ExecutablePath, CommandLine, "
           "@{n='CreatedMs';e={([DateTimeOffset]$_.CreationDate).ToUnixTimeMilliseconds()}} | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                         check=False).stdout.strip()
    rows = [] if not out else json.loads(out)
    rows = rows if isinstance(rows, list) else [rows]
    return [r for r in rows if r.get("ExecutablePath") and Path(r["ExecutablePath"]).resolve() == Path(exe).resolve()]


# A plain "terminal64.exe" (no /config) of this install that ARQVIO itself caused is closed gracefully
# (CloseMainWindow; positions and orders are untouched) and logged; any other running terminal is refused. Two cases:
# * started by THIS process (its parent pid is ours): the MetaTrader5 package's initialize() launches the terminal and
#   shutdown() leaves it running -- the full suite's MT5-reachability checks do exactly that at collection (S6 DEV-20);
# * the relaunch seen on this machine 3-6 s after one of ARQVIO's own tester terminals exits cleanly, whenever ARQVIO
#   does not start the next one first (S6 DEV-18/19): created within RELAUNCH_WINDOW_S of the recorded own exit.
RELAUNCH_WINDOW_S = 30.0
EXIT_STATE = "last_terminal_exit.json"
RELAUNCH_LOG = "relaunch_absorbed.log"


def _record_exit(pid: int) -> None:
    (arqvio_files_dir() / EXIT_STATE).write_text(json.dumps({"pid": pid, "exit_ms": int(time.time() * 1000)}),
                                                 encoding="ascii")


def _own_plain_terminal(r: dict) -> str | None:
    """Why this running terminal is ARQVIO's own to close, or None."""
    if "/config" in (r.get("CommandLine") or "").lower():
        return None
    if r.get("ParentProcessId") is not None and int(r["ParentProcessId"]) == os.getpid():
        return f"started by this process (pid {os.getpid()}, MetaTrader5 initialize)"
    state = arqvio_files_dir() / EXIT_STATE
    if not state.is_file() or r.get("CreatedMs") is None:
        return None
    own = json.loads(state.read_text(encoding="ascii"))
    delta = (int(r["CreatedMs"]) - int(own["exit_ms"])) / 1000.0
    if 0.0 <= delta <= RELAUNCH_WINDOW_S:
        return f"created {delta:.1f} s after own pid {own['pid']} exited"
    return None


def _close_gracefully(pid: int) -> bool:
    # A seconds-old terminal may not have its main window yet: ask once a second, for at most 60 s.
    cmd = (f"$p = Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; if (-not $p) {{ exit 0 }}; "
           "for ($i = 0; $i -lt 60; $i++) { [void]$p.CloseMainWindow(); if ($p.WaitForExit(1000)) { exit 0 } }; exit 3")
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], check=False).returncode == 0


def run_terminal(ini: Path, *, timeout_s: float) -> tuple[int, float]:
    running = terminal_processes()
    if running:
        own = [(r, _own_plain_terminal(r)) for r in running]
        if all(why is not None for _, why in own):
            for r, why in own:
                ok = _close_gracefully(int(r["ProcessId"]))
                with (arqvio_files_dir() / RELAUNCH_LOG).open("a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.now(tz=UTC).isoformat()} own plain terminal pid {r['ProcessId']} cmd "
                             f"{r['CommandLine']!r} {why}; closed gracefully={ok}\n")
                if not ok:
                    raise TesterError(f"ARQVIO's own plain terminal {r['ProcessId']} did not close; refusing")
            running = terminal_processes()
    if running:
        listed = "; ".join(f"{r['ProcessId']} {r['CommandLine']}" for r in running)
        raise TesterError(f"a terminal of this install is running ({listed}); close it before a tester run")
    started = time.time()
    proc = subprocess.Popen([str(terminal_exe()), f"/config:{ini}"])
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()  # our own child, started above; nothing else is touched
        proc.wait(timeout=60)
        _record_exit(proc.pid)
        raise TesterError(f"tester run over its {timeout_s:.0f} s wall cap: pid {proc.pid} stopped") from None
    _record_exit(proc.pid)
    return rc, time.time() - started


def _agent_log() -> Path:
    appdata = Path(os.environ["APPDATA"])
    logs = appdata / "MetaQuotes" / "Tester" / data_dir().name / "Agent-127.0.0.1-3000" / "logs"
    return logs / f"{datetime.now(tz=UTC).strftime('%Y%m%d')}.log"


def place_ea(ea: GeneratedEA, *, log_dir: Path) -> CompileResult:
    """Write the EA into the terminal's Experts tree and compile it; anything but 0 errors / 0 warnings raises."""
    d = mql5_dir() / "Experts" / "ARQVIO_S6" / "parity"
    d.mkdir(parents=True, exist_ok=True)
    src = d / f"{ea.name}.mq5"
    src.write_text(ea.source, encoding="utf-8", newline="\n")
    return require_clean(compile_mq5(src, log_path=log_dir / f"{ea.name}.compile.log"))


def tester_ini(ea: GeneratedEA, cell: CellConfig, run_id: str) -> str:
    first = datetime.fromtimestamp(cell.first_time, tz=UTC).strftime("%Y.%m.%d")
    last = datetime.fromtimestamp(cell.last_time + 86400, tz=UTC).strftime("%Y.%m.%d")
    return "\r\n".join([
        "[Tester]", f"Expert=ARQVIO_S6\\parity\\{ea.name}", f"ExpertParameters={run_id}.set",
        f"Symbol={cell.replica_names[0]}", "Period=H1", "Model=1", "ExecutionMode=0", "Optimization=0",
        "ForwardMode=0", f"FromDate={first}", f"ToDate={last}", f"Deposit={DEPOSIT}", "Currency=USD",
        f"Leverage={LEVERAGE}", f"Report=ARQVIO_S6\\reports\\{run_id}", "ReplaceReport=1", "Visual=0",
        "ShutdownTerminal=1", "",
    ])


def run_ea(ea: GeneratedEA, cell: CellConfig, *, run_id: str, log_dir: Path, timeout_s: float = 300.0,
           kill_after: int | None = None) -> TesterRun:
    log_dir.mkdir(parents=True, exist_ok=True)
    compiled = place_ea(ea, log_dir=log_dir)
    kill_file = f"ARQVIO_S6\\kill_{run_id}"
    kill_path = arqvio_files_dir() / f"kill_{run_id}"
    if kill_path.exists():
        kill_path.unlink()
    if kill_after is not None:
        kill_path.write_text(f"after={int(kill_after)}\n", encoding="ascii")
    sets = mql5_dir() / "Profiles" / "Tester"
    sets.mkdir(parents=True, exist_ok=True)
    # LF only: the tester keeps a CR as part of the value (S6 first parity run: "astra-mql5-v1\r").
    (sets / f"{run_id}.set").write_bytes(
        f"InpTemplateVersion={ea.template_version}\nInpRunId={run_id}\nInpKillFile={kill_file}\n".encode("ascii"))
    ini = log_dir / f"{run_id}.ini"
    ini.write_text(tester_ini(ea, cell, run_id), encoding="ascii")
    run_dir = arqvio_files_dir() / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (data_dir() / "ARQVIO_S6" / "reports").mkdir(parents=True, exist_ok=True)
    agent = _agent_log()
    agent_before = agent.stat().st_size if agent.exists() else 0
    rc, wall = run_terminal(ini, timeout_s=timeout_s)
    agent_lines = ""
    if agent.exists():
        with agent.open("rb") as fh:
            fh.seek(agent_before)
            agent_lines = fh.read().decode("utf-16", errors="replace")
    (log_dir / f"{run_id}.agent.log").write_text(agent_lines, encoding="utf-8")
    missing = [f for f in JOURNAL_FILES if not (run_dir / f).is_file()]
    if missing:
        tail = "\n".join(agent_lines.splitlines()[-25:])
        raise TesterError(f"run {run_id}: journal files missing {missing}; agent log tail:\n{tail}")
    report = data_dir() / "ARQVIO_S6" / "reports" / f"{run_id}.htm"
    if not report.is_file():
        raise TesterError(f"run {run_id}: no tester report at {report}")
    if kill_path.exists():
        kill_path.unlink()
    return TesterRun(run_id, ea.name, cell.name, run_dir, report, log_dir / f"{run_id}.agent.log", wall, rc,
                     compiled)


__all__ = ["JOURNAL_FILES", "TesterError", "TesterRun", "place_ea", "run_ea", "run_terminal", "terminal_processes",
           "tester_ini"]
