"""Every MetaTrader 5 terminal on this machine, and the account each running one is logged in to -- read live.

The app asks the owner which terminal a campaign reads its data from and which account a bot is deployed to, so it
needs the full list, not the one path ``resolve_terminal`` settles on. The list is read from the machine each time:

* installs: a running ``terminal64.exe`` process, each ``%APPDATA%\\MetaQuotes\\Terminal\\<hash>\\origin.txt`` (it
  names the install directory), ``Program Files\\*\\terminal64.exe``, the portable layout ``C:\\mt5\\<n>\\`` and the
  terminal the owner chose last (``configs/mt5.json``);
* the data directory: the install itself for a portable terminal (it holds ``MQL5``), otherwise the hash directory
  whose ``origin.txt`` names the install;
* the account: only a *running* terminal is asked (``MetaTrader5.initialize(path=...)`` attaches to it; asking a
  closed one would start it). A closed terminal shows the server its ``config/common.ini`` remembers, nothing more.

Nothing here comes from an old snapshot: an account removed from MT5 is gone from the next read.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from engine.research.data.mt5_terminal import CONFIG_PATH, EXE_NAME

# The MetaTrader5 package talks to one terminal per process at a time: every caller in the engine takes this lock.
MT5_LOCK = threading.RLock()
ATTACH_TIMEOUT_MS = 8000
log = logging.getLogger(__name__)

# The main pairs the campaign panel shows as checkboxes, with the names brokers give them (a broker suffix such as
# Exness's "m" or ".pro" is matched on top). Order is the panel's order.
MAIN_PAIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("EURUSD", ("EURUSD",)), ("GBPUSD", ("GBPUSD",)), ("USDJPY", ("USDJPY",)), ("USDCHF", ("USDCHF",)),
    ("AUDUSD", ("AUDUSD",)), ("USDCAD", ("USDCAD",)), ("NZDUSD", ("NZDUSD",)), ("EURGBP", ("EURGBP",)),
    ("EURJPY", ("EURJPY",)), ("GBPJPY", ("GBPJPY",)), ("XAUUSD", ("XAUUSD", "GOLD")),
    ("XAGUSD", ("XAGUSD", "SILVER")), ("US30", ("US30", "DJ30", "WS30", "DOW30")),
    ("US500", ("US500", "SPX500", "SP500")), ("US100", ("USTEC", "US100", "NAS100", "NDX100")),
    ("GER40", ("DE40", "GER40", "DE30", "GER30")), ("USOIL", ("USOIL", "WTI", "XTIUSD", "USOUSD")),
)
_SUFFIX_MAX = 5
# Portable layouts scanned for terminal64.exe one level down (the layout this project uses).
PORTABLE_ROOTS: tuple[Path, ...] = (Path("C:/mt5"),)


@dataclass(frozen=True)
class Terminal:
    id: str
    path: str
    install_dir: str
    data_dir: str | None
    portable: bool
    running: bool
    pids: tuple[int, ...]
    remembered_server: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pids"] = list(self.pids)
        return d


def terminal_id(path: str | Path) -> str:
    return hashlib.sha1(os.path.normcase(str(Path(path).resolve())).encode("utf-8")).hexdigest()[:12]


def running_terminals() -> dict[str, list[int]]:
    """normcase(exe path) -> PIDs of every running terminal64.exe (no subprocess: psapi + kernel32)."""
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    psapi, k32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = wintypes.HANDLE
    pids = (wintypes.DWORD * 8192)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
        return {}
    out: dict[str, list[int]] = {}
    for pid in pids[: needed.value // ctypes.sizeof(wintypes.DWORD)]:
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            continue
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)) and \
                    buf.value.lower().endswith("\\" + EXE_NAME):
                out.setdefault(os.path.normcase(buf.value), []).append(int(pid))
        finally:
            k32.CloseHandle(h)
    return out


def _read_utf16(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-16", "utf-8-sig"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""


def _appdata_dirs() -> dict[str, Path]:
    """normcase(install dir) -> its %APPDATA% data directory."""
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) / "MetaQuotes" / "Terminal" if appdata else None
    out: dict[str, Path] = {}
    if root is None or not root.is_dir():
        return out
    for child in root.iterdir():
        origin = child / "origin.txt"
        if origin.is_file():
            install = _read_utf16(origin).strip().strip("\ufeff")
            if install:
                out[os.path.normcase(str(Path(install)))] = child
    return out


def _candidates(appdata: dict[str, Path], running: dict[str, list[int]]) -> list[Path]:
    seen: dict[str, Path] = {}

    def add(p: Path) -> None:
        if p.is_file():
            seen.setdefault(os.path.normcase(str(p.resolve())), p.resolve())

    for exe in running:
        add(Path(exe))
    for install in appdata:
        add(Path(install) / EXE_NAME)
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if base and Path(base).is_dir():
            for exe in Path(base).glob(f"*/{EXE_NAME}"):
                add(exe)
    for root in PORTABLE_ROOTS:
        if root.is_dir():
            for exe in root.glob(f"*/{EXE_NAME}"):
                add(exe)
    chosen = chosen_path()
    if chosen:
        add(Path(chosen))
    return sorted(seen.values(), key=lambda p: str(p).lower())


def _remembered_server(data_dir: Path | None) -> str | None:
    ini = data_dir / "config" / "common.ini" if data_dir else None
    if ini is None or not ini.is_file():
        return None
    for line in _read_utf16(ini).splitlines():
        if line.startswith("Server="):
            return line.split("=", 1)[1].strip() or None
    return None


def discover() -> list[Terminal]:
    """Every terminal install on this machine, read now."""
    appdata, running = _appdata_dirs(), running_terminals()
    out: list[Terminal] = []
    for exe in _candidates(appdata, running):
        install = exe.parent
        portable = (install / "MQL5").is_dir()
        data_dir = install if portable else appdata.get(os.path.normcase(str(install)))
        pids = tuple(running.get(os.path.normcase(str(exe)), ()))
        out.append(Terminal(id=terminal_id(exe), path=str(exe), install_dir=str(install),
                            data_dir=str(data_dir) if data_dir else None, portable=portable, running=bool(pids),
                            pids=pids, remembered_server=_remembered_server(data_dir)))
    return out


def find(terminal: str) -> Terminal:
    """A terminal by id or by exe path; KeyError when it is not on this machine."""
    for t in discover():
        if terminal in (t.id, t.path) or os.path.normcase(terminal) == os.path.normcase(t.path):
            return t
    raise KeyError(terminal)


def _mt5() -> Any:
    import MetaTrader5 as mt5  # noqa: N813

    return mt5


def _parent_pids() -> dict[int, int]:
    """PID -> parent PID of every process (a toolhelp snapshot; no subprocess)."""
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    k32 = ctypes.WinDLL("kernel32")
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == ctypes.c_void_p(-1).value:
        return {}
    out: dict[int, int] = {}
    try:
        e = Entry()
        e.dwSize = ctypes.sizeof(Entry)
        ok = k32.Process32FirstW(wintypes.HANDLE(snap), ctypes.byref(e))
        while ok:
            out[int(e.th32ProcessID)] = int(e.th32ParentProcessID)
            ok = k32.Process32NextW(wintypes.HANDLE(snap), ctypes.byref(e))
    finally:
        k32.CloseHandle(wintypes.HANDLE(snap))
    return out


def started_by_us(path: str) -> list[int]:
    """PIDs of this install's terminal64.exe whose parent is this process -- a terminal ``mt5.initialize`` started."""
    pids = running_terminals().get(os.path.normcase(str(Path(path))), [])
    if not pids:
        return []
    parents, me = _parent_pids(), os.getpid()
    return sorted(p for p in pids if parents.get(p) == me)


CLOSE_WAIT_S = 60.0


def _alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not h:
        return False
    try:
        return k32.WaitForSingleObject(wintypes.HANDLE(h), 0) == 0x102  # WAIT_TIMEOUT: still running
    finally:
        k32.CloseHandle(wintypes.HANDLE(h))


def _post_close(pid: int) -> None:
    """WM_CLOSE to every visible top-level window of ``pid``: MetaTrader closes the way its own Exit does."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd: int, _: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(wintypes.HWND(hwnd)):
            user32.PostMessageW(wintypes.HWND(hwnd), 0x0010, 0, 0)  # WM_CLOSE
        return True

    user32.EnumWindows(each, 0)


def close_started(pids: list[int]) -> None:
    """Close, in the background, terminals a read started by accident; one that will not close is left and logged."""
    def run() -> None:
        left, deadline = set(pids), time.monotonic() + CLOSE_WAIT_S
        while left and time.monotonic() < deadline:
            for pid in sorted(left):
                if _alive(pid):
                    _post_close(pid)
                else:
                    left.discard(pid)
                    log.info("closed MetaTrader 5 pid %s that a read had started", pid)
            if left:
                time.sleep(1.0)
        for pid in sorted(left):
            log.warning("MetaTrader 5 pid %s that a read started did not close in %.0f s", pid, CLOSE_WAIT_S)

    threading.Thread(target=run, name="arqvio-close-started-terminal", daemon=True).start()


def _attach(t: Terminal, mt5: Any) -> bool | None:
    """``mt5.initialize`` on a terminal the listing saw running. None when the terminal had closed in between.

    ``initialize(path=...)`` STARTS a terminal that is not running. ``t.running`` was read a moment earlier, and a
    terminal the owner is closing still has its process for a few seconds -- so a read that lands in that gap opened
    MT5 again (owner's machine, 2026-10-26: the tester terminal came back 23 s after it was closed, parent
    arqvio-engine.exe). A terminal of this install that is our child after the call and was not before was started
    by the call: it is closed again and the read answers as for a closed terminal."""
    before = set(started_by_us(t.path))
    ok = bool(mt5.initialize(path=t.path, timeout=ATTACH_TIMEOUT_MS))
    started = [p for p in started_by_us(t.path) if p not in before]
    if started:
        if ok:
            mt5.shutdown()
        close_started(started)
        return None
    return ok


def _fields(obj: Any) -> dict[str, Any]:
    return dict(obj._asdict()) if hasattr(obj, "_asdict") else {}


MODES = {0: "DEMO", 1: "CONTEST", 2: "REAL"}


def account(t: Terminal, *, mt5: Any | None = None) -> dict[str, Any]:
    """The account the running terminal is logged in to, read now; {"state": ...} says why when there is none."""
    if not t.running:
        return {"state": "CLOSED", "detail": "MetaTrader 5 is not running; open it to read its account"}
    mt5 = mt5 or _mt5()
    with MT5_LOCK:
        ok = _attach(t, mt5)
        if ok is None:
            return {"state": "CLOSED", "detail": "MetaTrader 5 was closed while ARQVIO read it; open it to read its account"}
        if not ok:
            return {"state": "UNREACHABLE", "detail": f"the terminal did not answer: {mt5.last_error()}"}
        try:
            a, ti = mt5.account_info(), mt5.terminal_info()
        finally:
            mt5.shutdown()
    info = _fields(ti)
    if a is None:
        return {"state": "LOGGED_OUT", "detail": "the terminal is open but not logged in to an account",
                "connected": bool(info.get("connected"))}
    f = _fields(a)
    return {"state": "OK", "login": int(f.get("login", 0)), "server": f.get("server"), "company": f.get("company"),
            "name": f.get("name"), "currency": f.get("currency"), "balance": f.get("balance"),
            "equity": f.get("equity"), "leverage": f.get("leverage"),
            "mode": MODES.get(int(f.get("trade_mode", 0)), "DEMO"), "connected": bool(info.get("connected")),
            "algo_trading": bool(info.get("trade_allowed")), "max_bars": info.get("maxbars")}


def symbols(t: Terminal, *, mt5: Any | None = None) -> list[dict[str, Any]]:
    """Every symbol the running terminal offers: name, description, its folder and whether Market Watch shows it."""
    if not t.running:
        raise RuntimeError("MetaTrader 5 is not running; open it to list its symbols")
    mt5 = mt5 or _mt5()
    with MT5_LOCK:
        ok = _attach(t, mt5)
        if ok is None:
            raise RuntimeError("MetaTrader 5 was closed while ARQVIO read it; open it to list its symbols")
        if not ok:
            raise RuntimeError(f"the terminal did not answer: {mt5.last_error()}")
        try:
            rows = mt5.symbols_get() or ()
        finally:
            mt5.shutdown()
    out = []
    for s in rows:
        f = _fields(s)
        out.append({"name": f.get("name"), "description": f.get("description") or "", "path": f.get("path") or "",
                    "visible": bool(f.get("visible"))})
    return sorted(out, key=lambda r: r["name"])


def main_pairs(names: list[str] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each main pair with the broker's name for it (None when the terminal does not offer it)."""
    rows = [n if isinstance(n, dict) else {"name": n, "visible": False} for n in names]
    out = []
    for base, aliases in MAIN_PAIRS:
        matches = [r for r in rows if any(_is_alias(str(r["name"]).upper(), a) for a in aliases)]
        # Market Watch first, then the shortest name (EURUSDm before EURUSDm_raw)
        best = min(matches, key=lambda r: (not r.get("visible"), len(str(r["name"])), str(r["name"])), default=None)
        out.append({"pair": base, "symbol": best["name"] if best else None})
    return out


_LETTER_SUFFIXES = {"", "M", "C", "I", "Z", "X", "PRO", "ECN", "RAW", "MICRO", "MINI", "CASH"}


def _is_alias(name: str, alias: str) -> bool:
    """EURUSD, EURUSDm, EURUSD.pro, EURUSD#, EURUSD_i -- but not EURUSDJPY or GOLDEUR."""
    if not name.startswith(alias):
        return False
    rest = name[len(alias):]
    return rest in _LETTER_SUFFIXES or (len(rest) <= _SUFFIX_MAX and not rest[0].isalnum())


def chosen_path() -> str | None:
    import json

    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("terminal_path") or None
    except (OSError, ValueError):
        return None


__all__ = ["MAIN_PAIRS", "MT5_LOCK", "Terminal", "account", "chosen_path", "discover", "find", "main_pairs",
           "running_terminals", "symbols", "terminal_id"]
