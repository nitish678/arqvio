"""Boot checks (``GET /api/v2/app/boot``): what the app's boot screen shows, one line per real check.

ARQVIO_V4.md section 12.2: every boot line is the result of a check run now -- the engine answering, its runtime and
backtest kernel, the data folder and disk, the MetaTrader 5 terminals on this machine, the vault, the gene registry, the
launch queue, the deployed bots, a launch still running (which the reopened app reattaches to). Lines carry a group
(engine / environment / library / session) so the boot screen reads as the start-up sequence it is. A check that fails
says why and is never shown as passed; nothing here is decorative.
Reads only, except a probe file written and removed in the data folder.
"""
from __future__ import annotations

import os
import platform
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from engine import __version__
from engine.core.paths import get_arqvio_home

router = APIRouter(prefix="/api/v2/app", tags=["app-boot"])

STARTED = time.monotonic()


def _check(group: str, name: str, fn) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as exc:  # the line says what broke; the boot screen turns it red
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    return {"group": group, "check": name, "ok": bool(ok), "detail": str(detail),
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


def _runtime() -> tuple[bool, str]:
    from engine.api.routes_app_support import _memory

    mem = _memory()
    ram = f", {mem['available_gb']:.1f} of {mem['total_gb']:.1f} GB memory free" if mem.get("total_gb") else ""
    return True, f"Python {platform.python_version()}, {os.cpu_count() or '?'} logical cores{ram}"


def _kernel() -> tuple[bool, str]:
    # 2026-11-01: importing the kernel only registers its jitted functions (numba compiles on first use), so the line
    # reports whether a compiled cache already sits next to it instead of compiling during boot.
    import numba

    from engine.research.backtest import kernel

    cache = Path(kernel.__file__).parent / "__pycache__"
    cached = len(list(cache.glob("kernel.*.nbi"))) if cache.is_dir() else 0
    state = (f"{cached} compiled function{'s' if cached != 1 else ''} cached" if cached
             else "not compiled yet: the first campaign compiles it")
    return True, f"numba {numba.__version__}, {state}"


def _data_folder() -> tuple[bool, str]:
    home = get_arqvio_home()
    home.mkdir(parents=True, exist_ok=True)
    probe = home / f".boot-{uuid.uuid4().hex}"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return True, f"{home} (writable)"


def _disk() -> tuple[bool, str]:
    du = shutil.disk_usage(get_arqvio_home())
    free = du.free / 2**30
    if free < 2:
        return False, f"only {free:.1f} GB free: campaigns write bars and caches, free some space"
    return True, f"{free:.1f} GB free of {du.total / 2**30:.0f} GB"


def _terminals() -> tuple[bool, str]:
    from engine.research.data import terminals as mt5_terminals

    ts = mt5_terminals.discover()
    running = sum(1 for t in ts if t.running)
    if not ts:
        return False, "no MetaTrader 5 terminal found (data and deploy need one)"
    return True, f"{len(ts)} terminal{'s' if len(ts) != 1 else ''} found, {running} running"


def _vault() -> tuple[bool, str]:
    from engine.research.vault import library

    rows = library.load()
    certified = sum(1 for r in rows if r.get("tier") == "CERTIFIED")
    v1 = sum(1 for r in rows if r.get("engine") == "v1")
    return True, f"{len(rows)} strateg{'ies' if len(rows) != 1 else 'y'}, {certified} certified, {v1} from engine V1"


def _genes() -> tuple[bool, str]:
    from engine.research.genes import registry

    c = registry.counts()
    problems = registry.problems()
    if problems:
        return False, "; ".join(problems[:3])
    return True, f"{c['total']} genes, {c['by_status'].get('READY', 0)} ready to breed"


def _queue() -> tuple[bool, str]:
    from engine.api import routes_app_launch as launch

    n = len(launch.queue_items())
    return True, (f"{n} campaign{'s' if n != 1 else ''} waiting" if n else "empty")


def _bots() -> tuple[bool, str]:
    from engine.research.deploy import full_deploy

    bots = [b for b in full_deploy.load_bots() if b.get("status") != "REMOVED"]
    return True, (f"{len(bots)} bot{'s' if len(bots) != 1 else ''} deployed" if bots else "none deployed")


def _launch() -> tuple[bool, str]:
    from engine.api import routes_app_launch as launch

    return True, ("running -- reattached" if launch.busy() else "none running")


@router.get("/boot")
def boot() -> dict[str, Any]:
    checks = [
        {"group": "engine", "check": "engine", "ok": True,
         "detail": f"ARQVIO engine {__version__} on 127.0.0.1:8741, up {time.monotonic() - STARTED:.1f} s", "ms": 0.0},
        _check("engine", "runtime", _runtime),
        _check("engine", "backtest kernel", _kernel),
        _check("environment", "data folder", _data_folder),
        _check("environment", "disk", _disk),
        _check("environment", "MetaTrader 5", _terminals),
        _check("library", "vault", _vault),
        _check("library", "gene registry", _genes),
        _check("library", "launch queue", _queue),
        _check("library", "deployed bots", _bots),
        _check("session", "campaign", _launch),
    ]
    # no MetaTrader 5 or a low disk is a warning, not a stop: the vault, guide and settings still work
    return {"version": __version__, "checks": checks,
            "ok": all(c["ok"] for c in checks if c["check"] not in ("MetaTrader 5", "disk"))}


__all__ = ["boot", "router"]
