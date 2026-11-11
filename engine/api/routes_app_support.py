"""Diagnostics and the support report (``/api/v2/app/diagnostics``, ``/api/v2/app/support-report``; worker4 W4-9 9.7,
idea: research APP-8 "diagnostics card (RAM, disk, generations, errors) + support report builder, one-click COPY").

Every number is read from this machine or the engine's own files when asked. The report is plain text the owner
pastes to whoever helps them; long digit runs (an account login, a phone number) are masked before it leaves the
engine, except ARQVIO's own magic numbers (610500000 and up) which identify bots, not people.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import sys
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from engine.core import applog
from engine.core.paths import get_arqvio_home

router = APIRouter(prefix="/api/v2/app", tags=["app-support"])

_DIGITS = re.compile(r"\b\d{6,}\b")
MAGIC_LO, MAGIC_HI = 610_500_000, 610_999_999


def _memory() -> dict[str, Any]:
    if os.name != "nt":
        return {"total_gb": None, "available_gb": None}

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return {"total_gb": None, "available_gb": None}
    return {"total_gb": round(m.ullTotalPhys / 2**30, 2), "available_gb": round(m.ullAvailPhys / 2**30, 2),
            "load_pct": int(m.dwMemoryLoad)}


def mask(text: str) -> str:
    def one(m: re.Match[str]) -> str:
        v = int(m.group(0))
        return m.group(0) if MAGIC_LO <= v <= MAGIC_HI else "#" * len(m.group(0))
    return _DIGITS.sub(one, text)


@router.get("/diagnostics")
def diagnostics() -> dict[str, Any]:
    from engine.research.vault import library

    home = get_arqvio_home()
    try:
        du = shutil.disk_usage(home)
        disk = {"path": str(home), "free_gb": round(du.free / 2**30, 2), "total_gb": round(du.total / 2**30, 2)}
    except OSError:
        disk = {"path": str(home), "free_gb": None, "total_gb": None}
    try:
        rows = library.load()
        vault = {"strategies": len(rows), **{t: sum(r.get("tier") == t for r in rows) for t in library.TIERS},
                 "engine_v1": sum(r.get("engine") == "v1" for r in rows)}
    except Exception as exc:  # a damaged vault file is a diagnosis too
        vault = {"error": f"{type(exc).__name__}: {exc}"}
    log = applog.read(limit=2000)["lines"]
    errors = [{"utc": x.get("utc"), "src": x.get("src"), "msg": mask(str(x.get("msg", "")))}
              for x in log if x.get("level") in ("ERROR", "WARN")][-15:]
    try:
        from engine.config.settings import get_settings
        version = getattr(get_settings(), "engine_version", None)
    except Exception:
        version = None
    return {"utc": datetime.now(UTC).isoformat(timespec="seconds"), "version": version,
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "python": sys.version.split()[0], "cores": os.cpu_count(), "memory": _memory(), "disk": disk,
            "vault": vault, "recent_problems": errors}


def report_text() -> str:
    d = diagnostics()
    lines = [f"ARQVIO support report · {d['utc']}", f"version {d['version']} · {d['os']} · Python {d['python']}",
             f"cores {d['cores']} · memory {d['memory']} · disk {d['disk']}", f"vault {d['vault']}", "",
             "recent problems:"]
    lines += [f"  {e['utc']} {e['src']}: {e['msg']}" for e in d["recent_problems"]] or ["  none"]
    lines += ["", "last 150 log lines:"]
    lines += [mask(f"  {x.get('utc')} {x.get('level')} {x.get('src')}: {x.get('msg')}")
              for x in applog.read(limit=150)["lines"]]
    return "\n".join(lines) + "\n"


@router.get("/support-report")
def support_report() -> dict[str, str]:
    """The report as JSON ({"text": ...}): the app copies it to the clipboard with one press."""
    return {"text": report_text()}


__all__ = ["diagnostics", "mask", "report_text", "router", "support_report"]
