"""Hard suspend / resume of a process and every process under it (Windows; idea: research APP-2 "pause = hard
suspend/resume of the engine + worker process tree (ctypes-level on Windows), cooperative pause is NOT the design").

A V1 breeding run has no pause points of its own and its worker pool lives in child processes, so PAUSE freezes the
whole tree where it stands and RESUME lets it run on -- nothing is lost and nothing re-runs. ctypes only (no new
dependency): a Toolhelp32 snapshot names the children, ntdll's NtSuspendProcess / NtResumeProcess act on each.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

PROCESS_SUSPEND_RESUME = 0x0800
TH32CS_SNAPPROCESS = 0x00000002


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def _children() -> dict[int, list[int]]:
    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap in (-1, 0xFFFFFFFF, None):
        raise OSError("CreateToolhelp32Snapshot failed")
    out: dict[int, list[int]] = {}
    try:
        e = _PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            out.setdefault(int(e.th32ParentProcessID), []).append(int(e.th32ProcessID))
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    return out


def tree(pid: int) -> list[int]:
    """``pid`` and every process under it, parents first."""
    kids = _children()
    out, todo, seen = [], [int(pid)], set()
    while todo:
        p = todo.pop(0)
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def _act(pid: int, fn_name: str) -> int:
    if os.name != "nt":
        raise OSError("hard suspend is implemented for Windows only")
    k32, nt = ctypes.windll.kernel32, ctypes.windll.ntdll
    done = 0
    for p in tree(pid):
        h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, p)
        if not h:
            continue
        try:
            if getattr(nt, fn_name)(h) == 0:
                done += 1
        finally:
            k32.CloseHandle(h)
    return done


def suspend(pid: int) -> int:
    """Suspend ``pid`` and its tree; returns how many processes were suspended."""
    return _act(pid, "NtSuspendProcess")


def resume(pid: int) -> int:
    """Resume ``pid`` and its tree; returns how many processes were resumed."""
    return _act(pid, "NtResumeProcess")


__all__ = ["resume", "suspend", "tree"]
