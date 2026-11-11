"""The app's one log: everything the owner should see, in one stream, in plain words.

The app shows a single log panel (no per-source picker): data pulls, campaign progress, vault imports, deploys and
refusals all land here as one JSON line each, numbered so the panel can ask for "everything after line N".

    <home>/logs/arqvio.log      {"seq": 17, "utc": "...", "level": "INFO", "src": "data", "msg": "..."}

At ``MAX_BYTES`` the file rolls to ``arqvio.1.log`` (one generation kept); ``seq`` keeps counting across the roll.
Writing never raises: a log that cannot be written must not stop a campaign.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.core.paths import get_logs_dir

MAX_BYTES = 4_000_000
LEVELS = ("INFO", "OK", "WARN", "ERROR")
_lock = threading.Lock()
_seq: dict[str, int] = {}


def log_path() -> Path:
    return get_logs_dir() / "arqvio.log"


def _last_seq(p: Path) -> int:
    for f in (p, p.with_name("arqvio.1.log")):
        if not f.is_file():
            continue
        try:
            with f.open("rb") as h:
                h.seek(max(0, f.stat().st_size - 4096))
                tail = h.read().decode("utf-8", "replace").splitlines()
            for line in reversed(tail):
                try:
                    return int(json.loads(line)["seq"])
                except (ValueError, KeyError, TypeError):
                    continue
        except OSError:
            continue
    return 0


def write(src: str, msg: str, level: str = "INFO") -> None:
    """Append one line. ``src`` is a short word (engine, data, breed, vault, deploy, mt5); never raises."""
    try:
        p = log_path()
        with _lock:
            key = str(p)
            if key not in _seq:
                _seq[key] = _last_seq(p)
            _seq[key] += 1
            rec = {"seq": _seq[key], "utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                   "level": level if level in LEVELS else "INFO", "src": src, "msg": str(msg)}
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.is_file() and p.stat().st_size > MAX_BYTES:
                os.replace(p, p.with_name("arqvio.1.log"))
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - the log must never stop the work it records
        pass


def read(after: int = 0, limit: int = 500) -> dict[str, Any]:
    """Lines with seq > ``after`` (oldest first, at most ``limit``: the newest ones when there are more)."""
    p = log_path()
    rows: list[dict[str, Any]] = []
    for f in (p.with_name("arqvio.1.log"), p):
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and int(rec.get("seq", 0)) > after:
                rows.append(rec)
    rows = rows[-limit:]
    return {"lines": rows, "last": rows[-1]["seq"] if rows else after, "path": str(p)}


__all__ = ["LEVELS", "MAX_BYTES", "log_path", "read", "write"]
