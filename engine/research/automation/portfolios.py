"""Named portfolios: a team of vault strategies with a name (research: + NEW PORTFOLIO, ADD TO PORTFOLIO, the
automation PORTFOLIO step). One JSON file under the data folder, written atomically; a member is a vault id plus
who added it and when, so an automation can remove only what it added itself.
idea: research RF 437-439, PN 367/471, automation PORTFOLIO step (RF 47334-47455); code ARQVIO's.
"""
from __future__ import annotations

import json
import re
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.core import atomic
from engine.core.paths import get_artifacts_dir

SCHEMA = "arqvio.portfolios"
VERSION = 1
NAME_MAX = 48
_lock = threading.RLock()


class PortfolioError(ValueError):
    """A refused portfolio change; the message says why."""


def path() -> Path:
    return get_artifacts_dir() / "portfolios.json"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def load() -> list[dict[str, Any]]:
    try:
        doc = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(doc.get("portfolios") or []) if doc.get("schema") == SCHEMA else []


def _save(rows: list[dict[str, Any]]) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": SCHEMA, "version": VERSION, "portfolios": rows}, indent=2) + "\n",
                   encoding="utf-8")
    atomic.replace(tmp, p)


def _clean_name(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name or "")).strip()
    if not n:
        raise PortfolioError("name the portfolio")
    if len(n) > NAME_MAX:
        raise PortfolioError(f"a portfolio name is at most {NAME_MAX} characters")
    return n


def get(pid_or_name: str) -> dict[str, Any]:
    for p in load():
        if p["id"] == pid_or_name or p["name"].lower() == str(pid_or_name).lower():
            return p
    raise PortfolioError(f"no portfolio {pid_or_name!r}")


def create(name: str) -> dict[str, Any]:
    with _lock:
        rows = load()
        n = _clean_name(name)
        if any(p["name"].lower() == n.lower() for p in rows):
            raise PortfolioError(f"a portfolio named {n!r} already exists")
        p = {"id": f"pf{secrets.token_hex(4)}", "name": n, "created_utc": _now(), "members": []}
        rows.append(p)
        _save(rows)
        return p


def rename(pid: str, name: str) -> dict[str, Any]:
    with _lock:
        rows = load()
        n = _clean_name(name)
        if any(p["name"].lower() == n.lower() and p["id"] != pid for p in rows):
            raise PortfolioError(f"a portfolio named {n!r} already exists")
        for p in rows:
            if p["id"] == pid:
                p["name"] = n
                _save(rows)
                return p
        raise PortfolioError(f"no portfolio {pid!r}")


def delete(pid: str) -> None:
    """The portfolio only: its strategies stay in the vault, its deployed bots keep running."""
    with _lock:
        rows = load()
        keep = [p for p in rows if p["id"] != pid]
        if len(keep) == len(rows):
            raise PortfolioError(f"no portfolio {pid!r}")
        _save(keep)


def add(pid: str, ids: list[str], *, by: str = "you") -> dict[str, Any]:
    with _lock:
        rows = load()
        for p in rows:
            if p["id"] == pid:
                have = {m["id"] for m in p["members"]}
                for i in ids:
                    if i not in have:
                        p["members"].append({"id": i, "added_utc": _now(), "by": by})
                        have.add(i)
                _save(rows)
                return p
        raise PortfolioError(f"no portfolio {pid!r}")


def remove(pid: str, ids: list[str]) -> dict[str, Any]:
    with _lock:
        rows = load()
        for p in rows:
            if p["id"] == pid:
                drop = set(ids)
                p["members"] = [m for m in p["members"] if m["id"] not in drop]
                _save(rows)
                return p
        raise PortfolioError(f"no portfolio {pid!r}")


def protected_ids() -> set[str]:
    """Every strategy in any portfolio: purges never touch them (research: portfolio members are never purged)."""
    return {m["id"] for p in load() for m in p["members"]}


def ensure(name: str) -> dict[str, Any]:
    """The portfolio with this name, created when missing (the automation's + NEW PORTFOLIO)."""
    try:
        return get(name)
    except PortfolioError:
        return create(name)


__all__ = ["PortfolioError", "add", "create", "delete", "ensure", "get", "load", "protected_ids", "remove", "rename"]
