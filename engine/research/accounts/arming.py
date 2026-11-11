"""Owner arming for real-money accounts.

A DEMO account trades with no extra step. A REAL account may OPEN positions only after its owner has
armed it: one explicit, recorded act per (login, server) in ``ARQVIO_HOME/config/live_arming.json``.
The engine never arms an account by itself and never judges whether a trade is worth taking; the file
records that the owner chose to. Closing the engine's OWN positions never needs arming, so a disarm,
a kill switch or a flatten always works.

CLI (the owner's own terminal)::

    python -m engine.research.accounts.arming arm    --login 12345678 --server Exness-MT5Real8 [--note "..."]
    python -m engine.research.accounts.arming disarm --login 12345678 --server Exness-MT5Real8
    python -m engine.research.accounts.arming list
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from engine.core.paths import get_arqvio_home
from engine.core.schema import same_schema
from engine.research.mql5.records import write_json_atomic

SCHEMA = "arqvio.live_arming"
VERSION = 1


class ArmingFileError(ValueError):
    """The arming file exists but is not a valid arming record."""


def arming_path() -> Path:
    return get_arqvio_home() / "config" / "live_arming.json"


def load(path: Path | None = None) -> list[dict]:
    """The armed accounts; an absent file means none is armed."""
    path = path or arming_path()
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not same_schema(doc.get("schema"), SCHEMA) or not isinstance(doc.get("accounts"), list):
        raise ArmingFileError(f"{path} is not a {SCHEMA} record")
    for a in doc["accounts"]:
        if not isinstance(a.get("login"), int) or not isinstance(a.get("server"), str):
            raise ArmingFileError(f"{path}: every entry needs an integer login and a server name")
    return doc["accounts"]


def is_armed(login: int, server: str, path: Path | None = None) -> bool:
    return any(a["login"] == int(login) and a["server"] == server for a in load(path))


def _write(accounts: list[dict], path: Path) -> None:
    write_json_atomic(path, {"schema": SCHEMA, "version": VERSION, "accounts": accounts})


def arm(login: int, server: str, *, note: str = "", path: Path | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(tz=UTC)) -> dict:
    """Record the owner's decision to let ``login@server`` open positions. Re-arming refreshes the stamp."""
    path = path or arming_path()
    rest = [a for a in load(path) if not (a["login"] == int(login) and a["server"] == server)]
    entry = {"login": int(login), "server": server, "armed_utc": now().isoformat(), "note": note}
    _write(rest + [entry], path)
    return entry


def disarm(login: int, server: str, *, path: Path | None = None) -> bool:
    """Remove the arming; True if the account had been armed."""
    path = path or arming_path()
    accounts = load(path)
    rest = [a for a in accounts if not (a["login"] == int(login) and a["server"] == server)]
    if len(rest) == len(accounts):
        return False
    _write(rest, path)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m engine.research.accounts.arming",
                                description="Arm or disarm a real-money account for ARQVIO (owner only).")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("arm", "disarm"):
        s = sub.add_parser(name)
        s.add_argument("--login", type=int, required=True)
        s.add_argument("--server", required=True)
        if name == "arm":
            s.add_argument("--note", default="")
    sub.add_parser("list")
    args = p.parse_args(argv)
    if args.cmd == "arm":
        e = arm(args.login, args.server, note=args.note)
        print(f"ARMED login={e['login']} server={e['server']} at {e['armed_utc']} -> {arming_path()}")
    elif args.cmd == "disarm":
        print(("DISARMED" if disarm(args.login, args.server) else "NOT ARMED") + f" login={args.login} server={args.server}")
    else:
        for a in load():
            print(f"login={a['login']} server={a['server']} armed_utc={a['armed_utc']} note={a.get('note', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ArmingFileError", "SCHEMA", "arm", "arming_path", "disarm", "is_armed", "load", "main"]
