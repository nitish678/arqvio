"""MetaTrader 5 terminals API (``/api/v2/app/terminals``): which terminals this machine has, the account each running
one is logged in to (read live, engine/research/data/terminals.py), the symbols one offers, and the owner's choice.

The campaign panel reads its data from the chosen terminal; the deploy dialog names the terminal (and so the
account) a bot goes to. Reads only: nothing here logs in, changes a terminal setting or places an order.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.research.data import terminals as mt5_terminals
from engine.research.data.mt5_terminal import save_terminal_path

router = APIRouter(prefix="/api/v2/app/terminals", tags=["app-terminals"])


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _find(terminal: str) -> mt5_terminals.Terminal:
    try:
        return mt5_terminals.find(terminal)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no MetaTrader 5 terminal {terminal!r} on this machine") from None


def chosen(ts: list[mt5_terminals.Terminal]) -> mt5_terminals.Terminal | None:
    """The owner's choice when it is still installed; otherwise the only running terminal, if exactly one runs."""
    path = mt5_terminals.chosen_path()
    for t in ts:
        if path and t.path.lower() == str(path).lower():
            return t
    running = [t for t in ts if t.running]
    return running[0] if len(running) == 1 else None


@router.get("")
def overview() -> dict[str, Any]:
    ts = mt5_terminals.discover()
    pick = chosen(ts)
    rows = [{**t.to_dict(), "account": mt5_terminals.account(t)} for t in ts]
    return {"terminals": rows, "chosen": pick.id if pick else None, "read_utc": _now(),
            "note": "read from this machine now; a closed terminal is not started to read its account"}


class Choose(BaseModel):
    terminal: str


@router.post("/choose")
def choose(body: Choose) -> dict[str, Any]:
    t = _find(body.terminal)
    save_terminal_path(t.path)
    return {"chosen": t.id, "path": t.path}


@router.get("/{terminal}/symbols")
def symbols(terminal: str) -> dict[str, Any]:
    t = _find(terminal)
    try:
        rows = mt5_terminals.symbols(t)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"terminal": t.id, "symbols": rows, "main_pairs": mt5_terminals.main_pairs(rows), "read_utc": _now()}
