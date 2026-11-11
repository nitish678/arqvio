"""Add market API (``/api/v2/app/markets``): a new QUICK cell from the owner's MetaTrader 5, in the app.

One run at a time, in a background thread of this sidecar (a pull of years of M5 bars takes a while); its state is
``<ARQVIO_HOME>/artifacts/cells/add_market_job.json``: RUNNING, DONE (with the new cell) or REFUSED (with the step
and the reason, verbatim). The run reads market data only (engine/research/cells/add_market.py).
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.core.paths import get_artifacts_dir
from engine.research.cells import add_market
from engine.research.registry import cells as registry
from engine.research.registry.annualisation import TF_MINUTES

router = APIRouter(prefix="/api/v2/app/markets", tags=["app-markets"])
_lock = threading.Lock()
_thread: dict[str, threading.Thread] = {}


def job_path() -> Path:
    return get_artifacts_dir() / "cells" / "add_market_job.json"


def _write(doc: dict[str, Any]) -> None:
    p = job_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _job() -> dict[str, Any] | None:
    p = job_path()
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    t = _thread.get("job")
    if doc.get("state") == "RUNNING" and not (t and t.is_alive()):
        doc["state"] = "INTERRUPTED"  # the sidecar stopped while it ran
    return doc


def _known_symbols() -> list[str]:
    """Symbols this home already has specs and a spread matrix for (an add of one of these skips steps 1-2)."""
    from engine.data.spread_calibration import spread_dir
    from engine.research.data.symbol_specs import specs_dir

    sp = {p.stem for p in spread_dir().glob("*.json")} if spread_dir().is_dir() else set()
    sc = {p.stem for p in specs_dir().glob("*.json")} if specs_dir().is_dir() else set()
    return sorted(sp & sc)


@router.get("")
def overview() -> dict[str, Any]:
    try:
        research = registry.research_entries()
    except registry.RegistryError as exc:
        raise HTTPException(status_code=409, detail=f"REFUSED: {exc}") from exc
    quick = [{"cell_id": e["cell_id"], "symbol": e["symbol"], "timeframe": e["timeframe"],
              "bars": e["dataset_window"]["bars"], "start": e["dataset_window"]["start"],
              "end": e["dataset_window"]["end"], "cross_cell": e["cross_cell"], "entered_at": e["entered_at"],
              "campaign_status": e["campaign_status"]} for e in research if e["status"] == registry.QUICK]
    return {"quick_cells": quick, "research_cells": [e["cell_id"] for e in research],
            "timeframes": sorted(TF_MINUTES, key=TF_MINUTES.get), "known_symbols": _known_symbols(),
            "min_bars": add_market.MIN_BARS, "job": _job(),
            "label": "QUICK cells are pinned in the app from your terminal's data; the repo's wizard ceremony did "
                     "not prove them"}


class Add(BaseModel):
    symbol: str
    timeframe: str
    start: str
    end: str
    cross_cell: str


def _run(body: Add) -> None:
    import MetaTrader5 as mt5

    from engine.research.data.mt5_terminal import resolve_terminal

    started = add_market.utc_now()
    try:
        terminal = resolve_terminal().path
        record = add_market.run(body.symbol, body.timeframe, body.start, body.end, body.cross_cell, mt5=mt5,
                                terminal=terminal)
        doc = {"state": "DONE", "request": body.model_dump(), "started_utc": started, "utc": add_market.utc_now(),
               "cell_id": record["cell_id"], "bars": record["window"]["bars"], "dataset_hash16": record["dataset_hash16"],
               "specs": record["specs"]["action"], "spread": record["spread"]["action"]}
    except add_market.AddMarketRefused as exc:
        doc = {"state": "REFUSED", "request": body.model_dump(), "started_utc": started, "utc": add_market.utc_now(),
               "step": exc.step, "reason": exc.detail}
    except Exception as exc:  # recorded verbatim: the screen shows what failed
        doc = {"state": "FAILED", "request": body.model_dump(), "started_utc": started, "utc": add_market.utc_now(),
               "reason": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
    with _lock:
        _write(doc)


@router.post("/add")
def add(body: Add) -> dict[str, Any]:
    with _lock:
        t = _thread.get("job")
        if t and t.is_alive():
            raise HTTPException(status_code=409, detail="REFUSED: an Add market run is in progress; one at a time")
        _write({"state": "RUNNING", "request": body.model_dump(), "started_utc": add_market.utc_now()})
        t = threading.Thread(target=_run, args=(body,), name="add-market", daemon=True)
        _thread["job"] = t
        t.start()
    return {"state": "RUNNING", "cell_id": f"{body.symbol.lower()}_{body.timeframe.lower()}"}


@router.delete("/{cell_id}")
def remove(cell_id: str) -> dict[str, Any]:
    with _lock:
        try:
            add_market.remove_cell(cell_id)
        except add_market.AddMarketRefused as exc:
            raise HTTPException(status_code=404, detail=exc.detail) from exc
    return {"removed": cell_id, "note": "the pinned bars, specs and spread files stay on disk"}
