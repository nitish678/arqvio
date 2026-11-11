"""S7 read-only API: the portfolio replay, diversity, heat, drift, the audited alert spool, the 2-account health.

Every path resolves at CALL time (engine/core/paths, CLAUDE.md s7) and every route is GET-only: S7 exposes no
control surface and no screens (S5 renders). Fail-closed: a missing artifact answers ``exists: false`` with the path
it looked for; an unreadable one answers ``exists: true`` with an ``error`` and no data -- a value is never filled in.

* /portfolio -- run_hash, inputs, members, the verification verdicts (conservation, reconciliation), refusal and
  event counts (the per-bar series stays in the artifact);
* /diversity, /heat, /drift -- the analysis artifacts as written;
* /alerts?n= -- the spool's last n lines (1..500) and its chain verification;
* /health -- the 2-account status snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from engine.core.paths import get_artifacts_dir
from engine.research.alerts.spool import spool_path, tail, verify
from engine.research.health.probe import snapshot_path

router = APIRouter(prefix="/api/v1/s7", tags=["s7"])


class Artifact(BaseModel):
    exists: bool
    path: str
    error: str | None = None
    data: Any = None


def _portfolio(name: str) -> Path:
    return get_artifacts_dir() / "portfolio" / f"{name}.json"


def _read(path: Path) -> Artifact:
    if not path.is_file():
        return Artifact(exists=False, path=str(path))
    try:
        return Artifact(exists=True, path=str(path), data=json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return Artifact(exists=True, path=str(path), error=f"unreadable: {type(exc).__name__}: {exc}")


@router.get("/portfolio", response_model=Artifact)
async def portfolio() -> Artifact:
    a = _read(_portfolio("replay"))
    if a.data is None:
        return a
    d = a.data
    kinds: dict[str, int] = {}
    for e in d.get("events", []):
        kinds[e.get("kind")] = kinds.get(e.get("kind"), 0) + 1
    return Artifact(exists=True, path=a.path, data={
        "run_hash": d.get("run_hash"), "inputs": d.get("inputs"), "members": d.get("members"),
        "verification": d.get("verification"), "refused": len(d.get("refused", {})), "event_counts": kinds,
        "bars": len(d.get("bars", [])), "final_equity": (d.get("bars") or [{}])[-1].get("equity")})


@router.get("/diversity", response_model=Artifact)
async def diversity() -> Artifact:
    return _read(_portfolio("diversity"))


@router.get("/heat", response_model=Artifact)
async def heat() -> Artifact:
    return _read(_portfolio("heat"))


@router.get("/drift", response_model=Artifact)
async def drift() -> Artifact:
    return _read(_portfolio("drift"))


class AlertsView(BaseModel):
    exists: bool
    path: str
    chain_ok: bool | None = None
    chain_lines: int = 0
    chain_reason: str = ""
    lines: list[dict] = []
    error: str | None = None


@router.get("/alerts", response_model=AlertsView)
async def alerts(n: int = Query(50, ge=1, le=500)) -> AlertsView:
    p = spool_path()
    if not p.is_file():
        return AlertsView(exists=False, path=str(p))
    try:
        rep = verify(p)
        return AlertsView(exists=True, path=str(p), chain_ok=rep.ok, chain_lines=rep.lines, chain_reason=rep.reason,
                          lines=tail(n, p))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return AlertsView(exists=True, path=str(p), error=f"unreadable: {type(exc).__name__}: {exc}")


@router.get("/health", response_model=Artifact)
async def health() -> Artifact:
    return _read(snapshot_path())


__all__ = ["router"]
