"""S6 read-only API: the generated-EA registry, parity verdicts, the DEMO worker's status.

Every path resolves at CALL time through engine/research/mql5/records.py (CLAUDE.md s7), and every
route is GET-only: S6 exposes no control surface. Fail-closed: a missing artifact answers
``exists: false`` with the path it looked for; an unreadable one answers ``exists: true`` with an
``error`` and no data -- a value is never filled in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.research.mql5.records import ea_registry_path, parity_dir, worker_status_path

router = APIRouter(prefix="/api/v1/s6", tags=["s6"])


class Artifact(BaseModel):
    exists: bool
    path: str
    error: str | None = None
    data: Any = None


def _read(path: Path) -> Artifact:
    if not path.is_file():
        return Artifact(exists=False, path=str(path))
    try:
        return Artifact(exists=True, path=str(path), data=json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        return Artifact(exists=True, path=str(path), error=f"unreadable: {type(exc).__name__}: {exc}")


@router.get("/ea-registry", response_model=Artifact)
async def ea_registry() -> Artifact:
    """magic <-> dna_hash <-> template version (and digest), per generated EA and cell."""
    return _read(ea_registry_path())


class ParityIndex(BaseModel):
    exists: bool
    path: str
    subjects: int = 0
    passed: int = 0
    rows: list[dict[str, Any]] = []
    errors: list[str] = []


@router.get("/parity", response_model=ParityIndex)
async def parity_index() -> ParityIndex:
    """One row per recorded subject: its verdict and each level's verdict."""
    d = parity_dir()
    if not d.is_dir():
        return ParityIndex(exists=False, path=str(d))
    rows, errors = [], []
    for p in sorted(d.glob("*.json")):
        a = _read(p)
        if a.error is not None:
            errors.append(f"{p.name}: {a.error}")
            continue
        doc = a.data
        rows.append({"subject": doc.get("subject"), "cell": doc.get("cell"), "ok": doc.get("ok"),
                     "levels": {lv.get("level"): lv.get("ok") for lv in doc.get("levels", [])}})
    return ParityIndex(exists=True, path=str(d), subjects=len(rows), passed=sum(1 for r in rows if r["ok"] is True),
                       rows=rows, errors=errors)


@router.get("/parity/{subject}", response_model=Artifact)
async def parity_subject(subject: str) -> Artifact:
    if not subject.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="subject id must be alphanumeric with - or _")
    a = _read(parity_dir() / f"{subject}.json")
    if not a.exists:
        raise HTTPException(status_code=404, detail=f"no parity record for {subject}")
    return a


@router.get("/worker", response_model=Artifact)
async def worker_status() -> Artifact:
    """The DEMO worker's last state and reconciliation (written by the worker; never a hashed input)."""
    return _read(worker_status_path())


__all__ = ["router"]
