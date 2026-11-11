"""AUTOMATION and named portfolios (``/api/v2/app/automation``, ``/api/v2/app/portfolios``).

Research contract (worker4/RESEARCH_GAP_AUDIT.md §2.4), in ARQVIO's routes: the list with its presets, the live
state for the NOW panel, save / toggle / run now / cancel / delete / preview, the activity log; and the portfolios
the PORTFOLIO step fills (create, rename, delete, add, remove).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.research.automation import portfolios as pf
from engine.research.automation import runner
from engine.research.automation import steps as S

router = APIRouter(prefix="/api/v2/app/automation", tags=["app-automation"])
portfolios_router = APIRouter(prefix="/api/v2/app/portfolios", tags=["app-portfolios"])


def _view(a: dict[str, Any]) -> dict[str, Any]:
    return {**a, "every_text": S.every_text(a.get("every_hours", 24)),
            "alongside": not any(s.get("kind") == "campaign" for s in a.get("steps") or [])}


@router.get("")
def list_automations() -> dict[str, Any]:
    from engine.research.data import terminals as mt5_terminals

    groups: dict[str, int] = {}
    terms: list[dict[str, Any]] = []
    try:
        ts = mt5_terminals.discover()
        terms = [{"id": t.id, "path": t.path, "server": t.remembered_server, "running": t.running} for t in ts]
        chosen = mt5_terminals.chosen_path()
        t = next((x for x in ts if x.path == chosen), None) or next((x for x in ts if x.running), None)
        if t is not None and t.running:
            groups = S.group_counts(mt5_terminals.symbols(t))
    except Exception:  # no MetaTrader 5: the page still lists automations; group counts stay empty
        pass
    return {"automations": [_view(a) for a in runner.load()], "log": runner.activity_log(),
            "groups": [{"id": g, "label": S.GROUP_TEXT[g], "count": groups.get(g)} for g in S.MARKET_GROUPS],
            "timeframes": list(S.TIMEFRAMES), "kinds": list(S.KINDS), "built": list(S.BUILT),
            "planned_why": S.PLANNED_WHY, "defaults": {k: S.default_step(k) for k in S.KINDS},
            "launch_defaults": S.launch_defaults(), "terminals": terms, "portfolios": pf.load()}


@router.get("/state")
def automation_state() -> dict[str, Any]:
    return runner.state()


class SaveBody(BaseModel):
    id: str | None = None
    name: str = ""
    enabled: bool = True
    every_hours: float = Field(24, ge=0, le=24 * 7 * 52)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    terminal: str = "AUTO"


@router.post("/save")
def save_automation(body: SaveBody) -> dict[str, Any]:
    try:
        a = runner.save(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    note = "AUTOMATION SAVED" + (" · NEW STEPS START WITH THE NEXT RUN" if a.get("next_steps") else "")
    return {"automation": _view(a), "note": note}


class IdBody(BaseModel):
    id: str
    enabled: bool | None = None


@router.post("/toggle")
def toggle(body: IdBody) -> dict[str, Any]:
    try:
        return {"automation": _view(runner.toggle(body.id, bool(body.enabled)))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/run")
def run_now(body: IdBody) -> dict[str, Any]:
    try:
        a = runner.run_now(body.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"automation": _view(a), "toast": a["toast"]}


@router.post("/cancel")
def cancel(body: IdBody) -> dict[str, Any]:
    """STOP: this run ends after its current step; a campaign this automation started stops too, the owner's
    own campaigns keep running."""
    return {"stopped": runner.SCHEDULER.cancel(body.id)}


@router.post("/delete")
def delete(body: IdBody) -> dict[str, Any]:
    try:
        runner.delete(body.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": body.id}


@router.post("/preview")
def preview(body: SaveBody) -> dict[str, Any]:
    """What each step would do right now; nothing starts."""
    return {"lines": S.preview(body.model_dump()),
            "note": "Nothing was started. The numbers come from your broker and vault as they are now."}


# ---- named portfolios ----------------------------------------------------------------------------------------------

class NameBody(BaseModel):
    name: str


class IdsBody(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=500)


def _pf(fn, *a: Any) -> Any:
    try:
        return fn(*a)
    except pf.PortfolioError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@portfolios_router.get("")
def list_portfolios() -> dict[str, Any]:
    return {"portfolios": pf.load()}


@portfolios_router.get("/cards")
def portfolio_cards() -> dict[str, Any]:
    """One card per portfolio (research PORTFOLIOS list): members, AVG RET, AVG DD, how many are live, members that
    left the vault, and the diversity grade once the portfolio was judged."""
    from engine.api.routes_app_library import portfolio_grade
    from engine.research.automation.steps import _live_ids
    from engine.research.vault import library

    rows = {e["id"]: e for e in library.load()}
    live = _live_ids()
    cards = []
    for p in pf.load():
        here = [rows[m["id"]] for m in p["members"] if m["id"] in rows]
        rets = [float(e["metrics"]["total_return"]) for e in here if e["metrics"].get("total_return") is not None]
        dds = [float(e["metrics"]["max_dd"]) for e in here if e["metrics"].get("max_dd") is not None]
        cards.append({"id": p["id"], "name": p["name"], "created_utc": p["created_utc"], "members": len(p["members"]),
                      "in_vault": len(here), "live": sum(1 for e in here if e["id"] in live),
                      "avg_return": sum(rets) / len(rets) if rets else None,
                      "avg_dd": sum(dds) / len(dds) if dds else None,
                      "brokers": sorted({e.get("dataset_id") and _broker(e["dataset_id"]) or "" for e in here} - {""}),
                      "grade": portfolio_grade([e["id"] for e in here]) if len(here) >= 2 else None})
    return {"cards": cards}


def _broker(dataset_id: str) -> str:
    from engine.api.routes_app_library import _meta

    return str(_meta(dataset_id).get("server") or "")


@portfolios_router.post("")
def create_portfolio(body: NameBody) -> dict[str, Any]:
    return {"portfolio": _pf(pf.create, body.name)}


@portfolios_router.post("/{pid}/rename")
def rename_portfolio(pid: str, body: NameBody) -> dict[str, Any]:
    return {"portfolio": _pf(pf.rename, pid, body.name)}


@portfolios_router.delete("/{pid}")
def delete_portfolio(pid: str) -> dict[str, Any]:
    _pf(pf.delete, pid)
    return {"deleted": pid}


@portfolios_router.post("/{pid}/add")
def add_members(pid: str, body: IdsBody) -> dict[str, Any]:
    return {"portfolio": _pf(pf.add, pid, body.ids)}


@portfolios_router.post("/{pid}/remove")
def remove_members(pid: str, body: IdsBody) -> dict[str, Any]:
    return {"portfolio": _pf(pf.remove, pid, body.ids)}


__all__ = ["portfolios_router", "router"]
