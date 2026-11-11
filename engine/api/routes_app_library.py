"""The strategy vault API (``/api/v2/app/library``): import finalists, score and tier them, run robustness.

The app calls this screen "Vault" (strategies); the secrets store is the Keychain. Every value served here is
read from the vault file, which traces each entry to its campaign manifest (engine/research/vault/library.py).
Robustness runs one strategy at a time, in a background thread of this sidecar, on the strategy's own pinned
cell; its progress is the entry's ``robustness.state`` (RUNNING, DONE or FAILED).
"""
from __future__ import annotations

import contextlib
import functools
import json
import threading
import traceback
from collections import deque
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.research.vault import library

router = APIRouter(prefix="/api/v2/app/library", tags=["app-library"])

_lock = threading.Lock()
_job: dict[str, Any] = {}  # the one robustness run in flight: {"id", "thread"}


def _refuse(reason: str, status: int = 409) -> HTTPException:
    return HTTPException(status_code=status, detail=f"REFUSED: {reason}")


def _running(entry_id: str) -> bool:
    return _job.get("id") == entry_id and bool(_job.get("thread")) and _job["thread"].is_alive()


def _robustness_state(e: dict[str, Any]) -> str | None:
    state = (e.get("robustness") or {}).get("state")
    # RUNNING on file but no thread of this sidecar running it: the sidecar stopped mid-run
    return "INTERRUPTED" if state == "RUNNING" and not _running(e["id"]) else state


@functools.lru_cache(maxsize=4096)
def _meta(dataset_id: str | None) -> dict[str, Any]:
    """The pinned bars' sidecar: broker server, first and last bar (cached; snapshots never change)."""
    from engine.research.data.dataset_snapshot import load_meta

    try:
        m = load_meta(dataset_id) if dataset_id else None
    except Exception:
        m = None
    return m.to_dict() if m else {}


def _win_rate(e: dict[str, Any]) -> float | None:
    """From the inspector's cache when the strategy was opened once (no re-run just for a column)."""
    from engine.research.vault import inspect

    try:
        doc = json.loads(inspect.cache_path(e).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return None
    w = (doc.get("stats") or doc).get("win_rate")
    return float(w) if isinstance(w, int | float) else None


def _extra(e: dict[str, Any], groups: dict[str, list[str]]) -> dict[str, Any]:
    """Research vault columns: BROKER, WINDOW, FIRST TRADE, CAGR, PARITY, WIN RATE, ROBUST, PORTFOLIO."""
    meta = _meta(e.get("dataset_id"))
    years = None
    try:
        t0 = datetime.fromisoformat(meta["start_utc"])
        t1 = datetime.fromisoformat(meta["end_utc"])
        years = max((t1 - t0).days / 365.25, 1 / 365.25)
    except (KeyError, ValueError, TypeError):
        pass
    ret = e["metrics"].get("total_return")
    cagr = ((1 + ret) ** (1 / years) - 1) if years and ret is not None and ret > -1 else None
    par = e.get("parity") or {}
    rob = e.get("robustness") or {}
    return {"broker": meta.get("server"), "window_years": round(years, 2) if years else None,
            "first_bar": meta.get("start_utc"), "cagr": cagr,
            "parity_grade": par.get("grade") if par.get("state") == "DONE" else par.get("state"),
            "win_rate": _win_rate(e), "robust": rob.get("verdict") or rob.get("state"),
            "portfolios": groups.get(e["id"], [])}


def _summary(e: dict[str, Any]) -> dict[str, Any]:
    m = e["metrics"]
    return {"id": e["id"], "family": e["family"], "cell": e["cell"], "symbol": e["symbol"], "timeframe": e["timeframe"],
            "engine": e.get("engine", "v2"), "tier": e["tier"], "score": e["score"], "score_max": e["score_max"], "grade": e["grade"],
            "sharpe": m["sharpe"], "n_trades": m["n_trades"], "max_dd": m["max_dd"], "dsr_final": m["dsr_final"],
            "total_return": m.get("total_return"),
            "sham_margin": m["sham_margin"], "cross_sharpe": m["cross_sharpe"], "alive_on_both": m["alive_on_both"],
            "robustness": _robustness_state(e), "campaign_id": e["source"]["campaign_id"], "added_utc": e["added_utc"],
            "held_out": {k: (e.get("held_out") or {}).get(k) for k in ("state", "return", "trades", "sharpe")}}


@router.get("")
def overview() -> dict[str, Any]:
    try:
        rows = library.load()
    except (library.VaultError, ValueError) as exc:
        raise _refuse(str(exc)) from exc
    rows = sorted(rows, key=lambda e: (-(e["tier"] == "CERTIFIED"), -e["score"], e["id"]))
    imported = {e["source"]["campaign_id"] for e in rows}
    campaigns = [{**c, "imported": c["campaign_id"] in imported} for c in library.completed_campaigns()]
    dup = library.copies(rows)
    from engine.research.automation import portfolios as pf

    groups: dict[str, list[str]] = {}
    for p in pf.load():
        for mem in p["members"]:
            groups.setdefault(mem["id"], []).append(p["name"])
    return {"strategies": [{**_summary(e), **dup[e["id"]], **_extra(e, groups)} for e in rows], "campaigns": campaigns,
            "counts": {t: sum(e["tier"] == t for e in rows) for t in library.TIERS},
            "robustness_running": _job["id"] if _job.get("id") and _running(_job["id"]) else None}


@router.get("/strategies/{entry_id}")
def detail(entry_id: str) -> dict[str, Any]:
    try:
        e = library.get(entry_id)
    except library.VaultError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {**e, "robustness_state": _robustness_state(e), "address": library.address(e)}


@router.get("/export")
def export(ids: str = "") -> dict[str, Any]:
    """The whole vault, or the comma-separated ``ids``, as one JSON document (research GET JSON)."""
    try:
        return library.export([i for i in ids.split(",") if i] or None)
    except library.VaultError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/audit")
def audit(limit: int = 200) -> dict[str, Any]:
    """The vault's change log: every strategy sent to the recycle bin (deleted, purged) or restored."""
    return {"entries": library.audit_log(max(1, min(limit, 2000)))}


class Import(BaseModel):
    campaign_id: str


@router.post("/import")
def import_campaign(body: Import) -> dict[str, Any]:
    with _lock:
        try:
            return library.import_campaign(body.campaign_id)
        except library.VaultError as exc:
            raise _refuse(str(exc), 404) from exc


@router.delete("/strategies/{entry_id}")
def remove(entry_id: str) -> dict[str, Any]:
    with _lock:
        if _running(entry_id):
            raise _refuse("its robustness run is in progress")
        try:
            library.remove(entry_id)
        except library.VaultError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"removed": entry_id}


def _robustness(entry_id: str) -> None:
    from engine.research.ga.runner import prepare_cell
    from engine.research.search.dna import DNA
    from engine.research.vault import robustness

    try:
        e = library.get(entry_id)
        from engine.research.search.real_backtest import load_cell_bars_from_snapshot

        from engine.research.search.real_backtest import spread_cost_mode

        with spread_cost_mode(library.recorded_cost_model(e)):  # its own bars and cost model, as recorded
            ev = prepare_cell(e["cell"], dataset_id=e["dataset_id"])
        times = load_cell_bars_from_snapshot(ev.symbol, ev.timeframe, ev.dataset_id).bars.time
        result = robustness.run(DNA(**e["dna"]), timeframe=e["timeframe"], recorded_sharpe=e["metrics"]["sharpe"],
                                evaluate=ev.evaluate, bar_times=times,
                                random_sharpe=lambda d, n, k: float(
                                    ev.sham_fn(d, n, salt=f"{robustness.RANDOM_SALT}-{k}").sharpe))
    except Exception as exc:  # recorded on the entry: the screen shows why it failed
        result = {"state": "FAILED", "reason": f"{type(exc).__name__}: {exc}",
                  "trace": traceback.format_exc(limit=4)}
    with _lock, contextlib.suppress(library.VaultError):  # removed while it ran: nothing to record on
        library.set_robustness(entry_id, result)


@router.post("/strategies/{entry_id}/robustness")
def robustness_start(entry_id: str) -> dict[str, Any]:
    with _lock:
        if _job.get("thread") and _job["thread"].is_alive():
            raise _refuse(f"a robustness run is in progress ({_job['id']}); one at a time")
        try:
            e = library.get(entry_id)
        except library.VaultError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if e["timeframe"] is None:
            raise _refuse(f"its cell {e['cell']} is not a validated cell any more")
        library.set_robustness(entry_id, {"state": "RUNNING"})
        t = threading.Thread(target=_robustness, args=(entry_id,), name=f"robustness-{entry_id}", daemon=True)
        _job.update(id=entry_id, thread=t)
        t.start()
    return {"id": entry_id, "robustness": "RUNNING"}


def _certify(entry_id: str) -> None:
    """CERTIFY (V1 -> V2): engine V2's funnel on a V1 strategy -- its bred window, the cross market, the sham and the
    DSR against its V1 campaign's own trials (a VALIDATION_RUN: the bar does not move)."""
    from engine.research.ga.config import cross_cell_for
    from engine.research.ga.fitness import candidate_dsr, dsr_trials
    from engine.research.ga.runner import prepare_cell
    from engine.research.search.dna import DNA
    from engine.research.search.real_backtest import spread_cost_mode

    try:
        e = library.get(entry_id)
        with spread_cost_mode(library.recorded_cost_model(e)):
            ev = prepare_cell(e["cell"], dataset_id=e["dataset_id"])
            cross = cross_cell_for(e["cell"])
            evx = prepare_cell(cross)
        dna = DNA(**{k: tuple(v) if isinstance(v, list) else v for k, v in e["dna"].items()})
        base, x = ev.evaluate(dna), evx.evaluate(dna)
        n, sharpes = dsr_trials(library.v1_trial_sharpes(e), n_effective=10**9)
        dsr = candidate_dsr(base.result, n_trials=n, trial_sharpes=sharpes, symbol=ev.symbol, timeframe=ev.timeframe)
        sham = ev.sham_sharpe(base.result)
        updated = library.certify_metrics(e, primary=base, cross=x, dsr=dsr, sham_sharpe=sham, n_trials=n,
                                          cross_cell=cross)
        with _lock, contextlib.suppress(library.VaultError):
            library.set_certify(entry_id, updated)
    except Exception as exc:  # recorded on the entry: the screen shows why it failed
        with _lock, contextlib.suppress(library.VaultError):
            library.set_certify(entry_id, None, state="FAILED", reason=f"{type(exc).__name__}: {exc}")


@router.post("/strategies/{entry_id}/certify")
def certify_start(entry_id: str) -> dict[str, Any]:
    with _lock:
        if _job.get("thread") and _job["thread"].is_alive():
            raise _refuse(f"a vault job is in progress ({_job['id']}); one at a time")
        try:
            e = library.get(entry_id)
        except library.VaultError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if e.get("engine") != "v1":
            raise _refuse("CERTIFY is for an engine V1 strategy; this one was bred by V2 and is judged already")
        library.set_certify(entry_id, None, state="RUNNING")
        t = threading.Thread(target=_certify, args=(entry_id,), name=f"certify-{entry_id}", daemon=True)
        _job.update(id=entry_id, thread=t)
        t.start()
    return {"id": entry_id, "certify": "RUNNING"}


def _parity(entry_id: str) -> None:
    from engine.core.paths import get_artifacts_dir
    from engine.research.vault import parity_job

    try:
        result = parity_job.run(library.get(entry_id), get_artifacts_dir() / "parity" / entry_id.replace("@", "_"))
    except parity_job.ParityRefused as exc:
        result = {"state": "REFUSED", "reason": str(exc)}
    except Exception as exc:  # recorded on the entry: the screen shows why it failed
        result = {"state": "FAILED", "reason": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
    with _lock, contextlib.suppress(library.VaultError):
        library.set_parity(entry_id, result)


@router.post("/strategies/{entry_id}/parity")
def parity_start(entry_id: str) -> dict[str, Any]:
    """PARITY TEST: the strategy's EA in the Strategy Tester of the TEST terminal vs the oracle (grade A-F)."""
    from engine.research.vault import parity_job

    with _lock:
        if _job.get("thread") and _job["thread"].is_alive():
            raise _refuse(f"a vault job is in progress ({_job['id']}); one at a time")
        try:
            e = library.get(entry_id)
        except library.VaultError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            parity_job.preflight(e)
        except parity_job.ParityRefused as exc:
            raise _refuse(str(exc)) from exc
        library.set_parity(entry_id, {"state": "RUNNING"})
        t = threading.Thread(target=_parity, args=(entry_id,), name=f"parity-{entry_id}", daemon=True)
        _job.update(id=entry_id, thread=t)
        t.start()
    return {"id": entry_id, "parity": "RUNNING"}


@router.get("/strategies/{entry_id}/inspect")
def inspect_strategy(entry_id: str, refresh: bool = False) -> dict[str, Any]:
    """Trade-by-trade statistics and the equity curve, re-run on the strategy's own pinned data (cached)."""
    from engine.research.vault import inspect

    try:
        e = library.get(entry_id)
    except library.VaultError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return inspect.inspect(e, refresh=refresh)
    except Exception as exc:  # the data of a removed cell, a damaged snapshot: said, never guessed
        raise _refuse(f"the strategy could not be re-run: {type(exc).__name__}: {exc}", 409) from exc


_portfolio_cache: dict[str, dict[str, Any]] = {}


@router.get("/portfolio")
def portfolio(source: str = "deployed", n: int = 8) -> dict[str, Any]:
    """The DECK portfolio view (engine/research/vault/portfolio_view.py): ``deployed`` = the strategies of the bots
    in ARQVIO's deploy list; ``top`` = the n best distinct vault strategies by score; ``pf:<id>`` = a named
    portfolio's members."""
    from engine.research.deploy import full_deploy as fd
    from engine.research.vault import portfolio_view

    rows = library.load()
    by_id = {e["id"]: e for e in rows}
    if source == "deployed":
        ids = [b["source"]["vault_id"] for b in fd.load_bots() if b.get("status") != "REMOVED"
               and (b.get("source") or {}).get("vault_id") in by_id]
        members = [by_id[i] for i in dict.fromkeys(ids)]
    elif source.startswith("pf:"):  # a named portfolio (automation/portfolios.py): its members still in the vault
        from engine.research.automation import portfolios as pf

        try:
            named = pf.get(source[3:])
        except pf.PortfolioError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        members = [by_id[m["id"]] for m in named["members"] if m["id"] in by_id]
    elif source == "top":
        ranked = sorted(rows, key=lambda e: (-(e["tier"] == "CERTIFIED"), -e["score"], e["id"]))
        dup = library.copies(ranked)
        members = [e for e in ranked if dup[e["id"]]["copy_of"] is None][: max(2, min(int(n), 20))]
    else:
        raise HTTPException(status_code=422, detail=f"source must be deployed, top or pf:<id>, not {source!r}")
    if len(members) < 2:
        return {"source": source, "members": [{"id": m["id"]} for m in members], "overall": None,
                "note": "a portfolio needs at least two strategies"}
    key = ",".join(sorted(m["id"] for m in members))
    if key not in _portfolio_cache:
        _portfolio_cache[key] = portfolio_view.build(members)
    return {"source": source, **_portfolio_cache[key]}


_truesim_cache: dict[str, dict[str, Any]] = {}


@router.get("/portfolio/truesim")
def truesim(source: str = "deployed", n: int = 8) -> dict[str, Any]:
    """TRUE SIM: the portfolio's members on ONE account (vault/truesim.py, the shared-account replay), cached per
    member set."""
    from engine.research.vault import truesim as ts

    view = portfolio(source, n)
    ids = [m["id"] for m in view.get("members", [])]
    if len(ids) < 2:
        return {"state": "EMPTY", "members": len(ids), "note": "a portfolio needs at least two strategies"}
    key = ",".join(sorted(ids))
    if key not in _truesim_cache:
        by_id = {e["id"]: e for e in library.load()}
        _truesim_cache[key] = ts.run([by_id[i] for i in ids if i in by_id])
    return {"source": source, **_truesim_cache[key]}


@router.get("/recycle")
def recycle_bin() -> dict[str, Any]:
    rows = library.recycled()
    return {"days": library.RECYCLE_DAYS, "strategies": [{**_summary(e), "deleted_utc": e["deleted_utc"],
                                                         "deleted_reason": e.get("deleted_reason")} for e in rows]}


@router.post("/recycle/{entry_id}/restore")
def restore(entry_id: str) -> dict[str, Any]:
    with _lock:
        try:
            e = library.restore(entry_id)
        except library.VaultError as exc:
            raise _refuse(str(exc)) from exc
    return {"restored": e["id"]}


class Purge(BaseModel):
    rule: str
    apply: bool = False


@router.post("/purge")
def purge(body: Purge) -> dict[str, Any]:
    with _lock:
        try:
            return library.purge(body.rule, apply=body.apply)
        except library.VaultError as exc:
            raise _refuse(str(exc)) from exc


# ---- research vault actions: N SELECTED, criteria purge, DEDUPE, RESTORE ALL, DELETE CAMPAIGN, GET JSON ------------
# idea: research VAULT right-click menu and toolbar (RF 69568-71660, PN 163/255/367/451); code ARQVIO's.

_queue: deque[tuple[str, str]] = deque()  # (kind, id) waiting for the one vault job slot: REASSESS / PARITY TEST
_qthread: dict[str, Any] = {}


def _drain() -> None:
    """Run queued jobs one at a time in the vault job slot (robustness and parity never run side by side)."""
    while True:
        running = _job.get("thread")
        if running is not None and running.is_alive():
            running.join()
            continue
        with _lock:
            if not _queue:
                _qthread.clear()
                return
            kind, eid = _queue.popleft()
            try:
                if kind == "reassess":
                    library.set_robustness(eid, {"state": "RUNNING"})
                else:
                    library.set_parity(eid, {"state": "RUNNING"})
            except library.VaultError:  # deleted while it waited
                continue
            t = threading.Thread(target=_robustness if kind == "reassess" else _parity, args=(eid,),
                                 name=f"{kind}-{eid}", daemon=True)
            _job.update(id=eid, thread=t)
            t.start()


class Bulk(BaseModel):
    action: str
    ids: list[str]


@router.post("/bulk")
def bulk(body: Bulk) -> dict[str, Any]:
    """N SELECTED: DELETE (to the recycle bin), REASSESS (robustness re-run) or PARITY TEST, queued one at a time."""
    from engine.research.vault import parity_job

    ids = list(dict.fromkeys(body.ids))
    if not ids:
        raise _refuse("select at least one strategy", 422)
    if body.action == "delete":
        with _lock:
            have = {e["id"] for e in library.load()}
            busy = [i for i in ids if _running(i) or i not in have]
            go = [i for i in ids if i not in busy]
            removed = library.remove_many(go, reason=f"deleted ({len(ids)} selected)") if go else []
        return {"action": "delete", "done": removed, "skipped": busy}
    if body.action not in ("reassess", "parity"):
        raise _refuse(f"unknown action {body.action!r}", 422)
    queued: list[str] = []
    refused: dict[str, str] = {}
    rows = {e["id"]: e for e in library.load()}
    for i in ids:
        if i not in rows:
            refused[i] = "not in the vault"
            continue
        if body.action == "parity":
            try:
                parity_job.preflight(rows[i])
            except parity_job.ParityRefused as exc:
                refused[i] = str(exc)
                continue
        if (body.action, i) not in _queue:
            _queue.append((body.action, i))
            queued.append(i)
    with _lock:
        if queued and not _qthread.get("t"):
            t = threading.Thread(target=_drain, name="vault-queue", daemon=True)
            _qthread["t"] = t
            t.start()
    return {"action": body.action, "queued": queued, "refused": refused, "waiting": len(_queue)}


@router.get("/jobs")
def jobs() -> dict[str, Any]:
    return {"running": _job.get("id") if _job.get("id") and _running(_job["id"]) else None,
            "waiting": [{"kind": k, "id": i} for k, i in _queue]}


class Criteria(BaseModel):
    criteria: dict[str, float | None] = {}
    apply: bool = False


@router.post("/purge-criteria")
def purge_criteria(body: Criteria) -> dict[str, Any]:
    """PURGE with the research criteria (min score, trades, max DD %, min return %, min OOS return %, max age):
    preview first, then apply; portfolio members and live bots are never touched; 30 days in the recycle bin."""
    from engine.research.automation import portfolios as pf
    from engine.research.automation.steps import _live_ids, purge_matches, saved_or

    with _lock:
        ids = purge_matches(saved_or(body.criteria), library.load(), pf.protected_ids() | _live_ids())
        if body.apply and ids:
            library.remove_many(ids, reason="purge: criteria")
    return {"ids": ids, "count": len(ids), "applied": bool(body.apply and ids)}


@router.post("/dedupe")
def dedupe(body: Criteria) -> dict[str, Any]:
    """DEDUPE: strategies that made the same trades as a better-ranked one go to the recycle bin; the best stays."""
    with _lock:
        return library.purge("copies", apply=body.apply)


@router.post("/recycle/restore-all")
def restore_all() -> dict[str, Any]:
    done = []
    with _lock:
        for e in library.recycled():
            with contextlib.suppress(library.VaultError):
                done.append(library.restore(e["id"])["id"])
    return {"restored": done}


@router.post("/campaign/{campaign_id}/delete")
def delete_campaign(campaign_id: str) -> dict[str, Any]:
    """DELETE CAMPAIGN: every strategy of one campaign to the recycle bin (restorable for 30 days)."""
    with _lock:
        ids = [e["id"] for e in library.load() if e["source"]["campaign_id"] == campaign_id and not _running(e["id"])]
        if not ids:
            raise _refuse(f"no strategies of campaign {campaign_id!r} in the vault", 404)
        library.remove_many(ids, reason=f"campaign {campaign_id} deleted")
    return {"removed": ids}


@router.get("/strategies/{entry_id}/passport")
def passport(entry_id: str) -> dict[str, Any]:
    """GET JSON · FULL PASSPORT: the vault record, its address, the bars' sidecar and the inspector's trade record."""
    from engine.research.vault import inspect

    try:
        e = library.get(entry_id)
    except library.VaultError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        trades = json.loads(inspect.cache_path(e).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        trades = None
    return {"schema": "arqvio.passport", "version": 1, "strategy": e, "address": library.address(e),
            "bars": _meta(e.get("dataset_id")), "inspector": trades}


def portfolio_grade(ids: list[str]) -> str | None:
    """The diversity grade of a member set when it was judged already this session (never computed just for a card)."""
    v = _portfolio_cache.get(",".join(sorted(ids)))
    return v.get("overall") if v else None
