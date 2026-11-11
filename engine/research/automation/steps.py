"""The automation steps: defaults, validation, a dry-run line, and the work itself.

Research (worker4/RESEARCH_GAP_AUDIT.md §2.3): CAMPAIGN breeds on market groups, PORTFOLIO fills a named team that
trades differently, PARITY tests members in the Strategy Tester, PURGE sends weak strategies to the recycle bin
(portfolio members never), REPLACE swaps live bots, REASSESS re-checks strategies on new data. REPLACE (it swaps live
bots: a live-trading action, HARD STOP for the owner) and REASSESS (needs a broker refresh of the bars) are accepted
in saved automations as PLANNED and refused at save until they are built; everything else runs.
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

KINDS = ("campaign", "portfolio", "parity", "purge", "replace", "reassess")
BUILT = ("campaign", "portfolio", "parity", "purge")
PLANNED_WHY = {"replace": "REPLACE swaps live bots; it comes with the live-bot health checks",
               "reassess": "REASSESS needs the broker refresh of a strategy's bars; it comes with REFRESH"}

# research CAMPAIGN market groups; each name is matched to this broker's symbol (EURUSDm, EURUSD.pro, ...)
MARKET_GROUPS: dict[str, tuple[str, ...]] = {
    "fx_majors": ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"),
    "fx_minors": ("EURGBP", "EURJPY", "GBPJPY", "EURCHF", "EURAUD", "AUDJPY", "GBPCHF", "CADJPY", "CHFJPY", "AUDNZD"),
    "indices": ("US30", "US500", "USTEC", "US100", "DE40", "GER40", "UK100", "JP225"),
    "metals": ("XAUUSD", "XAGUSD"),
    "energy": ("USOIL", "UKOIL", "XNGUSD", "XTIUSD", "XBRUSD"),
}
GROUP_TEXT = {"fx_majors": "FX MAJORS", "fx_minors": "FX MINORS", "indices": "INDICES", "metals": "METALS",
              "energy": "ENERGY"}
TIMEFRAMES = ("M5", "M15", "M30", "H1", "H4", "D1")
HISTORY_YEARS = {"FOCUSED": 3, "DEEP": 6}
PASS_RANK = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
DEFAULT_PURGE = {"min_trades": 30, "max_dd": 35.0, "min_return": 0.0}  # used for blanks once any criterion is set
CANDIDATE_FACTOR = 4  # PORTFOLIO: at most this many x TEAM SIZE candidates get a correlation run (bounded cost)


def default_step(kind: str, *, after_campaign: bool = False) -> dict[str, Any]:
    if kind == "campaign":
        return {"kind": kind, "preset_groups": ["fx_majors"], "extra_symbols": [], "timeframes": ["H1"],
                "skip_graveyards": True, "graveyard_runs": 2, "graveyard_retest_every": 4, "overrides": {}}
    if kind == "portfolio":
        return {"kind": kind, "name": "", "mode": "top_up", "min_score": 55, "target_size": 30, "max_per_symbol": 4,
                "max_per_tf": 1, "max_avg_corr": 0.35, "max_age_days": 0, "use_run_pool": True, "criteria": {},
                "deploy_new_bots": False}
    if kind == "parity":
        return {"kind": kind, "pass_mark": "B", "tests_per_run": 10, "portfolio": ""}
    if kind == "purge":
        return {"kind": kind, "criteria": {}}
    if kind == "replace":
        return {"kind": kind, "portfolio": "", "where": "same", "ban_genes": True, "min_score": 55, "max_age_days": 30}
    if kind == "reassess":
        return {"kind": kind, "scope": "run" if after_campaign else "vault", "max_checks": 25, "max_age_days": 30,
                "min_score": 55}
    raise ValueError(f"unknown step kind {kind!r}")


def validate(auto: dict[str, Any]) -> list[str]:
    """Research's save messages, in its order; empty when the automation can be saved."""
    out: list[str] = []
    if not str(auto.get("name") or "").strip():
        out.append("NAME THIS AUTOMATION")
    every = auto.get("every_hours")
    if not isinstance(every, int | float) or every < 0 or (every != 0 and every <= 0):
        out.append("THE NUMBER MUST BE MORE THAN 0, OR PICK NON-STOP")
    steps = auto.get("steps") or []
    if not steps:
        out.append("ADD AT LEAST ONE STEP")
    if sum(1 for s in steps if s.get("kind") == "parity") > 1:
        out.append("ONE PARITY STEP PER AUTOMATION")
    has_portfolio = any(s.get("kind") == "portfolio" for s in steps)
    for i, s in enumerate(steps, 1):
        k = s.get("kind")
        if k not in KINDS:
            out.append(f"STEP {i}: UNKNOWN STEP {str(k).upper()}")
        elif k not in BUILT:
            out.append(f"STEP {i}: {k.upper()} IS NOT BUILT YET -- {PLANNED_WHY[k].upper()}")
        elif k == "campaign":
            if not (s.get("preset_groups") or s.get("extra_symbols")):
                out.append(f"STEP {i}: PICK AT LEAST ONE MARKET GROUP OR EXTRA MARKET")
            if not s.get("timeframes"):
                out.append(f"STEP {i}: PICK AT LEAST ONE TIMEFRAME")
            bad = [g for g in s.get("preset_groups") or [] if g not in MARKET_GROUPS]
            bad += [t for t in s.get("timeframes") or [] if t not in TIMEFRAMES]
            if bad:
                out.append(f"STEP {i}: UNKNOWN {', '.join(bad).upper()}")
        elif k == "portfolio":
            if not str(s.get("name") or "").strip():
                out.append(f"STEP {i}: NAME THE PORTFOLIO")
            if s.get("deploy_new_bots"):
                out.append(f"STEP {i}: DEPLOY NEW BOTS IS NOT BUILT YET -- IT PUTS BOTS LIVE AND WAITS FOR YOUR OK")
        elif k == "parity":
            n = s.get("tests_per_run")
            if not isinstance(n, int) or not 1 <= n <= 100:
                out.append(f"STEP {i}: TESTS PER RUN MUST BE A WHOLE NUMBER FROM 1 TO 100")
            if not has_portfolio and not str(s.get("portfolio") or "").strip():
                out.append(f"STEP {i}: PICK THE PORTFOLIO TO PARITY TEST")
    return out


def every_text(hours: float) -> str:
    """Research's interval word: NON-STOP, WEEKLY, EVERY nW / nD / nH."""
    h = int(hours)
    if h == 0:
        return "NON-STOP"
    if h == 168:
        return "WEEKLY"
    if h % 168 == 0:
        return f"EVERY {h // 168}W"
    if h > 24 and h % 24 == 0:
        return f"EVERY {h // 24}D"
    return f"EVERY {h}H"


# ---- context shared with the steps ------------------------------------------------------------------------------------

class Run:
    """One run of one automation: what the steps hand each other, and how they ask for a stop."""

    def __init__(self, auto: dict[str, Any], *, cancel: threading.Event, log: Callable[[str], None],
                 phase: Callable[[str, dict[str, Any] | None], None],
                 warn: Callable[[str], None] | None = None) -> None:
        self.auto = auto
        self.cancel = cancel
        self.log = log
        self.warn = warn or log
        self.phase = phase                          # (label, progress) for the NOW panel
        self.run_pool: list[str] = []               # vault ids this run's campaigns added
        self.campaigned = False
        self.portfolio_id: str | None = None        # the last PORTFOLIO step's portfolio
        self.added: list[str] = []                  # members this run added
        self.started_utc = datetime.now(tz=UTC).isoformat(timespec="seconds")

    @property
    def origin(self) -> str:
        return f"automation:{self.auto['id']}"


def _vault() -> list[dict[str, Any]]:
    from engine.research.vault import library

    return library.load()


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---- CAMPAIGN -------------------------------------------------------------------------------------------------------

def _terminal(auto: dict[str, Any]) -> Any:
    from engine.research.data import terminals as mt5_terminals

    ts = mt5_terminals.discover()
    want = str(auto.get("terminal") or "AUTO")
    if want != "AUTO":
        return next((t for t in ts if t.id == want or t.path == want), None)
    path = mt5_terminals.chosen_path()
    return next((t for t in ts if t.path == path), None) or next((t for t in ts if t.running), None)


def broker_names(groups: list[str], extra: list[str], offered: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """(this broker's symbols for the groups + extras, the names it does not offer)."""
    from engine.research.data.terminals import _is_alias

    names = [str(r["name"]) for r in offered]
    out: list[str] = []
    missing: list[str] = []
    for base in [b for g in groups for b in MARKET_GROUPS.get(g, ())] + list(extra):
        b = base.upper().strip()
        if not b:
            continue
        hits = sorted((n for n in names if n.upper() == b or _is_alias(n.upper(), b)), key=len)
        if hits:
            if hits[0] not in out:
                out.append(hits[0])
        elif base in extra:
            missing.append(base)
    return out, missing


def group_counts(offered: list[dict[str, Any]]) -> dict[str, int]:
    return {g: len(broker_names([g], [], offered)[0]) for g in MARKET_GROUPS}


def _graveyard_path():
    from engine.core.paths import get_artifacts_dir

    return get_artifacts_dir() / "automation" / "dead_markets.json"


def _graveyard() -> dict[str, Any]:
    import json

    try:
        return json.loads(_graveyard_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_graveyard(doc: dict[str, Any]) -> None:
    import json

    p = _graveyard_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def dead(step: dict[str, Any], symbol: str, tf: str, now: datetime) -> bool:
    """Research DEAD MARKETS: skip a market after N runs without a survivor; try again after M runs or 30 days."""
    if not step.get("skip_graveyards", True):
        return False
    g = _graveyard().get(f"{symbol}|{tf}") or {}
    empty = int(g.get("empty_runs", 0))
    if empty < int(step.get("graveyard_runs", 2)):
        return False
    skipped = int(g.get("skipped", 0))
    last = g.get("last_utc")
    stale = bool(last) and now - datetime.fromisoformat(last) > timedelta(days=30)
    return not (skipped + 1 >= int(step.get("graveyard_retest_every", 4)) or stale)


def _mark_skip(symbol: str, tf: str) -> None:
    doc = _graveyard()
    g = doc.setdefault(f"{symbol}|{tf}", {})
    g["skipped"] = int(g.get("skipped", 0)) + 1
    _save_graveyard(doc)


def _mark_result(symbol: str, tf: str, added: int) -> None:
    doc = _graveyard()
    g = doc.setdefault(f"{symbol}|{tf}", {})
    g["empty_runs"] = 0 if added else int(g.get("empty_runs", 0)) + 1
    g["skipped"] = 0
    g["last_utc"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
    _save_graveyard(doc)


def launch_defaults() -> dict[str, Any]:
    """The Launch page values at run time: the last launch the owner started by hand (blank = these)."""
    from engine.api import routes_app_launch as launch

    d = launch.manual_defaults()
    return {"years": int(d.get("years", 3)), "depth": d.get("depth", "STANDARD"), "power": d.get("power", "medium"),
            "search": d.get("search", "standard")}


def plan_campaign(step: dict[str, Any], auto: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """What the step would launch now: terminal, symbols, timeframes, skipped dead markets, launch settings."""
    from engine.research.data import terminals as mt5_terminals

    now = now or datetime.now(tz=UTC)
    term = _terminal(auto)
    if term is None:
        return {"error": "no MetaTrader 5 terminal to read data from"}
    try:
        offered = mt5_terminals.symbols(term)
    except Exception as exc:
        return {"error": f"MetaTrader 5 ({term.path}) did not list its symbols: {exc}"}
    symbols, missing = broker_names(list(step.get("preset_groups") or []), list(step.get("extra_symbols") or []),
                                    offered)
    tfs = [t for t in step.get("timeframes") or [] if t in TIMEFRAMES]
    keep, skipped = [], []
    for s in symbols:
        if all(dead(step, s, tf, now) for tf in tfs):
            skipped.append(s)
        else:
            keep.append(s)
    d = launch_defaults()
    o = step.get("overrides") or {}
    hist = str(o.get("history") or "INHERIT")
    years = [d["years"]] if hist == "INHERIT" else [3, 6] if hist == "BOTH" else [HISTORY_YEARS.get(hist, d["years"])]
    return {"terminal": term.id, "terminal_path": term.path, "symbols": keep, "skipped_dead": skipped,
            "missing": missing, "timeframes": tfs, "years": years, "depth": o.get("depth") or d["depth"],
            "power": o.get("power") or d["power"], "search": o.get("engine") or d["search"],
            "runs": len(keep) * len(tfs) * len(years)}


def run_campaign(step: dict[str, Any], run: Run) -> str:
    from fastapi import HTTPException

    from engine.api import routes_app_launch as launch

    p = plan_campaign(step, run.auto)
    if p.get("error"):
        raise RuntimeError(p["error"])
    for s in p["skipped_dead"]:
        for tf in p["timeframes"]:
            _mark_skip(s, tf)
    if not p["symbols"]:
        return f"nothing to breed: {len(p['skipped_dead'])} dead market(s) skipped, {len(p['missing'])} not offered"
    total_added = 0
    # a launch takes at most 12 symbols: bigger groups breed in turns, BOTH history runs focused then deep
    chunks = [p["symbols"][i:i + 12] for i in range(0, len(p["symbols"]), 12)]
    for years, symbols in [(y, c) for y in p["years"] for c in chunks]:
        if run.cancel.is_set():
            break
        req = launch.LaunchRequest(terminal=p["terminal"], symbols=symbols, timeframes=p["timeframes"],
                                   years=years, depth=p["depth"], power=p["power"], search=p["search"],
                                   origin=run.origin)
        while True:  # the owner's own launch goes first: wait for the engine to be free
            if run.cancel.is_set():
                return "stopped before the campaign started"
            try:
                st = launch.launch(req)["launch"]
                break
            except HTTPException as exc:
                if exc.status_code != 409 or not launch.busy():
                    raise RuntimeError(str(exc.detail)) from exc
                run.phase("waiting for your campaign to finish", None)
                time.sleep(5)
        since = st["utc"]
        while launch.busy():
            if run.cancel.is_set():
                cur = launch.latest() or {}
                if (cur.get("request") or {}).get("origin") == run.origin:
                    launch.stop()  # only this automation's own campaign; the owner's never
            cur = launch.latest() or {}
            tasks = [t for t in cur.get("tasks", []) if t.get("kind") == "breed"]
            done = sum(1 for t in tasks if t.get("state") not in ("WAITING", "RUNNING", "CHECKING", "PAUSED"))
            run.phase(f"campaign {done}/{len(tasks)} markets · {years}y", {"done": done, "total": max(1, len(tasks))})
            time.sleep(3)
        cur = launch.latest() or {}
        for t in cur.get("tasks", []):
            if t.get("kind") == "breed" and t.get("role") != "cross" and t.get("state") == "DONE":
                _mark_result(t["symbol"], t["timeframe"], int(t.get("added") or 0))
        new = [r["id"] for r in _vault() if str(r.get("added_utc", "")) >= since]
        run.run_pool += [i for i in new if i not in run.run_pool]
        total_added += int(cur.get("vaulted", 0))
        # Preserve completed markets and additions before interpreting a partial/failing run.
        state = cur.get("state")
        if state == "PARTIAL_DATA":
            run.warn(f"campaign PARTIAL_DATA: {cur.get('detail') or 'some markets refused data'}; continuing")
        elif state not in ("DONE", "COMPLETED_ZERO_SURVIVORS"):
            raise RuntimeError(f"campaign {state or 'UNKNOWN'}: {cur.get('detail') or 'did not complete'}")
    run.campaigned = True
    skipped = f", {len(p['skipped_dead'])} dead market(s) skipped" if p["skipped_dead"] else ""
    return (f"bred {len(p['symbols'])} market(s) x {len(p['timeframes'])} timeframe(s): "
            f"{total_added} strateg{'y' if total_added == 1 else 'ies'} to the vault{skipped}")


# ---- PORTFOLIO ------------------------------------------------------------------------------------------------------

def _passes(r: dict[str, Any], step: dict[str, Any], now: datetime) -> bool:
    m = r.get("metrics") or {}
    c = step.get("criteria") or {}
    if float(r.get("score") or 0) < float(step.get("min_score") or 0):
        return False
    age = int(step.get("max_age_days") or 0)
    if age and now - datetime.fromisoformat(r["added_utc"]) > timedelta(days=age):
        return False
    checks = [("min_trades", m.get("n_trades"), lambda v, x: v >= x),
              ("max_dd", None if m.get("max_dd") is None else 100 * float(m["max_dd"]), lambda v, x: v <= x),
              ("min_return", None if m.get("total_return") is None else 100 * float(m["total_return"]),
               lambda v, x: v >= x),
              ("min_oos_return", None if (r.get("held_out") or {}).get("return") is None
               else 100 * float(r["held_out"]["return"]), lambda v, x: v >= x)]
    for key, val, ok in checks:
        lim = _f(c.get(key))
        if lim is not None and (val is None or not ok(float(val), lim)):
            return False
    return True


_series_cache: dict[str, dict[str, float]] = {}


def _series(entry: dict[str, Any]) -> dict[str, float]:
    """Daily returns of one strategy on its own bars (portfolio_view's run), cached for the engine's life."""
    if entry["id"] not in _series_cache:
        from engine.research.vault import portfolio_view

        one = portfolio_view.build_series([entry])
        _series_cache[entry["id"]] = one.get(entry["id"], {})
    return _series_cache[entry["id"]]


def correlation(a: dict[str, float], b: dict[str, float]) -> float | None:
    from engine.research.vault.portfolio_view import MIN_SHARED_DAYS

    days = sorted(set(a) & set(b))
    if len(days) < MIN_SHARED_DAYS:
        return None
    x = [a[d] for d in days]
    y = [b[d] for d in days]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0 or sy == 0:
        return None
    return sum((p - mx) * (q - my) for p, q in zip(x, y, strict=True)) / (sx * sy)


def pick_team(step: dict[str, Any], pool: list[dict[str, Any]], seated: list[dict[str, Any]],
              series: Callable[[dict[str, Any]], dict[str, float]] = _series, *,
              now: datetime | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fill the seats left: best score first, at most MAX PER MARKET / MAX PER TIMEFRAME, and skip a pick whose
    daily returns correlate above SIMILARITY LIMIT with any member (research PN 205). Returns (picks, skip counts)."""
    now = now or datetime.now(tz=UTC)
    size = int(step.get("target_size") or 30)
    per_sym, per_tf = int(step.get("max_per_symbol") or 0), int(step.get("max_per_tf") or 0)
    limit = float(step.get("max_avg_corr") or 0.99)
    team = list(seated)
    seated_ids = {r["id"] for r in team}
    cands = sorted((r for r in pool if r["id"] not in seated_ids and _passes(r, step, now)),
                   key=lambda r: (-float(r.get("score") or 0), r["id"]))[: max(1, CANDIDATE_FACTOR * size)]
    skips = {"market": 0, "timeframe": 0, "similar": 0}
    picks: list[dict[str, Any]] = []
    for r in cands:
        if len(team) >= size:
            break
        if per_sym and sum(1 for t in team if t["symbol"] == r["symbol"]) >= per_sym:
            skips["market"] += 1
            continue
        if per_tf and sum(1 for t in team if t["symbol"] == r["symbol"] and t["timeframe"] == r["timeframe"]) >= per_tf:
            skips["timeframe"] += 1
            continue
        if limit < 0.99:
            s = series(r)
            too = any((c := correlation(s, series(t))) is not None and c > limit for t in team)
            if too:
                skips["similar"] += 1
                continue
        team.append(r)
        picks.append(r)
    return picks, skips


def _live_ids() -> set[str]:
    from engine.research.deploy import full_deploy as fd

    try:
        return {b["source"]["vault_id"] for b in fd.load_bots() if b.get("status") != "REMOVED"}
    except Exception:
        return set()


def run_portfolio(step: dict[str, Any], run: Run) -> str:
    from engine.research.automation import portfolios

    p = portfolios.ensure(str(step["name"]))
    run.portfolio_id = p["id"]
    rows = {r["id"]: r for r in _vault()}
    live = _live_ids()
    mine = run.origin
    if step.get("mode") == "re_pick":  # rebuild: this automation's non-live members leave first
        drop = [m["id"] for m in p["members"] if m.get("by") == mine and m["id"] not in live]
        p = portfolios.remove(p["id"], drop)
    seated = [rows[m["id"]] for m in p["members"] if m["id"] in rows]
    use_run = bool(step.get("use_run_pool", True)) and run.campaigned
    pool = [rows[i] for i in run.run_pool if i in rows] if use_run else list(rows.values())
    from engine.research.vault import library

    copies = library.copies(list(rows.values()))
    pool = [r for r in pool if not (copies.get(r["id"]) or {}).get("copy_of")]
    run.phase(f"picking a team for {p['name']}", None)
    picks, skips = pick_team(step, pool, seated)
    if picks:
        p = portfolios.add(p["id"], [r["id"] for r in picks], by=mine)
        run.added += [r["id"] for r in picks]
    size = int(step.get("target_size") or 30)
    over = len(p["members"]) - size
    removed = 0
    if over > 0:  # lowering a limit removes the excess: this automation's non-live members, lowest score first
        cand = sorted((m for m in p["members"] if m.get("by") == mine and m["id"] not in live),
                      key=lambda m: float((rows.get(m["id"]) or {}).get("score") or 0))[:over]
        p = portfolios.remove(p["id"], [m["id"] for m in cand])
        removed = len(cand)
    why = [f"{v} too similar" if k == "similar" else f"{v} over MAX PER {'MARKET' if k == 'market' else 'TIMEFRAME'}"
           for k, v in skips.items() if v]
    short = len(p["members"]) < size
    return (f"added {len(picks)} member(s) to portfolio '{p['name']}' (now {len(p['members'])} of {size})"
            + (f", removed {removed} over the limit" if removed else "")
            + (f"; skipped {', '.join(why)}" if why else "")
            + ("; not enough variety to fill every seat" if short and why else ""))


# ---- PARITY ---------------------------------------------------------------------------------------------------------

def run_parity(step: dict[str, Any], run: Run) -> str:
    from engine.research.automation import portfolios
    from engine.research.vault import library, parity_job

    target = str(step.get("portfolio") or "") or run.portfolio_id
    if not target:
        return "no portfolio to test"
    p = portfolios.get(target)
    rows = {r["id"]: r for r in _vault()}
    mark = PASS_RANK.get(str(step.get("pass_mark") or "B"), 4)

    def needs(r: dict[str, Any]) -> bool:
        par = r.get("parity") or {}
        return par.get("state") != "DONE" or not par.get("grade")

    order = [m["id"] for m in p["members"] if m["id"] in run.added] + [m["id"] for m in p["members"]]
    todo, seen = [], set()
    for i in order:
        if i in rows and i not in seen and needs(rows[i]):
            todo.append(rows[i])
            seen.add(i)
    todo = todo[: int(step.get("tests_per_run") or 10)]
    graded: list[str] = []
    refused: dict[str, int] = {}
    from engine.core.paths import get_artifacts_dir

    for n, e in enumerate(todo, 1):
        if run.cancel.is_set():
            break
        run.phase(f"parity test {n}/{len(todo)} · {e['symbol']} {e['timeframe']}", {"done": n - 1, "total": len(todo)})
        try:
            parity_job.preflight(e)
            res = parity_job.run(e, get_artifacts_dir() / "parity" / e["id"].split("@")[0])
            library.set_parity(e["id"], res)
            graded.append(str(res.get("grade") or "?"))
        except parity_job.ParityRefused as exc:
            refused[str(exc)[:120]] = refused.get(str(exc)[:120], 0) + 1
    below = sum(1 for g in graded if PASS_RANK.get(g, 0) < mark)
    out = f"{len(graded)} tested ({', '.join(graded) or 'none'}), {below} below {step.get('pass_mark', 'B')}"
    if refused:
        out += "; could not run: " + "; ".join(f"{v} x {k}" for k, v in refused.items())
    if not todo:
        out = f"every member of '{p['name']}' already has a grade"
    return out


# ---- PURGE ----------------------------------------------------------------------------------------------------------

def purge_matches(criteria: dict[str, Any], rows: list[dict[str, Any]], protect: set[str],
                  now: datetime | None = None) -> list[str]:
    """Ids the PURGE step sends to the recycle bin: failing any set criterion; portfolio members never."""
    now = now or datetime.now(tz=UTC)
    c = {k: v for k, v in (criteria or {}).items() if _f(v) is not None}
    if not c:
        c = dict(DEFAULT_PURGE)
    out = []
    for r in rows:
        if r["id"] in protect:
            continue
        m = r.get("metrics") or {}
        ret = None if m.get("total_return") is None else 100 * float(m["total_return"])
        dd = None if m.get("max_dd") is None else 100 * float(m["max_dd"])
        oos = None if (r.get("held_out") or {}).get("return") is None else 100 * float(r["held_out"]["return"])
        fails = [
            "min_score" in c and float(r.get("score") or 0) < float(c["min_score"]),
            "min_trades" in c and int(m.get("n_trades") or 0) < float(c["min_trades"]),
            "max_dd" in c and dd is not None and dd > float(c["max_dd"]),
            "min_return" in c and ret is not None and ret < float(c["min_return"]),
            "min_oos_return" in c and oos is not None and oos < float(c["min_oos_return"]),
            "max_age_days" in c and float(c["max_age_days"]) > 0
            and now - datetime.fromisoformat(r["added_utc"]) > timedelta(days=float(c["max_age_days"])),
        ]
        if any(fails):
            out.append(r["id"])
    return out


def saved_or(criteria: dict[str, Any] | None) -> dict[str, Any]:
    """Blank criteria = the saved SETTINGS purge criteria (research), else the built-in defaults."""
    c = {k: v for k, v in (criteria or {}).items() if _f(v) is not None}
    if c:
        return c
    from engine.core import app_settings

    return dict(app_settings.load()["purge"]["criteria"]) or dict(DEFAULT_PURGE)


def run_purge(step: dict[str, Any], run: Run) -> str:
    from engine.research.automation import portfolios
    from engine.research.vault import library

    rows = _vault()
    protect = portfolios.protected_ids() | _live_ids()
    ids = purge_matches(saved_or(step.get("criteria")), rows, protect)
    if ids:
        library.remove_many(ids, reason=f"purge by automation '{run.auto['name']}'")
    return (f"{len(ids)} strateg{'y' if len(ids) == 1 else 'ies'} went to the recycle bin. "
            "You can restore them for 30 days.")


RUNNERS: dict[str, Callable[[dict[str, Any], Run], str]] = {
    "campaign": run_campaign, "portfolio": run_portfolio, "parity": run_parity, "purge": run_purge,
}


def preview(auto: dict[str, Any]) -> list[str]:
    """PREVIEW: what each step would do right now; nothing starts."""
    from engine.research.automation import portfolios

    lines: list[str] = []
    rows = _vault()
    for i, s in enumerate(auto.get("steps") or [], 1):
        k = s.get("kind")
        try:
            if k == "campaign":
                p = plan_campaign(s, auto)
                lines.append(f"STEP {i} CAMPAIGN – " + (p["error"] if p.get("error") else
                             f"{p['runs']} run(s): {', '.join(p['symbols']) or 'no market'} x "
                             f"{', '.join(p['timeframes'])} · {', '.join(str(y) for y in p['years'])}y · depth "
                             f"{p['depth']} · power {p['power']} · engine {'V1' if p['search'] == 'v1' else 'V2'}"
                             + (f" · skips {len(p['skipped_dead'])} dead" if p["skipped_dead"] else "")
                             + (f" · not offered: {', '.join(p['missing'])}" if p["missing"] else "")))
            elif k == "portfolio":
                try:
                    have = len(portfolios.get(str(s.get("name"))).get("members", []))
                except portfolios.PortfolioError:
                    have = 0
                ok = sum(1 for r in rows if _passes(r, s, datetime.now(tz=UTC)))
                lines.append(f"STEP {i} PORTFOLIO – '{s.get('name')}' has {have} of {s.get('target_size')}; "
                             f"{ok} vault strategies pass its criteria today (this run's campaign output is used "
                             f"when a campaign step comes first)")
            elif k == "parity":
                lines.append(f"STEP {i} PARITY – up to {s.get('tests_per_run')} member(s) without a grade, pass mark "
                             f"{s.get('pass_mark')}; needs a test MetaTrader 5 without bots")
            elif k == "purge":
                n = len(purge_matches(saved_or(s.get("criteria")), rows, portfolios.protected_ids() | _live_ids()))
                lines.append(f"STEP {i} PURGE – {n} strateg{'y' if n == 1 else 'ies'} would go to the recycle bin "
                             "(portfolio members and live bots never)")
            else:
                lines.append(f"STEP {i} {str(k).upper()} – not built yet: {PLANNED_WHY.get(str(k), '')}")
        except Exception as exc:
            lines.append(f"STEP {i} {str(k).upper()} – cannot preview: {type(exc).__name__}: {exc}")
    return lines


__all__ = ["BUILT", "GROUP_TEXT", "KINDS", "MARKET_GROUPS", "RUNNERS", "Run", "broker_names", "correlation",
           "default_step", "every_text", "group_counts", "pick_team", "plan_campaign", "preview", "purge_matches",
           "validate"]
