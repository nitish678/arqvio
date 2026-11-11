"""Alerts from the health snapshot (DR §6/§7): STALE -> HEALTH_STALE (warn); UNAVAILABLE -> HEALTH_FAIL (crit);
an unresolved time basis on an UNAVAILABLE account -> BASIS_REFUSED (crit) as well. A STALE account whose clock is
unmeasurable (no fresh tick, S7 DEV-4) is HEALTH_STALE only. Emitting is the caller's ``alerts.dispatch.emit``."""
from __future__ import annotations

from collections.abc import Callable

from engine.research.alerts.schema import make_alert, utc_now


def from_health(snap: dict, source: str, *, clock: Callable[[], str] = utc_now) -> list[dict]:
    out = []
    for r in snap["accounts"]:
        who = {"id": r["id"], "login": r["login"], "server": r["server"]}
        basis = r.get("basis") or {}
        if basis.get("basis") == "unresolved" and r["status"] == "UNAVAILABLE":
            out.append(make_alert("BASIS_REFUSED", "crit", who | {"basis": basis}, source, clock=clock))
        if r["status"] == "STALE":
            out.append(make_alert("HEALTH_STALE", "warn", who | {"last_tick_age_s": r.get("last_tick_age_s"),
                                                                 "reason": r.get("reason")}, source, clock=clock))
        elif r["status"] == "UNAVAILABLE":
            out.append(make_alert("HEALTH_FAIL", "crit", who | {"reason": r.get("reason")}, source, clock=clock))
    return out


__all__ = ["from_health"]
