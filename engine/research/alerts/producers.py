"""Alerts from the S7 producers (DR §6): drift fires, replay heat / refusal events.

Each function turns a producer's result into schema-valid alerts (``schema.make_alert``); emitting them is the
caller's ``dispatch.emit``. The health producer lives with the health snapshot (engine/research/health).
"""
from __future__ import annotations

from collections.abc import Callable

from engine.research.alerts.schema import make_alert, utc_now
from engine.research.portfolio.drift import MonitorResult


def from_drift(result: MonitorResult, source: str, *, clock: Callable[[], str] = utc_now) -> list[dict]:
    out = []
    for member in sorted({c.member for c in result.rejected}):
        ff = result.first_fire(member)
        rejected = [(c.t, c.p) for c in result.rejected if c.member == member]
        out.append(make_alert("DRIFT_FIRE", "crit", {"member": member, "first_t": ff.t, "p": ff.p, "d": ff.d,
                                                     "by_cut": result.cut, "rejected_checks": rejected}, source,
                              clock=clock))
    return out


def from_replay(events: list[dict], source: str, *, clock: Callable[[], str] = utc_now) -> list[dict]:
    out = []
    for e in events:
        if e["kind"] == "HEAT_HIGH":
            out.append(make_alert("HEAT_HIGH", "warn", {"t": e["t"], "heat": e["heat"], "unbounded": e["unbounded"],
                                                        "per_symbol": e["per_symbol"]}, source, clock=clock))
        elif e["kind"] == "REFUSED":
            out.append(make_alert("REFUSED", "warn", {"t": e["t"], "member": e["member"], "trade": e["trade"],
                                                      "cap": e["cap"], "values": e["values"]}, source, clock=clock))
    return out


__all__ = ["from_drift", "from_replay"]
