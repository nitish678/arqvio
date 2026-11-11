"""Emit an alert: AUDIT FIRST, then deliver (DR §6).

``emit`` appends the alert line to the spool (fsynced) BEFORE any adapter is called. Each adapter's result is then
appended as a DELIVERY line referencing the alert's seq. An adapter that raises is recorded as FAILED, never
propagated: delivery is best effort and never crashes the producer. The spool itself is the always-on adapter.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from engine.research.alerts.adapters import Delivery
from engine.research.alerts.schema import utc_now
from engine.research.alerts.spool import append


def emit(alert: dict, adapters: Iterable = (), *, path: Path | None = None,
         clock: Callable[[], str] = utc_now) -> dict:
    rec = append(alert, path)  # durable before any delivery attempt
    for a in adapters:
        try:
            res = a.deliver(rec)
        except Exception as exc:  # noqa: BLE001 -- best effort, audited
            res = Delivery("FAILED", f"adapter raised {type(exc).__name__}")
        append({"kind": "DELIVERY", "ref_seq": rec["seq"], "adapter": a.name, "status": res.status,
                "detail": res.detail, "emitted_utc": clock()}, path)
    return rec


__all__ = ["emit"]
