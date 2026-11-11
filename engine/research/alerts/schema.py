"""The alert event schema (DR §6): arqvio.alert v1 -- kind, severity, payload, source artifact, emitted_utc.

``emitted_utc`` comes from an INJECTED clock: it is recorded evidence and never enters logic. Kinds and severities
are closed sets; anything else is refused.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

KINDS = ("DRIFT_FIRE", "HEAT_HIGH", "REFUSED", "HEALTH_STALE", "HEALTH_FAIL", "BASIS_REFUSED")
SEVERITIES = ("info", "warn", "crit")


class AlertSchemaError(ValueError):
    """An alert outside the schema."""


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def make_alert(kind: str, severity: str, payload: dict, source: str, *,
               clock: Callable[[], str] = utc_now) -> dict:
    if kind not in KINDS:
        raise AlertSchemaError(f"unknown alert kind {kind!r}; known: {KINDS}")
    if severity not in SEVERITIES:
        raise AlertSchemaError(f"unknown severity {severity!r}; known: {SEVERITIES}")
    if not isinstance(payload, dict):
        raise AlertSchemaError("payload must be a JSON object")
    if not source:
        raise AlertSchemaError("an alert names its source artifact path")
    return {"kind": kind, "severity": severity, "payload": payload, "source": str(source), "emitted_utc": clock()}


__all__ = ["AlertSchemaError", "KINDS", "SEVERITIES", "make_alert", "utc_now"]
