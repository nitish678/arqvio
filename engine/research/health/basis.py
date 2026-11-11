"""Time-basis labels for live account data (DR §7, G12) and the basis-drift DR's retry rule.

``probe_with_retry`` is DR-2026-10-01-mt5-live-basis-drift rule 4, implemented in the health layer: at most
RETRY_DELAYS_S + 1 = 3 probes of ``time_basis.probe_basis`` (the S3 machinery), 30 s and then 60 s apart. A resolved
basis on any attempt returns a label with the attempt count; three unresolved results return an ``unresolved``
label carrying all three deltas. The sleep and the clock are injected (tests never wait).

``basis_hour`` is the ONLY door for an hour-keyed rule on account data: with no label, or an ``unresolved`` label,
it REFUSES (``BasisRefused``). A ``broker_offset`` label converts with its measured offset; ``server_equals_utc``
passes through. ``to_utc`` is the same conversion for a timestamp.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from engine.research.data.time_basis import probe_basis

RETRY_DELAYS_S = (30.0, 60.0)
RESOLVED = ("server_equals_utc", "broker_offset")


class BasisRefused(ValueError):
    """An hour-keyed rule was handed account data without a resolved time-basis label (G12)."""


@dataclass(frozen=True)
class BasisLabel:
    basis: str
    measured_offset_s: int
    delta_s: float
    attempts: int
    deltas_s: tuple[float, ...]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["deltas_s"] = list(self.deltas_s)
        return d


def probe_with_retry(mt5: Any, symbol: str, *, sleep: Callable[[float], None] = time.sleep,
                     now: Callable[[], datetime] = lambda: datetime.now(tz=UTC)) -> BasisLabel:
    deltas: list[float] = []
    for attempt in range(1, len(RETRY_DELAYS_S) + 2):
        p = probe_basis(mt5, symbol, now=now())
        deltas.append(float(p.delta_s))
        if p.basis in RESOLVED:
            return BasisLabel(p.basis, int(p.measured_offset_s), float(p.delta_s), attempt, tuple(deltas))
        if attempt <= len(RETRY_DELAYS_S):
            sleep(RETRY_DELAYS_S[attempt - 1])
    return BasisLabel("unresolved", int(deltas[-1]), deltas[-1], len(deltas), tuple(deltas))


def _require(label: BasisLabel | dict | None) -> tuple[str, int]:
    if label is None:
        raise BasisRefused("refusing an hour-keyed rule on account data with NO time-basis label (G12)")
    basis = label["basis"] if isinstance(label, dict) else label.basis
    offset = label["measured_offset_s"] if isinstance(label, dict) else label.measured_offset_s
    if basis not in RESOLVED:
        raise BasisRefused(f"refusing an hour-keyed rule: time basis is {basis!r} (G12)")
    return basis, int(offset)


def to_utc(server_ts: int, label: BasisLabel | dict | None) -> int:
    basis, offset = _require(label)
    return int(server_ts) - (offset if basis == "broker_offset" else 0)


def basis_hour(server_ts: int, label: BasisLabel | dict | None) -> int:
    """The UTC hour of a server timestamp, through its account's basis label (refuses without one)."""
    return (to_utc(server_ts, label) // 3600) % 24


__all__ = ["BasisLabel", "BasisRefused", "RESOLVED", "RETRY_DELAYS_S", "basis_hour", "probe_with_retry", "to_utc"]
