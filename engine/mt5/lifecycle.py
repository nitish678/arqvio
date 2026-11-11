"""MT5 connection lifecycle.

State machine:
    DISCONNECTED  →  CONNECTING  →  CONNECTED  →  SYNCING  →  LIVE
          ↓                ↓              ↓            ↓
       (any state) ──────────────────────────── DEGRADED (lost mid-session)

UI rules:
  - UI renders BEFORE the probe completes. Never block the window on a
    broker handshake. 3s timeout, 30s background retry.
  - On CONNECTED: compute the gap per cell, fetch only the gap, report
    "USDCADm H2: 37 bars added".
  - On DISCONNECTED: fall back to bundled/cached and say so.
  - Symbols with no local data render greyed out and unselectable, not blank.
"""
from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


class ConnectionState(str, enum.Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    SYNCING = "SYNCING"
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"


@dataclass
class ConnectionHeader:
    """The text rendered at the top of the UI."""

    state: ConnectionState
    last_synced_utc: str | None = None
    data_as_of_utc: str | None = None
    can_reconnect: bool = False

    def render(self) -> str:
        if self.state is ConnectionState.LIVE and self.last_synced_utc:
            delta = _humanise_delta(self.last_synced_utc)
            return f"MT5 LIVE · synced {delta} ago"
        if self.state is ConnectionState.DEGRADED:
            if self.data_as_of_utc:
                return (
                    f"MT5 DEGRADED · using cache (data as of "
                    f"{self.data_as_of_utc[:10]})"
                )
            return "MT5 DEGRADED · using cache"
        if self.state in (ConnectionState.DISCONNECTED, ConnectionState.CONNECTING):
            if self.data_as_of_utc:
                as_of = self.data_as_of_utc[:10]
                return f"MT5 OFFLINE · data as of {as_of}"
            return "MT5 OFFLINE · no data"
        return f"MT5 {self.state.value}"


def _humanise_delta(then_utc: str, now: datetime | None = None) -> str:
    then = datetime.fromisoformat(then_utc.replace("Z", "+00:00"))
    now = now or datetime.now(tz=UTC)
    delta = int((now - then).total_seconds())
    if delta < 60:
        return f"{delta}s"
    if delta < 3600:
        return f"{delta // 60}m"
    if delta < 86400:
        return f"{delta // 3600}h"
    return f"{delta // 86400}d"


@dataclass
class ConnectionController:
    """Tracks lifecycle state. The actual probe lives in engine.mt5.connection."""

    state: ConnectionState = ConnectionState.DISCONNECTED
    last_synced_utc: str | None = None
    data_as_of_utc: str | None = None
    transition_log: list[tuple[datetime, ConnectionState]] = field(default_factory=list)

    def transition(self, new: ConnectionState) -> None:
        if new is self.state:
            return
        self.transition_log.append((datetime.now(tz=UTC), new))
        self.state = new

    def header(self) -> ConnectionHeader:
        return ConnectionHeader(
            state=self.state,
            last_synced_utc=self.last_synced_utc,
            data_as_of_utc=self.data_as_of_utc,
            can_reconnect=self.state in (
                ConnectionState.DISCONNECTED,
                ConnectionState.DEGRADED,
            ),
        )


# Probe parameters. The 3 s timeout / 30 s retry pair is the spec.
PROBE_TIMEOUT_SECONDS = 3.0
RETRY_INTERVAL_SECONDS = 30.0


def default_probe_loop(
    controller: ConnectionController,
    probe_fn: Callable[[], bool],
    *,
    stop_condition: Callable[[], bool] | None = None,
) -> None:
    """Cooperative probe loop. Yields via ``time.sleep`` so the UI can run.

    The UI starts rendering immediately; this loop runs in the background.
    """
    while True:
        if stop_condition is not None and stop_condition():
            return
        controller.transition(ConnectionState.CONNECTING)
        try:
            ok = probe_fn()
        except Exception:
            ok = False
        if ok:
            controller.transition(ConnectionState.CONNECTED)
            return
        controller.transition(ConnectionState.DISCONNECTED)
        # sleep, then retry
        for _ in range(int(RETRY_INTERVAL_SECONDS)):
            if stop_condition is not None and stop_condition():
                return
            time.sleep(1.0)


__all__ = [
    "ConnectionController",
    "ConnectionHeader",
    "ConnectionState",
    "PROBE_TIMEOUT_SECONDS",
    "RETRY_INTERVAL_SECONDS",
    "default_probe_loop",
]
