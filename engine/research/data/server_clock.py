"""The MT5 server's clock, measured even when the market is closed.

``time_basis.probe_basis`` compares a symbol's newest tick with the system clock. That only works while the symbol
trades: on a weekend or after the close the newest tick is hours or days old and the probe says "unresolved" -- every
launch then failed (owner's logs, 2026-10-24/25). A symbol never subscribed in Market Watch can even answer tick time 0.

``resolve`` therefore tries, in order:

1. the symbol itself (waiting briefly for a first tick after selecting it);
2. a symbol that trades around the clock (crypto: BTCUSD, ETHUSD, ...) when the broker offers one;
3. the clock this server was measured at before (``<home>/artifacts/time_basis/<server>.json``), for a server that
   was UTC -- a UTC server has no daylight-saving change, so the answer does not go stale;
4. on a weekend, the week's last tick against the weekly close: forex stops at 17:00 New York time on Friday, so a
   last tick a few seconds before that hour in server time names the server's offset (0, +1, +2 or +3 h). Owner
   2026-11-01: campaigns must run all Saturday and Sunday, also on a server never measured before.

Every fresh measurement is remembered. What cannot be measured stays "unresolved" and the caller refuses with the
reason (``closed_market``) -- nothing is guessed.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from engine.core import atomic
from engine.core.errors import DataError
from engine.core.paths import get_artifacts_dir
from engine.research.data.time_basis import BasisProbe, probe_basis

STALE_S = 600            # a newest tick older than this: the market is closed, the tick says nothing of the clock
FIRST_TICK_WAIT_S = 3.0  # a freshly selected symbol needs a moment before its first tick arrives
ALWAYS_OPEN = ("BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "SOLUSD", "BTCUSDT", "ETHUSDT")
MAX_TRIES = 4
# the weekly close (step 4): 17:00 New York on Friday; a server's offset from UTC is one of these whole hours
WEEKLY_CLOSE_NY_HOUR = 17
CLOSE_OFFSETS_S = (0, 3600, 7200, 10800)
CLOSE_TOLERANCE_S = 15 * 60  # the last tick lands up to this long before the close (thin Friday evenings)


@dataclass(frozen=True)
class Clock:
    probe: BasisProbe
    source: str          # "symbol" | "always-open <name>" | "remembered <utc>" | "unmeasured"
    newest_tick_age_s: float | None


def _memory(server: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", server or "unknown")
    return get_artifacts_dir() / "time_basis" / f"{safe}.json"


def remember(server: str, probe: BasisProbe, symbol: str, now: datetime) -> None:
    p = _memory(server)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"server": server, "basis": probe.basis, "measured_offset_s": probe.measured_offset_s,
                               "symbol": symbol, "utc": now.isoformat(timespec="seconds")}, indent=2) + "\n",
                   encoding="utf-8")
    atomic.replace(tmp, p)


def recall(server: str) -> dict[str, Any] | None:
    try:
        return json.loads(_memory(server).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _probe(mt5: Any, symbol: str, now: datetime) -> BasisProbe:
    """probe_basis, waiting up to FIRST_TICK_WAIT_S for a symbol that has no tick yet (time 0)."""
    deadline = time.monotonic() + FIRST_TICK_WAIT_S
    while True:
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None and int(getattr(tick, "time", 0)) > 0:
            return probe_basis(mt5, symbol, now=now)
        if time.monotonic() >= deadline:
            raise DataError(f"{symbol}: no tick from the terminal yet")
        time.sleep(0.25)


def _always_open(mt5: Any, symbol: str) -> list[str]:
    rows = mt5.symbols_get() or ()
    names = {str(getattr(s, "name", "")): bool(getattr(s, "visible", False)) for s in rows}
    out = []
    for base in ALWAYS_OPEN:
        for n, _visible in sorted(names.items(), key=lambda kv: (not kv[1], len(kv[0]))):
            if n != symbol and n.upper().startswith(base) and len(n) - len(base) <= 4 and n not in out:
                out.append(n)
                break
    return out[:MAX_TRIES]


def resolve(mt5: Any, symbol: str, *, server: str, now: datetime | None = None) -> Clock:
    """The server's clock basis for pinning ``symbol``'s bars; see the module docstring for the order."""
    now = now or datetime.now(tz=UTC)
    first: BasisProbe | None = None
    try:
        first = _probe(mt5, symbol, now)
    except DataError:
        first = None
    age = -first.delta_s if first is not None else None
    if first is not None and first.basis != "unresolved":
        remember(server, first, symbol, now)
        return Clock(first, "symbol", age)
    for name in _always_open(mt5, symbol):
        if not mt5.symbol_select(name, True):
            continue
        try:
            p = _probe(mt5, name, now)
        except DataError:
            continue
        if p.basis != "unresolved":
            remember(server, p, name, now)
            return Clock(p, f"always-open {name}", age)
    mem = recall(server)
    if mem and mem.get("basis") == "server_equals_utc":
        p = BasisProbe(basis="server_equals_utc", measured_offset_s=0,
                       delta_s=first.delta_s if first is not None else float("nan"), sample_n=0)
        return Clock(p, f"remembered {mem.get('utc')}", age)
    if first is not None:
        weekly = from_weekly_close(now.timestamp() + first.delta_s, now)
        if weekly is not None:
            remember(server, weekly, symbol, now)
            return Clock(weekly, "weekly close", age)
    unmeasured = first or BasisProbe(basis="unresolved", measured_offset_s=0, delta_s=float("nan"), sample_n=0)
    return Clock(replace(unmeasured, basis="unresolved"), "unmeasured", age)


def _us_dst(d: date) -> bool:
    """US daylight saving on day ``d`` (second Sunday of March to first Sunday of November); a Friday is never a
    switch day, so the date alone decides."""
    march = date(d.year, 3, 8) + timedelta(days=(6 - date(d.year, 3, 8).weekday()) % 7)
    november = date(d.year, 11, 1) + timedelta(days=(6 - date(d.year, 11, 1).weekday()) % 7)
    return march <= d < november


def last_weekly_close(now: datetime) -> datetime:
    """The most recent Friday 17:00 New York (as UTC) at or before ``now``."""
    d = now.astimezone(UTC).date()
    for back in range(8):
        day = d - timedelta(days=back)
        if day.weekday() != 4:
            continue
        hour = WEEKLY_CLOSE_NY_HOUR + (4 if _us_dst(day) else 5)
        close = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(hours=hour)
        if close <= now:
            return close
    raise AssertionError("a Friday close lies within any 8 days")


def from_weekly_close(tick_time: float, now: datetime) -> BasisProbe | None:
    """The server's offset from a last tick at the weekly close; None when the tick is not at the close."""
    close = last_weekly_close(now).timestamp()
    if now.timestamp() - close > 3 * 86400:
        return None  # not a weekend: a stale tick means something else (a holiday, a dead symbol)
    for off in CLOSE_OFFSETS_S:
        lead = close - (tick_time - off)
        if -60 <= lead <= CLOSE_TOLERANCE_S:
            return BasisProbe(basis="server_equals_utc" if off == 0 else "broker_offset", measured_offset_s=off,
                              delta_s=float(tick_time - now.timestamp()), sample_n=1)
    return None


def closed_market(clock: Clock) -> bool:
    """True when the clock could not be measured because the market is closed (the newest tick is stale)."""
    return clock.probe.basis == "unresolved" and (clock.newest_tick_age_s is None or clock.newest_tick_age_s > STALE_S)


__all__ = ["ALWAYS_OPEN", "Clock", "STALE_S", "closed_market", "from_weekly_close", "last_weekly_close", "recall",
           "remember", "resolve"]
