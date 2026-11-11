"""Read-only MT5 live-detect probe (DR-2026-10-16-s9-community §2, owner decision ld-yes).

Permitted MT5 calls (enforced by a static allowlist scan in the test suite):
``initialize``, ``shutdown``, ``terminal_info``, ``account_info``,
``symbols_get``, ``symbol_info``.  Any trading/order/deal/history call is a
build failure -- this module attaches MetaTrader5 only to READ whether a
terminal answered, never to mutate it.

Failure states are first-class and honest:
  - ``MT5_NOT_RUNNING``     -- the terminal binary could not be found or did not answer.
  - ``MT5_PROBE_TIMEOUT``   -- the probe did not complete within the 5 s ceiling.
  - ``MT5_PROBE_ERROR``     -- the verbatim exception text from a call that failed.
  - ``MT5_PROBE_OK``        -- the terminal answered; carries terminal_info, account_info
                               (with a DEMO flag derived from ``trade_mode == 0``) and symbol_count.

The result is cached for 15 s (matches ``_probe_mt5_live`` in server.py) so the
environment endpoint does not re-initialise per request.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from engine.research.data.mt5_terminal import resolve_terminal

PROBE_TIMEOUT_S = 5.0
CACHE_TTL_S = 15.0

# The DR allowlist names six mutation-safe calls.  ``last_error`` is added here
# because it is the read-only error-inquiry companion to every call above -- it
# returns a tuple and mutates no state.  Any trading/order/deal/history call
# (the FORBIDDEN_CALLS set in tests) is still a build failure.
ALLOWED_CALLS = frozenset({
    "initialize",
    "shutdown",
    "terminal_info",
    "account_info",
    "symbols_get",
    "symbol_info",
    "last_error",
})

DEMO_MODE = 0  # MetaTrader5 ENUM_ACCOUNT_TRADE_MODE_DEMO


@dataclass
class Mt5LiveResult:
    """The honest, serialisable shape returned by the probe.

    ``state`` is one of the MT5_PROBE_* constants.  When state is OK the
    ``terminal_info`` and ``account_info`` dicts carry the raw field values
    from the API (only what the API returns; nothing invented).  When state
    is a failure the reason is verbatim.
    """

    state: str
    source: str
    as_of: str
    terminal_path: str | None
    terminal_source: str
    terminal_info: dict[str, Any] | None = None
    account_info: dict[str, Any] | None = None
    symbol_count: int | None = None
    detail: str | None = None
    candidates_checked: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "state": self.state,
            "source": self.source,
            "as_of": self.as_of,
            "terminal_path": self.terminal_path,
            "terminal_source": self.terminal_source,
            "terminal_info": self.terminal_info,
            "account_info": self.account_info,
            "symbol_count": self.symbol_count,
            "candidates_checked": list(self.candidates_checked),
        }
        if self.detail is not None:
            d["detail"] = self.detail
        return d


class _Mt5Shim(Protocol):
    """Minimal structural type for the MetaTrader5 module -- allows injection in tests."""

    def initialize(self, path: str | None = None, **kw: Any) -> bool: ...
    def shutdown(self) -> None: ...
    def terminal_info(self) -> Any | None: ...
    def account_info(self) -> Any | None: ...
    def symbols_get(self) -> list[Any] | None: ...
    def last_error(self) -> tuple[int, str]: ...

    def symbol_info(self, name: str) -> Any | None: ...


_cache: dict[str, tuple[float, Mt5LiveResult]] = {}
_cache_lock = threading.Lock()


def _fields(obj: Any) -> dict[str, Any]:
    """The struct's data fields. MetaTrader5 returns named tuples, whose ``count`` and ``index`` methods are not
    data (a method in the payload made the environment route fail to serialize with a live terminal)."""
    if obj is None:
        return {}
    if hasattr(obj, "_asdict"):
        return dict(obj._asdict())
    return {k: v for k in dir(obj) if not k.startswith("_") and not callable(v := getattr(obj, k))}


def _terminal_info_dict(ti: Any) -> dict[str, Any]:
    return _fields(ti)


def _account_info_dict(ai: Any) -> dict[str, Any]:
    if ai is None:
        return {}
    out = _fields(ai)
    if "trade_mode" in out:
        out["demo"] = int(out["trade_mode"]) == DEMO_MODE
    return out


def _probe_once(mt5: _Mt5Shim) -> Mt5LiveResult:
    """Run a single probe against the resolved terminal.

    The caller is responsible for timeout and caching.  This function performs
    only allowlisted calls.
    """
    res = resolve_terminal()
    if res.path is None:
        return Mt5LiveResult(
            state="MT5_NOT_RUNNING",
            source="mt5_terminal.resolve_terminal",
            as_of=_now_iso(),
            terminal_path=None,
            terminal_source=res.source,
            detail=f"terminal not found; checked: {list(res.candidates_checked) or '(none)'}",
            candidates_checked=res.candidates_checked,
        )

    terminal_path = res.path
    as_of = _now_iso()

    if not mt5.initialize(path=terminal_path):
        err = mt5.last_error()
        return Mt5LiveResult(
            state="MT5_NOT_RUNNING",
            source="mt5_live_probe",
            as_of=as_of,
            terminal_path=terminal_path,
            terminal_source=res.source,
            detail=f"initialize failed: {err}",
            candidates_checked=res.candidates_checked,
        )

    try:
        ti = mt5.terminal_info()
        if ti is None:
            err = mt5.last_error()
            return Mt5LiveResult(
                state="MT5_PROBE_ERROR",
                source="mt5_live_probe",
                as_of=as_of,
                terminal_path=terminal_path,
                terminal_source=res.source,
                detail=f"terminal_info() returned None: {err}",
                candidates_checked=res.candidates_checked,
            )

        ai = mt5.account_info()
        if ai is None:
            err = mt5.last_error()
            return Mt5LiveResult(
                state="MT5_PROBE_ERROR",
                source="mt5_live_probe",
                as_of=as_of,
                terminal_path=terminal_path,
                terminal_source=res.source,
                detail=f"account_info() returned None: {err}",
                candidates_checked=res.candidates_checked,
            )

        syms = mt5.symbols_get()
        if syms is None:
            err = mt5.last_error()
            return Mt5LiveResult(
                state="MT5_PROBE_ERROR",
                source="mt5_live_probe",
                as_of=as_of,
                terminal_path=terminal_path,
                terminal_source=res.source,
                detail=f"symbols_get() returned None: {err}",
                candidates_checked=res.candidates_checked,
            )

        return Mt5LiveResult(
            state="MT5_PROBE_OK",
            source="mt5_live_probe",
            as_of=as_of,
            terminal_path=terminal_path,
            terminal_source=res.source,
            terminal_info=_terminal_info_dict(ti),
            account_info=_account_info_dict(ai),
            symbol_count=len(syms),
            candidates_checked=res.candidates_checked,
        )
    except Exception as exc:
        return Mt5LiveResult(
            state="MT5_PROBE_ERROR",
            source="mt5_live_probe",
            as_of=as_of,
            terminal_path=terminal_path,
            terminal_source=res.source,
            detail=f"{type(exc).__name__}: {exc}",
            candidates_checked=res.candidates_checked,
        )
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _run_with_timeout(mt5: _Mt5Shim) -> Mt5LiveResult:
    """Run the probe in a worker thread with a hard 5 s ceiling.

    The MetaTrader5 module is not thread-safe in general, but we only call it
    from this one worker thread during the brief probe window, then shut down.
    """
    box: list[Mt5LiveResult | Exception | None] = [None]

    def worker() -> None:
        try:
            box[0] = _probe_once(mt5)
        except Exception as exc:  # noqa: BLE001 -- caught verbatim
            box[0] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=PROBE_TIMEOUT_S)

    if t.is_alive():
        return Mt5LiveResult(
            state="MT5_PROBE_TIMEOUT",
            source="mt5_live_probe",
            as_of=_now_iso(),
            terminal_path=None,
            terminal_source="timeout",
            detail=f"probe did not complete within {PROBE_TIMEOUT_S}s",
            candidates_checked=(),
        )
    if isinstance(box[0], Exception):
        exc = box[0]
        return Mt5LiveResult(
            state="MT5_PROBE_ERROR",
            source="mt5_live_probe",
            as_of=_now_iso(),
            terminal_path=None,
            terminal_source="error",
            detail=f"{type(exc).__name__}: {exc}",
            candidates_checked=(),
        )
    return box[0]  # type: ignore[return-value]


def probe_mt5_live(*, mt5: _Mt5Shim | None = None) -> Mt5LiveResult:
    """Public entry point: returns the cached or freshly-probed result.

    Pass an injected ``mt5`` (the real ``MetaTrader5`` module or a test double)
    to avoid a hard import of MetaTrader5 at module load time.  When ``mt5`` is
    ``None`` the real module is imported lazily inside the call.
    """
    key = _cache_key()
    now = _cache_time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None and (now - entry[0]) < CACHE_TTL_S:
            return entry[1]

    real_mt5: _Mt5Shim
    if mt5 is not None:
        real_mt5 = mt5
    else:
        try:
            import MetaTrader5 as _mt5  # noqa: PLC0415
        except ImportError as exc:
            # MetaTrader5 not installed (or blocked in a test): report honestly.
            return Mt5LiveResult(
                state="MT5_NOT_RUNNING",
                source="mt5_live_probe",
                as_of=_now_iso(),
                terminal_path=None,
                terminal_source="NOT_FOUND",
                detail=f"MetaTrader5 module unavailable: {type(exc).__name__}: {exc}",
                candidates_checked=(),
            )
        real_mt5 = _mt5  # type: ignore[assignment]

    result = _run_with_timeout(real_mt5)

    with _cache_lock:
        _cache[key] = (_cache_time(), result)

    return result


def _cache_key() -> str:
    """Cache is keyed on the resolved terminal path + its mtime, so a moved
    terminal or a new MT5 version (different binary mtime) is not served stale."""
    res = resolve_terminal()
    if res.path is None:
        return "NOT_FOUND"
    try:
        mtime = os.path.getmtime(res.path)
    except OSError:
        mtime = 0
    return f"{res.path}:{mtime}"


def _cache_time() -> float:
    import time
    return time.monotonic()


def clear_cache() -> None:
    """Test helper: drop the probe cache so the next call re-probes."""
    with _cache_lock:
        _cache.clear()


__all__ = [
    "ALLOWED_CALLS",
    "DEMO_MODE",
    "Mt5LiveResult",
    "PROBE_TIMEOUT_S",
    "clear_cache",
    "probe_mt5_live",
]
