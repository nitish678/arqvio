"""Deploy refuses a broker whose clock differs from the clock a strategy learned its hours on.

Bars are pinned in UTC (a UTC server) or in a GMT+2/+3 server's own clock (``time_basis`` "server_time", owner
2026-11-01). The live EA reads hours with TimeCurrent() -- the target server's clock. A strategy bred on a UTC clock
and run on a GMT+3 server would open its session and close on Friday three hours off its backtest, so that deploy is
refused with the reason. A target server never measured yet (no ``time_basis/<server>.json``) is let through with a
warning: nothing is guessed either way.
"""
from __future__ import annotations

from typing import Any

from engine.research.data import server_clock


def _offset(server: str | None) -> int | None:
    """A measured server's offset from UTC in seconds; None when this server was never measured."""
    mem = server_clock.recall(server or "")
    if not mem:
        return None
    return 0 if mem.get("basis") == "server_equals_utc" else int(mem.get("measured_offset_s") or 0)


def _clock(offset: int) -> str:
    return "UTC" if offset == 0 else f"GMT{'+' if offset > 0 else '-'}{abs(offset) // 3600}"


def check(dataset_id: str | None, target_server: str | None) -> dict[str, Any]:
    """{"ok": bool, "warn": str | None, "reason": str | None} for deploying bars ``dataset_id`` to ``target_server``."""
    from engine.research.data.dataset_snapshot import load_meta

    try:
        meta = load_meta(dataset_id) if dataset_id else None
    except Exception:  # an old record without a readable snapshot: the guard has nothing to compare
        meta = None
    if meta is None:
        return {"ok": True, "warn": None, "reason": None}
    if meta.time_basis == "server_time":
        source, src_off = meta.server, _offset(meta.server)
    else:
        source, src_off = meta.server or "a UTC server", 0
    if target_server and target_server == meta.server:
        return {"ok": True, "warn": None, "reason": None}
    dst_off = _offset(target_server)
    if dst_off is None or src_off is None:
        return {"ok": True, "reason": None,
                "warn": f"{target_server}'s clock has not been measured yet; this strategy's hours were learned on "
                        f"{source}. Launch one campaign on this broker first so ARQVIO can check the two clocks match."}
    if dst_off == src_off:
        return {"ok": True, "warn": None, "reason": None}
    hours = abs(dst_off - src_off) // 3600
    return {"ok": False, "warn": None,
            "reason": f"this strategy learned its hours on a {_clock(src_off)} clock ({source}); {target_server} runs "
                      f"{_clock(dst_off)}, so its session and Friday-close hours would fire {hours} hour"
                      f"{'s' if hours != 1 else ''} off its backtest. Deploy it to a {_clock(src_off)} broker, or "
                      f"breed it again on {target_server}."}


__all__ = ["check"]
