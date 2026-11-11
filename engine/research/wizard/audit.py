"""W1 AUDIT (DR-2026-10-09-u4-wizard §4): a read-only symbol inventory from install 1's terminal.

Every field is recorded with its source and read time, badged DETECTED (the attribute was read) or ABSENT (no such
attribute), and compared with the pinned specs file when one exists (MATCH / DRIFT). Detected facts are never merged
into anything: W3 validates the specs; W1 only reports. Session hours and the broker calendar are badged NOT-EXPOSED:
the MetaTrader5 package has no session or holiday call (its session_* names are per-symbol statistics). The time
basis comes from probe_basis, which needs a fresh tick; a closed market records "unresolved", never a guess.
Refuses: a symbol that is absent or not selectable, or one whose trade mode is DISABLED.
"""
from __future__ import annotations

import json
from typing import Any

from engine.research.wizard.common import (
    ACCOUNT_ID,
    WizardRefused,
    attach_acct1,
    cell_dir,
    cell_id,
    require_registry,
    utc_now,
    write_product,
)

SCHEMA = "arqvio.wizard.w1_audit"
TRADE_MODE_DISABLED = 0  # MetaTrader5.SYMBOL_TRADE_MODE_DISABLED
# The 21 broker keys of engine.research.data.symbol_specs.snapshot_from_mt5_symbol_info, plus the trade-mechanics keys.
FIELDS = (
    "name", "path", "visible", "digits", "point", "spread", "currency_base", "currency_profit", "trade_contract_size",
    "trade_tick_size", "trade_tick_value", "trade_tick_value_profit", "trade_tick_value_loss", "trade_stops_level",
    "trade_freeze_level", "volume_min", "volume_max", "volume_step", "swap_long", "swap_short", "swap_mode",
    "swap_rollover3days", "trade_mode", "trade_calc_mode", "trade_exemode", "filling_mode", "order_mode",
    "expiration_mode", "currency_margin",
)


def _plain(v: Any) -> Any:
    return v if isinstance(v, (bool, int, float, str)) or v is None else str(v)


def inventory(mt5: Any, symbol: str, pinned: dict[str, Any] | None, read_utc: str) -> dict[str, dict[str, Any]]:
    info = mt5.symbol_info(symbol)
    out: dict[str, dict[str, Any]] = {}
    for f in FIELDS:
        if hasattr(info, f):
            rec = {"value": _plain(getattr(info, f)), "badge": "DETECTED", "source": "MetaTrader5.symbol_info",
                   "read_utc": read_utc}
            if pinned is not None and f in pinned:
                rec["vs_pinned_specs"] = "MATCH" if pinned[f] == rec["value"] else "DRIFT"
                if rec["vs_pinned_specs"] == "DRIFT":
                    rec["pinned_value"] = pinned[f]
        else:
            rec = {"value": None, "badge": "ABSENT", "source": "MetaTrader5.symbol_info", "read_utc": read_utc}
        out[f] = rec
    return out


def run(symbol: str, tf: str, *, mt5: Any) -> dict[str, Any]:
    """Audit (symbol, tf); write <cells>/<cell_id>/w1_audit.json; return the document. Refuses (raises) otherwise."""
    from engine.research.data.symbol_specs import specs_dir
    from engine.research.data.time_basis import probe_basis

    cid = cell_id(symbol, tf)
    require_registry()
    acct = attach_acct1(mt5)
    try:
        if not mt5.symbol_select(symbol, True) or mt5.symbol_info(symbol) is None:
            raise WizardRefused("symbol-absent", f"{symbol!r} is not in {ACCOUNT_ID}'s catalog: {mt5.last_error()}")
        trade_mode = int(mt5.symbol_info(symbol).trade_mode)
        if trade_mode == TRADE_MODE_DISABLED:
            raise WizardRefused("trade-disabled", f"{symbol!r} trade_mode is DISABLED")
        read_utc = utc_now()
        pinned_path = specs_dir() / f"{symbol}.json"
        pinned = json.loads(pinned_path.read_text(encoding="utf-8")) if pinned_path.is_file() else None
        fields = inventory(mt5, symbol, pinned, read_utc)
        try:
            probe = probe_basis(mt5, symbol)
            basis = {"basis": probe.basis, "delta_s": probe.delta_s, "source": "time_basis.probe_basis",
                     "read_utc": read_utc}
        except Exception as exc:  # no tick (closed market): recorded, never guessed
            basis = {"basis": "unresolved", "error": f"{type(exc).__name__}: {exc}", "source": "time_basis.probe_basis",
                     "read_utc": read_utc}
        terminal = mt5.terminal_info()
        doc = {
            "schema": SCHEMA, "version": 1, "cell_id": cid, "symbol": symbol, "timeframe": tf, "read_utc": read_utc,
            "attach": {"account": ACCOUNT_ID, "terminal": acct["terminal"], "server": acct["server"],
                       "login_verified": True, "demo": True},
            "package_version": str(getattr(mt5, "__version__", "unknown")),
            "terminal_build": _plain(getattr(terminal, "build", None)),
            "fields": fields,
            "pinned_specs": {"path": f"<ARQVIO_HOME>/artifacts/specs/{symbol}.json", "present": pinned is not None,
                             "drift": sorted(f for f, r in fields.items() if r.get("vs_pinned_specs") == "DRIFT")},
            "sessions": {"badge": "NOT-EXPOSED",
                         "note": "the MetaTrader5 package has no session-hours or holiday call; its session_* names "
                                 "are per-symbol statistics. W2's gap rule takes its calendar from the registry's pins"},
            "time_basis": basis,
        }
    finally:
        mt5.shutdown()
    write_product(cell_dir(cid) / "w1_audit.json", doc)
    return doc
