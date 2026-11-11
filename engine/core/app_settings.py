"""The owner's engine settings (SETTINGS tab): ``<home>/configs/app_settings.json``, validated, with defaults.

Only settings the engine really applies live here -- each key names where it is read:
- campaign.train_split          -> routes_app_launch: the share of the window breeding sees (the rest is held out)
- campaign.held_out_min_trades  -> V2 vault gate (vault/library.gates) and V1's vault gate (v1/cli.py)
- v1.*  (None = the depth preset's value) -> v1/cli.py: proving grounds, generations, tribe size, stagnation,
                                   drawdown limit, risk %
- purge.criteria                -> the vault PURGE and an automation PURGE step left blank
- auto_purge                    -> automation/runner.py: the scheduled vault sweep (research AUTOMATIC PURGE)
idea: research SETTINGS CAMPAIGN / SYSTEM tabs (RF 54202-54578); keys and code ARQVIO's.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from engine.core import atomic
from engine.core.paths import get_arqvio_home

SPLITS = (0.6, 0.7, 0.8)
DEFAULTS: dict[str, Any] = {
    "campaign": {"train_split": 0.7, "held_out_min_trades": 15},
    "v1": {"proving_grounds": None, "generations": None, "tribe_size": None, "stagnation_after": None,
           "drawdown_limit": None, "risk_pct": None},
    "purge": {"criteria": {}},
    "auto_purge": {"enabled": False, "every_hours": 24, "older_than_days": 45, "cap": 3000},
}
# (min, max, integer?) per number
RANGES: dict[str, tuple[float, float, bool]] = {
    "campaign.held_out_min_trades": (5, 200, True),
    "v1.proving_grounds": (100, 20000, True), "v1.generations": (1, 100, True), "v1.tribe_size": (4, 200, True),
    "v1.stagnation_after": (1, 50, True), "v1.drawdown_limit": (0.05, 0.9, False), "v1.risk_pct": (0.001, 0.05, False),
    "auto_purge.every_hours": (1, 720, True), "auto_purge.older_than_days": (0, 3650, True),
    "auto_purge.cap": (100, 100000, True),
}
PURGE_KEYS = ("min_score", "min_trades", "max_dd", "min_return", "min_oos_return", "max_age_days")


class SettingsError(ValueError):
    """A refused setting; the message names it."""


def path() -> Path:
    return get_arqvio_home() / "configs" / "app_settings.json"


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load() -> dict[str, Any]:
    try:
        stored = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    return _merge(DEFAULTS, {k: v for k, v in stored.items() if k in DEFAULTS})


def validate(s: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if s["campaign"]["train_split"] not in SPLITS:
        out.append(f"campaign.train_split must be one of {list(SPLITS)}")
    for key, (lo, hi, whole) in RANGES.items():
        sec, name = key.split(".")
        v = s[sec].get(name)
        if v is None and sec == "v1":
            continue
        if isinstance(v, bool) or not isinstance(v, int | float) or not lo <= v <= hi or (whole and v != int(v)):
            out.append(f"{key} must be {'a whole number ' if whole else ''}from {lo:g} to {hi:g}")
    crit = s["purge"]["criteria"]
    for k, v in crit.items():
        if k not in PURGE_KEYS:
            out.append(f"purge.criteria.{k} is not a purge criterion")
        elif v is not None and (isinstance(v, bool) or not isinstance(v, int | float)):
            out.append(f"purge.criteria.{k} must be a number or blank")
    if not isinstance(s["auto_purge"].get("enabled"), bool):
        out.append("auto_purge.enabled must be true or false")
    return out


def update(patch: dict[str, Any]) -> dict[str, Any]:
    """Merge a partial update, validate the whole, save; refused as a whole with every problem named."""
    unknown = sorted(set(patch) - set(DEFAULTS))
    if unknown:
        raise SettingsError(f"unknown settings section(s): {', '.join(unknown)}")
    merged = _merge(load(), patch)
    if "purge" in patch and "criteria" in (patch["purge"] or {}):  # the criteria list is replaced, not merged
        merged["purge"]["criteria"] = {k: v for k, v in patch["purge"]["criteria"].items() if v is not None}
    problems = validate(merged)
    if problems:
        raise SettingsError("; ".join(problems))
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    atomic.replace(tmp, p)
    return merged


def v1_overrides() -> dict[str, Any]:
    """The owner's V1 numbers that differ from the preset (None entries left out)."""
    return {k: v for k, v in load()["v1"].items() if v is not None}


__all__ = ["DEFAULTS", "PURGE_KEYS", "SPLITS", "SettingsError", "load", "update", "v1_overrides", "validate"]
