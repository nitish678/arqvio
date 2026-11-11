"""SETTINGS (``/api/v2/app/settings``): the owner's engine settings, the config echo of the next launch, and the
local bug report (research SUPPORT: what happened + contact; ARQVIO sends nothing over the network, so the report is
saved in the data folder and copied for the owner to send).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.core import app_settings

router = APIRouter(prefix="/api/v2/app/settings", tags=["app-settings"])


@router.get("")
def get_settings() -> dict[str, Any]:
    return {"settings": app_settings.load(), "defaults": app_settings.DEFAULTS, "splits": list(app_settings.SPLITS)}


@router.patch("")
def patch_settings(body: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"settings": app_settings.update(body)}
    except app_settings.SettingsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/echo")
def echo(depth: str = "STANDARD", engine: str = "standard") -> dict[str, Any]:
    """CONFIG ECHO: what the next launch at this depth and engine resolves to, from the settings in force now."""
    from engine.api.routes_app_campaigns import presets
    from engine.api.routes_app_launch import DEPTHS, V1_PRESETS

    s = app_settings.load()
    out: dict[str, Any] = {"train_split": s["campaign"]["train_split"],
                           "held_out": round(1 - s["campaign"]["train_split"], 4),
                           "held_out_min_trades": s["campaign"]["held_out_min_trades"], "depth": depth, "engine": engine}
    if engine == "v1":
        from engine.research.v1.config import v1_config

        over = app_settings.v1_overrides()
        risk = over.pop("risk_pct", 0.01)
        try:
            cfg = v1_config(V1_PRESETS.get(depth, "research"), risk_pct=risk,
                            held_out_min_trades=s["campaign"]["held_out_min_trades"], **over)
            out["v1"] = {k: v for k, v in cfg.block().items() if k in (
                "preset", "proving_grounds", "generations", "tribe_size", "stagnation_after", "drawdown_limit",
                "risk_pct", "finalists", "held_out_min_trades", "random_n", "shuffle_n")}
        except ValueError as exc:
            out["error"] = str(exc)
    else:
        p = presets()["presets"].get(DEPTHS.get(depth, ""), {})
        out["v2"] = {k: p.get(k) for k in ("pop", "gen", "budget", "final_n")}
    line = [f"split={s['campaign']['train_split']:.2f}/{1 - s['campaign']['train_split']:.2f}",
            f"oos_min_trades={s['campaign']['held_out_min_trades']}", f"engine={'V1' if engine == 'v1' else 'V2'}",
            f"depth={depth}"]
    line += [f"{k}={v}" for k, v in (out.get("v1") or out.get("v2") or {}).items()]
    out["line"] = "[CONFIG] " + " ".join(line)
    return out


class BugReport(BaseModel):
    what: str = Field(min_length=3, max_length=8000)
    contact: str = Field("", max_length=200)


@router.post("/bug-report")
def bug_report(body: BugReport) -> dict[str, Any]:
    """SAVE BUG REPORT: what happened + contact + the support report, as a text file in the data folder."""
    from engine.api.routes_app_support import report_text
    from engine.core.paths import get_arqvio_home

    now = datetime.now(tz=UTC)
    text = (f"ARQVIO BUG REPORT {now.isoformat(timespec='seconds')}\n\nWHAT HAPPENED\n{body.what.strip()}\n\n"
            f"CONTACT\n{body.contact.strip() or '(none given)'}\n\n{report_text()}\n")
    d = get_arqvio_home() / "support"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"bug_{now.strftime('%Y%m%dT%H%M%SZ')}.txt"
    p.write_text(text, encoding="utf-8")
    return {"path": str(p), "text": text}


__all__ = ["router"]
