"""W5 POWER-CHECK (DR-2026-10-09-u4-wizard §4; AMENDMENT-U4-#2 R1 option A, #3-Q4; ARQVIO_U4_PROMPT_v2_EN §2 Power).

U2's power check (engine.research.ga.power_check) is used AS-IS, and it needs a campaign: its manifest_v2.json
(run.dsr_n_final) and individuals.json (the 200 hosts and the prior). A cell never searched has none, and designing a
campaign-free protocol is forbidden. So W5 verifies the cell is UNRUN with no ga-<cell_id>-* campaign directory, and
records the decision power_check = {status: BLOCKED:no-campaign, ref: docs/cells/<cell_id>.md#power}. Nothing is run
or invented. decide() is pure (no clock): a re-run in a fresh process is byte-identical. run() writes w5_power.json,
then applies the decision through the guarded registry ceremony (power_check and an evidence append only).
Refuses: an unknown cell, a cell with a campaign (U2's machinery applies), a power_check already decided.
"""
from __future__ import annotations

from typing import Any

from engine.research.wizard.common import (
    WizardRefused,
    cell_dir,
    cell_id,
    registry_ceremony,
    require_registry,
    write_product,
)

SCHEMA = "arqvio.wizard.w5_power"
DECISION = "BLOCKED:no-campaign"
DISCHARGE = ("the first R-chain campaign on this cell flips power_check to COMPUTED (computing stage named in ref), "
             "via U2's machinery as-is")


def campaign_dirs(cid: str) -> list[str]:
    from engine.core.paths import get_artifacts_dir

    ga = get_artifacts_dir() / "ga"
    return sorted(p.name for p in ga.glob(f"ga-{cid}-*") if p.is_dir()) if ga.is_dir() else []


def decide(symbol: str, tf: str) -> dict[str, Any]:
    """The W5 decision document for (symbol, tf). Refuses (raises) a cell that has, or can have, a campaign."""
    cid = cell_id(symbol, tf)
    entry = next((e for e in require_registry() if e["cell_id"] == cid and e["status"] == "VALIDATED"), None)
    if entry is None:
        raise WizardRefused("unknown-cell", f"{cid} is not a VALIDATED registry cell")
    dirs = campaign_dirs(cid)
    if entry["campaign_status"] != "UNRUN" or dirs:
        raise WizardRefused("campaign-run", f"{cid}: campaign_status {entry['campaign_status']!r}, campaign "
                                            f"directories {dirs}; U2's power check applies as-is")
    return {
        "schema": SCHEMA, "version": 1, "cell_id": cid, "symbol": entry["symbol"], "timeframe": entry["timeframe"],
        "decision": DECISION, "ref": f"docs/cells/{cid}.md#power",
        "ruling": "AMENDMENT-U4-#2 R1 option A; #3-Q4; ARQVIO_U4_PROMPT_v2_EN §2 Power",
        "why": ("U2's power check (engine.research.ga.power_check) is used as-is and needs a campaign: "
                "manifest_v2.json (run.dsr_n_final) and individuals.json (the 200 hosts and the prior); a "
                "campaign-free protocol is forbidden"),
        "checked": {"campaign_status": entry["campaign_status"], "campaign_dirs": dirs},
        "control_at_n": "not applicable: there is no N without a campaign (U2-Q4)",
        "discharge": DISCHARGE,
    }


def run(symbol: str, tf: str) -> dict[str, Any]:
    """Decide, write w5_power.json, then record power_check through the guarded ceremony. The product exists before
    the registry cites it; a refused ceremony removes a product this run created."""
    doc = decide(symbol, tf)
    cid = doc["cell_id"]
    product = f"<ARQVIO_HOME>/artifacts/cells/{cid}/w5_power.json"
    current = next(e for e in require_registry() if e["cell_id"] == cid)
    if current["power_check"]["status"] != "UNKNOWN":
        raise WizardRefused("power-decided", f"{cid}: power_check is already {current['power_check']}")

    def change(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for e in entries:
            if e["cell_id"] == cid:
                if e["power_check"]["status"] != "UNKNOWN":
                    raise WizardRefused("power-decided", f"{cid}: power_check is already {e['power_check']}")
                e["power_check"] = {"status": DECISION, "ref": doc["ref"]}
                e["evidence"] = [*e["evidence"], product]
        return entries

    path = cell_dir(cid) / "w5_power.json"
    existed = path.exists()
    write_product(path, doc)
    try:
        result = registry_ceremony(change, allowed={cid: frozenset({"power_check", "evidence"})})
    except BaseException:
        if not existed:
            path.unlink()
        raise
    return {"cell_id": cid, "decision": doc, **result}
