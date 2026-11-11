"""W7 RUNBOOK (DR-2026-10-09-u4-wizard §4; ARQVIO_U4_PROMPT_v2_EN §2 Runbook, §3 W7).

Renders docs/cells/<cell_id>.md from data only: the verified registry entry, the cell's W1/W2/W3/W5 products, its
admitted proof, the registry ceremony commits naming the cell, and any pre-registry search database on disk that
holds trials for the same (symbol, timeframe). Every number is read, none is typed. The page carries: what the cell is;
every pin and hash with its ceremony commit; how to re-prove it; Known Holes; Known Limitations; the power section
(the ref power_check points to) with the discharge sentence; the worker proof and its citation; the app semantics.
The W7S ceremony then appends the page to the entry's evidence (an evidence append only).
Refuses: an unknown or non-VALIDATED cell, a cell without an admitted proof, a missing product.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from engine.research.wizard.common import WizardRefused, cell_dir, cell_id, registry_ceremony, require_registry

PRODUCTS = ("w1_audit.json", "w2_manifest.json", "w3_pin.json", "w5_power.json")
APP_SEMANTICS = ("PROVEN-PRECAMPAIGN + BLOCKED:no-campaign + UNRUN = \"ready for first search\"; "
                 "PROVEN + COMPUTED = fully validated. campaign_status UNRUN is a first-class state, not a gap.")


def runbook_ref(cid: str) -> str:
    return f"docs/cells/{cid}.md"


def _ceremony_commits(cid: str) -> list[str]:
    from engine.core import paths

    out = subprocess.run(["git", "-C", str(paths.PROJECT_ROOT), "log", "--format=%h %s", "--",
                          "configs/cells_registry.json"], capture_output=True, text=True, check=True).stdout
    return [line for line in reversed(out.splitlines()) if cid in line]


def _prior_searches(symbol: str, tf: str) -> list[dict[str, Any]]:
    """Pre-registry search databases under <ARQVIO_HOME>/artifacts holding trials for (symbol, tf); read-only."""
    from engine.core.paths import get_artifacts_dir

    found = []
    for db in sorted(get_artifacts_dir().glob("*.sqlite")):
        try:
            with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as con:
                camps = con.execute("SELECT campaign_id, dataset_hash, created_at FROM campaigns WHERE symbol = ? "
                                    "AND timeframe = ?", (symbol, tf)).fetchall()
                counts = {c: con.execute("SELECT op_type, source, COUNT(*) FROM trials WHERE campaign_id = ? "
                                         "GROUP BY op_type, source", (c,)).fetchall() for c, _, _ in camps}
        except sqlite3.Error:
            continue  # not a campaign database
        for cid_, ds, created in camps:
            found.append({"file": f"<ARQVIO_HOME>/artifacts/{db.name}", "campaign": cid_, "dataset": ds,
                          "created": created, "trials": [f"{n} {op} ({src})" for op, src, n in counts[cid_]]})
    return found


def _spread_bases(entries: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
    """(registered cells whose spread file carries no time_basis, cells, their distinct calibration dates)."""
    from engine.data.spread_calibration import spread_dir

    cells = [e for e in entries if e["status"] == "VALIDATED"]
    docs = [json.loads((spread_dir() / f"{e['symbol']}.json").read_text(encoding="utf-8")) for e in cells]
    return (sum(d.get("time_basis") is None for d in docs), len(docs),
            sorted({str(d.get("calibrated_utc", ""))[:10] for d in docs}))


def render(symbol: str, tf: str) -> tuple[str, str]:
    """(cell_id, the runbook markdown). Refuses (raises) a cell that is not ready to document."""
    from engine.core import paths

    cid = cell_id(symbol, tf)
    entries = require_registry()
    entry = next((e for e in entries if e["cell_id"] == cid and e["status"] == "VALIDATED"), None)
    if entry is None:
        raise WizardRefused("unknown-cell", f"{cid} is not a VALIDATED registry cell")
    proofs = [r for r in entry["evidence"] if r.endswith(".proof.json")]
    if not proofs or entry["worker_status"]["status"] == "UNKNOWN":
        raise WizardRefused("no-proof", f"{cid}: no admitted proof in its evidence")
    prod = {}
    for name in PRODUCTS:
        p = cell_dir(cid) / name
        if not p.is_file():
            raise WizardRefused("product-missing", f"{p} is absent")
        prod[name] = json.loads(p.read_text(encoding="utf-8"))
    w1, w2, w3, w5 = (prod[n] for n in PRODUCTS)
    proof = json.loads((paths.PROJECT_ROOT / proofs[-1]).read_text(encoding="utf-8"))
    win, g = entry["dataset_window"], w2["gaps"]
    holes = g.get("owner_accepted_holes", [])
    held = g.get("owner_accepted_bars", 0)
    L = [f"# {cid} -- cell runbook", "",
         f"Rendered by `python -m engine.research.wizard runbook --symbol {entry['symbol']} --tf {tf}` from the "
         "registry and the cell's products (engine/research/wizard/runbook.py); every number below is read, none "
         "typed. Re-render, never hand-edit.", "",
         "## What the cell is", "",
         f"- {entry['symbol']} x {tf}, status {entry['status']}, entered by stage {entry['entered_by_stage']} "
         f"({entry['entered_at']}).",
         f"- cross_cell `{entry['cross_cell']}`; annualisation {entry['annualisation_ppy']} bars/year (the law "
         "table).",
         f"- campaign_status `{entry['campaign_status']}`: no R-chain GA campaign (`ga-{cid}-*`) exists for it.",
         f"- power_check `{entry['power_check']['status']}` ({entry['power_check']['ref']}); worker_status "
         f"`{entry['worker_status']['status']}`.", "",
         "## Pins and hashes", "",
         "| what | value |", "|---|---|",
         f"| dataset pin | `{entry['dataset_hash16']}`, {win['start']} .. {win['end']}, {win['bars']} bars |",
         *(f"| `{k}` | sha256 `{v}` |" for k, v in sorted(w2["files_sha256"].items())),
         f"| spread `{w3['spread']['file']}` | sha256 `{entry['spread_artifact_hash']['sha256']}` |",
         f"| specs `{w3['specs']['file']}` | sha256 `{entry['specs_artifact_hash']['sha256']}` |",
         f"| audit row | `{entry['audit_row_ref']['file']}`:{entry['audit_row_ref']['line']} (AS_OF "
         f"{entry['audit_row_ref']['as_of']}) |",
         f"| R4 threshold at the pin | {w2['threshold']['bars']} bars ({w2['threshold']['rule']}) |",
         *([f"| accepted-holes list | `{g['acceptance']['file']}`, canonical sha256 "
            f"`{g['acceptance']['canonical_sha256']}` |"] if g.get("acceptance") else []),
         "", "The registry ceremonies that wrote this entry:", "",
         *(f"- `{c}`" for c in _ceremony_commits(cid)), "",
         "## How to re-prove it", "",
         "```",
         f"python -m engine.research.reproof --cell {cid} --out <dir> --battery-log <a current closing battery log>",
         "```", "",
         "The registry selects the tier: UNRUN gives the pre-campaign tier (seven checks; power and rederive not "
         "applicable), RUN@<campaign> the full tier. Exit 0 PROVEN-PRECAMPAIGN@/PROVEN@, 1 BLOCKED, 2 REFUSED.", "",
         "## Known Holes", "",
         f"{len(holes)} owner-accepted hole(s), {held} bars of {w2['n_bars']} ({100 * held / w2['n_bars']:.2f}%), "
         "a closed list (never a rate):" if holes else "None: every gap is a weekend or explained by the trusted pins.",
         ""]
    if holes:
        L += [f"Citation: {g['acceptance']['citation']}.", "",
              "| missing from (UTC) | missing to (UTC) | bars | also missing in | note |", "|---|---|---|---|---|",
              *(f"| {h['missing_from']} | {h['missing_to']} | {h['bars_missing']} | "
                f"{', '.join(h['seen_in']) or '-'} | {h['note']} |" for h in holes), ""]
    basis = w3["spread"]["provenance"]["time_basis"]
    unbased, n_cells, dates = _spread_bases(entries)
    L += ["Gap tally at the pin: " + ", ".join(f"{k} {v}" for k, v in sorted(g["counts"].items())) + ". The gap "
          f"calendar is the registry's own pins ({', '.join(g['trusted_pins'])}); {w1['sessions']['note']}.", "",
          "## Known Limitations", "",
          "- **Spread time basis**: the spread file records "
          + (f"time_basis `{basis}`" if basis is not None else "no time_basis (absent, null)")
          + f"; the loader assumes UTC. Calibrated {w3['spread']['provenance']['calibrated_utc']}, "
          f"{w3['spread']['provenance']['sample_count']} samples; reused, not recalibrated. Across the registry, "
          f"{unbased} of {n_cells} VALIDATED cells' spread files carry no time_basis, calibrated on "
          f"{', '.join(dates)}" + (" (one batch: consistent across cells)" if len(dates) == 1 else "") + ".",
          f"- **Live drift vs the pinned specs** (W1, read {w1['read_utc']}, badge DETECTED; the pinned specs are "
          "used unchanged until a recalibration ruling):", "",
          "| field | pinned | live |", "|---|---|---|",
          *(f"| {k} | {w1['fields'][k]['pinned_value']} | {w1['fields'][k]['value']} |"
            for k in w1["pinned_specs"]["drift"]), "",
          f"- **Sessions**: badged {w1['sessions']['badge']}.",
          "- **Canary slice**: the reproof canary runs on the pin's first 1200 bars (FAST_BARS).",
          "- **Campaign-time constraint** (#2-R4): \"N >= 4,760\" binds a campaign, not this ceremony -- U2's power "
          "curve held the false-positive rate at N = 4,760 and not at N = 20 or 200.", ""]
    prior = _prior_searches(entry["symbol"], tf)
    L += ["- **Pre-registry searches on disk** (not R-chain campaigns; whether a first campaign's trial accounting "
          "counts them is an owner question):" if prior else "- **Pre-registry searches on disk**: none found.", ""]
    L += [f"  - `{p['file']}`: campaign `{p['campaign']}`, dataset `{p['dataset']}`, created {p['created']}; "
          f"trials {', '.join(p['trials'])}" for p in prior]
    L += ["", "## Power", "",
          f"power_check = `{w5['decision']}` ({w5['ruling']}).", "", w5["why"] + ".", "",
          f"Control at N: {w5['control_at_n']}.", "",
          f"Discharge: {w5['discharge']}.", "",
          "## Worker proof", "",
          f"worker_status = `{entry['worker_status']['status']}`, proof `{proofs[-1]}` (tier `{proof.get('tier')}`).",
          "", f"Citation: {entry['worker_status']['note']}", "",
          "Ran: " + ", ".join(f"{k} {'PASS' if v['ok'] else 'FAIL'}" for k, v in sorted(proof["checks"].items())) + ".",
          "",
          *(f"Not applicable: {k} -- {v}." for k, v in sorted(proof.get("not_applicable", {}).items())), "",
          "## App semantics", "", APP_SEMANTICS, ""]
    return cid, "\n".join(L)


def run(symbol: str, tf: str) -> dict[str, Any]:
    """Render the runbook, write it, and append it to the entry's evidence (the W7S ceremony)."""
    from engine.core import paths

    cid, text = render(symbol, tf)
    ref = runbook_ref(cid)
    path = paths.PROJECT_ROOT / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")

    def change(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for e in entries:
            if e["cell_id"] == cid and ref not in e["evidence"]:
                e["evidence"] = [*e["evidence"], ref]
        return entries

    result = registry_ceremony(change, allowed={cid: frozenset({"evidence"})})
    return {"cell_id": cid, "runbook": ref, **result}
