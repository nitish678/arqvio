"""The analysis CLI on the pinned cells (isolated home): every artifact, one run_hash, canonical bytes, the drift
monitor reported-not-judged, and the alerts audited in the isolated spool."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research.alerts.spool import spool_path, tail, verify


def test_the_analysis_writes_every_artifact_and_audits_its_alerts(pinned_home: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.research.alerts.adapters import DESKTOP_VAR, SMTP_VARS, TG_VARS
    from engine.research.portfolio.analyze import analyze

    for var in TG_VARS + SMTP_VARS + (DESKTOP_VAR,):  # a test never sends for real
        monkeypatch.delenv(var, raising=False)

    out = pinned_home / "artifacts" / "portfolio"
    paths = analyze(out)
    assert sorted(paths) == ["allocation", "diversity", "drift", "heat", "replay"]
    docs = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    assert len({d["run_hash"] for d in docs.values()}) == 1
    assert docs["replay"]["verification"]["conservation_exact"] and docs["replay"]["verification"]["recon_ok"]
    assert docs["drift"]["judged"] is False and len(docs["drift"]["checks"]) == 8 * 5
    assert docs["heat"]["series"][0].keys() == {"t", "heat", "unbounded", "per_symbol"}
    assert docs["diversity"]["enb"]["equal"] == 8.0
    rep = verify(spool_path())
    rows = tail(1000, spool_path())
    heat_alerts = [r for r in rows if r["kind"] == "HEAT_HIGH"]
    assert rep.ok and len(heat_alerts) == len(docs["heat"]["crossings_t"])
    assert all(r["kind"] != "DELIVERY" or r["status"] in ("SKIPPED", "SENT") for r in rows)
    again = analyze(pinned_home / "again", alerts=False)
    assert all(again[k].read_bytes() == paths[k].read_bytes() for k in paths)  # byte-identical
