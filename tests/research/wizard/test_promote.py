"""W4 REGISTRY PROMOTE on a COPY of the real registry (the committed file is never written): the entry built from
re-verified W1-W3 products, appended by the guarded ceremony with every existing entry byte-equal; every refusal; and
the ceremony's own guard (an unallowed field, a removed cell, a file changed underfoot)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from engine.research.data.dataset_snapshot import datasets_dir, save_snapshot
from engine.research.registry import cells as registry
from engine.research.wizard import pin as w2
from engine.research.wizard import promote as w4
from engine.research.wizard.__main__ import main
from engine.research.wizard.common import WizardRefused, cell_dir, registry_ceremony, write_product

AS_OF = "2026-09-24T18:39:39.338452+00:00"
ROW = "AAAm,H1,100000,16.1070,100000,2010-08-16T19:00:00+00:00,2026-09-24T18:00:00+00:00,MT5_LIVE,utc"


@pytest.fixture
def home(wizard_home: Path, tmp_path: Path, monkeypatch) -> dict:
    """A copy of the real registry, and AAAm H1's pin, files, audit row and W1-W3 products in the isolated home."""
    reg = tmp_path / "cells_registry.json"
    reg.write_bytes(registry.registry_path().read_bytes())
    monkeypatch.setattr(registry, "registry_path", lambda: reg)
    t = np.arange(1_700_000_000, 1_700_000_000 + 50 * 3600, 3600, dtype=np.int64)
    px = np.linspace(1.0, 1.1, t.size)
    meta = save_snapshot(symbol="AAAm", timeframe="H1", time=t, open_=px, high=px + 0.01, low=px - 0.01, close=px,
                         spread=np.full(t.size, 7))
    art = wizard_home / "artifacts"
    sha = {}
    for sub, text in (("spread", '{"x": 1}'), ("specs", '{"y": 2}')):
        (art / sub).mkdir(parents=True, exist_ok=True)
        (art / sub / "AAAm.json").write_text(text, encoding="utf-8")
        sha[sub] = hashlib.sha256((art / sub / "AAAm.json").read_bytes()).hexdigest()
    (art / "history_depth_audit_target.csv").write_text(f"symbol,tf\n# AS_OF {AS_OF}\nUSDCADm,H1,1\n{ROW}\n",
                                                         encoding="utf-8")
    base = {"cell_id": "aaam_h1", "symbol": "AAAm", "timeframe": "H1"}
    write_product(cell_dir("aaam_h1") / "w1_audit.json", base)
    holes = tmp_path / "aaam_h1.accepted_holes.json"
    holes.write_text(json.dumps({"schema": w2.HOLES_SCHEMA, "version": 1, "cell_id": "aaam_h1", "citation": "C",
                                 "evidence": {}, "holes": [{"missing_from": w2.iso(1_700_000_000 + 9 * 3600),
                                                            "missing_to": w2.iso(1_700_000_000 + 9 * 3600),
                                                            "bars_missing": 1, "seen_in": [], "note": "n"}]}),
                     encoding="utf-8")
    monkeypatch.setattr(w2, "accepted_holes_path", lambda cid: holes)
    acc = w2.load_accepted("aaam_h1", 3600)
    npz = next(datasets_dir().glob(f"AAAm_H1_{meta.dataset_id}.npz"))
    # the W2 manifest's real shape (engine/research/wizard/pin.py run()), not an invented one
    write_product(cell_dir("aaam_h1") / "w2_manifest.json", {
        **base, "dataset_hash16": meta.dataset_id, "n_bars": meta.n_bars,
        "window": {"first_bar": meta.start_utc, "last_bar": meta.end_utc},
        "files_sha256": {f"<ARQVIO_HOME>/artifacts/datasets/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (npz, npz.with_suffix(".json"))},
        "gaps": {"counts": {"weekend": 0, "pin_explained": 0, "owner_accepted_hole": 1, "unexplained": 0},
                 "acceptance": {"file": acc["file"], "canonical_sha256": acc["canonical_sha256"]}}})
    write_product(cell_dir("aaam_h1") / "w3_pin.json", {
        **base, "spread": {"sha256": sha["spread"], "provenance": {"calibrated_utc": "2026-09-24T08:52:29+00:00"}},
        "specs": {"sha256": sha["specs"], "provenance": {"snapshot_utc": "2026-09-23T15:58:19+00:00"}}})
    return {"reg": reg, "meta": meta, "before": json.loads(reg.read_text(encoding="utf-8"))}


def test_the_entry_is_appended_and_every_existing_entry_stays_byte_equal(home) -> None:
    out = w4.run("AAAm", "H1", cross_cell="usdcadm_h1")
    after = registry.load(home["reg"])
    old = home["before"]["entries"]
    assert [e["cell_id"] for e in after] == [e["cell_id"] for e in old] + ["aaam_h1"]
    assert after[:len(old)] == old and out["changed"] == {e["cell_id"]: [] for e in old}
    e = after[-1]
    meta = home["meta"]
    assert (e["status"], e["dataset_hash16"], e["cross_cell"], e["annualisation_ppy"]) == \
        ("VALIDATED", meta.dataset_id, "usdcadm_h1", 6048)
    assert e["dataset_window"] == {"start": meta.start_utc, "end": meta.end_utc, "bars": 50}
    assert (e["campaign_status"], e["power_check"], e["worker_status"]["status"], e["entered_by_stage"]) == \
        ("UNRUN", {"status": "UNKNOWN", "ref": "U4 W5"}, "UNKNOWN", "U4")
    assert e["audit_row_ref"] == {"file": "<ARQVIO_HOME>/artifacts/history_depth_audit_target.csv", "line": 4,
                                  "as_of": AS_OF, "row": ROW}
    assert e["evidence"] == ["docs/decisions/DR-2026-10-09-u4-wizard.md",
                             *(f"<ARQVIO_HOME>/artifacts/cells/aaam_h1/{n}" for n in w4.PRODUCTS),
                             "configs/cells/aaam_h1.accepted_holes.json"]
    assert "manifest_v2.json" not in e["spread_artifact_hash"]["source"]  # cites no campaign
    assert out["chain_after"] == json.loads(home["reg"].read_text(encoding="utf-8"))["chain_hash"]
    with pytest.raises(WizardRefused, match="cell-exists"):
        w4.run("AAAm", "H1", cross_cell="usdcadm_h1")


def _unchanged(home) -> bool:
    return json.loads(home["reg"].read_text(encoding="utf-8")) == home["before"]


@pytest.mark.parametrize("cross", ["eurusdm_h1", "aaam_h1", ""])
def test_a_cross_cell_that_is_not_another_validated_cell_refuses(home, cross) -> None:
    with pytest.raises(WizardRefused, match="cross-cell-invalid"):
        w4.run("AAAm", "H1", cross_cell=cross)
    assert _unchanged(home)


def _edit(name: str, change) -> None:
    p = cell_dir("aaam_h1") / name
    doc = json.loads(p.read_text(encoding="utf-8"))
    change(doc)
    write_product(p, doc)


@pytest.mark.parametrize("mutate, reason", [
    (lambda: (cell_dir("aaam_h1") / "w3_pin.json").unlink(), "product-missing"),
    (lambda: _edit("w1_audit.json", lambda d: d.update(cell_id="bbbm_h1")), "product-contradiction"),
    (lambda: _edit("w3_pin.json", lambda d: d["spread"].update(sha256="0" * 64)), "product-contradiction"),
    (lambda: _edit("w2_manifest.json", lambda d: d.update(n_bars=49)), "product-contradiction"),
    (lambda: _edit("w2_manifest.json", lambda d: d["gaps"]["counts"].update(unexplained=1)), "product-contradiction"),
    (lambda: _edit("w2_manifest.json", lambda d: d.update(dataset_hash16="0123456789abcdef")),
     "product-contradiction"),
    (lambda: _edit("w2_manifest.json", lambda d: d.pop("gaps")), "product-contradiction"),
    (lambda: _edit("w2_manifest.json", lambda d: d["gaps"]["acceptance"].update(canonical_sha256="0" * 64)),
     "product-contradiction"),
    (lambda: _edit("w2_manifest.json", lambda d: d.update(files_sha256={})), "product-contradiction"),
], ids=["missing", "other-cell", "spread-moved", "bars", "unexplained", "no-pin", "malformed", "holes-changed",
        "files-unrecorded"])
def test_a_missing_or_contradictory_product_refuses(home, mutate, reason) -> None:
    mutate()
    with pytest.raises(WizardRefused, match=reason):
        w4.run("AAAm", "H1", cross_cell="usdcadm_h1")
    assert _unchanged(home)


def test_a_tampered_pin_refuses(home) -> None:
    npz = next(datasets_dir().glob("AAAm_H1_*.npz"))
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["close"] = arrays["close"] + 1.0
    np.savez(npz, **arrays)
    with pytest.raises(WizardRefused, match="product-contradiction"):
        w4.run("AAAm", "H1", cross_cell="usdcadm_h1")
    assert _unchanged(home)


def test_a_sidecar_that_disagrees_with_its_bars_refuses(home) -> None:
    side = next(datasets_dir().glob("AAAm_H1_*.json"))
    doc = json.loads(side.read_text(encoding="utf-8"))
    doc["n_bars"] += 1
    side.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(WizardRefused, match="product-contradiction"):
        w4.run("AAAm", "H1", cross_cell="usdcadm_h1")
    assert _unchanged(home)


def test_an_audit_row_that_is_not_unique_refuses(home, wizard_home: Path) -> None:
    csv = wizard_home / "artifacts" / "history_depth_audit_target.csv"
    csv.write_text(csv.read_text(encoding="utf-8") + ROW + "\n", encoding="utf-8")
    with pytest.raises(WizardRefused, match="audit-row"):
        w4.run("AAAm", "H1", cross_cell="usdcadm_h1")
    assert _unchanged(home)


@pytest.mark.parametrize("change, kw", [
    (lambda es: [dict(es[0], cross_cell="usdcadm_h1"), *es[1:]], {}),
    (lambda es: es[1:], {}),
    (lambda es: [*es, dict(es[0], cell_id="zzzm_h1", symbol="ZZZm")], {}),
    (lambda es: [dict(es[0], power_check={"status": "X", "ref": "y"}), *es[1:]],
     {"allowed": {"usdcadm_h1": frozenset({"worker_status"})}}),
    (lambda es: [dict(es[0], schema_version=2.0), *es[1:]], {}),
    (lambda es: [dict(sorted(es[0].items())), *es[1:]], {}),
    (lambda es: [dict(es[0], dataset_window=dict(reversed(es[0]["dataset_window"].items()))), *es[1:]], {}),
], ids=["unallowed-field", "removed-cell", "undeclared-new-cell", "other-field-than-allowed", "int-to-float",
        "key-order", "nested-key-order"])
def test_the_ceremony_guard_refuses_any_diff_outside_the_declaration(home, change, kw) -> None:
    raw = home["reg"].read_bytes()
    with pytest.raises(WizardRefused, match="ceremony-diff"):
        registry_ceremony(change, **{"allowed": {}, **kw})
    assert home["reg"].read_bytes() == raw


def test_a_sealed_document_that_does_not_verify_never_touches_the_registry(home) -> None:
    raw = home["reg"].read_bytes()
    bad = lambda es: [*es, dict(es[0], cell_id="zzzm_h1", symbol="ZZZm", annualisation_ppy=1)]  # noqa: E731
    with pytest.raises(WizardRefused, match="registry-RegistrySchemaError"):
        registry_ceremony(bad, allowed={}, new=frozenset({"zzzm_h1"}))
    assert home["reg"].read_bytes() == raw and not list(home["reg"].parent.glob("*.ceremony"))


def test_the_ceremony_refuses_a_registry_changed_underfoot(home) -> None:
    def change(es):
        home["reg"].write_text(home["reg"].read_text(encoding="utf-8") + " ", encoding="utf-8")
        return es
    with pytest.raises(WizardRefused, match="ceremony-race"):
        registry_ceremony(change, allowed={})


def test_the_ceremony_writes_an_allowed_change_and_reverifies(home) -> None:
    out = registry_ceremony(lambda es: [dict(es[0], worker_status={**es[0]["worker_status"], "note": "n"}), *es[1:]],
                            allowed={"usdcadm_h1": frozenset({"worker_status"})})
    after = registry.load(home["reg"])
    old = home["before"]["entries"]
    assert out["changed"] == {e["cell_id"]: ["worker_status"] if i == 0 else [] for i, e in enumerate(old)}
    assert after[0]["worker_status"]["note"] == "n" and after[1:] == old[1:]


def test_the_cli_promote_needs_a_cross_cell(wizard_home: Path, tmp_path: Path) -> None:
    assert main(["promote", "--symbol", "AAAm", "--tf", "H1", "--out", str(tmp_path)]) == 2
    row = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert (row["step"], row["outcome"], row["reason"]) == ("promote", "REFUSED", "no-cross-cell")
