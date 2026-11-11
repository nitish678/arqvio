"""W3 CALIBRATE = verify + pin (#2-R5, zero recalibration): the files are re-hashed against the recorded sha256s and
validated (load_spec, the spread loader, non-degenerate buckets, the gate, no override); every refusal writes no
product; the product is deterministic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.config import settings
from engine.data.spread_calibration import SpreadMatrix
from engine.research.wizard import calibrate as w3
from engine.research.wizard.__main__ import main
from engine.research.wizard.common import WizardRefused, cell_dir

SPECS = {"name": "AAAm", "path": "Standard\\Forex\\AAAm", "visible": True, "digits": 3, "point": 0.001, "spread": 22,
         "currency_base": "AAA", "currency_profit": "JPY", "trade_contract_size": 100000.0, "trade_tick_size": 0.001,
         "trade_tick_value": 0.63, "trade_tick_value_loss": 0.63, "trade_tick_value_profit": 0.63,
         "volume_min": 0.01, "volume_step": 0.01, "volume_max": 200.0, "swap_long": 0.0, "swap_short": -17.7,
         "swap_mode": 1, "swap_rollover3days": 3, "trade_freeze_level": 0, "trade_stops_level": 0,
         "server": "Fake-Demo", "account_login": 900001, "snapshot_utc": "2026-09-23T15:58:19+00:00", "source": "MT5"}
MEDIAN = [22.0] * 20 + [65.0, 210.0, 60.0, 22.0]
MEAN = [21.8 + 0.01 * h for h in range(20)] + [66.2, 213.9, 56.0, 21.9]


def matrix(buckets=MEDIAN, means=MEAN) -> str:
    return SpreadMatrix(symbol="AAAm", buckets=tuple(buckets), mean_buckets=tuple(means), sample_count=1401834,
                        source="MT5", calibrated_utc="2026-09-24T08:52:29+00:00").to_json()


def put(home: Path, spread: str | None = None, specs: dict | list | None = None) -> dict[str, str]:
    """Write the cell's files; returns their sha256s (the 'recorded' hashes)."""
    out = {}
    for kind, text in (("spread", matrix() if spread is None else spread),
                       ("specs", json.dumps(SPECS if specs is None else specs, indent=2))):
        p = home / "artifacts" / kind / "AAAm.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        out[kind] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def run(sha: dict[str, str]) -> dict:
    return w3.run("AAAm", "H1", spread_sha256=sha["spread"], specs_sha256=sha["specs"])


def no_product() -> bool:
    return not (cell_dir("aaam_h1") / "w3_pin.json").exists()


def test_the_files_are_verified_and_pinned(wizard_home: Path) -> None:
    sha = put(wizard_home)
    doc = run(sha)
    assert doc["spread"]["sha256"] == sha["spread"] and doc["specs"]["sha256"] == sha["specs"]
    assert doc["spread"]["provenance"] == {"calibrated_utc": "2026-09-24T08:52:29+00:00", "sample_count": 1401834,
                                           "source": "MT5", "calibrator_version": 2, "time_basis": "utc"}
    assert doc["specs"]["provenance"] == {"snapshot_utc": "2026-09-23T15:58:19+00:00", "server": "Fake-Demo",
                                          "account_login": 900001, "source": "MT5"}
    assert doc["spread"]["evaluate"]["status"] == "OK" and doc["spread"]["no_trade_hours"] == []
    assert doc["spread"]["mean_buckets_distinct_positive"] == 24 and doc["spread"]["spread_gate_override"] == {}
    product = cell_dir("aaam_h1") / "w3_pin.json"
    first = product.read_bytes()
    assert json.loads(first) == doc
    run(sha)
    assert product.read_bytes() == first  # no clock: deterministic


def test_a_missing_file_refuses(wizard_home: Path) -> None:
    sha = put(wizard_home)
    (wizard_home / "artifacts" / "specs" / "AAAm.json").unlink()
    with pytest.raises(WizardRefused, match="artifact-missing"):
        run(sha)
    assert no_product()


def test_a_hash_mismatch_refuses(wizard_home: Path) -> None:
    sha = put(wizard_home)
    with pytest.raises(WizardRefused, match="hash-mismatch"):
        run({**sha, "spread": "0" * 64})
    assert no_product()


@pytest.mark.parametrize("spread, specs, reason", [
    (None, {k: v for k, v in SPECS.items() if k != "digits"}, "specs-invalid"),
    (json.dumps({**json.loads(matrix()), "calibrator_version": 1}), None, "spread-invalid"),
    (matrix(means=[20.0] * 24), None, "degenerate-buckets"),
    (matrix(means=[0.0] * 23 + [20.0]), None, "degenerate-buckets"),
    (matrix(buckets=[22.0] * 24), None, "spread-gate-blocks"),
    ('["mean_buckets", "calibrator_version"]', None, "spread-invalid"),
    (json.dumps({**json.loads(matrix()), "sample_count": float("inf")}), None, "spread-invalid"),
    (None, [], "specs-invalid"),
], ids=["specs", "spread-v1", "flat-means", "one-positive-mean", "flat-row", "spread-list", "spread-inf",
        "specs-list"])
def test_an_invalid_file_refuses(wizard_home: Path, spread, specs, reason) -> None:
    sha = put(wizard_home, spread, specs)
    with pytest.raises(WizardRefused, match=reason):
        run(sha)
    assert no_product()


def test_a_gate_override_refuses(wizard_home: Path, monkeypatch) -> None:
    sha = put(wizard_home)
    def fake():
        return SimpleNamespace(max_spread_points_by_symbol={"AAAm": 30})

    fake.cache_clear = lambda: None  # isolated_arqvio_home clears the real cache at teardown
    monkeypatch.setattr(settings, "get_settings", fake)
    with pytest.raises(WizardRefused, match="spread-gate-override"):
        run(sha)
    assert no_product()


def test_a_registered_cell_refuses(wizard_home: Path) -> None:
    with pytest.raises(WizardRefused, match="cell-exists"):
        w3.run("USDCADm", "H1", spread_sha256="0" * 64, specs_sha256="0" * 64)


def test_the_cli_calibrate_needs_the_recorded_hashes(wizard_home: Path, tmp_path: Path) -> None:
    assert main(["calibrate", "--symbol", "AAAm", "--tf", "H1", "--out", str(tmp_path)]) == 2
    row = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert (row["step"], row["outcome"], row["reason"]) == ("calibrate", "REFUSED", "no-expected-hash")
