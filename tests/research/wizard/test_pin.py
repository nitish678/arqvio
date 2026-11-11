"""W2 DATASET PIN on a fake terminal against synthetic trusted pins: the gap rule (weekend by rule, EXPLAINED by every
trusted pin, an owner-accepted hole only by EXACT match with the cell's closed list, else refuse), the R4 threshold
read from the registry, the time-basis gate, and the no-overwrite pin. The last block pins gbpjpym_h1's ceremony list
(AMENDMENT-U4-#4) to exactly its five holes."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine.research.data import time_basis
from engine.research.data.dataset_snapshot import datasets_dir, load_snapshot, save_snapshot
from engine.research.wizard import pin as w2
from engine.research.wizard.__main__ import main
from engine.research.wizard.common import WizardRefused, canonical, cell_dir
from tests.research.wizard.conftest import FakeMT5

H = 3600
START, END = "2024-01-01", "2024-01-15"  # Monday to Monday
T0, T1 = (int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()) for d in (START, END))
HOLIDAY = int(datetime(2024, 1, 3, 11, tzinfo=UTC).timestamp())  # a Wednesday
ODD = int(datetime(2024, 1, 9, 5, tzinfo=UTC).timestamp())  # a Tuesday
ODD2 = int(datetime(2024, 1, 10, 7, tzinfo=UTC).timestamp())  # a Wednesday


def day(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())


def hours(*drop: int, t0: int = T0, t1: int = T1) -> np.ndarray:
    """Every hour of [t0, t1] but the weekend close (Friday 21:00 to Sunday 22:00 UTC) and `drop`."""
    t = np.arange(t0, t1 + H, H, dtype=np.int64)
    d = [datetime.fromtimestamp(int(x), tz=UTC) for x in t]
    keep = [not (x.weekday() == 4 and x.hour >= 21 or x.weekday() == 5 or x.weekday() == 6 and x.hour < 22)
            for x in d]
    t = t[np.array(keep)]
    return t[~np.isin(t, np.array(drop, dtype=np.int64))]


def rates(t: np.ndarray) -> np.ndarray:
    dt = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
          ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]
    r = np.zeros(t.size, dtype=dt)
    r["time"] = t
    px = 1.0 + (t - T0) / 1e7
    r["open"], r["high"], r["low"], r["close"], r["spread"] = px, px + 0.01, px - 0.01, px, 7
    return r


class PinMT5(FakeMT5):
    TIMEFRAME_H1 = 16385

    def __init__(self, t: np.ndarray, symbol: str = "AAAm", **kw):
        super().__init__({symbol: {}}, **kw)
        self.t = t

    def copy_rates_range(self, symbol, tf, start, end):
        self.pulled = (symbol, tf, start, end)
        return rates(self.t)


def pinned(symbol: str, t: np.ndarray) -> str:
    r = rates(t)
    return save_snapshot(symbol=symbol, timeframe="H1", time=r["time"], open_=r["open"], high=r["high"], low=r["low"],
                         close=r["close"], spread=r["spread"]).dataset_id


def stub(monkeypatch, tmp_path: Path, pins: list[tuple[str, np.ndarray, int]]) -> list[dict]:
    """A registry stub naming the given trusted H1 pins; time basis proven; acceptance lists read from tmp_path."""
    entries = [{"cell_id": f"{s.lower()}_h1", "symbol": s, "timeframe": "H1", "status": "VALIDATED",
                "dataset_hash16": pinned(s, t), "dataset_window": {"bars": bars}} for s, t, bars in pins]
    monkeypatch.setattr(w2, "require_registry", lambda: entries)
    monkeypatch.setattr(time_basis, "probe_basis", lambda mt5, s: SimpleNamespace(basis="server_equals_utc"))
    monkeypatch.setattr(w2, "accepted_holes_path", lambda cid: tmp_path / f"{cid}.accepted_holes.json")
    return entries


@pytest.fixture
def setup(wizard_home: Path, tmp_path: Path, monkeypatch):
    """Two trusted H1 pins sharing a Wednesday holiday; BBBm alone also misses ODD."""
    holiday = [HOLIDAY - H, HOLIDAY, HOLIDAY + H]
    return stub(monkeypatch, tmp_path, [("BBBm", hours(*holiday, ODD), 230), ("CCCm", hours(*holiday), 220)])


def test_the_gap_rule() -> None:
    t = hours()
    assert w2.gaps(t, H, T0, T1) == [(int(t[i]), int(t[i + 1])) for i in np.flatnonzero(np.diff(t) > H)]
    assert all(w2.is_weekend(*g) for g in w2.gaps(t, H, T0, T1)) and len(w2.gaps(t, H, T0, T1)) == 2
    assert w2.gaps(t[5:], H, T0, T1)[0] == (T0 - H, int(t[5]))  # the window's edge is a gap
    assert not w2.is_weekend(HOLIDAY - H, HOLIDAY + H)
    pin = {"p": w2.gaps(hours(HOLIDAY - H, HOLIDAY), H, T0, T1)}
    out = w2.classify(hours(HOLIDAY), H, pin, T0, T1)
    assert out["weekend"] == 2 and not out["unexplained"]
    assert out["explained"][0]["explained_by"]["p"][0]["missing_from"].startswith("2024-01-03T10:00")
    assert w2.classify(hours(ODD), H, pin, T0, T1)["unexplained"][0]["seen_in"] == []


def test_the_pin_writes_the_snapshot_and_the_manifest(setup) -> None:
    fake = PinMT5(hours(HOLIDAY))
    doc = w2.run("AAAm", "H1", START, END, mt5=fake)
    assert fake.pulled[1] == PinMT5.TIMEFRAME_H1 and fake.kw["login"] == 900001 and fake.shut
    assert doc["threshold"]["bars"] == 220 and doc["n_bars"] == hours(HOLIDAY).size
    g = doc["gaps"]
    assert g["trusted_pins"] == ["bbbm_h1", "cccm_h1"]
    assert g["counts"] == {"weekend": 2, "pin_explained": 1, "owner_accepted_hole": 0, "unexplained": 0}
    assert g["acceptance"] is None and g["owner_accepted_holes"] == [] and g["owner_accepted_bars"] == 0
    assert sorted(g["explained"][0]["explained_by"]) == ["bbbm_h1", "cccm_h1"]
    assert np.array_equal(load_snapshot(doc["dataset_hash16"]).time, hours(HOLIDAY))
    assert sorted(doc["files_sha256"]) == [f"<ARQVIO_HOME>/artifacts/datasets/AAAm_H1_{doc['dataset_hash16']}.{x}"
                                           for x in ("json", "npz")]
    assert json.loads((cell_dir("aaam_h1") / "w2_manifest.json").read_text(encoding="utf-8")) == doc


def _no_pin(symbol: str = "AAAm") -> bool:
    return (not list(datasets_dir().glob(f"{symbol}_*"))
            and not (cell_dir(f"{symbol.lower()}_h1") / "w2_manifest.json").exists())


def test_an_unexplained_gap_refuses_and_pins_nothing(setup) -> None:
    with pytest.raises(WizardRefused, match="unexplained-gap") as exc:
        w2.run("AAAm", "H1", START, END, mt5=PinMT5(hours(ODD)))
    assert "'seen_in': ['bbbm_h1']" in exc.value.detail and _no_pin()


def test_below_the_threshold_refuses(setup) -> None:
    setup[0]["dataset_window"]["bars"] = 10**6
    n = hours().size
    setup[1]["dataset_window"]["bars"] = n + 1  # the threshold is the minimum over the registered cells
    with pytest.raises(WizardRefused, match=f"small-dataset: {n} bars < the R4 threshold {n + 1}"):
        w2.run("AAAm", "H1", START, END, mt5=PinMT5(hours()))
    assert _no_pin()


def test_an_unproven_time_basis_refuses(setup, monkeypatch) -> None:
    monkeypatch.setattr(time_basis, "probe_basis", lambda mt5, s: SimpleNamespace(basis="unresolved"))
    fake = PinMT5(hours())
    with pytest.raises(WizardRefused, match="time-basis"):
        w2.run("AAAm", "H1", START, END, mt5=fake)
    assert fake.shut and not hasattr(fake, "pulled") and _no_pin()


def test_a_pin_is_never_rewritten(setup) -> None:
    w2.run("AAAm", "H1", START, END, mt5=PinMT5(hours()))
    with pytest.raises(WizardRefused, match="pin-exists"):
        w2.run("AAAm", "H1", START, END, mt5=PinMT5(hours()))


def test_a_tampered_trusted_pin_refuses_before_any_attach(setup) -> None:
    npz = next(datasets_dir().glob(f"BBBm_H1_{setup[0]['dataset_hash16']}.npz"))
    r = rates(hours())
    np.savez(npz, **{k: r[k] for k in r.dtype.names})
    fake = PinMT5(hours())
    with pytest.raises(WizardRefused, match="trusted-pin-unverifiable"):
        w2.run("AAAm", "H1", START, END, mt5=fake)
    assert not hasattr(fake, "kw")


def test_a_registered_cell_refuses(setup) -> None:
    with pytest.raises(WizardRefused, match="cell-exists"):
        w2.run("BBBm", "H1", START, END, mt5=PinMT5(hours()))


def test_the_cli_pin_needs_a_window(wizard_home: Path, tmp_path: Path) -> None:
    assert main(["pin", "--symbol", "AAAm", "--tf", "H1", "--out", str(tmp_path)]) == 2
    row = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert (row["step"], row["outcome"], row["reason"]) == ("pin", "REFUSED", "no-window")


# --- owner-accepted holes (#4-Q5): a closed list, matched EXACTLY; no rate anywhere ---

def hole(t: int, seen_in: list[str], bars: int = 1) -> dict:
    return {"missing_from": w2.iso(t), "missing_to": w2.iso(t + (bars - 1) * H), "bars_missing": bars,
            "seen_in": seen_in, "note": "test"}


def holes_doc(cid: str, holes: list[dict]) -> dict:
    return {"schema": w2.HOLES_SCHEMA, "version": 1, "cell_id": cid, "citation": "AMENDMENT-TEST",
            "evidence": {"diag_log": "test"}, "holes": holes}


def write_holes(cid: str, doc: dict) -> Path:
    path = w2.accepted_holes_path(cid)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_an_exactly_listed_hole_is_owner_accepted(setup) -> None:
    listed = holes_doc("aaam_h1", [hole(ODD, ["bbbm_h1"])])
    write_holes("aaam_h1", listed)
    doc = w2.run("AAAm", "H1", START, END, mt5=PinMT5(hours(ODD)))
    g = doc["gaps"]
    assert g["counts"] == {"weekend": 2, "pin_explained": 0, "owner_accepted_hole": 1, "unexplained": 0}
    assert g["owner_accepted_holes"] == [{**hole(ODD, ["bbbm_h1"]), "category": "owner_accepted_hole",
                                         "citation": "AMENDMENT-TEST"}]
    assert g["owner_accepted_bars"] == 1
    assert g["acceptance"] == {"file": "configs/cells/aaam_h1.accepted_holes.json", "citation": "AMENDMENT-TEST",
                               "canonical_sha256": hashlib.sha256(canonical(listed).encode()).hexdigest(),
                               "evidence": {"diag_log": "test"}}


@pytest.mark.parametrize("drop, listed, reason", [
    ((ODD, ODD2), [hole(ODD, ["bbbm_h1"])], "unexplained-gap"),  # a hole off the list
    ((ODD + H,), [hole(ODD, ["bbbm_h1"])], "unexplained-gap"),  # a shifted hole
    ((ODD,), [hole(ODD, ["bbbm_h1"]), hole(ODD2, [])], "accepted-hole-unmatched"),  # a listed hole the pull lacks
    ((ODD,), [hole(ODD, [])], "unexplained-gap"),  # the same interval, a different seen_in
], ids=["sixth", "shifted", "absent", "seen-in"])
def test_only_the_exact_list_is_accepted(setup, drop, listed, reason) -> None:
    write_holes("aaam_h1", holes_doc("aaam_h1", listed))
    with pytest.raises(WizardRefused, match=reason):
        w2.run("AAAm", "H1", START, END, mt5=PinMT5(hours(*drop)))
    assert _no_pin()


@pytest.mark.parametrize("edit", [
    lambda d: d.update(cell_id="bbbm_h1"),
    lambda d: d.update(citation=" "),
    lambda d: d.update(holes=[]),
    lambda d: d.update(tolerance_pct=0.1),
    lambda d: d["holes"][0].update(bars_missing=2),
    lambda d: d["holes"][0].update(missing_from="2024-01-09T05:00:00Z", missing_to="2024-01-09T05:00:00Z"),
    lambda d: d["holes"][0].update(max_bars=3),
    lambda d: d["holes"].append(dict(d["holes"][0])),
    lambda d: d["holes"][0].update(seen_in=[["bbbm_h1"]]),
    lambda d: d["holes"][0].update(seen_in=["cccm_h1", "bbbm_h1"]),
    lambda d: d["holes"][0].update(seen_in=["bbbm_h1", "bbbm_h1"]),
    lambda d: d["holes"][0].update(bars_missing=True),
    lambda d: d.update(version=True),
], ids=["cell", "citation", "empty", "rate-key", "bars", "non-canonical", "hole-key", "duplicate",
        "seen-in-nested", "seen-in-unsorted", "seen-in-repeat", "bars-bool", "version-bool"])
def test_a_malformed_list_refuses_before_any_attach(setup, edit) -> None:
    doc = holes_doc("aaam_h1", [hole(ODD, ["bbbm_h1"])])
    edit(doc)
    write_holes("aaam_h1", doc)
    fake = PinMT5(hours(ODD))
    with pytest.raises(WizardRefused, match="accepted-holes-malformed"):
        w2.run("AAAm", "H1", START, END, mt5=fake)
    assert not hasattr(fake, "kw") and _no_pin()


# --- gbpjpym_h1's ceremony list (configs/cells/gbpjpym_h1.accepted_holes.json) is EXACTLY these five ---

FIVE = [
    ("2017-08-14T00:00:00+00:00", "2017-08-14T12:00:00+00:00", 13, ()),
    ("2020-09-02T16:00:00+00:00", "2020-09-03T13:00:00+00:00", 22, ("xauusdm_h1",)),
    ("2021-03-08T13:00:00+00:00", "2021-03-08T14:00:00+00:00", 2, ()),
    ("2021-04-06T22:00:00+00:00", "2021-04-07T00:00:00+00:00", 3, ()),
    ("2021-06-08T23:00:00+00:00", "2021-06-09T01:00:00+00:00", 3, ()),
]
W_START, W_END = "2017-03-01", "2026-09-01"


def span(lo: str, hi: str) -> list[int]:
    a, b = (int(datetime.fromisoformat(x).timestamp()) for x in (lo, hi))
    return list(range(a, b + H, H))


def test_the_ceremony_list_is_exactly_the_five() -> None:
    real = w2.accepted_holes_path("gbpjpym_h1")
    acc = w2.load_accepted("gbpjpym_h1", H)
    assert acc["file"] == "configs/cells/gbpjpym_h1.accepted_holes.json"
    doc = json.loads(real.read_text(encoding="utf-8"))
    assert acc["canonical_sha256"] == hashlib.sha256(canonical(doc).encode()).hexdigest()
    assert acc["doc"]["evidence"]["diag_log"] == "artifacts/reports/u4/w2_gap_diag.log"
    assert [w2.hole_key(h) for h in acc["doc"]["holes"]] == FIVE
    assert sum(h["bars_missing"] for h in acc["doc"]["holes"]) == 43
    assert acc["doc"]["citation"].startswith("AMENDMENT-U4-#4")
    assert "symbol-pair-shared" in acc["doc"]["holes"][1]["note"]


@pytest.fixture
def five(wizard_home: Path, tmp_path: Path, monkeypatch):
    """A full-window synthetic gbpjpym_h1: two trusted pins, the xau one sharing the 2020 hole; the REAL list."""
    real = w2.accepted_holes_path("gbpjpym_h1")
    t0, t1 = day(W_START), day(W_END)
    stub(monkeypatch, tmp_path, [("USDCADm", hours(t0=t0, t1=t1), 100),
                                 ("XAUUSDm", hours(*span(*FIVE[1][:2]), t0=t0, t1=t1), 100)])
    monkeypatch.setattr(w2, "accepted_holes_path", lambda cid: real)
    return t0, t1


def test_the_five_are_owner_accepted(five) -> None:
    t0, t1 = five
    doc = w2.run("GBPJPYm", "H1", W_START, W_END,
                 mt5=PinMT5(hours(*[x for h in FIVE for x in span(*h[:2])], t0=t0, t1=t1), "GBPJPYm"))
    g = doc["gaps"]
    assert g["counts"]["owner_accepted_hole"] == 5 and g["counts"]["pin_explained"] == 0
    assert [w2.hole_key(r) for r in g["owner_accepted_holes"]] == FIVE and g["owner_accepted_bars"] == 43
    assert all(r["citation"].startswith("AMENDMENT-U4-#4") for r in g["owner_accepted_holes"])
    assert g["acceptance"]["file"] == "configs/cells/gbpjpym_h1.accepted_holes.json"


@pytest.mark.parametrize("change, reason", [
    (lambda d: d + span("2022-05-10T10:00:00+00:00", "2022-05-10T10:00:00+00:00"), "unexplained-gap"),
    (lambda d: [x + H if x in span(*FIVE[0][:2]) else x for x in d], "unexplained-gap"),
    (lambda d: [x for x in d if x not in span(*FIVE[2][:2])], "accepted-hole-unmatched"),
], ids=["sixth-hole", "shifted-hole", "a-listed-hole-filled"])
def test_anything_but_the_five_refuses(five, change, reason) -> None:
    t0, t1 = five
    drop = change([x for h in FIVE for x in span(*h[:2])])
    with pytest.raises(WizardRefused, match=reason):
        w2.run("GBPJPYm", "H1", W_START, W_END, mt5=PinMT5(hours(*drop, t0=t0, t1=t1), "GBPJPYm"))
    assert _no_pin("GBPJPYm")
