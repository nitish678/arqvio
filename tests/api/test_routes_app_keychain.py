"""S5-APP vault contract (DR-2026-10-10-s5-app §13, owner ruling D-9): CRUD through the backend, lock/unlock and
auto-lock, the secret absent from every ARQVIO file, the audit chain verifiable, the label a function of the backend
actually resolved, and UNAVAILABLE -- never a plain-file fallback -- when no ruled backend exists. Hermetic: an
in-memory backend registered under its own class name; the real Credential Manager is never written."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.api import routes_app_keychain as rv
from engine.api.server import create_app

TEST_VALUE = "fixture-value-never-on-disk"
D9 = "Windows Credential Manager (OS-backed, per-user store) — S9 adds DPAPI hardening"


class MemoryBackend:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, user: str, secret: str) -> None:
        self.data[(service, user)] = secret

    def get_password(self, service: str, user: str) -> str | None:
        return self.data.get((service, user))

    def delete_password(self, service: str, user: str) -> None:
        del self.data[(service, user)]


@pytest.fixture
def vault(isolated_arqvio_home: Path, monkeypatch) -> dict:
    mem = MemoryBackend()
    monkeypatch.setattr(rv, "store", lambda: mem)
    monkeypatch.setitem(rv.BACKEND_LABELS, f"{__name__}.MemoryBackend", "test label for the in-memory backend")
    monkeypatch.setattr(rv, "_session", {"unlocked_at": None, "last_activity": None})
    return {"mem": mem, "home": isolated_arqvio_home, "c": TestClient(create_app())}


def test_crud_lock_unlock_and_the_secret_lives_only_in_the_backend(vault) -> None:
    c, mem, home = vault["c"], vault["mem"], vault["home"]
    assert c.get("/api/v2/app/keychain").json()["state"] == "NOT SET UP"
    assert c.post("/api/v2/app/keychain/setup", json={"passphrase": "correct horse"}).json()["state"] == "UNLOCKED"
    assert c.post("/api/v2/app/keychain/entries", json={"name": "demo-acct1", "secret": TEST_VALUE, "note": "MT5 demo"}
                  ).status_code == 200
    assert mem.data[("ARQVIO-vault", "demo-acct1")] == TEST_VALUE
    s = c.get("/api/v2/app/keychain").json()
    assert s["entries"] == [{"name": "demo-acct1", "note": "MT5 demo", "created_utc": s["entries"][0]["created_utc"]}]
    assert TEST_VALUE not in json.dumps(s)  # masked: the listing never carries the value
    assert c.post("/api/v2/app/keychain/lock").json()["state"] == "LOCKED"
    assert "entries" not in c.get("/api/v2/app/keychain").json()
    assert c.post("/api/v2/app/keychain/entries/demo-acct1/reveal").status_code == 423
    assert c.post("/api/v2/app/keychain/unlock", json={"passphrase": "wrong horse!"}).status_code == 403
    assert c.post("/api/v2/app/keychain/unlock", json={"passphrase": "correct horse"}).status_code == 200
    r = c.post("/api/v2/app/keychain/entries/demo-acct1/reveal").json()
    assert (r["secret"], r["clipboard_clear_s"]) == (TEST_VALUE, 20)
    assert c.put("/api/v2/app/keychain/entries/demo-acct1", json={"secret": "rotated-value"}).status_code == 200
    assert mem.data[("ARQVIO-vault", "demo-acct1")] == "rotated-value"
    assert c.delete("/api/v2/app/keychain/entries/demo-acct1").status_code == 200 and mem.data == {}
    on_disk = "".join(p.read_text(encoding="utf-8") for p in (home / "vault").iterdir())
    assert TEST_VALUE not in on_disk and "rotated-value" not in on_disk and "correct horse" not in on_disk


def test_the_audit_chain_verifies_and_names_operations_only(vault) -> None:
    c = vault["c"]
    c.post("/api/v2/app/keychain/setup", json={"passphrase": "correct horse"})
    c.post("/api/v2/app/keychain/entries", json={"name": "k1", "secret": TEST_VALUE})
    c.post("/api/v2/app/keychain/entries/k1/reveal")
    c.post("/api/v2/app/keychain/unlock", json={"passphrase": "nope-nope"})
    lines = [json.loads(x) for x in (vault["home"] / "vault" / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["op"] for x in lines] == ["setup", "create", "reveal", "unlock-refused"]
    prev = "0" * 64
    for x in lines:  # V3: operation + timestamp + hash, each line chained to the previous
        body = {k: x[k] for k in ("op", "name", "utc", "prev_sha256")}
        assert x["prev_sha256"] == prev
        assert x["sha256"] == hashlib.sha256((prev + json.dumps(body, sort_keys=True, separators=(",", ":"))
                                              ).encode()).hexdigest()
        prev = x["sha256"]


def test_the_view_auto_locks_after_ten_idle_minutes(vault, monkeypatch) -> None:
    c = vault["c"]
    c.post("/api/v2/app/keychain/setup", json={"passphrase": "correct horse"})
    clock = [rv.time.monotonic()]
    monkeypatch.setattr(rv.time, "monotonic", lambda: clock[0])
    rv._session.update(last_activity=clock[0])
    clock[0] += 599
    assert c.get("/api/v2/app/keychain").json()["state"] == "UNLOCKED"
    clock[0] += 601
    assert c.get("/api/v2/app/keychain").json()["state"] == "LOCKED"
    assert json.loads((vault["home"] / "vault" / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
                      )["op"] == "auto-lock"


def test_no_ruled_backend_means_unavailable_and_nothing_is_written(isolated_arqvio_home, monkeypatch) -> None:
    monkeypatch.setattr(rv, "_session", {"unlocked_at": None, "last_activity": None})
    c = TestClient(create_app())
    for kr, why in (("pywin32's win32cred is not importable in this environment", "not importable"),
                    (MemoryBackend(), "is not the Windows Credential Manager")):
        monkeypatch.setattr(rv, "store", lambda kr=kr: kr)
        s = c.get("/api/v2/app/keychain").json()
        assert (s["state"], s["label"]) == ("UNAVAILABLE", None) and why in s["reason"]
        assert c.post("/api/v2/app/keychain/setup", json={"passphrase": "correct horse"}).status_code == 503
        assert c.post("/api/v2/app/keychain/entries", json={"name": "k", "secret": TEST_VALUE}).status_code == 503
    assert not (isolated_arqvio_home / "vault").exists()  # D-9: no silent plain-file fallback


def test_the_label_is_the_backend_fact_on_this_machine(isolated_arqvio_home, monkeypatch) -> None:
    monkeypatch.setattr(rv, "_session", {"unlocked_at": None, "last_activity": None})
    s = TestClient(create_app()).get("/api/v2/app/keychain").json()
    kr = rv.store()  # resolving the backend reads nothing from it and writes nothing
    if isinstance(kr, str):
        assert (s["state"], s["reason"], s["label"]) == ("UNAVAILABLE", kr, None)
    else:  # V2: the label is shown exactly when the resolved backend is the win32cred store
        assert isinstance(kr, rv.WinCredStore) and (s["state"], s["label"]) == ("NOT SET UP", D9)
        assert s["backend"] == rv.WINDOWS_BACKEND == f"{rv.WinCredStore.__module__}.{rv.WinCredStore.__qualname__}"
    assert rv.BACKEND_LABELS == {rv.WINDOWS_BACKEND: D9}  # the D-9 label, verbatim, and only for that backend


def test_the_real_credential_manager_round_trips_a_namespaced_smoke_credential() -> None:
    """Optional real-store smoke (owner, 2026-10-10 Decision 2): an ARQVIO-VAULT-SMOKE-* credential, deleted in finally."""
    kr = rv.store()
    if isinstance(kr, str):
        pytest.skip(kr)
    import uuid

    service, value = f"ARQVIO-VAULT-SMOKE-{uuid.uuid4().hex}", "smoke-ünïcode-value"
    try:
        kr.set_password(service, "probe", value)
        assert kr.get_password(service, "probe") == value
    finally:
        try:
            kr.delete_password(service, "probe")
        except kr.pywintypes.error:
            pass
    assert kr.get_password(service, "probe") is None


def test_nothing_outside_the_api_reads_the_keychain() -> None:
    root = Path(__file__).resolve().parents[2] / "engine"
    users = [p for p in root.rglob("*.py") if p.name != "routes_app_keychain.py"
             and "routes_app_keychain" in p.read_text(encoding="utf-8", errors="ignore")]
    assert [p.name for p in users] == ["server.py"]  # V5: only mounted; it feeds no gate, deploy or refusal
