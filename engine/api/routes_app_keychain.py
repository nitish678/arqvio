"""The Keychain: ARQVIO's secrets store, formerly the S5-APP "vault" (in ARQVIO the Vault is the strategy
library). DR-2026-10-10-s5-app §13; owner ruling D-9, backend per the 2026-10-10 Decision 2: pywin32's
``win32cred``): secrets in the Windows Credential Manager.

The secret bytes live only in the Credential Manager (generic credentials, target ``ARQVIO-vault/<name>``, persisted
for this Windows user on this machine); ARQVIO keeps a names-only index ``<ARQVIO_HOME>/vault/index.json`` (name, note,
created_utc, the passphrase's scrypt digest) and an audit chain ``<ARQVIO_HOME>/vault/audit.jsonl`` (operation, entry
name, utc, sha256 chained over the previous line -- never the secret). The passphrase locks the app view only; the view
auto-locks after ``AUTO_LOCK_S`` idle. The honesty label is a function of the backend actually resolved: only the
win32cred store carries D-9's label, and any other state -- win32cred not importable, another backend -- is
UNAVAILABLE with its reason. There is no plain-file fallback (D-9: banned). Nothing outside this module reads the
keychain: it feeds no gate, deploy or refusal (V5). The on-disk names (``ARQVIO-vault/``, ``<ARQVIO_HOME>/vault/``)
are kept so secrets saved before the rename stay readable.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.core import atomic
from engine.core.paths import get_arqvio_home

router = APIRouter(prefix="/api/v2/app/keychain", tags=["app-keychain"])

SERVICE = "ARQVIO-vault"
AUTO_LOCK_S = 600
CLIPBOARD_CLEAR_S = 20
WINDOWS_BACKEND = "engine.api.routes_app_keychain.WinCredStore"
# Owner ruling D-9, verbatim; keyed by the backend class it is true of.
BACKEND_LABELS = {WINDOWS_BACKEND: "Windows Credential Manager (OS-backed, per-user store) — S9 adds DPAPI hardening"}
SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}
ERROR_NOT_FOUND = 1168

_session: dict[str, float | None] = {"unlocked_at": None, "last_activity": None}


class WinCredStore:
    """Generic credentials in the Windows Credential Manager through pywin32's win32cred."""

    def __init__(self, win32cred: Any, pywintypes: Any) -> None:
        self.cred, self.pywintypes = win32cred, pywintypes

    @staticmethod
    def target(service: str, user: str) -> str:
        return f"{service}/{user}"

    def set_password(self, service: str, user: str, secret: str) -> None:
        self.cred.CredWrite({"Type": self.cred.CRED_TYPE_GENERIC, "TargetName": self.target(service, user),
                             "UserName": user, "CredentialBlob": secret,
                             "Persist": self.cred.CRED_PERSIST_LOCAL_MACHINE}, 0)

    def get_password(self, service: str, user: str) -> str | None:
        try:
            blob = self.cred.CredRead(self.target(service, user), self.cred.CRED_TYPE_GENERIC)["CredentialBlob"]
        except self.pywintypes.error as exc:
            if exc.winerror == ERROR_NOT_FOUND:
                return None
            raise
        return blob.decode("utf-16-le") if isinstance(blob, bytes) else blob

    def delete_password(self, service: str, user: str) -> None:
        self.cred.CredDelete(self.target(service, user), self.cred.CRED_TYPE_GENERIC)


def store() -> Any:
    """The credential store in use, or the reason there is none (a str)."""
    try:
        import pywintypes
        import win32cred
    except ImportError:
        return "pywin32's win32cred is not importable in this environment"
    return WinCredStore(win32cred, pywintypes)


def backend() -> tuple[Any, str | None, str]:
    """(backend or None, D-9 label or None, backend name or the unavailability reason)."""
    kr = store()
    if isinstance(kr, str):
        return None, None, kr
    name = f"{type(kr).__module__}.{type(kr).__qualname__}"
    label = BACKEND_LABELS.get(name)
    return (kr, label, name) if label else (None, None, f"credential backend {name} is not the Windows Credential "
                                                        "Manager; D-9 rules no other store")


def vault_dir() -> Path:
    return get_arqvio_home() / "vault"


def index_path() -> Path:
    return vault_dir() / "index.json"


def audit_path() -> Path:
    return vault_dir() / "audit.jsonl"


def now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()


def read_index() -> dict[str, Any] | None:
    p = index_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write_index(doc: dict[str, Any]) -> None:
    p = index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    atomic.replace(tmp, p)


def audit(op: str, name: str | None) -> None:
    """Append one chained line: sha256 over the previous line's sha256 + this line's canonical body."""
    p = audit_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
    prev = json.loads(lines[-1])["sha256"] if lines else "0" * 64
    body = {"op": op, "name": name, "utc": now_utc(), "prev_sha256": prev}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**body, "sha256": hashlib.sha256((prev + canon).encode()).hexdigest()},
                           sort_keys=True) + "\n")


def digest(passphrase: str, salt: bytes) -> str:
    return hashlib.scrypt(passphrase.encode(), salt=salt, **SCRYPT).hex()


def unlocked() -> bool:
    """True while the view is unlocked and not idle past AUTO_LOCK_S; a lapse locks it (and is audited)."""
    last = _session["last_activity"]
    if last is None:
        return False
    if time.monotonic() - last > AUTO_LOCK_S:
        _session.update(unlocked_at=None, last_activity=None)
        audit("auto-lock", None)
        return False
    _session["last_activity"] = time.monotonic()
    return True


def require(*, unlocked_view: bool = True) -> tuple[Any, dict[str, Any]]:
    kr, _label, why = backend()
    if kr is None:
        raise HTTPException(status_code=503, detail=f"keychain UNAVAILABLE: {why}")
    idx = read_index()
    if idx is None:
        raise HTTPException(status_code=409, detail="keychain not set up: choose a passphrase first")
    if unlocked_view and not unlocked():
        raise HTTPException(status_code=423, detail="keychain LOCKED")
    return kr, idx


class Passphrase(BaseModel):
    passphrase: str = Field(min_length=8)


class EntryIn(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.\-]+$")
    secret: str = Field(min_length=1)
    note: str = ""


class EntryUpdate(BaseModel):
    secret: str | None = Field(default=None, min_length=1)
    note: str | None = None


@router.get("")
async def vault_state() -> dict[str, Any]:
    kr, label, why = backend()
    idx = read_index() if kr is not None else None
    state = ("UNAVAILABLE" if kr is None else "NOT SET UP" if idx is None
             else "UNLOCKED" if unlocked() else "LOCKED")
    out: dict[str, Any] = {"state": state, "label": label, "backend": why if kr is not None else None,
                           "reason": why if kr is None else None, "auto_lock_s": AUTO_LOCK_S,
                           "clipboard_clear_s": CLIPBOARD_CLEAR_S}
    if state == "UNLOCKED":
        out["entries"] = [{k: e[k] for k in ("name", "note", "created_utc")} for e in idx["entries"]]
    return out


@router.post("/setup")
async def setup(body: Passphrase) -> dict[str, Any]:
    kr, _label, why = backend()
    if kr is None:
        raise HTTPException(status_code=503, detail=f"keychain UNAVAILABLE: {why}")
    if read_index() is not None:
        raise HTTPException(status_code=409, detail="keychain already set up")
    salt = os.urandom(16)
    write_index({"schema": "arqvio.app.vault_index", "version": 1, "service": SERVICE, "created_utc": now_utc(),
                 "passphrase": {"kdf": "scrypt", **SCRYPT, "salt": salt.hex(), "digest": digest(body.passphrase, salt)},
                 "entries": []})
    audit("setup", None)
    _session.update(unlocked_at=time.monotonic(), last_activity=time.monotonic())
    return {"state": "UNLOCKED"}


@router.post("/unlock")
async def unlock(body: Passphrase) -> dict[str, Any]:
    _kr, idx = require(unlocked_view=False)
    pp = idx["passphrase"]
    if not hmac.compare_digest(digest(body.passphrase, bytes.fromhex(pp["salt"])), pp["digest"]):
        audit("unlock-refused", None)
        raise HTTPException(status_code=403, detail="REFUSED: wrong passphrase")
    _session.update(unlocked_at=time.monotonic(), last_activity=time.monotonic())
    audit("unlock", None)
    return {"state": "UNLOCKED"}


@router.post("/lock")
async def lock() -> dict[str, Any]:
    require(unlocked_view=False)
    _session.update(unlocked_at=None, last_activity=None)
    audit("lock", None)
    return {"state": "LOCKED"}


@router.post("/entries")
async def create_entry(body: EntryIn) -> dict[str, Any]:
    kr, idx = require()
    if any(e["name"] == body.name for e in idx["entries"]):
        raise HTTPException(status_code=409, detail=f"entry {body.name!r} exists")
    kr.set_password(SERVICE, body.name, body.secret)
    idx["entries"].append({"name": body.name, "note": body.note, "created_utc": now_utc()})
    write_index(idx)
    audit("create", body.name)
    return {"name": body.name}


@router.put("/entries/{name}")
async def update_entry(name: str, body: EntryUpdate) -> dict[str, Any]:
    kr, idx = require()
    entry = next((e for e in idx["entries"] if e["name"] == name), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no entry {name!r}")
    if body.secret is not None:
        kr.set_password(SERVICE, name, body.secret)
    if body.note is not None:
        entry["note"] = body.note
    write_index(idx)
    audit("update", name)
    return {"name": name}


@router.delete("/entries/{name}")
async def delete_entry(name: str) -> dict[str, Any]:
    kr, idx = require()
    if not any(e["name"] == name for e in idx["entries"]):
        raise HTTPException(status_code=404, detail=f"no entry {name!r}")
    kr.delete_password(SERVICE, name)
    idx["entries"] = [e for e in idx["entries"] if e["name"] != name]
    write_index(idx)
    audit("delete", name)
    return {"name": name}


@router.post("/entries/{name}/reveal")
async def reveal(name: str) -> dict[str, Any]:
    """The one route that returns a secret: unlocked view only, audited; the app masks it until asked."""
    kr, idx = require()
    if not any(e["name"] == name for e in idx["entries"]):
        raise HTTPException(status_code=404, detail=f"no entry {name!r}")
    secret = kr.get_password(SERVICE, name)
    if secret is None:
        raise HTTPException(status_code=409, detail=f"entry {name!r} is indexed but absent from the backend")
    audit("reveal", name)
    return {"name": name, "secret": secret, "clipboard_clear_s": CLIPBOARD_CLEAR_S}
