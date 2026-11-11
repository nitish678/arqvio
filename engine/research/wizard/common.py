"""Shared pieces of the new-cell wizard (DR-2026-10-09-u4-wizard §3-§4): the cell id, the cell-product paths, the
refusal and its audit row, canonical JSON, the registry precondition, the registry ceremony with its allowed-diff
guard, and the explicit-login acct1 attach.

Every step starts by loading the registry through the verifying loader: a tampered registry refuses before any work.
The MT5 attach is to acct1 by EXPLICIT login, verified (login and DEMO) -- never a bare initialize().
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.core.paths import get_artifacts_dir
from engine.research.registry import cells as registry
from engine.research.registry.annualisation import TF_MINUTES

ACCOUNT_ID = "acct1"  # install 1's account (accounts.json); every wizard MT5 read attaches to it by login
DEMO = 0  # MetaTrader5.ACCOUNT_TRADE_MODE_DEMO


class WizardRefused(Exception):
    """The wizard refuses a step: nothing is written but the audit row."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def canonical(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def cell_id(symbol: str, tf: str) -> str:
    """The U1 convention: <symbol lower>_<timeframe lower>. The timeframe must have an annualisation law row."""
    if tf not in TF_MINUTES:
        raise WizardRefused("unknown-timeframe", f"{tf!r} has no annualisation law row; known {sorted(TF_MINUTES)}")
    return f"{symbol.lower()}_{tf.lower()}"


def cell_dir(cid: str) -> Path:
    """<ARQVIO_HOME>/artifacts/cells/<cell_id>/, resolved now."""
    return get_artifacts_dir() / "cells" / cid


def write_product(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")


def audit_row(out_dir: Path, row: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(canonical({**row, "utc": utc_now()}) + "\n")


def require_registry() -> list[dict[str, Any]]:
    """The verified registry entries; a tampered or unreadable registry refuses the step."""
    try:
        return registry.load()
    except (registry.RegistryError, UnicodeDecodeError) as exc:
        raise WizardRefused(f"registry-{type(exc).__name__}", str(exc)) from exc


def registry_ceremony(change: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
                      allowed: dict[str, frozenset[str]], new: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Apply ``change`` to a copy of the verified entries, refuse any diff outside the guard, seal, write, re-verify.

    The guard: the existing entries keep their order; only the cell_ids in ``new`` are appended; an existing entry
    may change only the fields ``allowed[cell_id]`` names, compared by their exact JSON text (so 2 vs 2.0, 1 vs True
    or a reordered key is a change), and keeps its key order. The sealed document is written to a temp file beside the
    registry and verified by load() there; only then does it replace the registry, atomically, and only if the
    registry did not change since it was read. A refusal never touches the registry.
    """
    path = registry.registry_path()
    raw = path.read_bytes()
    before = require_registry()
    after = change(copy.deepcopy(before))
    ids_before, ids_after = [e["cell_id"] for e in before], [e["cell_id"] for e in after]
    if ids_after[:len(ids_before)] != ids_before or sorted(ids_after[len(ids_before):]) != sorted(new):
        raise WizardRefused("ceremony-diff", f"cells {ids_before} -> {ids_after}; only {sorted(new)} may be added")
    diffs = {}
    for b, a in zip(before, after):
        text = lambda e, k: json.dumps(e.get(k, "<absent>"), ensure_ascii=False)  # noqa: E731
        changed = sorted(k for k in set(a) | set(b) if k != "entry_hash" and text(a, k) != text(b, k))
        if [k for k in a if k in b] != [k for k in b if k in a]:
            changed.append("<key order>")
        if not set(changed) <= allowed.get(b["cell_id"], frozenset()):
            raise WizardRefused("ceremony-diff", f"{b['cell_id']}: fields {changed} changed; allowed "
                                                 f"{sorted(allowed.get(b['cell_id'], ()))}")
        diffs[b["cell_id"]] = changed
    doc = registry.seal(after)
    staged = path.with_name(path.name + ".ceremony")
    staged.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        registry.load(staged)
    except registry.RegistryError as exc:
        staged.unlink()
        raise WizardRefused(f"registry-{type(exc).__name__}", f"the sealed document does not verify: {exc}") from exc
    if path.read_bytes() != raw:
        staged.unlink()
        raise WizardRefused("ceremony-race", f"{path} changed while the ceremony ran")
    os.replace(staged, path)
    verified = registry.load(path)
    return {"chain_before": json.loads(raw)["chain_hash"], "chain_after": doc["chain_hash"], "changed": diffs,
            "entry_hash_before": {e["cell_id"]: e["entry_hash"] for e in before},
            "entry_hash_after": {e["cell_id"]: e["entry_hash"] for e in verified}}


def attach_acct1(mt5: Any) -> dict[str, Any]:
    """initialize() acct1 by path + login + server (+ password from its password_env when set); verify the login and
    DEMO. Returns the account row; refuses otherwise (the caller shuts the terminal down)."""
    from engine.research.health.probe import accounts_path, load_accounts

    if not accounts_path().is_file():
        raise WizardRefused("no-accounts-config", f"no accounts config at {accounts_path()}")
    rows = [a for a in load_accounts() if a["id"] == ACCOUNT_ID]
    if len(rows) != 1:
        raise WizardRefused("no-acct1", f"accounts config has {len(rows)} rows with id {ACCOUNT_ID!r}")
    row = rows[0]
    kw = {"path": row["terminal"], "login": int(row["login"]), "server": row["server"]}
    if os.environ.get(row["password_env"]):
        kw["password"] = os.environ[row["password_env"]]
    if not mt5.initialize(**kw):
        raise WizardRefused("mt5-unavailable", f"attach to {ACCOUNT_ID} failed: {mt5.last_error()}")
    info = mt5.account_info()
    if info is None or int(info.login) != int(row["login"]) or int(info.trade_mode) != DEMO:
        mt5.shutdown()
        raise WizardRefused("wrong-account", f"the attached terminal is not {ACCOUNT_ID} on DEMO")
    return row
