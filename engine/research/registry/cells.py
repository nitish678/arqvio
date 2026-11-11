"""The cell registry: schema, verifying loader and read API (DR-2026-10-07-u1 §2).

The registry is DATA -- configs/cells_registry.json in the repo, resolved at call time. Every read loads the
file and verifies it end to end before answering; a file that fails any check is REFUSED with a named error and
never repaired:

- schema / schema_version wrong (top level or any entry)        -> RegistrySchemaVersionError
- a field missing, extra, or malformed                           -> RegistrySchemaError
- a recomputed entry_hash or chain_hash differs (tampered)       -> RegistryIntegrityError
- two entries on one cell (a duplicate cell_id)                  -> AmbiguousCellError
- a lookup of a cell the registry does not hold                  -> UnknownCellError

entry_hash is the sha256 of the entry's canonical JSON (sorted keys, compact, UTF-8) without entry_hash;
chain_hash folds the entry hashes in file order from 64 zeros: h_i = sha256(h_{i-1} + entry_hash_i).
Only VALIDATED entries resolve for the research stack; a RETIRED entry stays in the chain as history.

QUICK cells (ARQVIO "Add market"): a market the owner adds in the app lives in <ARQVIO_HOME>/configs/user_cells.json,
outside the repo registry and its ceremony, with the SAME entry schema, sealed and verified the same way, and the
status QUICK (never VALIDATED: its data was pinned in-app, not proven by the U4 wizard). The research stack -- the
pinned-cell view, cross_of, annualisation -- resolves VALIDATED and QUICK cells; the repo registry's own readers
(list_cells, validated_cells) see the repo file only. A QUICK cell can never shadow a registry cell.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from engine.core import paths
from engine.core.errors import ConfigError
from engine.core.schema import same_schema
from engine.research.registry.annualisation import TF_MINUTES, bars_per_year

SCHEMA = "arqvio.cells_registry"
USER_SCHEMA = "arqvio.user_cells"
SCHEMA_VERSION = 2
STATUSES = ("VALIDATED", "RETIRED")
QUICK = "QUICK"
RESEARCH_STATUSES = ("VALIDATED", QUICK)
GENESIS = "0" * 64

_TOP_FIELDS = ("schema", "schema_version", "entries", "chain_hash")
_ENTRY_FIELDS = (
    "schema_version", "cell_id", "status", "symbol", "timeframe", "dataset_hash16", "dataset_window", "cross_cell",
    "annualisation_ppy", "spread_artifact_hash", "specs_artifact_hash", "audit_row_ref", "power_check",
    "worker_status", "campaign_status", "entered_by_stage", "entered_at", "evidence", "entry_hash",
)
# The nested objects: field -> its exact keys and each key's type.
_NESTED: dict[str, dict[str, type]] = {
    "dataset_window": {"start": str, "end": str, "bars": int},
    "spread_artifact_hash": {"sha256": str, "source": str},
    "specs_artifact_hash": {"sha256": str, "source": str},
    "audit_row_ref": {"file": str, "line": int, "as_of": str, "row": str},
    "power_check": {"status": str, "ref": str},
    "worker_status": {"status": str, "fills_in": str, "note": str},
}
_STR_FIELDS = ("cell_id", "status", "symbol", "timeframe", "dataset_hash16", "cross_cell", "campaign_status",
               "entered_by_stage", "entered_at")
# UNRUN: never searched; RUN@<the GA campaign directory the registry cites>.
CAMPAIGN_STATUS = re.compile(r"UNRUN|RUN@ga-[a-z0-9_-]+")
_HEX16 = re.compile(r"[0-9a-f]{16}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class RegistryError(ConfigError):
    """The cell registry cannot be trusted or cannot answer."""


class RegistryMissing(RegistryError):
    """The registry file does not exist."""


class RegistrySchemaVersionError(RegistryError):
    """The registry (or an entry) is not schema arqvio.cells_registry at this loader's SCHEMA_VERSION."""


class RegistrySchemaError(RegistryError):
    """A field is missing, extra, or malformed."""


class RegistryIntegrityError(RegistryError):
    """A recomputed entry_hash or chain_hash does not match: the file was edited outside the sealing path."""


class AmbiguousCellError(RegistryError):
    """Two entries claim the same cell."""


class UnknownCellError(RegistryError):
    """A lookup of a cell the registry does not hold."""


def registry_path() -> Path:
    """configs/cells_registry.json in the project root, resolved now (never at import)."""
    return paths.PROJECT_ROOT / "configs" / "cells_registry.json"


def user_cells_path() -> Path:
    """<ARQVIO_HOME>/configs/user_cells.json: the QUICK cells added in the app, resolved now."""
    return paths.get_configs_dir() / "user_cells.json"


def entry_hash(entry: Mapping[str, Any]) -> str:
    """sha256 of the entry's canonical JSON without its entry_hash field."""
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chain_hash(entry_hashes: list[str]) -> str:
    """The fold of the entry hashes in order, from GENESIS."""
    h = GENESIS
    for eh in entry_hashes:
        h = hashlib.sha256((h + eh).encode("ascii")).hexdigest()
    return h


def seal(entries: list[dict[str, Any]], *, schema: str = SCHEMA) -> dict[str, Any]:
    """The registry document for ``entries``: each entry's entry_hash set, the chain hash computed.

    Sealing is how an entry is written (the U1 migration; the U2 power, U3 worker_status and U4 new-cell
    ceremonies); the loader verifies, it never seals.
    """
    sealed = []
    for e in entries:
        e = {k: v for k, v in e.items() if k != "entry_hash"}
        e["entry_hash"] = entry_hash(e)
        sealed.append(e)
    return {"schema": schema, "schema_version": SCHEMA_VERSION, "entries": sealed,
            "chain_hash": chain_hash([e["entry_hash"] for e in sealed])}


def _check_fields(where: str, obj: Any, fields: tuple[str, ...]) -> None:
    if not isinstance(obj, dict):
        raise RegistrySchemaError(f"{where}: expected an object, got {type(obj).__name__}")
    missing = [f for f in fields if f not in obj]
    extra = sorted(set(obj) - set(fields))
    if missing or extra:
        raise RegistrySchemaError(f"{where}: missing fields {missing}, unexpected fields {extra}")


def _check_type(where: str, value: Any, kind: type) -> None:
    # bool is an int subclass; a count or a line number is never a bool.
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise RegistrySchemaError(f"{where}: expected {kind.__name__}, got {value!r}")


def _check_entry(i: int, e: Any, statuses: tuple[str, ...] = STATUSES) -> None:
    where = f"entries[{i}]"
    if isinstance(e, dict) and e.get("schema_version") != SCHEMA_VERSION:
        raise RegistrySchemaVersionError(
            f"{where}: schema_version {e.get('schema_version')!r}; this loader reads {SCHEMA_VERSION} only")
    _check_fields(where, e, _ENTRY_FIELDS)
    for f in _STR_FIELDS:
        _check_type(f"{where}.{f}", e[f], str)
    for f, keys in _NESTED.items():
        _check_fields(f"{where}.{f}", e[f], tuple(keys))
        for k, kind in keys.items():
            _check_type(f"{where}.{f}.{k}", e[f][k], kind)
    _check_type(f"{where}.annualisation_ppy", e["annualisation_ppy"], int)
    _check_type(f"{where}.evidence", e["evidence"], list)
    if not e["evidence"] or not all(isinstance(p, str) and p for p in e["evidence"]):
        raise RegistrySchemaError(f"{where}.evidence: expected a non-empty list of paths")
    if e["status"] not in statuses:
        raise RegistrySchemaError(f"{where}.status: {e['status']!r} not in {statuses}")
    if e["timeframe"] not in TF_MINUTES:
        raise RegistrySchemaError(f"{where}.timeframe: {e['timeframe']!r} has no annualisation law")
    expected_id = f"{e['symbol'].lower()}_{e['timeframe'].lower()}"
    if e["cell_id"] != expected_id:
        raise RegistrySchemaError(f"{where}.cell_id: {e['cell_id']!r} breaks the convention ({expected_id!r})")
    if not _HEX16.fullmatch(e["dataset_hash16"]):
        raise RegistrySchemaError(f"{where}.dataset_hash16: {e['dataset_hash16']!r} is not 16 lowercase hex")
    for f in ("spread_artifact_hash", "specs_artifact_hash"):
        if not _HEX64.fullmatch(e[f]["sha256"]):
            raise RegistrySchemaError(f"{where}.{f}.sha256: not 64 lowercase hex")
    if e["dataset_window"]["bars"] <= 0 or e["audit_row_ref"]["line"] <= 0:
        raise RegistrySchemaError(f"{where}: dataset_window.bars and audit_row_ref.line must be positive")
    law = bars_per_year(e["timeframe"])
    if e["annualisation_ppy"] != law:
        raise RegistrySchemaError(
            f"{where}.annualisation_ppy: {e['annualisation_ppy']} is off the law ({e['timeframe']} -> {law})")
    if not _HEX64.fullmatch(e["entry_hash"]):
        raise RegistrySchemaError(f"{where}.entry_hash: not 64 lowercase hex")
    if not CAMPAIGN_STATUS.fullmatch(e["campaign_status"]):
        raise RegistrySchemaError(
            f"{where}.campaign_status: {e['campaign_status']!r} is not UNRUN or RUN@ga-<campaign>")


def _verified(path: Path, schema: str, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
    """A sealed cells document's entries after the schema, hash and chain checks (shared by both files)."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistrySchemaError(f"{path}: not JSON ({exc})") from exc
    if not isinstance(doc, dict) or not same_schema(doc.get("schema"), schema) or doc.get("schema_version") != SCHEMA_VERSION:
        got = (doc.get("schema"), doc.get("schema_version")) if isinstance(doc, dict) else type(doc).__name__
        raise RegistrySchemaVersionError(f"{path}: {got!r}; this loader reads ({schema!r}, {SCHEMA_VERSION}) only")
    _check_fields(str(path), doc, _TOP_FIELDS)
    _check_type("entries", doc["entries"], list)
    _check_type("chain_hash", doc["chain_hash"], str)
    entries = doc["entries"]
    for i, e in enumerate(entries):
        _check_entry(i, e, statuses)
    for i, e in enumerate(entries):
        recomputed = entry_hash(e)
        if recomputed != e["entry_hash"]:
            raise RegistryIntegrityError(
                f"entries[{i}] ({e['cell_id']}): entry_hash {e['entry_hash']} != recomputed {recomputed}")
    chain = chain_hash([e["entry_hash"] for e in entries])
    if chain != doc["chain_hash"]:
        raise RegistryIntegrityError(f"chain_hash {doc['chain_hash']} != recomputed {chain}")
    # cell_id is pinned to (symbol, timeframe) above, so a duplicate id is the only way two entries share a cell.
    ids = [e["cell_id"] for e in entries]
    dupes = sorted({c for c in ids if ids.count(c) > 1})
    if dupes:
        raise AmbiguousCellError(f"duplicate cell_id {dupes}")
    return entries


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """The registry's entries, in file order, after every check passed. Refuses (raises) otherwise."""
    path = registry_path() if path is None else path
    if not path.is_file():
        raise RegistryMissing(f"cell registry not found: {path}")
    entries = _verified(path, SCHEMA, STATUSES)
    status = {e["cell_id"]: e["status"] for e in entries}
    for e in entries:
        if e["status"] != "VALIDATED":
            continue
        cross = e["cross_cell"]
        if cross == e["cell_id"] or status.get(cross) != "VALIDATED":
            raise RegistrySchemaError(
                f"{e['cell_id']}.cross_cell: {cross!r} must be another VALIDATED cell (it is {status.get(cross)!r})")
    return entries


def load_user(path: Path | None = None) -> list[dict[str, Any]]:
    """The QUICK cells added in the app (none when the file is absent), verified like the registry."""
    path = user_cells_path() if path is None else path
    if not path.is_file():
        return []
    return _verified(path, USER_SCHEMA, (QUICK,))


def research_entries() -> list[dict[str, Any]]:
    """The cells the research stack resolves: the registry's VALIDATED entries, then the QUICK cells.

    A QUICK cell whose id the registry already holds is refused (it would shadow a proven cell), and a QUICK cell's
    cross cell must be another VALIDATED or QUICK cell."""
    from engine.core.paths import app_mode

    every = [] if app_mode() else load()
    repo = [e for e in every if e["status"] == "VALIDATED"]
    taken = {e["cell_id"] for e in every}
    user = load_user()
    clash = sorted(e["cell_id"] for e in user if e["cell_id"] in taken)
    if clash:
        raise AmbiguousCellError(f"QUICK cells {clash} share an id with the repo registry")
    known = {e["cell_id"] for e in repo} | {e["cell_id"] for e in user}
    for e in user:
        if e["cross_cell"] == e["cell_id"] or e["cross_cell"] not in known:
            raise RegistrySchemaError(f"{e['cell_id']}.cross_cell: {e['cross_cell']!r} must be another research cell")
    return repo + user


def list_cells(path: Path | None = None) -> list[dict[str, Any]]:
    """Every entry, in file order."""
    return load(path)


def get_cell(cell_id: str, path: Path | None = None) -> dict[str, Any]:
    """The entry for ``cell_id``: from ``path`` when given, else from the research cells (VALIDATED and QUICK)
    and then the rest of the repo registry."""
    from engine.core.paths import app_mode

    for e in (load(path) if path is not None else [*research_entries(), *([] if app_mode() else load())]):
        if e["cell_id"] == cell_id:
            return e
    raise UnknownCellError(f"unknown cell {cell_id!r}; the registry holds {sorted(validated_cells(path))}")


def validated_cells(path: Path | None = None) -> list[str]:
    """The VALIDATED cell_ids, sorted."""
    return sorted(e["cell_id"] for e in load(path) if e["status"] == "VALIDATED")


def cross_of(cell_id: str, path: Path | None = None) -> str:
    """The cell a GA on ``cell_id`` re-tests its finalists on."""
    return get_cell(cell_id, path)["cross_cell"]


def annualisation(cell_id: str, path: Path | None = None) -> int:
    """The cell's bars per year (the law value for its timeframe, checked at load)."""
    return get_cell(cell_id, path)["annualisation_ppy"]


class PinnedCellsView(Mapping[str, tuple[str, str, str]]):
    """Read-only cell_id -> (symbol, timeframe, dataset_hash16) over the research cells: the VALIDATED registry
    entries and the QUICK cells added in the app (a view on an explicit registry path reads that file only).

    Every access loads and verifies the registry, so a tampered or missing file makes every lookup refuse. There
    is no setter: a cell enters or leaves the registry only by a sealed data change.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    def _cells(self) -> dict[str, tuple[str, str, str]]:
        entries = [e for e in load(self._path) if e["status"] == "VALIDATED"] if self._path is not None \
            else research_entries()
        return {e["cell_id"]: (e["symbol"], e["timeframe"], e["dataset_hash16"]) for e in entries}

    def __getitem__(self, cell_id: str) -> tuple[str, str, str]:
        return self._cells()[cell_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._cells())

    def __len__(self) -> int:
        return len(self._cells())

    def __repr__(self) -> str:
        return f"PinnedCellsView({self._cells()!r})"
