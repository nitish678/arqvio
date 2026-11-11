"""The audited alert spool: append-only JSONL, one SHA-256 chain, call-time path (DR §6).

Each line is a JSON object with ``prev_hash`` (the previous line's ``line_hash``; GENESIS for the first) and
``line_hash`` = SHA-256 of the canonical JSON of the line WITHOUT ``line_hash``. ``append`` writes the whole line in
one write, flushes and fsyncs before returning, so a line the caller has been handed back is on disk. ``verify``
re-derives every hash and names the first line that does not chain. No git_sha is ever written into a line (OQ4's
mask-by-rule would apply if a cross-commit identity proof ever referenced the spool).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from engine.core.paths import get_artifacts_dir

SCHEMA = "arqvio.alert"
VERSION = 1
GENESIS = "0" * 64


class SpoolError(RuntimeError):
    """A spool that does not chain, or a line that cannot be appended."""


def spool_path() -> Path:
    """Resolved at call time; creates nothing (``append`` creates the directory it writes into)."""
    return get_artifacts_dir() / "alerts" / "spool.jsonl"


def canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def line_hash(record: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in record.items() if k != "line_hash"}).encode("ascii")).hexdigest()


def _last(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        step = min(size, 1 << 16)
        while True:
            fh.seek(size - step)
            chunk = fh.read(step)
            lines = chunk.rstrip(b"\n").split(b"\n")
            if len(lines) > 1 or step == size:
                return json.loads(lines[-1].decode("ascii"))
            step = min(size, step * 2)


def append(record: dict, path: Path | None = None) -> dict:
    """Chain, hash and durably append ``record`` (its seq / prev_hash / line_hash are assigned here)."""
    path = path or spool_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    last = _last(path)
    out = {"schema": SCHEMA, "version": VERSION, **record,
           "seq": 0 if last is None else int(last["seq"]) + 1,
           "prev_hash": GENESIS if last is None else last["line_hash"]}
    out["line_hash"] = line_hash(out)
    data = (canonical(out) + "\n").encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    try:
        if os.write(fd, data) != len(data):
            raise SpoolError(f"short write appending seq {out['seq']} to {path}")
        os.fsync(fd)
    finally:
        os.close(fd)
    return out


@dataclass(frozen=True)
class ChainReport:
    ok: bool
    lines: int
    first_bad: int | None
    reason: str


def verify(path: Path | None = None) -> ChainReport:
    path = path or spool_path()
    if not path.is_file():
        return ChainReport(True, 0, None, "no spool")
    prev = GENESIS
    n = 0
    with path.open("rb") as fh:
        for n, raw in enumerate(fh, 1):
            try:
                rec = json.loads(raw.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return ChainReport(False, n, n, f"line {n} is not JSON: {exc}")
            if rec.get("prev_hash") != prev:
                return ChainReport(False, n, n, f"line {n}: prev_hash does not chain")
            if rec.get("line_hash") != line_hash(rec):
                return ChainReport(False, n, n, f"line {n}: line_hash does not match its content")
            if rec.get("seq") != n - 1:
                return ChainReport(False, n, n, f"line {n}: seq {rec.get('seq')} != {n - 1}")
            prev = rec["line_hash"]
    return ChainReport(True, n, None, "chain intact")


def tail(n: int, path: Path | None = None) -> list[dict]:
    path = path or spool_path()
    if not path.is_file():
        return []
    with path.open("rb") as fh:
        rows = fh.read().decode("ascii").splitlines()
    return [json.loads(r) for r in rows[-n:]] if n > 0 else []


__all__ = ["ChainReport", "GENESIS", "SCHEMA", "SpoolError", "VERSION", "append", "canonical", "line_hash", "spool_path",
           "tail", "verify"]
