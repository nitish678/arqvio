"""The MetaEditor compile driver: compile one .mq5, parse the log FAIL-LOUD.

MetaEditor's exit code says nothing (it is 1 on a clean compile, S6 probe F2),
so the verdict comes from its UTF-16 log alone. Every line must be one of:

* ``<path> : information: <text>``  /  `` : information: <text>``
* ``<path>(<line>,<col>) : error <code>: <text>``  /  ``... : warning <code>: <text>``
* ``Result: <e> error(s), <w> warning(s)[, <ms> ms elapsed, cpu='<cpu>']``  (exactly once; a failed
  compile has no timing suffix)
* a blank line

Anything else is an ambiguity, and an ambiguity is an error (``CompileLogError``),
never a pass. A compile is clean only with 0 errors, 0 warnings and a fresh .ex5.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from engine.research.mql5.terminal import metaeditor_exe

_INFO = re.compile(r"^(?:.*\S)? ?: information: .+$")
_DIAG = re.compile(r"^(?P<path>.+)\((?P<line>\d+),(?P<col>\d+)\) : (?P<kind>error|warning) (?P<code>\d+): (?P<msg>.+)$")
# A failed compile ends "Result: 9 errors, 0 warnings" (no timing); a clean one adds ", <ms> ms elapsed, cpu='..'".
_RESULT = re.compile(r"^Result: (?P<e>\d+) errors?, (?P<w>\d+) warnings?(?:, (?P<ms>\d+) ms elapsed, cpu='[^']*')?$")


class CompileLogError(RuntimeError):
    """A MetaEditor log the parser cannot read unambiguously."""


class CompileError(RuntimeError):
    """A compile that is not clean (errors, warnings, or no fresh .ex5)."""


@dataclass(frozen=True)
class CompileResult:
    source: Path
    ex5: Path
    errors: int
    warnings: int
    diagnostics: tuple[str, ...]
    elapsed_ms: int
    log_path: Path

    @property
    def clean(self) -> bool:
        return self.errors == 0 and self.warnings == 0


def parse_log(text: str) -> tuple[int, int, tuple[str, ...], int]:
    """(errors, warnings, diagnostic lines, elapsed ms) from a MetaEditor log, or CompileLogError."""
    result = None
    diags: list[str] = []
    n_err = n_warn = 0
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip("﻿").rstrip()
        if not line.strip():
            continue
        m = _RESULT.match(line)
        if m:
            if result is not None:
                raise CompileLogError(f"log line {no}: a second Result line: {line!r}")
            result = (int(m["e"]), int(m["w"]), int(m["ms"]) if m["ms"] is not None else -1)
            continue
        d = _DIAG.match(line)
        if d:
            diags.append(line)
            n_err += d["kind"] == "error"
            n_warn += d["kind"] == "warning"
            continue
        if _INFO.match(line):
            continue
        raise CompileLogError(f"log line {no} has no known shape: {line!r}")
    if result is None:
        raise CompileLogError("the log has no Result line")
    if (n_err, n_warn) != result[:2]:
        raise CompileLogError(f"Result says {result[0]} errors / {result[1]} warnings but the log lists "
                              f"{n_err} / {n_warn}")
    return result[0], result[1], tuple(diags), result[2]


def compile_mq5(source: Path, *, log_path: Path, timeout_s: float = 120.0, metaeditor: Path | None = None,
                include: Path | None = None) -> CompileResult:
    """Compile ``source`` with the resolved terminal's MetaEditor (or ``metaeditor``, with ``include`` as its MQL5
    include root); parse the log; never trust the exit code."""
    source = Path(source).resolve()
    ex5 = source.with_suffix(".ex5")
    if ex5.exists():
        ex5.unlink()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    started = time.time()
    argv = [str(metaeditor or metaeditor_exe()), f"/compile:{source}", f"/log:{log_path}"]
    if include is not None:
        argv.insert(2, f"/inc:{include}")
    subprocess.run(argv, timeout=timeout_s, check=False, capture_output=True)
    if not log_path.is_file():
        raise CompileLogError(f"MetaEditor wrote no log at {log_path}")
    errors, warnings, diags, ms = parse_log(log_path.read_bytes().decode("utf-16"))
    if not ex5.is_file() and errors == 0:
        raise CompileError(f"0 errors reported but no {ex5}")
    if ex5.is_file() and ex5.stat().st_mtime < started - 1.0:
        raise CompileError(f"{ex5} predates this compile")
    return CompileResult(source, ex5, errors, warnings, diags, ms, log_path)


def require_clean(result: CompileResult) -> CompileResult:
    if not result.clean:
        raise CompileError(f"{result.source.name}: {result.errors} errors, {result.warnings} warnings:\n  "
                           + "\n  ".join(result.diagnostics))
    return result


__all__ = ["CompileError", "CompileLogError", "CompileResult", "compile_mq5", "parse_log", "require_clean"]
