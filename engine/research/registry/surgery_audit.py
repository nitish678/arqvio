"""The SL1 diff audit: every change a stage made under the frozen paths must sit inside its pre-declared symbols.

  python -m engine.research.registry.surgery_audit --base <rev> [--head <rev>]

``git diff -U0 base..head`` over the frozen paths; every changed file must be declared, and every changed line is
mapped to its enclosing TOP-LEVEL symbol by ``ast`` on the version the line belongs to (a removed line on base, an
added line on head), so the check is by symbol and survives line shifts. A symbol is a top-level def/class name, an
assignment target, or ``<imports>`` for an import statement; comment and blank lines belong to the node below them,
lines after the last node to ``<trailer>``. Exit 0 only when no hunk touches an undeclared file or symbol.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FROZEN = (
    "engine/validation/", "engine/research/backtest/", "engine/research/ga/", "engine/research/mql5/",
    "engine/research/live/", "engine/research/portfolio/", "engine/research/alerts/", "engine/research/health/",
)
# U1's declaration (DR-2026-10-07-u1 §1): file -> the top-level symbols that may change.
DECLARED: dict[str, frozenset[str]] = {
    "engine/research/ga/config.py": frozenset({"<imports>", "PINNED_CELLS", "cross_cell_for"}),
    "engine/research/ga/runner.py": frozenset({"main"}),
    "engine/validation/cell_dsr.py": frozenset({"PERIODS_PER_YEAR"}),
}
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    path: str
    header: str
    removed: dict[int, str]  # base line -> symbol
    added: dict[int, str]    # head line -> symbol

    def symbols(self) -> set[str]:
        return set(self.removed.values()) | set(self.added.values())


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def _node_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return ["<imports>"]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    targets = node.targets if isinstance(node, ast.Assign) else (
        [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
    names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
    return names or [f"<{type(node).__name__} at line {node.lineno}>"]


def symbol_map(source: str) -> dict[int, str]:
    """line number -> the top-level symbol that owns it (comments and blanks go to the node below)."""
    nodes = ast.parse(source).body
    owner: dict[int, str] = {}
    prev_end = 0
    for node in nodes:
        name = "/".join(_node_names(node))
        for line in range(prev_end + 1, node.end_lineno + 1):  # the gap above (decorators, comments) + the node
            owner[line] = name
        prev_end = node.end_lineno
    for line in range(prev_end + 1, source.count("\n") + 2):
        owner[line] = "<trailer>"
    return owner


def hunks(repo: Path, base: str, head: str, path: str) -> list[Hunk]:
    diff = _git(repo, "diff", "-U0", "--no-color", f"{base}..{head}", "--", path)
    base_owner = symbol_map(_git(repo, "show", f"{base}:{path}"))
    head_owner = symbol_map(_git(repo, "show", f"{head}:{path}"))
    out = []
    for line in diff.splitlines():
        m = _HUNK.match(line)
        if not m:
            continue
        a, b, c, d = int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1)
        out.append(Hunk(path, line.split(" @@")[0] + " @@",
                        {n: base_owner[n] for n in range(a, a + b)},
                        {n: head_owner[n] for n in range(c, c + d)}))
    return out


def audit(repo: Path, base: str, head: str, declared: dict[str, frozenset[str]] = DECLARED,
          frozen: tuple[str, ...] = FROZEN) -> tuple[list[str], int]:
    """(report lines, breaches). A breach is an undeclared changed file or a hunk touching an undeclared symbol."""
    lines = [f"## SL1 surgery diff audit: {base}..{head}",
             f"# base {_git(repo, 'rev-parse', base).strip()}  head {_git(repo, 'rev-parse', head).strip()}",
             f"# frozen paths: {' '.join(frozen)}"]
    for path, symbols in sorted(declared.items()):
        lines.append(f"# declared {path}: {sorted(symbols)}")
    lines.append("# every changed file base..head (frozen = under a frozen path):")
    for row in _git(repo, "diff", "--name-status", "--no-renames", f"{base}..{head}").splitlines():
        status, path = row.split("\t", 1)
        lines.append(f"  {status} {path}{'  [frozen]' if path.startswith(frozen) else ''}")
    breaches = 0
    changed = _git(repo, "diff", "--name-status", "--no-renames", f"{base}..{head}", "--", *frozen).splitlines()
    for row in changed:
        status, path = row.split("\t", 1)
        if path not in declared or status != "M":
            breaches += 1
            lines.append(f"FILE {path} status {status}: OUT-OF-SCOPE (not a declared modification)")
            continue
        lines.append(f"FILE {path} status {status}: declared")
        for h in hunks(repo, base, head, path):
            bad = sorted(h.symbols() - declared[path])
            breaches += bool(bad)
            verdict = f"OUT-OF-SCOPE {bad}" if bad else "IN-SCOPE"
            lines.append(f"  {h.header} -{sorted(h.removed)} +{sorted(h.added)} symbols {sorted(h.symbols())}: "
                         f"{verdict}")
    n = sum(1 for line in lines if line.startswith("  @@"))
    lines.append(f"SURGERY BOUNDARY: {'HELD' if breaches == 0 else 'BREACHED'} ({len(changed)} frozen files changed, "
                 f"{n} hunks, {breaches} out of scope)")
    return lines, breaches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    lines, breaches = audit(args.repo, args.base, args.head)
    print("\n".join(lines))
    return 0 if breaches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
