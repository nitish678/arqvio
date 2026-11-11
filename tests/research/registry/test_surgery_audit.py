"""The SL1 diff audit, proven on a throwaway git repo: an in-scope edit holds, and every out-of-scope kind breaches."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from engine.research.registry.surgery_audit import audit, symbol_map

FROZEN = ("frozen/",)
DECLARED = {"frozen/mod.py": frozenset({"<imports>", "TABLE", "allowed"})}
BASE_SRC = '''"""A frozen module."""
import math

# The table.
TABLE = {"a": 1}


def allowed(x):
    return x + 1


@staticmethod
def forbidden(x):
    return math.sqrt(x)
'''


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "audit@test")
    _git(tmp_path, "config", "user.name", "audit")
    _git(tmp_path, "config", "core.autocrlf", "false")
    (tmp_path / "frozen").mkdir()
    (tmp_path / "frozen" / "mod.py").write_text(BASE_SRC, encoding="utf-8", newline="\n")
    (tmp_path / "frozen" / "other.py").write_text("X = 1\n", encoding="utf-8", newline="\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "tag", "base")
    return tmp_path


def _commit(repo: Path, rel: str, text: str | None) -> None:
    path = repo / rel
    if text is None:
        path.unlink()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"edit {rel}")


def test_symbol_map_gives_comments_to_the_node_below() -> None:
    owner = symbol_map(BASE_SRC)
    assert owner[1] == "<Expr at line 1>"
    assert owner[2] == "<imports>"
    assert owner[3] == owner[4] == owner[5] == "TABLE"
    assert owner[8] == owner[9] == "allowed"
    assert owner[12] == owner[13] == owner[14] == "forbidden"


def test_in_scope_edits_hold_even_when_they_shift_lines(repo: Path) -> None:
    src = BASE_SRC.replace("import math\n", "import math\nimport os\n")
    src = src.replace('TABLE = {"a": 1}', 'TABLE = {\n    "a": 1,\n    "b": 2,\n}').replace("x + 1", "x + 2")
    _commit(repo, "frozen/mod.py", src)
    _commit(repo, "elsewhere/new.py", "Y = 2\n")  # outside the frozen paths: listed, never a breach
    lines, breaches = audit(repo, "base", "HEAD", DECLARED, FROZEN)
    assert breaches == 0, lines
    assert lines[-1].startswith("SURGERY BOUNDARY: HELD (1 frozen files changed, 3 hunks")


@pytest.mark.parametrize("rel, text, why", [
    ("frozen/mod.py", BASE_SRC.replace("math.sqrt(x)", "math.sqrt(x) + 0"), "forbidden"),
    ("frozen/mod.py", BASE_SRC.replace('"""A frozen module."""', '"""A frozen module, edited."""'), "<Expr"),
    ("frozen/mod.py", BASE_SRC + "\nNEW = 1\n", "NEW"),
    ("frozen/mod.py", BASE_SRC.replace("@staticmethod\n", "@classmethod\n"), "forbidden"),
    ("frozen/other.py", "X = 2\n", "not a declared modification"),
    ("frozen/added.py", "Z = 1\n", "status A"),
    ("frozen/other.py", None, "status D"),
], ids=["undeclared-function", "docstring", "new-symbol", "decorator", "undeclared-file", "added-file",
        "deleted-file"])
def test_out_of_scope_edits_breach(repo: Path, rel: str, text: str | None, why: str) -> None:
    _commit(repo, rel, text)
    lines, breaches = audit(repo, "base", "HEAD", DECLARED, FROZEN)
    assert breaches == 1, lines
    assert any("OUT-OF-SCOPE" in line and why in line for line in lines), lines
    assert lines[-1].startswith("SURGERY BOUNDARY: BREACHED")
