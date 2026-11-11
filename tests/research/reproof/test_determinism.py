"""Part 1.5: the FULL re-proof of a registered cell, run twice on the same HEAD and inputs, writes byte-identical
proof.json (no clock reading enters it). It runs on the real pinned inputs and the recorded U2 closing battery log;
the log is an untracked run product, so a staged worktree without it skips."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.research.reproof.checks import reproof
from tests.research.reproof.conftest import U2_BATTERY_LOG


def test_a_double_run_is_byte_identical(tmp_path: Path) -> None:
    if not U2_BATTERY_LOG.is_file():
        pytest.skip("the recorded U2 battery log (untracked artifacts) is not in this tree")
    first = reproof("xauusdm_h1", out_dir=tmp_path / "a", battery_log=U2_BATTERY_LOG)
    second = reproof("xauusdm_h1", out_dir=tmp_path / "b", battery_log=U2_BATTERY_LOG)
    a, b = ((tmp_path / d / "xauusdm_h1.proof.json").read_bytes() for d in ("a", "b"))
    assert a == b and first == second
    assert first["verdict"] == "PROVEN" and first["checks"]["rederive"]["equal"] == 20
