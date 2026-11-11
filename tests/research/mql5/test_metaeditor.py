"""The MetaEditor driver parses its log fail-loud; the real compile is probe-skipped when absent."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from engine.research.mql5 import metaeditor as me

FIX = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIX / name).read_bytes().decode("utf-16")


def test_a_real_clean_log_parses_to_zero_and_zero() -> None:
    e, w, diags, ms = me.parse_log(_fixture("metaeditor_clean.utf16.txt"))
    assert (e, w, diags) == (0, 0, ()) and ms == 1625


def test_a_real_error_log_lists_every_error() -> None:
    e, w, diags, ms = me.parse_log(_fixture("metaeditor_errors.utf16.txt"))
    assert (e, w, ms) == (3, 0, -1) and len(diags) == 3 and "error 125: variable already defined" in diags[0]


@pytest.mark.parametrize("mutate, needle", [
    (lambda t: t.replace("Result:", "Summary:"), "no known shape"),
    (lambda t: "\n".join(ln for ln in t.splitlines() if not ln.startswith("Result")), "no Result line"),
    (lambda t: t + "\nResult: 0 errors, 0 warnings, 1 ms elapsed, cpu='X64 Regular'", "a second Result line"),
    (lambda t: t.replace("Result: 0 errors", "Result: 1 errors"), "lists 0 / 0"),
    (lambda t: t + "\nsomething MetaEditor never wrote", "no known shape"),
])
def test_every_ambiguity_is_an_error_not_a_pass(mutate, needle: str) -> None:
    with pytest.raises(me.CompileLogError, match=needle):
        me.parse_log(mutate(_fixture("metaeditor_clean.utf16.txt")))


def test_require_clean_refuses_errors_and_names_them(tmp_path: Path) -> None:
    e, w, diags, ms = me.parse_log(_fixture("metaeditor_errors.utf16.txt"))
    r = me.CompileResult(tmp_path / "x.mq5", tmp_path / "x.ex5", e, w, diags, ms, tmp_path / "x.log")
    with pytest.raises(me.CompileError, match="3 errors, 0 warnings"):
        me.require_clean(r)


def _metaeditor_or_skip() -> Path:
    from engine.research.mql5.terminal import TerminalPathError, metaeditor_exe, mql5_dir

    try:
        metaeditor_exe()
        return mql5_dir()
    except (FileNotFoundError, TerminalPathError) as exc:
        pytest.skip(f"LOUD SKIP: no MetaEditor on this machine ({exc}); the stage run log is the real-compile "
                    "evidence (artifacts/reports/s6)")


@pytest.mark.parametrize("family", ["MA_CROSS", "RSI_REVERSION", "STOCH_CROSS", "MACD_CROSS", "DONCHIAN_BREAKOUT"])
def test_a_generated_ea_compiles_with_zero_errors_and_zero_warnings(pinned_home: Path, family: str) -> None:
    mql5 = _metaeditor_or_skip()
    from engine.research.mql5.cell_config import battery_cell
    from engine.research.mql5.codegen import generate
    from engine.research.search.sampler import random_dna

    dna = dataclasses.replace(random_dna(np.random.default_rng(20261001)), signal=family)
    ea = generate(dna, battery_cell(), magic=610_000_000)
    out = mql5 / "Experts" / "ARQVIO_S6" / "pytest"
    out.mkdir(parents=True, exist_ok=True)
    src = out / f"{ea.name}.mq5"
    src.write_text(ea.source, encoding="utf-8", newline="\n")
    r = me.require_clean(me.compile_mq5(src, log_path=out / f"{ea.name}.log"))
    assert r.ex5.is_file() and (r.errors, r.warnings) == (0, 0)
