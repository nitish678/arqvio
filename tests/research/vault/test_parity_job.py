"""PARITY TEST guards (W4-7 7.8): a named test terminal only, never a deployed bot's terminal, only what the tester
parity EA expresses; grades on the research bands."""
from __future__ import annotations

import pytest

from engine.research.vault import parity_job as pj


def test_grade_bands():
    assert pj.grade(8, 8) == (100.0, "A")
    assert pj.grade(6, 8) == (75.0, "B")
    assert pj.grade(5, 8) == (62.5, "C")
    assert pj.grade(4, 8) == (50.0, "D")
    assert pj.grade(1, 8)[1] == "F" and pj.grade(0, 0)[1] == "F"


def test_no_terminal_named_is_refused(monkeypatch):
    monkeypatch.delenv("ARQVIO_MT5_TERMINAL", raising=False)
    with pytest.raises(pj.ParityRefused, match="no test terminal named"):
        pj.test_terminal()


def test_a_deployed_bots_terminal_is_refused(tmp_path, monkeypatch):
    from engine.research.deploy import full_deploy as fd

    exe = tmp_path / "terminal64.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("ARQVIO_MT5_TERMINAL", str(exe))
    monkeypatch.setattr(fd, "load_bots", lambda path=None: [{"bot": "b1", "status": "LIVE", "terminal": str(exe)}])
    with pytest.raises(pj.ParityRefused, match="runs deployed bot b1"):
        pj.test_terminal()
    monkeypatch.setattr(fd, "load_bots", lambda path=None: [])
    assert pj.test_terminal() == exe


def test_only_h1_and_plain_dna(tmp_path, monkeypatch):
    import dataclasses

    import numpy as np

    from engine.research.search.sampler import random_dna

    d = random_dna(np.random.default_rng(2))
    with pytest.raises(pj.ParityRefused, match="H1"):
        pj.preflight({"timeframe": "M15", "dna": d.to_dict()})
    gated = dataclasses.replace(d, filter_switches=("volatility",))
    with pytest.raises(pj.ParityRefused, match="Phase 6"):
        pj.preflight({"timeframe": "H1", "dna": gated.to_dict()})
