"""Diagnostics and the support report (W4-9 9.7): real machine numbers, masked digit runs, ARQVIO magics kept."""
from __future__ import annotations

from engine.api import routes_app_support as sup


def test_mask_hides_long_numbers_but_keeps_arqvio_magics():
    assert sup.mask("login 12345678 magic 610500002 ok 2026") == "login ######## magic 610500002 ok 2026"


def test_diagnostics_and_report_render(tmp_path, monkeypatch):
    monkeypatch.setenv("ARQVIO_HOME", str(tmp_path))
    d = sup.diagnostics()
    assert d["cores"] and d["disk"]["path"]
    assert set(d) >= {"utc", "version", "os", "python", "memory", "vault", "recent_problems"}
    text = sup.support_report()["text"]
    assert text.startswith("ARQVIO support report") and "last 150 log lines:" in text
