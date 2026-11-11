"""Engine mode resolution tests for run_campaign_cli.

Covers:
  - resolve_engine_mode: "auto" -> fast on pinned, slow on live;
    explicit "slow"/"fast" pass through.
  - main() CLI: --engine accepts slow|fast|auto and forwards correctly.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from engine.research.search.run_campaign_cli import (
    resolve_engine_mode,
)


def test_resolve_auto_pinned_is_fast() -> None:
    assert resolve_engine_mode("auto", dataset_id="pinned123") == "fast"


def test_resolve_auto_live_is_slow() -> None:
    assert resolve_engine_mode("auto", dataset_id=None) == "slow"


def test_resolve_explicit_fast_passes_through() -> None:
    assert resolve_engine_mode("fast", dataset_id=None) == "fast"
    assert resolve_engine_mode("fast", dataset_id="pinned123") == "fast"


def test_resolve_explicit_slow_passes_through() -> None:
    assert resolve_engine_mode("slow", dataset_id=None) == "slow"
    assert resolve_engine_mode("slow", dataset_id="pinned123") == "slow"


# --- CLI forwarding ---------------------------------------------------------


def test_cli_default_engine_is_auto() -> None:
    import inspect

    from engine.research.search import run_campaign_cli as mod

    sig = inspect.signature(mod.run_campaign_sync)
    assert sig.parameters["engine_mode"].default == "auto"


def test_cli_run_cell_default_is_auto() -> None:
    import inspect

    from engine.research.search import run_campaign_cli as mod

    sig = inspect.signature(mod.run_cell)
    assert sig.parameters["engine_mode"].default == "auto"


def _assert_printed_paths_inside(out: str, home: Path) -> None:
    """Every path main() reports -- ARQVIO_HOME, TARGET_AUDIT_CSV and the WROTE
    runner report -- lies under the isolated home (Law 7)."""
    lines = {ln.split("=", 1)[0]: ln.split("=", 1)[1] for ln in out.splitlines() if "=" in ln}
    wrote = next(ln[len("WROTE "):] for ln in out.splitlines() if ln.startswith("WROTE "))
    for path in (lines["ARQVIO_HOME"], lines["TARGET_AUDIT_CSV"], wrote):
        assert Path(path).is_relative_to(home), path


@pytest.mark.parametrize("mode", ["slow", "fast", "auto"])
def test_cli_main_forwards_engine(
    mode: str, isolated_arqvio_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from engine.research.search import run_campaign_cli as mod

    captured: dict = {}

    def fake_sync(**kwargs):
        captured.update(kwargs)
        return []

    with mock.patch.object(mod, "load_target_audit", return_value=({}, {})), \
         mock.patch.object(mod, "run_campaign_sync", side_effect=fake_sync):
        rc = mod.main(["--engine", mode])
    assert rc == 0
    assert captured["engine_mode"] == mode
    _assert_printed_paths_inside(capsys.readouterr().out, isolated_arqvio_home)


def test_cli_main_defaults_to_auto(
    isolated_arqvio_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from engine.research.search import run_campaign_cli as mod

    captured: dict = {}

    def fake_sync(**kwargs):
        captured.update(kwargs)
        return []

    with mock.patch.object(mod, "load_target_audit", return_value=({}, {})), \
         mock.patch.object(mod, "run_campaign_sync", side_effect=fake_sync):
        rc = mod.main([])
    assert rc == 0
    assert captured["engine_mode"] == "auto"
    _assert_printed_paths_inside(capsys.readouterr().out, isolated_arqvio_home)


def test_cli_main_rejects_invalid_engine() -> None:
    from engine.research.search import run_campaign_cli as mod

    with pytest.raises(SystemExit):
        mod.main(["--engine", "bogus"])
