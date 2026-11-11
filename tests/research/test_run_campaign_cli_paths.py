"""Law 7: run_campaign_cli resolves every artifact path at CALL time, inside ARQVIO_HOME.

An ARQVIO_HOME set after import -- isolated_arqvio_home -- governs every path a
campaign writes or reads: the runner report, the registries, the target audit,
and the spread directory the launch gate hashes and the cost model prices from.
These tests pin that call-time resolution, the containment guard, production
identity with the legacy path, and that importing the module creates nothing.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engine.core.errors import ConfigError
from engine.core.paths import get_artifacts_dir, get_arqvio_home
from engine.research.search import run_campaign_cli as cli

REPO = Path(__file__).resolve().parents[2]


def _real_home_runner_json() -> Path:
    """The production file: <USERPROFILE or HOME>/ARQVIO/artifacts/search_runner_real.json."""
    root = os.environ.get("USERPROFILE") or os.environ["HOME"]
    return Path(root).expanduser().resolve() / "ARQVIO" / "artifacts" / "search_runner_real.json"


def _fingerprint(path: Path) -> tuple[bool, str | None, int | None]:
    if not path.exists():
        return False, None, None
    return True, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def test_runner_json_follows_arqvio_home_set_after_import(isolated_arqvio_home: Path) -> None:
    assert cli.runner_json_path() == isolated_arqvio_home / "artifacts" / cli.RUNNER_JSON_NAME
    assert cli.artifact_path(cli.TARGET_AUDIT_CSV_NAME).parent == isolated_arqvio_home / "artifacts"


def test_the_cost_model_matrix_comes_from_the_gated_spread_dir(
    isolated_arqvio_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_cell gates on and hashes artifact_path("spread"); the backtest
    factories price trades with _load_spread_matrix. Both read one directory."""
    import engine.data.spread_calibration as sc
    from engine.research.search import real_backtest

    seen: list[tuple[str, Path]] = []

    def spy(symbol: str, target_dir: Path) -> str:
        seen.append((symbol, Path(target_dir)))
        return "matrix"

    monkeypatch.setattr(sc, "load", spy)
    assert real_backtest._load_spread_matrix("USDCADm") == "matrix"
    assert seen == [("USDCADm", cli.artifact_path("spread"))]
    assert cli.artifact_path("spread") == isolated_arqvio_home / "artifacts" / "spread"


def test_production_path_is_the_legacy_import_time_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no isolation the call-time path is byte-identical to the old
    module constant ``get_artifacts_dir() / "search_runner_real.json"`` -- the
    legacy tail ``artifacts/search_runner_real.json`` under ARQVIO_HOME."""
    monkeypatch.delenv("ARQVIO_HOME", raising=False)
    legacy = get_artifacts_dir() / "search_runner_real.json"
    assert cli.runner_json_path() == legacy
    assert str(cli.runner_json_path()) == str(get_arqvio_home() / "artifacts" / "search_runner_real.json")
    assert cli.runner_json_path() == _real_home_runner_json()


def test_artifact_path_refuses_to_leave_arqvio_home(isolated_arqvio_home: Path) -> None:
    with pytest.raises(ConfigError, match="escapes ARQVIO_HOME"):
        cli.artifact_path("..", "..", "elsewhere.json")
    assert cli.artifact_path("spread", "USDCADm.json").is_relative_to(isolated_arqvio_home)


def test_artifact_path_refuses_to_leave_the_artifacts_dir(isolated_arqvio_home: Path) -> None:
    """Inside ARQVIO_HOME is not enough: configs/ and data/ are not the
    campaign's to write."""
    with pytest.raises(ConfigError, match="escapes the artifacts dir"):
        cli.artifact_path("..", "configs", "gene_pool.json")


def test_artifact_path_refuses_an_artifacts_dir_outside_arqvio_home(
    isolated_arqvio_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "not_arqvio_home" / "artifacts"
    monkeypatch.setattr(cli, "get_artifacts_dir", lambda: outside)
    with pytest.raises(ConfigError, match="escapes ARQVIO_HOME"):
        cli.runner_json_path()


def test_artifact_path_refuses_a_symlink_escape(isolated_arqvio_home: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    artifacts = isolated_arqvio_home / "artifacts"
    artifacts.mkdir()
    link = artifacts / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Windows without the symlink privilege: a directory junction needs
        # none, and Path.resolve() follows it the same way.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                       check=True, capture_output=True)
    assert link.resolve() == outside.resolve()
    with pytest.raises(ConfigError, match="escapes ARQVIO_HOME"):
        cli.artifact_path("linked", "x.json")


def test_campaign_sync_writes_inside_the_isolated_home_only(
    isolated_arqvio_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real writer, end to end, under an isolated home: the report lands
    there and the real home's file is untouched (hash and mtime)."""
    real = _real_home_runner_json()
    before = _fingerprint(real)
    # Checked BEFORE the writer runs: if resolution ever regresses, this test
    # must fail without writing the real home's file.
    assert cli.runner_json_path().is_relative_to(isolated_arqvio_home)
    monkeypatch.setattr(cli, "load_target_audit", lambda: ({}, {("USDCADm", "H1"): {}}))
    monkeypatch.setattr(
        cli, "run_cell", lambda **kw: {"symbol": kw["symbol"], "timeframe": kw["timeframe"]},
    )
    reports = cli.run_campaign_sync(population=1)
    written = isolated_arqvio_home / "artifacts" / cli.RUNNER_JSON_NAME
    assert reports == [{"symbol": "USDCADm", "timeframe": "H1"}]
    assert written.is_file()
    assert _fingerprint(real) == before


def test_importing_the_module_creates_nothing(tmp_path: Path) -> None:
    """A fresh interpreter whose home is an empty tmp dir imports the module:
    nothing appears under that home, and the real home's file is untouched."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    real = _real_home_runner_json()
    before = _fingerprint(real)
    env = {k: v for k, v in os.environ.items() if k != "ARQVIO_HOME"}
    env.update(USERPROFILE=str(fake_home), HOME=str(fake_home), PYTHONPATH=str(REPO))
    code = (
        "import engine.research.search.run_campaign_cli as m; "
        "print(m.runner_json_path())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    created = sorted(str(p.relative_to(fake_home)) for p in fake_home.rglob("*"))
    assert created == [], f"import created {created}"
    assert Path(out.stdout.strip().splitlines()[-1]) == (
        fake_home.resolve() / "ARQVIO" / "artifacts" / cli.RUNNER_JSON_NAME
    )
    assert _fingerprint(real) == before
