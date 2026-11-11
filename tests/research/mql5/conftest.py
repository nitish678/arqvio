"""Shared fixtures for the S6 MQL5 tests."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

PINNED = (("USDCADm", "aed3bf314539ec5a"), ("XAUUSDm", "447b843f87438a9a"))


@pytest.fixture(scope="session")
def real_artifacts_dir() -> Path:
    """The artifacts dir of the ARQVIO_HOME this test process started with, resolved through
    engine/core/paths (Law 7). Session-scoped, so it is resolved before any function-scoped
    isolated_arqvio_home takes effect; tests only ever READ from it."""
    from engine.core.paths import get_artifacts_dir

    return get_artifacts_dir()


@pytest.fixture
def pinned_home(real_artifacts_dir: Path, isolated_arqvio_home: Path) -> Path:
    """isolated_arqvio_home seeded with COPIES of the two pinned cells' snapshots, spread matrices and specs."""
    need = [real_artifacts_dir / "datasets" / f"{s}_H1_{d}.{ext}" for s, d in PINNED for ext in ("json", "npz")]
    need += [real_artifacts_dir / sub / f"{s}.json" for sub in ("spread", "specs") for s, _ in PINNED]
    missing = [p for p in need if not p.is_file()]
    if missing:
        pytest.skip(f"pinned cells not on this machine: {missing[0]}")
    for p in need:
        target = isolated_arqvio_home / "artifacts" / p.relative_to(real_artifacts_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    return isolated_arqvio_home
