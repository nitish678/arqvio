"""Shared fixtures for the cell-registry tests: a sealed copy of the real registry to tamper with, and the pinned
cells' source artifacts copied into an isolated home."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from engine.research.registry.cells import registry_path

PINNED = (("USDCADm", "aed3bf314539ec5a"), ("XAUUSDm", "447b843f87438a9a"), ("GBPJPYm", "5979a019aa6b3e0c"))


@pytest.fixture
def registry_doc() -> dict[str, Any]:
    """The real registry document, parsed (a fresh copy per test)."""
    return json.loads(registry_path().read_text(encoding="utf-8"))


@pytest.fixture
def write_registry(tmp_path: Path):
    """write_registry(doc) -> the path of ``doc`` written as a registry file under tmp_path."""
    def write(doc: dict[str, Any], name: str = "cells_registry.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return path
    return write


@pytest.fixture(scope="session")
def real_artifacts_dir() -> Path:
    """The artifacts dir this test process started with (READ only), resolved before any isolated_arqvio_home."""
    from engine.core.paths import get_artifacts_dir

    return get_artifacts_dir()


@pytest.fixture
def pinned_sources(real_artifacts_dir: Path, isolated_arqvio_home: Path) -> Path:
    """isolated_arqvio_home seeded with COPIES of the pinned cells' sidecars, spread matrices, specs and the target
    audit -- the sources the registry's provenance quotes."""
    need = [real_artifacts_dir / "datasets" / f"{s}_H1_{d}.json" for s, d in PINNED]
    need += [real_artifacts_dir / sub / f"{s}.json" for sub in ("spread", "specs") for s, _ in PINNED]
    need.append(real_artifacts_dir / "history_depth_audit_target.csv")
    missing = [p for p in need if not p.is_file()]
    if missing:
        pytest.skip(f"pinned cells not on this machine: {missing[0]}")
    for p in need:
        target = isolated_arqvio_home / "artifacts" / p.relative_to(real_artifacts_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    return isolated_arqvio_home
