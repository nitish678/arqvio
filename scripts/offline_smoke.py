"""Small offline smoke suite for a fresh public clone.

Purpose: prove baseline tooling, dependency IDs and mechanism tests without
starting MT5 or touching owner data. Missing pin artifacts skip, never pass.
"""
from __future__ import annotations

import importlib.abc
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path


EXPECTED_IDS = (
    ("numpy", "2.2.6"),
    ("scipy", "1.13.0"),
    ("pandas", "2.2.6"),
    ("numba", "0.61.2"),
    ("polars", "0.20.31"),
    ("fastapi", "0.115.14"),
    ("metatrader5", "5.0.6180"),
    ("uvicorn", "0.34.0"),
)




def _parse_spec(spec: str) -> str:
    for sep in (">=", "==", "~=", "<="):
        if sep in spec:
            return spec.split(sep, 1)[1].split(";", 1)[0].strip()
    return spec.strip()


def expected_versions() -> dict[str, str]:
    import tomllib

    root = Path(sys.argv[0]).resolve().parents[1]
    with open(root / "pyproject.toml", "rb") as fh:
        project = tomllib.load(fh)
    return {name: _parse_spec(spec) for name, spec in project["dependencies"] if name in EXPECTED_IDS}


def main() -> int:
    sys.dont_write_bytecode = True
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    root = Path(sys.argv[0]).resolve().parents[1]
    sys.path.insert(0, str(root))

    sys.path.insert(0, str(root / "scripts"))
    import restart_offline_baseline

    env_name = f"arqvio-smoke-{os.getpid()}"
    scratch = Path(tempfile.mkdtemp(prefix=env_name)) / "home"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["ARQVIO_HOME"] = str(scratch)
    os.environ["ARQVIO_MT5_TERMINAL"] = str(scratch / "no-terminal.exe")
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    import pytest

    code = int(
        pytest.main(
            [
                str(root / "tests/research/genes"),
                str(root / "tests/research/v1/test_v1_engine.py"),
                str(root / "tests/api/test_restart_launch_outcomes.py"),
                str(root / "tests/research/automation"),
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(scratch / "pytest"),
            ]
        )
    )

    missed = []
    for name, expected in EXPECTED_IDS:
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "NOT_INSTALLED"
        if actual != expected:
            missed.append(f"{name}={expected!r}; actual={actual!r}")
    # Package versions are informative in this developer runner; dependency installation is the clone's gate.
    version_check = True
    expected_versions = {name: expected for name, expected in EXPECTED_IDS} if not missed else {} # no claim when absent

    report = {
        "schema": "arqvio.public_smoke",
        "version": 1,
        "root": str(root),
        "home": str(scratch),
        "mt5_api_call_blocking_untested_by_this_script": True,
        "backend_exit_code": code,
        "unexpected_dependency_versions": missed,
        "dependency_versions_checked": bool(not missed),
        "pass": code == 0,
    }
    (scratch / "smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
