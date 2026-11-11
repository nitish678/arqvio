"""Small MT5-blocked restart baseline; does not claim full acceptance or market performance."""
from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
from pathlib import Path


def block_terminal_calls() -> int:
    # Importing the official package exposes constants/types without initialize().
    # Replace API functions before importing any project module; never attach.
    import MetaTrader5 as mt5

    def refuse(*args, **kwargs):
        raise RuntimeError("Restart offline baseline prohibits real MetaTrader5 API calls")

    names = [name for name, value in vars(mt5).items()
             if not name.startswith("_") and (inspect.isbuiltin(value) or inspect.isfunction(value))]
    if "initialize" not in names or "shutdown" not in names:
        raise RuntimeError("Cannot establish MT5 call blocker; refusing baseline")
    for name in names:
        setattr(mt5, name, refuse)
    return len(names)


def main() -> None:
    root = Path(sys.argv[1]).resolve(strict=True)
    evidence = Path(sys.argv[2]).resolve(strict=True)
    report = evidence / "offline-baseline-pytest-v2.json"
    if report.exists():
        raise RuntimeError("Baseline report already exists; select a new evidence directory")
    sys.path.insert(0, str(root))
    blocked_api_functions = block_terminal_calls()
    sys.dont_write_bytecode = True
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest

    class Report:
        def __init__(self):
            self.rows = []
            self.collection_errors = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed or report.skipped:
                self.rows.append({"id": report.nodeid, "phase": report.when,
                                  "outcome": report.outcome, "seconds": report.duration})

        def pytest_collectreport(self, report):
            if report.failed:
                self.collection_errors.append(str(report.longrepr))

    collector = Report()
    targets = sys.argv[3:] or ["tests/research/genes", "tests/research/v1/test_v1_engine.py"]
    with tempfile.TemporaryDirectory(prefix="arqvio-restart-offline-") as scratch:
        os.environ["ARQVIO_HOME"] = str(Path(scratch) / "home")
        os.environ["ARQVIO_MT5_TERMINAL"] = str(Path(scratch) / "NO_TERMINAL.exe")
        args = [str(root / t) for t in targets] + ["-q", "-p", "no:cacheprovider", "--basetemp", str(Path(scratch) / "pytest")]
        code = int(pytest.main(args, plugins=[collector]))
    output = {"schema": "arqvio.restart.offline_baseline", "version": 1,
              "exit_code": code, "targets": targets, "mt5_api_functions_blocked": blocked_api_functions,
              "scope": "Selected offline test targets only; no real-terminal/parity/profit evidence",
              "reports": collector.rows, "collection_errors": collector.collection_errors}
    report.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Baseline evidence: {report}")
    sys.exit(code)


if __name__ == "__main__":
    main()
