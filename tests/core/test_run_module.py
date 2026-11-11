"""``arqvio-engine --run-module``: the packaged sidecar's stand-in for ``python -m``, whitelisted modules only."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from engine.__main__ import RUNNABLE_MODULES, run_module

ROOT = Path(__file__).resolve().parents[2]


def test_only_the_launcher_and_the_ga_runner_can_be_run() -> None:
    assert RUNNABLE_MODULES == ("engine.api.campaign_launch", "engine.research.ga.runner")
    assert run_module([]) == 2
    assert run_module(["os"]) == 2
    assert run_module(["engine.api.server"]) == 2


def test_the_dispatch_runs_the_module_main_with_the_rest_of_argv(capsys) -> None:
    # the launcher refuses an argv without "--" -- proof its main() received exactly the rest
    assert run_module(["engine.api.campaign_launch", "--runner", "x"]) == 2
    assert "REFUSED: launcher argv has no '--'" in capsys.readouterr().out


def test_the_entry_point_dispatches_before_starting_a_server() -> None:
    result = subprocess.run([sys.executable, "-m", "engine", "--run-module", "nope"], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2
    assert "REFUSED: --run-module takes one of" in result.stdout
