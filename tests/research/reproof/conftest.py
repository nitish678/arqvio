"""Shared fixtures for the re-proof tests: a battery log in the closing-chain format, stamped at a chosen HEAD."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from engine.core.paths import PROJECT_ROOT
from engine.research.reproof.checks import BANKED_BATTERY

# The recorded U2 closing battery (an untracked run product; present on the stage machine, absent in a staged tree).
U2_BATTERY_LOG = PROJECT_ROOT / "artifacts" / "reports" / "u2" / "closing_parity_battery.log"


def head() -> str:
    return subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def battery_text(at: str, counts: dict[str, tuple[str, int, int]] | None = None, passed: bool = True) -> str:
    """A closing-chain battery log: the HEAD line, one PARITY block per family, the pytest summary."""
    lines = [f"HEAD {at} tree 0000000000000000000000000000000000000000; status: ", "> python -m pytest ..."]
    for fam, (fired, trades, mism) in (counts or BANKED_BATTERY).items():
        lines += [f"PARITY {fam} n=60 bars=800 elapsed=1.0s",
                  f"  fired                  : {fired} (zero-trade rate 0.0%)",
                  f"  trades (slow)          : {trades}",
                  f"  mismatches             : {mism}"]
    lines += ["4 passed in 300.00s (0:05:00)" if passed else "1 failed, 3 passed in 300.00s", "EXIT=0 WALL=301s"]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def real_artifacts() -> Path:
    """The artifacts dir this test process started with (READ only), resolved before any isolated_arqvio_home."""
    from engine.core.paths import get_artifacts_dir

    return get_artifacts_dir()


@pytest.fixture
def write_battery_log(tmp_path: Path):
    def write(at: str | None = None, counts=None, passed: bool = True) -> Path:
        p = tmp_path / "closing_parity_battery.log"
        p.write_text(battery_text(at or head(), counts, passed), encoding="utf-8")
        return p
    return write
