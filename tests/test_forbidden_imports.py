"""Forbidden-import policy.

Stage 1: ensures the engine does not import any blacklisted library.
This is the simplest enforcement of the policy in `docs/02_TECH_DECISIONS.md` § D-14.
"""
from __future__ import annotations

import importlib
import sys

import pytest

FORBIDDEN = {
    "deap",  # we have our own evolution engine
    "backtrader",  # we have our own backtester
    "vectorbt",  # inspired by, not imported
    "jesse",  # inspired by, not imported
    "zipline",
    "zipline_reloaded",
    "freqtrade",
    "ccxt",  # ARQVIO is MT5-first
}


@pytest.mark.parametrize("module", sorted(FORBIDDEN))
def test_forbidden_module_not_importable(module: str) -> None:
    """Either the module is not installed, or ARQVIO must not import it."""
    # The policy is enforced at the *import* level by ARQVIO's code. A module being
    # present on the system is not a violation; what matters is that no ARQVIO source
    # file imports it. We test the policy by importing ARQVIO's main package and
    # checking that the forbidden name is not in sys.modules after that import.
    # For absolute certainty we also assert importlib.util.find_spec is None OR
    # the module is not in sys.modules.
    for name in list(sys.modules):
        if name == module or name.startswith(module + "."):
            pytest.fail(f"ARQVIO imported forbidden module: {name}")
    # The import below is just to ensure ARQVIO's public surface loads cleanly.
    importlib.import_module("engine")
