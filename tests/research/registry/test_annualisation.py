"""The annualisation law, and the one-scale tripwire over every annualisation map a registered cell meets."""
from __future__ import annotations

import ast

import pytest

from engine.core.paths import PROJECT_ROOT
from engine.research.registry.annualisation import (
    MINUTES_PER_DAY,
    TF_MINUTES,
    TRADING_DAYS_PER_YEAR,
    UnknownTimeframe,
    bars_per_year,
)
from engine.research.registry.cells import list_cells

LAW = {"M1": 362880, "M5": 72576, "M15": 24192, "M30": 12096, "H1": 6048, "H2": 3024, "H4": 1512, "D1": 252}

# The Sharpe-side files (engine/research/search): since the 2026-10-22 surgery (ARQVIO M5-D1 cells) they annualise
# through registry.annualisation.bars_per_year and hold no literal timeframe map. A map that comes back changes the
# count and fails here, so it cannot slip past the tripwire.
SHARPE_SIDE_MAPS = {"engine/research/search/real_backtest.py": 0, "engine/research/search/canary.py": 0}


def test_the_law_values() -> None:
    assert {tf: bars_per_year(tf) for tf in TF_MINUTES} == LAW


def test_the_formula_is_exact_for_every_row() -> None:
    for tf, minutes in TF_MINUTES.items():
        assert (TRADING_DAYS_PER_YEAR * MINUTES_PER_DAY) % minutes == 0, tf
        assert bars_per_year(tf) * minutes == 252 * 24 * 60


def test_an_unknown_timeframe_is_refused_never_defaulted() -> None:
    with pytest.raises(UnknownTimeframe):
        bars_per_year("W1")


def test_the_law_agrees_with_the_older_tables() -> None:
    from engine.data.timeframes import PERIODS_PER_YEAR as data_ppy
    from engine.validation.stats import PERIODS_PER_YEAR as stats_ppy

    for table in (stats_ppy, data_ppy):
        shared = sorted(set(table) & set(LAW))
        assert shared
        assert {tf: table[tf] for tf in shared} == {tf: LAW[tf] for tf in shared}


def _sharpe_side_maps() -> list[tuple[str, dict[str, int]]]:
    """Every literal {"D1": ..., ...} timeframe map in the Sharpe-side files, read from source by ast."""
    found = []
    for rel in SHARPE_SIDE_MAPS:
        tree = ast.parse((PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and node.keys and all(
                    isinstance(k, ast.Constant) and k.value in TF_MINUTES for k in node.keys):
                found.append((rel, ast.literal_eval(node)))
    return found


def test_one_scale_for_every_registered_cell() -> None:
    """cell_dsr's DSR table, the Sharpe-side maps, the registry and the law agree for every registered timeframe.

    The Sharpe-side maps default to 252 for a timeframe they lack, so registering a cell on such a timeframe fails
    here until those maps are aligned in a declared surgery (DR-2026-10-07-u1 §4, U1-Q2).
    """
    from engine.validation.cell_dsr import DEFAULT_PERIODS_PER_YEAR, PERIODS_PER_YEAR

    maps = _sharpe_side_maps()
    assert {rel: sum(r == rel for r, _ in maps) for rel in SHARPE_SIDE_MAPS} == SHARPE_SIDE_MAPS
    for e in list_cells():
        tf = e["timeframe"]
        assert e["annualisation_ppy"] == LAW[tf]
        assert PERIODS_PER_YEAR.get(tf, DEFAULT_PERIODS_PER_YEAR) == LAW[tf], ("cell_dsr", tf)
        for rel, table in maps:
            assert table.get(tf, 252) == LAW[tf], (rel, tf)
