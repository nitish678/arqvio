"""The diversity census (DR §4): ENB known answers, axis labels from the DNA config, per-axis shares."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from engine.research.portfolio.diversity import AXES, axis_labels, census, duplicate_clusters, enb
from engine.research.search.sampler import random_dna


def _member(mid: str, signal: str, symbol: str, start: int = 0, end: int = 24):
    dna = dataclasses.replace(random_dna(np.random.default_rng(1)), signal=signal, window_start_hour_utc=start,
                              window_end_hour_utc=end)
    return SimpleNamespace(mid=mid, dna=dna, symbol=symbol)


@pytest.mark.parametrize("k", [2, 4, 7])
def test_k_independent_members_with_equal_weights_are_k_bets(k: int) -> None:
    panel = np.random.default_rng(k).normal(size=(4000, k))
    members = [_member(f"m{j}", "MA_CROSS", "USDCADm") for j in range(k)]
    out = census(panel, members, {"equal": [1 / k] * k})
    assert out["enb"]["equal"] == pytest.approx(k, rel=1e-12)
    assert out["duplicate_clusters"] == [[m.mid] for m in members]


def test_a_single_member_is_one_bet() -> None:
    assert enb([1.0], [[0]]) == 1.0


def test_identical_members_are_one_bet() -> None:
    x = np.random.default_rng(5).normal(size=(500, 1))
    panel = np.hstack([x, x, x])
    members = [_member(f"m{j}", "RSI_REVERSION", "XAUUSDm") for j in range(3)]
    out = census(panel, members, {"equal": [1 / 3] * 3})
    assert out["duplicate_clusters"] == [["m0", "m1", "m2"]] and out["enb"]["equal"] == 1.0
    assert out["abs_corr_offdiag"]["min"] == pytest.approx(1.0, abs=1e-12)


def test_a_negatively_identical_member_is_the_same_bet() -> None:
    corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert duplicate_clusters(corr) == [[0, 1]] and enb([0.5, 0.5], [[0, 1]]) == 1.0


def test_axis_labels_come_from_the_dna_config() -> None:
    m = _member("x", "DONCHIAN_BREAKOUT", "USDCADm", 7, 19)
    assert list(AXES) == ["family", "symbol", "session_window"]
    assert axis_labels(m) == {"family": "DONCHIAN_BREAKOUT", "symbol": "USDCADm", "session_window": "07-19"}


def test_per_axis_shares() -> None:
    panel = np.random.default_rng(2).normal(size=(300, 3))
    members = [_member("a", "MA_CROSS", "USDCADm"), _member("b", "MA_CROSS", "XAUUSDm"),
               _member("c", "MACD_CROSS", "XAUUSDm")]
    out = census(panel, members, {"w": [0.5, 0.25, 0.25]})
    fam = out["axis_shares"]["family"]
    assert fam["counts"] == {"MACD_CROSS": 1, "MA_CROSS": 2} and fam["weights"]["w"] == {"MACD_CROSS": 0.25,
                                                                                          "MA_CROSS": 0.75}
    assert out["axis_shares"]["symbol"]["weights"]["w"] == {"USDCADm": 0.5, "XAUUSDm": 0.5}
    assert census(panel, members, {"w": [0.5, 0.25, 0.25]}) == out  # deterministic
