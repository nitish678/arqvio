"""Single switches turned off inside an open gate (owner 2026-11-01): never drawn, never bred back, hash unchanged
for pools that turn nothing off, and the random stream is the same."""
from __future__ import annotations

import numpy as np
import pytest

from engine.research.ga.operators import mutate
from engine.research.search import gene_pool as gp
from engine.research.search.sampler import random_dna

GATES = {g: True for g in ("bias_switches", "filter_switches", "mgmt_switches", "signal_switches")}


def pool(off: list[str] | None = None) -> gp.GenePool:
    data = {"families": {f: {"enabled": True, "weight": 1.0} for f in gp.SIGNAL_VALUES}, "gates": GATES}
    return gp.canonicalise({**data, **({"off": off} if off is not None else {})})


def test_no_off_list_keeps_the_hash() -> None:
    assert pool().hash == pool([]).hash and "off" not in pool([]).canonical()
    assert pool(["bias.sma"]).hash != pool().hash


def test_off_switches_are_never_drawn_and_the_stream_is_unchanged() -> None:
    off = ["bias.sma", "filter.rsi", "management.breakeven", "signal.cci"]
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(400):
        d = random_dna(a, gene_pool=pool(off))
        random_dna(b, gene_pool=pool())
        assert "sma" not in d.bias_switches and "rsi" not in d.filter_switches
        assert "breakeven" not in d.mgmt_switches and "cci" not in d.signal_switches
    assert a.random() == b.random()  # the same number of draws was consumed


def test_mutation_never_breeds_an_off_switch_back() -> None:
    p = pool(["bias.sma"])
    rng = np.random.default_rng(1)
    d = random_dna(rng, gene_pool=p)
    for _ in range(300):
        d, _ = mutate(rng, d, p, p=1.0, scale=1.0)
        assert "sma" not in d.bias_switches


def test_unknown_or_all_signals_off_is_refused() -> None:
    with pytest.raises(gp.GenePoolError, match="not a READY switch"):
        pool(["bias.nope"])
    from engine.research.backtest.signal_switches import READY_SIGNALS
    with pytest.raises(gp.GenePoolError, match="at least one signal"):
        pool([f"signal.{n}" for n in READY_SIGNALS])


def test_patch_replaces_the_off_list() -> None:
    p = gp.merge(pool(["bias.sma"]), {"off": ["filter.doji"]})
    assert p.off == ("filter.doji",)
