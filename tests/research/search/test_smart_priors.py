"""Smart priors: half uniform, half the vault's survivors; a family with no survivor keeps the control share."""
from __future__ import annotations

import pytest

from engine.research.search.gene_pool import default_gene_pool
from engine.research.search.smart_priors import suggest


def test_weights_mix_control_and_evidence():
    pool = default_gene_pool()
    enabled = [f for f, s in pool.families.items() if s["enabled"]]
    rows = [{"family": enabled[0], "tier": "CANDIDATE", "grade": "B"}] * 3 + \
           [{"family": enabled[1], "tier": "CANDIDATE", "grade": "F"}]
    s = suggest(pool, rows)
    k = len(enabled)
    assert s["weights"][enabled[0]] == pytest.approx(0.5 / k + 0.5)
    assert s["weights"][enabled[1]] == pytest.approx(0.5 / k)          # grade F is not a survivor
    assert sum(s["weights"].values()) == pytest.approx(1.0)
    assert s["basis"]["survivors"] == 3


def test_empty_vault_is_uniform():
    pool = default_gene_pool()
    w = suggest(pool, [])["weights"]
    assert len(set(round(v, 9) for v in w.values())) == 1
