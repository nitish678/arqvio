"""Capability declarations are not executed proof; legacy hashes remain unchanged."""
import dataclasses

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.research.ga.population import checked
from engine.research.genes import registry
from engine.research.search.sampler import random_dna


def test_capability_api_is_separate_and_never_invents_proofs(isolated_arqvio_home):
    client = TestClient(create_app())
    legacy = client.get('/api/v1/gene-pool/registry').json()
    response = client.get('/api/v1/gene-pool/capabilities')
    assert response.status_code == 200
    body = response.json()
    assert body['registry_hash'] == legacy['hash'] == registry.registry_hash()
    rows = {r['name']: r for r in body['genes']}
    assert len(rows) == len(registry.GENES)
    for row in rows.values():
        for target in row['support'].values():
            assert target['evidence_status'] == 'UNVERIFIED'
            assert target['proof_runs'] == []
    assert rows['bias.sma']['support']['oracle']['declared']
    assert not rows['bias.sma']['support']['tester_ea']['declared']
    assert rows['bias.sma']['parameterization'] == 'fixed_switch_rule'
    assert rows['bias.sma']['breeding_fields'] == ['bias_switches']
    assert rows['signal.macd']['parameterization'] == 'family_parameters'
    assert body['declared_grammar']['configurable_quorum'] is False
    assert not rows['management.partial_tp']['breedable_declared']
    assert rows['risk.daily_dd']['support']['live_ea']['declared']
    assert not rows['risk.daily_dd']['support']['oracle']['declared']
    assert 'support' not in legacy['genes'][0]


@pytest.mark.parametrize('changes,reason', [
    ({'risk_switches': ('daily_dd',)}, 'risk.daily_dd'),
    ({'mgmt_switches': ('partial_tp',)}, 'management.partial_tp'),
    ({'bias_switches': ('ema',)}, 'bias.ema'),
    ({'mgmt': 'BREAKEVEN'}, 'legacy management'),
])
def test_admission_refuses_known_but_unsupported_genomes(changes, reason):
    dna = dataclasses.replace(random_dna(np.random.default_rng(7)), **changes)
    with pytest.raises(ValueError, match=reason):
        checked(dna)


def test_valid_admission_keeps_effective_hash_and_dna_bytes():
    from engine.research.search.effective_hash import effective_gene_hash
    for dna in (random_dna(np.random.default_rng(seed)) for seed in range(20)):
        before = dna.to_dict(), dna.dna_hash, effective_gene_hash(dna)
        assert checked(dna) == before[2]
        assert (dna.to_dict(), dna.dna_hash, effective_gene_hash(dna)) == before


def test_mutating_restricted_signals_never_produces_empty_entry():
    from engine.research.ga.operators import mutate
    from engine.research.search import gene_pool
    from engine.research.backtest.signal_switches import READY_SIGNALS

    pool = gene_pool.merge(gene_pool.default_gene_pool(), {
        'gates': {'signal_switches': True},
        'off': [f'signal.{name}' for name in READY_SIGNALS if name != 'cci'],
    })
    dna = dataclasses.replace(random_dna(np.random.default_rng(1)), signal='SIGNALS', signal2='',
                              signal_switches=('cci',))
    # Some mutations keep SIGNALS; its active switch must remain non-empty and enabled.
    for seed in range(30):
        rng = np.random.default_rng(seed)
        # p below 1 permits mutation without always switching to a legacy family.
        child, _ = mutate(rng, dna, pool, p=0.5, scale=0.5)
        checked(child)
        if child.signal == 'SIGNALS':
            assert child.signal_switches == ('cci',)


def test_legacy_registry_hash_is_pinned():
    assert registry.registry_hash() == 'c482d8bc0d90f900'
