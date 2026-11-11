"""Gate views preserve real ledger values and legacy gate decisions."""
from types import SimpleNamespace

import numpy as np

from engine.research.search.evaluation_view import GateResult, full_run, kernel_view, oracle_view
from engine.research.search.sampler import random_dna
from engine.research.v1.config import v1_config
from engine.research.v1.gates import judge


def test_oracle_view_preserves_fill_index_and_open_trade():
    result = SimpleNamespace(equity_curve=[100, 101, 99], trades=[
        SimpleNamespace(entry_fill=SimpleNamespace(bar_index=1), exit_bar_index=2, realized_pnl=-1.0),
        SimpleNamespace(entry_fill=SimpleNamespace(bar_index=2), exit_bar_index=None, realized_pnl=0.0),
    ])
    view = oracle_view(result, reason='not expressible')
    assert view.fills.tolist() == [1, 2]
    assert view.exits.tolist() == [2, -1]
    assert view.net.tolist() == [-1.0, 0.0]
    assert view.execution_path == 'slow_fallback' and view.fallback_reason == 'not expressible'


def test_missing_kernel_capture_runs_exact_oracle(monkeypatch):
    from engine.research.search import real_backtest as rb
    dna = random_dna(np.random.default_rng(1))
    calls = []

    def fast(candidate):
        fast.fallback_state['fallbacks'] += 1
        fast.fallback_state['reasons'].append('unsupported fast feature')
    fast.capture = {}
    fast.fallback_state = {'fallbacks': 0, 'reasons': []}
    monkeypatch.setattr(rb, 'dna_to_strategy', lambda candidate: ('cfg', 'strategy'))
    monkeypatch.setattr(rb, 'slow_cell_setup', lambda cell: 'setup')
    def slow(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(trades=(), equity_curve=[100.0, 101.0])
    monkeypatch.setattr(rb, 'run_slow_backtest', slow)
    result = full_run(dna, 'cell', fast)
    assert calls == [('cell', 'setup', dna, 'cfg', 'strategy')]
    assert result.equity.tolist() == [100, 101]
    assert result.execution_path == 'slow_fallback'
    assert result.fallback_reason == 'unsupported fast feature'


def test_gate_decisions_match_legacy_kernel_tuple():
    from engine.research.backtest.kernel import TR_SIZE, TR_FILL_BAR, TR_EXIT_BAR, TR_NET
    dna = random_dna(np.random.default_rng(2))
    rows = np.zeros((4, TR_SIZE))
    rows[:, TR_FILL_BAR] = [1, 30, 61, 80]
    rows[:, TR_EXIT_BAR] = [10, 40, 70, -1]
    rows[:, TR_NET] = [2, -1, 3, 0]
    equity = 100 * np.cumprod(1 + np.tile([0.01, -0.002], 51))
    options = dict(cfg=v1_config('quick', random_n=2, shuffle_n=3), timeframe='H1', split=50,
                   twin_returns=lambda *args: np.zeros(101), evaluate_bred=lambda d: None,
                   neighbours=lambda d: [])
    old = judge(dna, full_run=lambda d: (rows, equity), **options)
    raw = kernel_view(rows, equity)
    slow = GateResult(raw.fills, raw.exits, raw.net, raw.equity, 'slow_fallback', 'test fallback')
    new = judge(dna, full_run=lambda d: slow, **options)
    assert new.pop('execution_path') == 'slow_fallback'
    assert new.pop('fallback_reason') == 'test fallback'
    old.pop('execution_path'); old.pop('fallback_reason')
    assert new == old
