"""Cost impact: how much of the strategy's gross edge was eaten by costs.

WHAT: ``(sum(gross_price_pnl) - sum(realized_pnl)) / sum(gross_price_pnl)``,
      restricted to CLOSED trades. Both gross and net are in account
      currency and are read from the trade ledger (populated by the
      engine) - this module never re-derives P&L.

WHY: A 2.0 Sharpe on gross P&L that becomes 0.3 Sharpe on net P&L is
     a cost-eaten strategy, not an edge. Cost impact is the single
     number that tells you "of the edge your signals generated, what
     fraction did the broker keep?". Strategies above ~0.5 should be
     rejected at the S4 gate.

HOW: Pure NumPy. ``None`` when there are no closed trades. Returns
     ``None`` (NOT 0.0 or 1.0) when gross P&L is exactly zero across
     the closed ledger - the ratio is undefined there too.

TRADE-OFF: We compute cost impact on the CLOSED ledger only, not on
           the mark-to-market open position. An open trade's gross
           and net P&L are equal in S3 (no exit costs charged yet),
           so including it would dilute the metric. Closed trades are
           the honest signal.
"""
from __future__ import annotations

import numpy as np


def cost_impact(gross_pnls: np.ndarray, net_pnls: np.ndarray) -> float | None:
    """Return (gross - net) / gross; ``None`` when undefined.

    Args:
        gross_pnls: per-trade gross price P&L (no costs, no swap).
        net_pnls: per-trade realised P&L (after entry costs + swap).

    Both arrays must be the same length. Returns ``None`` when:
        - the arrays are empty, or
        - gross sum is exactly zero (ratio undefined).
    """
    g = np.asarray(gross_pnls, dtype=np.float64).ravel()
    n = np.asarray(net_pnls, dtype=np.float64).ravel()
    if g.size == 0 or n.size == 0 or g.size != n.size:
        return None
    # Drop non-finite pairs in lockstep.
    m = np.isfinite(g) & np.isfinite(n)
    g = g[m]
    n = n[m]
    if g.size == 0:
        return None
    g_sum = float(np.sum(g))
    n_sum = float(np.sum(n))
    if g_sum == 0.0:
        return None
    return float((g_sum - n_sum) / g_sum)
