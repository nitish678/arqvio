"""Lookahead guard. The runner rejects any DNA with ``uses_future_bar=True``
before it ever reaches the backtester. This is the second half of I4.
"""
from __future__ import annotations

from engine.research.search.dna import is_lookahead_dna as _impl

# Re-export under the import path the I4 test uses.
is_lookahead_dna = _impl
__all__ = ["is_lookahead_dna"]
