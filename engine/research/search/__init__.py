"""engine/research/search package.

The search runner is the part of the system that produces an answer. It
generates random DNA over the gene space, evaluates each candidate
against the backtester, and pipes results into the S1 validation
pipeline. A separate vs-random control campaign matches the real
campaign on every axis except signal meaning.
"""
from __future__ import annotations

__all__: list[str] = []
