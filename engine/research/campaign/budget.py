"""Campaign budget calculator.

SEARCH_TRIALS = n_symbols × n_timeframes × population_per_cell × n_generations
               × eval_multiplier.
VALIDATION_RUN is excluded from the budget. A validation run is a
robustness re-evaluation of a single already-selected strategy; it
never inflates the multiple-testing correction.

The legacy SR=1.0 comparator and the default SR=2.0 target are both
computed and surfaced: at 3.8 years of history the two figures differ
by orders of magnitude and the user must see both.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from engine.validation.minbtl import (
    DEFAULT_TARGET_IS_SHARPE,
    LEGACY_TARGET_IS_SHARPE,
    MinBTLBudgetTable,
    minbtl_budget_table,
    n_trials_for_budget,
)


@dataclass(frozen=True)
class CampaignScope:
    """Selection the user has ticked in the campaign builder."""

    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    population_per_cell: int
    n_generations: int = 1
    eval_multiplier: int = 1
    target_is_annual_sharpe: float = DEFAULT_TARGET_IS_SHARPE

    def __post_init__(self) -> None:
        if self.population_per_cell < 0:
            raise ValueError("population_per_cell must be >= 0")
        if self.n_generations < 0:
            raise ValueError("n_generations must be >= 0")
        if self.eval_multiplier < 0:
            raise ValueError("eval_multiplier must be >= 0")

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_timeframes(self) -> int:
        return len(self.timeframes)

    @property
    def search_trials(self) -> int:
        if self.n_symbols == 0 or self.n_timeframes == 0:
            return 0
        return (
            self.n_symbols
            * self.n_timeframes
            * self.population_per_cell
            * self.n_generations
            * self.eval_multiplier
        )

    @property
    def is_empty(self) -> bool:
        return self.n_symbols == 0 or self.n_timeframes == 0


def budget_table_for_scope(
    scope: CampaignScope, history_years: float
) -> MinBTLBudgetTable:
    """Build the comparison table the campaign builder renders."""
    return minbtl_budget_table(history_years)


def expected_history_years(per_cell_years: Iterable[float]) -> float:
    """The campaign's history_years for MinBTL is the MIN over the scope.

    MinBTL must be satisfied per cell. If any cell has only 1 year of
    history, the campaign's MinBTL is computed against 1 year.
    """
    values = [float(v) for v in per_cell_years if v is not None]
    return min(values) if values else 0.0


def required_min_years(
    scope: CampaignScope,
    target_sharpe: float,
    *,
    history_years: float,
) -> float:
    """Required MinBTL for the configured scope.

    The MinBTL formula is symmetric in n_trials so we invert it.
    """
    if scope.is_empty:
        return 0.0
    n = scope.search_trials
    if n < 1:
        n = 1
    # For zero skew/kurt the formula simplifies to:
    #     min_years ≈ 1 + (n - 1) / SR^2
    return 1.0 + (n - 1) / max(target_sharpe * target_sharpe, 1e-12)


def legacy_comparator_at_sharpe_1(
    scope: CampaignScope, *, history_years: float
) -> int:
    """The legacy SR=1.0 budget number. Surfaced alongside SR=2.0."""
    if scope.is_empty:
        return 0
    return n_trials_for_budget(LEGACY_TARGET_IS_SHARPE, history_years)


def default_comparator_at_sharpe_2(
    scope: CampaignScope, *, history_years: float
) -> int:
    """The default SR=2.0 budget number. The campaign budget uses this."""
    if scope.is_empty:
        return 0
    return n_trials_for_budget(DEFAULT_TARGET_IS_SHARPE, history_years)


__all__ = [
    "CampaignScope",
    "budget_table_for_scope",
    "default_comparator_at_sharpe_2",
    "expected_history_years",
    "legacy_comparator_at_sharpe_1",
    "required_min_years",
]
