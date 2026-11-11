"""Genetic search over DNA space on pinned cells (S4 stage 2).

Seeded, budgeted and deterministic: population -> evaluate (kernel first)
-> gated fitness -> selection / crossover / mutation under dedupe ->
generations -> cross-cell robustness pass -> manifest v2. The GA
orchestrates the existing machinery -- sampler, backtest factories, sham
controls, DSR, degeneracy census -- and never touches the backtest physics
or the validation layer.
"""
