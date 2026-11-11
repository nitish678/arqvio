# Phase 7 — portfolio/automation/live health foundation

2026-11-11. Status: partial.

## Implemented

- Read-only `exposure_breakdown()` now counts planned vault-member exposure by symbol and both FX currency legs, with explicit source text. It does not pretend to be live MT5 exposure.
- Read-only `replacement_eligibility()` reports WEAKENING/TOO FEW/BELOW_PARITY reasons and `auto_executed: false`; no bot swap or lot/risk mutation.
- Automation run state now carries a run ID and per-step idempotency key in `run.json`; restart evidence can identify the exact step intent.

## Verification

- Automation + new portfolio tests: **11 passed**, 1 dependency warning, 6.03s.
- No live MT5/deploy/order/AutoTrading action.

## Remaining

- Crash/restart execution replay proof with persisted step locks.
- Live exposure from actual MT5 deal/position reads, separate from planned vault exposure.
- Full incubation/reassessment/refresh and replacement candidate ranking.
- Portfolio combined simulation and concurrency/margin evidence.
