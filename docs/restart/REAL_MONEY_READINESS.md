# Real-money readiness: NOT APPROVED

2026-11-11. Owner clarified ARQVIO is intended for their own capital. Short unit-test passes, source publication, phase labels and successful data downloads do not establish safe execution or a profitable edge.

## Safety-review finding fixed: historical-bar integrity

The previously published M1 positional fallback concatenated newest pages before older pages. Its 90-day smoke result printed a later first timestamp than last timestamp. It also lacked duplicate conflict detection, filtered only fallback data, removed UTC timezone information from requests and contained a legacy branch that could inflate the accepted count with out-of-window rows.

Corrections in `cells/add_market.py`:
- Require timezone-aware UTC boundaries.
- Filter all responses to the exact half-open requested interval before checking minimum count.
- Stable-sort whole structured records by timestamp; preserve aligned prices/volume/spread.
- Deduplicate identical records; refuse conflicting duplicate timestamps instead of selecting a price.
- Never inflate usable-row counts with terminal capacity or out-of-window rows.

Evidence:
- Test-first safety regressions: six failing cases covering page ordering, identical duplicates, conflict refusal, legacy out-of-window count, direct-range filtering and UTC requests.
- Focused safety/clock/launch suites: **55 passed** with 35 official MT5 API functions blocked.
- Broader offline data/cells/launch suites: **112 passed, 33 skipped**, exit 0. Skips require private pinned snapshots, account config or ingested specs and are not parity passes.
- Explicit tester read: GBPUSDm M1 **91,767 rows**, **2026-08-13 00:00 UTC → 2026-11-10 23:59 UTC**, strictly increasing unique timestamps within the requested interval. Raw saved NPZ reloaded and compared field-for-field. SHA-256 `3bf8db7d3e81034293d36062062fe6ca2ac8f342f1327195c2b540f9959ca126`.
- Raw broker data stays private in isolated `ARQVIO_HOME`; script `scripts/verify_history_safety.py` reproduces the bounded read and file verification.
- Pinned planted-edge canary: **PASS 12.5s, zero crashes**, official MT5 APIs blocked. This is a diagnostic, not proof of market edge.

A prior 91,767-row count alone must not be used as validated backtest data. Re-pull/revalidate datasets created with the affected fallback. Old IDs/hashes are not rewritten or auto-certified.

## Critical work still requiring review/proof

- Queue locking, concurrent edits, crash-after-launch replay and waiting-state semantics; a stored idempotency string alone does not provide exactly-once execution.
- Portfolio exposure: use actual contract/currency/side/notional information, distinguish missing data from zero, never call insufficient live history replacement evidence.
- Tick path: strict finite/integer inputs, causal timing, quote-update handling, realistic costs, explicit time latency, order/SL/TP management and Python-versus-authorized-tester parity. Current helper is experimental only.
- Lockbox: independent trusted seal, one-time access ledger and actual pipeline integration; self-hash metadata isn't immutability/security or an untouched-data guarantee.
- Trial provenance: individual and bulk writers must enforce compatible contracts without losing failed/cross-cell observations; legacy readers/selection accounting require end-to-end review.
- ML: strict split/index/hyperparameter checks, train-only feature availability and nested selection, calibration/controls, unseen net-of-cost benefit, immutable model identity; current standalone baseline isn't live-ready.
- Current UI horizon selectors don't themselves enforce holding-time/session behavior. Do not interpret timeframe presets as demonstrated intraday/swing semantics.
- Existing safety/guardrail defaults and generated EAs need actual supported-scope tester parity; no profit/safety claims from skipped tests or synthetic fixtures.

## Operating boundary

No live orders, risk/lot changes, AutoTrading changes, live EA/chart attach or owner terminal restart under general development approval. Authorized tester reads only. Inspect existing changes before expanding features; every fix needs a reproducing regression and recorded evidence. Release/real-money promotion remains blocked until separately verified and authorized.
