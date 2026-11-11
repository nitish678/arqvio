# Phase 3 — strategy horizons and execution capability

2026-11-11. Status: partial, first slice only. No live MT5, deployment, kernel change, risk change or commit yet.

## Implemented

- Added explicit launch horizon profiles: `research`, `intraday`, `swing`, `scalping`.
- Existing OHLC/next-bar-open execution is labelled `MARKET2_OHLC`; intraday accepts M5/M15/M30/H1; swing accepts H1/H4/D1.
- `scalping` fails before task creation with a truthful message requiring timestamped bid/ask ticks and execution parity. M1/M5 or a timeframe dropdown alone does not certify scalping.
- Added optional frontend request type field without changing default legacy behavior.

## Verification

- Targeted launch/horizon/API regressions: **39 passed**, 4.95s, no failure. Real MT5 calls blocked through the offline harness.
- Targeted Phase 3 plus Phase 1/2 regressions: **47 passed**, 1 dependency deprecation warning, 6.24s.
- Frontend TypeScript check passed; desktop Vitest: **29 passed / 6 files**, 10.46s. No browser screenshot claim.
- `py_compile` passed for `routes_app_launch.py`; `git diff --check` passed (CRLF notices only).
- Broader backtest run was not accepted as a Phase 3 result: private pinned cell fixtures absent and resource-heavy workers were stopped; no production failure attributed to horizon code. No canary was needed for this non-kernel slice.

## M1 fix validation

- Root cause: the previous fallback asked for one oversized positional/range request after a range pull returned too few bars. MT5 returned `Invalid params` for a 90-day GBPUSDm M1 range request while smaller windows worked.
- Fix: page `copy_rates_from_pos` in 10,000-bar chunks up to terminal `maxbars`, filter bars to the requested half-open UTC window, and refuse when the remaining history is insufficient. Legacy fake adapters retain compatibility.
- Tester read validation: GBPUSDm M1 returned **91,767 bars** for a 90-day request; a seven-day request returned **7,184 bars**, `out_of_window=0`. No order/EA/AutoTrading call.
- Regression suite after fix: **35 passed** (M1 fallback, launch horizon and launch routes), 4.39s.

## Not implemented/claimed

- No pending limit/stop-order fill model.
- No partial-close split-trade model.
- No bid/ask tick ingestion, latency/slippage model or scalping parity.
- No new strategy family added. Existing RSI/Stoch/MACD/Donchian/SIGNALS families remain subject to their existing capability proofs.
- Gold/major-FX support still requires broker data/spec/cost acceptance per cell.
