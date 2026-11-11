# Phase 3 — strategy horizons and execution capability

2026-11-11. Status: partial, horizon-contract slice plus desktop selector. No live MT5, deployment, kernel change, risk change or release build.

## Implemented

- Added explicit launch horizon profiles: `research`, `intraday`, `swing`, `scalping`.
- Existing OHLC/next-bar-open execution is labelled `MARKET2_OHLC`; intraday accepts M5/M15/M30/H1; swing accepts H1/H4/D1.
- `scalping` fails before task creation with a truthful message requiring timestamped bid/ask ticks and execution parity. M1/M5 or a timeframe dropdown alone does not certify scalping.
- Added optional frontend request type field without changing default legacy behavior.
- Campaign UI now lets users choose Research, Intraday, Swing or Scalping. Scalping remains selectable only to show the explicit unavailable reason; it cannot launch.

## Verification

- Targeted launch/horizon/API regressions: **39 passed**, 4.95s, no failure. Real MT5 calls blocked through the offline harness.
- Targeted Phase 3 plus Phase 1/2 regressions: **47 passed**, 1 dependency deprecation warning, 6.24s.
- Frontend TypeScript check passed; desktop Vitest: **29 passed / 6 files**, 10.46s. No browser screenshot claim.
- `py_compile` passed for `routes_app_launch.py`; `git diff --check` passed (CRLF notices only).
- Broader backtest run was not accepted as a Phase 3 result: private pinned cell fixtures absent and resource-heavy workers were stopped; no production failure attributed to horizon code. No canary was needed for this non-kernel slice.

## Not implemented/claimed

- No pending limit/stop-order fill model.
- No partial-close split-trade model.
- No bid/ask tick ingestion, latency/slippage model or scalping parity.
- No new strategy family added. Existing RSI/Stoch/MACD/Donchian/SIGNALS families remain subject to their existing capability proofs.
- Gold/major-FX support still requires broker data/spec/cost acceptance per cell.
