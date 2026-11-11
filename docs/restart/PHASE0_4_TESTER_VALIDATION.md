# Phases 0–4 tester validation — read-only

2026-11-11. Terminal explicitly approved: `C:\mt5\tester\terminal64.exe`. No owner/live terminal touched.

## Terminal readiness

- Discovered tester path: `C:\mt5\tester\terminal64.exe`.
- Before explicit launch: CLOSED.
- After explicit launch: running, account state `OK`, connected `True`, AutoTrading reported `True` by terminal. No setting was changed; no order/chart/EA action was performed.
- Server/account identity is kept in the local private report only; not public source/docs.

## Read-only market data probe

Local report: `C:\Users\n69474736\Documents\ARQVIO\tester-validation-home\phase4_tester_read_report.json`.

Probe actions: initialize read, symbol info, `copy_rates_range`, `copy_ticks_range` for seven days, shutdown. No trading APIs.

| Symbol | Offered | 7d ticks | M1 bars | M5 | H1 | H4 | D1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| XAUUSDm | yes | 1,722,795 | 6,889 | 1,380 | 115 | 31 | 6 |
| EURUSDm | yes | 253,972 | 482 | 1,438 | 120 | 31 | 6 |
| GBPUSDm | yes | 310,298 | 0 | 1,438 | 120 | 31 | 6 |
| USDJPYm | yes | 370,073 | 7,175 | 1,436 | 120 | 31 | 6 |

The seven-day result proves current terminal retrieval only. It does not prove multi-year history, stable broker clock, tick completeness, queue fills or scalping parity. GBPUSDm M1 returned zero in this interval; no fallback/fabrication permitted.

## Validation checks

- Purged CV property check: 4 chronological splits, 50-row purge and 20-row embargo each, disjoint/leak-free assertion passed.
- Tester-configured validation + horizon suite: **144 passed**, 3 documented numerical warnings, 198.54s. Warnings are existing PBO polynomial conditioning and PSR near-constant-return precision warnings.
- No full campaign, lockbox promotion or strategy profitability run was performed.

## Capability interpretation

- Current code can research/intraday/swing with `MARKET2_OHLC` under its existing bar/cost semantics.
- Current code must not claim tick-scalping: bid/ask retrieval exists in the probe, but execution model, latency, spread/slippage stress, fill/queue behavior, M1 parity and stable coverage are not accepted.
- Pending limit/stop, partial-close, lockbox and nested selection remain incomplete.

No release claim, profitability claim, deployment or AutoTrading action follows from this report.
