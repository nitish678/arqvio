# Tick scalping experiment — Phase 3

2026-11-11. Research-only execution model; no orders, EA attach, AutoTrading or deployment.

## Implemented

- `engine/research/data/mt5_source.py::fetch_ticks` reads UTC ticks with `COPY_TICKS_ALL`, refuses empty data, and shuts down the read session.
- `engine/research/backtest/ticks.py` validates ordered positive bid/ask ticks and simulates market fills:
  - long fills at ask; short fills at bid;
  - explicit tick-index latency;
  - explicit non-negative slippage in points;
  - requested/fill/spread/slippage values returned in `TickFill`.
- No queue-position, market-impact, partial-fill, latency-distribution or broker execution guarantee is implied. Model is not connected to strategy search or deployment yet.

## Tester evidence

Tester: `C:\mt5\tester\terminal64.exe`, read-only. GBPUSDm, one-hour tick range:

- ticks: **2,489**
- median spread: **0.00010000000000021103** price units
- sample long fill: tick index 1, requested/fill `1.3319`, zero configured slippage

This is a data/model smoke check, not a strategy result, profitability result or parity certificate.

## Tests

- Tick unit tests: **2 passed**.
- Existing M1 fallback/horizon/launch suite after the previous fix: **35 passed**.
- Full tester-backed strategy tester parity was not run. Tick model remains experimental and scalping remains non-deployable.
