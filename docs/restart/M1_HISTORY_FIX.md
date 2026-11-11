# GBPUSDm M1 history fix

2026-11-11. Tester: `C:\mt5\tester\terminal64.exe`, read-only.

## Root cause

`copy_rates_range` accepted smaller GBPUSDm M1 windows (7d/30d) but a 90d request returned MT5 `Invalid params`. The old fallback made one oversized `copy_rates_from` request and did not page positional history, so campaigns could report zero/insufficient M1 bars despite available recent history.

## Fix

`engine/research/cells/add_market.py::pull_bars` now:

- Reads terminal `maxbars`.
- Uses `copy_rates_from_pos` in 10,000-bar chunks.
- Stops at the terminal cap/short page.
- Filters to `[start, end)` UTC timestamps.
- Refuses if the filtered count stays below `MIN_BARS`; never fabricates bars.
- Keeps a compatibility path for narrow test adapters lacking `copy_rates_from_pos`.

## Evidence

- Tester GBPUSDm M1 90-day request: **91,767 bars**, first `2026-08-14T13:58:00+00:00`, last `2026-11-02T09:31:00+00:00` (terminal history boundary; output ordering is terminal-provided).
- Tester GBPUSDm M1 7-day request: **7,184 bars**, `out_of_window=0`, first `2026-11-04T08:32:00+00:00`, last `2026-11-11T08:31:00+00:00`.
- Regression suite: **35 passed**, 4.39s (`test_server_clock.py`, horizon profiles, launch routes).
- No order, EA, chart, AutoTrading or live-terminal operation.

This fixes retrieval, not scalping validation. Bid/ask fill/latency/queue parity remains unimplemented.
