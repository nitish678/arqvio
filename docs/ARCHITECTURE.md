# Architecture and current boundaries

## Current code

- `engine/api`: local FastAPI entry points.
- `engine/research/search`, `ga`, `v1`: strategy descriptions and discovery policies. Legacy V1/V2 workflows remain; a unified contract is being introduced incrementally.
- `engine/research/backtest`: event-driven reference engine and Numba fast path. A fast-path refusal must not become an approximate result.
- `engine/research/genes`: legacy registry plus additive capability declarations/evidence schema. A valid evidence record is not independent verification that it ran.
- `engine/validation`: statistics and trial-accounting primitives; not a promise of investable results.
- `engine/research/data`, `engine/mt5`: terminal/data interfaces.
- `engine/research/vault`, `portfolio`, `automation`, `deploy`, `mql5`: storage, analysis, workflow and optional execution components.
- `apps/desktop/src`: React/TypeScript UI; `src-tauri`: native shell.

## Intended shared pipeline

Data readiness → Discovery → Validation → Vault → Portfolio → MT5 parity → Explicitly authorized deployment → Monitoring.

One product engine does not mean one blocking process. Research must not block monitoring/UI. Reference and fast execution paths both remain. Dataset/cost/code/config identity and uncertainty should survive through the pipeline.

## Safety and evidence

- Hash/version old artifacts; never auto-upgrade a legacy strategy's validation status.
- Declare unsupported grammar/execution explicitly. Legacy READY does not establish current tester parity.
- No future features or training-only information in validation decisions.
- Missing data is not zero performance. A skipped check is not a pass.
- Broker fills/costs and currency/spec changes need dedicated evidence. OHLC backtests cannot demonstrate tick-scalping or historical queue-position fidelity.
- Model training/inference and automatic proof-run collection are not implemented acceptance claims in this source snapshot.
