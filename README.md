# ARQVIO

**An experimental, Windows-native research workbench for reproducible trading-strategy discovery.**

ARQVIO combines a Python research engine, a local FastAPI API, a React/TypeScript interface and MetaTrader 5 integration. It is being developed toward **v4.2.0**. This repository contains a **development source snapshot through the current Phase 3 horizon-contract slice**, not a finished release, a profitable trading system, or an installer.

[Getting started](docs/GETTING_STARTED.md) · [Architecture](docs/ARCHITECTURE.md) · [Roadmap](docs/ROADMAP.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## What is here

- Genetic/random search with versioned strategy DNA and constrained signal/filter/management combinations.
- An event-driven reference backtester and a faster Numba path with explicit fallback.
- Statistical research tools including DSR/PSR, PBO/CSCV, purged validation and random/sham controls. Their presence is **not** a certification of any strategy.
- Strategy vault, portfolio analysis, MT5 data/deployment components and local desktop UI under active development.
- Current Phase 1 changes: distinguish blocked data, partial completion, failed search and completed empty search; preserve successful partial results.
- Current Phase 2 changes: separate declared gene support from verified evidence; validate provenance records; reject unsupported active genomes; fix restricted-signal mutation; permit exact oracle input in finalist gates.
- Current Phase 3 slice: explicit research/intraday/swing horizon profiles and an honest refusal for OHLC-only scalping.

**Limits:** Phase 1, Phase 2 and Phase 3 are partial. Trusted proof-run collection, broader feature/interactions parity, tick-faithful scalping, trained ML, end-to-end data reliability and release acceptance remain open. Existing V1/V2 paths remain during migration to a shared engine. The interface's legacy `READY` label is not proof that a feature passed a current MT5 parity run.

## Why this project

Selecting the best backtest from many trials can reward luck. ARQVIO's intended contribution is an inspectable, reproducible workflow: record what was tried, which data and costs were used, why a candidate was refused, and which checks actually ran. Unsupported behavior should fail explicitly rather than produce an attractive approximation.

The project is early-stage. No claims of external adoption, downloads, production safety or investment performance are made.

## Quick start — Windows / another RDP

Prerequisites: Git, **Python 3.11 x64**, [uv](https://docs.astral.sh/uv/), and Node.js 22 LTS. MT5 is Windows-only. Node/npm are needed only for UI development. Rust/MSVC/WebView2 are additionally needed for native packaging, which is not part of this snapshot's setup.

```bash
git clone https://github.com/nitish678/arqvio.git
```

```bash
cd arqvio
```

```bash
uv sync --frozen --extra dev
```

Run the small offline smoke suite without connecting to a broker:

```bash
uv run --frozen --extra dev python scripts/offline_smoke.py
```

Frontend:

```bash
npm --prefix apps/desktop ci
```

```bash
npm --prefix apps/desktop run typecheck
```

```bash
npm --prefix apps/desktop test
```

For application startup, environment settings and migration to another RDP, read [Getting started](docs/GETTING_STARTED.md). Do **not** launch an unknown trading terminal or enable automatic trading to run tests.

## Repository layout

| Path | Purpose |
|---|---|
| `engine/` | API, reference/fast backtests, search, validation, data and MT5 adapters |
| `apps/desktop/` | React UI and Tauri native-shell source |
| `tests/` | Offline fixtures and optional data/MT5-dependent checks |
| `configs/` | Non-secret defaults; no user account configuration |
| `scripts/` | Offline smoke and source build helpers; no prebuilt runtime |
| `docs/` | Setup, design, current status and release gates |
| `.github/workflows/` | Windows offline checks; no trading credentials |

## Data, safety and reproducibility

This public repository deliberately excludes broker histories/ticks, the owner's vault strategies, credentials, account exports, competitor research/decompiled artifacts, worker archives, runtime bundles and release binaries. Some historic tests need private pinned datasets or a separately configured MT5 tester and cannot run from a source-only clone. **Skipped is not passed.** The offline suite uses mechanism fixtures, not investment-performance evidence.

MT5 integration code can affect trading when explicitly configured and invoked. Use an isolated data directory and a dedicated demo/test terminal. Never point automated tests at a live account. No financial advice, guaranteed returns, HFT capability or universal broker compatibility is offered.

## Research references

- Bailey & López de Prado, [Deflated Sharpe Ratio](https://ssrn.com/abstract=2460551).
- Bailey, Borwein, López de Prado & Zhu, [Probability of Backtest Overfitting](https://ssrn.com/abstract=2326253).
- [MetaTrader 5 Python documentation](https://www.mql5.com/en/docs/python_metatrader5).

References motivate research discipline; they do not imply author endorsement. Current provenance work used the paper abstracts; full equation/reference validation remains a roadmap task. No competitor source or private research bundle is included.

## License and project status

Copyright 2026 Nitish Kumar and ARQVIO contributors. Licensed under [Apache-2.0](LICENSE). Third-party packages retain their own licenses and are installed separately. See [NOTICE](NOTICE) for scope.

Maintained by [Nitish Kumar](https://github.com/nitish678). Issues, reproducibility fixes, documentation and careful tests are welcome. No installer or GitHub release is published with this snapshot.
