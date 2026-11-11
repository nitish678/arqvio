# Public snapshot facts

2026-11-11 source-only snapshot published at https://github.com/nitish678/arqvio, commit `4bfeb0a9ac2cb277a9399c4721106075c532109f`. The repository is public, has no release/tag, and is not the private development checkout.

## Contents

- Python source: `engine/`; frontend source: `apps/desktop/src/`; Tauri configuration: `apps/desktop/src-tauri/`.
- Non-secret defaults: `configs/` (cell registry/private market artifact removed from the public snapshot).
- Build/setup files: `pyproject.toml`, `uv.lock`, frontend `package.json`/`package-lock.json`.
- Tests: `tests/`, with optional-market/binary-fixture dependencies marked.
- Documentation: `docs/`.

## Excluded from this repository

- Private Git history/commits/branches/worktrees (`ARQVIO-v2` nested repo).
- Competitor research packages, decompiled artifacts, `algory_research`, `Downloads/arqvio_v4_extracted`.
- Worker history, `worker1/`–`worker5/`, private reports, logs and screenshots.
- Broker datasets, account exports, vault strategies, `artifacts/`, `research_data/`, `vault/`.
- Credentials, `.env`, OS keyring material, broker terminal configurations.
- Executables/installers, `build/runtime`, `src-tauri/target`, `.exe`, `.ex5`.
- Runtime bundle at `apps/desktop/src-tauri/resources/arqvio-engine/`; native packaging cannot be built from this snapshot without building that runtime first.
- Historical deploy bundle and cell-registry fixture with private evidence references.
- Sidecar `ARCHIVE/` directories.

## Preservation vs publication

Private full verification history is retained under the developer's local private archive, not published. Open-source publication can use old words and examples, but not exact records unless separately reviewed.

## Public snapshot verification

- Public source scan found no private profile paths, key/token/private-key patterns, account-number patterns or private-network addresses. Two dummy test strings were renamed to avoid secret-like scanner matches.
- Public offline smoke: **71 tests passed**, one dependency warning, 12.52s in the local environment. Backend exit 0. The runner reports dependency versions separately; this local environment did not match all lockfile pins (e.g. numpy 2.4.6 vs expected 2.2.6, pandas/polars not installed). This is not a fresh-clone dependency acceptance claim.
- Private development checkout Phase 3 targeted verification: **39 API/horizon tests**, then **47 combined backend regressions**, and **29 desktop tests / 6 files** with TypeScript clean. These numbers are evidence for the development checkout, not a claim that every public clone has the same private fixtures.
- Private development checkout Phase 4 validation suite: **140 passed**, 3 documented numerical warnings, 175.61s; tester terminal discovery remained read-only and CLOSED.
- Public repo contents and commit were verified through GitHub API. Sensitive `configs/cells_registry.json` and historical deploy fixture are absent.
- No MT5 API calls, terminal operations, live actions, package build or native installer run.

## What was not rerun

- Full historical backend suite requiring private pinned datasets.
- Installed/MT5-dependent integration checks.
- Vite/Vitest on this exact public tree after sanitization.
- Package builds and native packaging.
- Native setup/release or Windows installer verification.
