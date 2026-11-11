# Public snapshot facts

2026-11-11 snapshot of local source work, sanitized for public source-only publication.

## Contents

- Python source: `engine/`; frontend source: `apps/desktop/src/`; Tauri configuration: `apps/desktop/src-tauri/`.
- Non-secret defaults: `configs/`.
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
- Sidecar `ARCHIVE/` directories.

## Preservation vs publication

Private full verification history is retained under this developer's local private archive, not published. Open-source publication can use old words and examples, but not exact records unless separately reviewed.

## What was rerun on this public snapshot

None yet. Local offline checks above were run on a private copy before this public source-only export. Fresh checks on a fresh clone are welcome; private historical test counts are not claimed for a fresh clone.

## What was not rerun

- Full historical backend suite requiring private pinned datasets.
- Installed/MT5-dependent integration checks.
- Vite/Vitest on this exact public tree after directory reorganization.
- Package builds and native packaging.
- Native setup/release or Windows installer verification.
