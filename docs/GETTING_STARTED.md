# Development on a fresh Windows PC or RDP

## 1. Install tools

- Git for Windows.
- Python **3.11, 64-bit** and uv. This snapshot's MetaTrader5 package is Windows-only.
- Node.js 22 LTS for UI tests/development.
- Optional: MT5 plus a **separate demo/test terminal** for broker-dependent work. Offline checks do not need MT5 installed or a broker login.
- Optional native-shell builds: Rust stable, MSVC C++ Build Tools, WebView2. Bundled engine/runtime generation is separate and not necessary for browser-based UI development.

Never copy a virtual environment or `node_modules` between RDPs; reinstall from the lockfiles.

## 2. Clone and install

```bash
git clone https://github.com/nitish678/arqvio.git
```

```bash
cd arqvio
```

```bash
uv sync --frozen --extra dev
```

```bash
npm --prefix apps/desktop ci
```

The Python lockfile is retained from development. Fresh Windows dependency resolution/installation is checked separately from tests run in the maintainer's existing environment. If a platform-specific wheel is unavailable, report the exact error; do not silently change package versions.

## 3. Offline verification first

```bash
uv run --frozen --extra dev python scripts/offline_smoke.py
```

This creates a temporary home, blocks official MT5 API functions, and runs registry/evidence, synthetic GA/V1 mechanism, and launch-accounting fixtures. It does not open a terminal or prove market profitability. JSON/log output lives in `smoke-evidence/` (ignored).

```bash
npm --prefix apps/desktop run typecheck
```

```bash
npm --prefix apps/desktop test
```

The full historic backend suite additionally expects private market snapshots, symbol/cost metadata, old report fixtures and sometimes a configured tester. Those are intentionally not distributed. Missing-data skips and absent private-fixture failures must not be represented as a fully verified clone.

## 4. Start local API/UI (explicit opt-in)

Starting the app can discover installed terminals and start background application services. Review the configuration first; never use a live trading terminal for development. Keep each project/data home isolated.

Windows PowerShell, terminal one:

```powershell
$env:ARQVIO_HOME = Join-Path $PWD 'local-data'
$env:ARQVIO_API_PORT = '8742'
$env:ARQVIO_APP = '1'
uv run --frozen --extra dev python -m engine
```

PowerShell, terminal two in the same clone:

```powershell
$env:VITE_ARQVIO_ENGINE_PORT = '8742'
npm --prefix apps/desktop run dev
```

Open `http://localhost:1420`. Missing terminal/history is expected until you explicitly configure an authorized demo data source. Research and deployment are distinct; a campaign should not open trades. The code contains deployment/execution components, so review actions before invoking them.

Do not run native packaging merely to preview the UI. `src-tauri/tauri.conf.json` references a runtime bundle that is deliberately absent; build helpers must create it before a future native package build. No installer is supplied here.

## 5. Move work to another RDP

Source/code changes:

```bash
git status
```

```bash
git add <reviewed-source-files>
```

```bash
git commit -m "Describe the change"
```

```bash
git push
```

On the other RDP, clone or pull your branch and repeat locked dependency installation. Review `git diff --cached` before pushing: **never commit `.env`, accounts, vaults, broker histories, logs or generated EAs**.

Private operational data is not a GitHub source backup. If needed, create a separate encrypted backup using your own trusted storage and transfer keys/credentials through a secure channel. Do not upload broker-licensed data without permission. This repository does not migrate open positions, OS keyring entries, terminal installations or broker sessions.
