# Phase 1A — truthful launch outcomes and partial accounting

2026-11-10, Worker 1. Implemented in current source, not installed/packaged. Phase 1 overall remains PARTIAL.

## Root cause and fix

`routes_app_launch._work` always ran `_chain` and set DONE after prerequisite refusals or caught breed errors. This could report successful zero-strategy search when no search ran, and could repeat/purge after failure. `automation.steps.run_campaign` also called `_mark_result` for skipped markets, incorrectly treating absent data as empty completed search.

Classification now:
- owner stop → STOPPED;
- any failed breed → FAILED, preserving additions;
- no completed breed plus failed data → BLOCKED_DATA (a successful helper/cross pull does not change this);
- completed breed plus failed data → PARTIAL_DATA, preserving additions;
- other incomplete/unexpected task states → FAILED;
- all completed, additions zero → COMPLETED_ZERO_SURVIVORS;
- all completed, additions positive → DONE.

Only DONE/COMPLETED_ZERO_SURVIVORS run existing launch after-actions. Failure/partial reasons persist and appear on CAMPAIGN. Automation accounts only DONE breed markets; completed additions are retained before failure interpretation. PARTIAL_DATA warns and proceeds to remaining chunks/steps; unknown/failed/incomplete outcomes fail visibly. The scheduler records actual `warn` activity rather than green success.

Existing `int(cur.get('vaulted', 0))` and timestamp-based vault membership accounting are unchanged in this slice; exact launch-owned IDs and concurrent-run/restart identity are next-slice work. No terminal waiting/automatic retry is claimed yet. No candidate funnel counts invented.

## Files

- `engine/api/routes_app_launch.py` — completion classification and after-action gating.
- `engine/research/automation/{steps,runner}.py` — DONE-only accounting, partial warning/continuation.
- `apps/desktop/src/api/ws.ts`, `deck/CampaignTab.tsx` — additive outcome types and visible reason.
- `tests/api/test_restart_launch_outcomes.py` — new offline regressions and fake-MT5 pull boundary tests.
- `apps/desktop/src/__tests__/launch-outcomes.test.tsx` — outcome text/status rendering.
- `scripts/restart_offline_baseline.py` — optional target arguments; all official MT5 API functions blocked before project imports.

## Fresh verification

Evidence is in workspace `restart-archives/`:

| Run | Result | Evidence directory |
|---|---|---|
| Test-first unchanged production source | 12 failed, 1 passed, exit 1 | `phase1-red-20261110` |
| First fix + existing launch/automation suites | 39 passed, 1 dependency deprecation warning, 7.16s, exit 0 | `phase1-green-20261110` |
| Expanded final fake-pull/scheduler cases + same suites | 43 passed, 1 dependency deprecation warning, 7.19s, exit 0 | `phase1-final-20261110` |
| Desktop TypeScript | `tsc --noEmit`, exit 0 | same shell chain preceding UI test log |
| Full installed-dependency desktop Vitest | 27 passed across 5 files, 14.45s, exit 0 | `phase1-green-20261110/ui-tests.log` |
| Whitespace check | `git diff --check`, exit 0 | command transcript |

Backend command: Hermes Python `-I scripts/restart_offline_baseline.py <repo> <evidence> tests/api/test_restart_launch_outcomes.py tests/api/test_routes_app_launch.py tests/research/automation`. Each run uses temporary ARQVIO_HOME, no default owner-home writes, blocked real MT5 functions. FakeMt5 initialize/shutdown records are in-memory test calls, not real terminal actions. Tests exercise production `_work → _data → add_market.pull` for stale clock, 12-bar history, and logged-out refusal; only unrelated specs/cost calibration injected.

UI: existing Node dependencies, `node node_modules/typescript/bin/tsc --noEmit` then `node node_modules/vitest/vitest.mjs run`. No package installation/network.

Spec/standards self-review: outcome precedence, helper-data case, failed-breed precedence, missing/unknown state, preserved additions, no partial repeat, warning severity and frontend consumers checked. No thresholds/tolerances, frozen hashes, kernel, scores or validation interfaces changed. No independent agent review claimed. No browser screenshot/visual QA this slice; component text regression only. No full backend suite, canary or MT5 parity run (kernel unchanged); no profitability or latency claims.

## Next approved work

Persist terminal readiness/waiting intent and retain queue head without duplicate startup, then measured search funnel and stronger data provenance. Continue offline fixtures first; no silent broker switch, terminal auto-open, branch merge, deploy or installed runtime change authorized by this slice. No commit/push performed.
