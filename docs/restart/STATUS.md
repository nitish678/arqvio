# Restart status — Worker 1

Date: 2026-11-11. Release target V4.2.0; [nine-phase gates](V4_2_RELEASE_GATES.md). Source baseline `arqvio-v4@630fe0b` retained. No merge, commit, push, package build, installed runtime change, live terminal operation or trading action.

## Phase 0

| Work | Status | Evidence / limitation |
|---|---|---|
| Repo/ref and selected runtime inventory | VERIFIED (scoped) | AUDIT.md; static-fingerprints.json in local archive; staged engine differs; installed selected sources absent |
| Archive checkout + branch worker evidence | VERIFIED | 53 current files, 188 total ZIP members, all decompressed bytes/digests checked |
| Archive helper | IMPLEMENTED | scripts/restart_archive.py; refuses changed file set/digests/HEAD/branch refs and links/reparse points |
| Canonical audit/approved roadmap/sources | IMPLEMENTED | AUDIT.md, ROADMAP.md, RESEARCH_SOURCES.md; root MASTER_PLAN marked historical |
| Offline registry/V1 mechanism baseline | VERIFIED (scoped) | 23 passed, 1 dependency deprecation warning, 16.49s, exit 0; real MT5 functions blocked. First strict-import attempt stopped collection (2 errors, exit 2); both logs retained |
| Worker folder deletion | VERIFIED | 53 files from current worker1–5 removed after exact source/archive/HEAD/branch checks; 188 archived members independently verified; cleanup.json records 2026-11-10 17:55 UTC |
| Full project/RC acceptance | PLANNED | No whole-suite, primary statistical-paper review, current UI or MT5 tester coverage claimed |

Private evidence directory (workspace-relative): `restart-archives/worker-evidence-20261110-01/`. Archive is local only; do not upload/publish it without owner permission. Manifest contains filenames and hashes, not report excerpts or secrets.

## Later phases

1. MT5 data reliability and truthful launch outcomes — PARTIAL: launch accounting, M1 retrieval and queue-head readiness retention verified. Automatic polling/multi-process locks/funnel/provenance remain. See [launch outcomes](PHASE1_LAUNCH_OUTCOMES.md), [queue recovery](PHASE1_QUEUE_RECOVERY.md) and [M1 fix](M1_HISTORY_FIX.md).
2. Shared contracts and proven genome capabilities — PARTIAL: additive capability/evidence contract, admission guard, restricted-signal mutation fix and oracle gate fallback verified. 263 Python passed/3 skipped; 29 desktop passed; TypeScript clean; final pinned canary PASS (9.8s, 0 crashes). Actual proof collector and broad parameter/interaction coverage remain. See [Phase 2 evidence](PHASE2_CONTRACTS.md).
3. Horizon-aware strategy families/fills — PARTIAL: explicit research/intraday/swing profiles, paged M1 history and experimental tick fill path verified; pending/partial/full scalping parity remains. See [Phase 3 evidence](PHASE3_STRATEGIES.md), [M1 fix](M1_HISTORY_FIX.md) and [tick experiment](TICK_SCALPING_EXPERIMENT.md).
4. Validation/generalization — PARTIAL: strict trial provenance and 140 validation tests pass; lockbox/nested-selection/evidence-writer work remains. See [Phase 4 evidence](PHASE4_VALIDATION.md). Statistical formulas remain frozen.
5. Measured ML — PARTIAL: deterministic leakage-safe shadow baseline and 4 ML tests pass; no trained edge/deployment claim. See [Phase 5 evidence](PHASE5_ML.md).
6. Friendly UI/UX — PARTIAL: horizon selector and honest scalping-unavailable state tested; full readiness/wizard/visual accessibility review remains. See [Phase 6 evidence](PHASE6_UI_UX.md). App copy English, owner discussion Roman Hinglish.
7. Portfolio/automation/paper evidence — PARTIAL: planned exposure, read-only replacement eligibility and automation idempotency keys verified; crash replay/live exposure/refresh remain. See [Phase 7 evidence](PHASE7_PORTFOLIO_AUTOMATION.md).
8. Release acceptance — PLANNED; installer/live actions require separate authorization.

Phase 0 preservation/canonical docs/scoped cleanup finished. Tester read/data validation for Phases 0–4 recorded in [tester evidence](PHASE0_4_TESTER_VALIDATION.md); tester remained read-only. Broader source/test inventory and RC acceptance remain unverified work, not implied by this scoped baseline. Phase 1A launch outcome/partial accounting fix now implemented and scoped-tested. Next: terminal readiness intent/queue retention and measured funnel; no branch merge or terminal access is implicit. No commit was made because the owner has not asked for one.
