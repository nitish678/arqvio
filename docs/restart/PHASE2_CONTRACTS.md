# Phase 2 — shared evidence and genome capability foundation

2026-11-11, Worker 1 / main. Target release: V4.2.0. Status: PARTIAL, scoped implementation verified. No merge/commit/push/build/deploy or real MT5 calls.

## Implemented

- Additive GET `/api/v1/gene-pool/capabilities`. Existing `/registry` body and registry hash remain unchanged (`c482d8bc0d90f900`, pinned by test).
- Support declarations read existing runtime allowlists; tester entry-gate/switch limitations are explicit. All proof fields remain UNVERIFIED / empty: READY and test filenames are not execution attestations.
- Capability payload distinguishes fixed switch rules, actual family breeding fields, cell policies and unavailable parameterization; exposes current composition semantics, not unimplemented configurable quorum.
- GeneBrowser displays declarations and unverified evidence; mismatched registry hash or absent capability response cannot produce a support claim. No new visual framework or colors.
- Strict additive EvaluationEvidence contract records experiment/run/candidate/source/dataset/config/artifact IDs, cost/validation versions, target/path/stage, UTC completion, gene coverage and explicit PASS/FAIL/SKIP/ERROR. Missing hashes, naive time, unknown/duplicate genes and slow fallback presented as kernel pass are rejected. Record validity alone does not independently verify a run. No automatic deployment/promotion consumer added.
- GA admission rejects unsupported active risk, partial-close, legacy management and invalid switch combinations before effective-hash admission. Doesn't alter valid DNA serialization or RNG.
- Regression reproduced restricted SIGNALS mutation producing an empty tuple. Fixed deterministic first-allowed fallback, matching sampler behavior with no extra RNG draw. Valid existing mutation streams remain unchanged; the prior failing case now produces a supported nonempty entry.
- Common gate input (fills/closed net P&L/equity) removes V1 final-gate dependence on kernel row capture. Kernel capture converts directly; no capture reruns exact oracle and records path/reason. Legacy tuple callers still work. Fill/cost/kernel interfaces unchanged. Fallback may rerun oracle twice because existing fast wrapper may already have fallen back; correctness-first, optimization deferred until raw-result exposure is safely available.

## Files

`genes/capabilities.py`, `genes/evidence.py`, `search/evaluation_view.py`; additive route in `api/routes_gene_pool.py`; admission `ga/population.py`, mutation `ga/operators.py`; V1 `cli.py` / `gates.py`; frontend `api/ws.ts` and `deck/GeneBrowser.tsx`. Tests in `tests/research/genes/` and desktop `gene-capabilities.test.tsx`.

## Evidence

Workspace-private `restart-archives/` contains test logs/JSON:
- `phase2-red-20261111`: initial missing route/admission regression failures.
- `phase2-green-20261111`: 218 passed, 3 skipped (71.11s).
- `phase2-mutation-red-20261111`: empty SIGNALS mutation failure before fix.
- `phase2-acceptance-20261111`: **263 passed, 3 skipped, 1 dependency deprecation warning, 46.72s**, exit 0. Targets genes, GA, V1, schema-v4 compatibility and previous launch regressions. Real MT5 API functions blocked, temporary home.
- Skips: two GA integration fixtures lack pinned H1 dataset; one power-check fixture lacks Stage-2 smoke outputs. These are unverified, not passes.
- `phase2-final-20261111/ui-tests-fixed.log`: **29 passed / 6 files**, TypeScript preceding command exit 0. Initial TypeScript attempt caught an invalid extra field in the new test fixture; fixture corrected, no type escape.
- `phase2-acceptance-20261111/canary.log`: final pinned 2594 bars, **PASS 9.8s, 0 crashes**, after gate-view adapter. Earlier run passed in 12.3s; no speed comparison claimed. This is a planted-edge diagnostic, not market-edge discovery. No backtest kernel changed.

Self-review: legacy identity/hash fixtures, no changed statistical thresholds, explicit skip reasons, no fake kernel rows, no generated VERIFIED claims. Gate-view tests compare identical inputs' full gate decisions to legacy tuple path and assert exact oracle invocation on capture absence. No fresh real-data oracle/kernel/EA parity for all combinations, no browser visual QA, no full backend suite or independent agent review claimed.

## Research used

Algory pack standard genome groups/composition concepts; original source/format not copied. On 2026-11-11 SSRN abstracts for DSR (`2460551`) and PBO (`2326253`) retrieved through Exa. They motivate honest multiple-trial/provenance tracking, not hardcoded survival weights or formula changes. Full primary PDFs/mathematics remain validation-phase work.

## Remaining Phase 2 acceptance

- Trusted runner/artifact collector tying each proof to exact current source/data/config bytes and fired coverage; schema-only input must not grant VERIFIED.
- Full effective/ignored field mapping and per-parameter effects, real pinned feature-fired/pairwise interactions, legacy artifact migrations and actual writer integration across both workflows.
- Expanded configurable grammar/compatibility rules (AND/OR/quorum changes need matching oracle/kernel/EA semantics and proofs); current runtime grammar is exposed without claiming these options exist.
- Full process-wide unified engine migration not implemented; legacy V1/V2 execution paths still exist, sharing the additive gate representation.

Do not mark V4.2.0 or Phase 2 complete from these scoped green tests. See V4_2_RELEASE_GATES.md.
