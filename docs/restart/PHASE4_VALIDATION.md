# Phase 4 — validation/generalization foundation

2026-11-11. Status: partial. No release or trading action.

## Implemented

- Trial registry now requires `dataset_hash`, `source`, and `as_of` on every new `record` row. Search provenance must match the campaign dataset; validation reruns may use their explicit new-cell dataset. Legacy rows remain readable and are excluded when operation type/provenance is unknown.
- Existing purged/embargoed CV, DSR, PBO/CSCV and SEARCH_TRIAL versus VALIDATION_RUN paths retained; no formula or threshold changes.
- Tester discovery was read-only. `C:\mt5\tester\terminal64.exe` was found but currently CLOSED; no initialization, process start, data pull, order, EA or AutoTrading action occurred.

## Verification

- New strict provenance tests: 2 passed before broader suite.
- Focused accounting suite: **22 passed**, 26.95s.
- Complete validation suite: **140 passed**, 3 numerical/statistical warnings, 175.61s. Warnings are existing PBO polynomial conditioning and PSR near-constant-return precision warnings; no assertions were weakened.
- DSR/PBO formulas untouched. No canary needed for this registry-only change; kernel canary remains a separate gate.

## Remaining

- Lockbox governance: immutable final holdout identity and access/audit record.
- Nested training/model selection and feature preprocessing policy.
- Purge/embargo semantics tied to actual label and holding intervals, not only a configured horizon.
- Trusted evidence writer tying trial rows to source/config/artifact hashes.
- Full tester-backed data provenance after tester is explicitly ready, still read-only.

No Phase 4 completion claim.
