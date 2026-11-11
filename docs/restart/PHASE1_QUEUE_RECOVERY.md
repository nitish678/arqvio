# Phase 1 queue recovery — readiness-safe slice

2026-11-11. Status: partial.

## Implemented

- Queue schema version 2 records `WAITING_TERMINAL`, reason and last wait UTC.
- `start_next()` no longer removes the queue head before launch readiness is known.
- Terminal/refusal/failure keeps the original launch request at queue head; successful start removes it only after launch accepts it.
- Repeated readiness attempts retain the same queue ID/request, preventing dropped or duplicate intent.
- Existing queue order and user DELETE/CLEAR behavior remain.

## Verification

- Queue/launch outcome suite: **36 passed**, 5.79s.
- Queue + launch + M1 data suite: **49 passed**, 4.68s.
- No live terminal action; tests use isolated homes/fakes.

Remaining: automatic readiness polling, multi-process locking/restart crash proof, broker source provenance and full UI queue-state presentation.
