# THI3-47 durable fake Tick core

This slice establishes the local durable execution seam before any real model,
transcript, Home, chat, notification, schedule, speaker, or mobile behavior is
wired. It consumes the frozen v1 `Tick` DTO and stores state below the active
profile's `HERMES_HOME` by default.

## Public seams

- `DurableRunState` owns SQLite migrations and user-scoped transactions for
  queue items, live leases, fault Attempts, checkpoints, acknowledged fake-tool
  receipts, and immutable quarantine records.
- `RunCoordinator` leases and invokes at most one Tick, orders P0 before P1
  before P2 with FIFO inside a priority, signals a live background invocation
  when P0 arrives, and renews its lease while the invocation is active.
- `FakeFeaturePort` is the only feature seam in this slice. It accepts a typed
  command and returns an acknowledgement. It exposes no database or generic
  backend operation.
- `HarnessDiagnostics` is the read-only operator seam for queue, leases,
  Attempts, checkpoints, receipts, quarantine, and the accepted GPT-5.6 Sol
  medium/tool-search configuration.

## Attempt and recovery rules

The first inference is Attempt 1. A provider/runtime/finalization fault or an
expired live lease with no durable checkpoint closes that Attempt as a fault.
The third fault quarantines background work; P0 becomes terminal. Intentional
P0 preemption, cooperative yield, and bounded continuation persist a checkpoint
and requeue the same Logical Run with its current Attempt still open.

A checkpoint includes the stable receipt IDs known for that Logical Run. On
resume, both the checkpoint and acknowledged receipts are loaded into the next
invocation context. Repeating the same action identity and fingerprint returns
the stored receipt without calling the feature port. Reusing an action identity
with a different fingerprint fails closed.

## Verification

Run the focused suite through the repository wrapper:

```bash
scripts/run_tests.sh tests/thine_harness/test_durable_run_coordinator.py -q
```

The tests use real temporary SQLite files, fresh repository/coordinator
instances for restart cases, and two independent repository instances for the
live-lease single-flight case.
