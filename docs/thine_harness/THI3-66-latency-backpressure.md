# THI3-66 latency, cache, and backpressure verification

This verification is bound to the Hermes integration base
`f66b02eaca6cd5b6cf8af21194834330dac1808e`. It measures the local Harness
without selecting product SLAs or invoking a paid provider. The default probe
uses a temporary SQLite database, the real queue/search/Stop Hook limit code,
a fake invocation runtime, and no backend or live Hermes profile.

## Run it

From the pinned repository environment:

```bash
./.venv/bin/python -m thine_harness.performance_diagnostics --transcript-burst 1000
```

Provider cache evidence is deliberately opt-in. If an operator explicitly runs
the existing isolated live probe, its JSON can be folded into the same report:

```bash
./.venv/bin/python -m thine_harness.performance_diagnostics \
  --cache-evidence /path/to/integrated-probe.json
```

The offline report says `not_run_offline` instead of manufacturing cache-read
evidence. The controlled live proof recorded in `THI3-41-runtime-proof.md`
remains the current provider evidence: identical prompt-cache key, system
prompt hash, and wire tool array across primary and Stop Hook requests, with
1,536 Stop Hook cache-read tokens. Cache-read counts are observations, not pass
thresholds; a cold cache may validly report zero.

## Reproducible sample

One isolated run on 2026-08-29 with a 1,000-transcript burst observed:

| Measurement | Observation |
| --- | ---: |
| enqueue 1,005 ticks | 3,990.83 ms |
| drain 1,005 ticks | 9,478.15 ms |
| first completed kind | `p0_user_chat` |
| interaction index | 1,001 |
| promoted overdue schedule index | 1,003 |
| ordinary P2 schedule index | 1,004 |
| SQLite lock held | 58.87 ms |
| blocked writer completion | 65.26 ms, succeeded |
| SQLite busy timeout | 5,000 ms |
| SQLite journal mode | `delete` |
| fixed half-hour scan drift fixture | 137 ms |

These host timings are diagnostic samples, not acceptance budgets. The queue
invariants are deterministic: an arriving P0 runs first, the transcript burst
remains FIFO, the fixed-boundary interaction remains behind already queued
transcripts, a later transcript and promoted overdue schedule are not lost,
and ordinary P2 work eventually completes. This probe drains deliberately one
tick at a time, so its throughput includes one durable SQLite lifecycle per
tick and does not model provider latency.

## Prompt and Working Memory bounds

The 21 current Thine helper schemas occupy 11,971 canonical UTF-8 bytes and an
estimated 2,604 request tokens when eager. The actual deferred bridge occupies
1,514 bytes and an estimated 323 request tokens, saving 10,457 exact bytes and
approximately 2,281 tokens for this schema set. The token values use Hermes'
rough request estimator and are labeled as estimates; byte counts are exact.

The controlled Working Memory probe exercises the real `StopHookRunner` with
an exact deterministic counter. A 16,000-token candidate commits, a
16,001-token candidate requests one same-context correction to the 14,000-token
target, and that correction commits without changing the captured cache
identity. Production changed-memory writes still require configured-model
token evidence and fail closed when it is unavailable. The fixed runtime
envelope is:

```text
provider context                         272000
system + eager bridge prefix reserve       4096
Working Memory reserve                    16000
output + reasoning reserve                 32768
absolute transcript ceiling              200000
unallocated safety margin                  19136
routine transcript target                  8000
```

## P0 latency trace

Each admitted user chat now retains redacted first-occurrence milestones for
submission resolution, accepted/started publication, first safe progress,
first model output, model completion, canonical reply persistence, Stop Hook
completion, and the terminal lifecycle event. `P0ChatController.latency_trace`
returns elapsed milliseconds from durable admission to first progress, first
model output, reply persistence, and terminal publication. It stores no user,
assistant, or progress text. Only the latest 50 delivered P0 traces are kept;
active traces are never pruned by that bound.

The first-model-output milestone uses the first published assistant delta when
the session streams one. Non-streaming sessions fall back to model completion,
so that value is an upper bound on time to first model output for those
sessions. The terminal milestone means the backend accepted the lifecycle
publication call; canonical final content was already persisted separately.

## Explicit operating limits and external gaps

- There are no product latency or throughput budgets. Reports must remain
  measure-only until the product chooses them.
- Safe P0 progress may be silent for at most 5,000 ms by contract. The current
  heartbeat interval is 3,000 ms.
- SQLite waits up to 5,000 ms for a writer. The isolated database reports
  `journal_mode=delete`; a lock held beyond the busy timeout remains a retryable
  runtime fault, not hidden queue loss.
- Interaction scans use fixed local `:00`/`:30` boundaries. OS suspension adds
  scan drift, but the durable cursor makes the next scan claim the missed
  closed window.
- Schedules overdue by ten minutes are promoted to the P1 tail. A logical input
  has exactly three attempts total.
- The local backend watchdog checks the generated Quick Tunnel every 30
  seconds and fails the stack after three consecutive unhealthy checks. It does
  not mint a replacement tunnel in place. A replacement Quick Tunnel has a new
  host, while the phone persists `debug_override_url` and requires restart;
  therefore tunnel death currently requires operator stack restart and phone
  relaunch with the newly printed host.
- Mobile WebSocket recovery is stronger within the same tunnel origin: native
  retries are followed by an app-level reconnect after five seconds, chat
  dispatch has a 20-second acknowledgement watchdog, and a retry tap forces a
  fresh handshake with a ten-second default timeout. None of those paths can
  recover a changed Quick Tunnel hostname automatically.

The backend and mobile findings above are read-only inspection results. This
ticket changes neither repository and does not claim an end-to-end tunnel
reconnect benchmark.

## Verification

The focused behavior tests cover the report, priority/FIFO pressure, exact
Working Memory correction, SQLite contention, fixed-boundary drift, optional
cache evidence, P0 milestone ordering/redaction, and the 50-trace retention
bound. Run them through the repository wrapper:

```bash
scripts/run_tests.sh \
  tests/thine_harness/test_performance_diagnostics.py \
  tests/thine_harness/test_p0_chat_control.py -q
```
