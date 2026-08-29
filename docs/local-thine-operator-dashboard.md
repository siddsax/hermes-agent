# Local Thine Operator Dashboard

The built-in operator dashboard is a bounded projection over Hermes-owned
repositories. It does not maintain another state store, read backend/mobile
tables, or expose itself through the phone tunnel. Missing backend/mobile owner
helpers are shown as unavailable rather than guessed.

Configure it in the Hermes profile:

```yaml
thine_harness:
  operator_dashboard:
    enabled: true
    host: 127.0.0.1
    port: 8792
```

The normal private-service launcher starts this second loopback listener in the
same Hermes process, while the phone-facing private service remains on its own
port, when `operator_dashboard.enabled` is explicitly set to `true`. For
maintenance while that service is stopped, it can also be started directly:

```bash
hermes-thine-operator
```

Then open `http://127.0.0.1:8792`. The listener accepts only a literal loopback
address and rejects forwarded/proxy headers. Control requests also require a
process-local token embedded in the same-origin page.

The page shows queue/lease/attempt/checkpoint/receipt state, current-run status,
transcript claims, Working Memory and its last 50 versions, Home head/history,
interaction and speaker ingestion state, communication delivery and permission
state, a redacted push-registration summary, schedules, topics/preferences,
recent durable retention/reset events and plans, current live-work count, and
the last 50 redacted invocation records. Queue, Attempt, checkpoint, receipt,
and quarantine rows are bounded in SQLite before they enter the projection;
checkpoints and receipts are the newest 50, and the active-run receipt count is
queried for that run rather than inferred from the bounded timeline.

Each panel names its authoritative helper and reports two different clocks:
`read_at_ms` is when the dashboard called the helper, while
`owner_observed_at_ms` is the newest durable observation or mutation represented
by that value. An empty but successful owner read therefore says `read` and does
not pretend that state changed during refresh. Composite panels include the
same freshness fields per source value, and the page renders those component
clocks under **source freshness**.

The product-attached communications panel independently reads the backend-owned
notification permission and redacted push-registration summary. A failure in
one leaves local action/allowance history and the other backend value visible.
The registration summary contains only whether a registration exists, its
count, and when one was last observed; device tokens and provider credentials
never enter the dashboard projection.

The product-attached speakers panel uses the same authenticated backend client
to call the closed `POST /v1/maintenance/inspect` helper. It strictly projects
only the speaker event/outcome counts, normal cursor, newest 50 mapping
identities, and newest 50 speaker quarantines; unrelated backend maintenance
state and mapping payload content never enter the dashboard. If that backend
read fails, Hermes-retained mapping inputs and its cursor remain visible while
the canonical portion is marked unavailable.

Those values use the existing private backend communications client. The
standalone dashboard has no backend client, so communications and canonical
speaker state are explicitly partial. A failed backend value is isolated and
exposes neither a response body nor registration content.

Operator controls always use a preview followed by exact confirmation. They
include schedule run-now/edit/cancel, Home replacement/reactivation, explicit
quarantine/action retry when the corresponding owner helper is attached, and
scoped/full reset. Reset execution ignores client assertions and checks the
private service's process-lifecycle marker; it refuses while the harness is
running. Working Memory cannot be restored and there are no global autonomy
switches.
