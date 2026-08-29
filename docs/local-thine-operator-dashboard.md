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
state, schedules, topics/preferences, retention/reset policy, and the last 50
redacted invocation records. Each panel names its authoritative helper and
reports freshness and errors.

Operator controls always use a preview followed by exact confirmation. They
include schedule run-now/cancel, Home replacement/reactivation, explicit
quarantine/action retry when the corresponding owner helper is attached, and
scoped/full reset. Reset execution ignores client assertions and checks the
private service's process-lifecycle marker; it refuses while the harness is
running. Working Memory cannot be restored and there are no global autonomy
switches.
