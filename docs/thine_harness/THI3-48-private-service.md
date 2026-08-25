# THI3-48 Hermes private service

This maintained-fork service is the authenticated, loopback-only boundary from
the local Thine backend to Hermes. It is not a phone route, is never placed
behind the tunnel, and does not expose a generic tool, RPC, or SQLite surface.

## Configuration

Use a dedicated `HERMES_HOME` for the daily-driver profile. Put behavior in its
`config.yaml`:

```yaml
thine_harness:
  private_service:
    enabled: true
    host: 127.0.0.1
    port: 8789
    firebase_uid: YOUR_FIREBASE_UID
    request_timeout_seconds: 5
    credential:
      env: HERMES_CONTROL_TOKEN
      file: ""
```

`host` must be a loopback IP literal. Startup rejects wildcard, LAN, DNS, and
public addresses. `firebase_uid` binds the process to exactly one local
daily-driver identity.

The credential value must not be written into `config.yaml`. Set
`HERMES_CONTROL_TOKEN` in the profile's secret environment, or set
`credential.env` to an empty string and `credential.file` to a private token
file path. Exactly one source is required when the service is enabled.

## Start, verify, stop, and restart

With the repository environment active and `HERMES_HOME` plus the configured
secret available:

```bash
python -m thine_harness.private_server
```

The server listens at `http://127.0.0.1:8789`. Verify it from the Mac with the
same secret and configured UID:

```bash
curl --fail-with-body http://127.0.0.1:8789/health \
  -H "Authorization: Bearer $HERMES_CONTROL_TOKEN" \
  -H "X-Thine-Firebase-UID: YOUR_FIREBASE_UID" \
  -H "X-Request-ID: operator-health-1"
```

The response includes a process instance ID and start time. They remain stable
for the life of the process, so the backend can distinguish a normal reconnect
from a Hermes restart. Stop with `Ctrl-C`; restart with the same module command.

## Private wire contract

Every request requires all three headers:

- `Authorization: Bearer <HERMES_CONTROL_TOKEN>`
- `X-Thine-Firebase-UID: <configured UID>`
- `X-Request-ID: <non-empty ID, at most 128 characters>`

Missing or incorrect credentials return `401`; a different Firebase UID returns
`403`; an invalid request ID returns `400`. The authenticated `GET /health`
route is implemented here. `POST /v1/control` is reserved for the typed
`HermesControlPort` integration and deliberately returns `501` until that
ticket supplies feature behavior. Active requests exceeding
`request_timeout_seconds` return a redacted `504 request_timed_out` response.
Other paths return `404`.

The local backend launcher probes the fixed authenticated health boundary at
`http://127.0.0.1:8789/health` with the same `HERMES_CONTROL_TOKEN`, configured
UID, and a request ID. Hermes enforces the finite server-side request deadline.
The phone continues to use only the local Thine tunnel origin; it never receives
or calls this address.
