# Local Hermes contract bindings v1

Status: accepted THI3-44 consumer boundary. This document describes data
bindings only; it does not authorize runtime orchestration or product behavior.

## Source and provenance

`thine_harness/contracts/local-hermes-thine/v1` is a vendored, immutable copy
of the language-neutral contract pack accepted at controller commit
`2479efa6059ae2b0185cfdf575c53c74eb64ce59`. The adjacent provenance file pins
the SHA-256 digest of every copied artifact. Committed code never follows a
symlink or reads the controller checkout.

The pack remains the wire authority. The Python modules under
`thine_harness.contracts` are consumers: each of the 57 manifest targets has
one explicit immutable DTO class in the module assigned by
`metadata/serialization-map.json`.

Each DTO exposes a read-only `payload` view. Generated `Protocol` structures
cover nested objects, arrays become tuples, and closed enum/constant fields use
`Literal` discriminants, so ports can remain statically typed without changing
the frozen wire. Regenerate these views with
`python thine_harness/contracts/_codegen.py`, then run the repository formatter.

## Decoder behavior

`decode_contract(type_id, wire)` performs strict JSON parsing, schema
validation, and the pack's payload-level invariants before returning the typed
DTO. It rejects:

- duplicate object members;
- non-finite and unsafe-range JSON numbers;
- unknown major versions, top-level fields, and closed enum values;
- the frozen pack's reserved executable or authoritative concept keys when
  hidden in `extensions`;
- privacy-sensitive field names forbidden by the pack.

A same-major future minor remains decodable only when additions stay within the
declared inert `extensions` object. DTOs serialize back to JSON without adding
defaults or translating `snake_case` names.

Other accepted extension members are opaque round-trip data. Contract bindings
and later runtime implementations must never interpret them as tools, actions,
routes, prompts, code, or other executable instructions.

`validate_contract_pack()` verifies the vendored hashes and runs all 89
positive and 62 negative golden cases. This is a consumer conformance check,
not a second contract-authoring validator.

## Port boundary

`thine_harness.contracts.ports` defines behavior-free Protocols for invocation,
Working Memory, transcripts, actions, chat, Home, interactions,
notifications, schedules, speakers, preferences/recovery, control, and the
eventual loopback dashboard. Implementations belong to later owner tickets.

`thine_harness.contracts.tool_metadata` contains only concise discovery text
for the nine approved deferred namespaces. It registers no core tool, handler,
database access, transport, scheduler, or UI behavior, so the accepted
GPT-5.6 SOL route, prompt-cache prefix, and Stop Hook remain unchanged.

## Verification

Run the focused consumer suite through the repository wrapper:

```bash
scripts/run_tests.sh tests/thine_harness/test_contract_conformance_v1.py -q
```

Run the complete existing Harness regression suite before integration:

```bash
scripts/run_tests.sh tests/thine_harness -q
```
