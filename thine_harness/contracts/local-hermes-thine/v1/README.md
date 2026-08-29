# Local Hermes-Controlled Thine contract pack v1

This directory is the language-neutral wire authority shared by the Hermes
fork, the local Thine backend, and the iOS app. The JSON payloads use only JSON
primitives, stable `snake_case` names, explicit nulls, closed enums, and integer
epoch milliseconds within the interoperable JSON safe range. Larger exact
integers use canonical decimal strings. Python, TypeScript, and Swift bindings consume these files;
none of those languages owns the wire shape.

## Entry points

- `manifest.json` is the machine-readable index. A consumer starts here and
  resolves a type ID to one exact JSON Schema definition.
- `schemas/*.schema.json` contains strict JSON Schema 2020-12 definitions.
- `fixtures/valid/*.json` and `fixtures/invalid/*.json` are unchanged golden
  bundles for all three language-specific conformance tickets.
- `metadata/owners.json` separates authoritative state from derived read models.
- `metadata/serialization-map.json` assigns every binding and records shared
  integration touchpoints as non-exclusive.
- `metadata/state-machines.json` freezes queue, Attempt, preemption, Quarantine,
  retry, and half-hour interaction-clock semantics.
- `metadata/compatibility.json` is the version and migration policy.
- `metadata/source-locks.json` binds this pack to the accepted planning and spike
  commits.
- `metadata/home-registry.json`, `topology.json`, and
  `privacy-and-limits.json` freeze the mobile boundary, local-only deployment,
  redaction, and capacity guards.

Run the complete neutral validation from the repository root:

```bash
python3 -m tools.contract_validator contracts/local-hermes-thine/v1
python3 -m unittest discover -s tests -v
```

The validator is dependency-free. It checks schema conformance, expected
negative fixtures, complete fixture coverage, cross-file references, invocation
lifecycle correlation, unique identities, owner and serialization coverage, state-machine transitions,
priority semantics, source locks, compatibility examples, version/schema-ID
provenance, portable JSON equality and numbers, supported JSON Schema keywords,
manifest path confinement (including symlinks), exact redaction and size constraints,
Meeting Mode exclusion, and the local
fail-closed topology.

## Propagating a corrected snapshot

`metadata/consumer-propagation.json` is the canonical propagation manifest for
the backend, Hermes fork, and mobile app. From a clean consumer feature branch,
set `controller_root` to the controller checkout containing the accepted pack
commit and `consumer_root` to that consumer checkout. Replace the consumer's
destination tree with the controller tree using that consumer's exact
`propagation_command`, then run the listed generation and verification
commands. Backend and Hermes provenance record the controller `HEAD` plus every
path-sorted SHA-256 file digest. Mobile provenance records the same controller
`HEAD` plus the normalized digest over that path-sorted hash stream; the helper
also mirrors all eight mobile fixture suites. Mobile then updates its listed
handwritten closed-enum binding. A partial copy, a stale provenance digest, or
an omitted fixture mirror is not a valid propagation.

The helper also rewrites and verifies the provenance assertions declared in
the manifest. Backend pins the propagated controller commit in its Dataplane
conformance test. Hermes pins that commit plus the valid and invalid fixture
counts derived from the propagated suites, so adding golden cases cannot leave
its test expectations stale. Mobile likewise updates and verifies the valid and
invalid Swift-assigned fixture counts, derived from the serialization map rather
than from all language-neutral golden cases.

This is a coordinated correction to the not-yet-independently-negotiated v1
snapshot, not a compatible minor addition. All three consumers must take the
same corrected controller commit before the changed closed enums are used.

From the controller repository root, the exact copy/provenance/fixture step is:

```bash
controller_contract_commit=$(git rev-parse HEAD)
python3 -m tools.contract_pack_propagation backend --consumer-root "$backend_root" --controller-commit "$controller_contract_commit" --apply
python3 -m tools.contract_pack_propagation hermes --consumer-root "$hermes_root" --controller-commit "$controller_contract_commit" --apply
python3 -m tools.contract_pack_propagation mobile --consumer-root "$mobile_root" --controller-commit "$controller_contract_commit" --apply
```

Run each command again without `--apply` for a read-only byte, provenance, and
mobile-fixture verification. Then make the manifest's exact mobile binding
changes, run every listed generator and verification command, and commit each
consumer separately. The mobile conformance fixtures make an omitted closed
enum or the old `submitting` cancellability behavior fail at the public decoder
seam.

The public boundary includes the durable invocation request/event stream,
transcript claim/lease-renewal/reclaim/continuation and quarantined input-gap
records, separate interaction ingestion and processing-cursor receipts, complete
chat/action receipt correlations, revision-bound reset confirmation, and the
non-disableable proactive-chat capability. `context_messages` is a closed
visible `user | assistant | tool` union; system messages and hidden reasoning
are not wire data.

## Compatibility

The current version is `1.0`.

- An unknown major version is rejected.
- A future minor version is accepted only when its additions are inside the
  declared `extensions` object. Unknown top-level fields remain rejected.
- Authoritative command enums are closed. An unknown enum is never interpreted
  as a no-op or silently mapped to a default.
- Removing a field, changing meaning, adding a required field, or adding an
  unnegotiated closed-enum value requires a major bump.
- Adding an optional field inside the declared extension point requires a minor
  bump.
- Extension objects are open for storage compatibility but inert: they cannot
  carry commands, tool calls, navigation, code, or other executable meaning.
- Duplicate object members, non-`snake_case` declared wire properties, and
  namespace-local stable-ID collisions with different payloads are rejected.

## Consumer boundary

The next three tickets consume this directory without modifying it:

- THI3-44: Python DTOs/ports and conformance tests in the Hermes fork.
- THI3-45: TypeScript DTOs/ports and conformance tests in the backend.
- THI3-46: strict Swift DTOs with explicit `CodingKeys` and conformance tests in
  the mobile repository.

Those tickets implement decoders only. Runtime, persistence, Dataplane routing,
Home rendering, and transport behavior belong to later repository-owned work.
