# THI3-41 Hermes runtime, cache, and Stop Hook proof

This proof is bound to upstream Hermes commit
`41447a6d7063b2772b0c2f26a5b22d9bd444fb43` and controller packet commit
`1651b810e3e0115a20e21f43f68169346d53381c`. It is a maintained-fork change
under `thine_harness`; it is not a plugin and it adds no Thine database or
product tools.

## Model and provider result

Run from this worktree with its environment activated:

```bash
python -m thine_harness.probe
```

Live result on 2026-08-25:

```text
status=ok
provider=openai-codex
model=gpt-5.6-sol
api_mode=codex_responses
reasoning_effort=medium
context_window_tokens=272000
event_kinds=accepted,started,progress,progress,progress,progress,progress,final
final_marker=HERMES_RUNTIME_OK
usage={input_tokens: 451, output_tokens: 9, cache_read_tokens: 0, cache_write_tokens: 0}
```

The proof reads an existing, unexpired Codex CLI OAuth token through Hermes'
read-only token validator. It writes all Hermes state under a temporary
`HERMES_HOME` and never prints the credential. A fresh Hermes auth store has no
publicly resolved credential (`codex_auth_missing`), so the proof does not
silently fall back or mutate the normal Hermes profile. Direct API-key routing
was not exercised because no `OPENAI_API_KEY` was present; the live route above
is OpenAI's ChatGPT Codex Responses backend. Runtime construction sets no
fallback model and validates provider, model, protocol, reasoning effort, and
context window before admission.

The binding machine-readable selection is
`thine_harness/contracts/runtime-model-v1.json`.

The integrated executable exercises the complete live path on one actual
`AIAgent`:

```bash
python scripts/thi3_41_integrated_live_probe.py
```

It prints one JSON object. The object includes every exact flat `tools` array
observed immediately before the OpenAI SDK's `responses.create(...)`, its
SHA-256, the exact system-prompt character count and SHA-256, the live rough
prefix estimate, and the corresponding `prompt_cache_key` for all primary
iterations and the Stop Hook continuation. No credential or request message
content is printed.

## Prefix and cache invariant

The stable cache identity is the tuple:

```text
(AIAgent session_id, Responses prompt_cache_key, sha256(canonical tool array))
```

Working Memory and the Stop Hook are user-side message content. They may grow
the message suffix but must not change the system instructions, model,
reasoning configuration, session ID, or eager tool array. The Stop Hook uses
the same `AIAgent` instance and its completed conversation history. H9 captures
`CacheIdentity.from_request(...)` from the already-frozen primary request and
passes it to the Stop Hook; identity sampling never rebuilds the system prompt
or request envelope mid-conversation. Session and agent-tool drift are checked
after each continuation and fail closed.

Deterministic transport evidence from the focused test fixture:

```text
primary_prompt_cache_key=pck_2db97656e73e4f1009c7ded4
stop_prompt_cache_key=pck_2db97656e73e4f1009c7ded4
tool_schema_sha256=cb225ea0788c9fbfaad482d5439702738363068990005a23c427346d34e4912f
same_tools=true
reasoning={effort: medium, summary: auto}
```

A live integrated proof on 2026-08-25 produced:

```text
primary_final=THI3_41_INTEGRATED_OK
deferred_catalog_listing=off
primary_prompt_cache_key=pck_f7ffff828578ffd3dd372dfe (4/4 requests)
stop_hook_prompt_cache_key=pck_f7ffff828578ffd3dd372dfe (1/1 request)
primary_tool_schema_sha256=4707333416dce208c950c0cccf929e4da3691a6cc68e3c29c6f141e1192f0268 (4/4)
stop_hook_tool_schema_sha256=4707333416dce208c950c0cccf929e4da3691a6cc68e3c29c6f141e1192f0268
wire_tools=[tool_search,tool_describe,tool_call]
system_prompt_chars=[9214,9214,9214,9214,9214]
system_prompt_sha256=f65ed59f95922ea3a47e1d4ca914d18eb7a327d2750cde683e7781f8a3dbfdb6 (5/5)
fixed_prefix_estimated_tokens=[2627,2627,2627,2627,2627]
fixed_prefix_reserve_tokens=4096
fixed_prefix_within_reserve=true
same_prompt_cache_key=true
same_system_prompt_sha256=true
same_tool_schema_sha256=true
same_wire_tool_array=true
primary_usage={input_tokens: 3633, output_tokens: 90, cache_read_tokens: 6144, cache_write_tokens: 0}
stop_hook_outcome=unchanged
stop_hook_usage_delta={input_tokens: 1124, output_tokens: 33, cache_read_tokens: 1536, cache_write_tokens: 0}
deferred_result_sequence=[tool_search,tool_describe,thine_transcripts_probe_lookup]
helper_calls=[{sequence:41}]
helper_calls_during_stop_hook=0
memory_commits=0
unchanged_markers=1
```

The helper's full schema is absent from all wire arrays; Hermes unwraps
`tool_call` to the helper's real name before appending the canonical tool
result. This proves live search, description, and bridge execution rather than
an eager direct helper call. The same run shows an actual provider cache read
on the Stop Hook continuation, an identical cache envelope, no handler
execution during the hook, and no new memory version when nothing worth
remembering changed. Cache-read counts are evidence, not an acceptance
precondition; a cold primary with zero cache read remains valid.

## Deferred namespaced helper result

`DeferredNamespaceCatalog` delegates to Hermes' existing progressive tool
disclosure. A registered `thine_transcripts_lookup` helper is absent from the
eager request, while the eager tool array contains only `tool_search`,
`tool_describe`, and `tool_call`. Searching for canonical transcript sequence
returns the flat helper name with logical namespace `transcripts`; only
`describe` reveals its full parameter schema. Other reserved logical
namespaces are `working_memory`, `communications`, `permissions`, `schedules`,
`speakers`, `ui.state`, `topics`, and `run`.

## Working Memory and Stop Hook result

- The intended Working Memory contract remains a hard 16,000 configured-model
  token ceiling, but Hermes does not ship an exact tokenizer for
  `openai-codex/gpt-5.6-sol`. This is frozen as an unresolved tokenizer
  limitation: changed memory writes fail closed unless an exact
  configured-model counter is supplied, and `token_count` is nullable for an
  unmeasured existing snapshot. UTF-8 bytes are never stored or reported as
  tokens.
- A 16,000-byte ceiling remains only as a fail-closed auxiliary guard. It can
  request compaction early, but it does not satisfy or impersonate the token
  contract.
- An oversized proposal receives exactly one same-context, agent-directed
  correction and the corrected document must be at most 14,000 guarded tokens.
  Failure preserves the previous version.
- A normal hook writes a new version only for a changed, worth-remembering
  proposal. Otherwise it records `mark_unchanged`.
- Interrupted work skips the Stop Hook and performs neither commit nor
  unchanged-marker write.
- The Stop Hook accepts only the exact keys `worth_remembering` and `markdown`.
  `true` requires a string; `false` requires null. It never creates a separate
  model/provider session.

The frozen envelope is:

```text
provider context                         272000
system + eager bridge prefix reserve       4096
Working Memory reserve                    16000
output + reasoning reserve                 32768
measured residual                         219136
unallocated safety margin                  19136
absolute transcript ceiling               200000
routine transcript batch target             8000
```

The contract reserves 4,096 tokens for the system prompt and eager bridge
schemas. Each integrated run records Hermes'
`estimate_request_tokens_rough` result over its actual final outbound system
prompt and exact three eager bridge schemas, and accepts the run only when
every observation is at most that reserve. The observed value is evidence for
that run, not a cross-run fixture: randomized temporary profile paths may
legitimately change the prompt length. Full helper schemas are deferred. The
200,000-token absolute limit is 19,136 tokens below the 219,136-token residual;
the normal architecture target remains the earlier of 8,000 transcript tokens
or 10 minutes of audio. The binding fixture is
`thine_harness/contracts/runtime-envelope-v1.json`.

## Progress and cancellation trace

Foreground lifecycle is `accepted -> started -> zero or more ephemeral
progress -> one non-ephemeral final|failed|interrupted`. Session adapters may
emit only progress; the runtime owns terminal events. Hermes' raw `completed`
and `failed` flags are mapped directly. A failed or incomplete turn never emits
`final`, even if Hermes includes failure prose in `final_response`.

The public cancellation and resume reference-spike test trace is:

```text
accepted -> started -> quiet background work -> cancel(p0_user_tick)
-> wait for active tool result append+persistence -> interrupted(p0_user_tick)
resume_token=checkpoint:run-background-1
checkpoint={logical_run_id,input_prompt,remaining_work,context_messages,
            completed_tool_results,successful_action_receipts,
            partial_visible_assistant_output,updated_at}
P0 accepted -> P0 final -> resume(checkpoint) -> same Logical Run final
```

Background admission requires both a durable `resume_token` and a
`BackgroundCheckpointStorePort` before the provider is called. Background
progress deltas are consumed quietly; only P0 chat emits ephemeral progress.
Cancellation delivery is serialized with unbind at turn completion. A
completed turn clears any cancellation that won the final snapshot race, so a
late cancel cannot poison the next invocation on the reused agent. During an active tool call it is
deferred until Hermes clears its tool-execution fence, which happens after the
canonical result is appended and persisted; only then may the model loop be
interrupted. The runtime records every durable tool result, but promotes only
results explicitly marked `effect_disposition=applied` to successful action
receipts. Its continuation prompt says the original input is already present
and forbids repeating completed effects. `resume()` starts a new invocation of
the same Logical Run. The Stop Hook skip path
performs no memory write. `thine_harness/runtime.py` is the THI3-41 executable
reference spike and temporarily co-locates H3 coordinator behavior with H9
adapter behavior; it is not an exclusive production assignment. H3's future
production implementation owns the coordinator, requeue, checkpoint
implementation, and resume lookup. H9 owns AIAgent mapping, progress,
cancellation, and runtime selection. H3 must requeue the same Logical Run at
the front after P0 and its finalizer complete; hidden reasoning is not part of
the resume contract. Partial visible prose is retained for UI/reference
continuity, but is not an action receipt or execution authority.

## Contract handoff to THI3-43

THI3-43 should consume these immutable v1 inputs:

- `runtime-model-v1.json` for exact runtime diagnostics and fail-closed
  selection;
- `runtime-envelope-v1.json` for transcript segmentation limits;
- `hermes-ownership-v1.json` for H3-H10 production ownership, the temporary
  `runtime.py` reference-spike split, and non-exclusive shared core touchpoints;
- public Python ports in `runtime.py`, `working_memory.py`, and
  `deferred_tools.py` as the source behavior for neutral DTO/golden fixtures.

The language-neutral invocation schema must preserve `logical_run_id`, kind,
prompt, durable resume token, accepted/started/progress/final/failed/interrupted event
kinds, ephemeral flag, context messages, usage telemetry, and the stable model
diagnostics. Its checkpoint schema must preserve `resume_token`,
`logical_run_id`, `input_prompt`, `remaining_work`, `context_messages`,
`completed_tool_results`, `successful_action_receipts`,
`partial_visible_assistant_output`, and UTC `updated_at`; H3 owns the production
coordinator, requeue, checkpoint implementation, and resume lookup, while H9
owns AIAgent mapping, progress, cancellation, and runtime selection. The
finalizer schema must preserve
the cache-identity tuple,
memory version, outcome (`committed`, `unchanged`, or `skipped_interrupted`),
nullable exact-model token count, unresolved-tokenizer status, intended
16K/14K limits, and hook-only recovery. THI3-43 must not treat UTF-8 bytes as a
token count. Tool contracts must keep
flat OpenAI-safe names on the wire and carry logical namespace as metadata.

Do not reuse Hermes cron or its existing memory plugin as the stage-one
implementation. H7 owns a new one-shot scheduler module. H4 extends the
maintained-fork Working Memory/finalizer seam here. H9 extends the AIAgent and
progress adapter here; it must pass the captured primary request cache identity
and must not introduce a second agent loop.

Existing Hermes core integration points are deliberately non-exclusive: H9's
outbound observation uses `agent/codex_runtime.py` and
`agent/outbound_request_scope.py`; isolation and safe-boundary cancellation use
`agent/agent_init.py` and `run_agent.py`; Stop Hook tool denial uses
`agent/tool_execution_scope.py`, `model_tools.py`, and `tools/registry.py`.
These are shared core seams, not files assigned exclusively to H3 or H9.
