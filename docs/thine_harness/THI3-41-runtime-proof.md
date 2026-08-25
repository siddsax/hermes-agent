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

A live two-turn same-agent proof used a deliberately cacheable stable prefix:

```text
primary_final=HERMES_PRIMARY_OK
primary_usage={input_tokens: 2473, output_tokens: 9, cache_read_tokens: 0, cache_write_tokens: 0}
stop_hook_outcome=unchanged
stop_hook_usage={input_tokens: 3235, output_tokens: 25, cache_read_tokens: 1792, cache_write_tokens: 0}
same_cache_identity=true
prompt_cache_key=pck_5b5f1a202382b09edef5f306
memory_commits=0
unchanged_markers=1
```

This shows an actual provider cache read on the Stop Hook continuation and no
new memory version when nothing worth remembering changed.

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

- Working Memory has a hard 16,000-token ceiling. Core Hermes does not ship the
  configured GPT tokenizer, so the production default uses UTF-8 bytes as a
  conservative upper bound for its byte-level BPE tokens. A configured exact
  tokenizer may be injected, but a rough estimator is never the hard guard.
- An oversized proposal receives exactly one same-context, agent-directed
  correction and the corrected document must be at most 14,000 guarded tokens.
  Failure preserves the previous version.
- A normal hook writes a new version only for a changed, worth-remembering
  proposal. Otherwise it records `mark_unchanged`.
- Interrupted work skips the Stop Hook and performs neither commit nor
  unchanged-marker write.
- The Stop Hook accepts only the exact structured decision shape and never
  creates a separate model/provider session.

The frozen envelope is:

```text
provider context                         272000
measured stable system + 3 bridge tools    791
Working Memory reserve                    16000
output + reasoning reserve                 32768
measured residual                         222441
unallocated safety margin                  22441
absolute transcript ceiling               200000
routine transcript batch target             8000
```

The 791-token prefix is Hermes'
`estimate_request_tokens_rough` measurement for the stable proof system prompt
and the three eager bridge schemas. Full helper schemas are deferred. The
200,000-token absolute limit is intentionally below the measured residual; the
normal architecture target remains the earlier of 8,000 transcript tokens or
10 minutes of audio. The binding fixture is
`thine_harness/contracts/runtime-envelope-v1.json`.

## Progress and cancellation trace

Foreground lifecycle is `accepted -> started -> zero or more ephemeral
progress -> one non-ephemeral final`. Session adapters may emit only progress;
the runtime owns terminal events.

The public cancellation and resume test trace is:

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
Cancellation is unbound at turn completion. During an active tool call it is
deferred until Hermes clears its tool-execution fence, which happens after the
canonical result is appended and persisted; only then may the model loop be
interrupted. The runtime records every durable tool result, but promotes only
results explicitly marked `effect_disposition=applied` to successful action
receipts. Its continuation prompt says the original input is already present
and forbids repeating completed effects. `resume()` starts a new invocation of
the same Logical Run. The Stop Hook skip path
performs no memory write. H3 must back the checkpoint port durably and requeue
the same Logical Run at the front after P0 and its finalizer complete; hidden
reasoning is not part of the resume contract. Partial visible prose is retained
for UI/reference continuity, but is not an action receipt or execution authority.

## Contract handoff to THI3-43

THI3-43 should consume these immutable v1 inputs:

- `runtime-model-v1.json` for exact runtime diagnostics and fail-closed
  selection;
- `runtime-envelope-v1.json` for transcript segmentation limits;
- `hermes-ownership-v1.json` for H3-H10 module and serialization ownership;
- public Python ports in `runtime.py`, `working_memory.py`, and
  `deferred_tools.py` as the source behavior for neutral DTO/golden fixtures.

The language-neutral invocation schema must preserve `logical_run_id`, kind,
prompt, durable resume token, accepted/started/progress/final/interrupted event
kinds, ephemeral flag, context messages, usage telemetry, and the stable model
diagnostics. Its checkpoint schema must preserve `resume_token`,
`logical_run_id`, `input_prompt`, `remaining_work`, `context_messages`,
`completed_tool_results`, `successful_action_receipts`,
`partial_visible_assistant_output`, and UTC `updated_at`; H3 owns durable
persistence and H9 owns the AIAgent mapping. The finalizer schema must preserve
the cache-identity tuple,
memory version, outcome (`committed`, `unchanged`, or `skipped_interrupted`),
token count, 16K/14K limits, and hook-only recovery. Tool contracts must keep
flat OpenAI-safe names on the wire and carry logical namespace as metadata.

Do not reuse Hermes cron or its existing memory plugin as the stage-one
implementation. H7 owns a new one-shot scheduler module. H4 extends the
maintained-fork Working Memory/finalizer seam here. H9 extends the AIAgent and
progress adapter here; it must pass the captured primary request cache identity
and must not introduce a second agent loop.
