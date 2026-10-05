# BFCL workload integration

This offline adapter imports **generated BFCL `multi_turn_base` result JSONL**.
Raw task questions and possible-answer files are not execution traces and are
rejected. The simulator needs no BFCL installation, model weights or API access.
Both a synthetic mechanism fixture and a pinned real Qwen3-8B result excerpt
are included. See the [real-log audit and replay](../../tests/validation/agent_workloads/real_bfcl/README.md).

[Architecture and benchmark choices](../../docs/agent_workloads/README.md) ·
[Validation evidence](../../tests/validation/agent_workloads/README.md)

## 1. Reproduce the offline smoke

From the repository root, with the HYDRA Python environment active:

```bash
# Prepare once and reuse this directory across validation commands.
python tests/validation/prepare_inputs.py \
  --output-dir output_sanity_checks/validation_inputs --include-datasets

python -m integrations.workloads.prepare \
  --input output_sanity_checks/validation_inputs/bfcl_format.jsonl \
  --provenance output_sanity_checks/validation_inputs/provenance.json \
  --output output_sanity_checks/agent_demo/trace.json \
  --tool-latency-s 0.002 --session-interval-s 0 --user-think-s 0.001

python tests/validation/agent_workloads/validate.py \
  --trace output_sanity_checks/agent_demo/trace.json \
  --output-dir output_sanity_checks/agent_demo/analytic \
  --models qwen3-8b --backends hydra_sim --sessions 2 --time-limit 10
```

For the small three-backend execution test, first complete the
[Packet](../packet/README.md) and [CHIPSIM](../chipsim/README.md) setup:

```bash
python tests/validation/agent_workloads/validate.py \
  --trace output_sanity_checks/agent_demo/trace.json \
  --output-dir output_sanity_checks/agent_demo/three_backends \
  --models deepseek-decoder-fixture \
  --backends hydra_sim hydra_packet chipsim_contended \
  --sessions 2 --time-limit 0.1
```

Each command requires a fresh output path. The runner selects an existing
archived hardware point and uses static mapping, static scheduling and batch
size 1. It checks identical setup/trajectory inputs across backends, exact call
and token completion, and predecessor/tool-wait dependencies. It fails if the
fixed simulation window is too short; increase `--time-limit`. `--sessions`
selects the first N sessions, not N individual LLM calls.

## 2. Obtain real BFCL trajectories

Existing results are available from
[HuanzhiMao/BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result).
The real-log guide above provides a hash-checked download and complete replay
without collecting new model responses. The following collection instructions
remain useful for a custom model or serving configuration.

The inspected upstream revision is
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. Its
[handler](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py)
records `input_token_count[turn][step]`, `output_token_count[turn][step]`, and
`inference_log` with `begin_of_turn_query`, `step_0`, `step_1`, etc. State snapshots
between turns are ignored for timing. Actual tool-role entries determine tool
execution count; generated calls that were never executed are not timed as tools.

In a separate BFCL environment, follow the pinned upstream installation and
model-serving instructions. A collection invocation is:

```bash
bfcl generate --model MODEL_NAME --test-category multi_turn_base \
  --include-input-log --num-threads 1
```

Use the upstream `--run-ids` mechanism to select a small, fixed subset, and run
`bfcl evaluate` separately when task accuracy is wanted. No collection is
performed by the HYDRA commands above. See the upstream
[README](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/README.md)
and [log guide](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/LOG_GUIDE.md).

Create `collection.json` with actual collection information:

```json
{
  "benchmark_revision": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
  "source_model": "MODEL_NAME and checkpoint revision",
  "tokenizer": "tokenizer repository and revision",
  "chat_template": "template revision or hash and BFCL handler name",
  "sampling": {"temperature": 0, "reasoning_mode": "record the actual setting"}
}
```

Then replace `--input` and `--provenance` in step 1 with the result JSONL and
this manifest. The importer hashes the source, keeps task/step IDs and handler
events, and labels the token source as reported usage. It does not retokenize.
Unknown/zero usage and mismatched step arrays fail explicitly; no tasks or
token lengths are silently discarded to make a run fit.

Tool timing is **required**: use `--tool-latency-s` for an explicit per-tool
assumption, or `--tool-latencies timings.json` for recorded durations in seconds:

```json
{
  "multi_turn_base_0": {
    "turn_0.step_0": [0.012, 0.008],
    "turn_1.step_0": [0.004]
  }
}
```

The sidecar must cover every tool-bearing step exactly, with one duration per
tool entry in log order. Values are summed for the pinned serial executor.
`--user-think-s` adds a delay before later user turns. `--session-interval-s`
sets initial session spacing; dependent call arrivals come from simulation.
Sub-microsecond waits round up to HYDRA's one-microsecond tick.

## 3. Inspect the outputs

Each simulation directory contains:

- `effective_agent_workload.json`: selected sessions, dependencies and provenance.
- `effective_workload.csv`: all planned per-call token lengths, in session order;
  this file alone does not encode release times or dependencies.
- `agent_metrics.json`: released calls, first-token/completion timestamps,
  queue-inclusive TTFT, session latencies, and planned/released/completed counts.
- `metrics.json`: existing serving metrics over the fixed window.
- `config.json`, `system_snapshot.json`: hardware/runtime configuration.
- `effective_tool_config.json`: tool-profile content/hash and source/target model identity check.

An output length of O maps to one prefill token plus O−1 decode iterations.
Output length 1 is supported. No context truncation or prompt/decode scaling is
performed. This baseline recomputes full prompts. Each full prompt must fit the target model's
supported context limit. Token counts from another model remain a workload
assumption, not tokenizer validation for the simulated model.

## Extension points and current limits

- `BaseWorkloadImporter` → `BFCLTraceImporter`: offline benchmark normalization.
- `AgentWorkload`, `AgentSession`, `AgentStep`: validated serial trajectory data.
- `BaseRequestSource` → `TraceRequestGenerator` / `AgentRequestGenerator`: factory
  preserving independent CSV replay as the default.
- `AgentRequest`: completion event after request resource cleanup.
- `AgentReplayValidation`: workload-specific subclass of the existing model runner.

The main CLI can select this source directly through
`--workload-config.request-generator-config.generator agent` and
`--workload-config.request-generator-config.agent-trace-file PATH`.
The nested `agent-config` dataclass accepts `input-format` (`normalized` or `bfcl`),
`provenance-file`, `task-ids`, `tool-profiles-file`, fixed/recorded tool delays,
raw-log arrival/think times, `allow-model-mismatch`, and `stop-when-complete`.
These are native Tyro arguments and `AgentTraceConfig` members in Python.
The normal CSV trace path and static interval are unused in agent mode.

Raw logs require explicit delays or a tool profile. Normalized logs already
contain arrivals/delays; conflicting import-time overrides fail. Task selection
preserves recorded timestamps. If `metadata.collection.source_model_id` is
declared, mismatches with the target model fail unless explicitly permitted for
a counterfactual replay. Missing identity remains labeled unverified.

`stop-when-complete` ends after every selected session's final tool operation.
The time limit still guards incomplete runs; throughput then uses the elapsed
burst window. Compare runs using the same termination policy.

Agent replay requires the existing `static` scheduler with batch size 1, static
mapping and pipeline execution. Sessions may overlap, but calls within a session
are serial. KV/recurrent state is released between calls. CPU profile workers
and workspace contend, while external service/NIC capacity is not modeled.
There is no live agent execution, semantic evaluation, tool-code execution or
general fork/join scheduling. AppWorld and MetaTool adapters remain future work.

## Tool profiles and custom inputs

`BaseToolModel` has `ExternalToolModel` and `CPUProfileToolModel` subclasses.
Profiles select an implementation per tool name, with an optional `default`.
Missing profiles fail before simulation. Deployment location is never inferred
from a function name. See [complete profile examples](../../tests/validation/agent_workloads/real_bfcl/README.md#tool-profiles).

| Kind | Timing/resource contract |
|---|---|
| `external` | Request serialization + service + RTT + response serialization. Independent sessions overlap; no local CPU occupancy. |
| `cpu_profile` | Request serialization + worker/workspace queue + service + RTT + response serialization. Finite shared host worker/memory pools, released after service. |

Bandwidth is decimal **GB/s**; serialization uses `bytes / bandwidth`, rounded
up to a simulation tick. `service_s` excludes separately specified transfer and
RTT. If a measurement already includes end-to-end network time, omit bandwidth
and set RTT to zero to avoid double counting. CPU profiles name their target
machine; no automatic scaling across CPU architectures is performed. Workspace
occupies host memory, independently of accelerator HBM/KV.

BFCL payload bytes are UTF-8 lengths of decoded call expressions and returned
text. These are **proxies**: no HTTP/TLS framing, tensor conversion, shared NIC
contention or accelerator-to-host DMA is inferred. Custom normalized inputs can
supply measured bytes. Tool transport uses shared analytical logic across all
fidelity backends, not additional Garnet traffic. A placed CPU chiplet requires
another backend extension and calibrated instruction/memory workloads.

With profiles, normalized steps must have zero `tool_delay_s` to prevent double
counting. `agent_metrics.json.tool_execution` records individual tool payloads,
service/queue/transfer timestamps and host resource state at cutoff.

Any user or benchmark adapter can emit the following normalized format, then
select a model with the usual `--workload-config.model` argument:

```json
{
  "schema_version": 2, "cache_policy": "runtime",
  "metadata": {"collection": {"source_model_id": "Qwen/Qwen3-8B", "description": "Illustrative custom trace"}},
  "sessions": [{
    "session_id": "custom_0", "arrival_time_s": 0,
    "steps": [{
      "step_id": "0", "input_tokens": 1024, "output_tokens": 32,
      "tool_calls": 1, "tool_delay_s": 0,
      "tools": [{"name": "search_api", "request_bytes": 128, "response_bytes": 4096}]
    }]
  }]
}
```

Use `input-format normalized` and a matching tool profile; no BFCL fields are
required. Add subsequent steps for follow-up generations. The single-step
example includes completion of its tool even without another LLM call.

Schema 2 stores full logical prompt lengths. This replay baseline recomputes
all prompts. Legacy schema 1 (`cache_policy: recompute`) still loads and is
exported as schema 2. Optional fields `slo_s`, `estimated_service_s`, `prefix_id`
and `prefix_tokens` are preserved as metadata for future runtime policies;
this baseline does not use them for prioritization or prefix reuse. Prefix IDs
must identify identical tokenized prefixes; the pinned BFCL logs do not
establish those identities.
