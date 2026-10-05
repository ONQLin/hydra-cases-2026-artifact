# Continuous batching and host offload validation

`vllm_latest` now executes real model iterations and can refill a vacant slot
before another request finishes. The paper `vllm` policy is preserved through a
frozen base; the previous latest emulation remains available as `vllm_legacy`.
See [runtime contracts](../../../../docs/agent_workloads/scheduling/runtime_evolution/README.md).

These are **synthetic mechanism checks** on `gqa-prefix-fixture` and one archived
hardware point. They are not BFCL task scores or calibrated serving performance.
No recurrent checkpoint restoration is implemented.

## Reproduce

From the repository root, use a fresh output directory for each run:

```bash
# Unit checks; external Garnet tests remain environment-gated.
python -m unittest discover -s tests

# Eight points: batch-one equivalence, late refill, and host capacity/BW.
python tests/validation/agent_workloads/study_continuous.py \
  --output-dir output_sanity_checks/continuous_repro/study

# Optional: all three backends, with host CPU tools and prefix DMA.
# The study above generated these inputs; no archived results are needed.
python tests/validation/agent_workloads/validate.py \
  --trace output_sanity_checks/continuous_repro/study/inputs/workload.json \
  --tool-profiles-file output_sanity_checks/continuous_repro/study/inputs/host_tools.json \
  --output-dir output_sanity_checks/continuous_repro/three \
  --models gqa-prefix-fixture --backends hydra_sim hydra_packet chipsim_contended \
  --scheduler vllm_latest --batch-size 4 --batching timeout \
  --max-batch-wait-s 0.0002 --prefix-cache lru_offload \
  --prefix-capacity-bytes 49152 --sessions 12 --time-limit 1 --stop-when-complete

# Independent physical allocation audit using that run's exact command.
python tests/validation/operator_validation/audit_memory.py \
  --command output_sanity_checks/continuous_repro/three/gqa-prefix-fixture/hydra_sim/command.json \
  --output-dir output_sanity_checks/continuous_repro/memory
```

Packet/CHIPSIM require their existing [build/runtime setup](../../../../integrations/chipsim/README.md).
CHIPSIM uses a local Unix socket. Native configuration selects
`--cluster-config.local-scheduler vllm_latest`; the existing
`--cluster-config.agent-scheduler.*` options control priority, timeout and caches.
Use `max-active-batches` 0 or 1. `batch-size` bounds resident requests.

## Evidence

The script generates single-request, late-arrival and host-offload workloads
from Python definitions. Assertions check batch-one service equivalence, slot
refill and heterogeneous decode membership. `AgentReplayValidation` checks
call/token counts, dependencies, iteration progression and memory drainage.
Summaries, commands and call/iteration ledgers are written under `--output-dir`;
none are required as committed inputs. The observations below summarize prior
runs with the same generators.

- **Legacy isolation:** the frozen base equals the original latest source after
  undoing its class/name rename; paper scheduler source differs only in its base
  import/name. Factory/inheritance checks pass. This is not a rerun of the full
  paper experiments.
- **Exact execution:** each output has one complete iteration, with correct
  stage/context cursor, call timing and tool dependencies. No virtual tokens.
- **Batch size 1:** both executors take 400 us of model service for 8 outputs.
  Total latency is 600 us for `agent`, 500 us for latest, because dispatch differs.
- **Late refill:** the short call finishes at 264 us; the late call starts at
  310 us, joins a heterogeneous decode iteration, and finishes at 528 us while
  the original long call continues to 1116 us. Four calls produce exactly 25
  tokens. A plain heterogeneous CSV also completes 4 requests / 25 tokens.
- **Memory:** private state/reservations, cache pins, restore leases, queued tools
  and DMA drain. Retained HBM equals weights plus cached prefixes.
- **Unit checks:** `tests.test_continuous_batching` verifies state ownership,
  admission and legacy isolation. Real Garnet unit tests are optional; the
  combined backend command above exercises CHIPSIM directly.

| Backend | Calls / output tokens | Prefix hits / host restores | TP (tokens/s) | Process wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| Analytic | 36 / 240 | 24 / 6 | 29,680.9 | 1.77 |
| Packet | 36 / 240 | 24 / 6 | 25,295.1 | 1.82 |
| CHIPSIM | 36 / 240 | 24 / 6 | 25,295.1 | 9.23 |

This tiny run checks backend continuity, not general accuracy. Wall time includes
process startup. The host has 256 GiB DRAM, with 4 GiB reserved for tool workspace;
48 KiB HBM holds one prefix. The shared link uses assumed 25 GB/s and 10 us startup.

In the eight-point study, no CPU tool profile is attached, so host capacity is
entirely available to KV. HBM-only gives 28,564.6 tokens/s; fast offload gives
29,680.9; reducing the link to 0.01 GB/s gives 8,599.1; limiting host KV to one
48 KiB entry gives 28,523.9. This verifies that transfer delay and capacity affect
results rather than providing free cache hits. Costs are assumptions.

The new variant uses homogeneous separate prefill and longest-context **padded
decode**, with actual per-request KV capacity. It does not model exact ragged
kernels, chunked prefill, active-KV swapping or online paged admission. Hybrid
prefix/checkpoint restoration remains deferred. Arrival and restore readiness
are polled at the existing clock; iteration boundaries wake scheduling directly.
