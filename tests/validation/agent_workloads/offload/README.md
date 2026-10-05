# Host DRAM prefix offload validation

The baseline adds an inclusive host tier to the existing dense prefix cache.
It writes completed prefixes through asynchronously and restores host hits before
admission. Host capacity defaults to **256 GiB**, with an explicit reserved tool
workspace partition. It does not swap active requests or simulate a placed CPU.
See [implementation scope and planned runtime stages](../../../../docs/agent_workloads/scheduling/runtime_evolution/README.md).

## Run the capacity and bandwidth study

From the repository root with the HYDRA environment active:

```bash
python tests/validation/agent_workloads/study_offload.py \
  --output-dir output_sanity_checks/agent_offload/study
```

The OOP runner uses the existing hardware selection and `AgentReplayValidation`.
Its synthetic trace has 12 sessions, 36 LLM calls, 240 output tokens and three
declared prefixes. HBM holds one 48 KiB prefix. Two assumed CPU workers each use
64 MiB workspace per tool; the host reserves 4 GiB total for tool workspace.
Twelve of the 24 tool calls encountered CPU queueing in the fast-offload run.
The large host leaves **252 GiB for KV**; only 144 KiB is needed by this fixture.

All configurations use the same trace, tool profiles, hardware, FCFS priority,
max batch size 4, timeout collection, and one active batch. Results are controlled
sensitivity evidence, not measured BFCL or CPU/PCIe performance.

| Configuration | TP (tok/s) | Mean session (ms) | Prefix hits | Host restores | Host evictions |
|---|---:|---:|---:|---:|---:|
| HBM-only LRU | 25,284.5 | 7.868 | 0 | 0 | 0 |
| Offload, 256 GiB host, 25 GB/s link | 25,762.1 | 7.765 | 24 | 6 | 0 |
| Offload, 256 GiB host, 0.01 GB/s link | 8,271.3 | 20.890 | 12 | 2 | 0 |
| Offload, host KV partition limited to 48 KiB | 25,252.5 | 7.868 | 0 | 0 | 8 |

The fast host tier preserves three prefixes while HBM cycles through one. Slow
copies pin HBM longer and delay restores, so fewer useful hits occur and the
extra traffic hurts performance. The limited host tier thrashes. Its final
background write adds 12 microseconds to the run window without extending any
session's completion time. The baseline always restores an available host hit;
there is no cost-aware restore-versus-recompute selector.

The shared transfer model uses `startup + bytes/min(link BW, DRAM BW)`, with
100 GB/s host DRAM and 10 us startup in this study. Both directions serialize
on one effective link. These parameters are assumptions and do not model
contention with accelerator HBM ports, Garnet links or tool memory bandwidth.

## Generated inputs and checks

The study generates the workload and assumed CPU tool profile in `--output-dir`.
To generate only the inputs for a custom validation run:

```bash
python tests/validation/agent_workloads/study_offload.py \
  --output-dir output_sanity_checks/agent_offload/inputs --prepare-only
```

This writes `workload.json` and `host_tools.json` without running a simulation.
The optional [combined backend check](../continuous/README.md) reuses these
inputs to cover continuous batching, tools and offload in one run.

For an independent memory audit:

```bash
python tests/validation/operator_validation/audit_memory.py \
  --command output_sanity_checks/agent_offload/study/offload_fast/gqa-prefix-fixture/hydra_sim/command.json \
  --output-dir output_sanity_checks/agent_offload/memory
```

The audit passed with only weights plus retained HBM KV, host KV within its
partition, and zero request reservations, CPU workspaces, transfer pins or
restore-admission leases at completion. Fixed-cutoff runs instead report pending
transfers and pins explicitly.

## Validation scope

Unit tests cover source/destination pinning, restore-before-compute ordering,
serialized transfer timing, host capacity/LRU pressure and cutoff behavior:

```bash
python -m unittest tests.test_host_prefix_cache
```

The runner verifies completed calls/tokens and drained private memory. Detailed
metrics, commands, transfer records and the study summary are generated under
`--output-dir`; they are not repository fixtures. The table above retains a
compact historical comparison with the same synthetic input generator.

Native Tyro uses `--cluster-config.agent-scheduler.prefix-cache lru_offload` and
the `host-dram-capacity-bytes`, `host-link-bandwidth-gbps`,
`host-dram-bandwidth-gbps`, `host-transfer-latency-s` fields under that same
prefix. The validation runner exposes equivalent short flags. Existing HBM-only
`lru` and cache-disabled runs keep their previous behavior.
