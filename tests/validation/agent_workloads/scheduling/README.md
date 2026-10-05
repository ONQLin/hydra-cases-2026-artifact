# Agent runtime DSE validation

This study holds one archived HYDRA hardware point fixed and varies priority,
batch collection, and HBM prefix capacity. It validates mechanisms and sensitivity;
its synthetic sessions and SLOs are not BFCL performance measurements.
See [policy semantics and limitations](../../../../docs/agent_workloads/scheduling/README.md).

## Reproduce

Use the HYDRA Python environment from the repository root and fresh output paths:

```bash
python tests/validation/agent_workloads/study_scheduling.py \
  --output-dir output_sanity_checks/agent_scheduling/study
```

The OOP study runner uses the existing `AgentReplayValidation` hardware selection
and simulation flow. It creates 12 sessions, three calls per session, three
prompt shapes and three declared synthetic prefixes. Runtime concurrency is
capped at one active batch to expose queue ordering. Default service estimates
use a fixed output-token estimate; SJF/HRRN do not inspect recorded output lengths.
The native CLI exposes the same controls under `cluster-config.agent-scheduler`.

## Observed trends

Selected observations from the 12-point study (synthetic workload):

| Configuration | TP (tok/s) | Mean session (ms) | Mean queue (ms) | Session SLO met |
|---|---:|---:|---:|---:|
| FCFS, batch 1 | 11,709.6 | 17.031 | 5.119 | 25.0% |
| SJF, batch 1 | 11,709.6 | 10.298 | 2.875 | 66.7% |
| HRRN, batch 1 | 11,709.6 | 14.723 | 4.350 | 66.7% |
| EDF, batch 1 | 11,733.6 | 13.164 | 3.830 | 33.3% |
| FCFS, max batch 4 + timeout | 25,284.5 | 7.868 | 1.578 | 66.7% |
| Same, cache holds all 3 prefixes | 27,855.2 | 7.165 | 1.457 | 66.7% |

Priority changes queue distribution more than total work. Larger compatible
batches amortize weight traffic. A 49,151-byte cache holds no fixture prefix;
49,152 bytes holds one but thrashes (eight evictions, zero hits). At 147,456 bytes,
all three fit and 24 of 36 calls hit, improving TP by about 10.2% relative to the
same batching policy without cache. These observations apply to this controlled
trace and shared compute model, not a general policy ranking.

## Checks and generated outputs

The runner generates its workload, then verifies complete calls/tokens, tool
causality, compatible batches, prefix reuse and final memory accounting.
`summary.json`, `table.md` and per-run reports are written under `--output-dir`;
they are generated outputs and are not committed. Unit tests construct their
own small request and memory fixtures in Python:

```bash
python -m unittest tests.test_agent_scheduling
```

The optional [combined backend check](../continuous/README.md) covers continuous
batching, host tools and offload together. Real BFCL inputs and policy comparisons
are described in the [BFCL study](../bfcl_trade/README.md).

For an independent memory audit, run:

```bash
python tests/validation/operator_validation/audit_memory.py \
  --command output_sanity_checks/agent_scheduling/study/cache_one/gqa-prefix-fixture/hydra_sim/command.json \
  --output-dir output_sanity_checks/agent_scheduling/memory_audit
```
