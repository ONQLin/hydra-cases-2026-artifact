# Persistent contention validation: 2026-10-03

Historical measurement record. Paths below identify the original local runs;
raw logs and simulator output dumps are not included in the public repository.
Use the [integration guide](../../../integrations/chipsim/README.md) to generate
new runs.

This record covers the initial contention and pacing checks. The completed
[three-fidelity validation](../packet_validation/README.md) contains the six-point
five-second curves, accuracy, simulation cost, and additional stress cases.

The `chipsim_contended` backend completed bidirectional HYDRA/Garnet serving
experiments for LLaMA3-8B and Nemotron-H-4B. Each run used one persistent network,
with no isolated-transfer latency cache. Later submissions can delay packets
already in flight, and network completions control HYDRA's subsequent phases
and resource release.

## Packet-level checks

A 2-by-2 mesh, 128-bit links, 1 GHz network, eight VCs per vnet, and 16 KiB per
flow produced the following measured completion times. Times are relative to
the start of the network simulation.

| Scenario | Flow 1 completion (us) | Flow 2 completion (us) |
|---|---:|---:|
| Isolated two-hop flow | 1.032 | — |
| Two simultaneous flows sharing a destination | 2.054 | 2.052 |
| Second flow joins at 0.250 us | 1.808 | 2.056 |
| Two independent one-hop paths | 1.030 | 1.030 |
| Isolated one-hop reference | 1.030 | — |

The shared destination and staggered arrival tests demonstrate real packet
contention. Independent paths preserve the isolated completion time. All fully
drained cases verified exactly 1,024 injected and received flits per flow.
Advancing again after equal-time completions also passed. A separate 0.100 us
cutoff received 92 of 1,024 flits and correctly reported no completed flow.

Raw local outputs: network cases (`output_sanity_checks/chipsim_contention/network_final/contention_results.json`)
and cutoff case (`output_sanity_checks/chipsim_contention/network_final/cutoff/result.json`).
Each case retains the actual Garnet command, log, and statistics.

## End-to-end serving

The first batch-size-2 Pareto point from the existing Chat report was used for
each model. Both backends ran the first four original trace requests, two static
batches, static pipeline mapping, and a 250 ms window from t=0. Full model
sequences and original input/output lengths were retained. The comparison
verified equal effective workloads, hardware placement, weight allocation, and
prefill/decode task mapping. All four requests contributed TTFT samples in each
run.

| Model | HYDRA tokens/s | Contended tokens/s | HYDRA TTFT (ms) | Contended TTFT (ms) | TTFT difference |
|---|---:|---:|---:|---:|---:|
| LLaMA3-8B | 40 | 40 | 143.694 | 144.016 | +0.224% |
| Nemotron-H-4B | 48 | 48 | 91.722 | 92.1615 | +0.479% |

| Model | Submitted flows | Completed flows | In flight at cutoff | Peak concurrent flows |
|---|---:|---:|---:|---:|
| LLaMA3-8B | 1,516 | 1,516 | 0 | 2 |
| Nemotron-H-4B | 2,328 | 2,327 | 1 | 2 |

Both traces passed flow-conservation and causal-order audits. Each model had six
network-dominated phases among completed block/transfer records. Their summed
NoI extension beyond compute/HBM duration was approximately 579.5 us and
579.0 us, respectively; these sums are not end-to-end latency deltas because
work overlaps. Most phases remained compute/HBM dominated.

The small TTFT differences include phase/completion time quantization as well
as network effects; they are not a decomposition of contention cost. The network
microbenchmarks separately isolate and demonstrate contention. This serving
experiment verifies the closed loop for two configurations, not multi-point
ordering or steady-state throughput. The prior six-point isolated experiment
must not be relabeled as a contended-network validation.

Local serving artifacts:

- Paired metrics and input hashes (`output_sanity_checks/chipsim_contention/serving_250ms/comparison.csv`).
- Phase diagnostics and audited network accounting (`output_sanity_checks/chipsim_contention/serving_250ms/summary.json`).
- Point comparison (`output_sanity_checks/chipsim_contention/serving_250ms/tp_vs_ttft.png`).

## Reproduction and checks

```bash
python tests/validation/chipsim_validation/run_comparison.py \
  --detailed-backend chipsim_contended --models LLAMA3 NEMO \
  --points 1 --requests 4 --time-limit 0.25 --workers 2 \
  --output-dir output_sanity_checks/chipsim_contention/serving_250ms --resume
```

Thirteen HYDRA adapter, profile, factory, and coordinator tests passed. Six
CHIPSIM runtime/contention tests passed, including real Garnet runs and the
fixed-cutoff check. Both installation patches applied to clean upstream source;
all 15 resulting source files matched the tested working tree.

See [the contention guide](../../../integrations/chipsim/README.md) for setup,
network-test commands, a larger multi-point experiment, and the precise traffic
and timing contract. Packet contention is complete for the admitted burst
traffic in this contract. Optional HBM DMA pacing is now available (see below);
HBM command scheduling remains outside the exported analytical profiles.


## Optional DMA pacing and longer windows

A source-HBM DMA queue now supports round-robin bursts at the exported physical
HBM bandwidth. It is disabled by default. With 1024-byte bursts, 50 ns access
latency, the same 2-by-2 mesh, and 16 KiB per flow, real Garnet runs measured:

| Scenario | Flow 1 completion (us) | Flow 2 completion (us) |
|---|---:|---:|
| Single flow, 1 GB/s HBM | 16.504 | — |
| Two flows sharing a 1 GB/s HBM | 31.864 | 32.890 |
| Two independent 1 GB/s HBMs | 16.504 | 16.504 |
| Single flow, 2 GB/s HBM | 8.312 | — |

All drained cases received exactly 1,024 packets per flow. These checks cover
source bandwidth sharing, independent HBMs, and bandwidth sensitivity.
The unit uses decimal GB/s. This is a DMA service model, not a DRAM command model.
Raw results are in `output_sanity_checks/chipsim_pacing/network/pacing_results_us.json`.

The DMA pacing regression passed 14 HYDRA adapter/coordinator/recorder tests,
7 real Garnet runtime/contention/pacing tests, and one checkpoint pairing test.
The updated installation patches reproduce all 17 affected files from clean
upstream sources. Packet IDs and cumulative packet counters now use 64 bits.

The longer experiment uses three archived batch-size-2 Pareto points per model,
32 original Chat requests, static mapping, and a 5-second simulation window.
All six runs completed; original outputs were written under `output_sanity_checks/chipsim_long_5s`.
Matched windows and the final six-point comparison are retained in the
[joint validation record](../packet_validation/README.md#serving-comparison).


Paced serving smoke runs also completed using the same four requests and
250 ms window as the unpaced smoke above (4096-byte DMA bursts):

| Model | HYDRA tokens/s | Paced tokens/s | HYDRA TTFT (ms) | Paced TTFT (ms) | TTFT difference |
|---|---:|---:|---:|---:|---:|
| LLaMA3-8B | 40 | 40 | 143.694 | 143.9885 | +0.205% |
| Nemotron-H-4B | 48 | 48 | 91.722 | 91.954 | +0.253% |

Both cases passed input equality, causal trace, and flow accounting checks.
LLaMA completed 1,516 of 1,516 flows; Nemotron completed 2,328 of 2,328.
Both peaked at two concurrent flows, with no latency cache. Pacing can alter
packet interleaving as well as add source service time; its end-to-end penalty
need not be monotonically greater than unpaced injection. The source-bandwidth
microbenchmarks above isolate pacing itself.

Artifacts: `output_sanity_checks/chipsim_pacing/serving_250ms/comparison.csv`,
`summary.json`, and `tp_vs_ttft.png` / `.pdf`. A separate 5-second, 32-request
paced comparison for P1 of both models completed under
`output_sanity_checks/chipsim_pacing/serving_5s`, against the matching unpaced P1
cases in the six-point suite. Its `comparison.csv`, `summary.json`, and per-run
process wall times were recorded in the original local run. The main three-fidelity curves use
unpaced injection consistently across all backends.
