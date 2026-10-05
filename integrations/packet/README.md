# HYDRA-Packet: medium network fidelity

HYDRA-Packet (`hydra_packet`) sits between HYDRA-Analytic (`hydra_sim`) and
[CHIPSIM](../chipsim/README.md) (`chipsim_contended`). All three use HYDRA's setup,
workload, static batching/mapping, and compute profiles. Medium fidelity applies
to network execution; it is not a cycle-level compute simulator.

## Build and validate

From the repository root, activate the HYDRA Python environment. The medium
backend needs a C++17 compiler and Python's built-in `ctypes`; it does not need
the CHIPSIM checkout, gem5, CIMLoop, or a second Python environment.

```bash
bash scripts/setup_packet.sh
python -m unittest tests.test_packet_network tests.test_execution_backend \
  tests.test_execution_cleanup \
  tests.test_chipsim_contention.CoSimulationTest

# Paired low/medium serving, first with a short window.
python tests/validation/chipsim_validation/run_comparison.py \
  --detailed-backend hydra_packet --models LLAMA3 NEMO \
  --points 1 --requests 4 --time-limit 0.25 --workers 2 \
  --output-dir output_sanity_checks/packet_smoke

# Three archived points per model; no new sweep.
python tests/validation/chipsim_validation/run_comparison.py \
  --detailed-backend hydra_packet --models LLAMA3 NEMO \
  --points 3 --requests 32 --time-limit 5 --workers 6 \
  --output-dir output_sanity_checks/packet_comparison_5s
```

Use `--simulator-backend hydra_packet` on an existing `main.py` command to keep
its hardware and workload. Supported policies match CHIPSIM serving: static
batching, static pipeline mapping, full rectangular meshes, and one-byte profiles.
The validated workloads are LLaMA3-8B and Nemotron-H-4B, including SSM blocks.

The build writes `.hydra-packet-build/libhydra_packet.so`, atomically replacing
the library so running simulations keep their loaded version. `CXX` can select
the compiler. Linux is the tested platform. No Python ABI-specific extension
build or new package installation is required.

## Modeling contract

`PacketNetwork` is a C++ discrete-event simulator with deterministic XY routing
on row-major mesh IDs and uniform link rates. It models separate directed links
and endpoint injection/ejection resources, FIFO port queues, round-robin
allocation of downstream buffer credits, and finite aggregate buffering. Later
flows compete with transfers already in flight. There is no latency cache.

Packets are **aggregation units**, not a change to the physical wire packet
size. The default maximum is 1024 bytes. A packet's head can advance to the next
hop before its tail arrives, preserving cut-through overlap; a long isolated
transfer is serialized once across a pipelined path, not once per hop. Each
stage adds `(router_latency_cycles + link_latency_cycles) / frequency`, including
the two endpoint stages. Uniform link rates prevent the outgoing tail from
overtaking incoming data. Clock-edge/VC/credit-return details are approximated.

An upstream packet reserves downstream space before transmission. A reservation
covers the whole aggregate packet until its tail leaves that port, including
in-service traffic. This is conservative aggregate buffering, not Garnet's
per-flit/per-VC buffer implementation. Round-robin credit grants avoid fixed
port-order starvation. When buffers are full, blocked inputs wait for release;
source traffic is generated lazily rather than allocating every packet upfront.

Optional DMA pacing serves ready flows round-robin at each physical HBM's
bandwidth, with one access delay per flow. A completed DMA burst becomes ready
for network injection; independent HBMs operate independently. DMA staging is
not capacity-limited. Changing network quantum does not change DMA burst size.

The Python execution backend shares `NetworkExecutionMixin` with CHIPSIM:
compute/HBM phase timers overlap network execution, and subsequent work waits
for both local timing and network completion. HYDRA retains resource admission
and reservations. There is no transfer-latency fallback.

## Parameters

| Comparison option | `main.py` option | Default |
|---|---|---:|
| `--packet-quantum-bytes` | `--packet-config.quantum-bytes` | 1024 |
| `--packet-buffer-bytes` | `--packet-config.buffer-bytes` | 16384 per directed port |
| `--dma-pacing` | `--chipsim-config.dma-pacing` | Off |
| `--dma-burst-bytes` | `--chipsim-config.dma-burst-bytes` | 4096 |

Shared router/link/frequency/HBM settings keep their existing `--chipsim-config.*`
CLI names for compatibility; using those settings does not require CHIPSIM for
the medium backend. Link width is derived from HYDRA's NoI bandwidth and frequency.
Garnet VC counts and per-flit credit settings do not affect the medium model.
Quantum, buffer, and DMA burst sizes round down to whole flits; transfer bytes
round up. Sizes smaller than one flit are rejected. The buffer must hold at
least one effective aggregate packet. Exported `system.json` records requested
and effective sizes and the loaded library's SHA-256.

Larger aggregation reduces event count but can alter arbitration and short-flow
latency. Check sensitivity (for example 256/1024/4096 bytes) instead of treating
aggregation as a physically exact packet size. Buffer capacity is a modeling
parameter, not automatically equivalent to a Garnet VC configuration.

## Compare against CHIPSIM

After [building CHIPSIM](../chipsim/README.md#1-install), run isolated,
shared-destination/source, staggered, disjoint, DMA, and larger-flow comparisons:

```bash
.chipsim-venv/bin/python tests/validation/packet_validation/validate_network.py \
  --garnet --stress --quanta 256 1024 4096 \
  --output-dir output_sanity_checks/packet_network_comparison
```

`comparison.json` records per-flow completion errors and wall times. Startup
and network execution times are separated; microbenchmark speedups are not
end-to-end serving speedups. Omitting `--garnet` runs medium-only cases with
the HYDRA Python environment.

Inspect longer medium runs while they execute:

```bash
python -m tests.validation.chipsim_validation.collect_checkpoints \
  output_sanity_checks/packet_comparison_5s --detailed-backend hydra_packet \
  --windows 1 2 5 --watch-seconds 60
```

For a three-level curve, use completed checkpoints from matching medium and
CHIPSIM experiment directories (same model/dataset/point selection and pacing):

```bash
python tests/validation/packet_validation/compare_levels.py \
  --packet-dir output_sanity_checks/packet_comparison_5s \
  --chipsim-dir output_sanity_checks/chipsim_long_5s \
  --window 1 --output-dir output_sanity_checks/three_levels_1s
```

The report checks workload, runtime configuration, placement/mapping, and exported network parameters;
missing points are listed explicitly. Each window is transient from t=0, and
TTFT only includes requests with a first token by cutoff. See the
[validation record](../../tests/validation/packet_validation/README.md)
for measured results and limitations.

The joint report includes signed error and wall-time columns. The complete
5-second six-point comparison measured medium/CHIPSIM maximum TP error of 12.41%
and TTFT error of 3.17%, with medium 8.60–57.61x faster by process wall time.
A staggered hotspot microbenchmark also exposed 43.15% per-flow timing error:
medium has no general accuracy bound, and smaller quantum does not always
approach Garnet. See the validation record before using it for MoE fan-in or
tail-latency claims.

## Implementation and diagnostics

See the shared [execution architecture and extension guide](../README.md) for
factory registration, provider responsibilities, and lifecycle requirements.

- `packet_network.hh` / `.cc`: C++ flow, packet, port, DMA, and event-queue state.
- `c_api.cc`: a small versioned C ABI; C++ exceptions become explicit Python errors.
- `network.py`: ownership and integer-picosecond validation through `ctypes`.
- `Sim/execution/packet.py`: completion feedback and trace output through the
  existing execution factory; serving policy checks and phase sequencing are shared.

All packet events run inside C++. Python calls `submit` or `advance`, not a
callback per packet. An advance stops at the next completion or the next HYDRA
event boundary and returns completions in a batch. Each network instance owns
its state and has one owning simulation thread; concurrent experiments use
separate processes. The main integration
risks are time ordering, buffer accounting, and aggregation bias, rather than
the Python/C++ call itself.

Per-run `packet_runtime/` contains `system.json`, `execution.jsonl`,
`network_summary.json`, and `packet_statistics.json`. The latter counts aggregate
packets, processed events, received bytes, and peak buffer occupancy. Failed
runs do not produce final serving metrics; no simulator is silently substituted.
