# HYDRA–CHIPSIM/Garnet network co-simulation

Validate an existing HYDRA design point with packet-level network feedback,
then compare output-token throughput (TP) and time to first token (TTFT).
For the faster C++ medium level, see [HYDRA-Packet](../packet/README.md).
The [main README](../../README.md#multi-fidelity-validation) shows the architecture.
Run all commands below from the HYDRA repository root.

## Terminology and responsibilities

**CHIPSIM/Garnet** names the high-fidelity network backend (`chipsim_contended`).
Existing figures abbreviate it as **CHIPSIM**. The integration reuses CHIPSIM's
communication infrastructure and bundled gem5/Garnet checkout. The pinned
CHIPSIM fork includes HYDRA adapters, online control and optional DMA pacing.

HYDRA owns compute profiles and serving policy. Its `NetworkExecutionMixin`
sequences phases, coordinates compute/network overlap, and delivers completion
events to the runtime. The CHIPSIM-side service manages the persistent Garnet
network. CHIPSIM's CMOS analytical and CIMLoop compute backends are not used in
the three-fidelity serving comparison. This is network co-simulation under shared
compute profiles, rather than an independent validation of accelerator compute.

## Backends and scope

| Backend | Execution | Purpose |
|---|---|---|
| `hydra_sim` | HYDRA-Analytic timing | Native baseline; default for `main.py` |
| `chipsim_contended` | One persistent Garnet network with concurrent flows | Serving validation; recommended below |
| `chipsim_runtime` | Isolated Garnet transfers with latency caching | Earlier diagnostic baseline; comparison driver's default |
| `chipsim` | Standalone transformer-block snapshot | Adapter demo; does not produce a full-serving TP/TTFT comparison |

HYDRA retains workload generation, placement, weight allocation, static batching,
offline task mapping, admission, and token accounting. TSCS/MARCA export shared
compute/SRAM times and aggregate external-memory bytes. The execution backend
advances Garnet only to the next HYDRA event; a completed flow can wake HYDRA
earlier. Routers, links, buffers, VCs, and credits persist across advances.
There is no latency cache or analytical fallback in `chipsim_contended`.

```text
local phase time = max(shared compute/SRAM time,
                       bytes / allocated HBM bandwidth + access latency)
phase completion = local timer elapsed AND all network flits received
block completion = all sequential phases completed
```

This formula applies to each primitive phase. New dependency-aware KDA/MLA
profiles expand ordered phases into read, compute, then write primitives;
writes reverse the compute/HBM endpoints. Legacy aggregate phases retain their
overlap convention. See the [operator execution contract](../../docs/workload_modernization/operators/README.md).

Supported validation: LLaMA3-8B and Nemotron-H-4B, Attention and SSM blocks,
prefill/decode, static batching, static pipeline mapping, uniform meshes, and
one-byte parameter profiles. Elastic mapping and tensor parallelism are rejected.
Legacy aggregate memory traffic is represented as HBM-to-compute transfers;
inter-block HBM-to-HBM movement is explicit. KDA/MLA graph profiles distinguish
read and write traffic; their small serving fixture has also been validated.
HYDRA still controls bandwidth reservations; Garnet's routes are not pinned to
HYDRA's reserved paths. This validates network effects under a shared runtime
and compute model, not an independent cycle-level implementation of the LLM.

## 1. Install

Activate the HYDRA environment from the [root setup](../../README.md#environment-setup).
The tested CHIPSIM build uses Ubuntu 24.04, Python 3.12, and GCC 13.3. No GPU or
CIMLoop server is needed for the shared HYDRA compute path. CHIPSIM's pinned
Python dependencies are installed separately in `.chipsim-venv/`.

```bash
# Ubuntu/Debian build dependencies; install once.
sudo apt update
sudo apt install -y build-essential m4 python3-dev python3-venv pkg-config \
  zlib1g-dev protobuf-compiler libprotobuf-dev libgoogle-perftools-dev

git submodule update --init --recursive third_party/CHIPSIM
bash scripts/setup_chipsim.sh
```

The setup script checks out the pinned CHIPSIM submodule revision, creates
the isolated Python environment, and builds `gem5.opt`. No patch application
is needed. To select the build Python or reduce memory usage, use
`PYTHON_BIN=/usr/bin/python3 CHIPSIM_BUILD_JOBS=4 bash scripts/setup_chipsim.sh`.
The Python installation needs development headers and a shared `libpython`.
Allow several GB for dependencies and build products; packet-level serving runs
can take hours and write substantial trace logs.

## 2. Check adapters, timing, contention, and pacing

```bash
python -m unittest tests.test_chipsim_backend tests.test_execution_backend \
  tests.test_execution_cleanup \
  tests.test_chipsim_contention.CoSimulationTest tests.test_serving_recorder \
  tests.test_chipsim_checkpoints tests.test_chipsim_validation

RUN_CHIPSIM_GARNET_TESTS=1 \
CHIPSIM_TEST_OUTPUT=output_sanity_checks/chipsim_network \
.chipsim-venv/bin/python -m unittest tests.test_chipsim_runtime_service \
  tests.test_chipsim_contention.PacketContentionTest tests.test_chipsim_dma_pacing
```

The real Garnet checks cover shared destinations, staggered arrivals,
independent paths, flow completion at a fixed cutoff, and HBM source bandwidth
sharing. Microbenchmarks isolate these mechanisms from compute-dominated serving.

## 3. Run a short paired serving check

```bash
python tests/validation/chipsim_validation/run_comparison.py \
  --detailed-backend chipsim_contended --models LLAMA3 NEMO \
  --points 1 --requests 4 --time-limit 0.25 --workers 2 \
  --output-dir output_sanity_checks/chipsim_smoke
```

The driver selects archived batch-size-2 Pareto points from the VI-B reports
also used by the VI-F/Fig. 8 reproduction. Selection uses archived TTFT order;
it does not optimize for agreement. Both backends rerun the selected setup
using the same seed, HYDRA trace preprocessing, and static policies. No extra
length scaling is added; HYDRA's configured maximum-token limit still applies. Input workload
and placed/mapped system snapshots must match before the pair is accepted.

## 4. Compare several points over a longer window

```bash
python tests/validation/chipsim_validation/run_comparison.py \
  --detailed-backend chipsim_contended --models LLAMA3 NEMO \
  --datasets CHAT --points 3 --requests 32 --time-limit 5 \
  --workers 6 --run-timeout 86400 \
  --output-dir output_sanity_checks/chipsim_long_5s
```

This selects the endpoints and middle of the archived frontier without a sweep.
`--workers` controls independent runs, not parallelism within one Garnet network.
`--run-timeout` is the wall-clock limit per backend run; `--transfer-timeout`
(default 120 s) is the service-response timeout. BWB is available via
`--datasets BWB`; it can be more expensive. HYDRA checks its cutoff every
100 microseconds, so choose durations that are multiples of that interval.

While runs are active, collect paired 1/2/5-second checkpoints in another terminal:

```bash
python -m tests.validation.chipsim_validation.collect_checkpoints \
  output_sanity_checks/chipsim_long_5s --windows 1 2 5 --watch-seconds 60
```

The collector reports partial coverage explicitly and never pairs different
cutoffs. It stops when all simulations finish or after its `--timeout` (default
24 hours). It observes runs; it does not launch or restart them.

Use a fresh output directory for a changed setup, code revision, or failed run.
`--resume` reuses completed runs only when their command matches exactly; it
does not restore simulator state or overwrite incomplete logs/checkpoints.

## 5. Enable HBM DMA pacing or validate your own setup

Add `--dma-pacing` to the comparison command and choose a separate output
folder, for example `output_sanity_checks/chipsim_paced_5s`. Pacing requires
`--detailed-backend chipsim_contended`; it is disabled by default.
`--dma-burst-bytes` defaults to 4096.

Each physical HBM serves ready flows in round-robin bursts at its exported
bandwidth, with one access delay per flow. Bursts enter Garnet after their memory
service and overlap subsequent DMA service. Independent HBMs operate separately.
Burst sizes round down to full flits; transfer sizes round up. Completion still
waits for every flit. HYDRA's allocated-bandwidth phase timer remains active;
it is not added to the paced transfer time. This models source bandwidth sharing,
not DRAM commands, bounded DMA buffers, or detailed read/write scheduling.

For an existing `main.py` experiment, retain its hardware, workload, placement,
batch size, seed, and duration. Run it once with `--simulator-backend hydra_sim`
and once with these settings, using separate output directories:

```bash
--simulator-backend chipsim_contended \
--mapping-config.mapping-strategy static \
--mapping-config.task-parallelism pipeline \
--cluster-config.local-scheduler static \
--chipsim-config.virtual-channels-per-vnet 8 \
--chipsim-config.timeout-seconds 120
```

Add `--chipsim-config.dma-pacing` and optionally
`--chipsim-config.dma-burst-bytes 4096` for paced execution. Pass the trace-file
argument explicitly: the dataset label alone does not select a trace file.
The eight-VC setting is an explicit microarchitecture choice, not a fitted
latency correction.

## Outputs and interpretation

| File under the comparison directory | Meaning |
|---|---|
| `selected_points.json` | Archived source rows and selected hardware configurations |
| `comparison.csv`, `tp_vs_ttft.png` / `.pdf` | Completed paired metrics, sample counts, input hashes, and curves |
| `summary.json` | Errors, point-order agreement, phase diagnostics, and causal flow audit |
| `checkpoint_comparison/` | Per-window metrics/figures and live `status.json` |
| `<point>/<backend>/command.json`, `console.log` | Exact command and execution log |
| `<point>/<backend>/<run>/metrics.json` | Final serving metrics; absent until successful completion |
| `<point>/<backend>/<run>/checkpoints/` | Cumulative and previous-second TP, TTFT samples, request counts |
| `<point>/chipsim_contended/<run>/chipsim_runtime/` | Exported `system.json`, execution trace, network progress/summary, Garnet command/log/stats |

TP counts output tokens over elapsed simulated time. TTFT is the mean over
requests that produced a first token before the cutoff. Windows start at t=0
without warmup exclusion; inspect sample counts and unfinished requests before
comparing curves. A flow still in flight at cutoff is not a completed flow.
Small serving differences alone do not validate contention; use the network
microbenchmarks and the causal/flow-accounting audit too.

The native baseline retains its original timing conventions. In particular,
`Sim/processing.py` currently uses `50 * 10E-9` (500 ns) and `/ 10E9` in its
inter-HBM transfer estimate; the CHIPSIM path uses the configured HBM access
latency (50 ns by default) and Garnet link timing. Native kernel rounding also
differs from per-phase completion rounding. Thus reported deltas include these
modeling differences, not only contention. Correcting the legacy native units
requires a separately labeled baseline rerun; the ongoing runs retain their
original baseline.


Measured evidence is recorded in the [contention and pacing validation record](../../tests/validation/chipsim_validation/CONTENTION_VALIDATION.md)
and the earlier [isolated-transfer record](../../tests/validation/chipsim_validation/VALIDATION.md).
Raw outputs are local generated files, excluded from Git. A running experiment
or partially populated checkpoint report is not a completed curve validation.

## Upstream and maintenance

[CHIPSIM](https://github.com/ONQLin/CHIPSIM) is an open-source chiplet simulator
using gem5 Garnet for network timing; its
[paper](https://doi.org/10.1109/OJSSCS.2025.3626314) describes the upstream simulator.
HYDRA pins the fork through the `third_party/CHIPSIM` submodule commit.
The fork directly versions the HYDRA adapters and online Garnet/DMA extensions;
cloning the pinned revision provides the complete integration source.

Simulation entry points subclass `BaseSimulationBackend`; execution providers
subclass `BaseExecutionBackend` and register through HYDRA's existing subclass
factory pattern. Extend those interfaces instead of duplicating the serving
scheduler. Service drivers live here; policy and model ownership stays in HYDRA.

Commit and push CHIPSIM changes to the fork first, then update and commit
the `third_party/CHIPSIM` gitlink (mode `160000`) in HYDRA. The parent repository
records a commit, not uncommitted submodule edits. Keep `.gitmodules` versioned
and environments, binaries and raw experiment outputs outside source control.
