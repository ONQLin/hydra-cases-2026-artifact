# HYDRA Simulator and DSE Framework

HYDRA evaluates hybrid LLM inference on heterogeneous 2.5D chiplet systems. It
combines request traces, analytical accelerator models, detailed event-driven
simulation, and fast candidate filtering to study:

- memory and compute chiplet composition;
- prefill- and decode-specialized accelerators;
- chiplet placement and model-weight allocation;
- Network-on-Interposer (NoI) bandwidth;
- request batching and task scheduling;
- throughput and time to first token (TTFT).

## Execution Paths

HYDRA has three connected execution paths:

1. **Detailed simulation:** `main.py` and `Sim/` model requests, chiplets,
   memory, NoI communication, scheduling, and execution using SimPy.
2. **Analytical estimation:** `analytic_profile/` provides operation latency,
   area, and memory parameters for each accelerator type.
3. **Fast DSE:** `Fast_Estimate/` uses Roofline and Markov-based models to
   filter the design space before evaluating the retained candidates.

The paper reproduction procedures that connect these paths are under
`artifact/experiments/`.

## Repository Map

- `main.py`: command-line entry point for one detailed simulation.
- `Sim/config/`: model, architecture, workload, placement, mapping, and metrics
  configuration.
- `Sim/entities/`: requests, model blocks, compute chiplets, memory chiplets,
  network resources, and static mapping.
- `Sim/placer/`: Round-Robin, Random, communication-aware, and trace-based
  placement.
- `Sim/scheduler/`: static and dynamic request batching.
- `Sim/task_scheduler.py`: Static, FCFS, Work-stealing, and Elastic task
  assignment.
- `Sim/processing.py`: block execution, communication reservation, and runtime
  task dispatch.
- `Sim/metrics/`: throughput, TTFT, utilization, congestion, and request
  counters.
- `analytic_profile/`: accelerator models and YAML hardware descriptions.
- `dataset/`: request-length traces.
- `Fast_Estimate/`: fast-DSE models and candidate filtering.
- `artifact/`: curated inputs, reproduction scripts, result tables, and
  figures.

## One Detailed Simulation

Configuration is exposed through `tyro` dataclasses in
`Sim/config/sys_config.py`. All available options can be inspected with:

```bash
python main.py --help
```

A run constructs the configured model and workload, places chiplets, allocates
weights, generates the preferred static mapping, injects trace requests, and
advances the simulation until the configured time limit.

Typical outputs are:

- `config.json`: complete simulation configuration;
- `placement.json`: chiplet locations and network topology;
- `log_info.txt`: throughput, TTFT, area, and request statistics;
- optional utilization figures when verbose metrics are enabled.

## Simulation Pipeline

`Sim/simulator.py` initializes one simulation:

1. Load the model preset and request trace.
2. Build the chiplet graph and select a placement policy.
3. Instantiate HBM and compute chiplets.
4. Allocate model weights to memory chiplets.
5. Generate preferred prefill and decode mappings.
6. Admit requests with the selected request scheduler.
7. Dispatch each model-block task with the selected task scheduler.
8. Reserve compute, memory, and NoI resources until the task completes.
9. Update request state, token counters, throughput, and TTFT.

`Sim/sim_core.py` manages outstanding, ready, executable, running, and
completed requests. `Sim/processing.py` performs phase-specific model
operations and communication for each batch.

## Models and Workloads

Model presets are defined in `Sim/config/model_config.py`.

| Model | Composition |
| --- | --- |
| `llama3-8b` | 32 Attention blocks |
| `mamba2-3b` | 64 Mamba blocks |
| `nemotronh-4b` | 24 Mamba and 4 Attention blocks |
| `zamba2-7b` | 81 Mamba and 13 Attention blocks |
| `jamba-tiny` | 14 Mamba and 2 Attention blocks |
| `jamba-mini` | 24 Mamba and 8 Attention blocks |

The included traces cover:

- `dataset/arxiv/`: document summarization;
- `dataset/bwb/`: translation;
- `dataset/chat/`: conversational requests;
- `dataset/longwriter/`: long-form generation.

Trace files provide prefill and decode token lengths. Model-specific trace
variants are included for Llama3, Mamba2, and Nemotron-H, while the
specialization study uses a common Chat trace to compare hybrid models.

## Hardware Configuration

The paper experiments use a 24-chiplet, 6-by-4 interposer. A configuration
specifies:

- `Num M`: HBM chiplets;
- `Num Mp` and `Num Md`: Mamba prefill/decode chiplets;
- `Num Ap` and `Num Ad`: Attention prefill/decode chiplets;
- `NoI_bw(GBps)`: provisioned NoI bandwidth;
- batch size, placement policy, request scheduler, and task scheduler.

The main specialized accelerator types are:

- `marca_p` and `marca_d` for Mamba prefill and decode;
- `tscs_p` and `tscs_d` for Attention prefill and decode;
- `HBM3` for model weights, intermediate state, and cache data.

Alternative compute and memory definitions also exist in
`Sim/config/utils.py`. Hardware parameters are read from
`analytic_profile/comp_acc/*.yml` and `analytic_profile/mem/*.yml`.

## Analytical Models and Static Mapping

`analytic_profile/Base_Accmodel.py` provides a common interface for operation
latency models. Implementations include MARCA, TSCS, B200, systolic-array,
VU-array, and unified accelerators. They model operations such as Attention,
SSM, fully connected layers, Conv1D, normalization, and activations.

`Sim/entities/static_mapper.py` evaluates compatible task-chiplet pairs and
builds preferred mappings for prefill and decode. Its communication cost
depends on transferred data volume, provisioned bandwidth, and Manhattan
distance between the compute chiplet and the memory chiplet containing the
required weights.

The detailed simulator subsequently models NoI path reservation, link
availability, memory-port limits, and runtime contention.

## Placement Policies

Placement policies are loaded from `Sim/placer/`:

- `rr`: places HBM on the outer ring and assigns compute chiplets in
  Round-Robin order.
- `random`: keeps HBM on the outer ring, randomizes compute locations, and
  randomly assigns model weights across available HBM chiplets.
- `bw`: communication-aware placement that partitions HBM resources and places
  compute chiplets near their dominant memory traffic.
- `trace`: follows an explicit package-placement trace from
  `analytic_profile/`.

## Request and Task Scheduling

Request scheduling determines which requests form a batch:

- `static`: fixed-size batching;
- `vllm`: the dynamic batching policy;

Task scheduling determines which compatible compute chiplet executes each
model-block operation:

- `static`: follows the placement-selected preferred mapping.
- `fcfs`: assigns incoming tasks to compatible chiplets in Round-Robin order.
- `worksteal`: starts from FCFS and migrates queued work toward less-loaded
  compatible chiplets.
- `elastic`: considers the preferred mapping, queue pressure, and
  communication distance when selecting a chiplet.

The FCFS and Work-stealing baselines include runtime dispatch overhead in
`Sim/task_scheduler.py`; Work-stealing also changes queue assignment without
using communication cost in its decision.

## Fast DSE

The fast-DSE path is implemented by:

- `Fast_Estimate/fast_estimate_models.py`: physical service-rate and
  Markov-based runtime estimates;
- `Fast_Estimate/static_state_filter.py`: candidate filtering for static
  scheduling and batching;
- `Fast_Estimate/Rooflines_est.py`: the basic Roofline baseline.

For the static setting in Fig. 8, the model evaluates a static allocation
state and retains a small candidate set for detailed-profile lookup. For
elastic scheduling and dynamic batching, the Markov model represents changing
prefill/decode service states and runtime resource sharing. The artifact
evaluates how much of the exhaustive optimum is recovered under a fixed
candidate budget.

Placement and mapping evaluations are cached under `output_temp_runs/` so
repeated estimator runs do not rebuild unchanged mappings.

## Artifact Reproduction

The standard Section VI-B through VI-F workflow is:

```bash
bash artifact/reproduce_all.sh
```

Individual experiments and optional exhaustive sweeps are documented in
`artifact/README.md`. Curated measurements are under `artifact/run_outputs/`,
and reproduced figures are under `artifact/figures/`.
