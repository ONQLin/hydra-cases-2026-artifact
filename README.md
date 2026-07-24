# HYDRA

HYDRA is a design-space exploration framework for hybrid LLM inference on
heterogeneous chiplet systems. It combines a detailed event-driven simulator,
analytical accelerator models, workload traces, and fast DSE models for
exploring hardware composition, placement, scheduling, batching, and NoI
bandwidth.

This repository includes the source code and curated data needed to reproduce
the results in Sections VI-B through VI-F of the paper.

## Environment Setup

The pinned NumPy and SciPy versions in `requirements.txt` require Python 3.11
or newer. Python 3.11 on Linux is recommended.

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Verify the main dependencies and simulator entry point:

```bash
python -c "import matplotlib, networkx, numpy, pandas, scipy, simpy; print('HYDRA environment ready')"
python main.py --help
```

The virtual environment only needs to be activated once per terminal session:

```bash
source .venv/bin/activate
```

## Reproduce the Results

Run the complete standard artifact workflow:

```bash
bash artifact/reproduce_all.sh
```

The driver runs all non-optional experiments from Sections VI-B through VI-F.
It reports the current experiment, progress, completion status, output
locations, and elapsed time. Per-experiment logs and a final summary are
written under:

```text
output_temp_runs/artifact_reproduction/<timestamp>/
```

Reproduced data and figures are written to:

```text
artifact/run_outputs/
artifact/figures/
```

The placement and scheduling comparisons launch several simulations in
parallel. Their concurrency can be adjusted before starting the suite:

```bash
HYDRA_NUM_WORKERS=8 bash artifact/reproduce_all.sh
```

The exhaustive macro-architecture and hybrid-LLM DSE sweeps are intentionally
excluded from this command because they are substantially more expensive.

## Step-by-Step Reproduction

See [artifact/README.md](artifact/README.md) for the paper-section map,
individual `run.sh` commands, expected outputs, and optional exhaustive
procedures.

The standard experiments can also be run independently:

```bash
bash artifact/experiments/vi_b_dse_on_macro_architectures/run.sh
bash artifact/experiments/vi_c_specialization_study/run.sh
bash artifact/experiments/vi_d_e_policy_ablation/run.sh
bash artifact/experiments/vi_d_placement_strategies/run.sh
bash artifact/experiments/vi_e_scheduler_strategies/run.sh
bash artifact/experiments/vi_f_markov_fast_dse/run.sh
```

## Repository Layout

- `main.py`: entry point for one detailed simulation.
- `Sim/`: event-driven simulator, chiplet system, placement, request
  scheduling, task scheduling, batching, and metrics.
- `analytic_profile/`: analytical compute and memory models plus hardware
  configuration files.
- `dataset/`: ARXIV, BWB, Chat, and LongWriter request traces.
- `Fast_Estimate/`: Roofline and Markov-based fast-DSE models.
- `artifact/experiments/`: section-level reproduction scripts.
- `artifact/run_outputs/`: curated measurements and generated result tables.
- `artifact/figures/`: reproduced PNG figures.

For a detailed description of the simulator and DSE framework, see
[sim_info.md](sim_info.md).
