# HYDRA Artifact Reproduction Map

This directory reproduces the results in Sections VI-B through VI-F. Each
experiment contains a `run.sh` entry point, its processing code, expected data
locations, and generated PNG figures.

## Run the Standard Suite

From the repository root:

```bash
bash artifact/reproduce_all.sh
```

The driver runs every non-optional experiment in paper order. It prints the
active experiment, output paths, pass/fail status, and elapsed time. Logs and a
tabular summary are written to:

```text
output_temp_runs/artifact_reproduction/<timestamp>/
```

## Experiments Checklist

- **Section VI-B, Fig. 8: DSE on Macro-Architectures**

  ```bash
  bash artifact/experiments/vi_b_dse_on_macro_architectures/run.sh
  ```

  Regenerates the 12 throughput-versus-TTFT panels from the included
  exhaustive-DSE reports.

- **Section VI-C, Table IV: Specialization and Generality**

  ```bash
  bash artifact/experiments/vi_c_specialization_study/run.sh
  ```

  Reproduces the specialization study across Jamba, Zamba, and Nemotron-H.

- **Sections VI-D/E, Fig. 9: Policy Ablation**

  ```bash
  bash artifact/experiments/vi_d_e_policy_ablation/run.sh
  ```

  Compares communication-aware placement (CP), CP with elastic scheduling
  (CP+ET), and CP with elastic scheduling and dynamic batching (CP+ET+DB).

- **Section VI-D, Fig. 10: Placement Strategies**

  ```bash
  bash artifact/experiments/vi_d_placement_strategies/run.sh
  ```

  Reruns the selected Random-placement configurations and compares
  Round-Robin, Random, and communication-aware placement.

- **Section VI-E, Fig. 12: Task-Scheduling Strategies**

  ```bash
  bash artifact/experiments/vi_e_scheduler_strategies/run.sh
  ```

  Reruns Static, FCFS, Work-stealing, and HYDRA's elastic scheduler.

- **Section VI-F: Performance Model for Fast DSE**

  ```bash
  bash artifact/experiments/vi_f_markov_fast_dse/run.sh
  ```

  Completes Fig. 8 with fast-DSE selections and compares Exhaustive, Roofline,
  and Markov-guided search under elastic scheduling and dynamic batching.

## Optional Full Simulations

These paths are excluded from `reproduce_all.sh` because they require much
longer simulator sweeps:

```bash
bash artifact/experiments/exhaustive_sim_optional/run.sh
bash artifact/experiments/vi_c_hybrid_llm_dse_optional/run.sh
bash artifact/experiments/vi_d_e_policy_ablation_sim_optional/run.sh
```

The first two regenerate the profiles used by Sections VI-B and VI-C. The
third replays all selected configurations used by the Fig. 9 policy ablation.

## Workflow

```text
                          optional, longer path
                    +-----------------------------+
                    |                             v
[simulation runner] +--> [raw simulation outputs] --> [post-processing]
                                                           |
                          standard, fast path               v
                    [included run outputs] -----------> [result CSV]
                                                           |
                                                           v
                                                       [PNG figure]
```

## Data and Results

- `artifact/run_outputs/` contains curated configurations, measurements, and
  generated result CSVs.
- `artifact/figures/` contains the reproduced PNG figures.
- `output_temp_runs/` contains optional simulations, reusable placement and
  mapping caches, and reproduction logs.

Set `HYDRA_NUM_WORKERS` to control simulation concurrency.

For a quick check that reuses already completed placement and scheduling runs:

```bash
HYDRA_REUSE_COMPLETED=1 bash artifact/reproduce_all.sh
```

## Caveats

- Exhaustive DSE reruns can take substantial time and storage.
- Small numerical variations may occur across machines. These differences do
  not change the reported comparisons, trends, or conclusions.
- The artifact includes a corrected fast-estimation implementation, so some
  values may differ slightly from those in the submitted manuscript. The
  correction preserves the reported trends and conclusions, and the affected
  values will be updated in the camera-ready version.
