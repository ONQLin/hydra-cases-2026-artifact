# Artifact Run Outputs

This directory contains the configurations, simulation measurements, and
processed CSV results used by the artifact experiments:

- `vi_b_dse_on_macro_architectures/`: exhaustive macro-architecture DSE data.
- `vi_c_specialization_study/`: hybrid-LLM specialization profiles.
- `vi_d_e_policy_ablation/`: CP, CP+ET, and CP+ET+DB measurements.
- `vi_d_placement_strategies/`: Round-Robin, Random, and communication-aware
  placement measurements.
- `vi_e_scheduler_strategies/`: Static, FCFS, Work-stealing, and elastic
  scheduling measurements.
- `vi_f_markov_fast_dse/`: Fig. 8 fast-DSE selections and elastic
  scheduling/dynamic batching evaluation data.

Each `artifact/experiments/` README identifies its exact inputs and generated
outputs. Optional long-running simulations write intermediate data under
`output_temp_runs/`.
