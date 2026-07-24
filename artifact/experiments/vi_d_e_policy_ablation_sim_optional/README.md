# Optional Fig. 9 Simulator Replay

This optional path reruns the fixed configurations behind Fig. 9. The
standard reproduction uses the included results and does not require this
step.

## Run and Summarize

Run from the repository root:

```bash
bash artifact/experiments/vi_d_e_policy_ablation_sim_optional/run.sh
```

The wrapper reruns the selected configurations, summarizes them, and generates
a separate replay figure. Set `HYDRA_NUM_WORKERS` to control concurrency.
Outputs are written under:

```text
output_temp_runs/vi_d_e_policy_ablation/
  simulator_runs/
  processed/fig9_simulation_replay_results.csv
```

This keeps the replay separate from the standard artifact figure.
