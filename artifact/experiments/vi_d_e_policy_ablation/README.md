# Sections VI-D and VI-E: Policy Ablation

This experiment reproduces Fig. 9. It compares communication-aware placement
(`CP`), elastic scheduling (`CP+ET`), and dynamic batching (`CP+ET+DB`) for
Llama3, Mamba2, and Nemotron-H across the four datasets.

## Reproduce

Prepared inputs:

```text
artifact/run_outputs/vi_d_e_policy_ablation/raw_data/
  fig9_configurations.csv
  fig9_measurements.csv
```

Run from the repository root:

```bash
bash artifact/experiments/vi_d_e_policy_ablation/run.sh
```

The wrapper processes the included measurements and regenerates Fig. 9.

Generated results:

```text
artifact/run_outputs/vi_d_e_policy_ablation/fig9_policy_results.csv
artifact/figures/vi_d_e_policy_ablation/fig9_policy_ablation.png
```

The optional simulator replay is documented in:

```text
artifact/experiments/vi_d_e_policy_ablation_sim_optional/README.md
```
