# Section VI-E: Task-Scheduling Strategies

This experiment reproduces Fig. 12. It compares Static, FCFS, Work-steal, and
HYDRA Elastic task scheduling on the four Nemotron-H workloads for
$B^{\star}$ and $M_T$.

## Reproduce

The fixed configurations are stored in:

```text
artifact/experiments/vi_e_scheduler_strategies/fig12_configurations.csv
```

Run the complete experiment from the repository root:

```bash
bash artifact/experiments/vi_e_scheduler_strategies/run.sh
```

The wrapper reruns all four scheduling policies and then regenerates Fig. 12.
Simulation outputs are written under:

```text
artifact/run_outputs/vi_e_scheduler_strategies/fig12/raw_simulations/
```

Generated results:

```text
artifact/run_outputs/vi_e_scheduler_strategies/fig12/fig12_scheduler_results.csv
artifact/figures/vi_e_scheduler_strategies/fig12_scheduler_bars.png
```
