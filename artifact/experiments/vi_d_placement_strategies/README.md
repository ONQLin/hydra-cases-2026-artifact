# Section VI-D: Placement Strategies

This experiment reproduces Fig. 10. It compares round-robin, random, and
communication-aware placement on the four Nemotron-H workloads for
$B^{\star}$ and $M_T$.

## Reproduce

The fixed configurations are stored in:

```text
artifact/experiments/vi_d_placement_strategies/fig10_configurations.csv
```

Run the complete experiment from the repository root:

```bash
bash artifact/experiments/vi_d_placement_strategies/run.sh
```

The wrapper reruns the selected Random-placement configurations, combines them
with the prepared Round-Robin and communication-aware baselines, and then
regenerates Fig. 10. Simulation outputs are written under:

```text
artifact/run_outputs/vi_d_placement_strategies/fig10/raw_simulations/
```

Generated results:

```text
artifact/run_outputs/vi_d_placement_strategies/fig10/fig10_placement_results.csv
artifact/figures/vi_d_placement_strategies/fig10_placement_bars.png
```
