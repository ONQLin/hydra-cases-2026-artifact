# Optional Exhaustive DSE for Section VI-B

This optional path regenerates the detailed-simulation data used by the
Section VI-B design-space plots. The standard reproduction already includes
the required CSV files, so this sweep is only needed for a full rerun.

## Run and Summarize

Run from the repository root:

```bash
bash artifact/experiments/exhaustive_sim_optional/run.sh
```

The underlying runner is resumable. Set `HYDRA_NUM_WORKERS` to control
concurrency. Outputs are written under:

```text
output_temp_runs/vi_b_exhaustive_dse/
  simulator_runs/
  Summary_Reports/
  Pareto_Reports/
```

## Use a Completed Sweep

After the full sweep finishes, replace the prepared reports under:

```text
artifact/run_outputs/vi_b_dse_on_macro_architectures/raw_data/
```

Then regenerate the Section VI-B figure:

```bash
bash artifact/experiments/vi_b_dse_on_macro_architectures/run.sh
```
