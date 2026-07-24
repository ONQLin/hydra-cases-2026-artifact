# Section VI-F: Performance Model for Fast DSE

This experiment completes Fig. 8 with the fast-DSE points and evaluates the
roofline and Markov estimators. Fig. 8 uses static task scheduling and static
batching. A separate 64-candidate evaluation measures estimator recovery when
elastic scheduling and dynamic batching are enabled.

## Reproduce

Run from the repository root:

```bash
bash artifact/experiments/vi_f_markov_fast_dse/run.sh
```

The wrapper selects the Fig. 8 candidates, regenerates the complete figure,
evaluates Roofline and Markov-guided search, and plots their recovery. Placement
and mapping caches are stored under `output_temp_runs/vi_f_markov_fast_dse/`
and reused by later runs.

Generated results:

```text
artifact/run_outputs/vi_f_markov_fast_dse/fig8_static_static/
artifact/figures/vi_f_markov_fast_dse/static_static/fig8_complete_12panels.png
```

Estimator-evaluation results:

```text
artifact/run_outputs/vi_f_markov_fast_dse/nemo_et_db_profiles/
  nemo_et_db_roofline_candidates.csv
  nemo_et_db_roofline_mt_recovery.csv
  nemo_et_db_markov_candidates.csv
  nemo_et_db_mt_recovery.csv

artifact/figures/vi_f_markov_fast_dse/
  estimator_recovery_64_candidates.png
```
