# Optional Hybrid-LLM DSE for Section VI-C

This optional path regenerates the Jamba, Zamba, and Nemotron-H Chat results
used by the specialization study. The standard Table IV reproduction already
includes the required profile.

## Run and Summarize

Run from the repository root:

```bash
bash artifact/experiments/vi_c_hybrid_llm_dse_optional/run.sh
```

The underlying runner is resumable. Set `HYDRA_NUM_WORKERS` to control
concurrency. Outputs are written under:

```text
output_temp_runs/vi_c_hybrid_llm_dse/
  hybrid_llm_dse_runs/
  profiles/hybrid_llm_dse_profiles.csv
```

## Use a Completed Sweep

Replace the prepared profile at:

```text
artifact/run_outputs/vi_c_specialization_study/raw_data/hybrid_llm_dse_profiles.csv
```

Then regenerate Table IV:

```bash
bash artifact/experiments/vi_c_specialization_study/run.sh
```
