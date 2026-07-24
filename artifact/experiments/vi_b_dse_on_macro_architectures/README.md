# Section VI-B: DSE on Macro-Architectures

This experiment reproduces the exhaustive throughput-versus-TTFT results in
Fig. 8 for Llama3, Mamba2, and Nemotron-H across ARXIV, BWB, Chat, and
LongWriter.

## Reproduce

Prepared inputs:

```text
artifact/run_outputs/vi_b_dse_on_macro_architectures/raw_data/
  Summary_Reports/
  Pareto_Reports/
```

Run from the repository root:

```bash
bash artifact/experiments/vi_b_dse_on_macro_architectures/run.sh
```

The wrapper runs `reproduce_fig8_exhaustive.py` on the prepared reports.

Generated figures:

```text
artifact/figures/vi_b_dse_on_macro_architectures/
  fig8_exhaustive_12panels.png
  panels/*.png
```

This figure contains the exhaustive results only. Section VI-F adds the
fast-DSE points to produce the complete Fig. 8.

## Optional Full Sweep

The long-running data-generation procedure is documented in:

```text
artifact/experiments/exhaustive_sim_optional/README.md
```
