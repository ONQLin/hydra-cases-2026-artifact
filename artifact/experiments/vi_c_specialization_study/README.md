# Section VI-C: Specialization and Generality

This experiment reproduces Table IV. It compares hardware selected for one,
two, or all three hybrid LLMs and evaluates how specialization affects
performance on Jamba, Zamba, and Nemotron-H.

## Reproduce

Prepared input:

```text
artifact/run_outputs/vi_c_specialization_study/raw_data/hybrid_llm_dse_profiles.csv
```

Run from the repository root:

```bash
bash artifact/experiments/vi_c_specialization_study/run.sh
```

The wrapper analyzes the prepared hybrid-LLM profiles and renders the table.

Generated table:

```text
artifact/figures/vi_c_specialization_study/table_iv_reproduced.png
```

The optional full DSE is documented in:

```text
artifact/experiments/vi_c_hybrid_llm_dse_optional/README.md
```
