# Section VI-C Prepared Input

`raw_data/hybrid_llm_dse_profiles.csv` contains the prepared Jamba, Zamba, and
Nemotron-H Chat profiles used to reproduce the specialization--generality
study. It contains no run directory or simulator start timestamp.

The standard analysis reads this CSV directly:

```bash
python artifact/experiments/vi_c_specialization_study/reproduce_specialization_table.py
```

The optional simulator and profile-generation workflow is documented in:

```text
artifact/experiments/vi_c_hybrid_llm_dse_optional/README.md
```
