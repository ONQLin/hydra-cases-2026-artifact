# Simulator validation

These suites validate extensions to HYDRA. The original paper's reproduction
scripts and results remain in [`artifact/`](../../artifact/README.md).

| Suite | Coverage |
|---|---|
| [CHIPSIM](chipsim_validation/README.md) | Setup export, serving feedback, contention and DMA pacing |
| [Packet and fidelity comparison](packet_validation/README.md) | Network checks, TP/TTFT shifts and simulation cost |
| [Modern models](modern_models/README.md) | Qwen decoder profiles and serving replay |
| [Operators](operator_validation/README.md) | Attention/operator profiles, dependencies and memory accounting |
| [Kimi](kimi_models/README.md) | KDA, MLA and decoder replay |
| [DeepSeek](deepseek_models/README.md) | MoE decoder, capacity and workload replay |
| [Multi-package](multi_package/README.md) | Placement, transfer causality and bandwidth/count studies |
| [Agent workloads](agent_workloads/README.md) | BFCL import, dependent calls and host/external tools |

Unit tests construct small inputs in Python. Demo inputs are generated on demand:

```bash
python -m unittest discover -s tests
python tests/validation/prepare_inputs.py \
  --output-dir output_sanity_checks/validation_inputs --include-datasets
```

Use a fresh output directory, then reuse it across the suite commands. Omit
`--include-datasets` for synthetic inputs only. Dataset subsets preserve complete
requests and write selection/hash metadata beside their generated CSVs. The
modern-model runner also generates its default short trace when `--trace` is
omitted; package studies generate their own workloads.

Detailed metrics, manifests, per-call/iteration logs and profile dumps belong in
ignored `output_sanity_checks/` directories. Tests check invariants and calculated
expectations rather than comparing against committed simulator output dumps.

Retained source inputs are the real BFCL excerpt, license/provenance and example
tool profiles, plus pinned upstream model metadata under `docs/`. Measured data
supporting documentation lives with the figures: the six-case
[multi-fidelity CSV](../../docs/multi_fidelity/data/comparison.csv) and its compact
[coverage record](../../docs/multi_fidelity/data/summary.json), and the
[package study CSV](../../docs/multi_package/data/package_study.csv). These are
historical measurements, not generated synthetic fixtures. The paper's original
design-point CSVs remain in `artifact/run_outputs/`.

Architecture and interpretation belong in [`docs/`](../../docs/); backend setup
instructions belong in [`integrations/`](../../integrations/README.md).
