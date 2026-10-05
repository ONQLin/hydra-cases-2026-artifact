# CHIPSIM validation experiments

See the **[integration README](../../../integrations/chipsim/README.md)** for
setup and the ordered validation workflow. This optional experiment replays
archived Pareto points with static batching and static pipeline mapping; it is
separate from the original paper reproduction suite.

- [Completed three-fidelity comparison](../packet_validation/README.md):
  six-point five-second curves, accuracy, simulation cost, and stress tests.
- `run_comparison.py`: run paired HYDRA/CHIPSIM configurations and audit results.
  Pass `--detailed-backend chipsim_contended` for persistent packet contention;
  the default `chipsim_runtime` is the earlier isolated-transfer baseline.
- `collect_checkpoints.py`: compare matching windows from active or completed
  contended experiments; invoke from the repository root with
  `python -m tests.validation.chipsim_validation.collect_checkpoints <output-dir>`.
- [Contention and pacing validation](CONTENTION_VALIDATION.md): measured
  network checks and serving results, with longer-run output locations.
- [Isolated-transfer validation](VALIDATION.md): earlier baseline measurements.

`run.sh` forwards arguments to `run_comparison.py`. Generated raw outputs are
local and excluded from Git; the validation records preserve measured findings.
