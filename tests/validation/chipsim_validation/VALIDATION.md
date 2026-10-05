# Static serving validation: 2026-10-03

Historical measurement record. Paths below identify the original local runs;
raw logs and simulator output dumps are not included in the public repository.
Use the [integration guide](../../../integrations/chipsim/README.md) to generate
new runs.

The `chipsim_runtime` closed loop completed six paired runs: three archived
Pareto configurations each for LLaMA3-8B and Nemotron-H-4B on Chat. Each pair
used identical effective workloads, placed hardware, weight allocation, and
prefill/decode task mapping. The comparison driver verified these snapshots
before accepting the metrics.

Configuration: static batching, batch size 2, static pipeline mapping,
bandwidth-aware placement, 16 original trace requests arriving every 10 ms,
and a one-second measurement window starting at t=0. Full model sequences and
original request token lengths were retained. Points were selected from the
VI-B Pareto reports used by the VI-F reproduction, without a new search.

## Measurements

| Model | Point | HYDRA tokens/s | HYDRA + CHIPSIM tokens/s | HYDRA TTFT (s) | HYDRA + CHIPSIM TTFT (s) | TTFT samples per backend |
|---|---|---:|---:|---:|---:|---:|
| LLaMA3-8B | P1 | 74 | 74 | 0.4275865 | 0.4276130 | 16 |
| LLaMA3-8B | P2 | 204 | 206 | 0.5850600 | 0.5850685 | 16 |
| LLaMA3-8B | P3 | 58 | 58 | 0.8016180 | 0.8020930 | 8 |
| Nemotron-H-4B | P1 | 64 | 64 | 0.2550980 | 0.2554440 | 16 |
| Nemotron-H-4B | P2 | 376 | 370 | 0.2871765 | 0.2872245 | 16 |
| Nemotron-H-4B | P3 | 91 | 91 | 0.6843133 | 0.6843713 | 6 |

All three pairwise design-point orderings agree for each metric and model.
Maximum absolute relative errors are 0.9804% TP / 0.0593% TTFT for LLaMA and
1.5957% TP / 0.1356% TTFT for Nemotron-H.

Generated local artifacts:

- TP vs. TTFT figure (`output_sanity_checks/chipsim_comparison/tp_vs_ttft.png`)
  (PDF (`output_sanity_checks/chipsim_comparison/tp_vs_ttft.pdf`)).
- Paired measurements and hashes (`output_sanity_checks/chipsim_comparison/comparison.csv`).
- Selected source configurations (`output_sanity_checks/chipsim_comparison/selected_points.json`).
- Error, ordering, and phase diagnostics (`output_sanity_checks/chipsim_comparison/summary.json`).

These generated files are local run outputs. Recreate them from the repository
root with `bash tests/validation/chipsim_validation/run.sh`. Add `--resume`
to reuse completed runs with matching commands. The six experiments measured
3,306 uncached Garnet bursts, with exact payload/path-class reuse within each
service. Detailed runs took about 13–28 minutes each with six experiments
running concurrently on this machine; runtime is hardware-dependent.

## Interpretation

This validates the static setup-to-execution-to-metrics path and consistent
trends under a shared serving model. Compute/SRAM profiles, admission, bandwidth
reservations, and runtime policies are shared with HYDRA. Garnet executes
isolated admitted transfers, without packet contention across concurrent
reservations.

Every admitted phase in these six runs was dominated by compute or reserved
HBM service time. Garnet transport never exceeded both of those components.
Consequently, the close agreement does **not** establish independent full-system
accuracy or validate HYDRA's network-contention approximation. HBM access
latency and time rounding differ between the execution backends, and the
resulting event schedules can also differ. A separate real-Garnet sensitivity
test confirmed that increasing router latency from 1 to 32 cycles changed an
isolated transfer from 1.03 to 4.679 microseconds.

The one-second window includes startup. TTFT is conditional on a first token
arriving before the cutoff; P3 has fewer TTFT samples. These curves are not
steady-state capacity measurements or reproductions of the archived paper
values. The archived points need not remain Pareto-optimal in this window.

## Verification

- 11 HYDRA adapter/profile/factory tests passed.
- 5 CHIPSIM runtime tests passed, including real Garnet router-latency feedback.
- The original standalone `chipsim` transformer-block demo passed with Garnet.
- Native execution matched the original processing implementation exactly for
  seeded LLaMA and Nemotron-H regression runs.
- Both installation patches applied to clean pinned upstream source files;
  all 13 patched/added files matched the tested working tree byte-for-byte.

See [the runtime guide](../../../integrations/chipsim/README.md) for the execution
contract, setup, individual configuration validation, and longer experiments.
