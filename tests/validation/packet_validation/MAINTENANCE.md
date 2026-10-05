# Multi-fidelity maintenance

## Evidence and generated outputs

Retain the six-case source CSV and compact coverage record under
`docs/multi_fidelity/data/`, and the presentation figures and aggregation summary
under `docs/multi_fidelity/figures/`. Generate network cases and simulator reports
with the validation scripts; store new outputs under ignored
`output_sanity_checks/` directories. Historical raw runs and duplicate plots are
not required to regenerate the retained figures.

## Code review and regression checks

Simulation and execution providers retain HYDRA's subclass factories. Shared
classes own policy checks, system export, phase sequencing and conservative
time synchronization. The C++ provider owns its events behind a C ABI.
See the [extension guide](../../../integrations/README.md).

Cleanup tests cover repeated closure, failed summary writes, broken pipes,
timeout handling and preservation of caller-owned request messages.
Run from the repository root:

```bash
python -m unittest discover -s tests
RUN_CHIPSIM_GARNET_TESTS=1 .chipsim-venv/bin/python -m unittest \
  tests.test_chipsim_runtime_service \
  tests.test_chipsim_contention.PacketContentionTest tests.test_chipsim_dma_pacing
```

The optional command requires the [CHIPSIM setup](../../../integrations/chipsim/README.md).
These checks exercise lifecycle and communication contracts; measured serving
accuracy and simulation cost are described in the
[evidence guide](../../../docs/multi_fidelity/README.md).
