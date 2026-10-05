# Multi-fidelity validation

HYDRA-Analytic, HYDRA-Packet and CHIPSIM/Garnet share compute profiles, hardware,
workload and runtime settings. These checks validate network refinement and its
feedback into serving. They do not independently calibrate accelerator compute.

## Unit and network checks

From the repository root with the HYDRA environment active:

```bash
python -m unittest tests.test_packet_network tests.test_fidelity_comparison tests.test_fidelity_figures
```

These tests construct networks, flows and paired records in temporary directories.
They check conservation, contention, input equivalence and metric aggregation.
For an optional real Garnet comparison, complete the
[CHIPSIM setup](../../../integrations/chipsim/README.md), then run:

```bash
.chipsim-venv/bin/python tests/validation/packet_validation/validate_network.py \
  --garnet --stress --quanta 256 1024 \
  --output-dir output_sanity_checks/packet_validation/network
```

The script generates the cases and checks their results. Flow records and timing
reports stay in the selected output directory.

## Serving comparison

The [Packet guide](../../../integrations/packet/README.md) and
[CHIPSIM guide](../../../integrations/chipsim/README.md) give paired-run commands
using archived paper design points. `compare_levels.py` verifies identical
inputs and completed flows, then generates TP/TTFT shifts, ordering and cost
reports. `repeat_serving.py` is available for timing variability studies; repeated
run dumps are not committed.

The retained [six-case CSV](../../../docs/multi_fidelity/data/comparison.csv)
covers LLaMA3-8B and Nemotron-H-4B, three designs each, 32 Chat requests and five
simulated seconds, with static batch size 2 and unpaced injection. Its small
[coverage record](../../../docs/multi_fidelity/data/summary.json) guards the
window and paired-case count. CSV bytes and measured values are unchanged.

[Results and measurement scope](../../../docs/multi_fidelity/README.md) contain
the figures and accuracy/cost table. Regenerate their PNG/PDF/SVG exports:

```bash
python tests/validation/packet_validation/plot_reference_comparison.py \
  --comparison-csv docs/multi_fidelity/data/comparison.csv \
  --window 5 --figure-set documentation \
  --output-dir output_sanity_checks/packet_validation/figures
```

CHIPSIM is the comparison reference under shared compute. TTFT retains the
legacy scheduled-to-prefill convention, excluding initial queue delay. These
finite-window results and startup-inclusive wall times do not establish a
universal fidelity ranking or steady-state serving performance. Earlier raw
checkpoints and duplicate plots can be regenerated instead of versioned.
