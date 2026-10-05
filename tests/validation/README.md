# Simulator validation

Extension validation lives here; the original paper reproduction remains in
[`artifact/`](../../artifact/README.md).

| Suite | Coverage |
|---|---|
| [CHIPSIM](chipsim_validation/README.md) | Setup export, serving feedback, contention and DMA pacing |
| [Packet and fidelity comparison](packet_validation/README.md) | Network checks, TP/TTFT shifts and simulation cost |

Run `python -m unittest discover -s tests` from the repository root.
Unit tests construct small fixtures in Python and use temporary directories.
Validation scripts select archived paper design points or generate network cases.
Write simulation outputs to ignored `output_sanity_checks/` directories.

The measured six-case comparison and its compact coverage record are retained
under [`docs/multi_fidelity/data/`](../../docs/multi_fidelity/data/). These support
the documentation figures; generated run dumps and duplicate plots are omitted.
Backend setup and execution commands are in [`integrations/`](../../integrations/README.md).
