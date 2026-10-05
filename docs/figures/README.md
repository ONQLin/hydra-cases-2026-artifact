# Framework overview figure

`hydra_overview.svg` reproduces Figure 2 from the HYDRA paper,
*HYDRA: A Heterogeneous Chiplet DSE Framework for Serving Dynamic Hybrid LLM
Workloads*. The section numbers in the figure refer to the paper.

Source: a HYDRA manuscript revision, page 3. The source revision is identified
by the SHA-256 below; it differs from the release PDF `HYDRA_paper.pdf`. Only
the figure is included here; no additional PDF is required to render the README.

- Source PDF SHA-256: `97dff3953a6339ec870d8167720725598cea482073183a16b47392fe6688a32e`.
- Extracted using Poppler `pdftocairo -f 3 -l 3 -svg`; figure bounds in PDF points:
  `x=303, y=70, width=257, height=256`.
- Content outside the figure and unused glyph definitions were removed. The
  original vector drawing and labels are retained, with a white background
  and an accessible SVG title/description.

The subsequent [multi-fidelity diagram](../multi_fidelity/architecture.svg)
illustrates the network-backend extension. Its comparison data and assumptions
are documented in [multi-fidelity evidence](../multi_fidelity/README.md).
