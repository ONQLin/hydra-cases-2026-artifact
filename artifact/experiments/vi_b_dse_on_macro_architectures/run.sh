#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"

echo "[VI-B] Reproducing the exhaustive throughput-versus-TTFT panels."
"${python_bin}" artifact/experiments/vi_b_dse_on_macro_architectures/reproduce_fig8_exhaustive.py
echo "[VI-B] Data: artifact/run_outputs/vi_b_dse_on_macro_architectures/"
echo "[VI-B] Figures: artifact/figures/vi_b_dse_on_macro_architectures/"
