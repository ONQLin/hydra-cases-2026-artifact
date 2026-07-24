#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"

echo "[VI-D/E] Reproducing the CP, CP+ET, and CP+ET+DB policy ablation."
"${python_bin}" artifact/experiments/vi_d_e_policy_ablation/reproduce_fig9.py
echo "[VI-D/E] Data: artifact/run_outputs/vi_d_e_policy_ablation/"
echo "[VI-D/E] Figure: artifact/figures/vi_d_e_policy_ablation/fig9_policy_ablation.png"
