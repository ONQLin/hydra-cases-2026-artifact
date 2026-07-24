#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"

echo "[VI-C] Reproducing the hybrid-LLM specialization table."
"${python_bin}" artifact/experiments/vi_c_specialization_study/reproduce_specialization_table.py
echo "[VI-C] Data: artifact/run_outputs/vi_c_specialization_study/"
echo "[VI-C] Figure: artifact/figures/vi_c_specialization_study/table_iv_reproduced.png"
