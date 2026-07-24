#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"

echo "[VI-F] Selecting the 256 fast-DSE candidates used in Fig. 8."
"${python_bin}" artifact/experiments/vi_f_markov_fast_dse/select_fig8_candidates.py

echo "[VI-F] Reproducing the complete Fig. 8."
"${python_bin}" artifact/experiments/vi_f_markov_fast_dse/reproduce_fig8_complete.py

echo "[VI-F] Evaluating Roofline and Markov-guided search with a 64-candidate budget."
"${python_bin}" artifact/experiments/vi_f_markov_fast_dse/evaluate_nemo_et_db_roofline.py
"${python_bin}" artifact/experiments/vi_f_markov_fast_dse/evaluate_nemo_et_db_throughput.py
"${python_bin}" artifact/experiments/vi_f_markov_fast_dse/plot_estimator_recovery.py

echo "[VI-F] Data: artifact/run_outputs/vi_f_markov_fast_dse/"
echo "[VI-F] Figures: artifact/figures/vi_f_markov_fast_dse/"
