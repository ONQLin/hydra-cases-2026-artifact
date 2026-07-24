#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"
workers="${HYDRA_NUM_WORKERS:-8}"
runner_args=(--workers "${workers}")
if [[ "${HYDRA_REUSE_COMPLETED:-0}" != "1" ]]; then
    runner_args+=(--rerun-completed)
fi

echo "[VI-D] Running the selected Random-placement simulations with ${workers} workers."
"${python_bin}" artifact/experiments/vi_d_placement_strategies/run_fig10_simulations.py \
    "${runner_args[@]}"

echo "[VI-D] Combining the new Random results with the prepared RR and CP baselines."
"${python_bin}" artifact/experiments/vi_d_placement_strategies/reproduce_fig10.py
echo "[VI-D] Data: artifact/run_outputs/vi_d_placement_strategies/fig10/"
echo "[VI-D] Figure: artifact/figures/vi_d_placement_strategies/fig10_placement_bars.png"
