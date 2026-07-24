#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"
workers="${HYDRA_NUM_WORKERS:-16}"
runner_args=(--workers "${workers}")
if [[ "${HYDRA_REUSE_COMPLETED:-0}" != "1" ]]; then
    runner_args+=(--rerun-completed)
fi

echo "[VI-E] Running Static, FCFS, Work-stealing, and Elastic scheduling with ${workers} workers."
"${python_bin}" artifact/experiments/vi_e_scheduler_strategies/run_fig12_simulations.py \
    "${runner_args[@]}"

echo "[VI-E] Post-processing the scheduler comparison."
"${python_bin}" artifact/experiments/vi_e_scheduler_strategies/reproduce_fig12.py
echo "[VI-E] Data: artifact/run_outputs/vi_e_scheduler_strategies/fig12/"
echo "[VI-E] Figure: artifact/figures/vi_e_scheduler_strategies/fig12_scheduler_bars.png"
