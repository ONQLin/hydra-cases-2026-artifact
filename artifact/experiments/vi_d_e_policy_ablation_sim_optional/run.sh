#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"
workers="${HYDRA_NUM_WORKERS:-16}"
output_root="output_temp_runs/vi_d_e_policy_ablation"

echo "[VI-D/E optional] Replaying the selected Fig. 9 simulations with ${workers} workers."
"${python_bin}" artifact/experiments/vi_d_e_policy_ablation_sim_optional/run_fig9_simulations.py \
    --workers "${workers}"
"${python_bin}" artifact/experiments/vi_d_e_policy_ablation_sim_optional/summarize_fig9_simulations.py
"${python_bin}" artifact/experiments/vi_d_e_policy_ablation/reproduce_fig9.py \
    --results-csv "${output_root}/processed/fig9_simulation_replay_results.csv" \
    --figure-dir "${output_root}/figures" \
    --figure-stem fig9_simulation_replay
echo "[VI-D/E optional] Outputs: ${output_root}/"
