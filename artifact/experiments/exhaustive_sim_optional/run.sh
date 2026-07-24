#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"
workers="${HYDRA_NUM_WORKERS:-32}"

echo "[VI-B optional] Running the exhaustive simulator sweep with ${workers} workers."
"${python_bin}" artifact/experiments/exhaustive_sim_optional/run_exhaustive_dse.py \
    --num-in-parallel "${workers}"
"${python_bin}" artifact/experiments/exhaustive_sim_optional/summarize_exhaustive_dse.py
echo "[VI-B optional] Outputs: output_temp_runs/vi_b_exhaustive_dse/"
