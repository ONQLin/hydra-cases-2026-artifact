#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"
workers="${HYDRA_NUM_WORKERS:-32}"

echo "[VI-C optional] Running the Jamba, Zamba, and Nemotron-H DSE with ${workers} workers."
"${python_bin}" artifact/experiments/vi_c_hybrid_llm_dse_optional/run_hybrid_llm_dse.py \
    --num-in-parallel "${workers}"
"${python_bin}" artifact/experiments/vi_c_hybrid_llm_dse_optional/summarize_hybrid_llm_dse.py
echo "[VI-C optional] Outputs: output_temp_runs/vi_c_hybrid_llm_dse/"
