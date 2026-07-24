#!/usr/bin/env bash
set -euo pipefail

# Stage only the files added or changed for the VI-B through VI-F artifact.
# Existing tracked dependencies such as main.py, dataset/, analytic_profile/,
# and the rest of Sim/ are already part of the repository.

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

git_args=(add)
if [[ "${1:-}" == "--dry-run" ]]; then
    git_args+=(--dry-run)
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

git "${git_args[@]}" -- \
    .gitignore \
    README.md \
    requirements.txt \
    sim_info.md \
    git_add_artifact.sh \
    artifact/README.md \
    artifact/reproduce_all.sh \
    artifact/experiments/ \
    artifact/figures/ \
    artifact/run_outputs/ \
    Fast_Estimate/Rooflines_est.py \
    Fast_Estimate/fast_estimate_models.py \
    Fast_Estimate/static_state_filter.py \
    Sim/common.py \
    Sim/config/sys_config.py \
    Sim/entities/mem_sys.py \
    Sim/placer/random_placer.py \
    Sim/processing.py \
    Sim/scheduler/vllm.py \
    Sim/scheduler/vllm_latest.py \
    Sim/simulator.py \
    Sim/task_scheduler.py
