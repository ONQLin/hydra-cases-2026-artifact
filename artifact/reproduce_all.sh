#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_root="${HYDRA_REPRO_LOG_DIR:-${repo_root}/output_temp_runs/artifact_reproduction/${timestamp}}"
summary_path="${log_root}/summary.tsv"
mkdir -p "${log_root}"

experiments=(
    "vi_b|Section VI-B: DSE on Macro-Architectures|artifact/experiments/vi_b_dse_on_macro_architectures/run.sh|artifact/run_outputs/vi_b_dse_on_macro_architectures|artifact/figures/vi_b_dse_on_macro_architectures"
    "vi_c|Section VI-C: Specialization and Generality|artifact/experiments/vi_c_specialization_study/run.sh|artifact/run_outputs/vi_c_specialization_study|artifact/figures/vi_c_specialization_study"
    "vi_d_e|Sections VI-D/E: Placement, Scheduling, and Batching Ablation|artifact/experiments/vi_d_e_policy_ablation/run.sh|artifact/run_outputs/vi_d_e_policy_ablation|artifact/figures/vi_d_e_policy_ablation"
    "vi_d|Section VI-D: Placement Strategies|artifact/experiments/vi_d_placement_strategies/run.sh|artifact/run_outputs/vi_d_placement_strategies|artifact/figures/vi_d_placement_strategies"
    "vi_e|Section VI-E: Task-Scheduling Strategies|artifact/experiments/vi_e_scheduler_strategies/run.sh|artifact/run_outputs/vi_e_scheduler_strategies|artifact/figures/vi_e_scheduler_strategies"
    "vi_f|Section VI-F: Performance Model for Fast DSE|artifact/experiments/vi_f_markov_fast_dse/run.sh|artifact/run_outputs/vi_f_markov_fast_dse|artifact/figures/vi_f_markov_fast_dse"
)

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter not found: ${python_bin}" >&2
    exit 2
fi

printf 'HYDRA artifact reproduction\n'
printf 'Repository: %s\n' "${repo_root}"
printf 'Python:     %s\n' "$("${python_bin}" --version 2>&1)"
printf 'Logs:       %s\n' "${log_root}"
printf 'Experiments: %d\n\n' "${#experiments[@]}"
printf 'experiment\tstatus\telapsed_seconds\tlog\tdata\tfigures\n' > "${summary_path}"

failures=0
index=0
suite_start="$(date +%s)"

for entry in "${experiments[@]}"; do
    IFS='|' read -r key description runner data_dir figure_dir <<< "${entry}"
    index=$((index + 1))
    log_path="${log_root}/${key}.log"
    start="$(date +%s)"

    printf '%s\n' "================================================================"
    printf '[%d/%d] Running %s\n' "${index}" "${#experiments[@]}" "${description}"
    printf 'Runner:  %s\n' "${runner}"
    printf 'Data:    %s\n' "${data_dir}"
    printf 'Figures: %s\n' "${figure_dir}"
    printf 'Log:     %s\n' "${log_path}"
    printf '%s\n' "----------------------------------------------------------------"

    if PYTHON_BIN="${python_bin}" bash "${runner}" 2>&1 | tee "${log_path}"; then
        status="PASS"
    else
        status="FAIL"
        failures=$((failures + 1))
    fi

    elapsed="$(( $(date +%s) - start ))"
    printf '%s status: %s (%ss)\n\n' "${description}" "${status}" "${elapsed}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${key}" "${status}" "${elapsed}" "${log_path}" "${data_dir}" "${figure_dir}" \
        >> "${summary_path}"
done

total_elapsed="$(( $(date +%s) - suite_start ))"
printf '%s\n' "================================================================"
if [[ "${failures}" -eq 0 ]]; then
    printf 'HYDRA reproduction completed: PASS (%ss)\n' "${total_elapsed}"
else
    printf 'HYDRA reproduction completed: %d experiment(s) FAILED (%ss)\n' \
        "${failures}" "${total_elapsed}"
fi
printf 'Summary: %s\n' "${summary_path}"
printf 'Figures: artifact/figures/\n'
printf 'Data:    artifact/run_outputs/\n'

exit "${failures}"
