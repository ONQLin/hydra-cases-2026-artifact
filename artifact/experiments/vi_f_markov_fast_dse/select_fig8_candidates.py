#!/usr/bin/env python3
"""Retain static-state candidates for the second stage of Fig. 8 DSE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Fast_Estimate.fast_estimate_models as estimator
from Fast_Estimate.static_state_filter import CONFIG_FIELDS
from Fast_Estimate.static_state_filter import evaluate_states
from Fast_Estimate.static_state_filter import retain_candidates


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "artifact/run_outputs/vi_f_markov_fast_dse/fig8_static_static"
)
DEFAULT_SUMMARY_ROOT = (
    REPO_ROOT
    / "artifact/run_outputs/vi_b_dse_on_macro_architectures/raw_data/Summary_Reports"
)
DEFAULT_PROFILE_ROOT = EXPERIMENT_DIR / "workload_profiles"
DEFAULT_CACHE_ROOT = REPO_ROOT / "output_temp_runs/vi_f_markov_fast_dse"

MODELS = ("LLAMA3", "MAMBA2", "NEMO")
DATASETS = ("ARXIV", "BWB", "CHAT", "LW")
DEFAULT_CANDIDATE_BUDGET = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=DEFAULT_CANDIDATE_BUDGET,
        help="Number of configurations retained per model/dataset pair.",
    )
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    missing = [
        args.summary_root
        / f"results_summary_{model}-{dataset}_bw_static_static.csv"
        for model in args.models
        for dataset in args.datasets
        if not (
            args.summary_root
            / f"results_summary_{model}-{dataset}_bw_static_static.csv"
        ).is_file()
    ]
    missing.extend(
        args.profile_root / f"{model.lower()}_{dataset.lower()}.json"
        for model in args.models
        for dataset in args.datasets
        if not (args.profile_root / f"{model.lower()}_{dataset.lower()}.json").is_file()
    )
    if missing:
        raise FileNotFoundError(
            "Missing static-filter inputs:\n" + "\n".join(map(str, missing))
        )


def filter_workload(
    model: str,
    dataset: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, int]:
    profile_data, mapping_entries, filtered_results = (
        estimator.ensure_dataset_candidates(
            model=model,
            dataset=dataset,
            summary_root=args.summary_root,
            placement_cache_dir=args.cache_root / "placement_cache",
            mapping_cache_dir=args.cache_root / "mapping_cache",
        )
    )

    def key_for_config(config: dict) -> str:
        return estimator.mapping_key(
            model,
            dataset,
            config,
            profile_data,
        )

    evaluated = evaluate_states(
        rows=filtered_results,
        profile_data=profile_data,
        mapping_entries=mapping_entries,
        mapping_key_for_config=key_for_config,
    )
    retained = retain_candidates(
        evaluated,
        candidate_budget=args.candidate_budget,
    )
    retained.insert(0, "dataset", dataset)
    retained.insert(0, "model", model)
    return retained, len(evaluated)


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.summary_root = args.summary_root.resolve()
    args.profile_root = args.profile_root.resolve()
    args.cache_root = args.cache_root.resolve()
    if args.candidate_budget <= 0:
        raise ValueError("--candidate-budget must be positive.")

    require_inputs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    estimator.PROFILE_ROOT = args.profile_root

    frames: list[pd.DataFrame] = []
    for model in args.models:
        for dataset in args.datasets:
            retained, valid_count = filter_workload(
                model,
                dataset,
                args,
            )
            frames.append(retained)
            print(
                f"{model}-{dataset}: retained {len(retained)} "
                f"of {valid_count} valid static states."
            )

    manifest = pd.concat(frames, ignore_index=True)
    output_columns = [
        "model",
        "dataset",
        "candidate_rank",
        "selection_sources",
        *CONFIG_FIELDS,
        "static_mapper_throughput",
        "static_mapper_ttft_s",
        "estimated_throughput",
        "estimated_prefill_service_s",
        "throughput_per_prefill_s",
        "stage_balance",
        "prefill_pressure",
        "bandwidth_pressure",
    ]
    manifest_path = args.output_root / "fig8_filter_candidates.csv"
    manifest.to_csv(
        manifest_path,
        columns=output_columns,
        index=False,
        float_format="%.12g",
    )

    print(f"Wrote candidate manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
