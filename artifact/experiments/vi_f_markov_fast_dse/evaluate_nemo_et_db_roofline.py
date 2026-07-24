#!/usr/bin/env python3
"""Evaluate the static roofline baseline on Nemotron-H ET+DB profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fast_Estimate.Rooflines_est import CONFIG_FIELDS
from Fast_Estimate.Rooflines_est import score_configurations
from Fast_Estimate.fast_estimate_models import load_cache_entries


MODEL = "NEMO"
DATASETS = ("ARXIV", "BWB", "CHAT", "LW")
DEFAULT_DATA_ROOT = (
    REPO_ROOT
    / "artifact/run_outputs/vi_f_markov_fast_dse/nemo_et_db_profiles"
)
DEFAULT_MEASUREMENTS = DEFAULT_DATA_ROOT / "nemo_et_db_measurements.csv"
DEFAULT_SCORES = DEFAULT_DATA_ROOT / "nemo_et_db_roofline_candidates.csv"
DEFAULT_RECOVERY = DEFAULT_DATA_ROOT / "nemo_et_db_roofline_mt_recovery.csv"
DEFAULT_PROFILE_ROOT = Path(__file__).resolve().parent / "workload_profiles"
DEFAULT_MAPPING_CACHE = (
    REPO_ROOT
    / "output_temp_runs/vi_f_markov_fast_dse/mapping_cache"
    / "mapping_cache_nemo.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--mapping-cache", type=Path, default=DEFAULT_MAPPING_CACHE)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--recovery-output", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--candidate-budget", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_budget <= 0:
        raise ValueError("--candidate-budget must be positive.")

    measurements = pd.read_csv(args.measurements)
    mapping_entries = load_cache_entries(args.mapping_cache)
    score_frames: list[pd.DataFrame] = []
    recoveries: list[float] = []

    for dataset in DATASETS:
        profile_data = json.loads(
            (
                args.profile_root / f"nemo_{dataset.lower()}.json"
            ).read_text(encoding="utf-8")
        )
        dataset_measurements = measurements[
            measurements["dataset"] == dataset
        ].drop_duplicates(list(CONFIG_FIELDS), keep="last")
        scores = score_configurations(
            MODEL,
            dataset,
            (row for _, row in dataset_measurements.iterrows()),
            profile_data,
            mapping_entries,
        )
        scores.insert(0, "dataset", dataset)
        scores.insert(0, "model", MODEL)
        scores = scores.merge(
            dataset_measurements[
                [*CONFIG_FIELDS, "tokens_per_sec", "TTFT(s)"]
            ],
            on=list(CONFIG_FIELDS),
            how="inner",
            validate="one_to_one",
        ).sort_values(
            "roofline_throughput",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)
        scores["roofline_rank"] = scores.index + 1
        score_frames.append(scores)

        retained = scores.head(args.candidate_budget)
        recoveries.append(
            float(retained["tokens_per_sec"].max())
            / float(dataset_measurements["tokens_per_sec"].max())
        )

    all_scores = pd.concat(score_frames, ignore_index=True)
    recovery = pd.DataFrame(
        [
            {
                "model": MODEL,
                "objective": "MT",
                "estimator": "static_roofline",
                "datasets": "+".join(DATASETS),
                "candidate_budget_per_dataset": args.candidate_budget,
                "average_throughput_recovery": sum(recoveries) / len(recoveries),
            }
        ]
    )
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(args.scores_output, index=False, float_format="%.12g")
    recovery.to_csv(args.recovery_output, index=False, float_format="%.12g")

    average_recovery = float(recovery.iloc[0]["average_throughput_recovery"])
    print(
        f"Roofline M_T recovery with {args.candidate_budget} candidates "
        f"per dataset: {100.0 * average_recovery:.2f}%"
    )
    print(f"Wrote {args.scores_output}")
    print(f"Wrote {args.recovery_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
