#!/usr/bin/env python3
"""Evaluate 64-candidate M_T recovery on Nemotron-H ET+DB profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fast_Estimate.fast_estimate_models import MODEL_CONFIG_NAMES
from Fast_Estimate.fast_estimate_models import estimate_runtime_markov
from Fast_Estimate.fast_estimate_models import runtime_chiplet_service_rates
from Fast_Estimate.fast_estimate_models import runtime_state_capacities
from Sim.config.model_config import BaseModelConfig


MODEL = "NEMO"
DATASETS = ("ARXIV", "BWB", "CHAT", "LW")
CONFIG_FIELDS = (
    "batchsize",
    "Num M",
    "Num Mp",
    "Num Md",
    "Num Ap",
    "Num Ad",
    "NoI_bw(GBps)",
)
DEFAULT_DATA_ROOT = (
    REPO_ROOT
    / "artifact/run_outputs/vi_f_markov_fast_dse/nemo_et_db_profiles"
)
DEFAULT_MEASUREMENTS = DEFAULT_DATA_ROOT / "nemo_et_db_measurements.csv"
DEFAULT_SCORES = DEFAULT_DATA_ROOT / "nemo_et_db_markov_candidates.csv"
DEFAULT_RECOVERY = DEFAULT_DATA_ROOT / "nemo_et_db_mt_recovery.csv"
DEFAULT_PROFILE_ROOT = Path(__file__).resolve().parent / "workload_profiles"
TRACE_FILES = {
    "ARXIV": REPO_ROOT / "dataset/arxiv/arxiv_summarization_stats_nemo.csv",
    "BWB": REPO_ROOT / "dataset/bwb/bwb_translation_stats_nemo.csv",
    "CHAT": REPO_ROOT / "dataset/chat/chat1m_stats_nemo.csv",
    "LW": REPO_ROOT / "dataset/longwriter/long_writer_nemo_6k.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--recovery-output", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument("--candidate-budget", type=int, default=64)
    parser.add_argument("--prefilter-budget", type=int, default=2048)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--step-s", type=float, default=0.01)
    return parser.parse_args()


def load_measurements(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.measurements)
    required = {
        "model",
        "dataset",
        "tokens_per_sec",
        "TTFT(s)",
        *CONFIG_FIELDS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")
    frame = frame[
        (frame["model"] == MODEL)
        & frame["dataset"].isin(args.datasets)
        & frame["tokens_per_sec"].gt(0)
        & frame["TTFT(s)"].gt(0)
    ].drop_duplicates(["dataset", *CONFIG_FIELDS], keep="last")
    missing_datasets = set(args.datasets) - set(frame["dataset"])
    if missing_datasets:
        raise ValueError(
            f"Missing Nemotron-H ET+DB datasets: {sorted(missing_datasets)}"
        )
    return frame.reset_index(drop=True)


def config_from_row(row: pd.Series) -> dict[str, int]:
    return {field: int(row[field]) for field in CONFIG_FIELDS}


def service_rate_key(config: dict[str, int]) -> tuple[int, int, int]:
    return (
        config["batchsize"],
        config["Num M"],
        config["NoI_bw(GBps)"],
    )


def prefilter_candidates(
    measurements: pd.DataFrame,
    profile_data: dict,
    model_config: BaseModelConfig,
    budget: int,
) -> tuple[pd.DataFrame, dict[tuple[int, int, int], dict]]:
    service_cache: dict[tuple[int, int, int], dict] = {}
    records: list[dict[str, float | int]] = []
    prefill_tokens = float(profile_data["prefill_len"])
    decode_tokens = float(profile_data["decode_len"])

    for _, row in measurements.iterrows():
        config = config_from_row(row)
        key = service_rate_key(config)
        if key not in service_cache:
            service_cache[key] = runtime_chiplet_service_rates(
                model_config,
                config,
                profile_data,
            )
        states = runtime_state_capacities(
            config,
            profile_data,
            service_cache[key],
        )
        max_prefill = max(float(state["Gp"]) for state in states)
        max_decode = max(float(state["Gd"]) for state in states)
        request_cycle_s = (
            prefill_tokens / max(max_prefill, 1e-12)
            + decode_tokens / max(max_decode, 1e-12)
        )
        records.append(
            {
                **config,
                "prefilter_throughput": (
                    decode_tokens / max(request_cycle_s, 1e-12)
                ),
            }
        )

    scores = pd.DataFrame(records)
    retained = scores.nlargest(
        min(budget, len(scores)),
        "prefilter_throughput",
        keep="all",
    ).head(budget)
    return retained.reset_index(drop=True), service_cache


def evaluate_markov_candidates(
    dataset: str,
    retained: pd.DataFrame,
    measurements: pd.DataFrame,
    profile_data: dict,
    model_config: BaseModelConfig,
    service_cache: dict[tuple[int, int, int], dict],
    args: argparse.Namespace,
) -> pd.DataFrame:
    trace = pd.read_csv(TRACE_FILES[dataset])
    request_trace = (
        trace["num_prefill_tokens"].to_numpy(dtype=float),
        trace["num_decode_tokens"].to_numpy(dtype=float),
    )
    records: list[dict[str, float | int | str]] = []
    for rank, (_, row) in enumerate(retained.iterrows(), start=1):
        config = config_from_row(row)
        result = estimate_runtime_markov(
            model_config=model_config,
            config=config,
            profile_data=profile_data,
            request_trace=request_trace,
            duration_s=args.duration_s,
            step_s=args.step_s,
            allocation_interval_s=args.step_s,
            service_rates=service_cache[service_rate_key(config)],
            enable_token_budget_admission=False,
        )
        occupied_states = sum(
            state["pi_s"] > 0 for state in result["state_occupancy"]
        )
        records.append(
            {
                "model": MODEL,
                "dataset": dataset,
                "prefilter_rank": rank,
                **config,
                "prefilter_throughput": float(row["prefilter_throughput"]),
                "markov_estimated_throughput": result[
                    "estimated_throughput"
                ],
                "markov_realized_fluid_throughput": result[
                    "realized_fluid_throughput"
                ],
                "occupied_states": occupied_states,
                "state_transitions": sum(
                    transition["count"]
                    for transition in result["transition_rates"]
                ),
            }
        )
    scores = pd.DataFrame(records)
    return scores.merge(
        measurements[
            [*CONFIG_FIELDS, "tokens_per_sec", "TTFT(s)"]
        ],
        on=list(CONFIG_FIELDS),
        how="left",
        validate="one_to_one",
    ).sort_values(
        ["markov_estimated_throughput", "prefilter_throughput"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def config_payload(row: pd.Series, prefix: str) -> dict[str, int]:
    return {
        f"{prefix}_{field.replace(' ', '_').replace('(GBps)', 'gbps')}": int(
            row[field]
        )
        for field in CONFIG_FIELDS
    }


def measure_recovery(
    dataset: str,
    scores: pd.DataFrame,
    measurements: pd.DataFrame,
    budget: int,
) -> dict[str, float | int | bool | str]:
    truth = measurements.loc[measurements["tokens_per_sec"].idxmax()]
    retained = scores.head(min(budget, len(scores)))
    selected = retained.loc[retained["tokens_per_sec"].idxmax()]
    truth_key = tuple(int(truth[field]) for field in CONFIG_FIELDS)
    retained_keys = {
        tuple(int(row[field]) for field in CONFIG_FIELDS)
        for _, row in retained.iterrows()
    }
    return {
        "model": MODEL,
        "dataset": dataset,
        "policy": "bw_vllm_elastic",
        "budget": budget,
        "full_sweep_points": len(measurements),
        "prefilter_points": len(scores),
        "retained_fraction": budget / len(measurements),
        "throughput_recovery": (
            float(selected["tokens_per_sec"])
            / float(truth["tokens_per_sec"])
        ),
        "optimal_config_retained": truth_key in retained_keys,
        "optimal_throughput": float(truth["tokens_per_sec"]),
        "recovered_throughput": float(selected["tokens_per_sec"]),
        **config_payload(truth, "optimal"),
        **config_payload(selected, "recovered"),
    }


def main() -> int:
    args = parse_args()
    if args.candidate_budget <= 0:
        raise ValueError("--candidate-budget must be positive.")
    if args.prefilter_budget < args.candidate_budget:
        raise ValueError("--prefilter-budget must cover --candidate-budget.")
    if args.duration_s <= 0 or args.step_s <= 0:
        raise ValueError("Runtime duration and step size must be positive.")

    measurements = load_measurements(args)
    model_config = BaseModelConfig.create_from_name(MODEL_CONFIG_NAMES[MODEL])
    score_frames: list[pd.DataFrame] = []
    recovery_rows: list[dict] = []

    for dataset in args.datasets:
        dataset_measurements = measurements[
            measurements["dataset"] == dataset
        ].reset_index(drop=True)
        profile_data = json.loads(
            (
                args.profile_root / f"nemo_{dataset.lower()}.json"
            ).read_text(encoding="utf-8")
        )
        retained, service_cache = prefilter_candidates(
            dataset_measurements,
            profile_data,
            model_config,
            args.prefilter_budget,
        )
        scores = evaluate_markov_candidates(
            dataset,
            retained,
            dataset_measurements,
            profile_data,
            model_config,
            service_cache,
            args,
        )
        score_frames.append(scores)
        recovery_rows.append(
            measure_recovery(
                dataset,
                scores,
                dataset_measurements,
                args.candidate_budget,
            )
        )
        print(f"{dataset}: evaluated {len(scores)} Markov candidates.")

    all_scores = pd.concat(score_frames, ignore_index=True)
    per_dataset = pd.DataFrame(recovery_rows)
    recovery = pd.DataFrame(
        [
            {
                "model": MODEL,
                "objective": "MT",
                "policy": "bw_vllm_elastic",
                "datasets": "+".join(args.datasets),
                "candidate_budget_per_dataset": args.candidate_budget,
                "prefilter_budget_per_dataset": args.prefilter_budget,
                "average_throughput_recovery": per_dataset[
                    "throughput_recovery"
                ].mean(),
            }
        ]
    )
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(args.scores_output, index=False, float_format="%.12g")
    recovery.to_csv(args.recovery_output, index=False, float_format="%.12g")

    average_recovery = float(recovery.iloc[0]["average_throughput_recovery"])
    print(
        f"M_T recovery with {args.candidate_budget} candidates per dataset: "
        f"{100.0 * average_recovery:.2f}%"
    )
    print(f"Wrote {args.scores_output}")
    print(f"Wrote {args.recovery_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
