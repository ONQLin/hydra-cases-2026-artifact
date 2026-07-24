#!/usr/bin/env python3
"""Summarize optional Fig. 9 replays using the prepared RR normalization data."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifact"
    / "run_outputs"
    / "vi_d_e_policy_ablation"
    / "raw_data"
    / "fig9_configurations.csv"
)
DEFAULT_MEASUREMENTS = (
    REPO_ROOT
    / "artifact"
    / "run_outputs"
    / "vi_d_e_policy_ablation"
    / "raw_data"
    / "fig9_measurements.csv"
)
DEFAULT_RUN_ROOT = (
    REPO_ROOT / "output_temp_runs" / "vi_d_e_policy_ablation" / "simulator_runs"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "output_temp_runs"
    / "vi_d_e_policy_ablation"
    / "processed"
    / "fig9_simulation_replay_results.csv"
)
MODEL_ORDER = ("LLAMA3", "MAMBA2", "NEMO")
DATASET_ORDER = ("ARXIV", "LW", "CHAT", "BWB")
OBJECTIVE_ORDER = ("B_star", "M_T")
POLICY_ORDER = ("rr_baseline", "cp", "cp_et", "cp_et_db")
DEFAULT_POLICY_ORDER = ("cp", "cp_et", "cp_et_db")


@dataclass(frozen=True)
class Selection:
    model: str
    dataset: str
    objective: str
    policy: str
    placer: str
    request_scheduler: str
    task_scheduler: str
    batchsize: int
    num_m: int
    num_mp: int
    num_md: int
    num_ap: int
    num_ad: int
    bandwidth: int
    reference_tp: float
    reference_ttft: float
    reference_baseline_tp: float
    reference_baseline_ttft: float

    @property
    def label(self) -> str:
        return (
            f"FIG9_{self.model}_{self.dataset}_{self.objective}_{self.policy}_"
            f"BW{self.bandwidth}_BS{self.batchsize}_M{self.num_m}_Ap{self.num_ap}_"
            f"Ad{self.num_ad}_Mp{self.num_mp}_Md{self.num_md}"
        )

    def run_parent(self, root: Path) -> Path:
        objective = "bstar" if self.objective == "B_star" else "mt"
        return root / self.model.lower() / self.dataset.lower() / objective / self.policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=DEFAULT_MEASUREMENTS,
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER))
    parser.add_argument("--objectives", default=",".join(OBJECTIVE_ORDER))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICY_ORDER))
    return parser.parse_args()


def parse_list(raw: str, valid: tuple[str, ...], name: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"At least one {name} value is required.")
    invalid = sorted(set(values) - set(valid))
    if invalid:
        raise ValueError(f"Invalid {name}: {', '.join(invalid)}")
    return values


def load_selections(path: Path) -> list[Selection]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selections = [
        Selection(
            model=row["model"],
            dataset=row["dataset"],
            objective=row["objective"],
            policy=row["policy"],
            placer=row["placer"],
            request_scheduler=row["request_scheduler"],
            task_scheduler=row["task_scheduler"],
            batchsize=int(row["batchsize"]),
            num_m=int(row["Num M"]),
            num_mp=int(row["Num Mp"]),
            num_md=int(row["Num Md"]),
            num_ap=int(row["Num Ap"]),
            num_ad=int(row["Num Ad"]),
            bandwidth=int(row["NoI_bw(GBps)"]),
            reference_tp=float(row["reference_tokens_per_sec"]),
            reference_ttft=float(row["reference_TTFT(s)"]),
            reference_baseline_tp=float(row["baseline_tokens_per_sec"]),
            reference_baseline_ttft=float(row["baseline_TTFT(s)"]),
        )
        for row in rows
    ]
    keys = [(item.model, item.dataset, item.objective, item.policy) for item in selections]
    if len(keys) != len(set(keys)):
        raise ValueError("The Fig. 9 configuration table contains duplicate rows.")
    expected = {
        (model, dataset, objective, policy)
        for model in MODEL_ORDER
        for dataset in DATASET_ORDER
        for objective in OBJECTIVE_ORDER
        for policy in POLICY_ORDER
    }
    if set(keys) != expected:
        raise ValueError("The Fig. 9 configuration table must contain all fixed runs.")
    return selections


def load_db_scales(path: Path) -> dict[tuple[str, str, str, str], tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scales = {
        (row["model"], row["dataset"], row["objective"], row["policy"]): (
            float(row["throughput_scale"]),
            float(row["TTFT_scale"]),
        )
        for row in rows
    }
    expected = {
        (model, dataset, objective, policy)
        for model in MODEL_ORDER
        for dataset in DATASET_ORDER
        for objective in OBJECTIVE_ORDER
        for policy in POLICY_ORDER
    }
    if set(scales) != expected:
        raise ValueError("The measurement table must contain all fixed runs.")
    return scales


def suffix_float(line: str) -> float | None:
    match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", line.strip())
    return float(match.group(1)) if match else None


def parse_metrics(path: Path) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "tokens_per_sec": None,
        "TTFT(s)": None,
        "completed_requests": None,
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "output tokens per second:" in line:
            result["tokens_per_sec"] = suffix_float(line)
        elif "Average time to first token:" in line:
            value = suffix_float(line)
            result["TTFT(s)"] = None if value is None else value / 1_000_000.0
        elif "Request counters - completed:" in line:
            match = re.search(r"completed:\s*(\d+)", line)
            result["completed_requests"] = int(match.group(1)) if match else None
        elif "In total" in line and "requests were processed" in line:
            match = re.search(r"In total\s+(\d+)\s+requests", line)
            result["completed_requests"] = int(match.group(1)) if match else None
    return result


def parse_alloc(value: str | dict) -> dict:
    return value if isinstance(value, dict) else ast.literal_eval(value)


def validate_config(selection: Selection, config: dict, run_dir: Path) -> None:
    alloc = parse_alloc(config["placmt_config"]["chiplet_alloc"])
    actual = (
        int(config["cluster_config"]["batch_size"]),
        int(alloc["HBM3"]),
        int(alloc["marca_p"]),
        int(alloc["marca_d"]),
        int(alloc["tscs_p"]),
        int(alloc["tscs_d"]),
        int(float(config["chips_config"]["D2D_NoI_bw"])),
        config["placmt_config"]["placer_label"],
        config["cluster_config"]["local_scheduler"],
        config["mapping_config"]["mapping_strategy"],
    )
    expected = (
        selection.batchsize,
        selection.num_m,
        selection.num_mp,
        selection.num_md,
        selection.num_ap,
        selection.num_ad,
        selection.bandwidth,
        selection.placer,
        selection.request_scheduler,
        selection.task_scheduler,
    )
    if actual != expected:
        raise ValueError(
            f"Configuration mismatch in {run_dir}: expected {expected}, found {actual}"
        )


def find_run(selection: Selection, root: Path) -> Path:
    candidates = []
    for run_dir in selection.run_parent(root).glob(f"*+{selection.label}"):
        config_path = run_dir / "config.json"
        log_path = run_dir / "log_info.txt"
        if not config_path.is_file() or not log_path.is_file():
            continue
        metrics = parse_metrics(log_path)
        if metrics["tokens_per_sec"] is not None and metrics["TTFT(s)"] is not None:
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No complete run found for {selection.model}/{selection.dataset}/"
            f"{selection.objective}/{selection.policy}"
        )
    return max(candidates, key=lambda path: (path / "log_info.txt").stat().st_mtime)


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def collect_results(
    selections: list[Selection],
    run_root: Path,
    db_scales: dict[tuple[str, str, str, str], tuple[float, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for selection in selections:
        run_dir = find_run(selection, run_root)
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        validate_config(selection, config, run_dir)
        metrics = parse_metrics(run_dir / "log_info.txt")
        raw_tp = float(metrics["tokens_per_sec"])
        raw_ttft = float(metrics["TTFT(s)"])
        throughput_scale = 1.0
        ttft_scale = 1.0
        if selection.policy == "cp_et_db" and selection.request_scheduler == "vllm":
            key = (
                selection.model,
                selection.dataset,
                selection.objective,
                selection.policy,
            )
            throughput_scale, ttft_scale = db_scales[key]
        tp = raw_tp * throughput_scale
        ttft = raw_ttft * ttft_scale
        rows.append(
            {
                "model": selection.model,
                "dataset": selection.dataset,
                "objective": selection.objective,
                "policy": selection.policy,
                "tokens_per_sec": tp,
                "TTFT(s)": ttft,
                "raw_tokens_per_sec": raw_tp,
                "raw_TTFT(s)": raw_ttft,
                "throughput_scale": throughput_scale,
                "TTFT_scale": ttft_scale,
                "completed_requests": metrics["completed_requests"],
                "reference_tokens_per_sec": selection.reference_tp,
                "reference_TTFT(s)": selection.reference_ttft,
                "reference_throughput_norm_vs_rr": (
                    selection.reference_tp / selection.reference_baseline_tp
                ),
                "reference_TTFT_norm_vs_rr": (
                    selection.reference_ttft / selection.reference_baseline_ttft
                ),
                "throughput_norm_vs_rr": tp / selection.reference_baseline_tp,
                "TTFT_norm_vs_rr": ttft / selection.reference_baseline_ttft,
                "normalization_baseline": "prepared_rr_static",
                "baseline_tokens_per_sec": selection.reference_baseline_tp,
                "baseline_TTFT(s)": selection.reference_baseline_ttft,
                "throughput_delta_percent": 100.0
                * (tp / selection.reference_tp - 1.0),
                "TTFT_delta_percent": 100.0 * (ttft / selection.reference_ttft - 1.0),
                "placer": selection.placer,
                "request_scheduler": selection.request_scheduler,
                "task_scheduler": selection.task_scheduler,
                "batchsize": selection.batchsize,
                "Num M": selection.num_m,
                "Num Mp": selection.num_mp,
                "Num Md": selection.num_md,
                "Num Ap": selection.num_ap,
                "Num Ad": selection.num_ad,
                "NoI_bw(GBps)": selection.bandwidth,
                "run_dir": portable_path(run_dir),
            }
        )

    return rows


def write_results(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    models = parse_list(args.models.upper(), MODEL_ORDER, "models")
    datasets = parse_list(args.datasets.upper(), DATASET_ORDER, "datasets")
    objectives = parse_list(args.objectives, OBJECTIVE_ORDER, "objectives")
    policies = parse_list(args.policies.lower(), POLICY_ORDER, "policies")

    selections = [
        item
        for item in load_selections(args.manifest)
        if item.model in models
        and item.dataset in datasets
        and item.objective in objectives
        and item.policy in policies
    ]
    db_scales = load_db_scales(args.measurements)
    rows = collect_results(
        selections,
        args.run_root.resolve(),
        db_scales,
    )
    write_results(rows, args.output_csv)
    print(f"Summarized {len(rows)} simulations into {args.output_csv}")


if __name__ == "__main__":
    main()
