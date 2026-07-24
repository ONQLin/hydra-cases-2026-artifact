#!/usr/bin/env python3
"""Run the selected CP, CP+ET, and CP+ET+DB experiments used by Fig. 9."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
NUM_IN_PARALLEL = 32
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifact"
    / "run_outputs"
    / "vi_d_e_policy_ablation"
    / "raw_data"
    / "fig9_configurations.csv"
)
SIMULATOR_WRAPPER = EXPERIMENT_DIR / "run_seeded_simulator.py"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "output_temp_runs" / "vi_d_e_policy_ablation" / "simulator_runs"
)
MODEL_ORDER = ("LLAMA3", "MAMBA2", "NEMO")
DATASET_ORDER = ("ARXIV", "LW", "CHAT", "BWB")
OBJECTIVE_ORDER = ("B_star", "M_T")
POLICY_ORDER = ("rr_baseline", "cp", "cp_et", "cp_et_db")
DEFAULT_POLICY_ORDER = ("cp", "cp_et", "cp_et_db")
MODEL_ARGUMENTS = {
    "LLAMA3": "llama3-8b",
    "MAMBA2": "mamba2-3b",
    "NEMO": "nemotronh-4b",
}
DATASET_ARGUMENTS = {
    "ARXIV": "arxiv",
    "BWB": "bwb",
    "CHAT": "chat",
    "LW": "longwriter",
}
TRACE_FILES = {
    ("LLAMA3", "ARXIV"): "dataset/arxiv/arxiv_summarization_stats_llama3.csv",
    ("LLAMA3", "BWB"): "dataset/bwb/bwb_translation_stats_llama3.csv",
    ("LLAMA3", "CHAT"): "dataset/chat/chat1m_stats_llama3.csv",
    ("LLAMA3", "LW"): "dataset/longwriter/long_writer_transformer_1.3b_6k.csv",
    ("MAMBA2", "ARXIV"): "dataset/arxiv/arxiv_summarization_stats_mamba2.csv",
    ("MAMBA2", "BWB"): "dataset/bwb/bwb_translation_stats_mamba2.csv",
    ("MAMBA2", "CHAT"): "dataset/chat/chat1m_stats_mamba2.csv",
    ("MAMBA2", "LW"): "dataset/longwriter/long_writer_mamba_6k.csv",
    ("NEMO", "ARXIV"): "dataset/arxiv/arxiv_summarization_stats_nemo.csv",
    ("NEMO", "BWB"): "dataset/bwb/bwb_translation_stats_nemo.csv",
    ("NEMO", "CHAT"): "dataset/chat/chat1m_stats_nemo.csv",
    ("NEMO", "LW"): "dataset/longwriter/long_writer_nemo_6k.csv",
}
METRIC_PATTERNS = (
    re.compile(r"output tokens per second:\s*[-+]?\d*\.?\d+"),
    re.compile(r"Average time to first token:\s*[-+]?\d*\.?\d+"),
)


@dataclass(frozen=True)
class Experiment:
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

    @property
    def label(self) -> str:
        return (
            f"FIG9_{self.model}_{self.dataset}_{self.objective}_{self.policy}_"
            f"BW{self.bandwidth}_BS{self.batchsize}_M{self.num_m}_Ap{self.num_ap}_"
            f"Ad{self.num_ad}_Mp{self.num_mp}_Md{self.num_md}"
        )

    def output_dir(self, root: Path) -> Path:
        objective = "bstar" if self.objective == "B_star" else "mt"
        return root / self.model.lower() / self.dataset.lower() / objective / self.policy


def parse_list(raw: str, valid: tuple[str, ...], name: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(values) - set(valid))
    if invalid:
        raise ValueError(f"Invalid {name}: {', '.join(invalid)}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER))
    parser.add_argument("--objectives", default=",".join(OBJECTIVE_ORDER))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICY_ORDER))
    parser.add_argument("--workers", type=int, default=NUM_IN_PARALLEL)
    parser.add_argument("--time-limit", type=int, default=100)
    parser.add_argument("--command-timeout", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--simulator-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--placement-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def load_experiments(path: Path) -> list[Experiment]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    experiments = [
        Experiment(
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
        )
        for row in rows
    ]
    keys = [(item.model, item.dataset, item.objective, item.policy) for item in experiments]
    if len(keys) != len(set(keys)):
        raise ValueError("The Fig. 9 manifest contains duplicate model/dataset/objective/policy rows.")
    expected = {
        (model, dataset, objective, policy)
        for model in MODEL_ORDER
        for dataset in DATASET_ORDER
        for objective in OBJECTIVE_ORDER
        for policy in POLICY_ORDER
    }
    if set(keys) != expected:
        raise ValueError("The Fig. 9 manifest must contain all 96 fixed configurations.")
    return experiments


def prepare_simulator_root(args: argparse.Namespace) -> Path:
    simulator_root = args.simulator_root.resolve()
    if not (simulator_root / "main.py").is_file():
        raise FileNotFoundError(f"No simulator entry point under {simulator_root}")
    return simulator_root


def command_for(exp: Experiment, args: argparse.Namespace, simulator_root: Path) -> list[str]:
    output_dir = exp.output_dir(args.output_root)
    return [
        args.python_bin,
        str(SIMULATOR_WRAPPER),
        "--simulator-root",
        str(simulator_root),
        "--placement-seed",
        str(args.placement_seed),
        "--",
        "--workload-config.model",
        MODEL_ARGUMENTS[exp.model],
        "--workload-config.dataset",
        DATASET_ARGUMENTS[exp.dataset],
        "--placmt-config.chiplet-alloc.marca_p",
        str(exp.num_mp),
        "--placmt-config.chiplet-alloc.marca_d",
        str(exp.num_md),
        "--placmt-config.chiplet-alloc.tscs_p",
        str(exp.num_ap),
        "--placmt-config.chiplet-alloc.tscs_d",
        str(exp.num_ad),
        "--placmt-config.chiplet-alloc.HBM3",
        str(exp.num_m),
        "--arch-config.num_nodes",
        "24",
        "--arch-config.intp_width",
        "6",
        "--arch-config.intp_height",
        "4",
        "--metrics-config.label_name",
        exp.label,
        "--metrics-config.output_dir",
        str(output_dir.resolve()),
        "--chips-config.D2D_NoI_bw",
        str(exp.bandwidth),
        "--chips-config.area_limit",
        "144",
        "--chips-config.Fixed_chiplet_area",
        "--workload-config.request-generator-config.trace-length-generator-config.trace_file",
        TRACE_FILES[(exp.model, exp.dataset)],
        "--cluster-config.batch-size",
        str(exp.batchsize),
        "--cluster-config.local_scheduler",
        exp.request_scheduler,
        "--placmt-config.placer-label",
        exp.placer,
        "--mapping-config.mapping_strategy",
        exp.task_scheduler,
        "--time-limit",
        str(args.time_limit),
    ]


def log_has_metrics(path: Path) -> bool:
    if not path.is_file():
        return False
    contents = path.read_text(encoding="utf-8", errors="replace")
    return all(pattern.search(contents) for pattern in METRIC_PATTERNS)


def is_complete(exp: Experiment, output_root: Path) -> bool:
    output_dir = exp.output_dir(output_root)
    return any(log_has_metrics(run_dir / "log_info.txt") for run_dir in output_dir.glob(f"*+{exp.label}"))


def make_configs_portable(experiments: list[Experiment], output_root: Path) -> None:
    for exp in experiments:
        for config_path in exp.output_dir(output_root).glob(f"*+{exp.label}/config.json"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            output_dir = Path(config.get("metrics_config", {}).get("output_dir", ""))
            if not output_dir.is_absolute():
                continue
            try:
                config["metrics_config"]["output_dir"] = str(output_dir.relative_to(REPO_ROOT))
            except ValueError:
                continue
            config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")


def write_run_inputs(
    experiments: list[Experiment], args: argparse.Namespace, simulator_root: Path
) -> None:
    run_root = args.output_root.parent
    run_root.mkdir(parents=True, exist_ok=True)
    command_path = run_root / "run_commands.sh"
    with command_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for exp in experiments:
            command = command_for(exp, args, simulator_root)
            for index, value in enumerate(command):
                path = Path(value)
                if not path.is_absolute():
                    continue
                try:
                    relative = path.relative_to(REPO_ROOT)
                    command[index] = "." if relative == Path(".") else "./" + str(relative)
                except ValueError:
                    continue
            handle.write(shlex.join(command) + "\n")
    command_path.chmod(0o755)


def run_one(
    exp: Experiment,
    args: argparse.Namespace,
    simulator_root: Path,
    env: dict[str, str],
) -> dict[str, object]:
    exp.output_dir(args.output_root).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command_for(exp, args, simulator_root),
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.command_timeout or None,
            check=False,
        )
        status = "ok" if process.returncode == 0 and is_complete(exp, args.output_root) else "failed"
        return_code = process.returncode
        output = process.stdout or ""
    except subprocess.TimeoutExpired as error:
        status = "timeout"
        return_code = 124
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
    return {
        "model": exp.model,
        "dataset": exp.dataset,
        "objective": exp.objective,
        "policy": exp.policy,
        "label": exp.label,
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_tail": "\n".join(output.splitlines()[-20:]) if status != "ok" else "",
    }


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    models = parse_list(args.models.upper(), MODEL_ORDER, "models")
    datasets = parse_list(args.datasets.upper(), DATASET_ORDER, "datasets")
    objectives = parse_list(args.objectives, OBJECTIVE_ORDER, "objectives")
    policies = parse_list(args.policies.lower(), POLICY_ORDER, "policies")
    experiments = [
        exp
        for exp in load_experiments(args.manifest)
        if exp.model in models
        and exp.dataset in datasets
        and exp.objective in objectives
        and exp.policy in policies
    ]
    if not experiments:
        raise RuntimeError("No Fig. 9 experiments match the requested filters.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    simulator_root = prepare_simulator_root(args)
    make_configs_portable(experiments, args.output_root)
    write_run_inputs(experiments, args, simulator_root)
    pending = experiments if args.rerun_completed else [exp for exp in experiments if not is_complete(exp, args.output_root)]
    print(f"Selected {len(experiments)} Fig. 9 simulations; {len(pending)} require execution.")
    if args.dry_run or not pending:
        return 0

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplcache"))
    status_path = args.output_root.parent / "run_status.jsonl"
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
        futures = {
            pool.submit(run_one, exp, args, simulator_root, env): exp
            for exp in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            failures += result["status"] != "ok"
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                f"[{index}/{len(pending)}] {result['model']} {result['dataset']} "
                f"{result['objective']} {result['policy']}: {result['status']} "
                f"({result['elapsed_seconds']} s)",
                flush=True,
            )
    make_configs_portable(experiments, args.output_root)
    print(f"Finished with {failures} unsuccessful simulations.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
