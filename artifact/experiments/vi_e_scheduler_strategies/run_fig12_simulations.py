#!/usr/bin/env python3
"""Run the selected Nemotron-H task-scheduler experiments used by Fig. 12."""

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
DEFAULT_MANIFEST = EXPERIMENT_DIR / "fig12_configurations.csv"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "artifact" / "run_outputs" / "vi_e_scheduler_strategies" / "fig12" / "raw_simulations"
)
MODEL_NAME = "nemotronh-4b"
DATASET_ORDER = ("ARXIV", "BWB", "CHAT", "LW")
OBJECTIVE_ORDER = ("B_star", "M_T")
SCHEDULER_ORDER = ("static", "fcfs", "worksteal", "elastic")
DATASET_ARGUMENTS = {
    "ARXIV": "arxiv",
    "BWB": "bwb",
    "CHAT": "chat",
    "LW": "longwriter",
}
TRACE_FILES = {
    "ARXIV": "dataset/arxiv/arxiv_summarization_stats_nemo.csv",
    "BWB": "dataset/bwb/bwb_translation_stats_nemo.csv",
    "CHAT": "dataset/chat/chat1m_stats_nemo.csv",
    "LW": "dataset/longwriter/long_writer_nemo_6k.csv",
}
METRIC_PATTERNS = (
    re.compile(r"output tokens per second:\s*[-+]?\d*\.?\d+"),
    re.compile(r"Average time to first token:\s*[-+]?\d*\.?\d+"),
)


@dataclass(frozen=True)
class Experiment:
    dataset: str
    objective: str
    scheduler: str
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
            f"FIG12_{self.dataset}_{self.objective}_{self.scheduler}_"
            f"BW{self.bandwidth}_BS{self.batchsize}_M{self.num_m}_Ap{self.num_ap}_"
            f"Ad{self.num_ad}_Mp{self.num_mp}_Md{self.num_md}"
        )

    def output_dir(self, root: Path) -> Path:
        objective = "bstar" if self.objective == "B_star" else "mt"
        return root / self.dataset.lower() / objective / self.scheduler


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
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER))
    parser.add_argument("--objectives", default=",".join(OBJECTIVE_ORDER))
    parser.add_argument("--schedulers", default=",".join(SCHEDULER_ORDER))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--time-limit", type=int, default=100)
    parser.add_argument("--command-timeout", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def load_experiments(path: Path) -> list[Experiment]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    experiments = [
        Experiment(
            dataset=row["dataset"],
            objective=row["objective"],
            scheduler=row["scheduler"],
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
    keys = [(item.dataset, item.objective, item.scheduler) for item in experiments]
    if len(keys) != len(set(keys)):
        raise ValueError("The Fig. 12 manifest contains duplicate dataset/objective/scheduler rows.")
    return experiments


def command_for(exp: Experiment, args: argparse.Namespace) -> list[str]:
    output_dir = exp.output_dir(args.output_root)
    try:
        output_dir_arg = str(output_dir.resolve().relative_to(REPO_ROOT))
    except ValueError:
        output_dir_arg = str(output_dir)
    return [
        args.python_bin,
        str(REPO_ROOT / "main.py"),
        "--workload-config.model",
        MODEL_NAME,
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
        output_dir_arg,
        "--chips-config.D2D_NoI_bw",
        str(exp.bandwidth),
        "--chips-config.area_limit",
        "144",
        "--chips-config.Fixed_chiplet_area",
        "--workload-config.request-generator-config.trace-length-generator-config.trace_file",
        TRACE_FILES[exp.dataset],
        "--cluster-config.batch-size",
        str(exp.batchsize),
        "--cluster-config.local_scheduler",
        "static",
        "--placmt-config.placer-label",
        "bw",
        "--mapping-config.mapping_strategy",
        exp.scheduler,
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


def write_run_inputs(experiments: list[Experiment], args: argparse.Namespace) -> None:
    run_root = args.output_root.parent
    run_root.mkdir(parents=True, exist_ok=True)
    command_path = run_root / "run_commands.sh"
    with command_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for exp in experiments:
            command = command_for(exp, args)
            try:
                command[0] = "./" + str(Path(command[0]).absolute().relative_to(REPO_ROOT))
            except ValueError:
                pass
            command[1] = "./main.py"
            handle.write(shlex.join(command) + "\n")
    command_path.chmod(0o755)


def run_one(exp: Experiment, args: argparse.Namespace, env: dict[str, str]) -> dict[str, object]:
    exp.output_dir(args.output_root).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command_for(exp, args),
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
        "dataset": exp.dataset,
        "objective": exp.objective,
        "scheduler": exp.scheduler,
        "label": exp.label,
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_tail": "\n".join(output.splitlines()[-20:]) if status != "ok" else "",
    }


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    datasets = parse_list(args.datasets.upper(), DATASET_ORDER, "datasets")
    objectives = parse_list(args.objectives, OBJECTIVE_ORDER, "objectives")
    schedulers = parse_list(args.schedulers.lower(), SCHEDULER_ORDER, "schedulers")
    experiments = [
        exp
        for exp in load_experiments(args.manifest)
        if exp.dataset in datasets and exp.objective in objectives and exp.scheduler in schedulers
    ]
    if not experiments:
        raise RuntimeError("No Fig. 12 experiments match the requested filters.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    make_configs_portable(experiments, args.output_root)
    write_run_inputs(experiments, args)
    pending = experiments if args.rerun_completed else [exp for exp in experiments if not is_complete(exp, args.output_root)]
    print(f"Selected {len(experiments)} Fig. 12 simulations; {len(pending)} require execution.")
    if args.dry_run or not pending:
        return 0

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplcache"))
    status_path = args.output_root.parent / "run_status.jsonl"
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
        futures = {pool.submit(run_one, exp, args, env): exp for exp in pending}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            failures += result["status"] != "ok"
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                f"[{index}/{len(pending)}] {result['dataset']} {result['objective']} "
                f"{result['scheduler']}: {result['status']} ({result['elapsed_seconds']} s)",
                flush=True,
            )
    make_configs_portable(experiments, args.output_root)
    print(f"Finished with {failures} unsuccessful simulations.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
