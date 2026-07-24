import multiprocessing
import subprocess
from typing import Tuple
import os

def run_command(command: str) -> Tuple[str, bool]:
    """Run a bash command in a subprocess and return (command, success)."""
    try:
        print(f"[RUNNING] {command}")
        subprocess.run(command, shell=True, executable="/bin/bash", check=True)
        return (command, True)
    except subprocess.CalledProcessError:
        return (command, False)

def main():
    # Constants
    SCENES = "LLAMA3-ARXIV"
    model = "llama3-8b"
    dataset = "arxiv"

    sim_length = 100  # in seconds
    Num_nodes = 24
    width = 6
    height = 4
    fixed_chiplet_area = True
    chiplet_area = 144
    M = 8
    P = 8
    D = 8
    bw = 384
    dataset_path = "dataset/arxiv/arxiv_summarization_stats_llama3.csv"
    batch_sizes = [1, 2, 3, 4, 6, 8, 16]
    commands = []
    for bs in batch_sizes:
        command = (
            f"python ./main.py "
            f"--workload-config.model {model} "
            f"--workload-config.dataset {dataset} "
            f"--placmt-config.chiplet-alloc.tscs_p {P} "
            f"--placmt-config.chiplet-alloc.tscs_d {D} "
            f"--placmt-config.chiplet-alloc.HBM3 {M} "
            f"--arch-config.num_nodes {Num_nodes} "
            f"--arch-config.intp_width {width} "
            f"--arch-config.intp_height {height} "
            f"--metrics-config.label_name '{SCENES}_BMPD{bw}_{M}_{P}_{D}' "
            f"--chips-config.D2D_NoI_bw {bw} "
            f"--chips-config.area_limit {chiplet_area} "
            f"--chips-config.Fixed_chiplet_area "
            f"--workload-config.request-generator-config.trace-length-generator-config.trace_file {dataset_path} "
            f"--time-limit {sim_length} "
            f"--cluster_config.batch_size {bs} "
        )
        commands.append(command)
            
    # Run commands in parallel
    log_path = f"./scripts/run_{SCENES}_batch_sizes.log"
    num_workers = min(len(commands), 40)
    with multiprocessing.Pool(processes=num_workers) as pool, open(log_path, "w") as f:
        for cmd, ok in pool.imap_unordered(run_command, commands):
            if ok:
                msg = f"[OK] {cmd}\n"
            else:
                msg = f"[ERROR] {cmd}\n"
            print(msg.strip())
            f.write(msg)
            f.flush()


if __name__ == "__main__":
    main()