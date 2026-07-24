import multiprocessing
import subprocess
from typing import Tuple
import os
import time

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
    SCENES_list = [
        ["MAMBA2-ARXIV","arxiv","dataset/arxiv/arxiv_summarization_stats_mamba2.csv"],
        ["MAMBA2-CHAT","chat","dataset/chat/chat1m_stats_mamba2.csv"],
        ["MAMBA2-BWB","bwb","dataset/bwb/bwb_translation_stats_mamba2.csv"],
        ["MAMBA2-LW","longwriter","dataset/longwriter/long_writer_mamba_6k.csv"]
    ]

    model = "mamba2-3b"
    placer = "bw"
    batchsizes = [2,4,8,16,32,64]
    req_scheduler = "vllm"
    task_scheduler = "elastic"
    
    sim_length = 100  # in seconds
    wall_time_limit = 900  # in seconds (15 minutes)
    Num_nodes = 24
    width = 6
    height = 4
    fixed_chiplet_area = True
    chiplet_area = 144
    Ms = range(2, 17, 1)  # Number of chiplets outring constraint
    BWs = [256, 384, 512, 640]
    #BWs = [256]
    dataset_path = "dataset/arxiv/arxiv_summarization_stats_llama3.csv"
    
    for scene in SCENES_list:
        SCENES = scene[0]
        dataset = scene[1]
        dataset_path = scene[2]
        for batchsize in batchsizes:
            for bw in BWs:
                commands = []
                for M in Ms:
                    remain = Num_nodes - M
                    Ps = range(1, remain)
                    for P in Ps:
                        D = remain - P
                        if D < 0:
                            raise ValueError("D cannot be negative")

                        command = (
                            f"python ./main.py "
                            f"--workload-config.model {model} "
                            f"--workload-config.dataset {dataset} "
                            f"--placmt-config.chiplet-alloc.marca_p {P} "
                            f"--placmt-config.chiplet-alloc.marca_d {D} "
                            f"--placmt-config.chiplet-alloc.tscs_p {0} "
                            f"--placmt-config.chiplet-alloc.tscs_d {0} "
                            f"--placmt-config.chiplet-alloc.HBM3 {M} "
                            f"--arch-config.num_nodes {Num_nodes} "
                            f"--arch-config.intp_width {width} "
                            f"--arch-config.intp_height {height} "
                            f"--metrics-config.label_name '{SCENES}_BMPD{bw}_{M}_{P}_{D}' "
                            f"--metrics-config.output_dir 'simulator_output_{SCENES}_{placer}placement_bs{batchsize}_{req_scheduler}req_{task_scheduler}task' "  # output all metrics
                            f"--chips-config.D2D_NoI_bw {bw} "
                            f"--chips-config.area_limit {chiplet_area} "
                            f"--chips-config.Fixed_chiplet_area "
                            f"--workload-config.request-generator-config.trace-length-generator-config.trace_file {dataset_path} "
                            f"--cluster-config.batch-size {batchsize} "
                            f"--cluster-config.local_scheduler {req_scheduler} " # run-time config
                            f"--placmt-config.placer-label {placer} "
                            f"--mapping-config.mapping_strategy {task_scheduler} "
                            f"--time-limit {sim_length} "
                            f"--wall-time-limit {wall_time_limit} "
                        )
                        commands.append(command)
                
                # Run commands in parallel
                current_time = int(time.time())
                log_path = f"./scripts/run_{SCENES}_MPD_bw{bw}_{current_time}.log"
                num_workers = min(len(commands), 20)
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
