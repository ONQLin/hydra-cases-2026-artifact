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
        ["JAMBA-TINY-CHAT", "chat", "dataset/chat/chat1m_stats_nemo.csv"],
    ]

    model = "jamba-tiny"
    model_tag = "jamba"
    placer = "rr"
    batchsizes = [2, 4, 8, 16, 32, 64]
    req_scheduler = "static"
    task_scheduler = "static"

    sim_length = 100  # in seconds
    Num_nodes = 24
    width = 6
    height = 4
    chiplet_area = 144
    Ms = range(2, 17, 2)
    BWs = [256, 384, 512, 640]
    output_root = "revisions_simulator_output/comment1_1"

    os.makedirs(f"{output_root}/logs", exist_ok=True)

    for scene in SCENES_list:
        SCENES = scene[0]
        dataset = scene[1]
        dataset_path = scene[2]
        for batchsize in batchsizes:
            for bw in BWs:
                commands = []
                for M in Ms:
                    remain = Num_nodes - M
                    if remain < 4:
                        continue
                    for Ap in range(1, remain - 2, 2):
                        for Ad in range(1, remain - Ap - 1, 2):
                            for Mp in range(1, remain - Ap - Ad, 2):
                                Md = remain - Ap - Ad - Mp
                                command = (
                                    f"python ./main.py "
                                    f"--workload-config.model {model} "
                                    f"--workload-config.dataset {dataset} "
                                    f"--placmt-config.chiplet-alloc.marca_p {Mp} "
                                    f"--placmt-config.chiplet-alloc.marca_d {Md} "
                                    f"--placmt-config.chiplet-alloc.tscs_p {Ap} "
                                    f"--placmt-config.chiplet-alloc.tscs_d {Ad} "
                                    f"--placmt-config.chiplet-alloc.HBM3 {M} "
                                    f"--arch-config.num_nodes {Num_nodes} "
                                    f"--arch-config.intp_width {width} "
                                    f"--arch-config.intp_height {height} "
                                    f"--metrics-config.label_name '{SCENES}_BMPD{bw}_{M}_{Ap}{Mp}{Ad}{Md}' "
                                    f"--metrics-config.output_dir '{output_root}/single_model_dse/{model_tag}' "
                                    f"--chips-config.D2D_NoI_bw {bw} "
                                    f"--chips-config.area_limit {chiplet_area} "
                                    f"--chips-config.Fixed_chiplet_area "
                                    f"--workload-config.request-generator-config.trace-length-generator-config.trace_file {dataset_path} "
                                    f"--cluster-config.batch-size {batchsize} "
                                    f"--cluster-config.local_scheduler {req_scheduler} "
                                    f"--placmt-config.placer-label {placer} "
                                    f"--mapping-config.mapping_strategy {task_scheduler} "
                                    f"--time-limit {sim_length} "
                                )
                                commands.append(command)

                current_time = int(time.time())
                log_path = f"./{output_root}/logs/run_{SCENES}_MPD_bw{bw}_{current_time}.log"
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
