import multiprocessing
import subprocess
import pandas as pd
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


def load_configurations(csv_path: str):
    """
    Read the pareto CSV and return a list of configuration dicts with keys:
    'batchsize', 'Num M', 'Num Ap', 'Num Ad', 'NoI_bw(GBps)'
    """
    df = pd.read_csv(csv_path)
    required = ['batchsize', 'Num M', 'Num Ap', 'Num Ad', 'Num Mp', 'Num Md', 'NoI_bw(GBps)']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    configs = []
    for _, r in df.iterrows():
        configs.append({
            'batchsize': int(r['batchsize']),
            'Num M': int(r['Num M']),
            'Num Ap': int(r['Num Ap']),
            'Num Ad': int(r['Num Ad']),
            'Num Mp': int(r['Num Mp']),
            'Num Md': int(r['Num Md']),
            'NoI_bw(GBps)': float(r['NoI_bw(GBps)']),
            'tokens_per_sec': float(r['tokens_per_sec']),
            'TTFT(s)': float(r['TTFT(s)'])
        })
    return configs

def main():
    # Constants
    SCENES_list = [
        ["NEMO-ARXIV","arxiv","dataset/arxiv/arxiv_summarization_stats_nemo.csv"],
        ["NEMO-CHAT","chat","dataset/chat/chat1m_stats_nemo.csv"],
        ["NEMO-BWB","bwb","dataset/bwb/bwb_translation_stats_nemo.csv"],
        ["NEMO-LW","lw","dataset/longwriter/long_writer_nemo_6k.csv"]
    ]
    
    placer = "bw"
    base_placer = "bw"
    sim_length = 100  # in seconds
    Num_nodes = 24
    width = 6
    height = 4
    fixed_chiplet_area = True
    req_scheduler = ["static","vllm"]
    task_scheduler = ["static", "elastic"]
    model = "nemotronh-4b"
    model_name = "NEMO"
    estimate = False
    specify=False
    chiplet_area = 144
    configurations = {}
    # batchsize,Num M,Num Mp,Num Md,Num Ap,Num Ad,NoI_bw(GBps),tokens_per_sec,TTFT(s),is_pareto,Ap_Ad_ratio
    # 2,6,7,7,3,1,384,353.27,7.050611011235955,True,3.0
    # 16,4,9,7,3,1,256,569.14,66.5066976,True,3.0
    # 2,6,7,7,3,1,384,618.58,12.663907111111111
    # 4,6,7,7,3,1,512,1069.37,41.27593066666667
    # 8,6,7,9,1,1,384,1097.47,1.8695551232876713
    # 32,4,5,13,1,1,384,2821.44,12.258554285714286
    # 4,10,3,9,1,1,512,1511.54,3.082964
    # 16,6,5,9,1,3,512,5324.29,14.328573111111112
    selected_configs = {
        "NEMO-ARXIV_B": {
            "batchsize": 2,
            "Num M": 6,
            "Num Mp": 7,
            "Num Md": 7,
            "Num Ap": 3,
            "Num Ad": 1,
            "NoI_bw(GBps)": 384.0
        },
        "NEMO-ARXIV_M": {
            "batchsize": 16,
            "Num M": 4,
            "Num Mp": 9,
            "Num Md": 7,
            "Num Ap": 3,
            "Num Ad": 1,
            "NoI_bw(GBps)": 256.0
        },
        "NEMO-BWB_B": {
            "batchsize": 2,
            "Num M": 6,
            "Num Mp": 7,
            "Num Md": 7,
            "Num Ap": 3,
            "Num Ad": 1,
            "NoI_bw(GBps)": 384.0
        },
        "NEMO-BWB_M": {
            "batchsize": 4,
            "Num M": 6,
            "Num Mp": 7,
            "Num Md": 7,
            "Num Ap": 3,
            "Num Ad": 1,
            "NoI_bw(GBps)": 512.0
        },
        "NEMO-CHAT_B": {
            "batchsize": 8,
            "Num M": 6,
            "Num Mp": 7,
            "Num Md": 9,
            "Num Ap": 1,
            "Num Ad": 1,
            "NoI_bw(GBps)": 384.0
        },
        "NEMO-CHAT_M": {
            "batchsize": 32,
            "Num M": 4,
            "Num Mp": 5,
            "Num Md": 13,
            "Num Ap": 1,
            "Num Ad": 1,
            "NoI_bw(GBps)": 384.0
        },
        "NEMO-LW_B": {
            "batchsize": 4,
            "Num M": 10,
            "Num Mp": 3,
            "Num Md": 9,
            "Num Ap": 1,
            "Num Ad": 1,
            "NoI_bw(GBps)": 512.0
        },
        "NEMO-LW_M": {
            "batchsize": 16,
            "Num M": 6,
            "Num Mp": 5,
            "Num Md": 9,
            "Num Ap": 1,
            "Num Ad": 3,
            "NoI_bw(GBps)": 512.0
        }
    }
    
    commands = []
    for scene_idx, scene_configs in enumerate(SCENES_list):
        for sched_idx in range(len(req_scheduler)):
            for label in ["B", "M"]:
                scene_name = f"{scene_configs[0]}_{label}"
                config = selected_configs[scene_name]
                batchsize = config['batchsize']
                num_m = config['Num M']
                num_ap = config['Num Ap']
                num_ad = config['Num Ad']
                num_mp = config['Num Mp']
                num_md = config['Num Md']
                noi_bw = config['NoI_bw(GBps)']
                dataset = scene_configs[1]
                dataset_path = SCENES_list[scene_idx][2]
                command = (
                    f"python ./main.py "
                    f"--workload-config.model {model} "
                    f"--workload-config.dataset {dataset} "
                    f"--placmt-config.chiplet-alloc.marca_p {num_mp} "
                    f"--placmt-config.chiplet-alloc.marca_d {num_md} "
                    f"--placmt-config.chiplet-alloc.tscs_p {num_ap} "
                    f"--placmt-config.chiplet-alloc.tscs_d {num_ad} "
                    f"--placmt-config.chiplet-alloc.HBM3 {num_m} "
                    f"--arch-config.num_nodes {Num_nodes} "
                    f"--arch-config.intp_width {width} "
                    f"--arch-config.intp_height {height} "
                    f"--metrics-config.label_name '{scene_name}_BMPD{noi_bw}_{num_m}_{num_ap}_{num_ad}' "
                    f"--metrics-config.output_dir 'simulator_output_{scene_name}_{placer}placement_bs{batchsize}_{req_scheduler[sched_idx]}req_{task_scheduler[sched_idx]}task_util' "  # output all metrics
                    f"--chips-config.D2D_NoI_bw {int(noi_bw)} "
                    f"--chips-config.area_limit {chiplet_area} "
                    f"--chips-config.Fixed_chiplet_area "
                    f"--workload-config.request-generator-config.trace-length-generator-config.trace_file {dataset_path} "
                    f"--cluster-config.batch-size {batchsize} "
                    f"--cluster-config.local_scheduler {req_scheduler[sched_idx]} " # run-time config
                    f"--placmt-config.placer-label {placer} "
                    f"--mapping-config.mapping_strategy {task_scheduler[sched_idx]} "
                    f"--time-limit {sim_length} "
                    f"--metrics-config.verbose "
                )
                commands.append(command)
    # Run commands in parallel
    current_time = int(time.time())
    log_path = f"./scripts/run_Util_test_MPD_bw_Pareto_{current_time}.log"
    num_workers = min(len(commands), 32)
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