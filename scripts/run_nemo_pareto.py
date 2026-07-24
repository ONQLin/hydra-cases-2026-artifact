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
    req_scheduler = "vllm"
    task_scheduler = "elastic"
    model = "nemotronh-4b"
    model_name = "NEMO"
    estimate = False
    specify=False
    chiplet_area = 144
    configurations = {}
    for scene in SCENES_list:
        SCENES = scene[0]
        dataset_path = scene[2]
        dataset = scene[1]
        configurations_file = f"pareto_frontier_results_nemo{dataset.upper()}_BW_static_static.csv"
        configurations[SCENES] = load_configurations(configurations_file)
    
    if estimate:
        best_score = 0
        best_config = None
        for config in configurations:
            score = config['tokens_per_sec']/config['TTFT(s)']
            if score > best_score:
                best_score = score
                best_config = config
        print(f"Estimated best configuration based on pareto frontier: {best_config} with score {best_score}")
        exit(0)
   
    
    if specify:
        specified_configs = []
        for scene in SCENES_list:
            dataset = scene[1]
            scene_name = scene[0]
            file_name = f"results_summary_{model_name}-{dataset.upper()}_{base_placer}_{task_scheduler}_{req_scheduler}.csv"
            df = pd.read_csv(file_name)
            mean_ttft = df['TTFT(s)'].mean()
            mean_tp = df['tokens_per_sec'].mean()

            max_tp_idx = df['tokens_per_sec'].idxmax()
            max_tp_row = df.loc[max_tp_idx]
            ratio = df['tokens_per_sec'] / df['TTFT(s)'].replace(0, np.nan)
            max_ratio_idx = ratio[(df['tokens_per_sec'] > mean_tp) & (df['TTFT(s)'] < mean_ttft)].idxmax()
            max_ratio_row = df.loc[max_ratio_idx]
            special_points = [
                # ('Mean', mean_ttft, mean_tp, 'C2', 's'),            # green square
                # ('Min TTFT', min_ttft_row['TTFT(s)'], min_ttft_row['tokens_per_sec'], 'red', 'o', 400),  # red circle
                ('Max Tp', max_tp_row['TTFT(s)'], max_tp_row['tokens_per_sec'], 'red', '^', 400), # red triangle
                ('Max TP/TTFT', max_ratio_row['TTFT(s)'], max_ratio_row['tokens_per_sec'], 'red', '*', 600)# red star
            ]
            specified_configs = []
            MT_flag, Bstar_flag = False, False
            for config in configurations[scene_name]:
                if config['tokens_per_sec'] == max_tp_row['tokens_per_sec'] and config['TTFT(s)'] == max_tp_row['TTFT(s)'] and not MT_flag:
                    print(f"Found Max Tp configuration: {config}")   
                    MT_flag = True
                    specified_configs.append(config)
                if config['tokens_per_sec'] == max_ratio_row['tokens_per_sec'] and config['TTFT(s)'] == max_ratio_row['TTFT(s)'] and not Bstar_flag:
                    print(f"Found Max TP/TTFT configuration: {config}")   
                    Bstar_flag = True
                    specified_configs.append(config)
            configurations[scene_name] = specified_configs
        exit(0)
    
    commands = []
    for scene_idx, scene_name in enumerate(configurations.keys()):
        for config in configurations[scene_name]:
            batchsize = config['batchsize']
            num_m = config['Num M']
            num_ap = config['Num Ap']
            num_ad = config['Num Ad']
            num_mp = config['Num Mp']
            num_md = config['Num Md']
            noi_bw = config['NoI_bw(GBps)']
            dataset = scene_name.split("-")[1].lower()
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
                f"--metrics-config.output_dir 'simulator_output_{scene_name}_{placer}placement_bs{batchsize}_{req_scheduler}req_{task_scheduler}task_pareto' "  # output all metrics
                f"--chips-config.D2D_NoI_bw {int(noi_bw)} "
                f"--chips-config.area_limit {chiplet_area} "
                f"--chips-config.Fixed_chiplet_area "
                f"--workload-config.request-generator-config.trace-length-generator-config.trace_file {dataset_path} "
                f"--cluster-config.batch-size {batchsize} "
                f"--cluster-config.local_scheduler {req_scheduler} " # run-time config
                f"--placmt-config.placer-label {placer} "
                f"--mapping-config.mapping_strategy {task_scheduler} "
                f"--time-limit {sim_length} "
            )
            commands.append(command)
    # Run commands in parallel
    current_time = int(time.time())
    log_path = f"./scripts/run_{SCENES}_MPD_bw_Pareto_{current_time}.log"
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