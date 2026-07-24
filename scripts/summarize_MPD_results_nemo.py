import os
import json
import re
import glob
import pandas as pd
import ast

def parse_log_info_tokens(log_path: str) -> float | None:
    """Extract tokens/sec from log_info.txt."""
    with open(log_path, "r") as f:
        for line in f:
            if "output tokens per second:" in line:
                # Grab the last number in the line
                match = re.search(r"([\d.]+)\s*$", line.strip())
                if match:
                    return float(match.group(1))
    return None

def parse_log_info_ttft(log_path: str) -> float | None:
    """Extract tokens/sec from log_info.txt."""
    with open(log_path, "r") as f:
        for line in f:
            if "Average time to first token:" in line:
                # Grab the last number in the line
                match = re.search(r"([\d.]+)\s*$", line.strip())
                if match:
                    return float(match.group(1))/1000000
    return None

def parse_config(config_path: str) -> dict:
    """Load config.json as dict."""
    with open(config_path, "r") as f:
        return json.load(f)

def main(label="PLCMT", index=3, placer="bw", req_scheduler="static", task_scheduler="elastic", dataset="ARXIV", model="NEMO"):
    # 1. Define the list of result directories to process
    # placer = "bw"
    # task_scheduler = "elastic" # static elastic
    # req_scheduler = "static"
    # dataset = "ARXIV"
    # model = "NEMO"
    # index = 3
    # label = "ELASTIC"
    
    # Parent folder containing all simulator_output_* subfolders
    parent_dir = f"Results_{model}{index}/{model}-{dataset}-{label}"
    
    # Automatically discover all simulator_output_* subfolders
    results_dirs = sorted(glob.glob(os.path.join(parent_dir, "simulator_output_*")))

    thre_dict = {
        "ARXIV": 200,
        "LW": 1000,
        "CHAT": 300,
        "BWB": 300
    }
    
    Exp_name = parent_dir.split("/")[1].split(f"-{label}")[0]
    all_rows = [] # Use a list to collect data from all directories

    highest_tp = 0
    score = 0
    best_config = None
    # 2. Iterate over each defined results directory
    for results_dir in results_dirs:
        # Extract the batch size (bs) number from the directory name
        # Use a regular expression to find 'bs' followed by one or more digits
        match = re.search(r'bs(\d+)', results_dir)
        batchsize = int(match.group(1)) if match else None
        
        if batchsize is None:
            print(f"Warning: Could not extract batch size from {results_dir}. Skipping.")
            continue
            
        print(f"Processing directory: {results_dir} with batchsize: {batchsize}")

        # 3. Existing logic to iterate through subfolders within the current results_dir
        for folder in glob.glob(os.path.join(results_dir, "*")):
            folder_name = os.path.basename(folder)
            
            # This check might be redundant if the parent folder names are specific enough,
            # but it's kept for internal filtering consistency.
            if Exp_name not in folder_name:
                continue
            
            log_path = os.path.join(folder, "log_info.txt")
            config_path = os.path.join(folder, "config.json")

            if not (os.path.isfile(log_path) and os.path.isfile(config_path)):
                continue

            # Parse data
            tokens_per_sec = parse_log_info_tokens(log_path)
            if tokens_per_sec is None:
                continue
            ttft = parse_log_info_ttft(log_path)
            config = parse_config(config_path)
            
            # Extract config details
            M = 0
            Ap = 0
            Ad = 0
            Mp = 0
            Md = 0
            BW_config = config["chips_config"]["D2D_NoI_bw"]
            NoI_bw = int(BW_config)
            
            alloc = config["placmt_config"]["chiplet_alloc"]
            chiplet_alloc = ast.literal_eval(alloc) 
            Ap = chiplet_alloc["tscs_p"]
            Ad = chiplet_alloc["tscs_d"]
            Mp = chiplet_alloc["marca_p"]
            Md = chiplet_alloc["marca_d"]
            M = chiplet_alloc["HBM3"]

            # Append the row, **including the batchsize**
            all_rows.append({
                "batchsize": batchsize, # <--- Added column
                "Num Ap": Ap,
                "Num Ad": Ad,
                "Num Mp": Mp,
                "Num Md": Md,
                "Num M": M,
                "NoI_bw(GBps)": NoI_bw,
                "tokens_per_sec": tokens_per_sec,
                "TTFT(s)": ttft,
            })
            try:
                temp_score = tokens_per_sec/ttft
            except:
                temp_score = 0
            if temp_score > score and tokens_per_sec > thre_dict[dataset.upper()]:
                score = temp_score
                best_config = {
                    "batchsize": batchsize,
                    "Num Ap": Ap,
                    "Num Ad": Ad,
                    "Num Mp": Mp,
                    "Num Md": Md,
                    "Num M": M,
                    "NoI_bw(GBps)": NoI_bw,
                    "tokens_per_sec": tokens_per_sec,
                    "TTFT(s)": ttft,
                }
            if tokens_per_sec > highest_tp:
                highest_tp = tokens_per_sec
            

    print(f"Highest tokens_per_sec/TTFT score: {score}")
    print(f"Best configuration: {best_config}")
    print(f"Highest tokens_per_sec: {highest_tp}")
    # 4. Create DataFrame and save to CSV
    df = pd.DataFrame(all_rows)
    
    # Define the new column order, including "batchsize"
    summary_cols = ["batchsize", "Num M", "Num Mp", "Num Md", "Num Ap", "Num Ad",  "NoI_bw(GBps)", "tokens_per_sec", "TTFT(s)"]
    df = df[summary_cols]

    out_csv = f"results_summary_{Exp_name}_{placer}_{req_scheduler}_{task_scheduler}.csv"
    df.to_csv(out_csv, index=False)

    print("-" * 30)
    print(f"✅ Saved {len(df)} total results to {out_csv}")
    print(df.head())
    print("-" * 30)
    return df

if __name__ == "__main__":
    models = ["NEMO", "MAMBA2", "LLAMA3"]
    # labels = ["PLCMT", "ELASTIC"]
    # labels = ["E+D"]
    labels = ["Results"]
    datasets = ["ARXIV", "LW", "CHAT", "BWB"]
    indexes = [3,1,1]
    # task_schedulers = ["static", "elastic"]
    task_schedulers = ["static"]
    # placers = ["rr", "bw"]
    placers = ["bw"]
    req_scheduler = "static"
    
    for model in models:
        for dataset in datasets:
            for label in labels:
                index = indexes[models.index(model)]
                placer = placers[labels.index(label)]
                task_scheduler = task_schedulers[labels.index(label)]
                main(label=label, index=index, placer=placer, req_scheduler=req_scheduler, task_scheduler=task_scheduler, dataset=dataset, model=model)
    
    
    # df = main()
    # unique_tokens_per_sec = sorted(df['tokens_per_sec'].unique())
    # for i in range(20):
    #     print(f"Top {i+1}: {unique_tokens_per_sec[-(i+1)]} tokens/sec")
    #     print(f"Config: {df[df['tokens_per_sec']==unique_tokens_per_sec[-(i+1)]].to_dict(orient='records')}")      
