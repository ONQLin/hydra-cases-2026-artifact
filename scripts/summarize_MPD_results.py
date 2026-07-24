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

def main():
    # 1. Define the list of result directories to process
    placer = "bw"
    task_scheduler = "static"
    req_scheduler = "static"
    dataset = "BWB"
    model = "LLAMA3"
    results_dirs = [
        f"{model}-{dataset}-Results/simulator_output_{model}-{dataset}_{placer}placement_bs1_{req_scheduler}req_{task_scheduler}task",
        f"{model}-{dataset}-Results/simulator_output_{model}-{dataset}_{placer}placement_bs2_{req_scheduler}req_{task_scheduler}task",
        f"{model}-{dataset}-Results/simulator_output_{model}-{dataset}_{placer}placement_bs4_{req_scheduler}req_{task_scheduler}task",
        f"{model}-{dataset}-Results/simulator_output_{model}-{dataset}_{placer}placement_bs8_{req_scheduler}req_{task_scheduler}task",
        f"{model}-{dataset}-Results/simulator_output_{model}-{dataset}_{placer}placement_bs12_{req_scheduler}req_{task_scheduler}task",
    ]

    Exp_name = results_dirs[0].split("/")[0].split("-Res")[0]
    all_rows = [] # Use a list to collect data from all directories

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
            BW_config = config["chips_config"]["D2D_NoI_bw"]
            NoI_bw = int(BW_config)
            
            alloc = config["placmt_config"]["chiplet_alloc"]
            chiplet_alloc = ast.literal_eval(alloc) 
            Ap = chiplet_alloc["tscs_p"]
            Ad = chiplet_alloc["tscs_d"]
            M = chiplet_alloc["HBM3"]
            if Ap == 0:
                continue  # skip configurations with Ap = 0
            # Append the row, **including the batchsize**
            all_rows.append({
                "batchsize": batchsize, # <--- Added column
                "Num Ap": Ap,
                "Num Ad": Ad,
                "Num M": M,
                "NoI_bw(GBps)": NoI_bw,
                "tokens_per_sec": tokens_per_sec,
                "TTFT(s)": ttft,
            })
            
    # 4. Create DataFrame and save to CSV
    df = pd.DataFrame(all_rows)
    
    # Define the new column order, including "batchsize"
    summary_cols = ["batchsize", "Num M", "Num Ap", "Num Ad", "NoI_bw(GBps)", "tokens_per_sec", "TTFT(s)"]
    df = df[summary_cols]

    out_csv = f"results_summary_{Exp_name}_{placer}_{req_scheduler}_{task_scheduler}.csv"
    df.to_csv(out_csv, index=False)

    print("-" * 30)
    print(f"✅ Saved {len(df)} total results to {out_csv}")
    print(df.head())
    print("-" * 30)
    return df

if __name__ == "__main__":
    df = main()
    unique_tokens_per_sec = sorted(df['tokens_per_sec'].unique())
    for i in range(20):
        print(f"Top {i+1}: {unique_tokens_per_sec[-(i+1)]} tokens/sec")
        print(f"Config: {df[df['tokens_per_sec']==unique_tokens_per_sec[-(i+1)]].to_dict(orient='records')}")      
