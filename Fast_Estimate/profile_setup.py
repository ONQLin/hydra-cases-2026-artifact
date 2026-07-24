import sys
import os

# Get the path to the directory containing both Fast_Estimate and Sim
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add this parent directory to the system path
sys.path.append(parent_dir)

from Sim.config.model_config import Llama3_8BModelConfig, Llama3_8B_BlockT_Config, \
Mamba2_3BModelConfig, Mamba2_3B_BlockM_Config, Nemotron_4BModelConfig, BaseBlockConfig
import Sim.common as common

import importlib
import inspect
from typing import Tuple, List
import numpy as np
import pandas as pd

def get_means_from_csv(file_path, nrows=1000):
    """
    Reads the first `nrows` rows of a CSV and returns the mean of each column.
    
    Args:
        file_path (str): Path to the CSV file.
        nrows (int): Number of rows to read (default=1000).
    
    Returns:
        pandas.Series: Column means.
    """
    # Read the CSV, limited to the first nrows
    df = pd.read_csv(file_path, nrows=nrows)
    
    # Compute mean of each column
    means = df.mean(numeric_only=True)
    
    return means

def get_pf_attention_count(batch_size: int, L_seq: int, embedding_dim: int, q_heads: int, kv_heads: int):
    B = batch_size
    L = L_seq
    D = embedding_dim
    # Per-KV head inner dim; supports GQA where kv_heads may be < q_heads
    d_k = D / kv_heads  # float on purpose to avoid floor issues
    
    ops = 6.0 * B * L * D * D + 2.0 * B * q_heads * (L * L) * d_k + 2.0 * B * q_heads * (L * L) * d_k
    ele_ops = 0
    
    # count the memory access HBM traffic
    mem_access = 0
    mem_access += B * L * D + 3 * D * D + 2 * B * L * D # QKV proj: load X, load W_qkv, write K,V cache to HBM
    
    return (int(ops), int(ele_ops), int(mem_access))
def get_dc_attention_count(batch_size: int, L_seq: int, embedding_dim: int, q_heads: int, kv_heads: int):
    B = batch_size
    L = L_seq
    D = embedding_dim
    # Per-KV head inner dim; supports GQA where kv_heads may be < q_heads
    d_k = D / kv_heads  # float on purpose to avoid floor issues
    
    ops = B * D * D + B * q_heads * L * d_k + B * q_heads * L * d_k
    ele_ops = 0
    mem_access = 0
    mem_access += B * D + D * D + B * D # Q proj: load X, load W_q, (no K,V writeback)
    mem_access += B * D + B * L * D # QK^T: load Q, stream K from HBM
    mem_access += B * L * D + B * D # AttnV: load V from HBM, write context
    return (int(ops), int(ele_ops), int(mem_access))

def get_pf_ssm_count(batch_size: int, L_seq: int, state_size: int, ED: int):
    """
    Estimates counts based on the 'marca_d' analytical model logic.
    """
    B = batch_size
    L = L_seq
    D = ED
    n = state_size

    # 1. Linear Operations (MACs)
    # Corresponds to 'macs_linear' in reference
    ops = B * L * D * n

    # 2. Element-wise Operations
    # Sum of numerators from Compute_latency1 and Compute_latency2:
    # elem_ops_1 (B*L*D) + elem_ops_2 (B*L*D*n) + non_elem_ops_1 (B*L*D) + elem_ops_3 (3*B*L*D*n)
    ele_ops = (B * L * D) + (B * L * D * n) + (B * L * D) + (3 * B * L * D * n)
    # Simplifies to: 2*B*L*D + 4*B*L*D*n

    # 3. Memory Access (HBM Traffic)
    # Sum of bytes_input, bytes_paramA, bytes_paramB, bytes_paramC, bytes_output
    # Reference: bytes_input (B*L*D), paramA (D), paramB (L*n*B), paramC (L*n*B), output (B*L*D)
    mem_access = (B * L * D) + D + (L * n * B) + (L * n * B) + (B * L * D)
    
    return (int(ops), int(ele_ops), int(mem_access))

def get_dc_ssm_count(batch_size: int, L_seq: int, state_size: int, ED: int):
    """
    Estimates counts for decode step (per token) based on 'marca_d' analytical model.
    Note: L_seq is unused in the compute logic for decode (O(1)).
    """
    B = batch_size
    D = ED
    N = state_size

    # 1. Linear Operations (MACs)
    # Corresponds to 'mac_ops' in reference
    ops = B * D * N

    # 2. Element-wise Operations
    # Sum of numerators from compute_stage1 and compute_stage2:
    # elem_ops (B*D*N + B*D) + exp_ops (B*D) + recur_ops (3*B*D*N)
    ele_ops = (B * D * N + B * D) + (B * D) + (3 * B * D * N)
    # Simplifies to: 4*B*D*N + 2*B*D

    # 3. Memory Access (HBM Traffic)
    # Sum of bytes_input, bytes_paramA, bytes_paramB, bytes_paramC, bytes_output
    # Reference: bytes_input (B*D), paramA (D), paramB (N*B), paramC (N*B), output (B*D)
    mem_access = (B * D) + D + (N * B) + (N * B) + (B * D)

    return (int(ops), int(ele_ops), int(mem_access))

def get_pf_mlp_count(bs: int, f_in: int, f_out: int):
    ops = 2 * bs * f_in * f_out
    ele_ops = 0
    mem_access = bs*f_in + f_in*f_out + bs*f_out # bytes
    return (int(ops), int(ele_ops), int(mem_access))
def get_pf_conv1d_count(bs: int, c_in: int, c_out: int, f_in: int, f_out: int, kernel_size: int):
    ops = 2 * bs * c_in * kernel_size
    ele_ops = 0
    mem_access = c_in*kernel_size + bs*c_in + bs*c_out # bytes
    return (int(ops), int(ele_ops), int(mem_access))

count_est_dict = {
    "Count_MHA_p": get_pf_attention_count,
    "Count_MHA_d": get_dc_attention_count,
    "Count_FC": get_pf_mlp_count,
    "Count_Conv1D": get_pf_conv1d_count,
    "Count_SSM_p": get_pf_ssm_count,
    "Count_SSM_d": get_dc_ssm_count,
}

def get_workload_profile(model_name, workload_name):
    if model_name == "llama3":
        model_config = Llama3_8BModelConfig()
        block_config = Llama3_8B_BlockT_Config()
    elif model_name == "mamba2":
        # TODO: add more models
        model_config = Mamba2_3BModelConfig()
        block_config = Mamba2_3B_BlockM_Config()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if workload_name == "arxiv":
        p_len = 2566
        d_len = 297
    elif workload_name == "bwb":
        p_len = 2445
        d_len = 2119
    elif workload_name == "chat":
        p_len = 56
        d_len = 62
    elif workload_name == "lw":
        p_len = 245
        d_len = 2890
    profile_path = f"Fast_Estimate/workloads_profiles/{model_name}_{workload_name}.json"
    profile_data = {
        "model": model_name,
        "workload": workload_name,
        "prefill_len": p_len,
        "decode_len": d_len,
        "lin_ops_per_token": 0,
        "ele_ops_per_token": 0,
        "mem_access_per_token": 0,
    }
    lin_ops = 0
    ele_ops = 0
    mem_access = 0
    # prefill stage
    prefill_ops = 0 
    # profile_module = importlib.import_module("analytic_profile.tscs")
    for i, layer_items in enumerate(block_config.layers):
        for j, item in enumerate(layer_items):
            if "MHA" in item or "SSM" in item:
                est_name = item+"_p"
            else:
                est_name = item
            est_name = "Count_" + est_name
            
            if est_name in count_est_dict:
                layer_config = block_config.layer_configs[i][j]
                func = count_est_dict[est_name]
                func_args = layer_config.__dict__  # Convert dataclass to dict
                func_args["batch_size"] = 1
                if "_p" in est_name:
                    func_args["L_seq"] = p_len
                else:
                    func_args["bs"] = p_len
                func_params = inspect.signature(func).parameters
                filtered_params = {key: value for key, value in func_args.items() if key in func_params}
                res = func(**filtered_params)
                lin_ops += res[0]
                ele_ops += res[1]
                mem_access += res[2]
                if "MHA" in item:
                    mem_access += min(p_len, block_config.max_position_embeddings) * block_config.states
            else:
                continue
                    
    block_num = model_config.num_layers
    # count the resources req of prefilling stage
    lin_ops *= block_num
    ele_ops *= block_num
    mem_access *= block_num
    print("Prefilling lin ops/token:", lin_ops/(d_len*1e9), "G ele ops/token:", ele_ops/(d_len*1e9), "mem access (Gbytes)/token:", mem_access/(d_len*1e9))
    p_ops = lin_ops
    p_ele_ops = ele_ops
    p_mem_access = mem_access
    profile_data["prefill lin_ops(G)"] = lin_ops / (1e9)
    profile_data["prefill ele_ops(G)"] = ele_ops / (1e9)
    profile_data["prefill mem_access(GB)"] = mem_access / (1e9)
    # count the decoding stage
    for stage in range(d_len):
        decode_ops = 0
        decode_ele_ops = 0
        decode_mem_access = 0
        for i, layer_items in enumerate(block_config.layers):
            for j, item in enumerate(layer_items):
                if "MHA" in item or "SSM" in item:
                    est_name = item+"_d"
                else:
                    est_name = item
                est_name = "Count_" + est_name
                
                
                if est_name in count_est_dict:
                    layer_config = block_config.layer_configs[i][j]
                    func = count_est_dict[est_name]
                    func_args = layer_config.__dict__  # Convert dataclass to dict
                    func_args["batch_size"] = 1
                    if "_d" in est_name:
                        func_args["L_seq"] = min(p_len + stage + 1, block_config.max_position_embeddings)
                    else:                 
                        func_args["bs"] = 1
                    func_params = inspect.signature(func).parameters
                    filtered_params = {key: value for key, value in func_args.items() if key in func_params}
                    res = func(**filtered_params)
                    decode_ops += res[0]
                    decode_ele_ops += res[1]
                    decode_mem_access += res[2]
                    if "MHA" in item:
                        # for MHA, we need to add the KV cache update cost
                        decode_mem_access += min(p_len + stage + 1, block_config.max_position_embeddings) * block_config.states
                else:
                    continue
        decode_ops *= block_num
        decode_ele_ops *= block_num
        decode_mem_access *= block_num
        # whole inference done, accumulate the ops...
        lin_ops += decode_ops
        ele_ops += decode_ele_ops
        mem_access += decode_mem_access
    print("G lin ops/token:", lin_ops/(d_len*1e9), "G ele ops/token:", ele_ops/(d_len*1e9), "mem access (Gbytes)/token:", mem_access/(d_len*1e9))
    profile_data["decode lin_ops(G)"] = (lin_ops-profile_data['prefill lin_ops(G)']*1e9) / (1e9)
    profile_data["decode ele_ops(G)"] = (ele_ops-profile_data['prefill ele_ops(G)']*1e9) / (1e9)
    profile_data["decode mem_access(GB)"] = (mem_access-profile_data['prefill mem_access(GB)']*1e9) / (1e9)
    profile_data["lin_ops_per_token(G)"] = lin_ops / (d_len * 1e9)
    profile_data["ele_ops_per_token(G)"] = ele_ops / (d_len * 1e9)
    profile_data["mem_access_per_token(GB)"] = mem_access / (d_len * 1e9)
    profile_data["param_scale"] = model_config.param_scale
    
    import json
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=4)
        
def get_workload_profile_hybrid(model_name, workload_name):
    if "nemo" in model_name:
        model_config = Nemotron_4BModelConfig()

    if workload_name == "arxiv":
        p_len = 2566
        d_len = 297
    elif workload_name == "bwb":
        p_len = 2445
        d_len = 2119
    elif workload_name == "chat":
        p_len = 56
        d_len = 62
    elif workload_name == "lw":
        p_len = 245
        d_len = 2890
    profile_path = f"Fast_Estimate/workloads_profiles/{model_name}_{workload_name}.json"
    profile_data = {
        "model": model_name,
        "workload": workload_name,
        "prefill_len": p_len,
        "decode_len": d_len,
        "lin_ops_per_token": 0,
        "ele_ops_per_token": 0,
        "mem_access_per_token": 0,
    }
    lin_ops = 0
    ele_ops = 0
    mem_access = 0
    # prefill stage
    prefill_ops = 0 

    for blk_no, blk_idx in enumerate(model_config.block_type_sequence):
        block_config: BaseBlockConfig = model_config.hybrid_blocks[blk_idx]
        for i, layer_items in enumerate(block_config.layers):
            for j, item in enumerate(layer_items):
                if "MHA" in item or "SSM" in item:
                    est_name = item+"_p"
                else:
                    est_name = item
                est_name = "Count_" + est_name
                
                if est_name in count_est_dict:
                    layer_config = block_config.layer_configs[i][j]
                    func = count_est_dict[est_name]
                    func_args = layer_config.__dict__  # Convert dataclass to dict
                    func_args["batch_size"] = 1
                    if "_p" in est_name:
                        func_args["L_seq"] = p_len
                    else:
                        func_args["bs"] = p_len
                    func_params = inspect.signature(func).parameters
                    filtered_params = {key: value for key, value in func_args.items() if key in func_params}
                    res = func(**filtered_params)
                    lin_ops += res[0]
                    ele_ops += res[1]
                    mem_access += res[2]
                    if "MHA" in item:
                        mem_access += min(p_len, block_config.max_position_embeddings) * block_config.states
                else:
                    continue
    print("Prefilling lin ops/token:", lin_ops/(d_len*1e9), "G ele ops/token:", ele_ops/(d_len*1e9), "mem access (Gbytes)/token:", mem_access/(d_len*1e9))
    profile_data["prefill lin_ops(G)"] = lin_ops / (1e9)
    profile_data["prefill ele_ops(G)"] = ele_ops / (1e9)
    profile_data["prefill mem_access(GB)"] = mem_access / (1e9)
    
    for stage in range(d_len):
        decode_ops = 0
        decode_ele_ops = 0
        decode_mem_access = 0
        
        for blk_no, blk_idx in enumerate(model_config.block_type_sequence):
            block_config: BaseBlockConfig = model_config.hybrid_blocks[blk_idx]
            for i, layer_items in enumerate(block_config.layers):
                for j, item in enumerate(layer_items):
                    if "MHA" in item or "SSM" in item:
                        est_name = item+"_d"
                    else:
                        est_name = item
                    est_name = "Count_" + est_name
                    
                    
                    if est_name in count_est_dict:
                        layer_config = block_config.layer_configs[i][j]
                        func = count_est_dict[est_name]
                        func_args = layer_config.__dict__  # Convert dataclass to dict
                        func_args["batch_size"] = 1
                        if "_d" in est_name:
                            func_args["L_seq"] = min(p_len + stage + 1, block_config.max_position_embeddings)
                        else:                 
                            func_args["bs"] = 1
                        func_params = inspect.signature(func).parameters
                        filtered_params = {key: value for key, value in func_args.items() if key in func_params}
                        res = func(**filtered_params)
                        decode_ops += res[0]
                        decode_ele_ops += res[1]
                        decode_mem_access += res[2]
                        if "MHA" in item:
                            # for MHA, we need to add the KV cache update cost
                            decode_mem_access += min(p_len + stage + 1, block_config.max_position_embeddings) * block_config.states
                    else:
                        continue
    
    
    profile_data["prefill lin_ops(G)"] = lin_ops / (1e9)
    profile_data["prefill ele_ops(G)"] = ele_ops / (1e9)
    profile_data["prefill mem_access(GB)"] = mem_access / (1e9)
    # count the decoding stage
    for stage in range(d_len):
        decode_ops = 0
        decode_ele_ops = 0
        decode_mem_access = 0
        for i, layer_items in enumerate(block_config.layers):
            for j, item in enumerate(layer_items):
                if "MHA" in item or "SSM" in item:
                    est_name = item+"_d"
                else:
                    est_name = item
                est_name = "Count_" + est_name
                
                
                if est_name in count_est_dict:
                    layer_config = block_config.layer_configs[i][j]
                    func = count_est_dict[est_name]
                    func_args = layer_config.__dict__  # Convert dataclass to dict
                    func_args["batch_size"] = 1
                    if "_d" in est_name:
                        func_args["L_seq"] = min(p_len + stage + 1, block_config.max_position_embeddings)
                    else:                 
                        func_args["bs"] = 1
                    func_params = inspect.signature(func).parameters
                    filtered_params = {key: value for key, value in func_args.items() if key in func_params}
                    res = func(**filtered_params)
                    decode_ops += res[0]
                    decode_ele_ops += res[1]
                    decode_mem_access += res[2]
                    if "MHA" in item:
                        # for MHA, we need to add the KV cache update cost
                        decode_mem_access += min(p_len + stage + 1, block_config.max_position_embeddings) * block_config.states
                else:
                    continue
        # whole inference done, accumulate the ops...
        lin_ops += decode_ops
        ele_ops += decode_ele_ops
        mem_access += decode_mem_access
    print("G lin ops/token:", lin_ops/(d_len*1e9), "G ele ops/token:", ele_ops/(d_len*1e9), "mem access (Gbytes)/token:", mem_access/(d_len*1e9))
    profile_data["decode lin_ops(G)"] = (lin_ops-profile_data['prefill lin_ops(G)']*1e9) / (1e9)
    profile_data["decode ele_ops(G)"] = (ele_ops-profile_data['prefill ele_ops(G)']*1e9) / (1e9)
    profile_data["decode mem_access(GB)"] = (mem_access-profile_data['prefill mem_access(GB)']*1e9) / (1e9)
    profile_data["lin_ops_per_token(G)"] = lin_ops / (d_len * 1e9)
    profile_data["ele_ops_per_token(G)"] = ele_ops / (d_len * 1e9)
    profile_data["mem_access_per_token(GB)"] = mem_access / (d_len * 1e9)
    profile_data["param_scale"] = model_config.param_scale
    
    import json
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=4)

if __name__ == "__main__":
    # get_workload_profile("llama3", "arxiv")
    # get_workload_profile("llama3", "bwb")
    # get_workload_profile("llama3", "chat")
    # get_workload_profile("llama3", "lw")
    
    # get_workload_profile("mamba2", "chat")
    # get_workload_profile("mamba2", "bwb")
    # get_workload_profile("mamba2", "arxiv")
    # get_workload_profile("mamba2", "lw")
    
    get_workload_profile_hybrid("nemo", "chat")
    get_workload_profile_hybrid("nemo", "bwb")
    get_workload_profile_hybrid("nemo", "arxiv")
    get_workload_profile_hybrid("nemo", "lw")
    # a_results = get_means_from_csv("dataset/arxiv/arxiv_summarization_stats_llama3.csv", nrows=1000)
    # b_results = get_means_from_csv("dataset/bwb/bwb_translation_stats_llama3.csv", nrows=1000)
    # print("arxiv3k:", a_results)
    # print("bwb:", b_results)