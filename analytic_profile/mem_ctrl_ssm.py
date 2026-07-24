# Model the memory controller for computing chiplets (SSMs)

import numpy as np
from typing import Tuple, List
import math
from Sim.metrics.monitor import analytics

def get_pf_ssm_latency(ED, L_seq, Total_States:int, Col_PE:int, Row_PE:int, Num_Array:int, C_sram:int=16, DMAs:int=8, Sram_bw:int=32, DRAM="HBM3", acc_freq:int = 1e9) -> Tuple[int, int, int, List]:
    Word_dic = {"HBM2": 128, "DDR4": 64, "HBM3": 128}    # Word size of DRAM

    # left input
    Delta_size = [L_seq, ED]
    A_size = [ED]
    D_size = [ED]
    U_size = [L_seq, ED]

    # Upper input 
    B_size = [L_seq, Total_States]
    C_size = [L_seq, Total_States]

    c_time = 0
    comp_latency = 0
    dram_latency = []

    DMA_setup = 50

    # initialize 
    #Init_bits = C_sram * Col_PE * Row_PE * Num_Array
    Init_bits = C_sram * DMAs   # Assume 8 channels DMA
    Init_itr = int((Col_PE + Row_PE)*Num_Array/DMAs)
    Num_word = int(Init_bits / Word_dic[DRAM])   # HBM2
    Num_word_1channel = int(Num_word / DMAs)        # 1 channels #words
    Init_latency = Init_itr * int(Num_word_1channel*Word_dic[DRAM]/Sram_bw) + 50  # assume 32 SRAM bits width

    dram_latency.append((0, Init_latency, "both"))

    # Count the loaded params
    Col_Sram = C_sram
    Row_Sram = C_sram # monitor the data usage
    

    for n in range(0, int(ED/Row_PE), Num_Array):
        # Check Row
        Row_Sram -= 2*8 # A and D
        if Row_Sram <= 0:
            reload_latency = int(Row_PE/DMAs)*Num_Array * int(Num_word_1channel*Word_dic[DRAM]/Sram_bw) + 50
            Row_Sram = C_sram
            dram_latency.append((c_time, reload_latency, "left"))
            c_time += reload_latency
        for k in range(0, int(Total_States/Col_PE)):
            # Check Col
            c_time += 4+Row_PE-1+Col_PE+3 
            for l in range(1, L_seq):
                # check the input data
                Col_Sram -= 2*8 # BC used
                Row_Sram -= 2*8 # delta U used
                if Col_Sram <= 0:
                    reload_latency = int(Col_PE/DMAs) *  int(Num_word_1channel*Word_dic[DRAM]/Sram_bw) + 50
                    Col_Sram = C_sram
                    dram_latency.append((c_time, reload_latency, "top"))
                    c_time += reload_latency
                if Row_Sram <= 0:
                    reload_latency = int(Row_PE/DMAs)*Num_Array * int(Num_word_1channel*Word_dic[DRAM]/Sram_bw) + 50
                    Row_Sram = C_sram
                    dram_latency.append((c_time, reload_latency, "left"))
                    c_time += reload_latency
                #for r in range(0, Row_PE):
                #for m in range(0, Col_PE):
                c_time+=4
                
    # print("Total time: ", c_time)
    # print("DRAM latency: ", dram_latency)

    total_d_latency = sum(lat[1] for lat in dram_latency)
    # print("Total DRAM latency: ", total_d_latency)\
    used_bw = int(DMAs * Sram_bw / 8) # in B/cycle
    used_bw = used_bw * acc_freq/1e9 # in GB/s
    
    peak_bw = 1200 # Assume 1200 GB/s for HBM3
    indicator = min(used_bw / peak_bw, 1)  # in percentage
    # make up other analytical value avg util power
    avg_util = 45 + indicator*45  # Assume 50% utilization for simplicity
    power = 100 + indicator*150  # Assume 100mW power for simplicity

    return analytics(int(c_time), total_d_latency, used_bw, dram_latency, avg_util, power)



# TODO: integrate with the VU Array
def get_dc_ssm_latency(ED, L_seq, Total_States:int, Col_PE:int, Row_PE:int, Num_Array:int, C_sram:int=16, DMAs:int=8, Sram_bw:int=32, DRAM="HBM3", acc_freq:int = 1e9) -> Tuple[int, int, int, List]:
    Word_dic = {"HBM2": 128, "DDR4": 64, "HBM3": 128}    # Word size of DRAM

    # left input
    Delta_size = [L_seq, ED]
    A_size = [ED]
    D_size = [ED]
    U_size = [L_seq, ED]

    # Upper input 
    B_size = [L_seq, Total_States]
    C_size = [L_seq, Total_States]

    c_time = 0
    comp_latency = 0
    dram_latency = []

    DMA_setup = 50

    # initialize 
    #Init_bits = C_sram * Col_PE * Row_PE * Num_Array
    Init_bits = C_sram * DMAs   # Assume 8 channels DMA
    Init_itr = int((Col_PE + Row_PE)*Num_Array/DMAs)
    Num_word = int(Init_bits / Word_dic[DRAM])   # HBM2
    Num_word_1channel = int(Num_word / DMAs)        # 1 channels #words
    Init_latency = Init_itr * int(Num_word_1channel*Word_dic[DRAM]/Sram_bw) + 50  # assume 32 SRAM bits width

    dram_latency.append((0, Init_latency, "both"))

    # Count the loaded params
    Col_Sram = C_sram
    Row_Sram = C_sram # monitor the data usage
    

    for n in range(0, int(ED/Row_PE), Num_Array):
        # Check Row
        Row_Sram -= 2*8 # A and D
        if Row_Sram <= 0:
            reload_latency = int(Row_PE/DMAs)*Num_Array * int(Num_word_1channel*Word_dic[DRAM]/Sram_bw) + 50
            Row_Sram = C_sram
            dram_latency.append((c_time, reload_latency, "left"))
            c_time += reload_latency
        for k in range(0, int(Total_States/Col_PE)):
            # Check Col
            c_time += 4+Row_PE-1+Col_PE+3 
            for l in range(1, 2):
                # check the input data
                Col_Sram -= 2*8 # BC used
                Row_Sram -= 2*8 # delta U used
                if Col_Sram <= 0:
                    reload_latency = int(Col_PE/DMAs) *  int(Num_word_1channel*Word_dic[DRAM]/Sram_bw) + 50
                    Col_Sram = C_sram
                    dram_latency.append((c_time, reload_latency, "top"))
                    c_time += reload_latency
                if Row_Sram <= 0:
                    reload_latency = int(Row_PE/DMAs)*Num_Array * int(Num_word_1channel*Word_dic[DRAM]/Sram_bw) + 50
                    Row_Sram = C_sram
                    dram_latency.append((c_time, reload_latency, "left"))
                    c_time += reload_latency
                #for r in range(0, Row_PE):
                #for m in range(0, Col_PE):
                c_time+=4
                
    # print("Total time: ", c_time)
    # print("DRAM latency: ", dram_latency)

    total_d_latency = sum(lat[1] for lat in dram_latency) * L_seq
    c_time = c_time * L_seq
    #dram_latency = [(lat[0] * L_seq, lat[1] * L_seq, lat[2]) for lat in dram_latency]
    # print("Total DRAM latency: ", total_d_latency)\
    used_bw = int(DMAs * Sram_bw / 8) # in B/cycle
    used_bw = used_bw * acc_freq/1e9 # in GB/s
    
    peak_bw = 1200 # Assume 1200 GB/s for HBM3
    indicator = min(used_bw / peak_bw, 1)  # in percentage
    # make up other analytical value avg util power
    avg_util = 45 + indicator*45  # Assume 50% utilization for simplicity
    power = 100 + indicator*150  # Assume 100mW power for simplicity

    return analytics(int(c_time), total_d_latency, used_bw, dram_latency, avg_util, power)


def get_dc_mlp_latency(bs: int, f_in: int, f_out: int,
                       Col_PE: int, Row_PE: int, Num_Array: int,
                       C_sram: int = 16, DMAs: int = 8, Sram_bw: int = 32,
                       DRAM: str = "HBM3", acc_freq: int = int(1e9)) -> int:
    """
    Estimate latency (in cycles) for an MLP layer on a systolic array accelerator.
    """
    # 1. GEMM Shape: (bs x f_out) = (bs x f_in) x (f_in x f_out)
    total_MACs = bs * f_in * f_out

    # 2. Compute latency:
    # Systolic array pipeline latency ~ f_in + f_out + bs (simplified),
    # but for steady-state, the number of tiles matter more.
    tiles_bs = math.ceil(bs / Row_PE)
    tiles_fout = math.ceil(f_out / Col_PE)

    # Number of batches that can be handled in parallel
    parallel_batches = min(Num_Array, tiles_bs)

    total_tiles = tiles_bs * tiles_fout

    # Compute latency per tile (each array handles a bs_tile x fout_tile matrix)
    compute_latency_per_tile = f_in + max(Row_PE, Col_PE)  # Systolic wave cycles
    compute_cycles = math.ceil(total_tiles / Num_Array) * compute_latency_per_tile

    # 3. Memory latency (simplified, assuming streaming from DRAM)
    # Data movement:
    input_size = bs * f_in * 4     # 4 bytes per FP32 element
    weight_size = f_in * f_out * 4
    output_size = bs * f_out * 4

    # 4. SRAM bandwidth constraint (if applicable)
    # Assume streaming from SRAM to PEs also needs cycles
    sram_bw_bytes = DMAs * Sram_bw  # bytes per cycle
    sram_cycles = math.ceil((input_size + weight_size+output_size) / sram_bw_bytes)

    # 5. Total cycles
    total_cycles = max(compute_cycles, sram_cycles)
    
    used_bw = int(DMAs * Sram_bw / 8) # in B/cycle
    used_bw = used_bw * acc_freq/1e9 # in GB/s

    peak_bw = 1200 # Assume 1200 GB/s for HBM3
    indicator = min(used_bw / peak_bw, 1)  # in percentage
    # make up other analytical value avg util power
    avg_util = 38 + indicator*38  # Assume 50% utilization for simplicity
    power = 100 + indicator*100  # Assume 100mW power for simplicity

    return analytics(int(total_cycles), sram_cycles, used_bw, [], avg_util, power)

def get_pf_mlp_latency(bs: int, f_in: int, f_out: int,
                       Col_PE: int, Row_PE: int, Num_Array: int,
                       C_sram: int = 16, DMAs: int = 8, Sram_bw: int = 32,
                       DRAM: str = "HBM3", acc_freq: int = int(1e9)) -> int:
    """
    Estimate latency (in cycles) for an MLP layer on a tensor-core-like accelerator.
    """

    # 1. Tensor Core tile shape (typically 16x16x16 for FP16)
    TILE_M, TILE_K, TILE_N = 16, 16, 16

    # 2. GEMM shape: (bs x f_out) = (bs x f_in) x (f_in x f_out)
    total_MACs = bs * f_in * f_out

    # 3. Compute latency: count how many Tensor Core tiles are needed
    tiles_M = math.ceil(bs / TILE_M)
    tiles_K = math.ceil(f_in / TILE_K)
    tiles_N = math.ceil(f_out / TILE_N)

    total_tiles = tiles_M * tiles_K * tiles_N

    # 4. Tensor Core throughput: how many tiles can run per cycle per core
    # Assume each PE handles 1 tile every 4 cycles (based on NVIDIA estimates)
    tc_latency_per_tile = 4

    # Assume parallel tile execution across all PEs (Num_Array * Row_PE * Col_PE)
    total_PEs = Num_Array * Row_PE * Col_PE
    compute_cycles = math.ceil(total_tiles / total_PEs) * tc_latency_per_tile

    # 5. Memory latency
    # Assume FP16 = 2 bytes per element
    input_size = bs * f_in * 2
    weight_size = f_in * f_out * 2
    output_size = bs * f_out * 2    
    # 6. SRAM bandwidth latency
    sram_bw_bytes = DMAs * Sram_bw
    sram_cycles = math.ceil((input_size + weight_size+output_size) / sram_bw_bytes)

    # 7. Total latency (overlapping memory and compute where possible)
    total_cycles = max(compute_cycles, sram_cycles)

    used_bw = int(DMAs * Sram_bw / 8) # in B/cycle
    used_bw = used_bw * acc_freq/1e9 # in GB/s

    peak_bw = 1200 # Assume 1200 GB/s for HBM3
    indicator = min(used_bw / peak_bw, 1)  # in percentage
    # make up other analytical value avg util power
    avg_util = 38 + indicator*38  # Assume 50% utilization for simplicity
    power = 100 + indicator*100  # Assume 100mW power for simplicity

    return analytics(int(total_cycles), sram_cycles, used_bw, [], avg_util, power)



def get_dc_conv1d_latency(bs: int, c_in: int, c_out: int, f_in: int, f_out: int, kernel_size: int,
                          Col_PE: int, Row_PE: int, Num_Array: int,
                          C_sram: int = 16, DMAs: int = 8, Sram_bw: int = 32,
                          DRAM: str = "HBM3", acc_freq: int = int(1e9)) -> int:
    """
    Estimate latency (in cycles) for Conv1D on a systolic array accelerator.
    """

    # 1. Convert Conv1D to GEMM:
    # After im2col: matrix A = (bs * f_out) x (c_in * kernel)
    # Matrix B (weights) = (c_in * kernel) x c_out
    m = bs * f_out
    k = c_in * kernel_size
    n = c_out
    total_MACs = m * k * n

    # 2. Compute tiling based on PE shape
    tiles_m = math.ceil(m / Row_PE)
    tiles_n = math.ceil(n / Col_PE)

    # Distribute across Num_Array arrays
    total_tiles = tiles_m * tiles_n
    active_arrays = min(Num_Array, tiles_m)
    compute_latency_per_tile = k + max(Row_PE, Col_PE)
    compute_cycles = math.ceil(total_tiles / active_arrays) * compute_latency_per_tile

    # 3. Memory size
    # Assuming FP32 (4 bytes), im2col input, weights, output
    input_size = bs * f_in * c_in * 4
    weight_size = c_out * c_in * kernel_size * 4
    output_size = bs * f_out * c_out * 4

    # 5. SRAM bandwidth (simplified)
    sram_bw_bytes = DMAs * Sram_bw
    sram_cycles = math.ceil((input_size + weight_size+output_size) / sram_bw_bytes)

    # 6. Final latency (assuming perfect overlapping of memory and compute)
    total_cycles = max(compute_cycles, sram_cycles)

    used_bw = int(DMAs * Sram_bw / 8) # in B/cycle
    used_bw = used_bw * acc_freq/1e9 # in GB/s

    peak_bw = 1200 # Assume 1200 GB/s for HBM3
    indicator = min(used_bw / peak_bw, 1)  # in percentage
    # make up other analytical value avg util power
    avg_util = 38 + indicator*38  # Assume 50% utilization for simplicity
    power = 100 + indicator*100  # Assume 100mW power for simplicity

    return analytics(int(total_cycles), sram_cycles, used_bw, [], avg_util, power)

def get_pf_conv1d_latency(bs: int, c_in: int, c_out: int, f_in: int, f_out: int, kernel_size: int,
                          Col_PE: int, Row_PE: int, Num_Array: int,
                          C_sram: int = 16, DMAs: int = 8, Sram_bw: int = 32,
                          DRAM: str = "HBM3", acc_freq: int = int(1e9)) -> int:
    """
    Estimate latency (in cycles) for Conv1D on a tensor-core-like accelerator.
    """
    # 1. GEMM shape from im2col-reduced Conv1D
    M = bs * f_out                 # number of output positions per batch
    K = c_in * kernel_size        # filter size per output point
    N = c_out                     # number of filters

    # 2. Tensor Core tile shape (e.g., 16x16x16 FP16)
    TILE_M, TILE_K, TILE_N = 16, 16, 16

    tiles_M = math.ceil(M / TILE_M)
    tiles_K = math.ceil(K / TILE_K)
    tiles_N = math.ceil(N / TILE_N)
    total_tiles = tiles_M * tiles_K * tiles_N

    # 3. Execution: Assume each PE can do 1 tile per 4 cycles
    # Total PEs = Num_Array * Row_PE * Col_PE
    total_PEs = Num_Array * Row_PE * Col_PE
    tc_latency_per_tile = 4  # cycles per tile
    compute_cycles = math.ceil(total_tiles / total_PEs) * tc_latency_per_tile

    # 4. Memory cost (FP16 = 2 bytes)
    input_size = bs * f_in * c_in * 2  # original input
    weight_size = c_out * c_in * kernel_size * 2
    output_size = bs * f_out * c_out * 2
    total_mem = input_size + weight_size + output_size

    # 6. SRAM bandwidth
    sram_bw_bytes = DMAs * Sram_bw
    sram_cycles = math.ceil(total_mem / sram_bw_bytes)

    # 7. Final latency estimate
    total_cycles = max(compute_cycles, sram_cycles)
    used_bw = int(DMAs * Sram_bw / 8) # in B/cycle
    used_bw = used_bw * acc_freq/1e9 # in GB/s

    peak_bw = 1200 # Assume 1200 GB/s for HBM3
    indicator = min(used_bw / peak_bw, 1)  # in percentage
    # make up other analytical value avg util power
    avg_util = 38 + indicator*38  # Assume 50% utilization for simplicity
    power = 100 + indicator*100  # Assume 100mW power for simplicity

    return analytics(int(total_cycles), sram_cycles, used_bw, [], avg_util, power)



def get_pf_attention_latency(bs: int, L_seq: int, embedding_dim: int, q_heads: int, kv_heads: int,
                              Col_PE: int, Row_PE: int, Num_Array: int,
                              C_sram: int = 16, DMAs: int = 8, Sram_bw: int = 32,
                              DRAM: str = "HBM3", acc_freq: int = int(1e9)) -> Tuple[int, int, int, List]:
    """
    Estimate latency (in cycles) of attention prefill phase on systolic array.
    Returns: (QKV_proj_latency, QK_T_latency, AttnV_latency, breakdown_list)
    """

    d_q = embedding_dim // q_heads
    d_kv = embedding_dim // kv_heads

    # GEMM estimate for systolic array
    def systolic_gemm_latency(M, K, N, Row_PE, Col_PE, Num_Array):
        tiles_M = math.ceil(M / Row_PE)
        tiles_N = math.ceil(N / Col_PE)
        total_tiles = tiles_M * tiles_N
        active_arrays = min(Num_Array, tiles_M)
        latency_per_tile = K + max(Row_PE, Col_PE)
        return math.ceil(total_tiles / active_arrays) * latency_per_tile

    # --- Step 1: Q, K, V projections ---
    qkv_latency = 0
    for _ in range(3):  # Q, K, V
        qkv_latency += systolic_gemm_latency(bs * L_seq, embedding_dim, embedding_dim, Row_PE, Col_PE, Num_Array)

    # --- Step 2: Q × Kᵀ ---
    qk_latency = systolic_gemm_latency(bs * q_heads, L_seq, L_seq, Row_PE, Col_PE, Num_Array)

    # --- Step 3: Attn × V ---
    attnv_latency = systolic_gemm_latency(bs * q_heads, L_seq, d_q, Row_PE, Col_PE, Num_Array)

    # --- Memory size (FP16 = 2 bytes) ---
    input_size = bs * L_seq * embedding_dim * 2
    weight_size = embedding_dim * embedding_dim * 3 * 2
    kv_size = bs * L_seq * embedding_dim * 2
    output_size = bs * L_seq * embedding_dim * 2
    total_mem = input_size + weight_size + kv_size + output_size
    # --- SRAM stall cycles ---
    sram_bw_bytes = DMAs * Sram_bw
    sram_cycles = math.ceil(total_mem / sram_bw_bytes)

    total_latency = max(qkv_latency + qk_latency + attnv_latency, sram_cycles)
    used_bw = int(DMAs * Sram_bw / 8) # in B/cycle
    used_bw = used_bw * acc_freq/1e9 # in GB/s

    peak_bw = 1200 # Assume 1200 GB/s for HBM3
    indicator = min(used_bw / peak_bw, 1)  # in percentage
    # make up other analytical value avg util power
    avg_util = 30 + indicator*30  # Assume 50% utilization for simplicity
    power = 100 + indicator*80  # Assume 100mW power for simplicity

    return analytics(int(total_latency), sram_cycles, used_bw, [], avg_util, power)

def get_dc_attention_latency(bs: int, L_seq: int, embedding_dim: int, q_heads: int, kv_heads: int,
                              Col_PE: int, Row_PE: int, Num_Array: int,
                              C_sram: int = 16, DMAs: int = 8, Sram_bw: int = 32,
                              DRAM: str = "HBM3", acc_freq: int = int(1e9)) -> Tuple[int, int, int, List]:
    """
    Estimate latency (in cycles) of attention decoding (one token) on a systolic array.
    Returns: (Q_proj_latency, QK_T_latency, AttnV_latency, breakdown_list)
    """

    d_q = embedding_dim // q_heads
    d_kv = embedding_dim // kv_heads

    def systolic_gemm_latency(M, K, N, Row_PE, Col_PE, Num_Array):
        tiles_M = math.ceil(M / Row_PE)
        tiles_N = math.ceil(N / Col_PE)
        total_tiles = tiles_M * tiles_N
        active_arrays = min(Num_Array, tiles_M)
        latency_per_tile = K + max(Row_PE, Col_PE)
        return math.ceil(total_tiles / active_arrays) * latency_per_tile

    # --- Step 1: Q projection ---
    # Shape: (bs × 1) × embedding_dim
    q_proj_latency = systolic_gemm_latency(bs, embedding_dim, embedding_dim, Row_PE, Col_PE, Num_Array)

    # --- Step 2: Q × Kᵀ ---
    # Shape: (bs × q_heads × 1 × d_q) × (bs × kv_heads × L_seq × d_kv)
    qk_latency = systolic_gemm_latency(bs * q_heads, d_q, L_seq, Row_PE, Col_PE, Num_Array)

    # --- Step 3: Attn × V ---
    attnv_latency = systolic_gemm_latency(bs * q_heads, L_seq, d_q, Row_PE, Col_PE, Num_Array)

    # --- Memory movement estimate (cached keys/values) ---
    # 1 token input: (bs × 1 × embedding_dim)
    # cache: K/V are bs × L_seq × embedding_dim (already available)
    input_size = bs * embedding_dim * 2  # FP16
    cached_kv_size = bs * L_seq * embedding_dim * 2
    output_size = bs * embedding_dim * 2
    total_mem = input_size + cached_kv_size + output_size

    sram_bw_bytes = DMAs * Sram_bw
    sram_cycles = math.ceil(total_mem / sram_bw_bytes)
    
    total_latency = max(q_proj_latency + qk_latency + attnv_latency, sram_cycles)
    used_bw = int(DMAs * Sram_bw / 8) # in B/cycle
    used_bw = used_bw * acc_freq/1e9 # in GB/s

    peak_bw = 1200 # Assume 1200 GB/s for HBM3
    indicator = min(used_bw / peak_bw, 1)  # in percentage
    # make up other analytical value avg util power
    avg_util = 30 + indicator*30  # Assume 50% utilization for simplicity
    power = 100 + indicator*80  # Assume 100mW power for simplicity

    return analytics(int(total_latency), sram_cycles, used_bw, [], avg_util, power)

def rms_norm_est(in_size, Col_PE: int, Row_PE: int, num_array: int) -> int:
    return analytics(int(4*in_size/(Col_PE*Row_PE*num_array) + 4), -1, -1, [], 20, 100)  

def silu_est(in_size, Col_PE: int, Row_PE: int, num_array: int) -> int:
    return analytics(int(3*in_size/(Col_PE*Row_PE*num_array) + 4), -1, -1, [], 20, 100)  

def softplus_est(in_size, Col_PE: int, Row_PE: int, num_array: int) -> int:
    return analytics(int(3*in_size/(Col_PE*Row_PE*num_array) + 4), -1, -1, [], 20, 100)  