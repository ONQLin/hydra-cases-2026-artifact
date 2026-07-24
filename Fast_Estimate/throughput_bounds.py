import sys
import os
from math import sqrt
import math
import numpy as np
import itertools

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import json
from Sim.config.sys_config import ChipletsConfig
from Sim.config.utils import used_bws_dict
import Sim.config.utils as utils
from Sim.entities.chips_network import chip_graph
import Sim.common as common
from Sim.entities.mem_sys import mem_sys
from Sim.entities.comp_sys import comp_sys

from Sim.config.model_config import BaseModelConfig
from Sim.config.sys_config import BaseChipletPlacerConfig
from Sim.entities.static_mapper import static_mapper
from Sim.placer.BasePlacer import BasePlacer
from Sim.entities.mem_chiplet import mem_chiplet
from Sim.entities.mem_chiplet import Params
import tqdm

HW_configs = {
    "NoI_BW": 256,  # GB/s
    "IO_BW": 600, # GB/s
    "Int_Height": 4,
    "Int_Width": 6,
    "Num_Mem": 8,
    "Num_Ap": 8,
    "Num_Ad": 8,
    "Num_Mp": 0,
    "Num_Md": 0,
}

def roofline_throughput_llama(workload_profile):
    # LLAMA3
    BWs = [256, 384, 512, 640]
    batchsizes = [2,4,8,16]
    batchsizes = [2,4] if "arxiv" in workload_profile or "bwb" in workload_profile else batchsizes
    Ms = range(2, 17, 1)
    Num_nodes = 24
    width = 6
    height = 4
    chiplet_area = 144
    sim_time = 100  # seconds
    
    model_name = "llama3-8b"
    model_config = BaseModelConfig.create_from_name(model_name)
    
    with open(workload_profile, 'r') as f:
        profile_data = json.load(f)
    Pareto_configs = {
        'MT': {},
        'Bstar': {},
    }
    Max_tp = 0
    Max_ratio = 0
    
    for bs in tqdm.tqdm(batchsizes, desc="Batch Sizes", position=0):
        for bw in tqdm.tqdm(BWs, desc="Bandwidths", position=1, leave=False):
            for M in tqdm.tqdm(Ms, desc="Memory Counts", position=2, leave=False):
                remain = Num_nodes - M
                Ps = range(1, remain)
                for P in Ps:
                    D = remain - P
                    chiplet_alloc = {
                        "marca_p": 0,
                        "marca_d": 0,
                        "tscs_p": P,
                        "tscs_d": D,
                        "systolicarray_p": 0,
                        "HBM3": M, # 16G
                        "GDDR7": 0, # 16G
                        "HBM3e": 0, # 16G
                        "B200_p": 0,
                        "vuarray_d": 0,
                        "unified_p": 0,
                    }
                    utils.used_bws = used_bws_dict[bw]
                    utils.NoI_bw = bw
                    placmt_config = BaseChipletPlacerConfig(chiplet_alloc=chiplet_alloc)
                    resources_graph: chip_graph = placmt_config.chip_graph
                    plcmt_inst:BasePlacer = placmt_config.placer_inst(chiplet_num=resources_graph.num_nodes, chiplet_alloc=placmt_config.chiplet_alloc)
                    group_HBM_M, group_HBM_A = plcmt_inst.make_chiplet_placement(chip_graph=resources_graph, model_config=model_config, logging=False)
                    mem_sys_inst = mem_sys(resources_graph)
                    mem_sys_inst.load_model(label="bw", mod_config=model_config, 
                               group_HBM_M=group_HBM_M, group_HBM_A=group_HBM_A)
                    comp_sys_inst = comp_sys(resources_graph)
                    mapper = static_mapper()
                    chiplet_config = ChipletsConfig(D2D_NoI_bw=bw, area_limit=chiplet_area, Fixed_chiplet_area=True)
                    common.logic_lib = chiplet_config.chips_lib.logics
                    pf_len = profile_data['prefill_len']
                    dc_len = profile_data['decode_len']
                    # print("P:", P, "D:", D)
                    eff_bw = min(utils.used_bws[1], M * chiplet_config.HBM_IO_bw/(P+D))
                    mapping, lat_dict = mapper.generate_mapping(model_config, comp_sys_inst, mem_sys_inst, bs=bs, pf_c=pf_len, NoI_bw=eff_bw)
                    max_concurrent_requests = 999
                    for mem_chiplet_inst in mem_sys_inst._mem_chiplets:
                        assert isinstance(mem_chiplet_inst, mem_chiplet), "mem_chiplet_inst is not an instance of mem_chiplet."
                        num_blocks = len(mem_chiplet_inst.content_params.get('weights', []))
                        left_budget = (mem_chiplet_inst.dram_budget - mem_chiplet_inst.inuse_budget)*1024*1024  # Convert to Bytes
                        # this is the most conservative estimate, as it assumes all requests will use the max position embeddings (longest context length)
                        mem_req_per_request = 0 # count the memory requirement per request for hybrid blocks
                        for block_params in mem_chiplet_inst.content_params.get('weights', []):
                            assert isinstance(block_params, Params), "block_params is not an instance of Params."
                            type_idx = model_config.block_type_sequence[block_params.block_id]
                            block_config = model_config.hybrid_blocks[type_idx]
                            if block_config.cache_store:
                                mem_req_per_request += (block_config.states*model_config.max_position_embeddings + block_config.peak_intermed)
                            else:
                                mem_req_per_request += (block_config.states + block_config.peak_intermed)
                        
                        max_requests = (left_budget) // mem_req_per_request
                        if max_requests < max_concurrent_requests:
                            max_concurrent_requests = int(max_requests) # set the max 
                    if max_concurrent_requests < bs:
                        continue
                    
                    
                    # convert lat_dict['P'] dictionary to a list
                    p_lat_list = []
                    d_lat_list = []
                    for key in sorted(lat_dict['P'].keys()):
                        p_lat_list.append(lat_dict['P'][key])
                    for key in sorted(lat_dict['D'].keys()):
                        d_lat_list.append(lat_dict['D'][key])
                    
                        
                    
                    T_comp0 = np.max(d_lat_list) + np.max(p_lat_list)*(1/dc_len)
                    comp_tp0 = (1e6 / T_comp0) * bs 
                    comp_tp1 = (1e6/np.max(p_lat_list))*bs*dc_len
                    comp_tp = comp_tp0
                    
                    Total_Mem_IO = M * chiplet_config.HBM_IO_bw
                    mem_tp = Total_Mem_IO / ((profile_data['param_scale']/bs+1-profile_data['param_scale'])*profile_data['mem_access_per_token(GB)'])
                    
                    Total_IO_bw = bw * min((width+height-1)*2,3*M)
                    Tp_noi = Total_IO_bw / ((profile_data['param_scale']/bs+1-profile_data['param_scale'])*profile_data['mem_access_per_token(GB)'])

                    Tp = min(comp_tp, Tp_noi, mem_tp)
                    # Tp = comp_tp
                    TTFT = np.max(p_lat_list)
                    # batchsize,Num M,Num Mp,Num Md,Num Ap,Num Ad,NoI_bw(GBps)
                    configs = {
                        "batchsize": bs,
                        "Num M": M,
                        "Num Mp": 0,
                        "Num Md": 0,
                        "Num Ap": P,
                        "Num Ad": D,
                        "NoI_bw(GBps)": bw,
                    }
                    if Tp > Max_tp:
                        Max_tp = Tp
                        Pareto_configs['MT'] = configs
                    if Tp/TTFT > Max_ratio:
                        Max_ratio = Tp/TTFT
                        Pareto_configs['Bstar'] = configs
    return Pareto_configs
                            
def roofline_throughput_mamba(workload_profile):
    # LLAMA3
    BWs = [256, 384, 512, 640]
    batchsizes = [4,8,16,32]
    batchsizes = [2,4,8,16] if "arxiv" in workload_profile or "bwb" in workload_profile else batchsizes
    Ms = range(2, 17, 1)
    Num_nodes = 24
    width = 6
    height = 4
    chiplet_area = 144
    sim_time = 100  # seconds
    
    model_name = "mamba2-3b"
    model_config = BaseModelConfig.create_from_name(model_name)
    
    with open(workload_profile, 'r') as f:
        profile_data = json.load(f)
    Pareto_configs = {
        'MT': {},
        'Bstar': {},
    }
    Max_tp = 0
    Max_ratio = 0
    
    for bs in tqdm.tqdm(batchsizes, desc="Batch Sizes", position=0):
        for bw in tqdm.tqdm(BWs, desc="Bandwidths", position=1, leave=False):
            for M in tqdm.tqdm(Ms, desc="Memory Counts", position=2, leave=False):
                remain = Num_nodes - M
                Ps = range(1, remain)
                for P in Ps:
                    D = remain - P
                    chiplet_alloc = {
                        "marca_p": P,
                        "marca_d": D,
                        "tscs_p": 0,
                        "tscs_d": 0,
                        "systolicarray_p": 0,
                        "HBM3": M, # 16G
                        "GDDR7": 0, # 16G
                        "HBM3e": 0, # 16G
                        "B200_p": 0,
                        "vuarray_d": 0,
                        "unified_p": 0,
                    }
                    utils.used_bws = used_bws_dict[bw]
                    utils.NoI_bw = bw
                    placmt_config = BaseChipletPlacerConfig(chiplet_alloc=chiplet_alloc)
                    resources_graph: chip_graph = placmt_config.chip_graph
                    plcmt_inst:BasePlacer = placmt_config.placer_inst(chiplet_num=resources_graph.num_nodes, chiplet_alloc=placmt_config.chiplet_alloc)
                    group_HBM_M, group_HBM_A = plcmt_inst.make_chiplet_placement(chip_graph=resources_graph, model_config=model_config, logging=False)
                    mem_sys_inst = mem_sys(resources_graph)
                    mem_sys_inst.load_model(label="bw", mod_config=model_config, 
                               group_HBM_M=group_HBM_M, group_HBM_A=group_HBM_A)
                    comp_sys_inst = comp_sys(resources_graph)
                    mapper = static_mapper()
                    chiplet_config = ChipletsConfig(D2D_NoI_bw=bw, area_limit=chiplet_area, Fixed_chiplet_area=True)
                    common.logic_lib = chiplet_config.chips_lib.logics
                    pf_len = profile_data['prefill_len']
                    dc_len = profile_data['decode_len']
                    # print("P:", P, "D:", D)
                    eff_bw = min(utils.used_bws[1], M * chiplet_config.HBM_IO_bw/(P+D))
                    mapping, lat_dict = mapper.generate_mapping(model_config, comp_sys_inst, mem_sys_inst, bs=bs, pf_c=pf_len, NoI_bw=eff_bw)
                    max_concurrent_requests = 999
                    for mem_chiplet_inst in mem_sys_inst._mem_chiplets:
                        assert isinstance(mem_chiplet_inst, mem_chiplet), "mem_chiplet_inst is not an instance of mem_chiplet."
                        num_blocks = len(mem_chiplet_inst.content_params.get('weights', []))
                        left_budget = (mem_chiplet_inst.dram_budget - mem_chiplet_inst.inuse_budget)*1024*1024  # Convert to Bytes
                        # this is the most conservative estimate, as it assumes all requests will use the max position embeddings (longest context length)
                        mem_req_per_request = 0 # count the memory requirement per request for hybrid blocks
                        for block_params in mem_chiplet_inst.content_params.get('weights', []):
                            assert isinstance(block_params, Params), "block_params is not an instance of Params."
                            type_idx = model_config.block_type_sequence[block_params.block_id]
                            block_config = model_config.hybrid_blocks[type_idx]
                            if block_config.cache_store:
                                mem_req_per_request += (block_config.states*model_config.max_position_embeddings + block_config.peak_intermed)
                            else:
                                mem_req_per_request += (block_config.states + block_config.peak_intermed)
                        
                        max_requests = (left_budget) // mem_req_per_request
                        if max_requests < max_concurrent_requests:
                            max_concurrent_requests = int(max_requests) # set the max 
                    if max_concurrent_requests < bs:
                        continue
                    
                    
                    # convert lat_dict['P'] dictionary to a list
                    p_lat_list = []
                    d_lat_list = []
                    for key in sorted(lat_dict['P'].keys()):
                        p_lat_list.append(lat_dict['P'][key])
                    for key in sorted(lat_dict['D'].keys()):
                        d_lat_list.append(lat_dict['D'][key])
                    
                        
                    
                    T_comp0 = np.max(d_lat_list) + np.max(p_lat_list)*(1/dc_len)
                    comp_tp0 = (1e6 / T_comp0) * bs 
                    comp_tp1 = (1e6/np.max(p_lat_list))*bs*dc_len
                    comp_tp = comp_tp0
                    
                    Total_Mem_IO = M * chiplet_config.HBM_IO_bw
                    mem_tp = Total_Mem_IO / ((profile_data['param_scale']/bs+1-profile_data['param_scale'])*profile_data['mem_access_per_token(GB)'])
                    
                    Total_IO_bw = bw * min((width+height-1)*2,3*M)
                    Tp_noi = Total_IO_bw / ((profile_data['param_scale']/bs+1-profile_data['param_scale'])*profile_data['mem_access_per_token(GB)'])

                    Tp = min(comp_tp, Tp_noi, mem_tp)
                    # Tp = comp_tp
                    TTFT = np.max(p_lat_list)
                    # batchsize,Num M,Num Mp,Num Md,Num Ap,Num Ad,NoI_bw(GBps)
                    configs = {
                        "batchsize": bs,
                        "Num M": M,
                        "Num Mp": P,
                        "Num Md": D,
                        "Num Ap": 0,
                        "Num Ad": 0,
                        "NoI_bw(GBps)": bw,
                    }
                    if Tp > Max_tp:
                        Max_tp = Tp
                        Pareto_configs['MT'] = configs
                    if Tp/TTFT > Max_ratio:
                        Max_ratio = Tp/TTFT
                        Pareto_configs['Bstar'] = configs
    return Pareto_configs

def roofline_throughput_hybrid(workload_profile):

    BWs = [256, 384, 512, 640]
    batchsizes = [4,8,16,32]
    batchsizes = [2,4] if "arxiv" in workload_profile or "bwb" in workload_profile else batchsizes
    Ms = range(2, 17, 2)
    Num_nodes = 24
    width = 6
    height = 4
    chiplet_area = 144
    sim_time = 100  # seconds
    
    model_name = "nemotronh-4b"
    model_config = BaseModelConfig.create_from_name(model_name)
    
    with open(workload_profile, 'r') as f:
        profile_data = json.load(f)
    Pareto_configs = {
        'MT': {},
        'Bstar': {},
    }
    Max_tp = 0
    Max_ratio = 0
    
    for bs in tqdm.tqdm(batchsizes, desc="Batch Sizes", position=0):
        for bw in tqdm.tqdm(BWs, desc="Bandwidths", position=1, leave=False):
            for M in tqdm.tqdm(Ms, desc="Memory Counts", position=2, leave=False):
                remain = Num_nodes - M
                if remain < 4:
                    continue
                valid_combinations = [
                    (Ap, Ad, Mp, remain - Ap - Ad - Mp)
                    for Ap in range(1, remain - 2, 2)
                    for Ad in range(1, remain - Ap - 1, 2)
                    for Mp in range(1, remain - Ap - Ad, 2)
                    if remain - Ap - Ad - Mp >= 1
                ]
                for Ap, Ad, Mp, Md in valid_combinations:
                    chiplet_alloc = {
                        "marca_p": Mp,
                        "marca_d": Md,
                        "tscs_p": Ap,
                        "tscs_d": Ad,
                        "systolicarray_p": 0,
                        "HBM3": M, # 16G
                        "GDDR7": 0, # 16G
                        "HBM3e": 0, # 16G
                        "B200_p": 0,
                        "vuarray_d": 0,
                        "unified_p": 0,
                    }
                    utils.used_bws = used_bws_dict[bw]
                    utils.NoI_bw = bw
                    placmt_config = BaseChipletPlacerConfig(chiplet_alloc=chiplet_alloc)
                    resources_graph: chip_graph = placmt_config.chip_graph
                    plcmt_inst:BasePlacer = placmt_config.placer_inst(chiplet_num=resources_graph.num_nodes, chiplet_alloc=placmt_config.chiplet_alloc)
                    group_HBM_M, group_HBM_A = plcmt_inst.make_chiplet_placement(chip_graph=resources_graph, model_config=model_config, logging=False)
                    mem_sys_inst = mem_sys(resources_graph)
                    mem_sys_inst.load_model(label="bw", mod_config=model_config, 
                            group_HBM_M=group_HBM_M, group_HBM_A=group_HBM_A)
                    comp_sys_inst = comp_sys(resources_graph)
                    mapper = static_mapper()
                    chiplet_config = ChipletsConfig(D2D_NoI_bw=bw, area_limit=chiplet_area, Fixed_chiplet_area=True)
                    common.logic_lib = chiplet_config.chips_lib.logics
                    pf_len = profile_data['prefill_len']
                    dc_len = profile_data['decode_len']
                    # print("P:", P, "D:", D)
                    eff_bw = min(utils.used_bws[1], M * chiplet_config.HBM_IO_bw/(Ap+Ad+Mp+Md))
                    mapping, lat_dict = mapper.generate_mapping(model_config, comp_sys_inst, mem_sys_inst, bs=bs, pf_c=pf_len, NoI_bw=eff_bw)
                    max_concurrent_requests = 999
                    for mem_chiplet_inst in mem_sys_inst._mem_chiplets:
                        assert isinstance(mem_chiplet_inst, mem_chiplet), "mem_chiplet_inst is not an instance of mem_chiplet."
                        num_blocks = len(mem_chiplet_inst.content_params.get('weights', []))
                        left_budget = (mem_chiplet_inst.dram_budget - mem_chiplet_inst.inuse_budget)*1024*1024  # Convert to Bytes
                        # this is the most conservative estimate, as it assumes all requests will use the max position embeddings (longest context length)
                        mem_req_per_request = 0 # count the memory requirement per request for hybrid blocks
                        for block_params in mem_chiplet_inst.content_params.get('weights', []):
                            assert isinstance(block_params, Params), "block_params is not an instance of Params."
                            type_idx = model_config.block_type_sequence[block_params.block_id]
                            block_config = model_config.hybrid_blocks[type_idx]
                            if block_config.cache_store:
                                mem_req_per_request += (block_config.states*model_config.max_position_embeddings + block_config.peak_intermed)
                            else:
                                mem_req_per_request += (block_config.states + block_config.peak_intermed)
                        
                        max_requests = (left_budget) // mem_req_per_request
                        if max_requests < max_concurrent_requests:
                            max_concurrent_requests = int(max_requests) # set the max 
                    if max_concurrent_requests < bs:
                        continue
                    
                    
                    # convert lat_dict['P'] dictionary to a list
                    p_lat_list = []
                    d_lat_list = []
                    for key in sorted(lat_dict['P'].keys()):
                        p_lat_list.append(lat_dict['P'][key])
                    for key in sorted(lat_dict['D'].keys()):
                        d_lat_list.append(lat_dict['D'][key])
                    
                        
                    
                    T_comp0 = np.max(d_lat_list) + np.max(p_lat_list)*(1/dc_len)
                    comp_tp0 = (1e6 / T_comp0) * bs 
                    comp_tp1 = (1e6/np.max(p_lat_list))*bs*dc_len
                    comp_tp = comp_tp0
                    
                    Total_Mem_IO = M * chiplet_config.HBM_IO_bw
                    mem_tp = Total_Mem_IO / ((profile_data['param_scale']/bs+1-profile_data['param_scale'])*profile_data['mem_access_per_token(GB)'])
                    
                    Total_IO_bw = bw * min((width+height-1)*2,3*M)
                    Tp_noi = Total_IO_bw / ((profile_data['param_scale']/bs+1-profile_data['param_scale'])*profile_data['mem_access_per_token(GB)'])

                    Tp = min(comp_tp, Tp_noi, mem_tp)
                    # Tp = comp_tp
                    TTFT = np.max(p_lat_list)
                    # batchsize,Num M,Num Mp,Num Md,Num Ap,Num Ad,NoI_bw(GBps)
                    configs = {
                        "batchsize": bs,
                        "Num M": M,
                        "Num Mp": Mp,
                        "Num Md": Md,
                        "Num Ap": Ap,
                        "Num Ad": Ad,
                        "NoI_bw(GBps)": bw,
                    }
                    if Tp > Max_tp:
                        Max_tp = Tp
                        Pareto_configs['MT'] = configs
                    if Tp/TTFT > Max_ratio:
                        Max_ratio = Tp/TTFT
                        Pareto_configs['Bstar'] = configs
    return Pareto_configs




# This is for Transformer models workloads
# def Estimate_opt_throughput(workload_profile, Ms = None):
#     # read the json file from workload_profile
#     with open(workload_profile, 'r') as f:
#         profile_data = json.load(f)
#     Num_nodes = 24
#     width = 6
#     height = 4
#     chiplet_area = 144
#     # Ms = range(2, 18, 2)  # Number of chiplets outring constraint
#     BWs = [128, 256, 384, 512, 640]
#     dataset_path = "dataset/arxiv/arxiv_summarization_stats_llama3.csv"
#     mem_cap_req = 50
#     batch_size = 1
    
#     util_coeff = 1
#     HBM_coeff = 1
#     NoI_coeff = 0.8
    
#     results = []
    
#     for bw in BWs:
#         # based on NoI Bw, confirm the comp chiplet configs
#         chiplet_config = ChipletsConfig(D2D_NoI_bw=bw, area_limit=chiplet_area, Fixed_chiplet_area=True)
#         for M in Ms:
#             remain = Num_nodes - M
#             Ps = range(1, remain + 1)
#             for P in Ps:
#                 # TODO: compute the four upper bounds
#                 # Here we just get the # requests concurrently by # comp and capacity
#                 Num_Comp = P+M
#                 D = remain - P
#                 num_req_max = min(Num_Comp*batch_size,mem_cap_req)
#                 # 1. Memory IO cap
#                 Total_Mem_IO = M * chiplet_config.HBM_IO_bw * HBM_coeff
#                 # Ap Ops
#                 Ops_Ap = chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['config']['tscs']
#                 Ele_Ap = sqrt(Ops_Ap) # rough estimation
#                 num_core_Ap = chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['core']
#                 Total_Comp_Lin_Ops = P * Ops_Ap * num_core_Ap
#                 Total_Comp_Ele_Ops = P * Ele_Ap * num_core_Ap # assume ele ops is 10% of lin ops
#                 # Ad Ops
#                 Ops_Ad = chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['config']['tscs']
#                 Ele_Ad = sqrt(Ops_Ad)
#                 num_core_Ad = chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['core']
#                 Total_Comp_Lin_Ops += D * Ops_Ad * num_core_Ad
#                 Total_Comp_Ele_Ops += D * Ele_Ad * num_core_Ad
                
#                 # 2. Ops cap
#                 Total_Comp_Lin_Ops = Total_Comp_Lin_Ops * min(1, num_req_max/Num_Comp) * util_coeff
#                 Total_Comp_Ele_Ops = Total_Comp_Ele_Ops * min(1, num_req_max/Num_Comp) * util_coeff
                
#                 # 3. Comp IO cap
#                 Total_Comp_IO_bw = D*used_bws_dict[bw][1] + P*used_bws_dict[bw][1]

#                 # 4. NoI cap
#                 # Assume the HBMs are placed at outring
#                 Total_IO_bw = bw * min((width+height-1)*2, 3*M) * NoI_coeff * min(1, num_req_max/Num_Comp)
                
#                 # 1. TP Bound of memory IO
#                 Tp_memio = Total_Mem_IO / profile_data['mem_access_per_token(GB)']
                
#                 # 2. TP Bound of ops
#                 ratio_lin = profile_data['lin_ops_per_token(G)'] / (profile_data['lin_ops_per_token(G)'] + profile_data['ele_ops_per_token(G)'])
#                 ratio_ele = profile_data['ele_ops_per_token(G)'] / (profile_data['lin_ops_per_token(G)'] + profile_data['ele_ops_per_token(G)'])
#                 if ratio_ele != 0:
#                     Tp_ops = Total_Comp_Lin_Ops * ratio_lin / profile_data['lin_ops_per_token(G)'] + Total_Comp_Ele_Ops * ratio_ele / profile_data['ele_ops_per_token(G)']
#                 else:
#                     Tp_ops = Total_Comp_Lin_Ops / profile_data['lin_ops_per_token(G)']
                    
#                 # 3. TP Bound of Comp IO
#                 Tp_compio = Total_Comp_IO_bw / profile_data['mem_access_per_token(GB)']
#                 # 4. TP Bound of NoI
#                 Tp_noi = Total_IO_bw / profile_data['mem_access_per_token(GB)']
                
#                 total_TP = min(Tp_memio, Tp_ops, Tp_compio, Tp_noi)
#                 # print(f"BW:{bw} M:{M} P:{P} D:{D} TP:{total_TP:.2f} Tok/s, MemIO:{Tp_memio:.2f}, Ops:{Tp_ops:.2f}, CompIO:{Tp_compio:.2f}, NoI:{Tp_noi:.2f}")
#                 results.append({
#                     "bw": bw,
#                     "M": M,
#                     "P": P,
#                     "D": D,
#                     "Tp_memio": Tp_memio,
#                     "Tp_ops": Tp_ops,
#                     "Tp_compio": Tp_compio,
#                     "Tp_noi": Tp_noi,
#                     "total_TP": total_TP
#                 })
#     return results

def system_rhs(t, x, params):
    # Unpack parameters
    a, b, c = params
    # Define the system of equations
    dxdt = a * x + b * t + c
    return dxdt

def f_beta(x, C, k=5):
    return 0.5 + 0.5 * (1 - math.exp(-k * x / C)) / (1 - math.exp(-k))

def Estimate_opt_throughput_markov(workload_profile, Ms = None):
    with open(workload_profile, 'r') as f:
        profile_data = json.load(f)
    Num_nodes = 24
    width = 6
    height = 4
    chiplet_area = 144
    # Ms = range(2, 18, 2)  # Number of chiplets outring constraint
    BWs = [128, 256, 384, 512, 640]
    mem_cap_req = 50
    batch_size = 1
    Sim_time = 100 # seconds
    results = []
    for bw in BWs:
        # based on NoI Bw, confirm the comp chiplet configs
        chiplet_config = ChipletsConfig(D2D_NoI_bw=bw, area_limit=chiplet_area, Fixed_chiplet_area=True)
        Chip_graph = chip_graph(Num_nodes, width, height, np.zeros((width, height)), True, None)
        for M in Ms:
            remain = Num_nodes - M
            Ps = range(1, remain + 1)
            for P in Ps:
                D = remain - P
                print(f"Estimating BW:{bw} M:{M} P:{P} D:{D}")
                c_time = 0
                state = "Pure_P"
                chiplet_alloc = {
                    "marca_p": 0,
                    "marca_d": 0,
                    "tscs_p": P,
                    "tscs_d": D,
                    "HBM3": M, # 16G
                }
                plcmt_inst = common.chiplet_plcmt(chiplet_num=Num_nodes, chiplet_alloc=chiplet_alloc)
                plcmt_inst.init_plcmt(Chip_graph, output_folder="")
                Mem_sys = mem_sys(Chip_graph)
                Comp_sys = comp_sys(Chip_graph)
                # Bw_budget = min(M*utils.IO_bw*2/3, (width+height-2)*2*utils.NoI_bw, 2*M*utils.NoI_bw)
                # interference factor (1.5 ~ 2.0 means 30–50% loss due to contention)
                num_comp = Num_nodes - M
                # global mesh bisection bound
                mesh_bw_bound = (width + height - 2) * 2 * utils.NoI_bw
                beta = f_beta(M, Num_nodes)
                noi_bw_eff = beta * mesh_bw_bound

                Bw_budget = min(
                    M * utils.IO_bw*beta,
                    noi_bw_eff,
                    M * utils.NoI_bw*beta
                )
                
                N_batch = min(Comp_sys._total_budget, Mem_sys._num_io_limit)
                prefill_ops = int(profile_data['prefill lin_ops(G)']/profile_data['prefill_len']) # per token
                decode_bw = profile_data['mem_access_per_token(GB)']
                prefill_tokens = [int(profile_data['prefill_len']) for _ in range(N_batch)]
                decode_tokens = [int(profile_data['decode_len']) for _ in range(N_batch)]
                
                dec_ops = int(profile_data['lin_ops_per_token(G)']) # per token
                prefill_bw = int(profile_data['mem_access_per_token(GB)'] / profile_data['prefill_len']) # per token

                Total_tokens = 0
                
                while c_time < Sim_time:
                    if state == "Pure_P":
                        pf_rateP = (chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['config']['tscs']
                                * chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['core'] / prefill_ops)
                        pf_rateD = (chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['config']['tscs']
                                * chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['core'] / prefill_ops)
                        assigned_P = min(P, N_batch)
                        # prefill_tokens -= assigned_P * profile_data['prefill_len']
                        elapse_time = np.inf
                        # from low to high
                        prefill_tokens = sorted(prefill_tokens, reverse=False)
                        if any (item <= 0 for item in prefill_tokens):
                            raise ValueError("prefill tokens < 0 in Pure_P")
                        for index,item in enumerate(prefill_tokens):
                            if item <= 0:
                                prev_state = state
                                state = "Mix_PD"
                                if decode_tokens[index] == 0:
                                    decode_tokens[index] = int(profile_data['decode_len'])
                                    prefill_tokens[index] = int(profile_data['prefill_len'])
                                elapse_time = 0
                                break
                            else:
                                if index < assigned_P:
                                    time0 = item / pf_rateP
                                    elapse_time = min(elapse_time, time0)
                                else:
                                    time1 = item / pf_rateD
                                    elapse_time = min(elapse_time, time1)
                        if elapse_time == np.inf:
                            raise ValueError("elapse_time is inf")
                        if state == "Mix_PD":
                            print("Accidentally Switching to Mix_PD")
                            continue
                        for index,item in enumerate(prefill_tokens):
                            if index < assigned_P:
                                prefill_tokens[index] -= elapse_time * pf_rateP
                            else:
                                prefill_tokens[index] -= elapse_time * pf_rateD
                        prev_state = state
                        state = "Mix_PD"
                        c_time += elapse_time
                        
                        prefill_tokens = [0 if item < 0 else item for item in prefill_tokens]
                        
                    elif state == "Mix_PD":
                        prefill_zeros = sum([1 for item in prefill_tokens if item == 0])
                        if prefill_zeros == 0:
                            state = "Pure_P"
                            continue
                        if prefill_zeros == N_batch:
                            state = "Pure_D"
                            continue
                        init_P = N_batch - prefill_zeros # The number of prefilling tasks beginning in mix state
                        assigned_P = min(P, init_P) # assigned P chiplet for prefilling at beginning
                        
                        init_D = min(prefill_zeros, int(Bw_budget//utils.used_bws[1])) # The number of decoding tasks beginning in mix state
                        assigned_D = min(D, init_D) # assigned D chiplet for decoding at beginning
                        # Bw_budget -= (N_batch-prefill_zeros) * utils.used_bws[0]
                        prefill_tokens = sorted(prefill_tokens, reverse=False)
                        decode_tokens = sorted(decode_tokens, reverse=False)
                        
                        """
                        Two conditions:
                            Cond1: Mix_PD -> Mix_PD (Some decoding finished, while some prefilling still not finished)
                            Cond2: Mix_PD -> Pure_D (All prefilling finished, all decoding not finished)
                        """

                        Pf_rateP = (chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['config']['tscs']
                                * chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['core'] / prefill_ops)
                        Pf_rateD = (chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['config']['tscs']
                                * chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['core'] / prefill_ops)
                        bw_p = max(utils.used_bws[1]/2, utils.used_bws[0])
                        Dec_rateP = min((bw_p/decode_bw), chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['config']['tscs']
                                * chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['core'] / dec_ops)
                        Dec_rateD = min((utils.used_bws[1]/decode_bw), chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['config']['tscs']
                                * chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['core'] / dec_ops)
                        
                        decode_start_times = {}
                        elapse_time = 0
                        while True:
                            time_pf_first = 0
                            for item in prefill_tokens:
                                if item != 0:
                                    time_pf_first = item / Pf_rateP if assigned_P > 0 else item / Pf_rateD
                                    break
                            time_decoding = decode_tokens[0] / (utils.used_bws[1]/decode_bw) if assigned_D > 0 else decode_tokens[0] / (2*utils.used_bws[1]/decode_bw)
                            if time_decoding == 0:
                                raise ValueError("decoding time is zero in Mix_PD")
                            
                            # Condition 2: Mix_PD -> Pure_D potentially
                            if  time_pf_first < time_decoding:
                                P_index = 0
                                time_int = 0
                                for index,item in enumerate(prefill_tokens):
                                    if item == 0:
                                        decode_start_times[index] = 0
                                        continue
                                    else:
                                        if P_index < assigned_P:
                                            if time_int == 0:
                                                prefill_tokens[index] = 0
                                                time_int = item / Pf_rateP
                                            else:
                                                prefill_tokens[index] -= time_int * Pf_rateP
                                                if prefill_tokens[index] <= -1:
                                                    raise ValueError("Prefilling tokens < 0 in conditions")
                                                prefill_tokens[index] = check_zeros_bounds(prefill_tokens[index])
                                            P_index += 1
                                        else:
                                            if time_int == 0:
                                                prefill_tokens[index] = 0
                                                time_int = item / Pf_rateD
                                            else:
                                                prefill_tokens[index] -= time_int * Pf_rateD
                                                if prefill_tokens[index] <= -1:
                                                    raise ValueError("Prefilling tokens < 0 in conditions")   
                                                prefill_tokens[index] = check_zeros_bounds(prefill_tokens[index])     
                                if time_int != time_pf_first:
                                    raise ValueError("time_int not equal to time_pf_first")
                                temp_budget = Bw_budget
                                num_still_pf = N_batch - len(decode_start_times.values())
                                temp_budget -= num_still_pf * utils.used_bws[0]
                                final_D = 0 # resource constraints for decoding tasks
                                for i in range(P+D-num_still_pf):
                                    if i < D:
                                        temp_budget -= utils.used_bws[1]
                                    else:
                                        temp_budget -= utils.used_bws[1]/2
                                    if temp_budget < 0:
                                        break
                                    final_D += 1
                                for i in range(len(decode_start_times.values())):
                                    if i < final_D:
                                        if i < assigned_D:
                                            decode_tokens[i] -= time_int * Dec_rateD
                                            Total_tokens += time_int * Dec_rateD
                                            decode_tokens[i] = check_zeros_bounds(decode_tokens[i])
                                        else:
                                            decode_tokens[i] -= time_int * Dec_rateP
                                            Total_tokens += time_int * Dec_rateP
                                            decode_tokens[i] = check_zeros_bounds(decode_tokens[i])
                                    else:
                                        break
                                elapse_time += time_int
                                if any(item <= 0 for item in decode_tokens):
                                    if check_zeros_bounds(time_decoding-time_int) == 0:
                                        state = "Mix_PD"
                                        break
                                    raise ValueError("Some decoding finished in condition 1 early")
                                if all(item == 0 for item in prefill_tokens):
                                    prev_state = state
                                    state = "Pure_D"
                                    break
                            # Condition 1: Mix_PD -> Mix_PD Must
                            else:
                                temp_budget = Bw_budget
                                num_still_pf = N_batch - prefill_zeros
                                temp_budget -= num_still_pf * utils.used_bws[0]
                                final_D = 0 # resource constraints for decoding tasks
                                for i in range(P+D-num_still_pf):
                                    if i < D:
                                        temp_budget -= utils.used_bws[1]
                                    else:
                                        temp_budget -= utils.used_bws[0]
                                    if temp_budget < 0:
                                        break
                                    final_D += 1
                                for i in range(prefill_zeros):
                                    if i < final_D:
                                        if i < D:
                                            decode_tokens[i] -= time_decoding * Dec_rateD
                                            Total_tokens += time_decoding * Dec_rateD
                                            decode_tokens[i] = check_zeros_bounds(decode_tokens[i])
                                        else:
                                            decode_tokens[i] -= time_decoding * Dec_rateP
                                            Total_tokens += time_decoding * Dec_rateP
                                            decode_tokens[i] = check_zeros_bounds(decode_tokens[i])
                                    else:
                                        break
                                if not any(item <= 0 for item in decode_tokens):
                                    raise ValueError("No decoding finished in condition 1")
                                decode_tokens = [0 if item < 0 else item for item in decode_tokens]
                                P_index = 0
                                for index,item in enumerate(prefill_tokens):
                                    if item == 0:
                                        decode_start_times[index] = 0
                                        continue
                                    else:
                                        if P_index < assigned_P:
                                            prefill_tokens[index] -= time_decoding * Pf_rateP
                                            prefill_tokens[index] = check_zeros_bounds(prefill_tokens[index])
                                            P_index += 1
                                        else:
                                            prefill_tokens[index] -= time_decoding * Pf_rateD
                                            prefill_tokens[index] = check_zeros_bounds(prefill_tokens[index])
                                elapse_time += time_decoding
                                prev_state = state
                                state = "Mix_PD"
                                if all(item == 0 for item in prefill_tokens):
                                    if check_zeros_bounds(time_pf_first-time_decoding) != 0:
                                        raise ValueError("All prefill tokens are zero in Mix_PD")  
                                break
                        c_time += elapse_time
                        for index in range(len(decode_tokens)):
                            # fill in the request when both prefill and decode are done
                            if decode_tokens[index] == 0 and prefill_tokens[index] == 0:
                                if state == "Pure_D":
                                    raise ValueError("No prefill tokens available in Pure_D")
                                decode_tokens[index] = int(profile_data['decode_len'])
                                prefill_tokens[index] = int(profile_data['prefill_len'])
                        if all(item > 0 for item in prefill_tokens):
                            prev_state = state
                            state = "Pure_P"
                        elif all(item == 0 for item in prefill_tokens):
                            if any(item <= 0 for item in decode_tokens):
                                raise ValueError("Some decoding finished in Mix_PD")
                            prev_state = state
                            state = "Pure_D"
                                
                                
                    elif state == "Pure_D":
                        
                        Dec_rateD = min(utils.used_bws[1]/decode_bw, chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['config']['tscs']
                                * chiplet_config.chips_lib.comp_lib['tscs_d'][bw]['core'] / dec_ops)
                        bw_p = max(utils.used_bws[1]/2, utils.used_bws[0])
                        Dec_rateP = min((bw_p/decode_bw), chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['config']['tscs']
                                * chiplet_config.chips_lib.comp_lib['tscs_p'][bw]['core'] / dec_ops)
                        
                        temp_budget = Bw_budget
                        final_D = 0 # resource constraints for decoding tasks
                        for i in range(P+D):
                            if i < D:
                                temp_budget -= utils.used_bws[1]
                            else:
                                temp_budget -= utils.used_bws[0]
                            if temp_budget < 0:
                                break
                            final_D += 1
                        assigned_D = min(D, final_D)
                        assigned_P = final_D - assigned_D
                        elapse_time = np.inf
                        decode_tokens = sorted(decode_tokens, reverse=False)
                        for index,item in enumerate(decode_tokens):
                            if index >= final_D:
                                break
                            if item == 0:
                                raise ValueError("No prefill tokens available in Pure_D") #?
                            else:
                                if index < assigned_D:
                                    time0 = item * decode_bw / utils.used_bws[1]
                                    elapse_time = min(elapse_time, time0)
                                else:
                                    time1 = item * decode_bw / (utils.used_bws[1]/2)
                                    elapse_time = min(elapse_time, time1)
                        if elapse_time == np.inf:
                            raise ValueError("elapse_time is inf")

                        for index,item in enumerate(decode_tokens):
                            if index < final_D:
                                if index < D:
                                    decode_tokens[index] -= elapse_time * Dec_rateD
                                    Total_tokens += elapse_time * Dec_rateD
                                    decode_tokens[index] = check_zeros_bounds(decode_tokens[index])
                                else:
                                    decode_tokens[index] -= elapse_time * Dec_rateP
                                    Total_tokens += elapse_time * Dec_rateP
                                    decode_tokens[index] = check_zeros_bounds(decode_tokens[index])
                        for index in range(len(decode_tokens)):
                            if decode_tokens[index] < 0:
                                decode_tokens[index] = 0
                            if decode_tokens[index] == 0 and prefill_tokens[index] == 0:
                                decode_tokens[index] = int(profile_data['decode_len'])
                                prefill_tokens[index] = int(profile_data['prefill_len'])
                        prev_state = state
                        state = "Mix_PD"
                        c_time += elapse_time
                results.append({
                    "bw": bw,
                    "M": M,
                    "P": P,
                    "D": D,
                    "total_TP": 1.5*Total_tokens / c_time
                })
                if c_time < Sim_time:
                    raise ValueError("c_time < Sim_time, need longer sim time")
    return results

def check_zeros_bounds(input, bound=1e-8):
    if abs(input) < bound: 
        return 0
    else:
        return input

if __name__=="__main__":
    """
        profile_path = "Fast_Estimate/workloads_profiles/llama3-8b_arxiv4k.json"
        # profile_path = "Fast_Estimate/workloads_profiles/llama3-8b_bwb.json"
        results = Estimate_opt_throughput2(profile_path, Ms = range(2, 18, 2))
        # results = Estimate_opt_throughput2(profile_path, Ms = range(2, 17, 1))
        # results = Estimate_opt_throughput(profile_path, Ms = range(2, 17, 1))
        results_sorted = sorted(results, key=lambda x: x["total_TP"], reverse=True)
        # keep top 10%
        percentile = 1
        results_filtered = results_sorted[:max(1, int(len(results_sorted)*percentile))]

        workloads_name = profile_path.split("/")[-1].split(".")[0]
        # for res in results_sorted[:30]:
        #     print(f"BW:{res['bw']} M:{res['M']} P:{res['P']} D:{res['D']} TP:{res['total_TP']:.2f} Tok/s, MemIO:{res['Tp_memio']:.2f}, Ops:{res['Tp_ops']:.2f}, CompIO:{res['Tp_compio']:.2f}, NoI:{res['Tp_noi']:.2f}")
        output_path = f"Fast_Estimate/filtered_configs/throughput_bounds_{workloads_name}.json"
        if percentile == 1:
            output_path = f"Fast_Estimate/filtered_configs/throughput_bounds_full_{workloads_name}.json"
        with open(output_path, "w") as f:
            json.dump(results_filtered, f, indent=4)
    """
    data_sets = [
        # "arxiv",
        "bwb",
        # "chat",
        # "lw"   
    ]
    
    models = [
        "llama3",
        "mamba2",
        "nemo"
    ]
    #data_set = "arxiv"
    model = "nemo"
    for data_set in data_sets:
        profile_path = f"Fast_Estimate/workloads_profiles/{model}_{data_set}.json"
        #best_configs = roofline_throughput_llama(profile_path)
        #best_configs = roofline_throughput_mamba(profile_path)
        best_configs = roofline_throughput_hybrid(profile_path)
        print("Best MT config:", best_configs['MT'])    
        print("Best B* config:", best_configs['Bstar'])
        # load the dict to a json file.
        file_name = f"Fast_Estimate/filtered_configs/roofline_throughput_{model}_{data_set}.json"
        with open(file_name, "w") as f:
            json.dump(best_configs, f, indent=4)