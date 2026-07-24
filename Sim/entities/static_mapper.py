import numpy as np
import math
import networkx as nx
from Sim.entities.base_entity import BaseEntity
from Sim.config.model_config import BaseModelConfig, BaseBlockConfig, baselayerConfig
from Sim.entities.comp_sys import comp_sys
from Sim.entities.mem_sys import mem_sys
from Sim.entities.comp_chiplet import comp_chiplet
import Sim.config.utils as utils
from analytic_profile.Base_Accmodel import BaseAccModel
import Sim.common as common
from Sim.metrics.monitor import analytics

class static_mapper(BaseEntity):
    """
    Represents a mapping for a llm model into chiplets
    """

    def generate_mapping(self, model_config:BaseModelConfig, comp_sys:comp_sys, mem_sys:mem_sys, bs = 32, pf_c = 32, NoI_bw = None):
        '''
        Generate a static mapping for the model into the chip graph.
        This is a simple round-robin mapping for demonstration purposes.
        '''
        lat_dict = {'P':{}, 'D':{}} # profile the comp latencies
        for stage_type in lat_dict.keys():
            for blk_idx, block_type in enumerate(model_config.hybrid_blocks):
                # 'transformer' in block_type.type_name:
                assert isinstance(block_type, BaseBlockConfig), "block_type must be an instance of BaseBlockConfig"
                if 'transformer' in block_type.type_name:
                    acc_list = utils.T_P if stage_type == 'P' else utils.T_D
                elif 'mamba' in block_type.type_name:
                    acc_list = utils.M_P if stage_type == 'P' else utils.M_D
                else:
                    raise NotImplementedError(f"Block type {block_type.type_name} not supported in static_mapper")
                
                selected_accs = set()
                # look through the comp_sys to find the acc chiplets
                for acc in acc_list:
                    for comp_chip in comp_sys._comp_chiplets:
                        assert isinstance(comp_chip, comp_chiplet), "comp_chiplet must be an instance of comp_chiplet"
                        if comp_chip.chiplet_type == acc:
                            selected_accs.add(acc)
                # impl the profile
                for acc in selected_accs:
                    logic_name = utils.chiplet_types_list[acc]
                    analytic_profiles:BaseAccModel = BaseAccModel.create_from_name(logic_name)
                    total_latency = 0
                    for layer_config, layer_name in zip(block_type.layer_configs, block_type.layers):
                        assert len(layer_config) == len(layer_name), "layer_config and layer_name must have the same length"
                        for layer_idx, layer_c in enumerate(layer_config):
                            name = layer_name[layer_idx]
                            if name in ['MHA','SSM']: # prefill and decode mode
                                if stage_type == 'P':
                                    name += '_p'
                                else:
                                    name += '_d'
                            assert isinstance(layer_c, baselayerConfig), "layer_c must be a baselayerConfig instance"
                            func_args = analytic_profiles.config_params(**vars(layer_c))
                            batch_size = 1
                            if stage_type == 'P':
                                func_args['bs'] = pf_c 
                                func_args['batch_size'] = bs
                                func_args['L_seq'] = pf_c
                                if 'in_size' in func_args:
                                    func_args['in_size'] *= pf_c
                            else:
                                func_args['bs'] = 1
                                func_args['batch_size'] = bs
                                func_args['L_seq'] = pf_c
                            func_args['logic_name'] = logic_name
                            func_args['ext_bw'] = utils.NoI_bw if NoI_bw is None else NoI_bw
                            analytic_res:analytics = analytic_profiles.get_kernel(name, **func_args)
                            total_latency += common.convert_analy_time(analytic_res.total_latency)
                    lat_dict[stage_type][blk_idx] = lat_dict[stage_type].get(blk_idx, {})
                    lat_dict[stage_type][blk_idx][acc] = total_latency
        
        # After profiling, generate the mapping of chiplets in the comp_sys
        mapping = {'P':{}, 'D':{}}
        # The initial T are all zero, opt target: minimize the variance of latencies across chiplets
        mapping_latencies = {'P':{}, 'D':{}}
        for comp_chip in comp_sys._comp_chiplets:
            if comp_chip.chiplet_type in utils.T_P + utils.M_P:
                mapping_latencies['P'][comp_chip.chiplet_id] = 0 
            elif comp_chip.chiplet_type in utils.T_D + utils.M_D:
                mapping_latencies['D'][comp_chip.chiplet_id] = 0
            else:
                raise ValueError(f"Unknown chiplet type: {comp_chip.chiplet_type}")
        
        # A greedy assignment to minimize variance
        for stage_type in mapping.keys():
            for blk_no, blk_idx in enumerate(model_config.block_type_sequence):
                # search the comp_sys to map the task
                avail_chiplets = set()
                for comp_chip in comp_sys._comp_chiplets:
                    # constrain the selectiion
                    try:
                        if comp_chip.chiplet_type in lat_dict[stage_type][blk_idx]:      
                            avail_chiplets.add(comp_chip.chiplet_id)
                    except KeyError:
                        continue
                
                best_chiplet_id = None
                best_variance = float('inf')
                best_chiplet_type = None
                for chiplet_id in avail_chiplets:
                    # find the best chiplet to map that minimize the variance in mapping_latencies['P' or 'D']
                    # Find the chiplet object and its type
                    chiplet_obj = comp_sys.find_chiplet_by_id(chiplet_id)
                    if chiplet_obj is None:
                        raise ValueError(f"Chiplet with ID {chiplet_id} not found in comp_sys.")
                    # Get the latency for this chiplet type
                    chiplet_type = chiplet_obj.chiplet_type
                    if chiplet_type not in lat_dict[stage_type][blk_idx]:
                        raise ValueError(f"Latency data not found for chiplet type {chiplet_type} in stage {stage_type}, block {blk_idx}")
                    
                    block_latency = lat_dict[stage_type][blk_idx][chiplet_type]
                    
                    # Simulate assignment: compute variance after adding this block
                    temp_latencies = mapping_latencies[stage_type].copy()
                    temp_latencies[chiplet_id] += block_latency
                    
                    # Calculate variance across all chiplets in this stage
                    latency_values = np.array(list(temp_latencies.values()))
                    variance = np.var(latency_values)
                    
                    # Choose chiplet that minimizes variance (and break ties by lowest current load)
                    if variance < best_variance or (variance == best_variance and temp_latencies[chiplet_id] < mapping_latencies[stage_type][best_chiplet_id]):
                        best_variance = variance
                        best_chiplet_id = chiplet_id
                        best_chiplet_type = chiplet_type
                
                # Assign this block to the best chiplet
                if best_chiplet_id is not None:
                    mapping[stage_type][blk_no] = mapping[stage_type].get(blk_no, None)
                    mapping[stage_type][blk_no] = best_chiplet_id
                    mapping_latencies[stage_type][best_chiplet_id] += lat_dict[stage_type][blk_idx][best_chiplet_type]
                else:
                    raise ValueError(f"No valid chiplet found for block {blk_idx} in stage {stage_type}")
        
        mapping_explicit = mapping.copy()
        for stage_type in mapping.keys():
            # revise the keys from id to name
            mapping_explicit[stage_type] = {str(model_config.hybrid_blocks[model_config.block_type_sequence[blk_no]]).split('(')[0]+str(blk_no): chiplet_id for blk_no, chiplet_id in mapping[stage_type].items()}
            # revise the values from id to name
            mapping_explicit[stage_type] = {blk_no: utils.chiplet_types_list[comp_sys.find_chiplet_by_id(chiplet_id).chiplet_type]+str(chiplet_id) for blk_no, chiplet_id in mapping_explicit[stage_type].items()}
        return mapping, mapping_latencies