from Sim.entities.chips_network import chip_graph
from Sim.entities.mem_chiplet import mem_chiplet, Params, intermediate
from Sim.config.utils import chiplet_types_list
from Sim.config.model_config import BaseModelConfig, BaseBlockConfig
import Sim.common as common
from Sim.metrics.monitor import mem_monitor
from Sim.config import utils
from Sim.placer.BasePlacer import BasePlacer

import networkx as nx
import random
import os

class mem_sys:
    def __init__(self, chip_graph: chip_graph):
        self.graph: nx.Graph = chip_graph.graph
        self._total_budget: float = 0 # in MB
        self._current_used: float = 0 # in MB
        # self._current_allocated: float = 0 # in MB
        self._mem_chiplets = []
        for coord, attr in self.graph.nodes(data=True):
            if attr['dram(g)'] != 0:
                if 'HBM' not in chiplet_types_list[attr['chiplet_type']] and \
                   'GDDR' not in chiplet_types_list[attr['chiplet_type']]:
                    raise ValueError(f"Non-HBM/GDDR chiplet {attr['chiplet_type']} found with DRAM budget.")
                self._total_budget += attr['dram(g)'] * 1024  # convert to MB
                self._mem_chiplets.append(mem_chiplet(chiplet_type=attr['chiplet_type'],
                                                      dram_budget=attr['dram(g)'],
                                                      chiplet_id=attr['id'],
                                                      chiplet_loc=coord if isinstance(coord, tuple) else attr['coord'][0],
                                                      BW_budget=attr['bandwidth']))
        self.total_mem_chiplets = len(self._mem_chiplets)
        self.total_mem_bw = sum([chiplet.BW_budget for chiplet in self._mem_chiplets])
        self.blocks_alloc = {}
        used_bw = max(utils.used_bws[0], utils.used_bws[1]/2)
        self._num_io_limit = min(int(len(self._mem_chiplets)*utils.IO_bw/(used_bw)),
                            int(len(self._mem_chiplets)*3*utils.NoI_bw/(used_bw))) # limit of concurrent IOs based on memory bandwidth and NoI bandwidth
        mem_monitor.init_monitor(self._mem_chiplets)
        
        # If one have any new model loading strategies accompany with placer, 
        # add to the map here with corresponding function
        self.model_loader_map = {
            "bw": self.load_model_bw,
            "rr": self.load_model_rr,
            "random": self.load_model_random,
            "tp": self.load_model_tp, # for tensor-parallelism aware loading
        }

    @staticmethod
    def get_global_index_map(sequence: list[int], hybrid_blocks: list[BaseBlockConfig]) -> dict[str, list[int]]:
        """Maps block type ID (0 or 1) to a list of global layer indices."""
        global_index_map = {'mamba': [], 'transformer': []}
        for global_idx, block_type_id in enumerate(sequence):
            # We assume block_type_id is 0 or 1 here
            for key in global_index_map.keys():
                if key in hybrid_blocks[block_type_id].type_name.lower():
                    global_index_map[key].append(global_idx)
                    break
            # Handle cases where hybrid_blocks might have more than two types if necessary
        return global_index_map
    
    def load_model(self, label: str, mod_config: BaseModelConfig, group_HBM_M: list, group_HBM_A: list):
        # 1. Get the function from the map, defaulting to self.load_model_rr
        #    if the input 'label' is not a valid key.
        load_func = self.model_loader_map.get(label, self.load_model_rr)
        # 2. Call the selected function
        return load_func(mod_config, group_HBM_M, group_HBM_A)
    
    """
        load a model weights into the memory system according to grouping of mem chiplets
    """
    def load_model_bw(self, mod_config:BaseModelConfig, group_HBM_M: list, group_HBM_A: list) -> None:
        """
        Load a model weights into the memory system (by default uniformly).
        """
        # if mod_config.hybrid == False:
        global_index_map = mem_sys.get_global_index_map(mod_config.block_type_sequence, mod_config.hybrid_blocks)
        group_keys = list(global_index_map.keys())
        for group_idx, group in enumerate([group_HBM_M, group_HBM_A]): # Just two groups for Mamba(0) and Attention(1)
            if len(group) == 0 and mod_config.hybrid == True:
                raise ValueError("No memory chiplets assigned for one of the block groups in hybrid model.")
            if len(group) == 0:
                continue
            
            # confirm the block types in the mamba/attention groups N types => N k-v in the dict containing layer index
            blocks_dict = {}
            for idx in global_index_map[group_keys[group_idx]]:
                if mod_config.block_type_sequence[idx] not in blocks_dict:
                    blocks_dict[mod_config.block_type_sequence[idx]] = [idx]
                else:
                    blocks_dict[mod_config.block_type_sequence[idx]].append(idx)
            
            # load each block type
            for key, val in blocks_dict.items():  
                blk_config:BaseBlockConfig = mod_config.hybrid_blocks[key]
                params_per_layer = blk_config.parameter_count
                mem_req_per_layer = params_per_layer * common.ByteperParam / (1024 * 1024)  # convert to MB
                # random.shuffle(chiplets)
                num_layers = len(val)
                num_chiplets = len(group)
                base_layers_per_chiplet = num_layers // num_chiplets
                remainder_layers = num_layers % num_chiplets
                layer_offset = 0
                
                for i, chiplet_id in enumerate(group):
                    chiplet = self.get_mem_byid(chiplet_id)
                    layers_to_assign = base_layers_per_chiplet
                    if i < remainder_layers:
                        layers_to_assign += 1 # Distribute remainder layers (e.g., layers 12, 13 for 14/3 case)
                    for layer_block_i in range(layers_to_assign):
                        layer_index = layer_offset + layer_block_i
                        global_layer_index = val[layer_index]
                        chiplet.inuse_budget += mem_req_per_layer
                        chiplet._current_allocated += mem_req_per_layer
                        chiplet.content_params['weights'] = [] if 'weights' not in chiplet.content_params else chiplet.content_params['weights']
                        chiplet.content_params['weights'].append(
                            Params(size=mem_req_per_layer, block_id=global_layer_index, 
                                name=f"Block {global_layer_index} weights", chip_id=chiplet.chiplet_id)
                        )
                        self._current_used += mem_req_per_layer
                        # self._current_allocated += mem_req_per_layer
                        self.blocks_alloc[global_layer_index] = chiplet.chiplet_id
                    # Update the starting point for the next chiplet.
                    layer_offset += layers_to_assign
                # Sanity check: ensure all layers were assigned
                if layer_offset != num_layers:
                    raise ValueError(f"Error: Assigned {layer_offset} layers but expected {num_layers}.")
        if self._current_used > self._total_budget:
            raise MemoryError("Current memory usage exceeds total budget.")
        sorted_items = sorted(self.blocks_alloc.items())
        self.blocks_alloc = dict(sorted_items)
        mem_monitor.update_utilization(self._mem_chiplets, self._current_used, 0)
    
    """
        load a model weights into the memory system in round-robin fashion
    """
    def load_model_rr(self, mod_config:BaseModelConfig, group_HBM_M: list, group_HBM_A: list) -> None:
        """
        Load a model weights into the memory system (by default uniformly).
        """
        for idx, block_type_idx in enumerate(mod_config.block_type_sequence):
            blk_config:BaseBlockConfig = mod_config.hybrid_blocks[block_type_idx] if mod_config.hybrid else mod_config.block_config
            params_per_layer = blk_config.parameter_count
            mem_req_per_layer = params_per_layer * common.ByteperParam / (1024 * 1024)  # convert to MB
            chiplets: list[mem_chiplet] = self._mem_chiplets[:]
            num_chiplets = len(chiplets)
            chiplet = chiplets[idx % num_chiplets]
            if chiplet.dram_budget - chiplet.inuse_budget >= mem_req_per_layer:
                chiplet.inuse_budget += mem_req_per_layer
                chiplet._current_allocated += mem_req_per_layer
                chiplet.content_params['weights'] = [] if 'weights' not in chiplet.content_params else chiplet.content_params['weights']
                chiplet.content_params['weights'].append(Params(size=mem_req_per_layer, block_id=idx, name=f"Block {idx} weights", chip_id=chiplet.chiplet_id))
                self._current_used += mem_req_per_layer
                # self._current_allocated += mem_req_per_layer
                self.blocks_alloc[idx] = chiplet.chiplet_id
            else:
                raise ValueError("Not enough memory to load the model.")

        if self._current_used > self._total_budget:
            raise MemoryError("Current memory usage exceeds total budget.")
        
        mem_monitor.update_utilization(self._mem_chiplets, self._current_used, 0)

    """
        load model weights into randomly selected memory chiplets
    """
    def load_model_random(self, mod_config:BaseModelConfig, group_HBM_M: list, group_HBM_A: list) -> None:
        """
        Load model weights into HBM chiplets using a deterministic random order.
        """
        seed = int(os.environ.get("HYDRA_RANDOM_MEM_ALLOC_SEED", "20260202"))
        rng = random.Random(seed)

        for idx, block_type_idx in enumerate(mod_config.block_type_sequence):
            blk_config:BaseBlockConfig = mod_config.hybrid_blocks[block_type_idx] if mod_config.hybrid else mod_config.block_config
            params_per_layer = blk_config.parameter_count
            mem_req_per_layer = params_per_layer * common.ByteperParam / (1024 * 1024)

            chiplets: list[mem_chiplet] = self._mem_chiplets[:]
            rng.shuffle(chiplets)
            chiplet = next(
                (candidate for candidate in chiplets if candidate.dram_budget - candidate.inuse_budget >= mem_req_per_layer),
                None,
            )
            if chiplet is None:
                raise ValueError("Not enough memory to load the model.")

            chiplet.inuse_budget += mem_req_per_layer
            chiplet._current_allocated += mem_req_per_layer
            chiplet.content_params['weights'] = [] if 'weights' not in chiplet.content_params else chiplet.content_params['weights']
            chiplet.content_params['weights'].append(
                Params(size=mem_req_per_layer, block_id=idx, name=f"Block {idx} weights", chip_id=chiplet.chiplet_id)
            )
            self._current_used += mem_req_per_layer
            self.blocks_alloc[idx] = chiplet.chiplet_id

        if self._current_used > self._total_budget:
            raise MemoryError("Current memory usage exceeds total budget.")

        sorted_items = sorted(self.blocks_alloc.items())
        self.blocks_alloc = dict(sorted_items)
        mem_monitor.update_utilization(self._mem_chiplets, self._current_used, 0)
            
    def load_model_tp(self, mod_config:BaseModelConfig, group_HBM_M: list, group_HBM_A: list) -> None:
        """
        Load a model weights into the memory system for tensor-parallelism.
        """
        # Instead of assigning layers to a chiplet, we assign each layer's weights to all chiplets
        for idx, block_type_idx in enumerate(mod_config.block_type_sequence):
            blk_config:BaseBlockConfig = mod_config.hybrid_blocks[block_type_idx] if mod_config.hybrid else mod_config.block_config
            params_per_layer = blk_config.parameter_count
            mem_req_per_layer = params_per_layer * common.ByteperParam / (1024 * 1024)  # convert to MB
            mem_req_per_chiplet = mem_req_per_layer / self.total_mem_chiplets
            chunk_id = 0
            for chiplet in self._mem_chiplets:
                if chiplet.dram_budget - chiplet.inuse_budget >= mem_req_per_layer:
                    chiplet.inuse_budget += mem_req_per_chiplet
                    chiplet._current_allocated += mem_req_per_chiplet
                    chiplet.content_params['weights'] = [] if 'weights' not in chiplet.content_params else chiplet.content_params['weights']
                    chiplet.content_params['weights'].append(Params(size=mem_req_per_layer, chunk_id=chunk_id, block_id=idx, name=f"Block {idx} weights", chip_id=chiplet.chiplet_id))
                    self._current_used += mem_req_per_chiplet
                    # self._current_allocated += mem_req_per_layer
                    # record that this block (idx) is stored on multiple chiplets
                    if idx not in self.blocks_alloc:
                        self.blocks_alloc[idx] = []
                    self.blocks_alloc[idx].append(chiplet.chiplet_id)
                    chunk_id += 1
                else:
                    raise ValueError("Not enough memory to load the model for tensor-parallelism.")
        
        if self._current_used > self._total_budget:
            raise MemoryError("Current memory usage exceeds total budget.")
        mem_monitor.update_utilization(self._mem_chiplets, self._current_used, 0)

    def get_mem_byid(self, chiplet_id: int) -> mem_chiplet:
        """
        Get a memory chiplet by its ID.
        """
        for chiplet in self._mem_chiplets:
            assert isinstance(chiplet, mem_chiplet), "Chiplet is not of type mem_chiplet."
            if chiplet.chiplet_id == chiplet_id:
                return chiplet
        raise ValueError(f"Chiplet with ID {chiplet_id} not found.")
    
    def load_data_byid(self, chiplet_id: int, data: intermediate, timestep: int) -> None:
        """
        Load data into a memory chiplet by its ID.
        """
        chiplet = self.get_mem_byid(chiplet_id)
        if chiplet.inuse_budget + data.size > chiplet.dram_budget:
            raise MemoryError(f"Not enough memory in chiplet {chiplet_id} to load data.")
        chiplet.inuse_budget += data.size
        chiplet.content_params[data.req_id] = [] if data.req_id not in chiplet.content_params else chiplet.content_params[data.req_id]
        chiplet.content_params[data.req_id].append(data)
        self._current_used += data.size
        # self._current_allocated += data.size
        mem_monitor.update_utilization(self._mem_chiplets, self._current_used, timestep)

    def offload_data_byid(self, chiplet_id: int, data: intermediate, timestep: int) -> None:
        """
        Offload data from a memory chiplet by its ID.
        """
        chiplet: mem_chiplet = self.get_mem_byid(chiplet_id)
        if data not in chiplet.content_params.get(data.req_id, []):
            raise ValueError(f"Data {data.name} not found in chiplet {chiplet_id}.")
        assert isinstance(data, intermediate), "Data is not of type intermediate."
        chiplet.content_params[data.req_id].remove(data)
        chiplet.inuse_budget -= data.size
        self._current_used -= data.size
        # self._current_allocated -= data.size
        mem_monitor.update_utilization(self._mem_chiplets, self._current_used, timestep)
    
    def refresh_states_byid(self, chiplet_id: int, data_prev: intermediate, data_cur: intermediate, timestep: int) -> None:
        """
        Refresh states in a memory chiplet by its chip ID, and new intermediate.
        """
        self.offload_data_byid(chiplet_id, data_prev, timestep)
        self.load_data_byid(chiplet_id, data_cur, timestep)

    def list_allocated_mem_chiplets(self, block_name: str) -> list[mem_chiplet]:
        """
        List all memory chiplets allocated for a specific block of the model
        """
        mem_chiplets_mapped = []
        for mem_chiplet_inst in self._mem_chiplets:
            assert isinstance(mem_chiplet_inst, mem_chiplet), "mem_chiplet_inst is not an instance of mem_chiplet."
            for params in mem_chiplet_inst.content_params['weights']:
                if block_name in params.name:
                    mem_chiplets_mapped.append(mem_chiplet_inst)

        if not mem_chiplets_mapped:
            raise ValueError(f"Block {block_name} not found in any memory chiplet.")
        
        return mem_chiplets_mapped

    def check_memchiplets_availability(self, allocated_memory: float, block_name: str) -> bool:
        """
        Check allocated memory availability for a size on a certain block of the model
        allocated_memory: memory required per block in MB
        """
        if common.task_parallelism == 'tensor':
            return self.check_memchiplets_availability_tp(allocated_memory, block_name)

        check_flag = False
        for mem_chiplet_inst in self._mem_chiplets:
            assert isinstance(mem_chiplet_inst, mem_chiplet), "mem_chiplet_inst is not an instance of mem_chiplet."
            for params in mem_chiplet_inst.content_params['weights']:
                if block_name in params.name:
                    if mem_chiplet_inst.dram_budget - mem_chiplet_inst._current_allocated < allocated_memory:
                        return -1
                    else:
                        check_flag = True
                        break
        if not check_flag:
            raise ValueError(f"Block {block_name} not found in any memory chiplet.")
        else:
            return 1
        
    def check_memchiplets_availability_tp(self, allocated_memory: float, block_name: str) -> bool:
        """
        Check allocated memory availability for a size on a certain block of the model
        allocated_memory: memory required per block in MB
        """
        mem_chiplets_mapped = self.list_allocated_mem_chiplets(block_name)
        required_memory_per_chiplet = allocated_memory / len(mem_chiplets_mapped)

        for mem_chiplet_inst in mem_chiplets_mapped:
            if mem_chiplet_inst.dram_budget - mem_chiplet_inst._current_allocated < required_memory_per_chiplet:
                return -1
        return 1

    def allocate_memchiplet(self, allocated_memory: float, block_name: str) -> None:
        """
        Allocate memory on a memory chiplet for a specific block of the model
        """
        if common.task_parallelism == 'tensor':
            return self.allocate_memchiplet_tp(allocated_memory, block_name)
        
        for mem_chiplet_inst in self._mem_chiplets:
            assert isinstance(mem_chiplet_inst, mem_chiplet), "mem_chiplet_inst is not an instance of mem_chiplet."
            for params in mem_chiplet_inst.content_params['weights']:
                if block_name in params.name:
                    mem_chiplet_inst._current_allocated += allocated_memory
                    return
        raise ValueError(f"Block {block_name} not found in any memory chiplet.")
    
    def allocate_memchiplet_tp(self, allocated_memory: float, block_name: str) -> None:
        """
        Allocate memory on a memory chiplet on each block of the model
        """
        mem_chiplets_mapped = self.list_allocated_mem_chiplets(block_name)        
        required_memory_per_chiplet = allocated_memory / len(mem_chiplets_mapped)

        for mem_chiplet_inst in mem_chiplets_mapped:
            mem_chiplet_inst._current_allocated += required_memory_per_chiplet
        return

    def relieve_allocate_memchiplet(self, allocated_memory: float, block_name: str) -> None:
        """
        Relieve allocated memory on a memory chiplet on each block of the model
        """
        if common.task_parallelism == 'tensor':
            return self.relieve_allocate_memchiplet_tp(allocated_memory, block_name)
        
        for mem_chiplet_inst in self._mem_chiplets:
            assert isinstance(mem_chiplet_inst, mem_chiplet), "mem_chiplet_inst is not an instance of mem_chiplet."
            for params in mem_chiplet_inst.content_params['weights']:
                if block_name in params.name:
                    mem_chiplet_inst._current_allocated -= allocated_memory
                    if mem_chiplet_inst._current_allocated < 0:
                        raise ValueError("Allocated memory on chiplet went below zero.")
                    return
        raise ValueError(f"Block {block_name} not found in any memory chiplet.")

    def relieve_allocate_memchiplet_tp(self, allocated_memory: float, block_name: str) -> None:
        """
        Relieve allocated memory on a memory chiplet on each block of the model
        """
        mem_chiplets_mapped = self.list_allocated_mem_chiplets(block_name)
        required_memory_per_chiplet = allocated_memory / len(mem_chiplets_mapped)

        for mem_chiplet_inst in mem_chiplets_mapped:
            mem_chiplet_inst._current_allocated -= required_memory_per_chiplet
            if mem_chiplet_inst._current_allocated < 0:
                raise ValueError("Allocated memory on chiplet went below zero.")
        return

    # def load_bw_byid(self, chiplet_id: int, size: float, timestep: int) -> None:
    #     """
    #     Load bandwidth into a memory chiplet by its ID.
    #     """
    #     chiplet = self.get_mem_byid(chiplet_id)
    #     if chiplet.bw_inuse + size > chiplet.BW_budget:
    #         raise MemoryError(f"Not enough bandwidth in chiplet {chiplet_id} to load data.")
    #     chiplet.bw_inuse += size
    #     #mem_monitor.update_bandwidth_utilization(self._mem_chiplets, chiplet.bw_inuse)
        
    # def offload_bw_byid(self, chiplet_id: int, size: float, timestep: int) -> None:
    #     """
    #     Offload bandwidth from a memory chiplet by its ID.
    #     """
    #     chiplet = self.get_mem_byid(chiplet_id)
    #     if chiplet.bw_inuse < size:
    #         raise ValueError(f"Bandwidth {size} to offload is greater than current bandwidth in chiplet {chiplet_id}.")
    #     chiplet.bw_inuse -= size
    #     #mem_monitor.update_bandwidth_utilization(self._mem_chiplets, chiplet.bw_inuse)
    
