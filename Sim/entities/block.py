from typing import List
import Sim.common as common
from Sim.config.model_config import BaseBlockConfig
from Sim.entities.base_entity import BaseEntity

class block(BaseEntity):
    """
    Represents a block in the system.
    """
    def __init__(self, context_length, block_config:BaseBlockConfig, is_prefill: bool = False, block_num: int = 0):
        if block_config.memory_accounting_version >= 2:
            if not 0 < context_length <= block_config.max_position_embeddings:
                raise ValueError(f"Context {context_length} exceeds the supported range for {block_config.type_name}")
        self.context_length = min(context_length, block_config.max_position_embeddings)
        self.block_config = block_config
        self.layers_configs = block_config.layer_configs #detailed operation configurations
        self.states_store = 0  # states memory store
        if block_config.cache_store:
            self.states_store = block_config.states * context_length
        else:
            self.states_store = block_config.states
        
        if is_prefill:
            num_tokens = context_length
        else:
            num_tokens = 1

        self.intermediate_store = block_config.intermed_mem*num_tokens  # accumulated intermediate memory
        self.peak_intermediate_store = block_config.peak_intermed*num_tokens  # peak intermediate memory
        self.output_act = block_config.output_mem*num_tokens  # output activation

        if block_config.memory_accounting_version >= 3:
            memory = block_config.memory_requirements(context_length, is_prefill)
            # v3 operators explicitly report bytes under the one-byte serving
            # contract. Avoid scaling constant state or quadratic scratch by L.
            self.states_store = memory.state_bytes
            self.intermediate_store = memory.activation_bytes
            self.peak_intermediate_store = memory.activation_bytes
            self.output_act = memory.output_bytes

        self.layers = block_config.layers
        # total output mem needs to be states_store + output_act
        self._id = block.generate_id() # unique identifier for the block
        self._inf_id = -1  # inference ID, used to link with infer entity
        self.block_num = block_num # the absolute block number in the model 
        
    def additional_memory_mib(self, previous_state_mib):
        """Admission increment; v1 retains the archived element/MiB convention."""
        if self.block_config.memory_accounting_version >= 2:
            state_growth = max(0, common.convert_param_mB(self.states_store) - previous_state_mib)
            return common.convert_param_mB(self.peak_intermediate_store) + state_growth
        state_growth = max(0, self.states_store - previous_state_mib)
        return common.convert_param_mB(self.peak_intermediate_store + state_growth)

    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id
    
    def config_blk(self, inf_id: int):
        """
        Configures the block with the inf ID.
        """
        self._inf_id = inf_id
        
    def config_blk_request(self, req_id: int):
        """
        Configures the block with the request ID.
        """
        self._req_id = req_id

    # def __repr__(self):
    #     return f"Block(id={self.block_id}, size={self.size})"

    # def get_size(self) -> int:
    #     """
    #     Returns the size of the block.
    #     """
    #     return self.size

    # def get_id(self) -> int:
    #     """
    #     Returns the unique identifier of the block.
    #     """
    #     return self.block_id
