from typing import List
from Sim.config.model_config import BaseBlockConfig
from Sim.entities.base_entity import BaseEntity

class block(BaseEntity):
    """
    Represents a block in the system.
    """
    def __init__(self, context_length, block_config:BaseBlockConfig, is_prefill: bool = False, block_num: int = 0):
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

        self.layers = block_config.layers
        # total output mem needs to be states_store + output_act
        self._id = block.generate_id() # unique identifier for the block
        self._inf_id = -1  # inference ID, used to link with infer entity
        self.block_num = block_num # the absolute block number in the model 
        
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