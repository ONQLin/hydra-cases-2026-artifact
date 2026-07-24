from Sim.entities.block import block
from Sim.entities.base_entity import BaseEntity


class infer(BaseEntity):
    def __init__(self, type:str, blocks:list[block], max_decoding_length: int = -1):
        self.type = type
        self.blocks = blocks
        self._id = infer.generate_id()
        self._req_id = -1
        self.process_idx = -1 # the block under processing
        self.max_decoding_length = max_decoding_length  # only used for decode infers
        # configure each block with this infer ID
        for blk in blocks:
            blk.config_blk(self._id)
                
    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id
    
    def config_infer(self, req_id: int):
        self._req_id = req_id
        for blk in self.blocks:
            blk.config_blk_request(req_id)
        
    def step_processing(self):
        self.process_idx += 1
        if self.process_idx >= len(self.blocks):
            return True
        else:
            return False

    def step_iteration(self, context_length, model_config):
        blocks = [
            block(
                context_length=context_length,
                block_config=model_config.hybrid_blocks[type_idx],
                block_num=idx  # block_num starts from 1 for decode blocks
            ) for idx, type_idx in enumerate(model_config.block_type_sequence)
        ]
        self.blocks = blocks
        self.process_idx = 0