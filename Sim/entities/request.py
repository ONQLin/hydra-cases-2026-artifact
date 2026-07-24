from typing import Tuple, List
import networkx as nx
import numpy as np
from Sim.config.model_config import BaseModelConfig
from Sim.entities.infer import infer
from Sim.entities.block import block
from Sim.entities.base_entity import BaseEntity
from Sim.metrics.monitor import tokens_monitor

class Request(BaseEntity):
    def __init__(self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        context_length: int,
        # num_processed_tokens: int = 0,
        model_config: BaseModelConfig
    ):
        self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._num_prefill_tokens = num_prefill_tokens
        self._num_decode_tokens = num_decode_tokens
        self._context_length = context_length
        # self._num_processed_tokens = num_processed_tokens

        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0

        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False
        # one inference includes prefill/decode of multiple blocks
        # use a dict keyed by "prefill" and "decode"
        self._infs: dict[str, infer] = {}
        self._num_restarts = 0
        self._model_config = model_config
        self._process_idx = -1  # the inf under processing
        self._process_target = -1 # label the PE

    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id
    
    def set_batch_id(self, batch_id: int):
        self._batch_id = batch_id

    def fill_request(self):
        # TODO: generalize this to support hybrid model (different blocks)
        # self._model_config.hybrid == False
        # prefill infers

        if self._num_prefill_tokens > 0:
            blocks = [
                block(
                    context_length=self._num_prefill_tokens,
                    block_config=self._model_config.hybrid_blocks[type_idx],
                    block_num=idx  # block_num starts from 1 for decode blocks
                ) for idx, type_idx in enumerate(self._model_config.block_type_sequence)
            ]
            self._infs["prefill"] = infer(
                type="prefill",
                blocks=blocks
            )

        # decode infers
        context_length = self._context_length + 1
        if self._num_decode_tokens > 0:
            blocks = [
                block(
                    context_length=context_length,
                    block_config=self._model_config.hybrid_blocks[type_idx],
                    block_num=idx  # block_num starts from 1 for decode blocks
                ) for idx, type_idx in enumerate(self._model_config.block_type_sequence)
            ]
            self._infs["decode"] = infer(
                type="decode",
                blocks=blocks,
                max_decoding_length=self._num_decode_tokens
            )

        # configure the infer entities with the request ID
        for inf in self._infs.values():
            inf.config_infer(self._id)
    
    def has_prefill(self) -> bool:
        return "prefill" in self._infs

    def has_decode(self) -> bool:
        return "decode" in self._infs

    def is_only_prefill(self) -> bool:
        return self.has_prefill() and not self.has_decode()

    def is_only_decode(self) -> bool:
        return self.has_decode() and not self.has_prefill()

    def get_prefill_length(self) -> int:
        if self.has_prefill():
            return self._num_prefill_tokens
        else:
            return 0
    
    def get_decode_length(self) -> int:
        if self.has_decode():
            return self._num_decode_tokens
        else:
            return 0
    
    def step_processing(self, timestep=0) -> bool:
        # bookkeeping for timing/metrics
        if self._process_idx == -1:
            self._scheduled_at = timestep
        elif self._process_idx == 0:
            self._execution_time = timestep
            tokens_monitor.add_first_token(timestep - self._scheduled_at)
        else:
            tokens_monitor.add_time_dec_token(timestep - self._execution_time)
            self._execution_time = timestep

        self._process_idx += 1

        decode_inf = self._infs.get("decode", None)
        decode_len = decode_inf.max_decoding_length if decode_inf is not None else 0
        total_steps = 1 + decode_len

        # completion check
        if self._process_idx >= total_steps:
            if self._process_idx > total_steps:
                raise ValueError("Processing index exceeds the number of inferences in the request.")
            self._completed_at = timestep  # label the completion time
            return True
        else:
            # perform the current step
            if self._process_idx < 1:
                if self.has_prefill():
                    # prefill step
                    self._infs["prefill"].step_processing()
                    if not self.has_decode():
                        return True
                else:
                    self._infs["decode"].step_processing()
            elif self.has_decode():
                # decode step
                # align context length index whether prefill exists or not
                decode_step_index = self._process_idx
                self._infs["decode"].step_iteration(
                    context_length=self._context_length + decode_step_index,
                    model_config=self._model_config
                )
            return False
        
    def reset_request(self) -> None:
        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0

        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False
        self._process_idx = -1  # the inf under processing
        self._process_target = -1 # label the PE
        for inf in self._infs.values():
            inf.process_idx = -1
