import logging
from typing import Tuple

import numpy as np
import pandas as pd

from Sim.config.sys_config import TraceRequestLengthGeneratorConfig
from Sim.request_generator.base_request_length_generator import BaseRequestLengthGenerator

logger = logging.getLogger(__name__)


class TraceRequestLengthGenerator(BaseRequestLengthGenerator):

    def __init__(self, config: TraceRequestLengthGeneratorConfig, request_num: int = 128):
        super().__init__(config)

        self.trace_df = pd.read_csv(config.trace_file)
        # just keep the first token_num rows
        self.trace_df = self.trace_df.head(request_num)

        # convert request type to int. 0: end to end 1: prefill only, 2: decode only
        self.req_type = {"e2e": 0, "prefill": 1, "decode": 2}[config.request_type]
        
        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        self.trace_df["context_length"] = self.trace_df["num_prefill_tokens"]
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"] * config.decode_scale_factor

        if self.req_type == 1:
            self.trace_df["num_decode_tokens"] = 0
        elif self.req_type == 2:
            self.trace_df["num_prefill_tokens"] = 0

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(int)
        self.trace_df["context_length"] = self.trace_df["context_length"].astype(int)

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (self.trace_df["context_length"] + self.trace_df["num_decode_tokens"])
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)

        # deduct the diff tokens from the prefill and decode tokens proportionally
        prefill_tokens_ratio = self.trace_df["context_length"] / total_tokens
        decode_tokens_ratio = self.trace_df["num_decode_tokens"] / total_tokens

        self.trace_df["context_length"] -= (np.ceil(diff_tokens * prefill_tokens_ratio)).astype(int)
        self.trace_df["num_decode_tokens"] -= (np.ceil(diff_tokens * decode_tokens_ratio)).astype(int)
        
        self.trace_df["context_length"] = self.trace_df["context_length"].clip(lower=1)
        # if req type is e2e or prefill only,
        if self.req_type in [0, 1]:
            self.trace_df["num_prefill_tokens"] = self.trace_df["context_length"]
        
        

        # make sure that there is at least one prefill and decode token - only needed for e2e requests
        if self.req_type == 0:
            self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(lower=1)
            self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(lower=1)
            

        assert all(
            self.trace_df["context_length"] + self.trace_df["num_decode_tokens"]
            <= self.config.max_tokens
        )

        # for e2e and prefill
        if self.req_type in [0, 1]:
            assert any(self.trace_df["num_prefill_tokens"] > 0)
        # for e2e and decode
        if self.req_type in [0, 2]:
            assert all(self.trace_df["num_decode_tokens"] > 0)
        bad_idxs = self.trace_df.index[self.trace_df["context_length"] <= 0].tolist()
        assert all(self.trace_df["context_length"] > 0)

        self.next_request_idx = 0

    def get_next_num_tokens(self) -> Tuple[float, float, float]:
        if self.next_request_idx >= len(self.trace_df):
            return None, None, None

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        return (
            row["num_prefill_tokens"],
            row["num_decode_tokens"],
            row["context_length"]
        )
