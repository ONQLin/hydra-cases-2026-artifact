from dataclasses import dataclass, field
import random
import sys
from typing import Dict, Any
import os
import json
import numpy as np
from Sim.config.model_config import BaseModelConfig
import math


from Sim.logger import init_logger, _setup_logger
_setup_logger()   # root logger handles console + file
logger = init_logger(__name__)  # child logger inherits handlers

class RequestQueue:
    outstanding: list = []
    ready: list = []
    executable: list = []
    running: list = []
    completed: list = []
    
    @classmethod
    def remove_running_batch(cls, batch_id: int):
        cls.running = [batch for batch in cls.running if batch._id != batch_id]


max_requests_in_parallel = 5000
max_requests_inj = 5000
inj_finish = False
num_wk_threads = 50

init_placment = 'heuristic0'

output_folder = 'output'

congested_blocks = []
no_comp_blocks = []
preempted_requests = {}

ByteperParam = 1 # 1 Byte per parameter, can be changed later

simulation_clk = 100 # 100 1/time_granularity = 100 microseconds
max_sim_length = 50_000_000 # 100 seconds

time_granularity = 1_000_000 # 1s = 1_000_000 us

analy_clk_time = 1_000_000_000 # 1s = 1_000_000_000 ns

logic_lib = {} # they are set up at simulator.py
mem_lib = {}
component_lib = {}
req_allocated_mems = {}
congest_counts = {}
# Task mapping strategy: static, fcfs, worksteal, or elastic.
mapping_strategy = 'elastic'
task_parallelism = 'pipeline'
local_scheduler = 'static'
batch_size = 1
# Elastic runtime rebalance (temporary patch):
# keep static mapper preference unless queue imbalance indicates retargeting.
elastic_runtime_rebalance_enabled = True
elastic_runtime_rebalance_queue_imbalance_threshold = 2
elastic_runtime_rebalance_prefer_static_mapping = True
elastic_runtime_rebalance_reselect_interval = 8
allocate_mem = 0 
total_pf_tokens = 0
total_dc_tokens = 0

job_mapping = {} # static job mapping for block-wise pipeline

def convert_param_mB(param: int) -> int:
    """
    Convert parameter size in bytes to megabytes.
    """
    return param * ByteperParam / (1024 * 1024) # Convert bytes to megabytes

def convert_analy_time(latency):
    return latency*time_granularity / analy_clk_time

@dataclass
class AnalyticModel:
    func_name: str
    args: Dict[str, Any] = field(default_factory=dict)

    def config_params(self, **params):
        """
        Configures the parameters for the analytic model.

        Args:
            **params: A dictionary of parameter names and their values.
        """
        for key, value in params.items():
            if key in self.args:
                self.args[key] = value

@dataclass
class AnalyticModelsContainer:
    RMS_Norm: AnalyticModel
    MHA_p: AnalyticModel
    MHA_d: AnalyticModel
    Softplus: AnalyticModel
    Silu: AnalyticModel
    SSM_p: AnalyticModel
    SSM_d: AnalyticModel
    FC: AnalyticModel
    Conv1D: AnalyticModel
    Count_MHA_p: AnalyticModel = None
    Count_MHA_d: AnalyticModel = None
    Count_FC: AnalyticModel = None

analytic_models_config = AnalyticModelsContainer(
    RMS_Norm=AnalyticModel(
        func_name="rms_norm_est",
        args={'in_size': 0, 'Col_PE': 0, 'Row_PE': 0, 'Num_Array': 0}
    ),
    MHA_p=AnalyticModel(
        func_name="get_pf_attention_latency",
        args={'L_seq': 0, 'embedding_dim': 0, 'q_heads': 0, 'kv_heads': 0, 'batch_size': 1, # workload feature
              'Col_PE': 16, 'Row_PE': 16, 'Num_Array': 4, 'C_sram': 0, 'DMAs': 0, 'Sram_bw': 0} # hw feature
    ),
    MHA_d=AnalyticModel(
        func_name="get_dc_attention_latency",
        args={'L_seq': 0, 'embedding_dim': 0, 'q_heads': 0, 'kv_heads': 0, 'batch_size': 1,
              'Col_PE': 16, 'Row_PE': 16, 'Num_Array': 4, 'C_sram': 0, 'DMAs': 0, 'Sram_bw': 0}
    ),
    Softplus=AnalyticModel(
        func_name="softplus_est",
        args={'in_size': 0, 'Col_PE': 16, 'Row_PE': 16, 'Num_Array': 4}
    ),
    Silu=AnalyticModel(
        func_name="silu_est",
        args={'in_size': 0, 'Col_PE': 16, 'Row_PE': 16, 'Num_Array': 4}
    ),
    SSM_p=AnalyticModel(
        func_name="get_pf_ssm_latency",
        args={'ED': 0, 'L_seq': 0, 'Total_States': 0,
              'Col_PE': 16, 'Row_PE': 16, 'Num_Array': 4, 'C_sram': 0, 'DMAs': 0, 'Sram_bw': 0}
    ),
    SSM_d=AnalyticModel(
        func_name="get_dc_ssm_latency",
        args={'ED': 0, 'L_seq': 0, 'Total_States': 0,
              'Col_PE': 16, 'Row_PE': 16, 'Num_Array': 4, 'C_sram': 0, 'DMAs': 0, 'Sram_bw': 0}
    ),
    FC=AnalyticModel(
        func_name="get_pf_mlp_latency",
        args={'bs': 1, 'f_in': 0, 'f_out': 0,
              'Col_PE': 16, 'Row_PE': 16, 'Num_Array': 4, 'C_sram': 0, 'DMAs': 0, 'Sram_bw': 0}
    ),
    Conv1D=AnalyticModel(
        func_name="get_pf_conv1d_latency",
        args={'bs': 1, 'c_in': 0, 'c_out': 0, 'f_in': 0, 'f_out': 0, 'kernel_size': 0,
              'Col_PE': 0, 'Row_PE': 0, 'Num_Array': 0, 'C_sram': 0, 'DMAs': 0, 'Sram_bw': 0}
    )
)

analytic_models_config2 = AnalyticModelsContainer(
    RMS_Norm=AnalyticModel(
        func_name="rms_norm_est",
        args={'in_size': 0, 'logic_name': "", 'ext_bw': 0}
    ),
    MHA_p=AnalyticModel(
        func_name="get_pf_attention_latency",
        args={'L_seq': 0, 'embedding_dim': 0, 'q_heads': 0, 'kv_heads': 0, 'batch_size': 1, # workload feature
              'logic_name': "", 'ext_bw': 0} # hw feature
    ),
    MHA_d=AnalyticModel(
        func_name="get_dc_attention_latency",
        args={'L_seq': 0, 'embedding_dim': 0, 'q_heads': 0, 'kv_heads': 0, 'batch_size': 1,
              'logic_name': "", 'ext_bw': 0}
    ),
    Softplus=AnalyticModel(
        func_name="softplus_est",
        args={'in_size': 0, 'logic_name': "", 'ext_bw': 0}
    ),
    Silu=AnalyticModel(
        func_name="silu_est",
        args={'in_size': 0, 'logic_name': "", 'ext_bw': 0}
    ),
    SSM_p=AnalyticModel(
        func_name="get_pf_ssm_latency",
        args={'ED': 0, 'L_seq': 0, 'Total_States': 0,
              'logic_name': "", 'ext_bw': 0}
    ),
    SSM_d=AnalyticModel(
        func_name="get_dc_ssm_latency",
        args={'ED': 0, 'L_seq': 0, 'Total_States': 0,
              'logic_name': "", 'ext_bw': 0}
    ),
    FC=AnalyticModel(
        func_name="get_pf_mlp_latency",
        args={'bs': 1, 'f_in': 0, 'f_out': 0,
              'logic_name': "", 'ext_bw': 0}
    ),
    Conv1D=AnalyticModel(
        func_name="get_pf_conv1d_latency",
        args={'bs': 1, 'c_in': 0, 'c_out': 0, 'f_in': 0, 'f_out': 0, 'kernel_size': 0,
              'logic_name': "", 'ext_bw': 0}
    ),
    Count_MHA_p=AnalyticModel(
        func_name="get_pf_attention_count",
        args={'L_seq': 0, 'embedding_dim': 0, 'q_heads': 0, 'kv_heads': 0, 'batch_size': 1}
    ),
    Count_MHA_d=AnalyticModel(
        func_name="get_dc_attention_count",
        args={'L_seq': 0, 'embedding_dim': 0, 'q_heads': 0, 'kv_heads': 0, 'batch_size': 1}
    ),
    Count_FC=AnalyticModel(
        func_name="get_pf_mlp_count",
        args={'bs': 1, 'f_in': 0, 'f_out': 0}
    )
)

def configure_chip_features(models_container: AnalyticModelsContainer, **chip_features):
    """
    Configures chip-specific features for all analytic models in the container.

    Args:
        models_container: An instance of AnalyticModelsContainer.
        **chip_features: A dictionary of key-value pairs for the chip features to update.
                         Keys should match the argument names in the AnalyticModel.args.
    """
    # Get all AnalyticModel instances from the dataclass
    for field_name in models_container.__annotations__:
        model: AnalyticModel = getattr(models_container, field_name)
        if model is None:
            continue
        # Update the relevant arguments in each model's args dictionary
        for arg_name, arg_value in chip_features.items():
            if arg_name in model.args:
                model.args[arg_name] = arg_value
    
