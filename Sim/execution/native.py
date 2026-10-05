"""HYDRA accelerator profiling shared by both execution backends."""

import numpy as np
import Sim.common as common
import Sim.config.utils as utils
from analytic_profile.Base_Accmodel import BaseAccModel
from Sim.execution.BaseExecutionBackend import BaseExecutionBackend


class NativeExecutionBackend(BaseExecutionBackend):
    def __init__(self, config=None):
        self.config = config

    @staticmethod
    def get_name() -> str:
        return "hydra_sim"

    def start_block(self, env, *args):
        latency, utilization = self.execute_block(*args)
        return env.timeout(int(latency)), utilization

    def profile_block(self, block, stage, batch_size, chiplet, bandwidth):
        profiles = []
        if len(block.layers_configs) != len(block.layers):
            raise ValueError('Layer configuration groups and names must match.')
        logic_name = utils.chiplet_types_list[chiplet.chiplet_type]
        analytic_model = BaseAccModel.create_from_name(logic_name)
        for layer_configs, layer_names in zip(block.layers_configs, block.layers):
            if len(layer_configs) != len(layer_names):
                raise ValueError("Layer configuration and names must match.")
            for layer_config, name in zip(layer_configs, layer_names):
                if name in ("MHA", "SSM"):
                    name += "_p" if stage == "prefill" else "_d"
                arguments = analytic_model.config_params(**vars(layer_config))
                arguments['bs'] = block.context_length if stage == "prefill" else 1
                arguments['batch_size'] = batch_size
                arguments['L_seq'] = block.context_length
                if stage == "prefill" and 'in_size' in arguments:
                    arguments['in_size'] *= block.context_length
                arguments['stage'] = stage
                arguments['logic_name'] = logic_name
                arguments['ext_bw'] = bandwidth
                profiles.append(analytic_model.profile_kernel(name, **arguments))
        return profiles

    def execute_block(self, block, stage, batch_size, chiplet, memory_id, bandwidth):
        profiles = self.profile_block(block, stage, batch_size, chiplet, bandwidth)
        latency = sum(common.convert_analy_time(item.total_latency) for item in profiles)
        return int(latency), float(np.mean([item.utilization for item in profiles]))
