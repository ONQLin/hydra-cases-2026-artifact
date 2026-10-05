import simpy
import json
import random
from pathlib import Path
from pyvis.network import Network
import numpy as np

from Sim.config.sys_config import HPSim_Config
from Sim.request_generator.trace_request_generator import TraceRequestGenerator
import Sim.common as common
import Sim.config.utils as utils
from Sim.entities.chips_network import chip_graph
from Sim.entities.mem_sys import mem_sys
from Sim.entities.comp_sys import comp_sys
from Sim.metrics.monitor import mem_monitor, comp_monitor, bandwidth_monitor, tokens_monitor, congestion_monitor, request_counter
from Sim.metrics.serving_recorder import ServingMetricsRecorder
from Sim.sim_core import SimulationManager
from Sim.placer.BasePlacer import BasePlacer
from Sim.entities.static_mapper import static_mapper

from Sim.logger import init_logger, _setup_logger
_setup_logger()   # root logger handles console + file
logger = init_logger(__name__)  # child logger inherits handlers


class Simulator:
    def __init__(self, config: HPSim_Config) -> None:
        self._config: HPSim_Config = config

        self._time = 0
        self._terminate = False
        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        self._event_queue = []

        self._event_trace = []
        
        self.resources_graph: chip_graph = config.placmt_config.chip_graph
        self.plcmt_inst: BasePlacer = None
        self.mem_sys: mem_sys = None

        self.setup_config(config)

    def setup_config(self, config: HPSim_Config):
        random.seed(config.seed)
        np.random.seed(config.seed)
        # load some variables to common and utils from config
        #utils.NoI_bw = config.chips_config.D2D_NoI_bw
        common.logic_lib = config.chips_config.chips_lib.logics
        common.mem_lib = config.chips_config.chips_lib.mem_lib
        common.component_lib = config.chips_config.chips_lib.acc_components
        common.mapping_strategy = config.mapping_config.mapping_strategy
        common.task_parallelism = config.mapping_config.task_parallelism
        common.local_scheduler = config.cluster_config.local_scheduler
        common.ByteperParam = config.workload_config.bytes_per_param
        common.batch_size = config.cluster_config.batch_size

    def run(self):
        from Sim.execution import BaseExecutionBackend
        backend_class = BaseExecutionBackend.create_from_name(self._config.simulator_backend)
        self.execution_backend = backend_class(self._config)
        env = self.execution_backend.create_environment()
        sim_done = env.event()
        tokens_monitor.reset()
        congestion_monitor.reset()
        request_counter.reset()
        common.RequestQueue.outstanding.clear()
        common.RequestQueue.ready.clear()
        common.RequestQueue.executable.clear()
        common.RequestQueue.running.clear()
        common.RequestQueue.completed.clear()
        common.preempted_requests.clear()
        common.req_allocated_mems.clear()
        common.inj_finish = False

        logger.info("Feed Request ...")
        self._request_generator = TraceRequestGenerator(sim_done=sim_done, tr_config=self._config.workload_config.request_generator_config,
                                                        mod_config=self._config.workload_config.model_config, env=env)

        common.output_folder = self._config.metrics_config.output_dir
        file = common.output_folder + "/log_info.txt"
        open(file, "w").close()  # Clear the log file
        
        if self._time_limit != float("inf"):
            common.max_sim_length = int(self._time_limit * common.time_granularity)  # convert to time granularity unit
        
        logger.info("Init Placement ...")
        self.plcmt_inst = self._config.placmt_config.placer_inst(chiplet_num=self.resources_graph.num_nodes, chiplet_alloc=self._config.placmt_config.chiplet_alloc)
        group_HBM_M, group_HBM_A = self.plcmt_inst.make_chiplet_placement(chip_graph=self.resources_graph, model_config=self._config.workload_config.model_config, 
                                               mem_size=self._config.chips_config.mem_sizes[0], output_folder=common.output_folder)
        for key,value in self.plcmt_inst.chiplet_alloc.items():
            if value > 0:
                extra_bw = int((utils.NoI_bw - utils.HW_D2D_BW)/128)
                extra_area = extra_bw * ((common.component_lib['PHY'][128]['area']) * 4 + 
                                         common.component_lib['bump'][128]['area'] + common.component_lib['dma'][128]['area'])
                if key in common.logic_lib.keys():
                    c_area = extra_area + common.logic_lib[key]['area']
                else:
                    m_area = 121
                #utils.total_area += value * area
        utils.total_area = max(c_area, m_area) * sum(self.plcmt_inst.chiplet_alloc.values())

        logger.info("Init Memory System and load model ...")

        # if common task parallelism is pipelined, use load_label as placmt label
        # else use tp mapping label as placmt label
        if common.task_parallelism == 'pipeline':
            load_label = self._config.placmt_config.placer_label
        elif common.task_parallelism == 'tensor':
            load_label = 'tp'
            utils.used_bws = utils.used_bws_dict_tensor[utils.NoI_bw] # overwrite the used_bws for tensor parallelism
        else:
            raise ValueError(f"Unknown task parallelism: {common.task_parallelism}")
        self.mem_sys = mem_sys(self.resources_graph)
        self.mem_sys.load_model(label=load_label, mod_config=self._config.workload_config.model_config, 
                               group_HBM_M=group_HBM_M, group_HBM_A=group_HBM_A)
        logger.info("Init Comp system ...")
        self.comp_sys = comp_sys(self.resources_graph)

        # it is aborted because we now just input the hw logic instance to the analytical model
        # common.configure_chip_features(common.analytic_models_config, Col_PE=24, Row_PE=24, Num_Array=8, C_sram=16, DMAs=8, Sram_bw=32)
        self.mapper = static_mapper()
        common.job_mapping, _ = self.mapper.generate_mapping(self._config.workload_config.model_config, self.comp_sys, self.mem_sys)
        self._request_generator.export_workload(common.output_folder)
        setup = {
            "weight_mapping": self.mem_sys.blocks_alloc,
            "task_mapping": common.job_mapping,
            "chiplets": [{"id": int(data['id']), "type": int(data['chiplet_type'])}
                         for _, data in self.resources_graph.graph.nodes(data=True)],
            "links": sorted([sorted([int(self.resources_graph.graph.nodes[u]['id']),
                                     int(self.resources_graph.graph.nodes[v]['id'])])
                             for u, v in self.resources_graph.graph.edges]),
        }
        (Path(common.output_folder) / "system_snapshot.json").write_text(json.dumps(setup, indent=2))
        logger.info("Init Simulation Manager ...")
        self.metrics_recorder = ServingMetricsRecorder(common.output_folder)
        self._sim_manager = SimulationManager(env, sim_done, self.resources_graph, self.mem_sys, self.comp_sys, 
                                              self._config.workload_config.model_config, self._config.cluster_config.local_scheduler_inst,
                                              self.execution_backend, self.metrics_recorder)
        

        try:
            self.execution_backend.initialize(self.resources_graph, self.mem_sys,
                                              self.comp_sys, common.job_mapping)
            env.run(until=sim_done)
        finally:
            self.execution_backend.close()

        total_tokens = tokens_monitor.output_tokens
        total_finish_tokens = tokens_monitor.finished_tokens
        elapsed_seconds = max(env.now / common.time_granularity, 1e-9)
        token_p_second = total_tokens / elapsed_seconds
        ftoken_p_second = total_finish_tokens / elapsed_seconds
        total_preempted_requests = len(common.preempted_requests)
        total_preempted_tokens = sum(common.preempted_requests.values())

        logger.info(f"output tokens per second: {token_p_second}")
        logger.info(f"finished tokens per second: {ftoken_p_second}")
        logger.info(f"Total output tokens: {total_tokens}")
        logger.info(f"Total preempted tokens: {total_preempted_tokens}; Total preempted requests: {total_preempted_requests}")
        logger.info(
            f"Request counters - completed: {request_counter.completed_requests}, "
            f"running: {request_counter.running_requests}, pending: {request_counter.pending_requests}"
        )

        if tokens_monitor.time_to_first_token:
            avg_time_first_token = float(np.mean(tokens_monitor.time_to_first_token))
            logger.info(f"Average time to first token: {avg_time_first_token}")
        else:
            logger.info("Average time to first token: N/A (no first token produced yet)")

        if tokens_monitor.time_per_output_token:
            avg_time_output_token = float(np.mean(tokens_monitor.time_per_output_token))
            logger.info(f"Average time per output token: {avg_time_output_token}")
        else:
            logger.info("Average time per output token: N/A (no output token interval recorded yet)")

        if total_tokens == 0 and (request_counter.pending_requests > 0 or request_counter.running_requests > 0):
            logger.warning(
                "No output tokens were produced before simulation cutoff. "
                "This usually means time_limit is too short for long prefill workloads."
            )

        logger.info(f"Total area: {utils.total_area} mm^2")
        
        if utils.verbose:        
            avg_mem_util = mem_monitor.visualize_trace(self.mem_sys._total_budget, int(self.mem_sys._total_budget/len(self.mem_sys._mem_chiplets)), output_dir=common.output_folder)
            avg_comp_util = comp_monitor.visualize_trace(output_dir=common.output_folder)
            logger.info(f"Average Compute Utilization: {avg_comp_util*100/self.comp_sys._total_budget}%")
            bandwidth_monitor.visualize_trace(output_dir=common.output_folder)
        # bandwidth_monitor.visualize_IO_trace(output_dir=common.output_folder)
        # bandwidth_monitor.visualize_Link_trace(output_dir=common.output_folder)
        # while True:
        #     if sim_done.triggered:
        #         break
        #     try:
        #         env.step()
        #     except simpy.core.EmptySchedule:
        #         # No more events to process
        #         break
        

        print("Outputs are saved in ", common.output_folder)
        print("Simulation finished.")
        metrics = {
            "backend": self._config.simulator_backend,
            "elapsed_simulation_s": elapsed_seconds,
            "output_tokens": total_tokens,
            "tokens_per_sec": token_p_second,
            "finished_tokens_per_sec": ftoken_p_second,
            "ttft_s": float(np.mean(tokens_monitor.time_to_first_token)) / common.time_granularity
                      if tokens_monitor.time_to_first_token else None,
            "ttft_samples": len(tokens_monitor.time_to_first_token),
            "completed_requests": request_counter.completed_requests,
            "running_requests": request_counter.running_requests,
            "pending_requests": request_counter.pending_requests,
            "preempted_requests": total_preempted_requests,
            "area_mm2": utils.total_area,
            "measurement_window": "from t=0 through fixed simulation cutoff; no warmup exclusion",
        }
        metrics_path = Path(common.output_folder) / "metrics.json"
        temporary = metrics_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(metrics, indent=2))
        temporary.replace(metrics_path)
        return metrics_path
