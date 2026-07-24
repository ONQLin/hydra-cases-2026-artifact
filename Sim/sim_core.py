import simpy
import time

import Sim.common as common
from Sim.entities.chips_network import chip_graph
from Sim.entities.mem_sys import mem_sys
from Sim.entities.comp_sys import comp_sys
from Sim.metrics.monitor import mem_monitor, comp_monitor, request_counter
from Sim.processing import processing
from Sim.entities.request import Request
from Sim.entities.batch_of_req import BatchOfRequests
from Sim.config.model_config import BaseModelConfig
from Sim.scheduler.BaseReqScheduler import BaseReqScheduler

from Sim.logger import init_logger, _setup_logger

class SimulationManager:
    '''!
    Define the SimulationManager class to handle the simulation events.
    '''
    def __init__(self, env:simpy.Environment, sim_done:simpy.Event, chip_graph:chip_graph, mem_sys:mem_sys, comp_sys:comp_sys, model_config: BaseModelConfig, scheduler: BaseReqScheduler):

        self.env: simpy.Environment = env                          # SimPy environment
        self.sim_done = sim_done                # Event to signal the end of simulation
        self.chip_graph = chip_graph            # The chip graph representing the NoI (network on interposer)
        self.mem_sys = mem_sys                  # Memory system instance
        self.comp_sys = comp_sys                # Compute system instance  
        self.scheduler:BaseReqScheduler = scheduler(chip_graph, mem_sys, comp_sys, bs=common.batch_size)  # Request scheduler instance
        self.scheduler.set_max_concurrent_requests(model_config, mem_sys)
        self.PEs: list[processing] = [processing() for _ in range(common.num_wk_threads)]
        self.action = env.process(self.run())   # starts the run() method as a SimPy process

    def run(self):
        stat_time = time.time()
        
        # Later than that, all info will be logged to the file
        _setup_logger(f"{common.output_folder}/log_info.txt")   # root logger handles console + file
        logger = init_logger(__name__)  # If need log to file please apply to _setup_logger() before init_logger()

        while True:
            # Evaluate chiplets for DTPM TODO

            """
                Schedule the request in the queue, allocate memory
            """
            if len(common.RequestQueue.ready) != 0:
                self.scheduler.start_request(self.env, self.PEs)
            sched = self.scheduler.schedule_request(self.mem_sys, self.comp_sys, batchsize=common.batch_size)

            if sched == 1:
                cur_batch_id = common.RequestQueue.ready[-1]._id
                logger.info(f"batch {cur_batch_id} scheduled at time {self.env.now}.")

            """
                Start the request processing
            """
            if len(common.RequestQueue.executable) != 0:
                # Iterate over a snapshot since we remove items from executable in-loop.
                for executable_batch in list(common.RequestQueue.executable):
                    cur_batch: BatchOfRequests = executable_batch
                    PE_id = cur_batch._processing_target
                    selected_PE: processing = self.PEs[PE_id]
                    if selected_PE.IDLE:
                        if common.task_parallelism == 'pipeline':
                            self.env.process(selected_PE.run(self.env, self.chip_graph, self.mem_sys, self.comp_sys))
                        elif common.task_parallelism == 'tensor':
                            self.env.process(selected_PE.run_tp(self.env, self.chip_graph, self.mem_sys, self.comp_sys))
                        else:
                            raise ValueError(f"Unknown mapping strategy: {common.mapping_strategy}")
                    else:
                        raise ValueError(f"Processing element {PE_id} is not idle, for batch {cur_batch._id} but it should be.")
                    common.RequestQueue.running.append(executable_batch)
                    common.RequestQueue.executable.remove(executable_batch)
                
            if common.max_sim_length != -1 and self.env.now >= common.max_sim_length:
                self.sim_done.succeed()
                end_time = time.time()
                
                logger.info(f"Simulation reached the maximum length of {common.max_sim_length} time units.")
                logger.info(f"In total {request_counter.completed_requests} requests were processed.")
                logger.info(f"Total simulation time: {end_time - stat_time} seconds.")
                break
            yield self.env.timeout(common.simulation_clk)
            
            