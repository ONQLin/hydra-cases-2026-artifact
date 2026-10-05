from typing import List

import simpy.exceptions
from Sim.request_generator.base_request_generator import BaseRequestGenerator
from Sim.request_generator.base_request_source import BaseRequestSource
from pathlib import Path
from Sim.request_generator.trace_request_length_generator import TraceRequestLengthGenerator
from Sim.request_generator.static_request_generator import StaticRequestIntervalGenerator
from Sim.config.sys_config import TraceRequestGeneratorConfig
from Sim.config.model_config import BaseModelConfig,Llama3_8BModelConfig
from Sim.entities.request import Request
import simpy
import Sim.common as common
import Sim.metrics.monitor as monitor
from Sim.logger import init_logger, _setup_logger
_setup_logger()   # root logger handles console + file
logger = init_logger(__name__)  # child logger inherits handlers

# We now use simpy for request generation.
class TraceRequestGenerator(BaseRequestSource):

    @staticmethod
    def get_name():
        return 'trace'

    def export_workload(self, output_dir):
        self.request_length_generator.trace_df.to_csv(
            Path(output_dir) / 'effective_workload.csv', index=False)

    def __init__(self, sim_done: simpy.Event, tr_config: TraceRequestGeneratorConfig, mod_config:BaseModelConfig ,env: simpy.Environment):
        #super().__init__(config)
        self.tr_config = tr_config
        self.mod_config = mod_config
        # Generate the request that includes decode and prefill lengths
        self.request_length_generator = TraceRequestLengthGenerator(tr_config.trace_length_generator_config, tr_config.num_requests)
        self.mod_config.validate_batch_lengths(self.request_length_generator.trace_df, common.batch_size)
        self.request_interval_generator = StaticRequestIntervalGenerator(tr_config.trace_interval_generator_config)
        self.env = env
        self.generate_job = True
        self.max_num_jobs = common.max_requests_inj
        self.sim_done = sim_done
        self.action = env.process(self.run()) 
        
        
    def run(self):
        """Run the trace request generator."""
        
        logger.info(
            f"Starting generation with {self.request_length_generator.trace_df.shape[0]} requests"
        )
        
        inf_id = 0
        while (self.generate_job):
            jobs_exceed_max_in_parallel = monitor.request_counter.pending_requests + monitor.request_counter.running_requests >= common.max_requests_in_parallel
            
            if jobs_exceed_max_in_parallel:
                raise Exception("Maximum number of requests in parallel exceeded. Please increase the max_requests_in_parallel parameter in common.py.")

            pf_len, dc_len, c_len = self.request_length_generator.get_next_num_tokens()
            if pf_len is None or dc_len is None or c_len is None:
                logger.info("No more requests to generate.")
                common.inj_finish = True
                break
        
            Request_interval = self.request_interval_generator.get_next_inter_request_time()
            current_time = self.env.now
            request_gen = Request(
                arrived_at=current_time,
                num_prefill_tokens=int(pf_len),
                num_decode_tokens=int(dc_len),
                context_length=int(c_len),
                model_config=self.mod_config
            )
            request_gen.fill_request()  # Fill the request with prefill and decode blocks
            # Generate the request based on the model config and request config.
            common.RequestQueue.outstanding.append(request_gen)
            monitor.request_counter.increment_pending()
            monitor.request_counter.increment_total()
            

            if (monitor.request_counter.total_requests >= self.max_num_jobs):                                 # check if max number of jobs, given in config file, are created
                self.generate_job = False   
                common.inj_finish = True
                # make up
                # self.sim_done.succeed()
                
            try:
                yield self.env.timeout(int(Request_interval*common.time_granularity))  # Convert seconds to microseconds              
            except simpy.exceptions.Interrupt as e:
                pass
