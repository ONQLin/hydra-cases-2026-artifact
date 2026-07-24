import simpy

import Sim.common as common
import Sim.entities.chips_network as chips_network
import Sim.entities.mem_sys as mem_sys
import Sim.entities.comp_sys as comp_sys
import Sim.metrics.monitor as monitor
from Sim.entities.mem_chiplet import mem_chiplet
from Sim.entities.request import Request
from Sim.entities.infer import infer
from Sim.processing import processing
from Sim.entities.batch_of_req import BatchOfRequests
from Sim.config.model_config import BaseModelConfig
from Sim.scheduler.BaseReqScheduler import BaseReqScheduler
from Sim.entities.mem_chiplet import Params

# This is a request scheduler that schedules requests in a static manner.   
class StaticReplicaScheduler(BaseReqScheduler):
    def __init__(self, chip_graph: chips_network.chip_graph, mem_sys: mem_sys.mem_sys, comp_sys: comp_sys.comp_sys, bs = 1):
        self._max_concurrent_batch = min(common.max_requests_in_parallel, comp_sys._total_budget, mem_sys._num_io_limit)
        self._max_num_batch_tokens = -1  # TODO: Set this based on the model and memory system
        self._max_concurrent_requests = 9999
        self._current_bs_est = bs

    def set_max_concurrent_requests(self, model_config: BaseModelConfig, mem_sys: mem_sys.mem_sys) -> None:
        self.model_config = model_config
        for mem_chiplet_inst in mem_sys._mem_chiplets:
            assert isinstance(mem_chiplet_inst, mem_chiplet), "mem_chiplet_inst is not an instance of mem_chiplet."
            num_blocks = len(mem_chiplet_inst.content_params.get('weights', []))
            left_budget = (mem_chiplet_inst.dram_budget - mem_chiplet_inst.inuse_budget)*1024*1024  # Convert to Bytes
            # this is the most conservative estimate, as it assumes all requests will use the max position embeddings (longest context length)
            mem_req_per_request = 0 # count the memory requirement per request for hybrid blocks
            for block_params in mem_chiplet_inst.content_params.get('weights', []):
                assert isinstance(block_params, Params), "block_params is not an instance of Params."
                type_idx = model_config.block_type_sequence[block_params.block_id]
                block_config = model_config.hybrid_blocks[type_idx]
                if block_config.cache_store:
                    mem_req_per_request += (block_config.states*model_config.max_position_embeddings + block_config.peak_intermed)
                else:
                    mem_req_per_request += (block_config.states + block_config.peak_intermed)
            
            max_requests = (left_budget) // mem_req_per_request
            if max_requests < self._max_concurrent_requests:
                self._max_concurrent_requests = int(max_requests) # set the max 
        print(f"Set max concurrent requests to {self._max_concurrent_requests} based on memory chiplet budgets.")

    def schedule_request(self, mem_sys: mem_sys.mem_sys, comp_sys: comp_sys.comp_sys, batchsize = 1) -> int:
        if len(common.RequestQueue.outstanding) < batchsize:
            # No outstanding requests to schedule
            return 0

        if monitor.request_counter.running_batches >= self._max_concurrent_batch:
            # Too many requests running in parallel
            return -1
        
        if comp_sys._current_run >= comp_sys._total_budget:
            # Too many comp chiplets running
            return -1

        if monitor.request_counter.running_requests + batchsize > self._max_concurrent_requests:
            # Too many requests running in parallel
            return -1
        
        incoming_batch = BatchOfRequests(common.RequestQueue.outstanding[:batchsize])
        # incoming_request: Request = common.RequestQueue.outstanding[0]
        # Self check all requests in the batch
        for incoming_req in incoming_batch.ongoing_requests:
            if incoming_req.has_prefill():
                prefill_inf: infer = incoming_req._infs["prefill"]
                if prefill_inf.type != 'prefill':
                    raise ValueError("The first inference in the request must be a prefill inference.")
             # check if all infs are filled
            if len(incoming_req._infs) == 0:
                raise ValueError("The request has no inferences.")
            for inf in incoming_req._infs.values():
                if len(inf.blocks) == 0:
                    raise ValueError("The inference has no blocks.")
        
        batch_allocate_mem = []
        for incoming_req in incoming_batch.ongoing_requests:
            # only prefill length is known in the batching stage
            if incoming_req.has_prefill():
                inf_req: infer = incoming_req._infs["prefill"]
            else:
                inf_req: infer = incoming_req._infs["decode"]
            for block in inf_req.blocks:
                pf_memory = (block.peak_intermediate_store + block.states_store) if incoming_req.has_prefill() else 0
                dc_memory = (block.peak_intermediate_store//block.context_length + (self.model_config.max_position_embeddings*block.block_config.states if block.block_config.cache_store else block.states_store)) if incoming_req.has_decode() else 0
                
                allocated_memory = max(pf_memory, dc_memory) * common.ByteperParam / (1024 * 1024)  # convert to MB
                block_name = f'Block {block.block_num}'
                allocate_success = mem_sys.check_memchiplets_availability(allocated_memory, block_name)
                if allocate_success == -1:
                    # Not enough memory to schedule the request
                    for req_id, block_name, allocated_memory in batch_allocate_mem:
                        mem_sys.relieve_allocate_memchiplet(allocated_memory, block_name)
                    return -1
                else:
                    # mem_sys.allocate_memchiplet(allocated_memory, block.block_config.name)
                    mem_sys.allocate_memchiplet(allocated_memory, block_name)
                    batch_allocate_mem.append((incoming_req._id, block_name, allocated_memory))

        for req_id, block_name, allocated_memory in batch_allocate_mem:
            common.req_allocated_mems.setdefault(req_id, []).append((block_name, allocated_memory))
            
        
        # allocate memory on all chiplets
        
        # common.allocate_mem = allocated_memory
        incoming_batch.step_batch_id()
        common.RequestQueue.ready.append(incoming_batch)
        for _ in range(batchsize):
            common.RequestQueue.outstanding.pop(0)
            monitor.request_counter.increment_running()
            monitor.request_counter.decrement_pending()

        monitor.request_counter.increment_running_batches()
        return 1
            
    def start_request(self, env: simpy.Environment, PEs:list[processing]) -> None:

        if len(common.RequestQueue.ready) == 0:
            raise ValueError("No requests ready to start.")
        
        for idx in range(len(PEs)):
            if PEs[idx].IDLE:
                PEs[idx].config_processing(common.RequestQueue.ready[0]._id)
                break
        
        for i in range(common.RequestQueue.ready[0]._batch_size):
            assert isinstance(common.RequestQueue.ready[0], BatchOfRequests), "The ready queue does not contain a BatchOfRequests."
            current_request: Request = common.RequestQueue.ready[0].ongoing_requests[i]
            current_request._scheduled = True
            current_request._scheduled_at = env.now
            current_request._process_target = idx
            current_request.step_processing(env.now)
        common.RequestQueue.ready[0]._processing_target = idx
        common.RequestQueue.executable.append(common.RequestQueue.ready[0])
        common.RequestQueue.ready.pop(0)
    
    @staticmethod
    def get_name() -> str:
        return "static"