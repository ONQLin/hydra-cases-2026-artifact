from copy import deepcopy
import simpy
import inspect
import numpy as np

import Sim.common as common
import Sim.entities.chips_network as chips_network
import Sim.entities.mem_sys as mem_sys
import Sim.entities.comp_sys as comp_sys
from Sim.common import AnalyticModel
from Sim.metrics.monitor import analytics, tokens_monitor
from Sim.metrics import monitor
from Sim.task_scheduler import select_task_slot

from Sim.entities.request import Request
from Sim.entities.batch_of_req import BatchOfRequests
from Sim.entities.infer import infer
from Sim.entities.block import block
from Sim.entities.mem_chiplet import mem_chiplet, Params, intermediate
from Sim.entities.comp_chiplet import comp_chiplet
import Sim.config.utils as utils
from Sim.config.model_config import baselayerConfig
from Sim.entities.block import block
from analytic_profile.Base_Accmodel import BaseAccModel

from Sim.logger import init_logger, _setup_logger
_setup_logger()   # root logger handles console + file
logger = init_logger(__name__)  # child logger inherits handlers

# the worker thread in the simulator
class processing:
    def __init__(self, execution_backend=None):
        from Sim.execution.native import NativeExecutionBackend
        self.execution_backend = execution_backend or NativeExecutionBackend()
        self.name = "Processing on a block of the request"
        self.batch_id = -1 # it process on a certain request
        self.IDLE = True
        self.current_batch: BatchOfRequests = None
        self.requests = None
        self.infs_to_process = None
        self.blocks_to_process = None
        self.states_cache = None
        self.input_cache = None
        self._continuous_mode = False
        self._virtual_continuous_slots = []
        self._continuous_longest_req_id = None
        self._continuous_target_slots = 0
        self._cb_virtual_slot_cap_ratio = 0.1
        self._cb_virtual_slot_cap_abs = 2
        self._cb_virtual_emit_cap_ratio = 0.02
        self._cb_virtual_emit_cap_abs = 1
        self._cb_virtual_to_real_cap = 0.5
        self._cb_virtual_total_tokens = 0
        self._cb_real_total_tokens = 0
        # Continuous batching penalties (vllm_latest only):
        # model non-zero prefill overhead when new requests are admitted.
        self._cb_prefill_penalty_scale = 1.0
        self._cb_prefill_penalty_divisor = 80
        self._cb_prefill_penalty_cap_steps = 24
        # Per-slot warmup/throttle to avoid unrealistic instant decode gain.
        self._cb_warmup_divisor = 96
        self._cb_warmup_min_steps = 1
        self._cb_warmup_max_steps = 40
        self._cb_emit_interval_divisor = 48
        self._cb_emit_interval_min = 3
        self._cb_emit_interval_max = 40
        
    def config_processing(self, batch_id: int):
        self.batch_id = batch_id
        
    def reset_processing(self):
        self.batch_id = -1
        self.IDLE = True
        self.current_batch = None
        self.requests = None
        self.infs_to_process = None
        self.blocks_to_process = None
        self.states_cache = None
        self.input_cache = None
        self._continuous_mode = False
        self._virtual_continuous_slots = []
        self._continuous_longest_req_id = None
        self._continuous_target_slots = 0
        self._cb_virtual_total_tokens = 0
        self._cb_real_total_tokens = 0

    def _init_continuous_batching(self):
        # Continuous batching emulation is only enabled for vllm_latest.
        self._continuous_mode = (common.local_scheduler == "vllm_latest")
        self._virtual_continuous_slots = []
        self._continuous_longest_req_id = None
        self._continuous_target_slots = 0
        self._cb_virtual_total_tokens = 0
        self._cb_real_total_tokens = 0
        if self._continuous_mode and self.requests:
            # Target refill size is tied to the runtime admitted batch size,
            # not only configured batch_size. This is important for vLLM where
            # admission may exceed common.batch_size.
            self._continuous_target_slots = max(
                common.batch_size,
                min(len(self.requests), common.batch_size * 2),
            )
            self._continuous_longest_req_id = max(
                self.requests,
                key=lambda req: (req.get_decode_length() + 1)
            )._id

    def _get_sample_request_for_slot(self) -> Request | None:
        if len(common.RequestQueue.outstanding) == 0:
            return None
        return common.RequestQueue.outstanding[0]

    def _estimate_virtual_slot_tokens(self, req: Request | None = None) -> int:
        if req is not None:
            decode_len = max(1, req.get_decode_length())
            # Keep virtual decode chunks modest to avoid exaggerated gains.
            return 1
        if len(common.RequestQueue.outstanding) == 0:
            return 1
        sample = common.RequestQueue.outstanding[:min(len(common.RequestQueue.outstanding), 16)]
        decode_lengths = [max(1, r.get_decode_length()) for r in sample]
        return 1

    def _estimate_slot_warmup_steps(self, req: Request | None) -> int:
        if req is not None:
            prefill_len = max(1, req.get_prefill_length())
        elif len(common.RequestQueue.outstanding) > 0:
            sample = common.RequestQueue.outstanding[:min(len(common.RequestQueue.outstanding), 16)]
            prefill_len = int(max(1, np.median([max(1, r.get_prefill_length()) for r in sample])))
        else:
            prefill_len = 1
        return int(
            max(
                self._cb_warmup_min_steps,
                min(self._cb_warmup_max_steps, np.ceil(prefill_len / self._cb_warmup_divisor)),
            )
        )

    def _estimate_slot_emit_interval(self, req: Request | None) -> int:
        if req is not None:
            prefill_len = max(1, req.get_prefill_length())
        elif len(common.RequestQueue.outstanding) > 0:
            sample = common.RequestQueue.outstanding[:min(len(common.RequestQueue.outstanding), 16)]
            prefill_len = int(max(1, np.median([max(1, r.get_prefill_length()) for r in sample])))
        else:
            prefill_len = 1
        return int(
            max(
                self._cb_emit_interval_min,
                min(self._cb_emit_interval_max, np.ceil(prefill_len / self._cb_emit_interval_divisor)),
            )
        )

    def _estimate_prefill_penalty_ticks(self, admitted_reqs: list[Request]) -> int:
        if (not self._continuous_mode) or len(admitted_reqs) == 0:
            return 0
        total_prefill_tokens = sum(max(1, req.get_prefill_length()) for req in admitted_reqs)
        penalty_scale = self._cb_prefill_penalty_scale
        penalty_steps = int(
            max(
                1,
                min(
                    self._cb_prefill_penalty_cap_steps,
                    np.ceil(
                        (total_prefill_tokens * penalty_scale)
                        / (max(1, common.batch_size) * self._cb_prefill_penalty_divisor)
                    ),
                ),
            )
        )
        return penalty_steps * common.simulation_clk

    def _admit_virtual_continuous_slots(self):
        if not self._continuous_mode:
            return 0

        # Keep the effective active set near runtime target slots by "refilling"
        # freed slots. These are virtual decode streams that emulate continuous
        # generation but do not consume scheduler slots.
        target_slots = max(common.batch_size, self._continuous_target_slots)
        max_virtual_slots = int(
            max(
                1,
                min(
                    self._cb_virtual_slot_cap_abs,
                    np.ceil(target_slots * self._cb_virtual_slot_cap_ratio),
                ),
            )
        )
        free_slots = target_slots - (len(self.requests) + len(self._virtual_continuous_slots))
        free_slots = min(
            free_slots,
            max(0, max_virtual_slots - len(self._virtual_continuous_slots)),
        )
        admitted_reqs = []
        outstanding_len = len(common.RequestQueue.outstanding)
        outstanding_idx = 0
        while free_slots > 0 and len(common.RequestQueue.outstanding) > 0:
            req = common.RequestQueue.outstanding[outstanding_idx % outstanding_len]
            warmup_steps = self._estimate_slot_warmup_steps(req)
            emit_interval = self._estimate_slot_emit_interval(req)
            self._virtual_continuous_slots.append({
                "remaining_tokens": self._estimate_virtual_slot_tokens(req),
                "warmup_steps": warmup_steps,
                "emit_interval": emit_interval,
                "emit_cooldown": 0,
            })
            admitted_reqs.append(req)
            free_slots -= 1
            outstanding_idx += 1
        return self._estimate_prefill_penalty_ticks(admitted_reqs)

    def _complete_virtual_continuous_request(self, slot):
        # Virtual slots only emulate token-level overlap, so they must not
        # mutate request lifecycle counters.
        return

    def _advance_virtual_continuous_slots(self):
        if not self._continuous_mode or len(self._virtual_continuous_slots) == 0:
            return

        # Each virtual slot contributes throttled decode steps after warmup to
        # emulate continuous generation with prefill admission overhead.
        active_slots = []
        virtual_budget_left = int(
            max(0, np.floor(self._cb_virtual_to_real_cap * max(0, self._cb_real_total_tokens)) - self._cb_virtual_total_tokens)
        )
        if virtual_budget_left <= 0:
            return
        emit_cap = int(
            max(
                1,
                min(
                    self._cb_virtual_emit_cap_abs,
                    np.ceil(max(1, len(self.requests)) * self._cb_virtual_emit_cap_ratio),
                ),
            )
        )
        emit_cap = min(emit_cap, virtual_budget_left)
        if emit_cap <= 0:
            return
        emitted_tokens = 0
        for slot in self._virtual_continuous_slots:
            if slot["warmup_steps"] > 0:
                slot["warmup_steps"] -= 1
                active_slots.append(slot)
                continue
            if slot["emit_cooldown"] > 0:
                slot["emit_cooldown"] -= 1
                active_slots.append(slot)
                continue
            if emitted_tokens >= emit_cap:
                # Keep slot active but defer emission to next iteration.
                active_slots.append(slot)
                continue
            tokens_monitor.inc_output_tokens()
            common.total_dc_tokens += 1
            emitted_tokens += 1
            self._cb_virtual_total_tokens += 1
            slot["emit_cooldown"] = max(0, slot["emit_interval"] - 1)
            slot["remaining_tokens"] -= 1
            if slot["remaining_tokens"] <= 0:
                if len(common.RequestQueue.outstanding) > 0 or (not common.inj_finish):
                    # Assume outstanding requests keep decoding in continuous mode.
                    slot_req = self._get_sample_request_for_slot()
                    slot["remaining_tokens"] = self._estimate_virtual_slot_tokens(slot_req)
                    slot["warmup_steps"] = self._estimate_slot_warmup_steps(slot_req)
                    slot["emit_interval"] = self._estimate_slot_emit_interval(slot_req)
                    slot["emit_cooldown"] = 0
                    active_slots.append(slot)
                else:
                    self._complete_virtual_continuous_request(slot)
            else:
                active_slots.append(slot)
        self._virtual_continuous_slots = active_slots

    def _flush_virtual_continuous_slots(self):
        if not self._continuous_mode or len(self._virtual_continuous_slots) == 0:
            return

        # Do not flush remaining virtual tokens at batch end; we only model
        # overlap during active iterations.
        self._virtual_continuous_slots = []

    def _force_finish_when_longest_done(self, requests_status: list[bool]) -> list[bool]:
        """
        Continuous-batching simplification:
        if the longest original request in this batch finishes, end the whole
        batch as if all requests are finished.
        """
        if (not self._continuous_mode) or (self._continuous_longest_req_id is None):
            return requests_status
        for idx, req in enumerate(self.requests):
            if req._id == self._continuous_longest_req_id:
                if idx < len(requests_status) and requests_status[idx]:
                    return [True for _ in requests_status]
                return requests_status
        # Longest request is no longer tracked (e.g. preempted); disable forcing.
        self._continuous_longest_req_id = None
        return requests_status

    def _elastic_queue_load(self, chiplet: comp_chiplet) -> int:
        # Queue length plus running-slot occupancy approximates pending pressure.
        return chiplet.queue_length + (0 if chiplet.is_available() else 1)

    def _choose_elastic_chiplet_with_rebalance(
        self,
        target_comp_chiplets: list[comp_chiplet],
        block_id: int,
        stage_key: str,
        chip_graph: chips_network.chip_graph,
        target_weight_chiplet: mem_chiplet,
    ) -> comp_chiplet | None:
        if len(target_comp_chiplets) == 0:
            return None

        # Prefer static mapper target when queue imbalance is not severe.
        preferred_chiplet = None
        if getattr(common, "elastic_runtime_rebalance_prefer_static_mapping", True):
            try:
                preferred_id = common.job_mapping.get(stage_key, {}).get(block_id, None)
            except Exception:
                preferred_id = None
            if preferred_id is not None:
                preferred_chiplet = next(
                    (c for c in target_comp_chiplets if c.chiplet_id == preferred_id),
                    None,
                )

        threshold = int(
            max(
                0,
                getattr(
                    common,
                    "elastic_runtime_rebalance_queue_imbalance_threshold",
                    2,
                ),
            )
        )
        loads = [self._elastic_queue_load(c) for c in target_comp_chiplets]
        min_load = min(loads)
        max_load = max(loads)
        imbalance = max_load - min_load

        if preferred_chiplet is not None:
            pref_load = self._elastic_queue_load(preferred_chiplet)
            # Stay with static mapping when imbalance is mild.
            if imbalance < threshold or pref_load <= (min_load + threshold):
                return preferred_chiplet

        # Rebalance path: choose least-loaded chiplet; break ties by distance.
        def score(c: comp_chiplet):
            return (
                self._elastic_queue_load(c),
                chip_graph.get_distance(c.chiplet_loc, target_weight_chiplet.chiplet_loc),
                c.chiplet_id,
            )

        return min(target_comp_chiplets, key=score)

    def _maybe_reselect_elastic_chiplet(
        self,
        selected_chiplet: comp_chiplet | None,
        target_comp_chiplets: list[comp_chiplet],
        block_id: int,
        stage_key: str,
        chip_graph: chips_network.chip_graph,
        target_weight_chiplet: mem_chiplet,
        wait_ticks: int,
    ) -> tuple[comp_chiplet | None, int]:
        """
        Avoid sticking to a single congested chiplet for too long.
        Periodically re-evaluate and migrate queued job ownership if needed.
        """
        if selected_chiplet is None:
            return selected_chiplet, wait_ticks
        interval = int(max(1, getattr(common, "elastic_runtime_rebalance_reselect_interval", 8)))
        if wait_ticks < interval:
            return selected_chiplet, wait_ticks

        next_chiplet = self._choose_elastic_chiplet_with_rebalance(
            target_comp_chiplets=target_comp_chiplets,
            block_id=block_id,
            stage_key=stage_key,
            chip_graph=chip_graph,
            target_weight_chiplet=target_weight_chiplet,
        )
        if (next_chiplet is not None) and (next_chiplet.chiplet_id != selected_chiplet.chiplet_id):
            selected_chiplet.remove_job(self.batch_id)
            next_chiplet.enqueue_job(self.batch_id)
            return next_chiplet, 0
        return selected_chiplet, 0
    


    def run(self, env: simpy.Environment, chip_graph: chips_network.chip_graph, mem_sys: mem_sys.mem_sys, comp_sys: comp_sys.comp_sys):
        self.IDLE = False
        while(True):
            if len(common.RequestQueue.running) != 0:
                self.requests = [req for batch in common.RequestQueue.running if batch._id == self.batch_id for req in batch.ongoing_requests]
                if self.requests:
                    break
            # logger.warning(f"Request {self.request_id} not found in running requests. Waiting ...")
            yield env.timeout(common.simulation_clk)
        # if len(self.request) != 1:
        #     raise ValueError(f"Multiple requests found with ID {self.batch_id}. Expected only one.")
        self.current_batch = [batch for batch in common.RequestQueue.running if batch._id == self.batch_id][0]
        # self.requests = self.requests
        self.infs_to_process = [inf for req in self.requests for inf in req._infs.values() if inf.process_idx == req._process_idx]
        self.blocks_to_process = [inf.blocks[inf.process_idx] for inf in self.infs_to_process]
        batch_done = False
        batch_size = len(self.requests)
        self.input_cache : list[intermediate] = [] # intermediate activations
        self.states_cache = {req._id: [] for req in self.requests} # KV and state cachees
        pre_states_cache = None # 
        self._init_continuous_batching()
        
        # initialize input embedding into the input cache and memory system
        
        while not batch_done:
            """
            Step 0: find the HBM store the required model params
            """
            block_id = self.blocks_to_process[0].block_num
            weight_Mem_id = mem_sys.blocks_alloc[block_id] 
            target_weight_chiplet:mem_chiplet = mem_sys.get_mem_byid(weight_Mem_id)
            
            """
            Step 1: Transmit input data (embedding tokens)
            """
            for item in self.input_cache:
                if not isinstance(item, intermediate):
                    raise ValueError(f"Invalid item in input_cache: {item}")
                
                if item.chip_id == target_weight_chiplet.chiplet_id:
                    continue  # No transmission needed

                if self.execution_backend.is_package_transfer(item.chip_id, target_weight_chiplet.chiplet_id):
                    yield from self.execution_backend.transfer_package_input(
                        env, chip_graph, mem_sys, item, target_weight_chiplet.chiplet_id)
                    continue
                
                int_mem_chiplet = mem_sys.get_mem_byid(item.chip_id)
                alloc_io_bw = min(int(utils.IO_bw / 3), 128)  # in GBps, allocate part of the IO bandwidth for data transmission

                # Unified availability check loop
                while True:
                    # Check capacity and bandwidth
                    capacity_ok = target_weight_chiplet.is_capacity_available(item.size)
                    bw_src_ok = chip_graph.check_bw_availability(int_mem_chiplet.chiplet_id, alloc_io_bw)
                    bw_dst_ok = chip_graph.check_bw_availability(target_weight_chiplet.chiplet_id, alloc_io_bw)

                    if capacity_ok and bw_src_ok and bw_dst_ok:
                        # Try to find a valid route
                        route = chip_graph.find_unused_path(
                            int_mem_chiplet.chiplet_loc,
                            target_weight_chiplet.chiplet_loc,
                            alloc_io_bw
                        )
                        if route is not None:
                            break  # All checks passed
                        # else:
                        #     logger.warning(f"No unused path found from {int_mem_chiplet.chiplet_id} to {target_weight_chiplet.chiplet_id}. Retrying...")
                    # Wait and retry next simulation tick
                    yield env.timeout(common.simulation_clk)

                # Update model state after successful check
                item.chip_id = target_weight_chiplet.chiplet_id
                chip_graph.load_bw_byid(target_weight_chiplet.chiplet_id, alloc_io_bw, env.now)
                chip_graph.load_bw_byid(int_mem_chiplet.chiplet_id, alloc_io_bw, env.now)
                # Transmit the data
                chip_graph.reserve_path(route, alloc_io_bw, env.now)
                mem_sys.load_data_byid(target_weight_chiplet.chiplet_id, item, env.now)
                # Calculate the communication latency
                
                # convert to ids if chip_graph.is_2d_grid() is false
                if chip_graph.is_2d_grid():
                    src_id = route[0][0]
                    dst_id = route[0][1]
                else:
                    src_id = chip_graph.coord_to_id[route[0][0]] if isinstance(route[0][0], tuple) else route[0][0]
                    dst_id = chip_graph.coord_to_id[route[0][1]] if isinstance(route[0][1], tuple) else route[0][1]

                comm_latency = len(route)* chip_graph.graph.edges[src_id, dst_id]['latency(ns)'] / 10E9
                transf_latency = ((item.size / (alloc_io_bw*1024)) + comm_latency + 50*10E-9) * common.time_granularity  # Convert to base time unit #TODO
                transfer_done = self.execution_backend.start_transfer(
                    env,
                    int_mem_chiplet.chiplet_id, target_weight_chiplet.chiplet_id,
                    item.size, alloc_io_bw, transf_latency,
                )
                yield transfer_done
                # Release the path and bandwidth after transmission
                chip_graph.release_path(route, alloc_io_bw, env.now)
                chip_graph.offload_bw_byid(target_weight_chiplet.chiplet_id, alloc_io_bw, env.now)
                chip_graph.offload_bw_byid(int_mem_chiplet.chiplet_id, alloc_io_bw, env.now)

                # Release the capacity of the input mem chiplet
                mem_sys.offload_data_byid(int_mem_chiplet.chiplet_id, item, env.now)
            # input_cache.clear()
            if self.infs_to_process[0].type == 'prefill' and self.blocks_to_process[0].block_num == 0:
                if self.input_cache != []:
                    raise ValueError(f"Input cache is not empty for prefill block {block_id}.")
                # input embedding always same size as output embedding
                for request in self.requests[:]:
                    input_intermediate = intermediate(size=common.convert_param_mB(self.blocks_to_process[0].output_act), block_id=block_id,
                                                    name=f"Block {block_id} input", chip_id=target_weight_chiplet.chiplet_id, req_id=request._id)
                    self.input_cache.append(input_intermediate)
                    # TODO: check the capacity
                    if not target_weight_chiplet.is_capacity_available(input_intermediate.size):
                        logger.warning(f"Target chiplet {target_weight_chiplet.chiplet_id} does not have enough capacity for input intermediate. preempt the request...")
                        self.drop_request_fromsystem(request._id, mem_sys, env.now)
                        continue
                    mem_sys.load_data_byid(target_weight_chiplet.chiplet_id, input_intermediate, env.now)  # load the input embedding into the target chiplet

            block_config = self.blocks_to_process[0].block_config
            block_type = block_config.get_accelerators(self.infs_to_process[0].type)
            block_types_t = [block_config.get_accelerators(stage) for stage in ('prefill', 'decode')]

            selected_C_chiplet = None
            route = None  
            set_priority = False #TODO: priority control
            congest_count = 0

            if self.infs_to_process[0].type == 'prefill':
                select_bw = utils.used_bws[0]
            else:
                if utils.used_bws[1] <= utils.NoI_bw:
                    select_bw = utils.used_bws[1]
                else:
                    select_bw = utils.used_bws[0]

            selected_C_chiplet, route = yield from select_task_slot(
                owner=self,
                env=env,
                chip_graph=chip_graph,
                comp_sys=comp_sys,
                block_type=block_type,
                fallback_types=block_types_t,
                block_id=block_id,
                infer_type=self.infs_to_process[0].type,
                target_weight_chiplet=target_weight_chiplet,
                select_bw=select_bw,
            )

            """
            Step 5, Allocate the memory for intermediates and states cache
            """
            temp_int_mems = []
            drop_number = 0
            for idx, block_to_process in enumerate(self.blocks_to_process[:]):
                int_load = block_to_process.peak_intermediate_store # total intermediate memory required for the block
                idx = idx - drop_number
                req_id = self.requests[idx]._id
                # Either prefill or 1st decode step
                if (self.infs_to_process[idx].type == 'prefill') or \
                    (self.requests[idx].is_only_decode() and self.requests[idx]._process_idx == 0):
                    # new states to load
                    # self.states_cache[req_id].append(intermediate(size=common.convert_param_mB(block_to_process.states_store), block_id=block_id, name=f"Block {block_id} states cache",
                    #                             chip_id=target_weight_chiplet.chiplet_id, req_id=req_id))# states cache
                    states_cache_vol = block_to_process.states_store # total states memory required for the block (kv cache, states, etc.)
                    total_inc_mem = common.convert_param_mB(int_load + states_cache_vol)
                else:
                    # update the states cache
                    pre_states_cache = self.states_cache[req_id][block_id]
                    # self.states_cache[req_id][block_id] = intermediate(size=common.convert_param_mB(block_to_process.states_store), block_id=block_id, name=f"Block {block_id} states cache",
                    #                             chip_id=target_weight_chiplet.chiplet_id, req_id=req_id)
                    total_inc_mem = block_to_process.additional_memory_mib(pre_states_cache.size)
                if not target_weight_chiplet.is_capacity_available(total_inc_mem):
                # if idx == 2:
                    logger.warning(f"Target chiplet {target_weight_chiplet.chiplet_id} does not have enough capacity for intermediates and states cache. preempt the request...")
                    self.drop_request_fromsystem(req_id, mem_sys, env.now)
                    drop_number += 1
                    # TODO: feedback to request scheduler
                    continue
                # Update the state cache
                if (self.infs_to_process[idx].type == 'prefill') or \
                    (self.requests[idx].is_only_decode() and self.requests[idx]._process_idx == 0):
                    self.states_cache[req_id].append(intermediate(size=common.convert_param_mB(block_to_process.states_store), block_id=block_id, name=f"Block {block_id} states cache",
                                            chip_id=target_weight_chiplet.chiplet_id, req_id=req_id))# states cache
                else:
                    self.states_cache[req_id][block_id] = intermediate(size=common.convert_param_mB(block_to_process.states_store), block_id=block_id, name=f"Block {block_id} states cache",
                                            chip_id=target_weight_chiplet.chiplet_id, req_id=req_id)
                # Allocate the intermediate (acts...)
                temp_int_mem = intermediate(size=common.convert_param_mB(int_load), block_id=block_id, name=f"Block {block_id} intermediate", chip_id=target_weight_chiplet.chiplet_id, req_id=req_id)
                temp_int_mems.append(temp_int_mem)
                mem_sys.load_data_byid(target_weight_chiplet.chiplet_id, temp_int_mem, env.now)
                # Refresh the states cache (kv cache, etc.)
                if (self.infs_to_process[idx].type == 'prefill') or \
                    (self.requests[idx].is_only_decode() and self.requests[idx]._process_idx == 0):
                    mem_sys.load_data_byid(target_weight_chiplet.chiplet_id, self.states_cache[req_id][block_id], env.now)
                else:
                    mem_sys.refresh_states_byid(target_weight_chiplet.chiplet_id, pre_states_cache, self.states_cache[req_id][block_id], env.now)

            """
            Step 6, comp latency, run the comp chiplet
            """
            self.execution_backend.record_package_block(
                selected_C_chiplet.chiplet_id, block_id, [r._id for r in self.requests],
                self.infs_to_process[0].type, 'start')
            block_done, avg_util = self.execution_backend.start_block(
                env,
                self.blocks_to_process[0], self.infs_to_process[0].type,
                len(self.requests), selected_C_chiplet,
                target_weight_chiplet.chiplet_id, select_bw,
            )
            comp_sys.set_used_byid(chiplet_id=selected_C_chiplet.chiplet_id, util=avg_util, power=30.0, time=env.now)
            yield block_done
            self.execution_backend.record_package_block(
                selected_C_chiplet.chiplet_id, block_id, [r._id for r in self.requests],
                self.infs_to_process[0].type, 'complete')
            comp_sys.set_free_byid(chiplet_id=selected_C_chiplet.chiplet_id, time=env.now)  # set the chiplet to idle after processing
            
            for item in self.input_cache:
                mem_sys.offload_data_byid(target_weight_chiplet.chiplet_id, item, env.now)
            self.input_cache.clear()
            
            """
            Step 7, release the routing, memory resource (intermediate/states) and bandwidth, save the output intermediates
            """
            chip_graph.release_path(route, select_bw, env.now)
            chip_graph.offload_bw_byid(selected_C_chiplet.chiplet_id, select_bw, env.now)
            chip_graph.offload_bw_byid(target_weight_chiplet.chiplet_id, select_bw, env.now)
            for temp_int_mem in temp_int_mems:
                mem_sys.offload_data_byid(target_weight_chiplet.chiplet_id, temp_int_mem, env.now)
            self.check_members_length()
            requests_status = [False for _ in self.requests]
            completed_inferences = []
            for index in range(len(self.requests)):
                blk_process_status = self.infs_to_process[index].step_processing()
                completed_inferences.append(blk_process_status)
                inf_process_status = False
                if blk_process_status == True:
                    # update the monitor
                    tokens_monitor.inc_output_tokens()
                    common.total_dc_tokens += 1
                    self._cb_real_total_tokens += 1
                    
                    inf_process_status = self.requests[index].step_processing(env.now)
                    if inf_process_status == True:
                        requests_status[index] = True
            requests_status = self._force_finish_when_longest_done(requests_status)
            pop_indexs = []    
            for index, request_status in enumerate(requests_status[:]):
                if request_status == False:
                    output_intermediate = intermediate(size=common.convert_param_mB(block_to_process.output_act), block_id=block_id, name=f"Block {block_id} output", chip_id=target_weight_chiplet.chiplet_id, req_id=self.requests[index]._id)
                    if completed_inferences[index] and getattr(self.execution_backend, 'package_network', None) is not None:
                        # The next autoregressive iteration consumes a token ID,
                        # not the entire previous prefill activation sequence.
                        output_intermediate.size = self.execution_backend.config.package_config.token_bytes / 1024**2
                        output_intermediate.name = f'Block {block_id} token feedback'
                    # TODO: check the capacity
                    if not target_weight_chiplet.is_capacity_available(output_intermediate.size):
                        logger.warning(f"Target chiplet {target_weight_chiplet.chiplet_id} does not have enough capacity for output intermediate. preempt the request...")
                        self.drop_request_fromsystem(self.requests[index]._id, mem_sys, env.now)
                        continue
                    mem_sys.load_data_byid(target_weight_chiplet.chiplet_id, output_intermediate, env.now) # it would be released when transferred to new chiplet
                    self.input_cache.append(output_intermediate)
                    self.infs_to_process[index] = self.requests[index]._infs['prefill'] if (self.requests[index]._process_idx == 0 and self.requests[index].has_prefill()) else self.requests[index]._infs['decode']
                    self.blocks_to_process[index] = self.infs_to_process[index].blocks[self.infs_to_process[index].process_idx]
                else:
                    # release all states
                    if self.requests[index].has_decode():
                        tokens_monitor.inc_finished_tokens(self.requests[index]._infs['decode'].max_decoding_length+1) # TODO: only for decode inf
                    elif self.requests[index].has_prefill():
                        tokens_monitor.inc_finished_tokens(1)
                    for state in self.states_cache[self.requests[index]._id]:
                        assert isinstance(state, intermediate), "State is not of type intermediate."
                        mem_sys.offload_data_byid(state.chip_id, state, env.now)
                    # no output intermediate
                    # input_cache.clear()
                    # states_cache.clear()
                    # common.RequestQueue.running.remove(current_request)
                    # common.RequestQueue.completed.append(current_request)
                    for block_name, allocated_memory in common.req_allocated_mems[self.requests[index]._id]:
                        mem_sys.relieve_allocate_memchiplet(allocated_memory, block_name)
                    del common.req_allocated_mems[self.requests[index]._id]
                        
                    monitor.request_counter.decrement_running()
                    monitor.request_counter.increment_completed()
                    self.requests[index].on_completion(env.now)
                    pop_indexs.append(index)
            # clean up finished requests
            for index in sorted(pop_indexs, reverse=True):
                del self.states_cache[self.requests[index]._id]
                common.total_pf_tokens -= self.requests[index].get_prefill_length()
                common.total_dc_tokens -= self.requests[index].get_decode_length()
                self.requests.pop(index)
                self.infs_to_process.pop(index)
                self.blocks_to_process.pop(index)
            batch_size = len(self.requests)

            prefill_penalty_ticks = self._admit_virtual_continuous_slots()
            if prefill_penalty_ticks > 0:
                # New-admission prefill overhead before virtual streams decode.
                yield env.timeout(prefill_penalty_ticks)
            self._advance_virtual_continuous_slots()
        
            if all(requests_status):
                batch_done = True
                self._flush_virtual_continuous_slots()
                if len(self.requests) != 0 or batch_size != 0:
                    raise ValueError("Current requests should be empty when batch is done.")
                common.RequestQueue.remove_running_batch(self.current_batch._id)
                common.RequestQueue.completed.append(self.current_batch)
                monitor.request_counter.decrement_running_batches()
                logger.info(f"Batch {self.current_batch._id} completed at time {env.now}.")
                self.reset_processing()
                # mem_sys._current_allocated -= common.allocate_mem
    
    def run_tp(self, env: simpy.Environment, chip_graph: chips_network.chip_graph, mem_sys: mem_sys.mem_sys, comp_sys: comp_sys.comp_sys):
        self.IDLE = False
        while(True):
            if len(common.RequestQueue.running) != 0:
                self.requests = [req for batch in common.RequestQueue.running if batch._id == self.batch_id for req in batch.ongoing_requests]
                if self.requests:
                    break
            # logger.warning(f"Request {self.request_id} not found in running requests. Waiting ...")
            yield env.timeout(common.simulation_clk)
        
        self.current_batch = [batch for batch in common.RequestQueue.running if batch._id == self.batch_id][0]
        self.infs_to_process = [inf for req in self.requests for inf in req._infs.values() if inf.process_idx == req._process_idx] 
        self.blocks_to_process = [inf.blocks[inf.process_idx] for inf in self.infs_to_process]
        batch_done = False
        batch_size = len(self.requests)
        self.input_cache = {req._id: [] for req in self.requests} # intermediate activations
        self.states_cache = {req._id: [] for req in self.requests} # KV and state cachees
        pre_states_cache = None
        self._init_continuous_batching()

        while not batch_done:
            """
            Step 0: find the HBMs that store the required model params
            """
            block_id = self.blocks_to_process[0].block_num
            weight_Mem_id_list = mem_sys.blocks_alloc[block_id]
            target_weight_chiplet_list = [mem_sys.get_mem_byid(mem_id) for mem_id in weight_Mem_id_list]

            # identify the workload type #TODO
            block_type = self.blocks_to_process[0].block_config.type_name
            # Hybrid LLM only Mamba/Transformer's P or D
            # TODO: do this more elegantly
            if 'transformer' in block_type:
                block_type = [2,4] if self.infs_to_process[0].type == 'prefill' else [3]
            elif 'mamba' in block_type:
                block_type = [0,4] if self.infs_to_process[0].type == 'prefill' else [1]
            else:
                raise ValueError(f"Unknown block type {block_type} for block {block_id}.")

            selected_C_chiplet = None
            route = None  
            set_priority = False #TODO: priority control
            """
            Step 1: Find compute chiplets
            """
            select_bw = mem_sys.total_mem_bw // comp_sys.total_comp_chiplets
            comp_sys_chiplets = []
            while True:
                comp_sys_chiplets = comp_sys.find_all_available_chiplets(chip_graph, select_bw)

                if len(comp_sys_chiplets) > 0:
                    # set all compute chiplets busy
                    for c_chiplet in comp_sys_chiplets:
                        comp_sys.set_busy_byid(c_chiplet.chiplet_id)
                        chip_graph.load_bw_byid(c_chiplet.chiplet_id, select_bw, env.now)
                    break  # found available compute chiplets
                
                yield env.timeout(common.simulation_clk)

            mem_bw_per_mem_chiplet = mem_sys.total_mem_bw // len(target_weight_chiplet_list)
            for mem_chiplet in target_weight_chiplet_list:
                chip_graph.load_bw_byid(mem_chiplet.chiplet_id, mem_bw_per_mem_chiplet, env.now)

            # now comp_sys_chiplets should not be empty at time
            assert len(comp_sys_chiplets) > 0, f"No available compute chiplets found at time {env.now} {comp_sys_chiplets} batch id {self.batch_id}."

            """
            Step 2: Allocate the memory for intermediates and states cache
            """
            temp_int_mems = []
            drop_number = 0
            for idx, block_to_process in enumerate(self.blocks_to_process[:]):
                int_load = block_to_process.peak_intermediate_store
                idx = idx - drop_number
                req_id = self.requests[idx]._id
                # Either prefill or 1st decode step
                if (self.infs_to_process[idx].type == 'prefill') or \
                    (self.requests[idx].is_only_decode() and self.requests[idx]._process_idx == 0):

                    states_cache_vol = block_to_process.states_store
                    total_inc_mem = common.convert_param_mB(int_load + states_cache_vol)
                else:
                    pre_states_cache = deepcopy(self.states_cache[req_id][block_id])
                    total_pre_state_size = sum([state.size for state in pre_states_cache])
                    if block_to_process.states_store > total_pre_state_size:
                        states_cache_vol = block_to_process.states_store - total_pre_state_size
                    total_inc_mem = common.convert_param_mB(int_load + states_cache_vol)

                # check all target chiplets for capacity
                total_inc_mem_per_chiplet = total_inc_mem // len(target_weight_chiplet_list)
                capacity_ok = all([chiplet.is_capacity_available(total_inc_mem_per_chiplet) for
                   chiplet in target_weight_chiplet_list])
                if not capacity_ok:
                    logger.warning(f"Target chiplets do not have enough capacity for intermediates and states cache. preempt the request...")
                    self.drop_request_fromsystem(req_id, mem_sys, env.now)
                    drop_number += 1
                    continue
                # Update the state cache
                if (self.infs_to_process[idx].type == 'prefill') or \
                    (self.requests[idx].is_only_decode() and self.requests[idx]._process_idx == 0):
                    # add 2D list for each block and each chiplet
                    if len(self.states_cache[req_id]) <= block_id:
                        self.states_cache[req_id].append([])
                    for chiplet in target_weight_chiplet_list:
                        self.states_cache[req_id][block_id].append(
                            intermediate(size=common.convert_param_mB(block_to_process.states_store) // len(target_weight_chiplet_list),
                                         block_id=block_id,
                                         name=f"Block {block_id} states cache",
                                         chip_id=chiplet.chiplet_id,
                                         req_id=req_id)
                        )
                else:
                    for i, chiplet in enumerate(target_weight_chiplet_list):
                        self.states_cache[req_id][block_id][i] = intermediate(
                            size=common.convert_param_mB(block_to_process.states_store) // len(target_weight_chiplet_list),
                            block_id=block_id,
                            name=f"Block {block_id} states cache",
                            chip_id=chiplet.chiplet_id,
                            req_id=req_id
                        )
                
                # Allocate the intermediate (acts...)
                temp_int_mem = intermediate(size=common.convert_param_mB(int_load) // len(target_weight_chiplet_list),
                                           block_id=block_id,
                                           name=f"Block {block_id} intermediate",
                                           chip_id=chiplet.chiplet_id,
                                           req_id=req_id)
                temp_int_mems.append(temp_int_mem)
                for chiplet in target_weight_chiplet_list:
                    mem_sys.load_data_byid(chiplet.chiplet_id, temp_int_mem, env.now)
                # Refresh the states cache (kv cache, etc.)
                if (self.infs_to_process[idx].type == 'prefill') or \
                    (self.requests[idx].is_only_decode() and self.requests[idx]._process_idx == 0):

                    for i, chiplet in enumerate(target_weight_chiplet_list):
                        mem_sys.load_data_byid(chiplet.chiplet_id,
                                              self.states_cache[req_id][block_id][i],
                                              env.now)
                else:
                    for i, chiplet in enumerate(target_weight_chiplet_list):
                        mem_sys.refresh_states_byid(
                            chiplet.chiplet_id,
                            pre_states_cache[i],
                            self.states_cache[req_id][block_id][i],
                            env.now
                        )
            
            """
            Step 3, comp latency, run the comp chiplets
            """
            total_latency = 0.0
            avg_util = []
            layer_index = 0
            # blocks to process are same in the batch
            for layer_config, layer_name in zip(self.blocks_to_process[0].layers_configs,self.blocks_to_process[0].layers):
                assert len(layer_config) == len(layer_name), "Layer configuration and layer names must match in length."
                assert isinstance(self.blocks_to_process[0], block), "Invalid block configuration."
                # computation latency

                first_layer = layer_index == 0
                last_layer = layer_index == len(self.blocks_to_process[0].layers) - 1
                layer_index += 1

                for idx, item in enumerate(layer_config):
                    name = layer_name[idx]
                    if name in ['MHA','SSM']: # prefill and decode mode
                        if self.infs_to_process[0].type == 'prefill':
                            name += '_p'
                        else:
                            name += '_d'

                    assert isinstance(item, baselayerConfig), "Invalid layer configuration."

                    assert len(comp_sys_chiplets) > 0, f"No compute chiplets available for processing at time {env.now} batch id {self.batch_id}."

                    # if all the compute chiplets are of the same type, we can use the first one
                    logic_name = utils.chiplet_types_list[comp_sys_chiplets[0].chiplet_type] if all(
                        c_chiplet.chiplet_type == comp_sys_chiplets[0].chiplet_type for c_chiplet in comp_sys_chiplets) \
                        else ValueError("Compute chiplets have different types, not supported yet.")
                    analytic_profiles:BaseAccModel = BaseAccModel.create_from_name(logic_name)
                    func_args = analytic_profiles.config_params(**vars(item))
                    batch_size = len(self.requests)
                    # adjust the args for prefill/decode
                    if self.infs_to_process[0].type == 'prefill':
                        func_args['bs'] = self.blocks_to_process[0].context_length # for kernels aside from mha and ssm
                        func_args['batch_size'] = batch_size
                        func_args['L_seq'] = self.blocks_to_process[0].context_length
                        if 'in_size' in func_args:
                            func_args['in_size'] *= self.blocks_to_process[0].context_length
                    else:
                        func_args['bs'] = 1
                        func_args['batch_size'] = batch_size
                        func_args['L_seq'] = self.blocks_to_process[0].context_length
                    func_args['logic_name'] = logic_name
                    func_args['ext_bw'] = select_bw
                    func_args['concurrent_chiplets'] = len(comp_sys_chiplets)
                    func_args['first_layer'] = first_layer
                    func_args['last_layer'] = last_layer
                    analytic_res:analytics = analytic_profiles.get_kernel(name, **func_args)
                    total_latency += common.convert_analy_time(analytic_res.total_latency)
                    avg_util.append(analytic_res.utilization)
            # comp latency for one block, set utilization of the
            avg_util = np.mean(avg_util)
            for c_chiplet in comp_sys_chiplets:
                comp_sys.set_used_byid(chiplet_id=c_chiplet.chiplet_id, util=avg_util, power=30.0, time=env.now)
            yield env.timeout(int(total_latency))
            for c_chiplet in comp_sys_chiplets:
                comp_sys.set_free_byid(chiplet_id=c_chiplet.chiplet_id, time=env.now)
            
            for req in self.requests:
                for chiplet_input in self.input_cache[req._id]:
                    assert isinstance(chiplet_input, intermediate), "Input is not of type intermediate."
                    mem_sys.offload_data_byid(chiplet_input.chip_id, chiplet_input, env.now)
            self.input_cache = {req._id: [] for req in self.requests}
            

            # No input cache in TP, since all intermediates are distributed across chiplets
            """
            Step 4, release the routing, memory resource (intermediate/states) and bandwidth, save the output intermediates
            """
            # we didn't reserve any routing or bandwidth in TP for simplicity
            for c_chiplet in comp_sys_chiplets:
                chip_graph.offload_bw_byid(c_chiplet.chiplet_id, select_bw, env.now)
            for temp_int_mem in temp_int_mems:
                for chiplet in target_weight_chiplet_list:
                    mem_sys.offload_data_byid(chiplet.chiplet_id, temp_int_mem, env.now)
            for mem_chiplet in target_weight_chiplet_list:
                chip_graph.offload_bw_byid(mem_chiplet.chiplet_id, mem_bw_per_mem_chiplet, env.now)
            self.check_members_length()
            requests_status = [False for _ in self.requests]
            for index in range(len(self.requests)):
                blk_process_status = self.infs_to_process[index].step_processing()
                inf_process_status = False
                if blk_process_status == True:
                    # update the monitor
                    tokens_monitor.inc_output_tokens()
                    self._cb_real_total_tokens += 1
                    inf_process_status = self.requests[index].step_processing(env.now)
                    if inf_process_status == True:
                        requests_status[index] = True
            requests_status = self._force_finish_when_longest_done(requests_status)
            pop_indexs = []    
            for index, request_status in enumerate(requests_status[:]):
                if request_status == False:
                    output_intermediate = []
                    for i, chiplet in enumerate(target_weight_chiplet_list):
                        output_intermediate.append(intermediate(size=common.convert_param_mB(
                            block_to_process.output_act) // len(target_weight_chiplet_list),
                                                        block_id=block_id,
                                                        name=f"Block {block_id} output",
                                                        chip_id=chiplet.chiplet_id,
                                                        req_id=self.requests[index]._id))
                    check_capacity = all([chiplet.is_capacity_available(output_intermediate[i].size) for i, chiplet in enumerate(target_weight_chiplet_list)])
                    if not check_capacity:
                        logger.warning(f"Target chiplets do not have enough capacity for output intermediate. preempt the request...")
                        self.drop_request_fromsystem(self.requests[index]._id, mem_sys, env.now)
                        continue
                    for i, chiplet in enumerate(target_weight_chiplet_list):
                        mem_sys.load_data_byid(chiplet.chiplet_id, output_intermediate[i], env.now)
                        self.input_cache[self.requests[index]._id].append(output_intermediate[i])
                    # Not using input_cache in TP
                    self.infs_to_process[index] = self.requests[index]._infs['prefill'] if (self.requests[index]._process_idx == 0 and self.requests[index].has_prefill()) else self.requests[index]._infs['decode']
                    self.blocks_to_process[index] = self.infs_to_process[index].blocks[self.infs_to_process[index].process_idx]
                else:
                    # release all states
                    if self.requests[index].has_decode():
                        tokens_monitor.inc_finished_tokens(self.requests[index]._infs['decode'].max_decoding_length+1)
                    elif self.requests[index].has_prefill():
                        tokens_monitor.inc_finished_tokens(1)
                    for block_cache in self.states_cache[self.requests[index]._id]:
                        for chiplet_state in block_cache:
                            assert isinstance(chiplet_state, intermediate), "State is not of type intermediate."
                            mem_sys.offload_data_byid(chiplet_state.chip_id, chiplet_state, env.now)
                    
                    for block_name, allocated_memory in common.req_allocated_mems[self.requests[index]._id]:
                        mem_sys.relieve_allocate_memchiplet(allocated_memory, block_name)
                    del common.req_allocated_mems[self.requests[index]._id]

                    monitor.request_counter.decrement_running()
                    monitor.request_counter.increment_completed()
                    self.requests[index].on_completion(env.now)
                    pop_indexs.append(index)
            # clean up finished requests
            for index in sorted(pop_indexs, reverse=True):
                del self.states_cache[self.requests[index]._id]
                del self.input_cache[self.requests[index]._id]
                self.requests.pop(index)
                self.infs_to_process.pop(index)
                self.blocks_to_process.pop(index)
            batch_size = len(self.requests)

            prefill_penalty_ticks = self._admit_virtual_continuous_slots()
            if prefill_penalty_ticks > 0:
                # New-admission prefill overhead before virtual streams decode.
                yield env.timeout(prefill_penalty_ticks)
            self._advance_virtual_continuous_slots()

            if all(requests_status):
                batch_done = True
                self._flush_virtual_continuous_slots()
                if len(self.requests) != 0 or batch_size != 0:
                    raise ValueError("Current requests should be empty when batch is done.")
                common.RequestQueue.remove_running_batch(self.current_batch._id)
                common.RequestQueue.completed.append(self.current_batch)
                monitor.request_counter.decrement_running_batches()
                logger.info(f"Batch {self.current_batch._id} completed at time {env.now}.")
                self.reset_processing()

    def drop_request_fromsystem(self, req_id, mem_sys: mem_sys.mem_sys, timestep: int):
        # clear the mem_sys
        logger.warning(f"Request {req_id} is dropped from the system at time {timestep}.")
        offload_flag = False
        for mem_chiplet_inst in mem_sys._mem_chiplets:
            assert isinstance(mem_chiplet_inst, mem_chiplet), "Chiplet is not of type mem_chiplet."
            if req_id not in mem_chiplet_inst.content_params.keys():
                continue
            for data in list(mem_chiplet_inst.content_params[req_id]):
                if not isinstance(data, intermediate):
                    raise ValueError(f"Data is not of type intermediate: {data}")
                if data.req_id != req_id:
                    raise ValueError(f"Data req_id {data.req_id} does not match the dropping req_id {req_id}.")
                    # drop all data related to the request (kv cache, input output intermediate, etc.)
                mem_sys.offload_data_byid(mem_chiplet_inst.chiplet_id, data, timestep)
                offload_flag = True
        if not offload_flag:
            logger.warning(
                f"Request {req_id} has no data in mem chiplets at drop time {timestep}; "
                "continuing lifecycle cleanup."
            )
        matched_requests = [
            req for batch in common.RequestQueue.running for req in batch.ongoing_requests if req._id == req_id
        ]
        if len(matched_requests) == 0:
            logger.warning(f"Request {req_id} not found in running batches during drop cleanup.")
            return
        temp_request:Request = matched_requests[0]
        # clear the current workthread's status
        # Keep request/infer/block arrays aligned by removing the same index.
        if self.requests is not None:
            keep_indices = [i for i, req in enumerate(self.requests) if req._id != req_id]
            self.requests = [self.requests[i] for i in keep_indices]

            if self.infs_to_process is not None:
                self.infs_to_process = [self.infs_to_process[i] for i in keep_indices if i < len(self.infs_to_process)]
            if self.blocks_to_process is not None:
                self.blocks_to_process = [self.blocks_to_process[i] for i in keep_indices if i < len(self.blocks_to_process)]
        else:
            if self.infs_to_process is not None:
                self.infs_to_process = [inf for inf in self.infs_to_process if getattr(inf, "_req_id", None) != req_id]
            if self.blocks_to_process is not None:
                self.blocks_to_process = [blk for blk in self.blocks_to_process if getattr(blk, "_req_id", None) != req_id]
        if self.states_cache is not None:
            if req_id in self.states_cache:
                del self.states_cache[req_id]
        if self.input_cache is not None:
            if isinstance(self.input_cache, list):
                self.input_cache = [item for item in self.input_cache if getattr(item, "req_id", None) != req_id]
            elif isinstance(self.input_cache, dict):
                self.input_cache.pop(req_id, None)
            
        # update the monitor and task queue
        # current_batch:BatchOfRequests = [batch for batch in common.RequestQueue.running if temp_request in batch.ongoing_requests][0]   
        # remove the request from the current batch
        copy_request:Request = temp_request.retry_request()
        if self.current_batch is not None:
            self.current_batch.remove_ongoing_request(temp_request._id)
        common.RequestQueue.outstanding.append(copy_request)
        monitor.request_counter.decrement_running()
        monitor.request_counter.increment_pending()
        # monitor.tokens_monitor.dec_output_tokens(copy_request._process_idx)
        common.preempted_requests[req_id] = copy_request._process_idx
        copy_request.reset_request()
        self.check_members_length()

    # for debug
    def check_members_length(self):
        len_requests = len(self.requests) if self.requests is not None else 0
        len_infs = len(self.infs_to_process) if self.infs_to_process is not None else 0
        len_blocks = len(self.blocks_to_process) if self.blocks_to_process is not None else 0
        len_states = len(self.states_cache) if self.states_cache is not None else 0
        len_input_cache = len(self.input_cache) if self.input_cache is not None else 0
        if not (len_requests == len_infs == len_blocks == len_states):
            raise ValueError(f"Length mismatch: requests({len_requests}), infs({len_infs}), blocks({len_blocks}), states({len_states}), input_cache({len_input_cache})")
