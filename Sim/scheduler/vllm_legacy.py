import simpy
import numpy as np

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
from Sim.config.model_config import BaseModelConfig, BaseBlockConfig
from Sim.scheduler.BaseReqScheduler import BaseReqScheduler
from Sim.entities.mem_chiplet import Params
import Sim.config.utils as utils
from Sim.metrics.monitor import congestion_monitor


class VllmLegacyReplicaScheduler(BaseReqScheduler):
    """
    Dynamic request scheduler inspired by paged-attention style serving.

    In this framework, "dynamic batching" admission has two layers:
    1) Request-count controller (`_current_bs_est`) with congestion feedback.
    2) Token-budget controller (`_token_budget_est`) that caps the number of
       requests admitted this step based on request token cost, similar in
       spirit to vLLM's max batched tokens.

    """

    def __init__(self, chip_graph: chips_network.chip_graph, mem_sys: mem_sys.mem_sys, comp_sys: comp_sys.comp_sys, bs=1):
        # Upper bound on concurrently running batches from system resources.
        self._max_concurrent_batch = min(common.max_requests_in_parallel, comp_sys._total_budget, mem_sys._num_io_limit)

        self._max_num_batch_tokens = -1  # reserved for future use
        self._max_concurrent_requests = 9999

        # Dynamic estimate of request-count batch size (AIMD-like controller).
        self._current_bs_est = max(1, int(bs))

        self._get_eff_bw(mem_sys)

        # Optional old context-bytes guard (coarse model); keep disabled by default.
        self.enable_context_bw_guard = False

        # Throughput-priority mode keeps admission close to configured batchsize.
        # This is the default for DSE throughput studies.
        self.throughput_priority_mode = True
        # Allow bounded expansion over configured batch size in throughput mode.
        self.throughput_bs_boost = 1.45

        # Token-budget admission is optional. In throughput mode we keep it off by
        # default to avoid frequent under-filled dispatches.
        self.enable_token_budget_admission = True

        # Minimum queue fill ratio (vs configured batchsize) before dispatch while
        # trace injection is still ongoing.
        self.min_start_batch_ratio = 0.75
        self.min_start_batch_ratio_cold = 0.5

        # Only a bounded decode contribution is counted in admission token cost,
        # so extremely long decode tails do not fully block admission.
        self.decode_window_tokens = 96

        # Runtime token budget estimate (updated online from queue stats).
        self._token_budget_est: int | None = None
        self._min_token_budget: int = 1
        self._max_token_budget: int = int(1e9)

    def set_max_concurrent_requests(self, model_config: BaseModelConfig, mem_sys: mem_sys.mem_sys) -> None:
        """
        Compute a conservative upper bound of concurrent requests from memory.

        Same principle as static scheduler, but with a relaxed KV factor (1/4)
        to mimic paged-cache sharing behavior.
        """
        self.model_config = model_config

        for mem_chiplet_inst in mem_sys._mem_chiplets:
            assert isinstance(mem_chiplet_inst, mem_chiplet), "mem_chiplet_inst is not an instance of mem_chiplet."
            left_budget = (mem_chiplet_inst.dram_budget - mem_chiplet_inst.inuse_budget) * 1024 * 1024  # Bytes

            mem_req_per_request = 0
            for block_params in mem_chiplet_inst.content_params.get('weights', []):
                assert isinstance(block_params, Params), "block_params is not an instance of Params."
                type_idx = model_config.block_type_sequence[block_params.block_id]
                block_config = model_config.hybrid_blocks[type_idx]
                if block_config.cache_store:
                    mem_req_per_request += (
                        block_config.states * model_config.max_position_embeddings * (1 / 4)
                        + block_config.peak_intermed
                    )
                else:
                    mem_req_per_request += (block_config.states + block_config.peak_intermed)

            max_requests = left_budget // mem_req_per_request
            if max_requests < self._max_concurrent_requests:
                self._max_concurrent_requests = int(max_requests)

        # Initialize token-budget bounds after model is known.
        # Upper bound follows configured batch size and model max context.
        self._max_token_budget = max(
            1, int(common.batch_size * self.throughput_bs_boost * self.model_config.max_position_embeddings)
        )

        # Lower bound allows at least one request to be admitted.
        # Keep this small and let _cap_bs_by_token_budget force bs>=1 anyway.
        self._min_token_budget = 1

        # Start from a moderate token budget proportional to configured batch size.
        init_per_req = max(1, min(512, self.model_config.max_position_embeddings // 8))
        self._token_budget_est = max(
            1,
            min(self._max_token_budget, int(common.batch_size * self.throughput_bs_boost * init_per_req)),
        )

        print(f"Set max concurrent requests to {self._max_concurrent_requests} based on memory chiplet budgets.")

    def _get_eff_bw(self, mem_sys: mem_sys.mem_sys) -> float:
        """Estimate effective platform BW budget used by optional admission guards."""
        self.bw_budget = min(
            int(len(mem_sys._mem_chiplets) * utils.IO_bw),
            int(len(mem_sys._mem_chiplets) * 3 * utils.NoI_bw),
        )

    def _get_byte_per_token(self, context_lens=None) -> float:
        """
        Coarse bytes-per-token estimator (legacy guard).
        Kept for compatibility; token-budget admission does not depend on this.
        """
        if context_lens is not None:
            pf_length, dc_length = context_lens
        else:
            pf_length = common.total_pf_tokens
            dc_length = common.total_dc_tokens

        bytes_per_token = 0.0
        for block_type in self.model_config.block_type_sequence:
            block_config: BaseBlockConfig = self.model_config.hybrid_blocks[block_type]
            if block_config.cache_store:
                context_length = pf_length + dc_length
            else:
                context_length = 1

            num_job = monitor.request_counter.running_batches + 1 if context_lens is not None else monitor.request_counter.running_batches
            bytes_per_token += block_config.parameter_count * common.ByteperParam * num_job
            bytes_per_token += block_config.intermed_mem * num_job + (block_config.states * context_length)
        return bytes_per_token

    def _bs_context_length_threshold(self, total_pf: int, max_bs: int, target_tp=1) -> int:
        """
        Optional context-length bandwidth guard.
        This is intentionally off by default due model coarseness.
        """
        eff_bw = self.bw_budget * 1e9
        cur_bs = max_bs

        if common.total_dc_tokens < 0 or common.total_pf_tokens < 0:
            raise ValueError("Total prefill or decode tokens is negative.")

        while cur_bs > 0:
            post_bytes_per_token = self._get_byte_per_token(
                (common.total_pf_tokens + total_pf * cur_bs, common.total_dc_tokens)
            )
            use_bw = target_tp * post_bytes_per_token
            if use_bw <= eff_bw:
                break
            cur_bs -= 1

        return max(1, min(cur_bs, max_bs))

    def _request_token_cost(self, req: Request) -> int:
        """
        Token cost used by token-budget admission for one request.

        We count full prefill tokens and a bounded decode window to approximate
        "active token pressure" without letting very long decode tails dominate.
        """
        pf = req.get_prefill_length()
        dc = req.get_decode_length()
        return int(max(1, pf + min(dc, self.decode_window_tokens)))

    def _update_token_budget(self, sample_reqs: list[Request]) -> int:
        """
        Update token-budget estimate from current queue/system status.

        Target budget is proportional to configured batch size and median token
        cost in outstanding requests. We then adjust with congestion/occupancy
        factors and smooth via EMA for stability.
        """
        if len(sample_reqs) == 0:
            return int(self._token_budget_est or self._min_token_budget)

        costs = [self._request_token_cost(r) for r in sample_reqs]
        # Use a slightly lower percentile than median to reduce sensitivity to
        # long-tail requests and avoid under-admitting in high-throughput mode.
        median_cost = float(np.percentile(costs, 40))

        # Base target: what "batch_size requests" means in tokens for this queue.
        target = (common.batch_size * self.throughput_bs_boost) * median_cost

        # Back off under congestion and high worker occupancy.
        factor = 1.0
        if congestion_monitor.total_congested() > 100:
            factor *= 0.9

        if self._max_concurrent_batch > 0:
            occupancy = monitor.request_counter.running_batches / self._max_concurrent_batch
            if occupancy > 0.9:
                factor *= 0.92

        # When backlog is deep, bias budget upward to reduce starvation and
        # keep batches better filled.
        if len(common.RequestQueue.outstanding) >= max(32, 4 * common.batch_size):
            factor *= 1.1

        target = int(max(self._min_token_budget, min(self._max_token_budget, target * factor)))

        if self._token_budget_est is None:
            self._token_budget_est = target
        else:
            # Smooth to avoid oscillations.
            self._token_budget_est = int(0.6 * self._token_budget_est + 0.4 * target)

        self._token_budget_est = int(max(self._min_token_budget, min(self._max_token_budget, self._token_budget_est)))
        return self._token_budget_est

    def _cap_bs_by_token_budget(self, bs: int) -> int:
        """
        Convert token budget into a request-count cap for this scheduling step.
        """
        if not self.enable_token_budget_admission or bs <= 1:
            return max(1, bs)

        sample_n = min(len(common.RequestQueue.outstanding), max(bs * 2, 32))
        sample_reqs = common.RequestQueue.outstanding[:sample_n]
        budget = self._update_token_budget(sample_reqs)

        admitted = 0
        used_tokens = 0
        for req in common.RequestQueue.outstanding[:bs]:
            c = self._request_token_cost(req)
            if admitted > 0 and used_tokens + c > budget:
                break
            admitted += 1
            used_tokens += c

        return max(1, admitted)

    def _expand_bs_by_token_budget(self, baseline_bs: int, max_candidate_bs: int) -> int:
        """
        Throughput-oriented token-budget policy:
        - Never decreases baseline batch size.
        - Only increases batch size if token budget suggests it is safe.
        """
        if (not self.enable_token_budget_admission) or max_candidate_bs <= baseline_bs:
            return max(1, baseline_bs)

        sample_n = min(len(common.RequestQueue.outstanding), max(max_candidate_bs * 2, 32))
        sample_reqs = common.RequestQueue.outstanding[:sample_n]
        budget = self._update_token_budget(sample_reqs)

        admitted = 0
        used_tokens = 0
        for req in common.RequestQueue.outstanding[:max_candidate_bs]:
            c = self._request_token_cost(req)
            if admitted > 0 and used_tokens + c > budget:
                break
            admitted += 1
            used_tokens += c

        return max(1, max(baseline_bs, admitted))

    def _can_allocate_batch_memory(self, mem_sys: mem_sys.mem_sys, batch_reqs: list[Request]) -> tuple[bool, list[tuple[int, str, float]]]:
        """
        Try allocating memory for a candidate batch (same logic as schedule_request).
        Returns (ok, allocations).
        """
        batch_allocate_mem: list[tuple[int, str, float]] = []

        for incoming_req in batch_reqs:
            if incoming_req.has_prefill():
                inf_req: infer = incoming_req._infs["prefill"]
            else:
                inf_req: infer = incoming_req._infs["decode"]

            for block in inf_req.blocks:
                pf_memory = (block.peak_intermediate_store + block.states_store) if incoming_req.has_prefill() else 0
                dc_memory = (
                    block.peak_intermediate_store // block.context_length
                    + (
                        self.model_config.max_position_embeddings * (1 / 4) * block.block_config.states
                        if block.block_config.cache_store
                        else block.states_store
                    )
                ) if incoming_req.has_decode() else 0

                allocated_memory = max(pf_memory, dc_memory) * common.ByteperParam / (1024 * 1024)  # MB
                block_name = f'Block {block.block_num}'

                if mem_sys.check_memchiplets_availability(allocated_memory, block_name) == -1:
                    for _, bname, amem in batch_allocate_mem:
                        mem_sys.relieve_allocate_memchiplet(amem, bname)
                    return (False, [])
                else:
                    mem_sys.allocate_memchiplet(allocated_memory, block_name)
                    batch_allocate_mem.append((incoming_req._id, block_name, allocated_memory))

        return (True, batch_allocate_mem)

    def _choose_effective_batchsize(self, mem_sys: mem_sys.mem_sys, comp_sys: comp_sys.comp_sys, max_batchsize: int, pf_len: int) -> int:
        """
        Choose effective batch size <= max_batchsize based on:
        - queue depth
        - request-count capacity
        - token-budget capacity
        - memory feasibility (shrink-to-fit)
        """
        if max_batchsize <= 0:
            return 0

        if monitor.request_counter.running_batches >= self._max_concurrent_batch:
            return 0
        if comp_sys._current_run >= comp_sys._total_budget:
            return 0

        outstanding_n = len(common.RequestQueue.outstanding)
        if outstanding_n == 0:
            return 0

        remaining_reqs = self._max_concurrent_requests - monitor.request_counter.running_requests
        if remaining_reqs <= 0:
            return 0

        # Baseline is configured batch size (or queue/resource limits), then
        # token budget can optionally increase but never decrease it.
        bs = min(max_batchsize, outstanding_n, remaining_reqs)
        if bs <= 0:
            return 0

        # Optional coarse bandwidth guard.
        if self.enable_context_bw_guard:
            bs = min(bs, self._bs_context_length_threshold(total_pf=pf_len, max_bs=bs, target_tp=1))

        # Token budget can only expand (not shrink) admission in throughput mode.
        if self.enable_token_budget_admission:
            max_expand_bs = min(outstanding_n, remaining_reqs, max_batchsize * 2)
            bs = self._expand_bs_by_token_budget(bs, max_expand_bs)
        bs = max(1, bs)

        # Final shrink-to-fit with concrete memory checks.
        while bs > 0:
            candidate_reqs = common.RequestQueue.outstanding[:bs]
            ok, allocs = self._can_allocate_batch_memory(mem_sys, candidate_reqs)
            if ok:
                for req_id, block_name, allocated_memory in allocs:
                    common.req_allocated_mems.setdefault(req_id, []).append((block_name, allocated_memory))
                return bs
            bs -= 1

        return 0

    def schedule_request(self, mem_sys: mem_sys.mem_sys, comp_sys: comp_sys.comp_sys, batchsize=1) -> int:
        """
        Dynamic admission entry point.

        `batchsize` is the configured max request-count batch size from sys_config.
        We adapt around it; we do not always admit that many requests.
        """
        if len(common.RequestQueue.outstanding) == 0:
            return 0

        if monitor.request_counter.running_batches >= self._max_concurrent_batch:
            return -1
        if comp_sys._current_run >= comp_sys._total_budget:
            return -1

        queue_depth = len(common.RequestQueue.outstanding)

        if self.throughput_priority_mode:
            # Throughput-first policy: use as many requests as currently available,
            # up to a bounded expansion over configured batchsize, and avoid
            # shrinking from transient congestion.
            max_bs = max(1, int(np.ceil(batchsize * self.throughput_bs_boost)))
            self._current_bs_est = min(max_bs, max(1, queue_depth))
        else:
            # AIMD-like request-count controller.
            if congestion_monitor.total_congested() > 100:
                self._current_bs_est = max(1, self._current_bs_est - 2)
            else:
                inc_step = 2 if queue_depth >= (2 * self._current_bs_est) else 1
                self._current_bs_est = min(batchsize, self._current_bs_est + inc_step)

        # During injection, wait for a minimally filled batch to avoid very small
        # dispatches that hurt throughput.
        if not common.inj_finish:
            # Cold start: don't over-wait for a large initial fill, otherwise
            # we can lose throughput from idle bubbles.
            fill_ratio = self.min_start_batch_ratio
            if monitor.request_counter.running_batches == 0:
                fill_ratio = self.min_start_batch_ratio_cold
            min_start_bs = max(1, int(np.ceil(batchsize * fill_ratio)))
            if queue_depth < min_start_bs:
                return 0

        pf_len = common.RequestQueue.outstanding[0].get_prefill_length()
        eff_bs = self._choose_effective_batchsize(mem_sys, comp_sys, self._current_bs_est, pf_len)
        if eff_bs <= 0:
            return -1

        incoming_batch = BatchOfRequests(common.RequestQueue.outstanding[:eff_bs])

        # Sanity checks for request integrity.
        for incoming_req in incoming_batch.ongoing_requests:
            if incoming_req.has_prefill():
                prefill_inf: infer = incoming_req._infs["prefill"]
                if prefill_inf.type != 'prefill':
                    raise ValueError("The first inference in the request must be a prefill inference.")
            if len(incoming_req._infs) == 0:
                raise ValueError("The request has no inferences.")
            for inf in incoming_req._infs.values():
                if len(inf.blocks) == 0:
                    raise ValueError("The inference has no blocks.")

        incoming_batch.step_batch_id()
        common.RequestQueue.ready.append(incoming_batch)

        for _ in range(eff_bs):
            common.total_pf_tokens += common.RequestQueue.outstanding[0].get_prefill_length()
            common.RequestQueue.outstanding.pop(0)
            monitor.request_counter.increment_running()
            monitor.request_counter.decrement_pending()

        monitor.request_counter.increment_running_batches()
        return 1

    def start_request(self, env: simpy.Environment, PEs: list[processing]) -> None:
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
        return "vllm_legacy"
