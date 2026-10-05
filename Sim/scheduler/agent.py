"""Agent request admission using existing static execution and reservations."""

import Sim.common as common
from Sim.config.agent_scheduler_config import AgentSchedulerConfig
from Sim.entities.block import block
from Sim.metrics.monitor import request_counter
from Sim.scheduler.static import StaticReplicaScheduler
from Sim.scheduler.agent_priority import BaseAgentPriority
from Sim.scheduler.agent_batching import BaseAgentBatching


class AgentReplicaScheduler(StaticReplicaScheduler):
    def __init__(self, chip_graph, mem_sys, comp_sys, bs=1):
        if type(bs) is not int or bs < 1:
            raise ValueError('Agent batch size must be a positive integer.')
        super().__init__(chip_graph, mem_sys, comp_sys, bs)

    @staticmethod
    def get_name():
        return 'agent'

    def configure_runtime(self, env, config):
        self.env = env
        self.config = config or AgentSchedulerConfig()
        self.policy = BaseAgentPriority.create_from_name(self.config.priority)
        self.batching = BaseAgentBatching.create_from_name(self.config.batching)
        self.restoring_request = None
        self._max_concurrent_batch = min(self._max_concurrent_batch, common.num_wk_threads)
        if self.config.max_active_batches:
            self._max_concurrent_batch = min(self._max_concurrent_batch, self.config.max_active_batches)

    def set_max_concurrent_requests(self, model_config, mem_sys):
        self.model_config = model_config
        # Exact per-request reservations below enforce placed HBM capacity.
        self._max_concurrent_requests = common.max_requests_in_parallel

    def estimate_service(self, request):
        if request.step.estimated_service_s is not None:
            seconds = request.step.estimated_service_s
        else:
            outputs = (request.step.output_tokens if self.config.service_estimator == 'oracle_output'
                       else self.config.estimated_output_tokens)
            cached = request.prefix_cache.peek(request) if request.prefix_cache else 0
            seconds = ((request.step.input_tokens - cached) * self.config.prefill_token_s
                       + outputs * self.config.decode_token_s)
        return max(1, seconds * common.time_granularity)

    @staticmethod
    def shape(request):
        cached = request.prefix_cache.peek(request) if request.prefix_cache else 0
        # Homogeneous prompts keep the existing representative batch profile
        # exact. Different output lengths drain independently without padding.
        return request.get_prefill_length(), request._context_length, cached

    def request_memory_mib(self, request, prefill_block):
        config = prefill_block.block_config
        # Actual replay output length is used for memory reservation, never
        # priority unless oracle_output was selected. This is a known-length
        # offline admission baseline, not online KV paging.
        final_context = request.step.input_tokens + request.step.output_tokens - 1
        decode = block(final_context, config, block_num=prefill_block.block_num)
        decode.states_store -= config.states * request.cached_prefix_tokens
        prefill = prefill_block.states_store + prefill_block.peak_intermediate_store
        decoding = decode.states_store + decode.peak_intermediate_store if request.has_decode() else 0
        # Include input/output handoff buffers while reservations remain held.
        return common.convert_param_mB(max(prefill, decoding) + 2 * prefill_block.output_act)

    def schedule_request(self, mem_sys, comp_sys, batchsize=1):
        if not common.RequestQueue.outstanding:
            return 0
        if (request_counter.running_batches >= self._max_concurrent_batch
                or comp_sys._current_run >= comp_sys._total_budget):
            return -1
        queue = common.RequestQueue.outstanding
        if any(not hasattr(request, 'step') for request in queue):
            raise ValueError('The agent scheduler requires agent requests.')
        now = self.env.now
        estimates = {request._id: self.estimate_service(request) for request in queue}
        ordered = sorted(queue, key=lambda request: self.policy.key(request, now, estimates[request._id]))
        if self.restoring_request in ordered:
            ordered.remove(self.restoring_request)
            ordered.insert(0, self.restoring_request)
        anchor = ordered[0]
        if anchor.prefix_cache and not anchor.prefix_cache.ensure_available(anchor):
            self.restoring_request = anchor
            return 0
        selected = [request for request in ordered if self.shape(request) == self.shape(anchor)][:batchsize]
        wait = self.config.max_batch_wait_s * common.time_granularity
        if not self.batching.ready(selected, batchsize, now, wait):
            return 0
        original = list(queue)
        while selected:
            for request in selected:
                request.prepare_prefix()
            queue[:] = selected + [request for request in original if request not in selected]
            result = super().schedule_request(mem_sys, comp_sys, batchsize=len(selected))
            if result == 1:
                self.restoring_request = None
                batch = common.RequestQueue.ready[-1]
                for request in selected:
                    request.admission = dict(tick=now, batch_id=batch._id, batch_size=len(selected),
                        priority=self.config.priority, estimated_service_ticks=estimates[request._id],
                        score=self.policy.score(request, now, estimates[request._id]))
                    if request.prefix_cache:
                        request.prefix_cache.record_admission(request)
                return 1
            for request in selected:
                # A failed hot admission must not endlessly reload the same
                # entry after eviction. Fall back to recompute for this call.
                if request.cached_prefix_tokens:
                    request.skip_host_restore = True
                request.rollback_prefix()
            queue[:] = original
            cache = anchor.prefix_cache
            if cache and cache.evict_one(now):
                # Cache eviction can change shape compatibility. Re-evaluate
                # on the next scheduler tick instead of using stale hit lengths.
                return -1
            if len(selected) > 1:
                selected = selected[:-1]
            else:
                if request_counter.running_batches == 0 and not (cache and cache.pending_events()):
                    raise MemoryError('Agent request cannot fit the placed HBM reservation budget.')
                return -1
        return -1
