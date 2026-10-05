"""Iteration-level continuous batching with resident private state.

Prefill batches have equal input/cache shapes. Decode batches use the longest
context as a padded profile while retaining each request's actual KV allocation.
One iteration batch runs at a time; new prefills and resident decodes alternate
when both can make progress. The paper scheduler inherits vllm_legacy instead.
"""

import json
from pathlib import Path

import Sim.common as common
from Sim.config.agent_scheduler_config import AgentSchedulerConfig
from Sim.entities.batch_of_req import BatchOfRequests
from Sim.entities.block import block
from Sim.metrics.monitor import request_counter
from Sim.scheduler.agent_batching import BaseAgentBatching
from Sim.scheduler.agent_priority import BaseAgentPriority
from Sim.scheduler.static import StaticReplicaScheduler


class VllmLatestReplicaScheduler(StaticReplicaScheduler):
    dispatch_ready_on_admission = True
    @staticmethod
    def get_name():
        return 'vllm_latest'

    def configure_runtime(self, env, config):
        if common.task_parallelism != 'pipeline' or common.mapping_strategy != 'static':
            raise ValueError('vllm_latest requires static mapping and pipeline execution.')
        self.env = env
        self.config = config or AgentSchedulerConfig()
        if self.config.max_active_batches > 1:
            raise ValueError('vllm_latest currently supports one iteration batch at a time.')
        self.policy = BaseAgentPriority.create_from_name(self.config.priority)
        self.batching = BaseAgentBatching.create_from_name(self.config.batching)
        self._max_concurrent_batch = 1
        self.resident = []
        self.state = {}
        self.iterations = []
        self.prefer_decode = False
        self.restoring_request = None
        self.wakeup = env.event()

    def wait_for_work(self, env):
        # Resume immediately at a token boundary. Poll arrivals/restore readiness
        # using the existing clock without adding one clock of delay per token.
        if self.wakeup.triggered:
            self.wakeup = env.event()
        return env.any_of([env.timeout(common.simulation_clk), self.wakeup])

    def set_max_concurrent_requests(self, model_config, mem_sys):
        self.model_config = model_config
        # Known replay lengths reserve the whole private lifetime at admission.
        self._max_concurrent_requests = common.max_requests_in_parallel

    def create_processing(self, execution_backend):
        from Sim.execution.continuous import ContinuousProcessing
        return ContinuousProcessing(execution_backend, self)

    def estimate_service(self, request):
        step = getattr(request, 'step', None)
        if step is not None and step.estimated_service_s is not None:
            return max(1, step.estimated_service_s * common.time_granularity)
        outputs = (request.get_decode_length() + 1 if self.config.service_estimator == 'oracle_output'
                   else self.config.estimated_output_tokens)
        cache = getattr(request, 'prefix_cache', None)
        cached = cache.peek(request) if cache else 0
        return max(1, ((request.get_prefill_length() - cached) * self.config.prefill_token_s
                       + outputs * self.config.decode_token_s) * common.time_granularity)

    def priority_key(self, request):
        if not hasattr(request, 'deadline_tick'):
            request.deadline_tick = request._arrived_at + self.config.default_session_slo_s * common.time_granularity
        return self.policy.key(request, self.env.now, self.estimate_service(request))

    @staticmethod
    def shape(request):
        cache = getattr(request, 'prefix_cache', None)
        return request.get_prefill_length(), request._context_length, cache.peek(request) if cache else 0

    def request_memory_mib(self, request, initial_block):
        config = initial_block.block_config
        final_context = request._context_length + request.get_decode_length()
        decode = block(final_context, config, block_num=initial_block.block_num)
        if config.cache_store:
            decode.states_store -= config.states * getattr(request, 'cached_prefix_tokens', 0)
        prefill = initial_block.states_store + initial_block.peak_intermediate_store
        decoding = decode.states_store + decode.peak_intermediate_store if request.has_decode() else 0
        return common.convert_param_mB(max(prefill, decoding) + 2 * max(initial_block.output_act, decode.output_act))

    def schedule_request(self, mem_sys, comp_sys, batchsize=1):
        if batchsize < 1:
            raise ValueError('Continuous batch size must be positive.')
        if request_counter.running_batches:
            return 0
        # Reserve at most batchsize resident requests, not batchsize new requests
        # per iteration. This also bounds private KV and prevents decode starvation.
        slots = batchsize - request_counter.running_requests
        if common.RequestQueue.outstanding and slots > 0 and not (self.prefer_decode and self.resident):
            result = self.admit(mem_sys, comp_sys, slots)
            if result == 1:
                self.prefer_decode = True
                return 1
        if self.resident:
            selected = sorted(self.resident, key=self.priority_key)[:batchsize]
            # The representative backend profile charges padded decode work at
            # the longest context. Private allocations stay at their real sizes.
            selected.sort(key=lambda r: r._infs['decode'].blocks[0].context_length, reverse=True)
            batch = BatchOfRequests(selected)
            batch.step_batch_id()
            common.RequestQueue.ready.append(batch)
            for request in selected:
                self.resident.remove(request)
            request_counter.increment_running_batches()
            self.prefer_decode = False
            return 1
        return 0

    def admit(self, mem_sys, comp_sys, slots):
        queue = common.RequestQueue.outstanding
        original = list(queue)
        ordered = sorted(queue, key=self.priority_key)
        if self.restoring_request in ordered:
            ordered.remove(self.restoring_request)
            ordered.insert(0, self.restoring_request)
        anchor = ordered[0]
        cache = getattr(anchor, 'prefix_cache', None)
        if cache and not cache.ensure_available(anchor):
            self.restoring_request = anchor
            return 0
        selected = [r for r in ordered if self.shape(r) == self.shape(anchor)][:slots]
        if not self.batching.ready(selected, slots, self.env.now,
                                   self.config.max_batch_wait_s * common.time_granularity):
            return 0
        estimates = {r._id: self.estimate_service(r) for r in selected}
        while selected:
            for request in selected:
                if hasattr(request, 'prepare_prefix'):
                    request.prepare_prefix()
            queue[:] = selected + [r for r in original if r not in selected]
            result = super().schedule_request(mem_sys, comp_sys, len(selected))
            if result == 1:
                self.restoring_request = None
                batch = common.RequestQueue.ready[-1]
                for request in selected:
                    request.admission = dict(tick=self.env.now, batch_id=batch._id, batch_size=len(selected),
                        priority=self.config.priority, estimated_service_ticks=estimates[request._id],
                        score=self.policy.score(request, self.env.now, estimates[request._id]))
                    cache = getattr(request, 'prefix_cache', None)
                    if cache:
                        cache.record_admission(request)
                    common.total_pf_tokens += request.get_prefill_length()
                    common.total_dc_tokens += request.get_decode_length()
                return 1
            for request in selected:
                if getattr(request, 'cached_prefix_tokens', 0):
                    request.skip_host_restore = True
                if hasattr(request, 'rollback_prefix'):
                    request.rollback_prefix()
            queue[:] = original
            if cache and cache.evict_one(self.env.now):
                return -1
            if len(selected) > 1:
                selected = selected[:-1]
            else:
                if not request_counter.running_requests and not (cache and cache.pending_events()):
                    raise MemoryError('Continuous request cannot fit the placed HBM reservation budget.')
                return -1
        return -1

    def start_request(self, env, PEs):
        batch = common.RequestQueue.ready.pop(0)
        index = next(i for i, pe in enumerate(PEs) if pe.IDLE)
        PEs[index].config_processing(batch._id)
        for request in batch.ongoing_requests:
            if request._process_idx == -1:
                request._scheduled = True
                request.step_processing(env.now)
            request._process_target = index
            request.set_batch_id(batch._id)
        batch._processing_target = index
        common.RequestQueue.executable.append(batch)
        requests = batch.ongoing_requests
        stage = 'prefill' if requests[0]._process_idx == 0 and requests[0].has_prefill() else 'decode'
        self.iterations.append(dict(batch_id=batch._id, stage=stage, started_tick=env.now,
            completed_tick=None, request_ids=[r._id for r in requests],
            contexts=[r._infs[stage].blocks[0].context_length for r in requests],
            completed_request_ids=[]))

    def finish_iteration(self, processor, tick):
        row = self.iterations[-1]
        row['completed_tick'] = tick
        remaining = {r._id for r in processor.requests}
        row['completed_request_ids'] = [i for i in row['request_ids'] if i not in remaining]
        for request in processor.requests:
            self.state[request._id] = (processor.states_cache[request._id],
                [item for item in processor.input_cache if item.req_id == request._id])
            self.resident.append(request)
        common.RequestQueue.remove_running_batch(processor.current_batch._id)
        # Keep completed request objects once, not one duplicate reference per
        # generated token. The iteration ledger already retains batch history.
        batch = processor.current_batch
        batch.completed_requests = [r for r in batch.ongoing_requests if r._id not in remaining]
        batch.ongoing_requests = []
        if batch.completed_requests:
            common.RequestQueue.completed.append(batch)
        request_counter.decrement_running_batches()
        if not self.wakeup.triggered:
            self.wakeup.succeed()

    def write_report(self, output_dir):
        report = dict(scheduler=self.get_name(), iterations=self.iterations,
            resident_request_ids=[r._id for r in self.resident], retained_private_states=len(self.state),
            running_batches=request_counter.running_batches,
            scope='One iteration batch; homogeneous prefill; longest-context padded decode; lifetime KV reservation; no active KV swap.')
        (Path(output_dir) / 'continuous_batching.json').write_text(json.dumps(report, indent=2)+'\n')
