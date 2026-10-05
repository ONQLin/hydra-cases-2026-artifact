"""Iteration admission, state ownership and legacy scheduler isolation."""

from types import SimpleNamespace
from unittest.mock import patch
import unittest

import Sim.common as common
from Sim.config.agent_scheduler_config import AgentSchedulerConfig
from Sim.entities.request import Request
from Sim.metrics.monitor import request_counter
from Sim.scheduler.BaseReqScheduler import BaseReqScheduler
from Sim.scheduler.vllm import VllmReplicaScheduler
from Sim.scheduler.vllm_legacy import VllmLegacyReplicaScheduler
from Sim.scheduler.vllm_latest import VllmLatestReplicaScheduler
from tests import test_agent_scheduling as fixtures
from tests import test_host_prefix_cache as host_fixtures


class ContinuousBatchingTest(unittest.TestCase):
    setUp = fixtures.AgentSchedulingTest.setUp
    request = fixtures.AgentSchedulingTest.request
    enqueue = fixtures.AgentSchedulingTest.enqueue
    offload = host_fixtures.HostPrefixCacheTest.offload

    def scheduler(self, **kwargs):
        context = patch.multiple(common, mapping_strategy='static', local_scheduler='vllm_latest',
                                 total_pf_tokens=0, total_dc_tokens=0)
        context.start()
        self.addCleanup(context.stop)
        result = VllmLatestReplicaScheduler(None, self.memory, self.comp, bs=2)
        result.configure_runtime(self.env, AgentSchedulerConfig(**kwargs))
        result.set_max_concurrent_requests(self.model, self.memory)
        return result

    def test_paper_factory_does_not_inherit_new_scheduler_or_executor(self):
        self.assertIs(BaseReqScheduler.create_from_name('vllm'), VllmReplicaScheduler)
        self.assertIs(BaseReqScheduler.create_from_name('vllm_latest'), VllmLatestReplicaScheduler)
        self.assertIs(BaseReqScheduler.create_from_name('vllm_legacy'), VllmLegacyReplicaScheduler)
        self.assertTrue(issubclass(VllmReplicaScheduler, VllmLegacyReplicaScheduler))
        self.assertFalse(issubclass(VllmReplicaScheduler, VllmLatestReplicaScheduler))
        self.assertIs(VllmReplicaScheduler.start_request, VllmLegacyReplicaScheduler.start_request)
        self.assertFalse(VllmReplicaScheduler.dispatch_ready_on_admission)

    def test_partial_prefill_and_full_lifetime_reservation(self):
        scheduler = self.scheduler()
        request = self.request(outputs=100)
        self.enqueue(request)
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 4), 1)
        self.assertEqual(common.RequestQueue.ready[0]._batch_size, 1)
        allocated = sum(size for _, size in common.req_allocated_mems[request._id])
        final_kv = sum(b.block_config.states * (128+99) for b in request._infs['prefill'].blocks)/1024**2
        self.assertGreater(allocated, final_kv)
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 4), 0)

    def test_host_restore_wait_keeps_resident_decode_runnable(self):
        scheduler = self.scheduler()
        cache = self.offload()
        newcomer = self.request()
        newcomer.prefix_cache = cache
        cache.publish(newcomer, 0)
        self.env.run()
        cache.evict_one(self.env.now)
        resident = self.request(tokens=256)
        resident.step_processing(0)
        resident.step_processing(10)
        scheduler.resident.append(resident)
        request_counter.increment_running()
        self.enqueue(newcomer)
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 2), 1)
        self.assertIs(scheduler.restoring_request, newcomer)
        self.assertEqual(common.RequestQueue.ready[0].ongoing_requests, [resident])
        self.assertEqual(cache.peek(newcomer), 0)
        self.env.run()
        self.assertEqual(cache.restores, 1)

    def test_decode_padding_order_and_resume_preserve_cursor_and_state_identity(self):
        scheduler = self.scheduler()
        requests = [self.request(tokens=128), self.request(tokens=256)]
        states, inputs = {}, {}
        for request in requests:
            request.step_processing(0)
            request.step_processing(10)
            states[request._id] = [object()]
            inputs[request._id] = [SimpleNamespace(req_id=request._id)]
            scheduler.state[request._id] = states[request._id], inputs[request._id]
            scheduler.resident.append(request)
            request_counter.increment_running()
        before = self.memory._mem_chiplets[0]._current_allocated
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 2), 1)
        pe = scheduler.create_processing(None)
        scheduler.start_request(self.env, [pe])
        batch = common.RequestQueue.executable[0]
        self.assertEqual(batch.ongoing_requests, list(reversed(requests)))
        pe.requests = batch.ongoing_requests
        pe.states_cache, pe.input_cache = {}, []
        pe.restore_iteration()
        self.assertEqual([r._process_idx for r in requests], [1, 1])
        self.assertEqual([inf.process_idx for inf in pe.select_inferences()], [0, 0])
        self.assertEqual(scheduler.state, {})
        self.assertIs(pe.states_cache[requests[0]._id], states[requests[0]._id])
        self.assertIs(pe.input_cache[-1], inputs[requests[0]._id][0])
        self.assertEqual(self.memory._mem_chiplets[0]._current_allocated, before)

    def test_plain_trace_request_and_timeout_are_supported(self):
        scheduler = self.scheduler(batching='timeout', max_batch_wait_s=.000010)
        request = Request(0, 128, 3, 128, self.model)
        request.fill_request()
        self.enqueue(request)
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 2), 0)
        self.env.run(until=11)
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 2), 1)
        pe = scheduler.create_processing(None)
        scheduler.start_request(self.env, [pe])
        self.assertEqual(request._process_idx, 0)
        self.assertEqual(request._scheduled_at, 11)

    def test_boundary_retains_private_state_and_only_archives_finished_members(self):
        scheduler = self.scheduler()
        first, second = self.request(), self.request(outputs=1)
        self.enqueue(first, second)
        scheduler.schedule_request(self.memory, self.comp, 2)
        pe = scheduler.create_processing(None)
        scheduler.start_request(self.env, [pe])
        batch = common.RequestQueue.executable.pop()
        common.RequestQueue.running.append(batch)
        pe.current_batch, pe.requests = batch, [first]
        states = [object()]
        pe.states_cache = {first._id: states}
        pe.input_cache = [SimpleNamespace(req_id=first._id)]
        wake = scheduler.wait_for_work(self.env)
        with patch.object(common.RequestQueue, 'completed', []):
            self.assertTrue(pe.yield_iteration(10))
            self.assertEqual(common.RequestQueue.completed[0].completed_requests, [second])
            self.assertEqual(batch.ongoing_requests, [])
        self.assertEqual(scheduler.resident, [first])
        self.assertIs(scheduler.state[first._id][0], states)
        self.assertEqual(request_counter.running_batches, 0)
        self.assertEqual(common.RequestQueue.running, [])
        self.assertTrue(pe.IDLE)
        self.env.run(until=1)
        self.assertTrue(wake.triggered)

    def test_impossible_request_fails_and_rolls_back_all_reservations(self):
        scheduler = self.scheduler()
        self.memory._mem_chiplets[0].dram_budget = 2
        self.enqueue(self.request())
        with self.assertRaises(MemoryError):
            scheduler.schedule_request(self.memory, self.comp, 2)
        self.assertEqual(common.req_allocated_mems, {})
        self.assertEqual(self.memory._mem_chiplets[0]._current_allocated, 2)


if __name__ == '__main__':
    unittest.main()
