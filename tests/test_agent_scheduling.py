"""Priority order, dependency-safe batching and physical prefix accounting."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

import Sim.common as common
from Sim.config.agent_scheduler_config import AgentSchedulerConfig
from Sim.config.model_config import BaseModelConfig
from Sim.entities.agent_workload import AgentSession, AgentStep, AgentWorkload
from Sim.entities.mem_chiplet import mem_chiplet, Params
from Sim.entities.mem_sys import mem_sys
from Sim.execution.prefix_cache import BasePrefixCache
from Sim.metrics.monitor import request_counter, tokens_monitor
from Sim.request_generator.agent_request_generator import AgentRequest
from Sim.scheduler.BaseReqScheduler import BaseReqScheduler
from Sim.scheduler.agent_priority import BaseAgentPriority
from analytic_profile.modern import GQAProfile


class AgentSchedulingTest(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.model = BaseModelConfig.create_from_name('gqa-prefix-fixture')
        patches = [patch.multiple(common, ByteperParam=1, task_parallelism='pipeline',
                                  req_allocated_mems={}, batch_size=2),
                   patch.multiple(common.RequestQueue, outstanding=[], ready=[], running=[], executable=[]),
                   patch('Sim.entities.mem_sys.mem_monitor.update_utilization')]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        request_counter.reset()
        tokens_monitor.reset()
        self.addCleanup(request_counter.reset)
        self.addCleanup(tokens_monitor.reset)
        self.memory = mem_sys.__new__(mem_sys)
        chip = mem_chiplet(chiplet_type=0, dram_budget=1, chiplet_id=0, chiplet_loc=(0, 0))
        chip.content_params['weights'] = [Params(size=1, block_id=i, name=f'Block {i} weights') for i in range(2)]
        chip.inuse_budget = chip._current_allocated = 2
        self.memory._mem_chiplets = [chip]
        self.memory._current_used = 2
        self.memory._num_io_limit = 4
        self.memory.blocks_alloc = {0: 0, 1: 0}
        self.comp = SimpleNamespace(_total_budget=4, _current_run=0)

    def request(self, tokens=128, outputs=3, prefix='shared', prefix_tokens=96):
        request = AgentRequest(self.env, AgentStep('0', tokens, outputs,
            prefix_id=prefix, prefix_tokens=prefix_tokens), self.model)
        request.fill_request()
        return request

    def cache(self, capacity=49152):
        return BasePrefixCache.create_from_name('lru', self.memory, self.model, capacity)

    def scheduler(self, **kwargs):
        scheduler = BaseReqScheduler.create_from_name('agent')(None, self.memory, self.comp, bs=2)
        scheduler.configure_runtime(self.env, AgentSchedulerConfig(**kwargs))
        scheduler.set_max_concurrent_requests(self.model, self.memory)
        return scheduler

    def enqueue(self, *requests):
        common.RequestQueue.outstanding.extend(requests)
        for _ in requests:
            request_counter.increment_pending()

    def test_classic_priority_scores_and_deterministic_ties(self):
        a = SimpleNamespace(_id=1, _arrived_at=0, deadline_tick=100)
        b = SimpleNamespace(_id=2, _arrived_at=9, deadline_tick=20)
        expected = dict(fcfs=a, sjf=b, hrrn=a, edf=b, least_slack=b)
        for name, first in expected.items():
            policy = BaseAgentPriority.create_from_name(name)
            selected = min([a, b], key=lambda r: policy.key(r, 10, 5 if r is a else 1))
            self.assertIs(selected, first, name)
        self.assertEqual(BaseAgentPriority.create_from_name('hrrn').score(a, 10, 5), -3)

    def test_estimator_does_not_read_output_length_without_oracle(self):
        short, long = self.request(outputs=1), self.request(outputs=100)
        scheduler = self.scheduler()
        self.assertEqual(scheduler.estimate_service(short), scheduler.estimate_service(long))
        oracle = self.scheduler(service_estimator='oracle_output')
        self.assertLess(oracle.estimate_service(short), oracle.estimate_service(long))
        long.step = replace(long.step, estimated_service_s=.5)
        self.assertEqual(scheduler.estimate_service(long), .5*common.time_granularity)

    def test_equal_shape_batching_skips_incompatible_prompts_and_records_members(self):
        first, other, third = self.request(), self.request(tokens=256), self.request(outputs=1)
        self.enqueue(first, other, third)
        self.assertEqual(self.scheduler().schedule_request(self.memory, self.comp, 2), 1)
        self.assertEqual(common.RequestQueue.ready[0].ongoing_requests, [first, third])
        self.assertEqual(common.RequestQueue.outstanding, [other])
        self.assertEqual(first.admission['batch_size'], 2)

    def test_timeout_flushes_partial_batch_even_while_session_has_future_steps(self):
        self.enqueue(self.request())
        scheduler = self.scheduler(batching='timeout', max_batch_wait_s=.000010)
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 2), 0)
        self.env.run(until=11)
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 2), 1)
        self.assertEqual(common.RequestQueue.ready[0]._batch_size, 1)

    def test_prefix_pins_share_physical_state_and_evict_after_release(self):
        cache = self.cache()
        producer = self.request()
        cache.publish(producer, 0)
        self.assertEqual(cache.used_bytes, 2*256*96)
        readers = [self.request(), self.request()]
        for reader in readers:
            reader.prefix_cache = cache
            reader.prepare_prefix()
        self.assertEqual(cache.snapshot()['pinned_references'], 2)
        self.assertFalse(cache.evict_one(1))
        self.assertEqual(self.memory._current_used, 2+49152/1024**2)
        for reader in readers:
            reader.rollback_prefix()
        self.assertTrue(cache.evict_one(2))
        chip = self.memory._mem_chiplets[0]
        self.assertEqual((chip.inuse_budget, chip._current_allocated), (2, 2))

    def test_lru_capacity_and_duplicate_publication(self):
        cache = self.cache(capacity=2*49152)
        a, b, c = [self.request(prefix=key) for key in ('a', 'b', 'c')]
        cache.publish(a, 0)
        cache.publish(b, 1)
        a.prefix_cache = cache
        a.prepare_prefix()
        a.rollback_prefix()
        cache.publish(c, 2)
        self.assertEqual(list(cache.entries), ['a', 'c'])
        cache.publish(c, 3)
        self.assertEqual(cache.insertions, 3)
        self.assertEqual(cache.evictions, 1)

    def test_partial_prefill_keeps_context_but_reduces_private_state_and_work(self):
        cache = self.cache()
        request = self.request()
        cache.publish(request, 0)
        request.prefix_cache = cache
        request.prepare_prefix()
        block = request._infs['prefill'].blocks[0]
        self.assertEqual((block.context_length, block.query_tokens, block.states_store), (128, 32, 32*256))
        request.step_processing(1)
        request.step_processing(2)
        decode = request._infs['decode'].blocks[0]
        self.assertEqual((decode.context_length, decode.states_store), (129, 33*256))
        config = dict(vars(self.model.block_config.attention), bs=32, batch_size=1, L_seq=128, stage='prefill')
        work = GQAProfile().work(**config)
        attention = next(item for item in work if item.name == 'causal_attention')
        self.assertEqual(attention.macs, 2*512*(32*96+32*33//2))
        self.assertEqual(attention.state_bytes, 128*256)
        # A failed admission must restore full prefill shapes.
        request.rollback_prefix()
        self.assertEqual(request._infs['prefill'].blocks[0].states_store, 128*256)

    def test_cache_rejects_recurrent_models_and_inconsistent_trace_identity(self):
        with self.assertRaisesRegex(ValueError, 'dense'):
            BasePrefixCache.create_from_name('lru', self.memory,
                BaseModelConfig.create_from_name('qwen3.5-9b-text'), 100)
        with self.assertRaises(ValueError):
            AgentWorkload([AgentSession('a', 0, (AgentStep('0',128,1,prefix_id='x',prefix_tokens=32),
                AgentStep('1',128,1,prefix_id='x',prefix_tokens=64)))], {})
        with self.assertRaises(ValueError):
            AgentStep('0', 128, 1, prefix_id='x', prefix_tokens=128)
        for length in (None, '128', True, 0):
            with self.subTest(length=length), self.assertRaises(ValueError):
                AgentStep('0', length, 1)

    def test_memory_pressure_shrinks_batch_without_leaking_reservations(self):
        first, second = self.request(), self.request()
        scheduler = self.scheduler()
        size = sum(scheduler.request_memory_mib(first,b) for b in first._infs['prefill'].blocks)
        chip = self.memory._mem_chiplets[0]
        chip.dram_budget = 2+size+.00001
        self.enqueue(first, second)
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 2), 1)
        self.assertEqual(common.RequestQueue.ready[0].ongoing_requests, [first])
        self.assertAlmostEqual(chip._current_allocated, 2+size)
        self.assertEqual(set(common.req_allocated_mems), {first._id})

    def test_impossible_memory_fails_explicitly(self):
        self.memory._mem_chiplets[0].dram_budget = 2
        self.enqueue(self.request())
        with self.assertRaises(MemoryError):
            self.scheduler().schedule_request(self.memory, self.comp, 2)
        self.assertEqual(self.memory._mem_chiplets[0]._current_allocated, 2)

    def test_idle_cache_evicts_under_admission_pressure_and_retry_keeps_event(self):
        cache = self.cache()
        cached = self.request()
        cache.publish(cached, 0)
        cold = self.request(prefix='other')
        cold.prefix_cache = cache
        scheduler = self.scheduler()
        private = sum(scheduler.request_memory_mib(cold, b) for b in cold._infs['prefill'].blocks)
        self.memory._mem_chiplets[0].dram_budget = 2+private+.00001
        self.enqueue(cold)
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 1), -1)
        self.assertEqual(cache.evictions, 1)
        self.assertEqual(common.req_allocated_mems, {})
        self.assertEqual(scheduler.schedule_request(self.memory, self.comp, 1), 1)
        # A retry must not deepcopy its environment, event or cache manager.
        event = cold.completion
        self.assertIs(cold.retry_request(), cold)
        self.assertIs(cold.completion, event)
        self.assertIs(cold.prefix_cache, cache)

    def test_counter_reset_includes_batches_and_configuration_rejects_invalid_values(self):
        request_counter.increment_running_batches()
        request_counter.reset()
        self.assertEqual(request_counter.running_batches, 0)
        for kwargs in (dict(priority='unknown'), dict(decode_token_s=0),
                       dict(max_batch_wait_s=float('nan')), dict(max_active_batches=-1),
                       dict(prefix_cache='lru'), dict(prefix_capacity_bytes=10)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AgentSchedulerConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
