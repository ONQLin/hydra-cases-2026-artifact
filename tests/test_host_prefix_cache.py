"""Explicit host tier ownership, transfer timing and resource partitions."""

from types import SimpleNamespace
import unittest

from Sim.config.agent_scheduler_config import AgentSchedulerConfig
from Sim.execution.prefix_cache import BasePrefixCache
from Sim.execution.host_prefix_cache import HostPrefixMemory, HostKVTransferEngine
from tests import test_agent_scheduling as scheduling_tests


class HostPrefixCacheTest(unittest.TestCase):
    # Reuse the physical-memory fixture without duplicating its test cases.
    setUp = scheduling_tests.AgentSchedulingTest.setUp
    request = scheduling_tests.AgentSchedulingTest.request
    def offload(self, host_capacity=256*1024**3, tool_bytes=0, **kwargs):
        config = AgentSchedulerConfig(prefix_cache='lru_offload', prefix_capacity_bytes=49152,
                                      host_dram_capacity_bytes=host_capacity, **kwargs)
        cache = BasePrefixCache.create_from_name('lru_offload', self.memory, self.model, 49152)
        cache.configure_runtime(self.env, config,
                                SimpleNamespace(memory=SimpleNamespace(capacity=tool_bytes) if tool_bytes else None))
        return cache

    def test_backup_pins_hbm_and_host_restore_blocks_admission(self):
        cache = self.offload()
        request = self.request()
        request.prefix_cache = cache
        cache.publish(request, 0)
        self.assertFalse(cache.evict_one(0))
        self.assertFalse(cache.host.entries['shared'].ready)
        self.env.run()
        self.assertEqual(cache.backups, 1)
        self.assertTrue(cache.evict_one(self.env.now))
        self.assertEqual(self.memory._current_used, 2)
        self.assertFalse(cache.ensure_available(request))
        self.assertEqual(cache.peek(request), 0)
        with self.assertRaisesRegex(RuntimeError, 'before host restore'):
            cache.acquire(request)
        self.env.run()
        self.assertTrue(cache.ensure_available(request))
        self.assertFalse(cache.evict_one(self.env.now))  # Admission lease.
        request.prepare_prefix()
        self.assertEqual(request.cached_prefix_tokens, 96)
        self.assertEqual(cache.entries['shared'].references, 1)
        request.rollback_prefix()
        self.assertEqual(cache.snapshot()['pinned_references'], 0)
        self.assertEqual(cache.host.snapshot()['pinned_references'], 0)
        self.assertEqual(cache.restores, 1)

    def test_transfer_queue_is_shared_and_accounts_for_startup_and_bandwidth(self):
        config = AgentSchedulerConfig(host_link_bandwidth_gbps=2,
                                      host_dram_bandwidth_gbps=1, host_transfer_latency_s=.000005)
        engine = HostKVTransferEngine(self.env, config)
        self.env.process(engine.transfer('a', 1000, 'hbm_to_host'))
        self.env.process(engine.transfer('b', 2000, 'host_to_hbm'))
        self.env.run()
        self.assertEqual([(r['started_tick'], r['completed_tick']) for r in engine.records], [(0, 6), (6, 13)])

    def test_host_capacity_reserves_tool_quota_and_never_evicts_pinned_source(self):
        host = HostPrefixMemory(120, 20)
        a = host.reserve('a', 1, 60)
        self.assertIsNone(host.reserve('b', 1, 60))
        a.ready, a.references = True, 0
        self.assertIsNotNone(host.reserve('b', 1, 60))
        self.assertEqual(list(host.entries), ['b'])
        self.assertLessEqual(host.used_bytes+host.tool_reservation, host.capacity)
        with self.assertRaises(ValueError):
            HostPrefixMemory(20, 20)

    def test_no_host_space_falls_back_to_hbm_only_and_cutoff_exposes_pending_copy(self):
        cache = self.offload(host_capacity=49151)
        cache.publish(self.request(), 0)
        self.assertEqual(cache.pending_events(), [])
        self.assertTrue(cache.evict_one(0))
        self.assertEqual(cache.host.used_bytes, 0)
        other = self.offload(host_link_bandwidth_gbps=.001)
        other.publish(self.request(), 0)
        self.env.run(until=1)
        report = other.snapshot()
        self.assertEqual(report['pending_transfers'], 1)
        self.assertEqual(report['pinned_references'], 1)
        self.assertIsNone(report['transfers'][0]['completed_tick'])

    def test_hbm_and_host_capacity_release_independently(self):
        cache = self.offload(host_capacity=49152)
        a, b = self.request(prefix='a'), self.request(prefix='b')
        cache.publish(a, 0)
        self.env.run()
        cache.publish(b, self.env.now)
        self.env.run()
        self.assertEqual(cache.host.evictions, 1)
        self.assertEqual(cache.used_bytes, 49152)
        self.assertEqual(cache.host.used_bytes, 49152)
        self.assertTrue(cache.ensure_available(a))  # Evicted in both tiers: recompute.


if __name__ == '__main__':
    unittest.main()
