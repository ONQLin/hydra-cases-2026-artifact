"""Operator arithmetic, tensor dependencies, memory admission and directed feedback."""

from dataclasses import asdict
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

import Sim.common as common
from Sim.config.attention_operator_config import KDAConfig, MLAConfig, OperatorBlockConfig
from Sim.config.model_config import BaseModelConfig
from Sim.config.sys_config import HPSim_Config, MetricsConfig
from Sim.entities.block import block
from Sim.entities.execution import ExecutionPhase
from Sim.entities.operator_graph import OperatorGraph
from Sim.entities.mem_sys import mem_sys
from Sim.entities.mem_chiplet import mem_chiplet, Params
from Sim.execution.native import NativeExecutionBackend
from Sim.execution.network import NetworkExecutionMixin
from Sim.simulator import Simulator
from Sim.scheduler.static import StaticReplicaScheduler
from analytic_profile.attention import KDAProfile, MLAProfile
from analytic_profile.modern import AcceleratorRates, BaseOperatorProfile


class AttentionOperatorTest(unittest.TestCase):
    def kda(self, stage='prefill', length=5):
        cfg = KDAConfig(embedding_dim=8, num_heads=2, head_dim=3, conv_kernel_size=4)
        graph = KDAProfile().graph(**dict(vars(cfg), bs=length if stage == 'prefill' else 1,
                                          batch_size=2, L_seq=length, stage=stage))
        return cfg, graph

    def mla(self, layout='absorbed', stage='prefill', length=5):
        cfg = MLAConfig(embedding_dim=8, num_heads=2, q_lora_rank=3,
                        kv_lora_rank=4, qk_nope_head_dim=2, qk_rope_head_dim=2,
                        v_head_dim=3, cache_layout=layout)
        graph = MLAProfile().graph(**dict(vars(cfg), bs=length if stage == 'prefill' else 1,
                                          batch_size=2, L_seq=length, stage=stage))
        return cfg, graph

    def test_kda_channel_gates_and_independent_weight_counts(self):
        cfg, graph = self.kda()
        self.assertIs(BaseOperatorProfile.find('ModernKDA'), KDAProfile)
        weights = sum(t.nbytes for t in graph.tensors.values() if t.storage == 'weight')
        self.assertEqual(weights, 375)  # Four large projections, four low-rank gates, beta, conv, scales.
        self.assertEqual(weights, cfg.parameter_count)
        self.assertEqual(graph.tensors['channel_decay'].shape, (10, 6))
        self.assertEqual(graph.tensors['beta_sigmoid'].shape, (10, 2))
        recurrence = next(n for n in graph.nodes if n.name == 'delta_recurrence')
        self.assertEqual(recurrence.macs, 540)  # 10 tokens * 2 heads * 3 products * 3x3 state.
        self.assertEqual(recurrence.recurrent_steps, 5)
        self.assertEqual(recurrence.recurrent_bytes, 144)
        self.assertLess(graph.export()['schedule'].index('k_normalize'),
                        graph.export()['schedule'].index('delta_recurrence'))

    def test_kda_initialization_and_decode_state_are_distinct(self):
        cfg, prefill = self.kda()
        _, decode = self.kda('decode', 100)
        for graph in (prefill, decode):
            written = sum(t.nbytes for k,t in graph.tensors.items() if k.endswith('_after'))
            self.assertEqual(written, 288)  # 2 sequences, each with 72 matrix and 72 convolution bytes.
            self.assertEqual(written, 2*cfg.state_size_bytes)
        self.assertEqual(sum(t.nbytes for k,t in prefill.tensors.items() if k.endswith('_before')), 0)
        self.assertEqual(sum(t.nbytes for k,t in decode.tensors.items() if k.endswith('_before')), 288)

    def test_mla_cache_layout_changes_compute_and_state_together(self):
        for layout, stride, qk, av in (('absorbed', 6, 600, 400), ('expanded', 14, 400, 300)):
            cfg, graph = self.mla(layout)
            with self.subTest(layout=layout):
                self.assertEqual(cfg.cache_bytes_per_token, stride)
                self.assertEqual(sum(t.nbytes for k,t in graph.tensors.items() if k.endswith('_append')), 10*stride)
                nodes = {n.name: n for n in graph.nodes}
                self.assertEqual(nodes['attention_scores'].macs, qk)
                self.assertEqual(nodes['attention_values'].macs, av)
                self.assertEqual(graph.tensors['scores'].nbytes, 400)
                self.assertEqual(graph.tensors['softmax_fp32'].nbytes, 400)
                self.assertEqual(nodes['probability_cast'].inputs, ('softmax_fp32',))
                spilled = graph.lower(AcceleratorRates(1,1,1,1))
                softmax = next(p for p in spilled if p.name == 'softmax')
                cast = next(p for p in spilled if p.name == 'probability_cast')
                self.assertEqual(softmax.write_bytes, 400)
                self.assertEqual(cast.read_bytes, 400)
                self.assertLess(spilled.index(softmax), spilled.index(cast))
                self.assertEqual(sum(t.nbytes for t in graph.tensors.values() if t.storage == 'weight'), cfg.parameter_count)
                self.assertEqual('key_absorb' in nodes, layout == 'absorbed')
                self.assertNotIn('key_before', nodes['attention_values'].inputs)
                self.assertLess(graph.export()['schedule'].index('cache_commit'),
                                graph.export()['schedule'].index('attention_scores'))

    def test_nope_retains_positional_projection_channels_without_rotary_work(self):
        cfg, _ = self.mla()
        cfg.rope_enabled = False
        cfg.q_lora_rank = 0
        graph = MLAProfile().graph(**dict(vars(cfg), bs=1, batch_size=1, L_seq=17, stage='decode'))
        self.assertNotIn('query_rope', graph.export()['schedule'])
        self.assertEqual(graph.tensors['query_projection'].shape, (1, 8))
        self.assertEqual(graph.tensors['key_before'].nbytes, 32)
        self.assertEqual(graph.tensors['value_before'].nbytes, 64)
        self.assertEqual(sum(t.nbytes for t in graph.tensors.values() if t.storage == 'weight'), cfg.parameter_count)

    def test_live_tensors_survive_branches_until_last_consumer(self):
        graph = OperatorGraph('branch')
        graph.tensor('x', (4,), storage='input')
        graph.tensor('a', (8,)); graph.tensor('b', (16,)); graph.tensor('y', (4,), storage='output')
        graph.add('join', ['a','b'], ['y'])  # Deliberately declared before producers.
        graph.add('left', ['x'], ['a'])
        graph.add('right', ['x'], ['b'])
        self.assertEqual([n.name for n in graph.ordered_nodes()], ['left','right','join'])
        self.assertEqual(graph.peak_activation_bytes(), 28)
        resident = graph.lower(AcceleratorRates(1,1,1,1000))
        spilled = graph.lower(AcceleratorRates(1,1,1,1))
        self.assertGreater(sum(p.memory_bytes for p in spilled), sum(p.memory_bytes for p in resident))
        self.assertEqual(spilled[-1].read_bytes, 24)
        scratch = OperatorGraph('scratch')
        scratch.tensor('x', (4,), storage='input')
        scratch.tensor('y', (4,), storage='output')
        scratch.add('materialize', ['x'], ['y'], workspace_bytes=16)
        self.assertEqual(scratch.peak_activation_bytes(), 24)
        with self.assertRaisesRegex(ValueError, 'explicit intermediate tensors'):
            scratch.lower(AcceleratorRates(1,1,1,1))
        phase = scratch.lower(AcceleratorRates(1,1,1,16))[0]
        self.assertEqual((phase.read_bytes, phase.write_bytes), (4, 4))

    def test_graph_rejects_cycles_missing_producers_and_duplicate_names(self):
        graph = OperatorGraph('bad')
        graph.tensor('a', (1,)); graph.tensor('b', (1,))
        graph.add('first', ['b'], ['a']); graph.add('second', ['a'], ['b'])
        with self.assertRaisesRegex(ValueError, 'cycle'): graph.ordered_nodes()
        graph.nodes.pop()
        with self.assertRaisesRegex(ValueError, 'Missing producer'): graph.ordered_nodes()
        with self.assertRaisesRegex(ValueError, 'Duplicate tensor'): graph.tensor('a',(1,))

    def test_spilled_recurrence_reads_previous_token_before_next_update(self):
        _, graph = self.kda()
        rates = AcceleratorRates(100,100,100,1)
        phases = [p for p in graph.lower(rates) if p.name.startswith('delta_recurrence')]
        first = next(p for p in phases if p.name.endswith(':first'))
        repeated = next(p for p in phases if p.name.endswith(':next'))
        self.assertEqual(first.read_bytes, 0)
        self.assertEqual(first.write_bytes, 144)
        self.assertEqual(repeated.iterations, 4)
        self.assertEqual(repeated.read_bytes, 144)
        self.assertEqual(repeated.write_bytes, 144)
        expanded = list(repeated.expanded())
        self.assertEqual([p.memory_direction for p in expanded if p.memory_bytes], ['read','write']*4)
        self.assertAlmostEqual(sum(p.latency_ns(1) for p in expanded), repeated.latency_ns(1))

    def test_network_waits_for_reads_then_compute_then_directed_writes(self):
        env = simpy.Environment()
        observed = []
        class Executor:
            config = SimpleNamespace(chipsim_config=SimpleNamespace(hbm_access_latency_ns=0))
            trace = io.StringIO()
            def _submit(self, source, destination, size):
                if size:
                    observed.append((env.now, source, destination, size))
                return env.timeout(0, value=env.now*1e12/common.time_granularity)
        execution = Executor(); execution.env = env
        phase = ExecutionPhase(2000, 4000, 'dependent', ordered=True, read_bytes=1000, write_bytes=3000)
        env.process(NetworkExecutionMixin._run_phases(execution, 0, 1, [asdict(phase)], 1e9, {}))
        env.run()
        self.assertEqual([(s,d,b) for _,s,d,b in observed], [(0,1,1000),(1,0,3000)])
        self.assertEqual(observed[1][0], 3*common.time_granularity/1e6)
        self.assertEqual(env.now, 6*common.time_granularity/1e6)

    def test_versioned_memory_admission_and_factory_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            config = HPSim_Config(metrics_config=MetricsConfig(output_dir=folder))
            Simulator(config)
            model = BaseModelConfig.create_from_name('kda-mla-operator-fixture')
            backend = NativeExecutionBackend(config)
            for block_config in model.hybrid_blocks:
                for prefill in (True, False):
                    work = block(64, block_config, is_prefill=prefill)
                    memory = block_config.memory_requirements(64, prefill)
                    self.assertEqual(work.peak_intermediate_store, memory.activation_bytes)
                    stage = 'prefill' if prefill else 'decode'
                    profile = backend.profile_block(work, stage, 1,
                        SimpleNamespace(chiplet_type=block_config.get_accelerators(stage)[0]), 128)[0]
                    self.assertAlmostEqual(profile.total_latency,
                        sum(p.latency_ns(128) for p in profile.execution_phases), delta=1)
            mla = model.hybrid_blocks[1]
            small = block(64, mla, is_prefill=True)
            large = block(128, mla, is_prefill=True)
            self.assertGreater(large.peak_intermediate_store, 2*small.peak_intermediate_store)
            dec = block(65, mla)
            self.assertEqual(dec.additional_memory_mib(common.convert_param_mB(small.states_store)),
                             common.convert_param_mB(dec.peak_intermediate_store+mla.states))

    def test_unsupported_algorithms_and_malformed_execution_fail_explicitly(self):
        with self.assertRaises(ValueError): KDAConfig(algorithm='unknown')
        with self.assertRaises(ValueError): KDAConfig(num_heads=0)
        with self.assertRaises(ValueError): MLAConfig(cache_layout='compressed-but-expanded-compute')
        with self.assertRaises(ValueError): self.kda(stage='unknown')
        with self.assertRaises(ValueError): ExecutionPhase(-1)
        with self.assertRaises(ValueError): ExecutionPhase(1,2,ordered=True,read_bytes=1)
        with self.assertRaises(ValueError): KDAProfile().graph(bs=2,batch_size=1,L_seq=3,stage='prefill')
        with self.assertRaises(ValueError): OperatorBlockConfig(operator=MLAConfig())

    def test_empty_hbm_and_exact_block_identity_during_reservation(self):
        system = mem_sys.__new__(mem_sys)
        system._mem_chiplets = [mem_chiplet(chiplet_type=0, dram_budget=1,
                                           chiplet_id=i, chiplet_loc=(0,i)) for i in range(3)]
        empty, other, target = system._mem_chiplets
        other.content_params['weights'] = [Params(block_id=10, name='Block 10 weights')]
        target.content_params['weights'] = [Params(block_id=1, name='Block 1 weights')]
        with patch.object(common, 'task_parallelism', 'pipeline'):
            self.assertEqual(system.list_allocated_mem_chiplets('Block 1'), [target])
            self.assertEqual(system.check_memchiplets_availability(100, 'Block 1'), 1)
            system.allocate_memchiplet(100, 'Block 1')
            self.assertEqual(target._current_allocated, 100)
            self.assertEqual(other._current_allocated, 0)
            self.assertEqual(empty._current_allocated, 0)
            system.relieve_allocate_memchiplet(100, 'Block 1')
            self.assertEqual(target._current_allocated, 0)

    def test_static_admission_ignores_empty_hbm_and_uses_graph_peak(self):
        model = BaseModelConfig.create_from_name('kda-mla-operator-fixture')
        cfg = model.hybrid_blocks[1]
        required = max(cfg.reservation_bytes(model.max_position_embeddings, stage)
                       for stage in (True, False))
        scheduler = StaticReplicaScheduler.__new__(StaticReplicaScheduler)
        scheduler._max_concurrent_requests = 9999
        empty = SimpleNamespace(content_params={}, dram_budget=1, inuse_budget=0)
        used = SimpleNamespace(content_params={'weights': [SimpleNamespace(block_id=1)]},
                               dram_budget=2.5*required/(1024*1024), inuse_budget=0)
        with patch('Sim.scheduler.static.mem_chiplet', SimpleNamespace), \
             patch('Sim.scheduler.static.Params', SimpleNamespace):
            scheduler.set_max_concurrent_requests(model, SimpleNamespace(_mem_chiplets=[empty, used]))
        self.assertEqual(scheduler._max_concurrent_requests, 2)


if __name__ == '__main__':
    unittest.main()
