"""Package locality, capacity, communication contention and feedback contracts."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
import simpy

import Sim.common as common
from Sim.config.model_config import BaseModelConfig
from Sim.config.package_config import PackageConfig
from Sim.config.sys_config import BaseChipletPlacerConfig
from Sim.config import utils
from Sim.entities.package_system import (BasePackagePartition, PackageGraph,
                                         PackageMemorySystem, PackageStaticMapper)
from Sim.execution.package_network import BasePackageNetwork
from Sim.execution.native import NativeExecutionBackend
from Sim.entities.mem_chiplet import intermediate
from integrations.packet.network import PacketNetwork
from tests.validation.multi_package.audit import PackageRunAudit


class PackageNetworkTest(unittest.TestCase):
    def test_failed_summary_closes_both_traces_and_remains_idempotent(self):
        with TemporaryDirectory() as directory:
            network = BasePackageNetwork.create_from_name('alpha_beta')(
                PackageConfig(count=2), simpy.Environment(), Path(directory))
            trace, blocks = network.trace, network.blocks
            with patch.object(Path, 'write_text', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    network.close()
            self.assertTrue(trace.closed and blocks.closed)
            network.close()

    def test_second_trace_open_failure_closes_first_trace(self):
        import io
        trace = io.StringIO()
        with patch.object(Path, 'open', side_effect=[trace, OSError('disk full')]):
            with self.assertRaisesRegex(OSError, 'disk full'):
                BasePackageNetwork.create_from_name('alpha_beta')(
                    PackageConfig(count=2), simpy.Environment(), Path('/unused'))
        self.assertTrue(trace.closed)

    def run_transfers(self, endpoints, **overrides):
        with TemporaryDirectory() as directory:
            env = simpy.Environment()
            config = PackageConfig(count=4, bandwidth_gbps=1, efficiency=1, latency_ns=10000)
            config = replace(config, **overrides)
            network = BasePackageNetwork.create_from_name('alpha_beta')(config, env, Path(directory))
            for source, destination in endpoints:
                network.transfer(env, source, destination, 1000, 100, {})
            env.run()
            network.close()
            network.close()
            records = [json.loads(row) for row in (Path(directory) / 'package_transfers.jsonl').read_text().splitlines()]
            return records, json.loads((Path(directory) / 'package_network_summary.json').read_text())

    def test_alpha_beta_units_and_port_contention(self):
        # Independently calculated: 10 us + 1000 bytes / (1e9 bytes/s) = 11 us.
        for endpoints in ([(0, 1), (0, 2)], [(0, 2), (1, 2)]):
            rows, summary = self.run_transfers(endpoints)
            self.assertEqual([r['completed_tick'] for r in rows], [11, 22])
            self.assertEqual(summary['completed_bytes'], 2000)
            self.assertEqual(summary['inflight_at_cutoff'], 0)

    def test_full_duplex_and_independent_packages_overlap(self):
        for endpoints in ([(0, 1), (1, 0)], [(0, 1), (2, 3)]):
            rows, _ = self.run_transfers(endpoints)
            self.assertEqual([r['completed_tick'] for r in rows], [11, 11])

    def test_lower_bandwidth_increases_time_without_changing_bytes(self):
        rows, _ = self.run_transfers([(0, 1)], efficiency=.5)
        self.assertEqual(rows[0]['completed_tick'], 12)
        rows, _ = self.run_transfers([(0, 1)], latency_ns=20000)
        self.assertEqual(rows[0]['completed_tick'], 21)

    def test_invalid_config_and_cutoff_are_explicit(self):
        for args in (dict(count=0), dict(count=1.5), dict(bandwidth_gbps=0),
                     dict(latency_ns=float('nan')), dict(efficiency=1.1), dict(token_bytes=0)):
            with self.assertRaises(ValueError): PackageConfig(**args)
        with self.assertRaises(ValueError): BasePackageNetwork.create_from_name('unknown')
        with TemporaryDirectory() as directory:
            env = simpy.Environment()
            network = BasePackageNetwork.create_from_name('alpha_beta')(PackageConfig(count=2), env, Path(directory))
            with self.assertRaises(ValueError): network.transfer(env, 0, 0, 1, 1, {})
            network.transfer(env, 0, 1, 1e9, 1, {})
            env.run(until=1)
            network.close()
            summary = json.loads((Path(directory) / 'package_network_summary.json').read_text())
            self.assertEqual(summary['inflight_at_cutoff'], 1)


class PackagePlacementTest(unittest.TestCase):
    def test_failed_transfer_rolls_back_destination_and_releases_bandwidth(self):
        graph = self.graph(2)
        with patch.object(utils, 'used_bws', [128, 256]):
            memory = PackageMemorySystem(graph, {})
        env = simpy.Environment()
        item = intermediate(size=1, block_id=0, name='activation', chip_id=0, req_id=0)
        memory.load_data_byid(0, item, 0)

        class FailingFabric:
            def transfer(self, env, *args):
                return env.process(self.fail(env))

            def fail(self, env):
                yield env.timeout(1)
                raise RuntimeError('failed fabric')

        backend = NativeExecutionBackend()
        backend.package_network = FailingFabric()
        with self.assertRaisesRegex(RuntimeError, 'failed fabric'):
            env.run(until=env.process(backend.transfer_package_input(env, graph, memory, item, 24)))
        self.assertEqual(memory.get_mem_byid(0).inuse_budget, 1)
        self.assertEqual(memory.get_mem_byid(24).inuse_budget, 0)
        self.assertEqual(memory._current_used, 1)
        self.assertEqual(item.chip_id, 0)
        for node in (0, 24):
            self.assertEqual(graph.graph.nodes[graph.id_to_coord[node]]['bw_inuse'], 0)

    def graph(self, count):
        # Real 6x4 package geometry, with 16 HBMs and eight compute chiplets.
        local = BaseChipletPlacerConfig().chip_graph
        for coord, attrs in local.graph.nodes(data=True):
            memory = attrs['id'] < 16
            attrs.update(chiplet_type=utils.chiplet_types_dict['HBM3' if memory else 'tscs_p'],
                         **{'dram(g)': 16 if memory else 0, 'bandwidth': 600, 'bw_inuse': 0})
        return PackageGraph(local, count)

    def test_graph_is_disconnected_and_coordinates_preserve_local_paths(self):
        graph = self.graph(4)
        self.assertEqual(nx.number_connected_components(graph.graph), 4)
        self.assertEqual(graph.num_nodes, 96)
        self.assertEqual(graph.package_of(95), 3)
        for source, destination in graph.graph.edges:
            self.assertEqual(graph.package_of(graph.coord_to_id[source]), graph.package_of(graph.coord_to_id[destination]))
        self.assertIsNone(graph.find_unused_path((0, 0), (6, 0), 1))
        self.assertEqual(len(graph.find_unused_path((6, 0), (11, 3), 1)), 8)

    def test_partition_factory_and_mapper_keep_both_phases_local(self):
        model = BaseModelConfig.create_from_name('deepseek-v3-text')
        partition = BasePackagePartition.create_from_name('contiguous')
        stages = partition.assign(model, 4)
        self.assertEqual([list(stages.values()).count(p) for p in range(4)], [16, 15, 15, 15])
        self.assertEqual(list(stages.values()), sorted(stages.values()))
        with self.assertRaises(ValueError): partition.assign(model, 62)
        mapper = PackageStaticMapper(self.graph(4), stages)
        self.assertTrue(mapper.is_eligible(SimpleNamespace(chiplet_id=24), 16))
        self.assertFalse(mapper.is_eligible(SimpleNamespace(chiplet_id=0), 16))

    def test_deepseek_capacity_failure_is_atomic_and_four_packages_fit(self):
        model = BaseModelConfig.create_from_name('deepseek-v3-text')
        with patch.object(common, 'ByteperParam', 1), patch.object(utils, 'used_bws', [128, 256]):
            for count in (2, 3, 4):
                graph = self.graph(count)
                stages = BasePackagePartition.create_from_name('contiguous').assign(model, count)
                memory = PackageMemorySystem(graph, stages)
                if count < 4:
                    with self.assertRaisesRegex(MemoryError, 'local HBM'):
                        memory.load_model('bw', model, [], [])
                    self.assertEqual(memory.blocks_alloc, {})
                    self.assertEqual(memory._current_used, 0)
                    self.assertTrue(all(m.inuse_budget == 0 for m in memory._mem_chiplets))
                else:
                    memory.load_model('bw', model, [], [])
                    self.assertEqual(len(memory.blocks_alloc), 61)
                    total_bytes = sum(model.hybrid_blocks[i].parameter_count for i in model.block_type_sequence)
                    self.assertEqual(memory._current_used * 1024**2, total_bytes)
                    for layer, hbm in memory.blocks_alloc.items():
                        self.assertEqual(graph.package_of(hbm), stages[layer])
                    self.assertTrue(all(m.inuse_budget <= m.dram_budget for m in memory._mem_chiplets))

    def test_packet_mesh_validator_rejects_package_shortcuts(self):
        graph = self.graph(2)
        snapshot = dict(chiplet_ids=list(range(48)), mesh_rows=12, nodes_per_package=24,
                        package_count=2, network_frequency_hz=1e9, link_width_bits=256,
                        packet_quantum_bytes=1024, packet_buffer_bytes=4096,
                        router_latency_cycles=1, link_latency_cycles=1, hbm_access_latency_ns=50,
                        adjacency=nx.to_numpy_array(graph.graph, weight=None).tolist())
        network = PacketNetwork.__new__(PacketNetwork)
        network.handle = None
        network.config = snapshot
        network._validate()
        # A seam between packages is adjacent in the global coordinate array,
        # but physically absent from the local network topology.
        snapshot['adjacency'][20][24] = 1
        with self.assertRaisesRegex(ValueError, 'within packages'):
            network._validate()


class PackageAuditTest(unittest.TestCase):
    def test_independent_trace_oracle_rejects_wrong_bytes_and_early_consumption(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            root.mkdir()
            setup = dict(chiplets=[dict(id=i, package_id=i // 2) for i in range(4)],
                         links=[[0, 1], [2, 3]], weight_mapping={0: 0, 1: 2},
                         task_mapping={p: {0: 1, 1: 3} for p in ('P', 'D')},
                         packages=dict(layer_packages={0: 0, 1: 1}))
            config = dict(workload_config=dict(model='deepseek-decoder-fixture', bytes_per_param='1'),
                          package_config=dict(token_bytes='4'))
            metrics = dict(completed_requests=1, running_requests=0, pending_requests=0, preempted_requests=0, output_tokens=2)
            summary = dict(submitted_transfers=3, completed_transfers=3, inflight_at_cutoff=0, completed_bytes=260)
            for name, data in (('system_snapshot', setup), ('config', config), ('metrics', metrics),
                               ('package_network_summary', summary)):
                (root / (name + '.json')).write_text(json.dumps(data))
            (root / 'effective_workload.csv').write_text('num_prefill_tokens,num_decode_tokens\n1,1\n')
            blocks = []
            for index, start in enumerate((0, 3, 6, 9)):
                for event, tick in (('start', start), ('complete', start + 1)):
                    blocks.append(dict(package=index % 2, block_id=index % 2, request_ids=[0],
                                       stage='prefill' if index < 2 else 'decode', event=event, tick=tick))
            moves = []
            for index, queued in enumerate((1, 4, 7)):
                source, destination = index % 2, 1 - index % 2
                moves.append(dict(source_hbm=2 * source, destination_hbm=2 * destination,
                    source_package=source, destination_package=destination, request_id=0,
                    producer_block=source, kind='token_feedback' if index == 1 else 'activation',
                    bytes=4 if index == 1 else 128, queued_tick=queued, start_tick=queued, completed_tick=queued + 1))
            def write_records():
                for name, rows in (('package_blocks', blocks), ('package_transfers', moves)):
                    (root / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            write_records()
            audit = PackageRunAudit(Path(directory))
            self.assertTrue(audit.verify()['passed'])
            moves[0]['bytes'] = 256
            write_records()
            with self.assertRaisesRegex(ValueError, 'Wrong payload'): audit.verify()
            moves[0]['bytes'] = 128
            moves[0]['completed_tick'] = 4
            write_records()
            with self.assertRaisesRegex(ValueError, 'before package transfer'): audit.verify()


if __name__ == '__main__':
    unittest.main()
