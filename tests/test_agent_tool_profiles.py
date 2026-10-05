"""Independent service-time arithmetic, CPU contention and real-log contracts."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import simpy
import tyro

from Sim.config.agent_config import AgentTraceConfig
from Sim.config.model_config import BaseModelConfig
from Sim.config.sys_config import TraceRequestGeneratorConfig
from Sim.entities.agent_workload import AgentSession, AgentStep, AgentToolCall, AgentWorkload
from Sim.execution.tool_system import ToolExecutionSystem
from Sim.request_generator.agent_request_generator import AgentRequestGenerator
import Sim.common as common
from Sim.metrics.monitor import request_counter


ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT/'tests/validation/agent_workloads/real_bfcl'


class ToolProfilesTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = simpy.Environment()

    def system(self, kind='external', workers=1, memory=100, workspace=0, **kwargs):
        config = dict(schema_version=1, default=dict(kind=kind, service_s=.000010,
            provenance='Unit-test assumption', workspace_bytes=workspace, **kwargs),
            cpu=dict(workers=workers, memory_bytes=memory, target_machine='test CPU'))
        path = self.directory/'profiles.json'
        path.write_text(json.dumps(config))
        return ToolExecutionSystem(self.env, path)

    @staticmethod
    def step(count=1, request=0, response=0):
        return AgentStep('0', 8, 2, count, tools=tuple(AgentToolCall('test', request, response) for _ in range(count)))

    def test_external_service_rtt_and_bidirectional_payload_time(self):
        system = self.system(round_trip_s=.000005, bandwidth_gbps=1)
        # 1000/1e9 + 10 us service + 5 us RTT + 2000/1e9 = 18 us.
        self.env.run(until=self.env.process(system.execute('a', self.step(request=1000, response=2000))))
        self.assertEqual(self.env.now, 18)
        record = system.records[0]
        self.assertEqual((record['request_transfer_ticks'], record['response_transfer_ticks']), (1, 2))
        self.assertEqual(record['queue_ticks'], 0)

    def test_external_sessions_overlap_but_tools_within_a_step_are_serial(self):
        system = self.system()
        self.env.process(system.execute('a', self.step(2)))
        self.env.process(system.execute('b', self.step()))
        self.env.run()
        self.assertEqual([r['completed_tick'] for r in system.records], [10, 10, 20])

    def test_cpu_worker_and_workspace_capacity_both_limit_concurrency(self):
        for workers, memory, workspace, expected in ((1, 100, 0, [10, 20]),
                (2, 100, 0, [10, 10]), (2, 100, 100, [10, 20])):
            with self.subTest(workers=workers, workspace=workspace):
                self.env = simpy.Environment()
                system = self.system('cpu_profile', workers, memory, workspace)
                for sid in ('a', 'b'):
                    self.env.process(system.execute(sid, self.step()))
                self.env.run()
                self.assertEqual([r['completed_tick'] for r in system.records], expected)
                self.assertEqual(system.snapshot()['cpu_workspace_used_bytes'], 0)
                self.assertEqual(system.cpu.count, 0)

    def test_interrupted_cpu_memory_wait_is_cancelled_without_leaking_capacity(self):
        system = self.system('cpu_profile', workers=2, memory=100, workspace=100)
        self.env.process(system.execute('a', self.step()))
        def interrupted():
            try:
                yield from system.execute('b', self.step())
            except simpy.Interrupt:
                pass
        other = self.env.process(interrupted())
        def cancel():
            yield self.env.timeout(1)
            other.interrupt()
        self.env.process(cancel())
        self.env.run()
        self.assertEqual(system.memory.level, system.memory.capacity)
        self.assertEqual(len(system.memory.get_queue), 0)
        self.assertEqual(system.cpu.count, 0)

    def test_profile_rejects_impossible_workspace_and_double_counted_wait(self):
        with self.assertRaisesRegex(ValueError, 'capacity'):
            self.system('cpu_profile', workspace=101)
        system = self.system()
        workload = AgentWorkload([AgentSession('a', 0, (replace(self.step(), tool_delay_s=1),))], {})
        with self.assertRaisesRegex(ValueError, 'double counting'):
            system.validate(workload)
        with self.assertRaises(ValueError):
            self.system(bandwidth_gbps=0)


class RealBFCLInputTest(unittest.TestCase):
    def options(self):
        # Exercise the same nested dataclass Tyro uses under HPSim_Config.
        return tyro.cli(AgentTraceConfig, args=[
            '--input-format', 'bfcl', '--provenance-file', str(REAL/'provenance.json'),
            '--tool-profiles-file', str(REAL/'assumed_host_cpu.json'), '--session-interval-s', '0',
            '--task-ids', 'multi_turn_base_0', 'multi_turn_base_1'])

    def test_raw_tyro_input_preserves_real_counts_reasoning_and_payloads(self):
        workload = self.options().load(REAL/'qwen3_8b_fc_base_0_1.jsonl', 2)
        self.assertEqual([len(s.steps) for s in workload.sessions], [16, 9])
        self.assertEqual([sum(s.output_tokens for s in session.steps) for session in workload.sessions], [10183, 2750])
        self.assertEqual(workload.sessions[0].steps[0].tools[0].name, 'mkdir')
        self.assertGreater(workload.sessions[0].steps[0].tools[0].request_bytes, 0)
        self.assertTrue(workload.metadata['reasoning_steps'])
        model = BaseModelConfig.create_from_name('qwen3-8b')
        self.assertTrue(workload.check_model(model)['identity_matches'])
        with self.assertRaisesRegex(ValueError, 'differs'):
            workload.check_model(BaseModelConfig.create_from_name('deepseek-decoder-fixture'))
        self.assertFalse(workload.check_model(BaseModelConfig.create_from_name('deepseek-decoder-fixture'), True)['identity_matches'])

    def test_custom_normalized_input_roundtrip_and_task_selection(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'custom.json'
            source = self.options().load(REAL/'qwen3_8b_fc_base_0_1.jsonl', 2)
            source.write(path)
            loaded = AgentTraceConfig(task_ids=['multi_turn_base_1']).load(path, 1)
            self.assertEqual(loaded.sessions, (source.sessions[1],))
            with self.assertRaisesRegex(ValueError, 'absent'):
                AgentTraceConfig(task_ids=['missing']).load(path, 1)
            with self.assertRaisesRegex(ValueError, 'already contain'):
                AgentTraceConfig(tool_latency_s=1).load(path, 1)

    def test_cutoff_during_tool_service_and_completion_stop(self):
        for cutoff in (True, False):
            with TemporaryDirectory() as directory, patch.multiple(common, batch_size=1,
                    local_scheduler='static', task_parallelism='pipeline', mapping_strategy='static', inj_finish=False), \
                    patch.object(common.RequestQueue, 'outstanding', []):
                request_counter.reset()
                path = Path(directory)/'custom.json'
                AgentWorkload([AgentSession('a', 0, (AgentStep('0', 8, 2, 1, .000010),))], {}).write(path)
                env = simpy.Environment()
                done = env.event()
                config = TraceRequestGeneratorConfig(agent_trace_file=str(path),
                    agent_config=AgentTraceConfig(stop_when_complete=True))
                generator = AgentRequestGenerator(done, config, BaseModelConfig.create_from_name('deepseek-decoder-fixture'), env)
                def worker():
                    yield env.timeout(1)
                    common.RequestQueue.outstanding[0].on_completion(env.now)
                env.process(worker())
                env.run(until=5 if cutoff else done)
                generator.finalize(directory, env.now)
                report = json.loads((Path(directory)/'agent_metrics.json').read_text())
                self.assertEqual(report['completed_sessions'], 0 if cutoff else 1)
                self.assertEqual(done.triggered, not cutoff)
                if not cutoff:
                    self.assertEqual(env.now, 11)
                request_counter.reset()


if __name__ == '__main__':
    unittest.main()
