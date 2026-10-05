"""BFCL provenance/shape checks and causal, completion-driven agent replay."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import simpy

import Sim.common as common
from Sim.config.model_config import BaseModelConfig
from Sim.config.sys_config import TraceRequestGeneratorConfig
from Sim.entities.agent_workload import AgentSession, AgentStep, AgentWorkload
from Sim.metrics.monitor import request_counter, tokens_monitor
from Sim.request_generator.agent_request_generator import AgentRequest, AgentRequestGenerator
from Sim.request_generator.base_request_source import BaseRequestSource
from integrations.workloads import BaseWorkloadImporter, BFCLTraceImporter
from tests.validation.chipsim_validation.collect_checkpoints import CheckpointRun
from tests.validation.prepare_inputs import ValidationInputs


class BFCLImporterTest(unittest.TestCase):
    def setUp(self):
        self.provenance = ValidationInputs.bfcl_provenance()
        self.records = ValidationInputs.bfcl_records()

    def load(self, records=None, **kwargs):
        if not kwargs:
            kwargs = dict(tool_latency_s=0.002, user_think_s=0.1)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'results.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in (self.records if records is None else records)))
            return BFCLTraceImporter(self.provenance, **kwargs).load(path)

    def test_factory_and_nested_turns_preserve_calls_tokens_and_waits(self):
        self.assertIs(BaseWorkloadImporter.create_from_name('bfcl'), BFCLTraceImporter)
        self.assertIs(BaseRequestSource.create_from_name('agent'), AgentRequestGenerator)
        self.assertEqual(BaseRequestSource.create_from_name('trace').get_name(), 'trace')
        for factory in (BaseWorkloadImporter, BaseRequestSource):
            with self.assertRaises(ValueError):
                factory.create_from_name('unknown')
        workload = self.load()
        self.assertEqual([len(s.steps) for s in workload.sessions], [3, 2])
        self.assertEqual([s.tool_delay_s for s in workload.sessions[0].steps], [.004, .002, 0])
        self.assertEqual(workload.sessions[0].steps[-1].user_delay_before_s, .1)
        self.assertEqual(sum(s.output_tokens for session in workload.sessions for s in session.steps), 11)
        self.assertTrue(workload.metadata['collection']['is_synthetic'])
        with TemporaryDirectory() as directory:
            path = Path(directory)/'normalized.json'
            workload.write(path)
            self.assertEqual(AgentWorkload.load(path).sessions, workload.sessions)
            self.assertEqual(len(AgentWorkload.load(path, 1).sessions), 1)

    def test_measured_serial_tool_waits_and_exact_sidecar_coverage(self):
        timings = {'multi_turn_base_fixture_0': {'turn_0.step_0': [.01, .02], 'turn_0.step_1': [.04]},
                   'multi_turn_base_fixture_1': {'turn_0.step_0': [.005]}}
        workload = self.load(tool_latencies=timings)
        self.assertEqual(workload.sessions[0].steps[0].tool_delay_s, .03)
        for replacement in ([], [float('nan'), .01], [-1, .01]):
            invalid = deepcopy(timings)
            invalid['multi_turn_base_fixture_0']['turn_0.step_0'] = replacement
            with self.assertRaises(ValueError):
                self.load(tool_latencies=invalid)
        timings['unused'] = {'turn_0.step_0': [1]}
        with self.assertRaisesRegex(ValueError, 'extra step'):
            self.load(tool_latencies=timings)

    def test_reject_raw_tasks_missing_usage_duplicate_ids_and_misaligned_steps(self):
        with self.assertRaisesRegex(ValueError, 'generated result'):
            self.load([{'id': 'multi_turn_base_0', 'question': []}])
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.load([self.records[0], self.records[0]])
        for value in (0, -1, None, True, 2.5, float('nan')):
            rows = deepcopy(self.records)
            rows[0]['output_token_count'][0][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(rows)
        rows = deepcopy(self.records)
        rows[0]['inference_log'][1]['step_2'] = []
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.load(rows)

    def test_handler_failures_are_preserved_without_claiming_task_success(self):
        self.records[0]['inference_log'][2]['step_0'].append(
            {'role': 'handler_log', 'content': 'Error decoding the model response.'})
        workload = self.load()
        self.assertEqual(len(workload.sessions[0].steps), 3)
        self.assertIn('Error decoding', workload.metadata['handler_events']['multi_turn_base_fixture_0/turn_1.step_0'][0])
        self.assertNotIn('success', workload.metadata)

    def test_backend_pair_rejects_changed_tool_wait_with_identical_token_rows(self):
        with TemporaryDirectory() as directory:
            runs = []
            for name in ('native', 'packet'):
                parent = Path(directory)/name
                run = parent/'run'
                run.mkdir(parents=True)
                (run/'system_snapshot.json').write_text('{}')
                (run/'effective_workload.csv').write_text('num_prefill_tokens,num_decode_tokens\n32,2\n')
                (run/'effective_agent_workload.json').write_text(json.dumps({'tool_delay_s': 1}))
                runs.append(CheckpointRun(parent))
            runs[0].verify_inputs(runs[1])
            (runs[1].run_dir/'effective_agent_workload.json').write_text(json.dumps({'tool_delay_s': 2}))
            with self.assertRaisesRegex(ValueError, 'effective_agent'):
                runs[0].verify_inputs(runs[1])


class AgentReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.model = BaseModelConfig.create_from_name('deepseek-decoder-fixture')
        self.env = simpy.Environment()
        self.patch = patch.multiple(common, batch_size=1, local_scheduler='static',
                                    task_parallelism='pipeline', mapping_strategy='static', inj_finish=False)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.queue_patch = patch.object(common.RequestQueue, 'outstanding', [])
        self.queue_patch.start()
        self.addCleanup(self.queue_patch.stop)
        request_counter.reset()
        tokens_monitor.reset()
        self.addCleanup(request_counter.reset)
        self.addCleanup(tokens_monitor.reset)

    def generator(self, sessions):
        path = self.directory/'trace.json'
        AgentWorkload(sessions, {'is_synthetic': True}).write(path)
        return AgentRequestGenerator(self.env.event(), TraceRequestGeneratorConfig(
            generator='agent', agent_trace_file=str(path), num_requests=len(sessions)), self.model, self.env)

    def test_completion_drives_next_call_and_sessions_overlap(self):
        # LLM duration is independently controlled at 7 us in this unit test.
        sessions = [AgentSession('a', 0, (AgentStep('0', 8, 3, 2, .000011),
                                          AgentStep('1', 16, 1, 1, .000005, .000003))),
                    AgentSession('b', .000002, (AgentStep('0', 8, 2),))]
        generator = self.generator(sessions)
        def complete(request):
            yield self.env.timeout(7)
            request.on_completion(self.env.now)
        def worker():
            while not common.inj_finish:
                while common.RequestQueue.outstanding:
                    request = common.RequestQueue.outstanding.pop(0)
                    self.env.process(complete(request))
                yield self.env.timeout(1)
        self.env.process(worker())
        self.env.run(until=generator.action)
        rows = generator.records
        a = [r for r in rows if r['session_id'] == 'a']
        b = next(r for r in rows if r['session_id'] == 'b')
        self.assertEqual(a[1]['arrived_tick'], a[0]['completed_tick'] + 11 + 3)
        self.assertLess(b['arrived_tick'], a[0]['completed_tick'])
        self.assertEqual(generator.completed['a'], a[1]['completed_tick'] + 5)
        self.assertTrue(common.inj_finish)
        self.assertEqual(request_counter.total_requests, 3)

    def test_cutoff_does_not_release_dependent_calls_or_finish_waiting_session(self):
        generator = self.generator([AgentSession('a', 0, (AgentStep('0', 8, 2, 1, 1), AgentStep('1', 8, 2)))])
        self.env.run(until=1)
        generator.requests[generator.records[0]['request_id']].on_completion(1)
        self.env.run(until=10)
        generator.finalize(self.directory, 10)
        report = json.loads((self.directory/'agent_metrics.json').read_text())
        self.assertEqual((report['planned_calls'], report['released_calls'], report['completed_calls']), (2, 1, 1))
        self.assertEqual(report['completed_sessions'], 0)
        self.assertFalse(common.inj_finish)

    def test_single_output_token_uses_prefill_only_and_signals_once(self):
        request = AgentRequest(self.env, AgentStep('0', 8, 1), self.model)
        request.fill_request()
        self.assertFalse(request.has_decode())
        request.step_processing(2)
        self.assertTrue(request.step_processing(9))
        request.on_completion(9)
        request.on_completion(9)
        self.assertEqual(request.completion.value, 9)
        self.assertEqual(request.first_token_at, 9)

    def test_reject_oversize_context_and_batch_deadlock_before_running(self):
        steps = (AgentStep('0', self.model.max_position_embeddings, 2),)
        with self.assertRaisesRegex(ValueError, 'context limit'):
            self.generator([AgentSession('a', 0, steps)])
        with patch.object(common, 'batch_size', 2), self.assertRaisesRegex(ValueError, 'batch size 1'):
            self.generator([AgentSession('a', 0, (AgentStep('0', 8, 2),))])
        self.assertEqual(AgentRequestGenerator.ticks(.000011), 11)
        self.assertEqual(AgentRequestGenerator.ticks(.0000001), 1)

    def test_invalid_trace_contracts(self):
        for kwargs in (dict(input_tokens=0), dict(output_tokens=True), dict(tool_delay_s=-1),
                       dict(tool_calls=0, tool_delay_s=1), dict(user_delay_before_s=float('inf'))):
            with self.assertRaises(ValueError):
                replace(AgentStep('0', 8, 2), **kwargs)
        with self.assertRaises(ValueError):
            AgentSession('a', -1, (AgentStep('0', 8, 2),))


if __name__ == '__main__':
    unittest.main()
