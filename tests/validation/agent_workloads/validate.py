#!/usr/bin/env python3
"""Replay normalized agent trajectories on a fixed archived HYDRA design point."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.validation.modern_models.validate import ModernModelValidation


class AgentReplayValidation(ModernModelValidation):
    allows_partial_batches = True

    def select_hardware(self):
        point = super().select_hardware()
        path = self.output_dir / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['scope'] = 'Fixed agent trajectory performance replay; task correctness is not evaluated.'
        manifest['selection'] = 'Selected complete sessions; static batching with full-prompt recomputation.'
        for directory in ('integrations/workloads', 'Sim/request_generator', 'Sim/scheduler'):
            for source in sorted((ROOT / directory).glob('*.py')):
                manifest['source_sha256'][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        for name in ('Sim/entities/agent_workload.py', 'Sim/entities/request.py',
                     'Sim/config/agent_config.py', 'Sim/execution/tool_system.py',
                     'Sim/sim_core.py',
                     'Sim/execution/native.py', 'analytic_profile/modern.py',
                     'tests/validation/agent_workloads/validate.py'):
            manifest['source_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2)+'\n')
        return point

    def configure_command(self, command):
        prefix = '--workload-config.request-generator-config.'
        command.extend([prefix+'generator', 'agent', prefix+'agent-trace-file', str(self.args.trace.resolve())])
        options = prefix+'agent-config.'
        for name in ('input_format', 'provenance_file', 'tool_profiles_file'):
            value = getattr(self.args, name, None)
            if value:
                command.extend([options+name.replace('_', '-'), str(value.resolve() if isinstance(value, Path) else value)])
        for name in ('tool_latency_s', 'session_interval_s', 'user_think_s'):
            value = getattr(self.args, name, None)
            if value is not None:
                command.extend([options+name.replace('_', '-'), str(value)])
        if getattr(self.args, 'task_ids', None):
            command.extend([options+'task-ids', *self.args.task_ids])
        for name in ('stop_when_complete', 'allow_model_mismatch'):
            if getattr(self.args, name, False):
                command.append(options+name.replace('_', '-'))

    @staticmethod
    def verify_complete(run, result):
        from Sim.entities.agent_workload import AgentWorkload
        report = run.read('agent_metrics.json')
        workload = AgentWorkload.load(run.run_dir / 'effective_agent_workload.json')
        if not report or report['completed_sessions'] != len(workload.sessions):
            raise RuntimeError('Agent sessions did not finish, including final tool waits; increase --time-limit.')
        count = sum(len(s.steps) for s in workload.sessions)
        tokens = sum(step.output_tokens for s in workload.sessions for step in s.steps)
        if (report['planned_calls'] != count or report['released_calls'] != count
                or report['completed_calls'] != count or len(report['calls']) != count
                or result['completed_requests'] != count or result['output_tokens'] != tokens
                or abs(result['finished_tokens_per_sec'] * result['elapsed_simulation_s'] - tokens) > 1e-6
                or result['running_requests'] or result['pending_requests'] or result['preempted_requests']):
            raise RuntimeError('Agent call/token accounting did not complete exactly.')
        from Sim.request_generator.agent_request_generator import AgentRequestGenerator
        ticks = AgentRequestGenerator.ticks
        for session in workload.sessions:
            calls = [r for r in report['calls'] if r['session_id'] == session.session_id]
            earliest = ticks(session.arrival_time_s)
            if [r['step_id'] for r in calls] != [s.step_id for s in session.steps]:
                raise RuntimeError('Agent step order mismatch.')
            for step, row in zip(session.steps, calls):
                expected_arrival = earliest + ticks(step.user_delay_before_s)
                if not (row['arrived_tick'] == expected_arrival
                        and row['arrived_tick'] <= row['scheduled_tick'] <= row['first_token_tick'] <= row['completed_tick']
                        and row['input_tokens'] == step.input_tokens and row['output_tokens'] == step.output_tokens
                        and row['tool_completed_tick'] == row['completed_tick'] + row['tool_delay_ticks']):
                    raise RuntimeError('Agent causal timing or token contract failed.')
                if report['tool_execution']['config'] is None and row['tool_delay_ticks'] != ticks(step.tool_delay_s):
                    raise RuntimeError('Stored tool delay was not replayed exactly.')
                earliest = row['tool_completed_tick']
            end = next(r for r in report['sessions'] if r['session_id'] == session.session_id)
            if end['completed_tick'] != earliest:
                raise RuntimeError('Session completion omitted final external work.')
        result['agent_audit'] = {key: report[key] for key in ('completed_sessions', 'completed_calls')}
        result['agent_sessions'] = report['sessions']
        AgentReplayValidation.verify_tools(workload, report)


    @staticmethod
    def verify_tools(workload, report):
        from Sim.execution.tool_system import BaseToolModel, ToolProfile
        execution = report['tool_execution']
        config = execution['config']
        if config is None:
            return
        if execution['cpu_active'] or execution['cpu_queued'] or execution['cpu_workspace_used_bytes']:
            raise RuntimeError('Host CPU resources did not drain.')
        records = execution['calls']
        expected = sum(step.tool_calls for session in workload.sessions for step in session.steps)
        if len(records) != expected:
            raise RuntimeError('Incorrect number of executed tool calls.')
        for session in workload.sessions:
            for step in session.steps:
                rows = [r for r in records if r['session_id'] == session.session_id and r['step_id'] == step.step_id]
                call = next(r for r in report['calls'] if r['session_id'] == session.session_id and r['step_id'] == step.step_id)
                previous = call['completed_tick']
                if len(rows) != len(step.tools):
                    raise RuntimeError('Missing or duplicate tool records.')
                for index, (tool, row) in enumerate(zip(step.tools, rows)):
                    profile = ToolProfile(**config.get('tools', {}).get(tool.name, config.get('default', {})))
                    if not (row['tool_index'] == index and row['name'] == tool.name
                            and row['request_bytes'] == tool.request_bytes and row['response_bytes'] == tool.response_bytes
                            and row['submitted_tick'] == previous
                            and row['service_started_tick'] == previous + row['request_transfer_ticks'] + row['queue_ticks']
                            and row['service_completed_tick'] == row['service_started_tick'] + BaseToolModel.ticks(profile.service_s)
                            and row['completed_tick'] == row['service_completed_tick'] + row['round_trip_ticks'] + row['response_transfer_ticks']):
                        raise RuntimeError('Tool profile timing or serial dependency failed.')
                    previous = row['completed_tick']
                if previous != call['tool_completed_tick']:
                    raise RuntimeError('LLM call released before all tools completed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['qwen3-8b'])
    parser.add_argument('--backends', nargs='+', choices=['hydra_sim', 'hydra_packet', 'chipsim_contended'], default=['hydra_sim'])
    parser.add_argument('--sessions', dest='requests', type=int, default=2)
    parser.add_argument('--time-limit', type=float, default=10.0)
    parser.add_argument('--run-timeout', type=float, default=1800)
    parser.add_argument('--packages', type=int, default=1)
    parser.add_argument('--input-format', choices=['normalized', 'bfcl'], default='normalized')
    parser.add_argument('--provenance-file', type=Path)
    parser.add_argument('--tool-profiles-file', type=Path)
    parser.add_argument('--tool-latency-s', type=float)
    parser.add_argument('--session-interval-s', type=float)
    parser.add_argument('--user-think-s', type=float)
    parser.add_argument('--task-ids', nargs='+')
    parser.add_argument('--stop-when-complete', action='store_true')
    parser.add_argument('--allow-model-mismatch', action='store_true')
    parser.set_defaults(batch_size=1, memory_chiplets=None, dataset_label='bfcl',
                        arrival_interval=0, require_complete=True, virtual_channels=8,
                        transfer_timeout=120, dma_pacing=False, dma_burst_bytes=4096,
                        packet_quantum_bytes=1024, packet_buffer_bytes=16384)
    args = parser.parse_args()
    AgentReplayValidation(args).run()


if __name__ == '__main__':
    main()
