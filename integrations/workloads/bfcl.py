"""Import BFCL multi_turn_base result JSONL using reported per-call token usage.

No prompts are executed, answers generated, or benchmark scores inferred.
The pinned BFCL handler executes each step's tool calls serially.
"""

import hashlib
import ast
import json
from pathlib import Path

from Sim.entities.agent_workload import AgentSession, AgentStep, AgentToolCall, AgentWorkload
from integrations.workloads.base import BaseWorkloadImporter


class BFCLTraceImporter(BaseWorkloadImporter):
    source_revision = '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8'

    @staticmethod
    def get_name():
        return 'bfcl'

    def __init__(self, provenance, tool_latency_s=None, tool_latencies=None,
                 session_interval_s=0.01, user_think_s=0.0):
        required = {'benchmark_revision', 'source_model', 'tokenizer', 'chat_template', 'sampling'}
        if not isinstance(provenance, dict) or not required <= provenance.keys():
            raise ValueError(f'Collection provenance requires {sorted(required)}.')
        if any(not provenance[key] for key in required):
            raise ValueError('Collection provenance fields must not be empty.')
        if (tool_latency_s is None) == (tool_latencies is None):
            raise ValueError('Specify exactly one tool latency assumption or measured sidecar.')
        if tool_latency_s is not None:
            AgentStep.validate_seconds(tool_latency_s, 'tool_latency_s')
        AgentStep.validate_seconds(session_interval_s, 'session_interval_s')
        AgentStep.validate_seconds(user_think_s, 'user_think_s')
        self.provenance = provenance
        self.tool_latency_s, self.tool_latencies = tool_latency_s, tool_latencies
        self.session_interval_s, self.user_think_s = session_interval_s, user_think_s

    @staticmethod
    def token_count(value):
        # Some handlers serialize integral counts as floats. Unknown/zero usage
        # is not a valid zero-cost inference and must be collected again.
        if type(value) not in (int, float) or value <= 0 or not float(value).is_integer():
            raise ValueError('BFCL token usage must contain positive integral counts.')
        return int(value)

    @staticmethod
    def tool_descriptors(entries):
        """Read executed calls without evaluating any benchmark or model code.

        UTF-8 log text is a payload proxy, not a measured HTTP/DMA byte count.
        Missing decoded call text is marked unknown rather than reconstructed.
        """
        results = [entry['content'] for entry in entries if entry.get('role') == 'tool']
        decoded = [entry['model_response_decoded'] for entry in entries
                   if entry.get('role') == 'handler_log' and entry.get('model_response_decoded')]
        calls = decoded[-1] if decoded else [''] * len(results)
        if not isinstance(calls, list) or len(calls) != len(results):
            if not results:
                return ()  # Decoded suggestions that were never executed cost no tool service.
            raise ValueError('Decoded tool calls and execution results cannot be aligned.')
        tools = []
        for call, result in zip(calls, results):
            name = 'unknown'
            if call:
                expression = ast.parse(call, mode='eval').body
                if not isinstance(expression, ast.Call):
                    raise ValueError('Expected a decoded function call expression.')
                if isinstance(expression.func, ast.Name):
                    name = expression.func.id
                elif isinstance(expression.func, ast.Attribute):
                    name = ast.unparse(expression.func)
                else:
                    raise ValueError('Unsupported decoded tool name.')
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, separators=(',', ':'))
            tools.append(AgentToolCall(name, len(call.encode('utf-8')), len(text.encode('utf-8'))))
        return tuple(tools)

    def load(self, path):
        source = Path(path).read_bytes()
        sessions, events, reasoning_steps = [], {}, []
        expected_delays = {}
        for line in source.decode('utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            session_id = record.get('id', '')
            if not session_id.startswith('multi_turn_base_'):
                raise ValueError('The first BFCL adapter supports multi_turn_base result JSONL only.')
            if not all(key in record for key in ('input_token_count', 'output_token_count', 'inference_log')):
                raise ValueError(f'{session_id}: use generated result logs, not raw questions or ground truth.')
            log = record['inference_log']
            if not isinstance(log, list):
                raise ValueError(f'{session_id}: unsupported inference_log structure.')
            turns = []
            for entry in log:
                if isinstance(entry, dict) and 'begin_of_turn_query' in entry:
                    turns.append(entry)
                elif not (isinstance(entry, list) and all(isinstance(e, dict) and e.get('role') == 'state_info' for e in entry)):
                    raise ValueError(f'{session_id}: unrecognized inference log entry.')
            inputs, outputs = record['input_token_count'], record['output_token_count']
            if not turns or len(inputs) != len(turns) or len(outputs) != len(turns):
                raise ValueError(f'{session_id}: turn/token array mismatch.')
            steps = []
            expected_delays[session_id] = {}
            for turn_id, turn in enumerate(turns):
                in_turn, out_turn = inputs[turn_id], outputs[turn_id]
                if not isinstance(in_turn, list) or not isinstance(out_turn, list) or not in_turn or len(in_turn) != len(out_turn):
                    raise ValueError(f'{session_id}: invalid per-turn token counts.')
                expected = {f'step_{i}' for i in range(len(in_turn))}
                if set(turn) != expected | {'begin_of_turn_query'}:
                    raise ValueError(f'{session_id}: step/token array mismatch.')
                for index, (prompt, output) in enumerate(zip(in_turn, out_turn)):
                    step_id = f'turn_{turn_id}.step_{index}'
                    entries = turn[f'step_{index}']
                    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
                        raise ValueError(f'{session_id}/{step_id}: invalid step log.')
                    if sum(e.get('role') == 'assistant' for e in entries) != 1:
                        raise ValueError(f'{session_id}/{step_id}: expected one assistant response.')
                    calls = sum(e.get('role') == 'tool' for e in entries)
                    tools = self.tool_descriptors(entries)
                    if any(e.get('reasoning_content') for e in entries if e.get('role') == 'assistant'):
                        reasoning_steps.append(f'{session_id}/{step_id}')
                    delay = 0.0
                    if calls:
                        if self.tool_latencies is None:
                            delay = self.tool_latency_s * calls
                        else:
                            values = self.tool_latencies.get(session_id, {}).get(step_id)
                            if not isinstance(values, list) or len(values) != calls:
                                raise ValueError(f'{session_id}/{step_id}: tool timing count mismatch.')
                            for value in values:
                                AgentStep.validate_seconds(value, 'measured tool latency')
                            delay = sum(values)
                        expected_delays[session_id][step_id] = calls
                    steps.append(AgentStep(step_id, self.token_count(prompt), self.token_count(output),
                                           calls, delay, self.user_think_s if turn_id and index == 0 else 0.0, tools))
                    events[f'{session_id}/{step_id}'] = [e.get('content') for e in entries if e.get('role') == 'handler_log']
            sessions.append(AgentSession(session_id, len(sessions)*self.session_interval_s, tuple(steps)))
        if self.tool_latencies is not None:
            expected_keys = {(sid, step) for sid, rows in expected_delays.items() for step in rows}
            actual_keys = {(sid, step) for sid, rows in self.tool_latencies.items() for step in rows}
            if actual_keys != expected_keys:
                raise ValueError('Measured tool timing sidecar contains missing or extra step IDs.')
        metadata = dict(benchmark='BFCL', category='multi_turn_base',
                        importer_revision=self.source_revision, collection=self.provenance,
                        source_file=Path(path).name, source_sha256=hashlib.sha256(source).hexdigest(),
                        token_source='BFCL reported input/output_token_count; not retokenized',
                        tool_execution='serial within each step, as in the pinned BFCL executor',
                        tool_latency_source='assumed constant per call' if self.tool_latencies is None else 'user supplied per-call measurements',
                        tool_latency_s=self.tool_latency_s, tool_latencies=self.tool_latencies,
                        user_think_s=self.user_think_s, session_interval_s=self.session_interval_s,
                        handler_events=events,
                        reasoning_steps=reasoning_steps,
                        payload_source='UTF-8 decoded call expression and tool response text; excludes protocol headers and memory traffic',
                        scope='Recorded trajectory replay, independent of task correctness. Source LLM latency is not replayed.')
        return AgentWorkload(sessions, metadata)
