"""Replay LLM calls only after their predecessor and external waits complete."""

import csv
import json
from pathlib import Path

import Sim.common as common
from Sim.entities.request import Request
from Sim.execution.tool_system import BaseToolModel, ToolExecutionSystem
from Sim.metrics.monitor import request_counter
from Sim.request_generator.base_request_source import BaseRequestSource


class AgentRequest(Request):
    def __init__(self, env, step, model_config):
        super().__init__(arrived_at=env.now, num_prefill_tokens=step.input_tokens,
                         num_decode_tokens=step.output_tokens - 1,
                         context_length=step.input_tokens, model_config=model_config)
        self.completion = env.event()
        self.first_token_at = None
        self.step = step

    def retry_request(self):
        # Preserve the SimPy completion event when retrying the same call.
        self.fill_request()
        return self

    def step_processing(self, timestep=0):
        if self._process_idx == 0:
            self.first_token_at = timestep
        completed = super().step_processing(timestep)
        return completed

    def on_completion(self, timestep):
        if self.completion.triggered:
            return
        super().on_completion(timestep)
        if not self.completion.triggered:
            self.completion.succeed(timestep)


class AgentRequestGenerator(BaseRequestSource):
    @staticmethod
    def get_name():
        return 'agent'

    def __init__(self, sim_done, tr_config, mod_config, env):
        if (common.local_scheduler != 'static' or common.batch_size != 1
                or common.task_parallelism != 'pipeline' or common.mapping_strategy != 'static'):
            raise ValueError('Agent replay requires static scheduler with batch size 1, static mapping and pipeline execution.')
        lengths = tr_config.trace_length_generator_config
        if lengths.request_type != 'e2e' or lengths.prefill_scale_factor != 1 or lengths.decode_scale_factor != 1:
            raise ValueError('Agent replay requires unscaled end-to-end calls.')
        self.options = tr_config.agent_config
        self.workload = self.options.load(tr_config.agent_trace_file, tr_config.num_requests)
        self.model_check = self.workload.check_model(mod_config, self.options.allow_model_mismatch)
        self.tools = ToolExecutionSystem(env, self.options.tool_profiles_file)
        self.tools.validate(self.workload)
        self.sim_done = sim_done
        self.env, self.model = env, mod_config
        self.records = []
        self.completed = {}
        self.requests = {}
        count = sum(len(session.steps) for session in self.workload.sessions)
        if count > common.max_requests_inj:
            raise ValueError('Agent trace exceeds max_requests_inj; select fewer sessions.')
        for session in self.workload.sessions:
            for step in session.steps:
                # Unlike legacy CSV replay, never truncate a dependent trajectory.
                if step.input_tokens + step.output_tokens > min(lengths.max_tokens, mod_config.max_position_embeddings):
                    raise ValueError(f'{session.session_id}/{step.step_id} exceeds the context limit; no truncation is allowed.')
                mod_config.validate_request_lengths(step.input_tokens, step.input_tokens, step.output_tokens - 1)
        self.action = env.process(self.run())

    @staticmethod
    def ticks(seconds):
        return BaseToolModel.ticks(seconds)

    def run(self):
        processes = [self.env.process(self.run_session(session)) for session in self.workload.sessions]
        yield self.env.all_of(processes)
        common.inj_finish = True
        if self.options.stop_when_complete and not self.sim_done.triggered:
            self.sim_done.succeed()

    def run_session(self, session):
        yield self.env.timeout(self.ticks(session.arrival_time_s))
        for step in session.steps:
            yield self.env.timeout(self.ticks(step.user_delay_before_s))
            if request_counter.pending_requests + request_counter.running_requests >= common.max_requests_in_parallel:
                raise ValueError('Agent replay exceeded max_requests_in_parallel.')
            request = AgentRequest(self.env, step, self.model)
            request.fill_request()
            row = dict(session_id=session.session_id, step_id=step.step_id, request_id=request._id,
                       arrived_tick=self.env.now, completed_tick=None, tool_completed_tick=None,
                       tool_delay_ticks=self.ticks(step.tool_delay_s),
                       user_delay_before_ticks=self.ticks(step.user_delay_before_s),
                       tool_calls=step.tool_calls, input_tokens=step.input_tokens,
                       output_tokens=step.output_tokens)
            self.records.append(row)
            self.requests[request._id] = request
            common.RequestQueue.outstanding.append(request)
            request_counter.increment_pending()
            request_counter.increment_total()
            completed = yield request.completion
            row['completed_tick'] = completed
            # Request KV/state has been released before host/external tool work.
            yield from self.tools.execute(session.session_id, step)
            row['tool_completed_tick'] = self.env.now
            row['tool_delay_ticks'] = self.env.now - completed
        self.completed[session.session_id] = self.env.now

    def export_workload(self, output_dir):
        path = Path(output_dir)
        self.workload.write(path / 'effective_agent_workload.json')
        (path / 'effective_tool_config.json').write_text(json.dumps(
            dict(model_check=self.model_check, profile_sha256=self.tools.source_sha256,
                 config=self.tools.config), indent=2)+'\n')
        with (path / 'effective_workload.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['session_id', 'step_id', 'num_prefill_tokens',
                                                        'num_decode_tokens', 'context_length'])
            writer.writeheader()
            for session in self.workload.sessions:
                for step in session.steps:
                    writer.writerow(dict(session_id=session.session_id, step_id=step.step_id,
                                         num_prefill_tokens=step.input_tokens, num_decode_tokens=step.output_tokens-1,
                                         context_length=step.input_tokens))

    def finalize(self, output_dir, tick):
        rows = []
        for record in self.records:
            row = dict(record)
            request = self.requests[row['request_id']]
            row['scheduled_tick'] = request._scheduled_at if request._scheduled else None
            row['first_token_tick'] = request.first_token_at
            row['queue_s'] = ((request._scheduled_at-row['arrived_tick']) / common.time_granularity
                              if request._scheduled else None)
            row['ttft_including_queue_s'] = ((request.first_token_at-row['arrived_tick']) / common.time_granularity
                                             if request.first_token_at is not None else None)
            rows.append(row)
        sessions = [dict(session_id=s.session_id, arrival_tick=self.ticks(s.arrival_time_s),
                         completed_tick=self.completed.get(s.session_id),
                         latency_s=((self.completed[s.session_id]-self.ticks(s.arrival_time_s)) / common.time_granularity
                                    if s.session_id in self.completed else None))
                    for s in self.workload.sessions]
        report = dict(cache_policy='recompute', tick_seconds=1/common.time_granularity,
                      cutoff_tick=tick, planned_sessions=len(sessions), completed_sessions=len(self.completed),
                      planned_calls=sum(len(s.steps) for s in self.workload.sessions),
                      released_calls=len(rows), completed_calls=sum(r['completed_tick'] is not None for r in rows),
                      sessions=sessions, calls=rows,
                      model_check=self.model_check, tool_execution=self.tools.snapshot(),
                      ttft_semantics='First generated token, including reasoning; not the first user-visible final answer.',
                      scope='Fixed trajectory performance replay; completion is not benchmark task success.')
        Path(output_dir, 'agent_metrics.json').write_text(json.dumps(report, indent=2)+'\n')
