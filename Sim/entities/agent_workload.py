"""Backend-independent, serial agent trajectories with explicit external waits."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AgentToolCall:
    name: str
    request_bytes: int
    response_bytes: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Tool calls need a nonempty name.')
        for value in (self.request_bytes, self.response_bytes):
            if type(value) is not int or value < 0:
                raise ValueError('Tool payload sizes must be nonnegative integer bytes.')


@dataclass(frozen=True)
class AgentStep:
    step_id: str
    input_tokens: int
    output_tokens: int
    tool_calls: int = 0
    tool_delay_s: float = 0.0
    user_delay_before_s: float = 0.0
    tools: tuple = ()
    estimated_service_s: Optional[float] = None
    prefix_id: str = ''
    prefix_tokens: int = 0

    def __post_init__(self):
        if self.estimated_service_s is not None:
            self.validate_seconds(self.estimated_service_s, 'estimated_service_s')
            if not self.estimated_service_s:
                raise ValueError('Service estimate must be positive.')
        if not isinstance(self.step_id, str) or not self.step_id:
            raise ValueError('Each agent step needs a nonempty string ID.')
        for name, minimum in (('input_tokens', 1), ('output_tokens', 1), ('tool_calls', 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        if (not isinstance(self.prefix_id, str) or type(self.prefix_tokens) is not int
                or not 0 <= self.prefix_tokens < self.input_tokens):
            raise ValueError('Prefix tokens must leave at least one uncached prompt token.')
        if bool(self.prefix_id) != bool(self.prefix_tokens):
            raise ValueError('Declare both prefix_id and prefix_tokens, or neither.')
        for name in ('tool_delay_s', 'user_delay_before_s'):
            self.validate_seconds(getattr(self, name), name)
        if not self.tool_calls and self.tool_delay_s:
            raise ValueError('Tool delay requires at least one recorded tool call.')
        if self.tools and (len(self.tools) != self.tool_calls or
                           not all(isinstance(tool, AgentToolCall) for tool in self.tools)):
            raise ValueError('Tool descriptors must match the recorded execution count.')

    @staticmethod
    def validate_seconds(value, name):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite nonnegative seconds.')


@dataclass(frozen=True)
class AgentSession:
    session_id: str
    arrival_time_s: float
    steps: tuple
    slo_s: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError('Each session needs a nonempty string ID.')
        AgentStep.validate_seconds(self.arrival_time_s, 'arrival_time_s')
        if self.slo_s is not None:
            AgentStep.validate_seconds(self.slo_s, 'slo_s')
            if not self.slo_s:
                raise ValueError('Session SLO must be positive.')
        if not self.steps or len({step.step_id for step in self.steps}) != len(self.steps):
            raise ValueError('A session must contain steps with unique IDs.')


class AgentWorkload:
    def __init__(self, sessions, metadata):
        self.sessions = tuple(sessions)
        self.metadata = metadata
        if not self.sessions or len({s.session_id for s in self.sessions}) != len(self.sessions):
            raise ValueError('A workload must contain sessions with unique IDs.')
        if not isinstance(metadata, dict):
            raise ValueError('Workload provenance metadata must be a dictionary.')
        prefixes = {}
        for session in self.sessions:
            for step in session.steps:
                if step.prefix_id:
                    if prefixes.setdefault(step.prefix_id, step.prefix_tokens) != step.prefix_tokens:
                        raise ValueError('A prefix_id must always describe the same tokenized prefix length.')

    @classmethod
    def load(cls, path, limit=None):
        payload = json.loads(Path(path).read_text())
        contract = (payload.get('schema_version'), payload.get('cache_policy'))
        if type(payload.get('schema_version')) is not int or contract not in ((1, 'recompute'), (2, 'runtime')):
            raise ValueError('Use agent schema 2 with runtime cache policy, or legacy schema 1/recompute.')
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('The session limit must be a positive integer.')
        sessions = [AgentSession(row['session_id'], row['arrival_time_s'],
                                 tuple(AgentStep(**dict(step, tools=tuple(AgentToolCall(**tool)
                                       for tool in step.get('tools', [])))) for step in row['steps']), row.get('slo_s'))
                    for row in payload['sessions']]
        # Validate the entire file before selecting a reproducible prefix.
        workload = cls(sessions, payload['metadata'])
        return cls(workload.sessions[:limit], workload.metadata)

    def select(self, task_ids=(), limit=None):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('The session limit must be positive.')
        if len(set(task_ids)) != len(task_ids):
            raise ValueError('Duplicate selected task IDs.')
        by_id = {session.session_id: session for session in self.sessions}
        if any(task_id not in by_id for task_id in task_ids):
            raise ValueError('A selected task ID is absent from the workload.')
        selected = [by_id[key] for key in task_ids] if task_ids else self.sessions
        # Arrival times remain exactly as recorded in the normalized workload.
        return AgentWorkload(selected[:limit], self.metadata)

    def check_model(self, model, allow_mismatch=False):
        collection = self.metadata.get('collection', {})
        declared = collection.get('source_model_id')
        target = getattr(model, 'source_model', model.get_name())
        matches = declared == target if declared else None
        if matches is False and not allow_mismatch:
            raise ValueError(f'Trace source model {declared} differs from {target}; explicitly enable allow_model_mismatch for a counterfactual replay.')
        return dict(source_model_id=declared, simulated_model_id=target,
                    identity_matches=matches, allow_model_mismatch=allow_mismatch,
                    scope='Model identity check only; checkpoint, tokenizer and serving precision require provenance audit.')

    def write(self, path):
        payload = dict(schema_version=2, cache_policy='runtime', metadata=self.metadata,
                       sessions=[asdict(session) for session in self.sessions])
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
