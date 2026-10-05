"""Profile-backed external tools and a bounded host CPU resource.

These resources sit outside the chiplet mesh. Service profiles must describe
the intended tool implementation and target machine; no hardware scaling or
benchmark Python execution is inferred from a function name.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
from typing import Optional

import simpy

import Sim.common as common
from Sim.config.utils import get_all_subclasses
from Sim.entities.agent_workload import AgentStep


@dataclass(frozen=True)
class ToolProfile:
    kind: str
    service_s: float
    provenance: str
    round_trip_s: float = 0.0
    bandwidth_gbps: Optional[float] = None
    workspace_bytes: int = 0

    def __post_init__(self):
        for name in ('service_s', 'round_trip_s'):
            AgentStep.validate_seconds(getattr(self, name), name)
        if self.bandwidth_gbps is not None:
            AgentStep.validate_seconds(self.bandwidth_gbps, 'bandwidth_gbps')
            if self.bandwidth_gbps == 0:
                raise ValueError('Tool transport bandwidth must be positive decimal GB/s.')
        if type(self.workspace_bytes) is not int or self.workspace_bytes < 0:
            raise ValueError('Tool workspace_bytes must be a nonnegative integer.')
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError('Every tool profile needs measurement or assumption provenance.')
        if self.kind == 'external' and self.workspace_bytes:
            raise ValueError('External-service workspace does not occupy the local CPU pool.')


class BaseToolModel(ABC):
    @classmethod
    def create_from_name(cls, name):
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass
        raise ValueError(f'[{cls.__name__}] Invalid name: {name}')

    @staticmethod
    def ticks(seconds):
        return int((Decimal(str(seconds))*common.time_granularity).to_integral_value(rounding=ROUND_CEILING))

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def service(self, system, profile, record):
        raise NotImplementedError


class ExternalToolModel(BaseToolModel):
    @staticmethod
    def get_name():
        return 'external'

    def service(self, system, profile, record):
        record['service_started_tick'] = system.env.now
        yield system.env.timeout(self.ticks(profile.service_s))
        record['service_completed_tick'] = system.env.now


class CPUProfileToolModel(BaseToolModel):
    @staticmethod
    def get_name():
        return 'cpu_profile'

    def service(self, system, profile, record):
        with system.cpu.request() as worker:
            yield worker
            allocation = system.memory.get(profile.workspace_bytes) if profile.workspace_bytes else None
            try:
                if allocation is not None:
                    yield allocation
                record['service_started_tick'] = system.env.now
                yield system.env.timeout(self.ticks(profile.service_s))
                record['service_completed_tick'] = system.env.now
            finally:
                if allocation is not None:
                    if allocation.triggered:
                        system.memory.put(profile.workspace_bytes)
                    else:
                        allocation.cancel()


class ToolExecutionSystem:
    def __init__(self, env, profile_file=''):
        self.env = env
        self.records = []
        self.profiles = {}
        self.default = None
        self.config = None
        self.cpu = self.memory = None
        self.source_sha256 = None
        if not profile_file:
            return
        source = Path(profile_file).read_bytes()
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        self.config = json.loads(source)
        if type(self.config.get('schema_version')) is not int or self.config['schema_version'] != 1:
            raise ValueError('Unsupported tool-profile schema version.')
        self.profiles = {name: ToolProfile(**row) for name, row in self.config.get('tools', {}).items()}
        if 'default' in self.config:
            self.default = ToolProfile(**self.config['default'])
        profiles = list(self.profiles.values()) + ([self.default] if self.default else [])
        if not profiles:
            raise ValueError('Tool profile file contains no profiles.')
        for profile in profiles:
            BaseToolModel.create_from_name(profile.kind)
        if any(profile.kind == 'cpu_profile' for profile in profiles):
            cpu = self.config.get('cpu', {})
            if (type(cpu.get('workers')) is not int or cpu['workers'] < 1
                    or type(cpu.get('memory_bytes')) is not int or cpu['memory_bytes'] < 1
                    or not cpu.get('target_machine')):
                raise ValueError('CPU profiles require workers, memory_bytes and target_machine.')
            if any(p.workspace_bytes > cpu['memory_bytes'] for p in profiles):
                raise ValueError('A tool workspace exceeds the host CPU memory capacity.')
            self.cpu = simpy.Resource(env, capacity=cpu['workers'])
            self.memory = simpy.Container(env, capacity=cpu['memory_bytes'], init=cpu['memory_bytes'])

    def profile_for(self, name):
        profile = self.profiles.get(name, self.default)
        if profile is None:
            raise ValueError(f'Missing tool profile for {name}.')
        return profile

    def validate(self, workload):
        if self.config is None:
            return
        for session in workload.sessions:
            for step in session.steps:
                if step.tool_delay_s:
                    raise ValueError('Tool profiles require zero stored tool_delay_s to avoid double counting.')
                if step.tool_calls != len(step.tools):
                    raise ValueError('Profiled tool execution needs per-tool descriptors.')
                for tool in step.tools:
                    self.profile_for(tool.name)

    def execute(self, session_id, step):
        if self.config is None:
            yield self.env.timeout(BaseToolModel.ticks(step.tool_delay_s))
            return
        # BFCL tools within each step are sequential; separate sessions overlap.
        for index, tool in enumerate(step.tools):
            profile = self.profile_for(tool.name)
            row = dict(session_id=session_id, step_id=step.step_id, tool_index=index,
                       name=tool.name, kind=profile.kind, submitted_tick=self.env.now,
                       request_bytes=tool.request_bytes, response_bytes=tool.response_bytes,
                       service_started_tick=None, service_completed_tick=None, completed_tick=None)
            self.records.append(row)
            request_s = response_s = Decimal(0)
            if profile.bandwidth_gbps is not None:
                bandwidth = Decimal(str(profile.bandwidth_gbps)) * 1_000_000_000
                request_s = Decimal(tool.request_bytes) / bandwidth
                response_s = Decimal(tool.response_bytes) / bandwidth
            row['request_transfer_ticks'] = BaseToolModel.ticks(request_s)
            row['response_transfer_ticks'] = BaseToolModel.ticks(response_s)
            row['round_trip_ticks'] = BaseToolModel.ticks(profile.round_trip_s)
            yield self.env.timeout(row['request_transfer_ticks'])
            queued = self.env.now
            model = BaseToolModel.create_from_name(profile.kind)()
            yield from model.service(self, profile, row)
            row['queue_ticks'] = row['service_started_tick'] - queued
            yield self.env.timeout(row['round_trip_ticks'] + row['response_transfer_ticks'])
            row['completed_tick'] = self.env.now

    def snapshot(self):
        return dict(profile_sha256=self.source_sha256, config=self.config,
                    scope='Host/external service profiles; no chiplet area, energy, NoI traffic or shared NIC contention.',
                    cpu_active=self.cpu.count if self.cpu else 0,
                    cpu_queued=len(self.cpu.queue) if self.cpu else 0,
                    cpu_workspace_used_bytes=self.memory.capacity-self.memory.level if self.memory else 0,
                    calls=self.records)
