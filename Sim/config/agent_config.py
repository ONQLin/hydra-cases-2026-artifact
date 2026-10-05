"""Configurable recorded workload input, independent of benchmark runtimes."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Optional


@dataclass
class AgentTraceConfig:
    input_format: str = 'normalized'
    provenance_file: str = ''
    task_ids: List[str] = field(default_factory=list)
    tool_latency_s: Optional[float] = None
    tool_latencies_file: str = ''
    tool_profiles_file: str = ''
    session_interval_s: float = 0.01
    user_think_s: float = 0.0
    allow_model_mismatch: bool = False
    stop_when_complete: bool = False

    def load(self, path, limit):
        from Sim.entities.agent_workload import AgentWorkload
        if self.input_format == 'normalized':
            if (self.provenance_file or self.tool_latency_s is not None or self.tool_latencies_file
                    or self.session_interval_s != .01 or self.user_think_s != 0):
                raise ValueError('Normalized traces already contain provenance, arrivals and delays; edit the trace instead.')
            workload = AgentWorkload.load(path)
        else:
            from integrations.workloads import BaseWorkloadImporter
            if not self.provenance_file:
                raise ValueError('Raw benchmark logs require an explicit provenance_file.')
            if self.tool_profiles_file and (self.tool_latency_s is not None or self.tool_latencies_file):
                raise ValueError('Tool profiles and fixed/measured delays are mutually exclusive.')
            importer = BaseWorkloadImporter.create_from_name(self.input_format)(
                json.loads(Path(self.provenance_file).read_text()),
                tool_latency_s=0.0 if self.tool_profiles_file else self.tool_latency_s,
                tool_latencies=json.loads(Path(self.tool_latencies_file).read_text()) if self.tool_latencies_file else None,
                session_interval_s=self.session_interval_s, user_think_s=self.user_think_s)
            workload = importer.load(path)
        return workload.select(self.task_ids, limit)
