#!/usr/bin/env python3
"""Controlled host-DRAM capacity/transport sensitivity with CPU tool work."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import tyro

from Sim.config.agent_scheduler_config import AgentSchedulerConfig
from Sim.config.model_config import BaseModelConfig
from Sim.entities.agent_workload import AgentToolCall, AgentWorkload
from tests.validation.agent_workloads.validate import AgentReplayValidation
from tests.validation.agent_workloads.study_scheduling import AgentSchedulingStudy, SchedulingStudyConfig


@dataclass
class HostOffloadStudyConfig:
    output_dir: Path
    model: str = 'gqa-prefix-fixture'
    backends: tuple[str, ...] = ('hydra_sim',)
    prepare_only: bool = False


class HostOffloadStudy:
    tool_quota = 4 * 1024**3

    def __init__(self, config):
        self.config = config
        self.output = config.output_dir.resolve()

    def prepare(self):
        if self.output.exists():
            raise ValueError('Use a fresh offload study output directory.')
        self.output.mkdir(parents=True)
        source = AgentSchedulingStudy(SchedulingStudyConfig(self.output)).workload()
        sessions = [replace(session, steps=tuple(replace(step, tool_delay_s=0,
            tools=(AgentToolCall('local_tool', 64, 128),) if step.tool_calls else ())
            for step in session.steps)) for session in source.sessions]
        trace = self.output/'workload.json'
        AgentWorkload(sessions, dict(is_synthetic=True, description=
            'Fixed synthetic identities and assumed CPU/transport costs; not BFCL performance.')).write(trace)
        profiles = self.output/'host_tools.json'
        profiles.write_text(json.dumps(dict(schema_version=1,
            cpu=dict(workers=2, memory_bytes=self.tool_quota, target_machine='Uncalibrated host profile'),
            default=dict(kind='cpu_profile', service_s=.0002, workspace_bytes=64*1024**2,
                         provenance='Controlled sensitivity assumption; no measured CPU speed.')), indent=2)+'\n')
        return trace, profiles

    def run(self):
        trace, profiles = self.prepare()
        if self.config.prepare_only:
            print(f'Generated workload and tool profiles in {self.output}')
            return
        model = BaseModelConfig.create_from_name(self.config.model)
        entry = 96*sum(model.hybrid_blocks[i].states for i in model.block_type_sequence)
        rows = []
        points = [('hbm_only', 'lru', 256*1024**3, 25.0),
                  ('offload_fast', 'lru_offload', 256*1024**3, 25.0),
                  ('offload_slow', 'lru_offload', 256*1024**3, .01),
                  ('host_limited', 'lru_offload', self.tool_quota+entry, 25.0)]
        for name, policy, host_capacity, bandwidth in points:
            scheduler = AgentSchedulerConfig(prefix_cache=policy, prefix_capacity_bytes=entry,
                host_dram_capacity_bytes=host_capacity, host_link_bandwidth_gbps=bandwidth,
                batching='timeout', max_batch_wait_s=.0002, max_active_batches=1)
            args = SimpleNamespace(**asdict(scheduler), output_dir=self.output/name,
                trace=trace, models=[self.config.model], backends=list(self.config.backends), requests=12,
                time_limit=60, run_timeout=1800, packages=1, input_format='normalized',
                tool_profiles_file=profiles, stop_when_complete=True, scheduler='agent', batch_size=4,
                memory_chiplets=None, dataset_label='synthetic-host-offload', arrival_interval=0,
                require_complete=True, virtual_channels=8, transfer_timeout=120, dma_pacing=False,
                dma_burst_bytes=4096, packet_quantum_bytes=1024, packet_buffer_bytes=16384)
            results = AgentReplayValidation(args).run()
            for result in results:
                path = next((args.output_dir/self.config.model/result['backend']).glob('*/agent_metrics.json'))
                report = json.loads(path.read_text())
                cache = report['prefix_cache']
                rows.append(dict(point=name, backend=result['backend'], tokens_per_s=result['tokens_per_sec'],
                    mean_session_s=statistics.mean(s['latency_s'] for s in report['sessions']),
                    process_wall_s=result['process_wall_seconds'], completed_calls=report['completed_calls'],
                    hits=cache['hits'], restores=cache.get('restores', 0),
                    host_evictions=cache.get('host', {}).get('evictions', 0),
                    host_config=asdict(scheduler), report=str(path.relative_to(self.output))))
            (self.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    HostOffloadStudy(tyro.cli(HostOffloadStudyConfig)).run()
