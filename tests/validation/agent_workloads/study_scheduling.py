#!/usr/bin/env python3
"""Small reproducible runtime DSE on fixed hardware, using synthetic sessions."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import tyro

from Sim.config.agent_scheduler_config import AgentSchedulerConfig
from Sim.entities.agent_workload import AgentSession, AgentStep, AgentWorkload
from tests.validation.agent_workloads.validate import AgentReplayValidation


@dataclass
class SchedulingStudyConfig:
    output_dir: Path
    model: str = 'gqa-prefix-fixture'
    sessions: int = 12
    time_limit: float = 60.0


@dataclass
class RuntimeDesignPoint:
    name: str
    priority: str = 'fcfs'
    batch_size: int = 1
    batching: str = 'immediate'
    capacity_bytes: int = 0


class AgentSchedulingStudy:
    def __init__(self, config):
        self.config = config
        self.output = config.output_dir.resolve()

    def workload(self):
        sessions = []
        for i in range(self.config.sessions):
            # Fixed synthetic identities; no prefix inference from token counts.
            prefix = f'synthetic:template-tokenizer-v1:tenant-{i % 3}'
            length = (128, 256, 512)[i % 3]
            outputs = (2, 6, 12)[i % 3]
            steps = tuple(AgentStep(str(j), length, outputs,
                tool_calls=int(j < 2), tool_delay_s=.0002 if j < 2 else 0,
                prefix_id=prefix, prefix_tokens=96) for j in range(3))
            sessions.append(AgentSession(f'session-{i:02}', i*.00002, steps,
                                         slo_s=(.006, .012, .018)[(i+1) % 3]))
        return AgentWorkload(sessions, dict(is_synthetic=True,
            scope='Controlled queueing/batching/cache sensitivity; not a BFCL benchmark score.',
            prefix_identity='Explicit equal fixture token prefixes within each tenant; no real prompt claim.'))

    def points(self):
        points = [RuntimeDesignPoint('priority_'+name, priority=name)
                  for name in ('fcfs', 'sjf', 'hrrn', 'edf', 'least_slack')]
        points += [RuntimeDesignPoint(f'batch_{size}_{policy}', batch_size=size, batching=policy)
                   for size in (2, 4) for policy in ('immediate', 'timeout')]
        # Fixture entries are 49,152 B; Qwen entries are 7,077,888 B.
        from Sim.config.model_config import BaseModelConfig
        model = BaseModelConfig.create_from_name(self.config.model)
        entry_bytes = 96*sum(model.hybrid_blocks[i].states for i in model.block_type_sequence)
        points += [RuntimeDesignPoint('cache_'+name, batch_size=4, batching='timeout', capacity_bytes=capacity)
                   for name, capacity in (('undersized', entry_bytes-1), ('one', entry_bytes), ('all', 3*entry_bytes))]
        return points

    def run(self):
        if self.output.exists():
            raise ValueError('Use a fresh study output directory.')
        if self.config.sessions < 4:
            raise ValueError('The study needs at least four sessions.')
        self.output.mkdir(parents=True)
        trace = self.output/'workload.json'
        self.workload().write(trace)
        rows = []
        for point in self.points():
            scheduler = AgentSchedulerConfig(priority=point.priority, batching=point.batching,
                max_batch_wait_s=.0002, estimated_output_tokens=6, max_active_batches=1,
                prefix_cache='lru' if point.capacity_bytes else 'none', prefix_capacity_bytes=point.capacity_bytes)
            args = SimpleNamespace(**asdict(scheduler), output_dir=self.output/point.name,
                trace=trace, models=[self.config.model], backends=['hydra_sim'], requests=self.config.sessions,
                time_limit=self.config.time_limit, run_timeout=1800, packages=1, input_format='normalized',
                stop_when_complete=True, scheduler='agent', batch_size=point.batch_size,
                memory_chiplets=None, dataset_label='synthetic-agent', arrival_interval=0, require_complete=True,
                virtual_channels=8, transfer_timeout=120, dma_pacing=False, dma_burst_bytes=4096,
                packet_quantum_bytes=1024, packet_buffer_bytes=16384)
            result = AgentReplayValidation(args).run()[0]
            report_path = next((args.output_dir/self.config.model/'hydra_sim').glob('*/agent_metrics.json'))
            report = json.loads(report_path.read_text())
            waits = [row['queue_s'] for row in report['calls']]
            latencies = [row['latency_s'] for row in report['sessions']]
            batches = {row['admission']['batch_id']: row['admission']['batch_size'] for row in report['calls']}
            rows.append(dict(design=asdict(point), tokens_per_s=result['tokens_per_sec'],
                elapsed_s=result['elapsed_simulation_s'], process_wall_s=result['process_wall_seconds'],
                mean_session_s=statistics.mean(latencies), max_session_s=max(latencies),
                mean_queue_s=statistics.mean(waits), mean_batch_size=statistics.mean(batches.values()),
                mean_ttft_including_queue_s=statistics.mean(row['ttft_including_queue_s'] for row in report['calls']),
                slo_attainment=sum(row['slo_met'] for row in report['sessions'])/len(latencies),
                prefix_cache=report['prefix_cache'], report=str(report_path.relative_to(self.output)),
                report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest()))
            (self.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
        self.write_table(rows)

    def write_table(self, rows):
        lines = ['| Design | TP (tok/s) | Mean session (ms) | Mean queue (ms) | Mean batch | Session SLO met | Prefix hits |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            hits = row['prefix_cache']['hits'] if row['prefix_cache'] else 0
            lines.append(f"| {row['design']['name']} | {row['tokens_per_s']:.1f} | {1e3*row['mean_session_s']:.3f} | "
                         f"{1e3*row['mean_queue_s']:.3f} | {row['mean_batch_size']:.2f} | {row['slo_attainment']:.1%} | {hits} |")
        (self.output/'table.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    AgentSchedulingStudy(tyro.cli(SchedulingStudyConfig)).run()
