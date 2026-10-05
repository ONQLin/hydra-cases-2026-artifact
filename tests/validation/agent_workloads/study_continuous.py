#!/usr/bin/env python3
"""Iteration lifecycle checks on a fixed design point; synthetic mechanism data."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import tyro
from Sim.config.agent_scheduler_config import AgentSchedulerConfig
from Sim.entities.agent_workload import AgentSession, AgentStep, AgentWorkload
from tests.validation.agent_workloads.validate import AgentReplayValidation
from tests.validation.agent_workloads.study_offload import HostOffloadStudy, HostOffloadStudyConfig


@dataclass
class ContinuousStudyConfig:
    output_dir: Path


class ContinuousBatchingStudy:
    def __init__(self, config):
        self.output = config.output_dir.resolve()

    def run_point(self, name, trace, scheduler, batch_size, **options):
        config = AgentSchedulerConfig(**options)
        args = SimpleNamespace(**asdict(config), output_dir=self.output/name,
            trace=trace, models=['gqa-prefix-fixture'], backends=['hydra_sim'], requests=12,
            time_limit=1, run_timeout=120, packages=1, input_format='normalized',
            stop_when_complete=True, scheduler=scheduler, batch_size=batch_size,
            memory_chiplets=None, dataset_label='synthetic-continuous', arrival_interval=0,
            require_complete=True, virtual_channels=8, transfer_timeout=120, dma_pacing=False,
            dma_burst_bytes=4096, packet_quantum_bytes=1024, packet_buffer_bytes=16384)
        result = AgentReplayValidation(args).run()[0]
        path = next((args.output_dir/'gqa-prefix-fixture'/'hydra_sim').glob('*/agent_metrics.json'))
        report = json.loads(path.read_text())
        row = dict(point=name, output_tokens=result['output_tokens'], tokens_per_s=result['tokens_per_sec'],
            completed_calls=report['completed_calls'], report=str(path.relative_to(self.output)),
            mean_session_s=sum(r['latency_s'] for r in report['sessions'])/len(report['sessions']))
        self.rows.append(row)
        return path.parent, report

    def run(self):
        if self.output.exists():
            raise ValueError('Use a fresh continuous study output directory.')
        self.output.mkdir(parents=True)
        self.rows = []
        single = self.output/'single.json'
        AgentWorkload([AgentSession('one', 0, (AgentStep('0', 128, 8),))],
                      dict(is_synthetic=True, description='Batch-size-one service equivalence.')).write(single)
        _, old = self.run_point('single_agent', single, 'agent', 1)
        _, new = self.run_point('single_continuous', single, 'vllm_latest', 1)
        service = lambda report: report['calls'][0]['completed_tick'] - report['calls'][0]['scheduled_tick']
        if service(old) != service(new):
            raise RuntimeError('Batch-size-one model service changed at iteration boundaries.')
        self.equivalent_service_ticks = service(new)
        ragged = self.output/'ragged.json'
        sessions = [AgentSession('long', 0, (AgentStep('0', 128, 16),)),
                    AgentSession('short', 0, (AgentStep('0', 128, 2),)),
                    AgentSession('late', .0003, (AgentStep('0', 128, 4),)),
                    AgentSession('other-shape', .00035, (AgentStep('0', 256, 3),))]
        AgentWorkload(sessions, dict(is_synthetic=True,
            description='Unequal output lengths, late arrivals and heterogeneous contexts.')).write(ragged)
        self.run_point('ragged_agent', ragged, 'agent', 2, max_active_batches=1)
        directory, report = self.run_point('ragged_continuous', ragged, 'vllm_latest', 2)
        iterations = json.loads((directory/'continuous_batching.json').read_text())['iterations']
        calls = {r['session_id']: r for r in report['calls']}
        if calls['late']['scheduled_tick'] >= calls['long']['completed_tick']:
            raise RuntimeError('A late request did not refill the vacant resident slot.')
        ids = {calls[name]['request_id'] for name in ('late', 'long')}
        if not any(row['stage'] == 'decode' and ids.issubset(row['request_ids'])
                   and len(set(row['contexts'])) > 1 for row in iterations):
            raise RuntimeError('Late request did not join a real heterogeneous decode iteration.')
        source, _ = HostOffloadStudy(HostOffloadStudyConfig(self.output/'inputs')).prepare()
        for name, policy, host, bw in [('hbm', 'lru', 256*1024**3, 25),
                                       ('offload_fast', 'lru_offload', 256*1024**3, 25),
                                       ('offload_slow', 'lru_offload', 256*1024**3, .01),
                                       ('offload_limited', 'lru_offload', 49152, 25)]:
            self.run_point(name, source, 'vllm_latest', 4, prefix_cache=policy,
                prefix_capacity_bytes=49152, host_dram_capacity_bytes=host,
                host_link_bandwidth_gbps=bw, batching='timeout', max_batch_wait_s=.0002)
        result = dict(points=self.rows, batch_one_service_ticks=self.equivalent_service_ticks,
                      late_refill_and_padded_decode=True,
                      scope='Synthetic lifecycle and offload checks, not BFCL performance or serving-engine calibration.')
        (self.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    ContinuousBatchingStudy(tyro.cli(ContinuousStudyConfig)).run()
