#!/usr/bin/env python3
"""Controlled runtime trade study using complete, hash-verified BFCL sessions."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import tyro
from Sim.config.agent_config import AgentTraceConfig
from Sim.config.agent_scheduler_config import AgentSchedulerConfig
from tests.validation.agent_workloads.validate import AgentReplayValidation


@dataclass
class BFCLStudyConfig:
    output_dir: Path
    archive: Path = Path('output_sanity_checks/agent_workloads/archive_verified/qwen_full_base.jsonl')
    sessions: int = 4
    time_limit: float = 12000.0


@dataclass(frozen=True)
class BFCLRuntimePoint:
    name: str
    scheduler: str = 'agent'
    priority: str = 'fcfs'
    batch_size: int = 1
    batching: str = 'immediate'
    active_batches: int = 1


class BFCLRuntimeStudy:
    def __init__(self, config):
        self.config = config
        self.output = config.output_dir.resolve()
        self.source = ROOT/'tests/validation/agent_workloads/real_bfcl'

    @staticmethod
    def points():
        return [BFCLRuntimePoint('whole_fcfs'),
                BFCLRuntimePoint('whole_sjf', priority='sjf'),
                BFCLRuntimePoint('whole_hrrn', priority='hrrn'),
                BFCLRuntimePoint('whole_batch4', batch_size=4, batching='timeout'),
                BFCLRuntimePoint('continuous1', scheduler='vllm_latest'),
                BFCLRuntimePoint('continuous2', scheduler='vllm_latest', batch_size=2),
                BFCLRuntimePoint('continuous4', scheduler='vllm_latest', batch_size=4),
                BFCLRuntimePoint('whole_parallel4', active_batches=4)]

    def prepare(self):
        if self.output.exists():
            raise ValueError('Use a fresh BFCL study output directory.')
        if not 2 <= self.config.sessions <= 200:
            raise ValueError('Select between 2 and 200 complete archived sessions.')
        provenance = json.loads((self.source/'provenance.json').read_text())
        data = self.config.archive.read_bytes()
        if hashlib.sha256(data).hexdigest() != provenance['source_sha256']:
            raise ValueError('Full BFCL archive does not match the pinned source hash.')
        ids = [f'multi_turn_base_{i}' for i in range(self.config.sessions)]
        lines = {json.loads(line)['id']: line for line in data.splitlines(keepends=True)}
        selected = b''.join(lines[key] for key in ids)
        self.output.mkdir(parents=True)
        raw = self.output/'selected.jsonl'
        raw.write_bytes(selected)
        provenance.update(selected_ids=ids, selected_sha256=hashlib.sha256(selected).hexdigest(),
            selection=f'First {len(ids)} numeric task IDs, complete sessions; not filtered by correctness or output length.')
        metadata = self.output/'provenance.json'
        metadata.write_text(json.dumps(provenance, indent=2)+'\n')
        tool_file = self.output/'tool_profiles.json'
        tool_file.write_bytes((self.source/'assumed_host_cpu.json').read_bytes())
        options = AgentTraceConfig(input_format='bfcl', provenance_file=str(metadata),
            tool_profiles_file=str(tool_file), session_interval_s=0)
        workload = options.load(raw, self.config.sessions)
        workload.metadata['study'] = dict(selection='First numeric IDs, complete trajectories; no token truncation.',
            arrivals='All sessions released at t=0: controlled burst, not recorded arrival timestamps.',
            prefix_cache='Disabled; token identities are unavailable.',
            tool_costs='Assumed 1 ms CPU service; not measured BFCL latency.')
        trace = self.output/'workload.json'
        workload.write(trace)
        inventory = dict(sessions=len(workload.sessions), calls=sum(len(s.steps) for s in workload.sessions),
            output_tokens=sum(t.output_tokens for s in workload.sessions for t in s.steps),
            tools=sum(t.tool_calls for s in workload.sessions for t in s.steps),
            unique_prompt_lengths=len({t.input_tokens for s in workload.sessions for t in s.steps}),
            model='qwen3-8b', source_model='Qwen/Qwen3-8B', session_ids=ids)
        (self.output/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
        return trace, tool_file

    def run(self):
        trace, tools = self.prepare()
        rows = []
        for point in self.points():
            rows.append(self.run_point(point, trace, tools))
            (self.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
        self.verify_comparison(rows)
        self.plot(rows)
        self.write_table(rows)

    def verify_comparison(self, rows):
        inventory = json.loads((self.output/'inventory.json').read_text())
        reference = {}
        for row in rows:
            if (row['completed_calls'] != inventory['calls']
                    or row['output_tokens'] != inventory['output_tokens']):
                raise RuntimeError('Runtime points did not replay identical complete work.')
            directory = next((self.output/row['design']['name']/'qwen3-8b/hydra_sim').glob('*/agent_metrics.json')).parent
            for name in ('system_snapshot.json', 'effective_workload.csv',
                         'effective_agent_workload.json', 'effective_tool_config.json'):
                digest = hashlib.sha256((directory/name).read_bytes()).hexdigest()
                if reference.setdefault(name, digest) != digest:
                    raise RuntimeError(f'Runtime points disagree on {name}.')
            policy = json.loads((directory/'effective_agent_scheduler.json').read_text())
            options, design = policy['options'], row['design']
            if (options['prefix_cache'] != 'none' or options['max_active_batches'] != design['active_batches']
                    or policy['scheduler'] != design['scheduler'] or policy['batch_size'] != design['batch_size']
                    or any(options[key] != design[key] for key in ('priority', 'batching'))):
                raise RuntimeError('Runtime control differs from the intended design.')
        baseline = next(r for r in rows if r['design']['name'] == 'whole_fcfs')
        single = next(r for r in rows if r['design']['name'] == 'continuous1')
        shift = single['elapsed_s']/baseline['elapsed_s'] - 1
        if abs(shift) > 1e-5:
            raise RuntimeError('Single-resident control changed the completion window beyond dispatch rounding.')
        report = dict(passed=True, points=len(rows), equal_input_sha256=reference,
                      continuous_one_relative_elapsed_shift=shift)
        (self.output/'paired_audit.json').write_text(json.dumps(report, indent=2)+'\n')

    def run_point(self, point, trace, tools):
        policy = AgentSchedulerConfig(priority=point.priority, batching=point.batching,
            max_batch_wait_s=.01, max_active_batches=point.active_batches, default_session_slo_s=1800)
        args = SimpleNamespace(**asdict(policy), output_dir=self.output/point.name,
            trace=trace, models=['qwen3-8b'], backends=['hydra_sim'], requests=self.config.sessions,
            time_limit=self.config.time_limit, run_timeout=3600, packages=1, input_format='normalized',
            tool_profiles_file=tools, stop_when_complete=True, scheduler=point.scheduler,
            batch_size=point.batch_size, memory_chiplets=None, dataset_label='bfcl-runtime-trade',
            arrival_interval=0, require_complete=True, virtual_channels=8, transfer_timeout=120,
            dma_pacing=False, dma_burst_bytes=4096, packet_quantum_bytes=1024, packet_buffer_bytes=16384)
        result = AgentReplayValidation(args).run()[0]
        directory = next((args.output_dir/'qwen3-8b/hydra_sim').glob('*/agent_metrics.json')).parent
        report = json.loads((directory/'agent_metrics.json').read_text())
        continuous = directory/'continuous_batching.json'
        iterations = json.loads(continuous.read_text())['iterations'] if continuous.exists() else []
        return self.summarize(point, result, report, iterations)

    @staticmethod
    def summarize(point, result, report, iterations):
        calls = report['calls']
        batches = {c['admission']['batch_id']: c['admission']['batch_size'] for c in calls}
        decode = [r for r in iterations if r['stage'] == 'decode']
        duration = sum(r['completed_tick']-r['started_tick'] for r in decode)
        padded = sum(max(r['contexts'])*len(r['contexts']) for r in decode)
        actual = sum(sum(r['contexts']) for r in decode)
        return dict(design=asdict(point), output_tokens=result['output_tokens'],
            completed_calls=report['completed_calls'], elapsed_s=result['elapsed_simulation_s'],
            tokens_per_s=result['tokens_per_sec'], process_wall_s=result['process_wall_seconds'],
            mean_ttft_s=statistics.mean(c['ttft_including_queue_s'] for c in calls),
            p95_ttft_s=float(np.percentile([c['ttft_including_queue_s'] for c in calls],95)),
            mean_queue_s=statistics.mean(c['queue_s'] for c in calls),
            mean_session_s=statistics.mean(s['latency_s'] for s in report['sessions']),
            session_s={s['session_id']: s['latency_s'] for s in report['sessions']},
            mean_admission_batch=statistics.mean(batches.values()),
            decode_time_weighted_batch=(sum((r['completed_tick']-r['started_tick'])*len(r['request_ids'])
                for r in decode)/duration if duration else None),
            decode_context_padding_ratio=padded/actual if actual else None,
            cpu_queued_calls=sum(r['queue_ticks']>0 for r in report['tool_execution']['calls']),
            checks='Complete call/token/causal/tool/memory audit passed; continuous iteration audit where applicable.')

    def write_table(self, rows):
        lines = ['| Runtime | TP (tok/s) | Mean TTFT incl. queue (s) | P95 TTFT (s) | Mean session (s) | Actual admission batch | Decode batch, time-weighted | Wall (s) |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            decode = row['decode_time_weighted_batch']
            decode_text = f'{decode:.2f}' if decode is not None else '—'
            lines.append(f"| {row['design']['name']} | {row['tokens_per_s']:.3f} | {row['mean_ttft_s']:.3f} | "
                f"{row['p95_ttft_s']:.3f} | {row['mean_session_s']:.3f} | {row['mean_admission_batch']:.2f} | "
                f"{decode_text} | {row['process_wall_s']:.2f} |")
        (self.output/'table.md').write_text('\n'.join(lines)+'\n')

    def plot(self, rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
        offsets = {'whole_fcfs': (-5, -34), 'whole_sjf': (-25, 10),
                   'whole_hrrn': (4, 10), 'continuous2': (-30, 10),
                   'continuous4': (5, 7), 'whole_parallel4': (5, -15)}
        baseline = next(r for r in rows if r['design']['name'] == 'whole_fcfs')
        overlap = {r['design']['name'] for r in rows
                   if r['design']['name'] in ('whole_batch4', 'continuous1')
                   and all(np.isclose(r[key], baseline[key], rtol=.001)
                           for key in ('mean_ttft_s', 'tokens_per_s'))}
        curve = [r for r in rows if r['design']['scheduler'] == 'vllm_latest']
        axes[0].plot([r['mean_ttft_s'] for r in curve], [r['tokens_per_s'] for r in curve],
                     color='0.65', linewidth=1, zorder=0)
        for row in rows:
            continuous = row['design']['scheduler']=='vllm_latest'
            label = row['design']['name']
            axes[0].scatter(row['mean_ttft_s'],row['tokens_per_s'],marker='o' if continuous else 's',s=55)
            # Collapse only numerically overlapping controls, not distinct data.
            if label not in overlap:
                cluster = ['FCFS'] + [short for name, short in
                    (('whole_batch4', 'batch4'), ('continuous1', 'C1')) if name in overlap]
                text = ' / '.join(cluster) if label == 'whole_fcfs' else label
                axes[0].annotate(text,(row['mean_ttft_s'],row['tokens_per_s']),
                    xytext=offsets.get(label, (4, 10)),textcoords='offset points',fontsize=8,
                    ha='right' if label == 'whole_fcfs' else 'left',
                    arrowprops=dict(arrowstyle='-',color='0.65',lw=.6))
        axes[0].set(xlabel='Mean call TTFT including queue (s)',ylabel='Completion-window throughput (tokens/s)',
                    title='Runtime trade-off (same session burst)')
        for row in rows:
            axes[1].plot(list(row['session_s']),list(row['session_s'].values()),'o-',label=row['design']['name'])
        axes[1].set(xlabel='BFCL session ID',ylabel='Session completion latency (s)',title='Per-session effects')
        axes[1].set_xticks(range(len(rows[0]['session_s'])), [s.rsplit('_',1)[-1] for s in rows[0]['session_s']])
        fig.legend(*axes[1].get_legend_handles_labels(), loc='outside lower center', fontsize=8, ncol=4)
        axes[0].margins(x=.12,y=.25)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.savefig(self.output/'tradeoff.png',dpi=180)
        fig.savefig(self.output/'tradeoff.pdf')
        plt.close(fig)


if __name__ == '__main__':
    BFCLRuntimeStudy(tyro.cli(BFCLStudyConfig)).run()
