"""Serving checkpoints without adding or reordering simulation events."""

import json
from pathlib import Path
import time

import numpy as np
import Sim.common as common
from Sim.metrics.monitor import tokens_monitor, request_counter


class ServingMetricsRecorder:
    def __init__(self, output_dir, interval_s=1.0):
        self.output_dir = Path(output_dir)
        self.interval_ticks = max(1, round(interval_s * common.time_granularity))
        self.next_tick = self.interval_ticks
        self.started_at = time.monotonic()
        self.previous_tick = 0
        self.previous_tokens = 0

    def sample(self, tick, force=False):
        if tick < self.next_tick and not force:
            return
        elapsed = tick / common.time_granularity
        window = (tick - self.previous_tick) / common.time_granularity
        record = {
            'elapsed_simulation_s': elapsed,
            'wall_seconds': time.monotonic() - self.started_at,
            'output_tokens': tokens_monitor.output_tokens,
            'tokens_per_sec': tokens_monitor.output_tokens / elapsed if elapsed else 0,
            'interval_start_s': self.previous_tick / common.time_granularity,
            'interval_tokens_per_sec': ((tokens_monitor.output_tokens - self.previous_tokens) / window
                                       if window else None),
            'ttft_s': (float(np.mean(tokens_monitor.time_to_first_token)) / common.time_granularity
                       if tokens_monitor.time_to_first_token else None),
            'ttft_samples': len(tokens_monitor.time_to_first_token),
            'generated_requests': request_counter.total_requests,
            'completed_requests': request_counter.completed_requests,
            'running_requests': request_counter.running_requests,
            'pending_requests': request_counter.pending_requests,
            'preempted_requests': len(common.preempted_requests),
        }
        checkpoints = self.output_dir / 'checkpoints'
        checkpoints.mkdir(exist_ok=True)
        name = f'{elapsed:g}s.json'
        self._write_record(checkpoints / name, record)
        self._write_record(self.output_dir / 'progress.json', record)
        self.previous_tick = tick
        self.previous_tokens = tokens_monitor.output_tokens
        self.next_tick = tick + self.interval_ticks

    @staticmethod
    def _write_record(path, record):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(path)
