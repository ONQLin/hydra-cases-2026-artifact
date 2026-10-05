"""Shared conservative network clock, phase execution, and completion accounting."""

from dataclasses import asdict
import json
import math
import time

import numpy as np
import simpy
from simpy.events import Event

import Sim.common as common
from Sim.entities.execution import ExecutionPhase


class NetworkCompletion(Event):
    def complete_at(self, when, value):
        if self.triggered or when < self.env.now:
            raise RuntimeError('Invalid network completion time or duplicate completion.')
        self._ok = True
        self._value = value
        self.env.schedule(self, delay=when - self.env.now)


class CoSimulationEnvironment(simpy.Environment):
    def __init__(self, execution):
        super().__init__()
        self.execution = execution

    def step(self):
        # Never run network beyond the next application event: that event may
        # submit a flow which changes completion times of packets in flight.
        while self.peek() != float('inf'):
            target = int(self.peek() * self.execution.ticks_per_hydra_tick)
            if self.execution.network_tick >= target:
                break
            self.execution.advance(target)
        return super().step()


class NetworkExecutionMixin:
    """Shared conservative clock and phase sequencing for network providers.

    Providers supply _request(message), memory_bandwidth, output_dir, and trace.
    The request protocol returns integer-picosecond ticks and flattened
    (flow_id, completion_tick) pairs; advancement may stop at an early completion.
    """
    ticks_per_hydra_tick = 1_000_000_000_000 // common.time_granularity

    def __init__(self, config):
        super().__init__(config)
        self.network_tick = 0
        self.flow_id = 0
        self.pending = {}
        self.completed_flows = 0
        self.peak_active_flows = 0
        self.started_at = time.monotonic()
        self.last_progress_at = self.started_at

    def create_environment(self):
        self.env = CoSimulationEnvironment(self)
        return self.env

    def _submit(self, source, destination, byte_count):
        if self.is_package_transfer(source, destination):
            raise ValueError('Remote traffic must use the package fabric, not the local network.')
        if byte_count <= 0 or source == destination:
            return self.env.timeout(0)
        self.flow_id += 1
        event = NetworkCompletion(self.env)
        self.pending[self.flow_id] = event
        result = self._request({'kind': 'submit', 'flow_id': self.flow_id,
                                'source': int(source), 'destination': int(destination),
                                'bytes': float(byte_count), 'hydra_tick': self.env.now})
        if result['tick'] != int(self.env.now * self.ticks_per_hydra_tick):
            raise RuntimeError('HYDRA and network disagree on submission time.')
        self.peak_active_flows = max(self.peak_active_flows, len(self.pending))
        return event

    def advance(self, target):
        result = self._request({'kind': 'advance', 'target_tick': target})
        tick = result['tick']
        if not self.network_tick <= tick <= target:
            raise RuntimeError('network advanced outside the conservative time window.')
        if tick == self.network_tick and not result['completed']:
            raise RuntimeError('network made no progress.')
        self.network_tick = tick
        values = result['completed']
        if len(values) % 2:
            raise RuntimeError('Malformed network completion list.')
        for flow, completed_at in zip(values[::2], values[1::2]):
            if flow not in self.pending or completed_at > tick:
                raise RuntimeError('Unexpected network completion.')
            when = (completed_at + self.ticks_per_hydra_tick - 1) // self.ticks_per_hydra_tick
            self.pending.pop(flow).complete_at(when, completed_at)
            self.completed_flows += 1
        if time.monotonic() - self.last_progress_at >= 10:
            self.last_progress_at = time.monotonic()
            path = self.output_dir / 'network_progress.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps({
                'elapsed_simulation_s': self.network_tick / 1e12,
                'wall_seconds': self.last_progress_at - self.started_at,
                'submitted_flows': self.flow_id,
                'completed_flows': self.completed_flows,
                'inflight_flows': len(self.pending),
                'peak_active_flows': self.peak_active_flows,
            }, indent=2))
            temporary.replace(path)

    def start_block(self, env, block, stage, batch_size, chiplet, memory_id, bandwidth):
        profiles = self.profile_block(block, stage, batch_size, chiplet, bandwidth)
        if any(not item.execution_phases for item in profiles):
            raise ValueError('Missing execution phases for contended execution.')
        phases = [asdict(phase) for item in profiles for phase in item.execution_phases]
        process = env.process(self._run_phases(
            int(memory_id), int(chiplet.chiplet_id), phases,
            min(bandwidth, self.memory_bandwidth[str(memory_id)]) * 1e9,
            {'kind': 'block', 'block_id': block.block_num, 'stage': stage,
             'batch_size': batch_size, 'context_length': block.context_length}))
        return process, float(np.mean([item.utilization for item in profiles]))

    def start_transfer(self, env, source, destination, size_mb, bandwidth, native_ticks):
        return env.process(self._run_phases(
            int(source), int(destination),
            [{'compute_ns': 0, 'memory_bytes': math.ceil(size_mb * 1024**2)}],
            bandwidth * 1024**3, {'kind': 'transfer'}))

    def _run_phases(self, source, destination, phases, bandwidth, metadata):
        start = self.env.now
        records = []
        expanded = (asdict(part) for phase in phases
                    for part in ExecutionPhase(**phase).expanded())
        for phase in expanded:
            phase_start = self.env.now
            memory_bytes = phase['memory_bytes']
            hbm_us = memory_bytes / bandwidth * 1e6
            if memory_bytes:
                hbm_us += self.config.chipsim_config.hbm_access_latency_ns / 1000
            compute_us = phase['compute_ns'] / 1000
            endpoints = ((destination, source) if phase['memory_direction'] == 'write'
                         else (source, destination))
            network = self._submit(*endpoints, memory_bytes)
            local_ticks = math.ceil(max(compute_us, hbm_us) * common.time_granularity / 1e6)
            yield self.env.all_of([network, self.env.timeout(local_ticks)])
            noi_us = (network.value / 1e6 - phase_start * 1e6 / common.time_granularity
                      if network.value is not None else 0.0)
            records.append({'name': phase['name'], 'memory_direction': phase['memory_direction'],
                            'compute_us': compute_us, 'hbm_us': hbm_us,
                            'noi_us': noi_us, 'latency_us': (self.env.now - phase_start) * 1e6 / common.time_granularity})
        self.trace.write(json.dumps({'request': dict(metadata, source=source, destination=destination,
                                                     start_tick=start),
                                     'result': {'phases': records, 'completed_tick': self.env.now}}) + '\n')

    def close(self):
        try:
            if hasattr(self, 'output_dir'):
                (self.output_dir / 'network_summary.json').write_text(json.dumps({
                    'submitted_flows': self.flow_id, 'completed_flows': self.completed_flows,
                    'inflight_at_cutoff': len(self.pending), 'peak_active_flows': self.peak_active_flows,
                    'network_tick': self.network_tick, 'cached_latencies': 0,
                }, indent=2))
        finally:
            super().close()
