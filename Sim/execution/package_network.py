"""Effective package fabric, independent of the intra-package network backend."""

from abc import ABC, abstractmethod
from contextlib import ExitStack
from decimal import Decimal, ROUND_CEILING
import json
import math

import simpy

import Sim.common as common
from Sim.config.utils import get_all_subclasses


class BasePackageNetwork(ABC):
    @classmethod
    def create_from_name(cls, name):
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass
        raise ValueError(f'Unknown package communication model: {name}')

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def transfer(self, env, source, destination, byte_count, endpoint_gbps, metadata):
        raise NotImplementedError


class AlphaBetaPackageNetwork(BasePackageNetwork):
    """Nonblocking switch with one FIFO Tx and Rx port per package.

    Each port is held for alpha + bytes/B, including startup. This conservative
    message-level model captures port contention, not packets, credits, shared
    switch bisection, retransmissions or concurrent local NoI gateway traffic.
    """
    @staticmethod
    def get_name():
        return 'alpha_beta'

    def __init__(self, config, env, output_dir):
        self.config = config
        self.env = env
        self.tx = [simpy.Resource(env, capacity=1) for _ in range(config.count)]
        self.rx = [simpy.Resource(env, capacity=1) for _ in range(config.count)]
        self.submitted = 0
        self.completed = 0
        self.total_bytes = 0
        self.output_dir = output_dir
        with ExitStack() as resources:
            self.trace = resources.enter_context((output_dir / 'package_transfers.jsonl').open('w'))
            self.blocks = resources.enter_context((output_dir / 'package_blocks.jsonl').open('w'))
            resources.pop_all()

    def record_block(self, package, block_id, request_ids, stage, event):
        self.blocks.write(json.dumps(dict(package=package, block_id=block_id, request_ids=request_ids,
                                          stage=stage, event=event, tick=self.env.now)) + '\n')

    def transfer(self, env, source, destination, byte_count, endpoint_gbps, metadata):
        if env is not self.env or source == destination or not (0 <= source < self.config.count and 0 <= destination < self.config.count):
            raise ValueError('Invalid package transfer endpoints or environment.')
        if not math.isfinite(byte_count) or byte_count <= 0 or not math.isfinite(endpoint_gbps) or endpoint_gbps <= 0:
            raise ValueError('Package transfer bytes and endpoint bandwidth must be positive.')
        return env.process(self._transfer(source, destination, math.ceil(byte_count), endpoint_gbps, metadata))

    def _transfer(self, source, destination, byte_count, endpoint_gbps, metadata):
        self.submitted += 1
        transfer_id = self.submitted
        queued = self.env.now
        # All flows acquire Tx before Rx: no resource cycle. Independent
        # directions may overlap; flows with the same sender/receiver serialize.
        with self.tx[source].request() as tx:
            yield tx
            with self.rx[destination].request() as rx:
                yield rx
                started = self.env.now
                bandwidth = min(self.config.bandwidth_gbps * self.config.efficiency, endpoint_gbps)
                # Decimal avoids adding a spurious microsecond at exact tick
                # boundaries (e.g. binary float 11.000000000000002 -> 12).
                nanoseconds = Decimal(str(self.config.latency_ns)) + Decimal(byte_count) / Decimal(str(bandwidth))
                ticks = nanoseconds * common.time_granularity / Decimal(1_000_000_000)
                yield self.env.timeout(int(ticks.to_integral_value(rounding=ROUND_CEILING)))
                self.completed += 1
                self.total_bytes += byte_count
                record = dict(metadata, transfer_id=transfer_id, source_package=source,
                              destination_package=destination, bytes=byte_count, bandwidth_gbps=bandwidth,
                              queued_tick=queued, start_tick=started, completed_tick=self.env.now)
                self.trace.write(json.dumps(record) + '\n')

    def close(self):
        if self.trace is None:
            return
        with ExitStack() as resources:
            resources.callback(self.blocks.close)
            resources.callback(self.trace.close)
            self.trace = None
            (self.output_dir / 'package_network_summary.json').write_text(json.dumps({
                'model': self.get_name(), 'submitted_transfers': self.submitted,
                'completed_transfers': self.completed, 'inflight_at_cutoff': self.submitted - self.completed,
                'completed_bytes': self.total_bytes,
            }, indent=2) + '\n')
