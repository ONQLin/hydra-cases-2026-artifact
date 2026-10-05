"""Inclusive host-DRAM prefix tier with explicit, serialized DMA transfers.

Completed prefixes are written through asynchronously. Only completed host
copies can restore an HBM miss. This effective off-chip link is shared by both
directions and by all prefixes; it is not a placed CPU or a Garnet endpoint.
"""

from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal

import simpy

from Sim.execution.prefix_cache import LRUPrefixCache
from Sim.execution.tool_system import BaseToolModel
from Sim.metrics.monitor import request_counter


@dataclass
class HostPrefixEntry:
    tokens: int
    size_bytes: int
    ready: bool = False
    references: int = 1


class HostPrefixMemory:
    """One host budget partitioned into reserved tool workspace and an LRU tier."""
    def __init__(self, capacity, tool_reservation):
        if tool_reservation >= capacity:
            raise ValueError('Host DRAM must exceed the reserved CPU tool workspace capacity.')
        self.capacity = capacity
        self.tool_reservation = tool_reservation
        self.cache_capacity = capacity - tool_reservation
        self.entries = OrderedDict()
        self.used_bytes = self.peak_bytes = self.evictions = 0

    def reserve(self, key, tokens, size):
        if key in self.entries:
            raise ValueError('A host prefix allocation already exists for this identity.')
        if size > self.cache_capacity:
            return None
        while self.used_bytes + size > self.cache_capacity:
            victim = next((k for k, entry in self.entries.items() if not entry.references), None)
            if victim is None:
                return None
            self.used_bytes -= self.entries.pop(victim).size_bytes
            self.evictions += 1
        entry = HostPrefixEntry(tokens, size)
        self.entries[key] = entry
        self.used_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        return entry

    def snapshot(self):
        return dict(capacity_bytes=self.capacity, tool_workspace_reserved_bytes=self.tool_reservation,
                    kv_capacity_bytes=self.cache_capacity, kv_used_bytes=self.used_bytes,
                    kv_peak_bytes=self.peak_bytes, entries=len(self.entries), evictions=self.evictions,
                    pinned_references=sum(e.references for e in self.entries.values()))


class HostKVTransferEngine:
    def __init__(self, env, config):
        self.env = env
        self.link = simpy.Resource(env, capacity=1)
        self.bandwidth = min(config.host_link_bandwidth_gbps, config.host_dram_bandwidth_gbps)
        self.latency = config.host_transfer_latency_s
        self.records = []

    def transfer(self, key, size, direction):
        seconds = Decimal(str(self.latency)) + Decimal(size) / (Decimal(str(self.bandwidth))*10**9)
        duration = BaseToolModel.ticks(seconds)
        row = dict(prefix_id=key, direction=direction, bytes=size, submitted_tick=self.env.now,
                   started_tick=None, completed_tick=None, service_ticks=duration)
        self.records.append(row)
        with self.link.request() as slot:
            yield slot
            row['started_tick'] = self.env.now
            yield self.env.timeout(duration)
            row['completed_tick'] = self.env.now


class HostOffloadPrefixCache(LRUPrefixCache):
    @staticmethod
    def get_name():
        return 'lru_offload'

    def configure_runtime(self, env, config, tools):
        self.env = env
        self.host = HostPrefixMemory(config.host_dram_capacity_bytes,
                                     tools.memory.capacity if tools.memory is not None else 0)
        self.transport = HostKVTransferEngine(env, config)
        self.pending = {}
        self.loading = {}
        self.restore_owners = {}
        self.restores = self.backups = 0

    def pending_events(self):
        return [event for event in self.pending.values() if not event.triggered]

    def peek(self, request):
        if request.prefix_key in self.loading:
            return 0
        return super().peek(request)

    def acquire(self, request):
        if request.prefix_key in self.loading:
            raise RuntimeError('Prefix KV cannot be read before host restore completes.')
        key = request.prefix_key
        if self.restore_owners.get(key) == request._id:
            # Convert the completed-restore lease to a normal request reference.
            self.entries[key].references -= 1
            del self.restore_owners[key]
        super().acquire(request)
        if key in self.host.entries:
            self.host.entries.move_to_end(key)

    def publish(self, request, tick):
        super().publish(request, tick)
        key = request.prefix_key
        if key not in self.entries or key in self.host.entries:
            return
        entry = self.entries[key]
        host_entry = self.host.reserve(key, entry.tokens, entry.size_bytes)
        if host_entry is None:
            return
        entry.references += 1
        self.pending[('backup', key)] = self.env.process(self.backup(key, entry, host_entry))

    def backup(self, key, entry, host_entry):
        yield from self.transport.transfer(key, entry.size_bytes, 'hbm_to_host')
        host_entry.ready = True
        host_entry.references -= 1
        entry.references -= 1
        self.backups += 1
        del self.pending[('backup', key)]

    def ensure_available(self, request):
        key = request.prefix_key
        if key in self.loading:
            return False
        if key in self.entries or getattr(request, 'skip_host_restore', False):
            return True
        host_entry = self.host.entries.get(key)
        if host_entry is None or not host_entry.ready or host_entry.size_bytes > self.capacity:
            return True  # Cold or uncacheable request: recompute.
        # Reserve the actual placed destination before starting DMA. Bypass
        # write-through publication: this immutable prefix already exists in DRAM.
        LRUPrefixCache.publish(self, request, self.env.now)
        entry = self.entries.get(key)
        if entry is None:
            return not (self.pending_events() or request_counter.running_batches)
        entry.references += 1
        host_entry.references += 1
        self.host.entries.move_to_end(key)
        self.restore_owners[key] = request._id
        process = self.env.process(self.restore(key, entry, host_entry))
        self.loading[key] = process
        self.pending[('restore', key)] = process
        return False

    def restore(self, key, entry, host_entry):
        yield from self.transport.transfer(key, entry.size_bytes, 'host_to_hbm')
        host_entry.references -= 1
        # Keep the destination pinned until its owner reaches admission. This
        # prevents another completion from evicting freshly restored data.
        self.restores += 1
        del self.loading[key]
        del self.pending[('restore', key)]

    def snapshot(self):
        result = super().snapshot()
        result.update(scope='Whole-prefix HBM + host DRAM; async write-through and demand restore; no active-request swapping.',
                      host=self.host.snapshot(), backups=self.backups, restores=self.restores,
                      pending_transfers=len(self.pending_events()),
                      restore_admission_leases=len(self.restore_owners),
                      transfer_model='Shared serialized alpha-beta link; effective endpoint bandwidth; outside Garnet/NoI/HBM port queues.',
                      transfers=self.transport.records)
        return result
