"""Whole-prefix KV retention in placed HBM, with pinned entries and LRU eviction.

Keys are trusted trace declarations of identical tokenized prefixes, including
tokenizer/template/tenant identity. This is not an automatic radix-tree matcher.
Only completed calls publish entries; cache hits share physical state in place.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import Sim.common as common
from Sim.config.modern_model_config import ModernAttentionBlockConfig
from Sim.config.utils import get_all_subclasses
from Sim.entities.mem_chiplet import intermediate


class BasePrefixCache(ABC):
    @classmethod
    def create_from_name(cls, name, memory, model, capacity):
        from Sim.execution import host_prefix_cache
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass(memory, model, capacity)
        raise ValueError(f'Unknown prefix cache: {name}')

    def configure_runtime(self, env, config, tools):
        """Optional host tier and transport context."""

    def ensure_available(self, request):
        """False delays admission while an asynchronous restore is in flight."""
        return True

    def pending_events(self):
        return []

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def peek(self, request):
        raise NotImplementedError

    @abstractmethod
    def acquire(self, request):
        raise NotImplementedError

    @abstractmethod
    def release(self, request):
        raise NotImplementedError

    @abstractmethod
    def record_admission(self, request):
        raise NotImplementedError

    @abstractmethod
    def evict_one(self, tick):
        raise NotImplementedError

    @abstractmethod
    def complete(self, request, tick):
        raise NotImplementedError

    @abstractmethod
    def snapshot(self):
        raise NotImplementedError


@dataclass
class PrefixEntry:
    tokens: int
    data: tuple
    size_bytes: int
    references: int = 0


class LRUPrefixCache(BasePrefixCache):
    def __init__(self, memory, model, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Prefix capacity must be a positive integer number of bytes.')
        if common.ByteperParam != 1 or any(type(config) is not ModernAttentionBlockConfig
                                          for config in model.hybrid_blocks):
            raise ValueError('Prefix reuse currently supports explicit dense ModernGQA/SwiGLU models with one-byte KV only.')
        self.memory, self.model, self.capacity = memory, model, capacity
        self.entries = OrderedDict()
        self.used_bytes = 0
        self.peak_bytes = 0
        self.hits = self.misses = self.evictions = self.insertions = 0
        self.reused_tokens = 0

    @staticmethod
    def get_name():
        return 'lru'

    def peek(self, request):
        entry = self.entries.get(request.prefix_key)
        return entry.tokens if entry is not None else 0

    def acquire(self, request):
        entry = self.entries.get(request.prefix_key)
        if entry is not None:
            entry.references += 1
            self.entries.move_to_end(request.prefix_key)
            request.cached_prefix_tokens = entry.tokens

    def record_admission(self, request):
        if request.cached_prefix_tokens:
            self.hits += 1
            self.reused_tokens += request.cached_prefix_tokens
        else:
            self.misses += 1

    def release(self, request):
        if request.cached_prefix_tokens:
            entry = self.entries[request.prefix_key]
            if entry.references <= 0:
                raise RuntimeError('Prefix reference count underflow.')
            entry.references -= 1
            self.entries.move_to_end(request.prefix_key)

    def evict_one(self, tick):
        for key, entry in self.entries.items():
            if entry.references:
                continue
            for data in entry.data:
                self.memory.offload_data_byid(data.chip_id, data, tick)
                self.memory.relieve_allocate_memchiplet(data.size, f'Block {data.block_id}')
                contents = self.memory.get_mem_byid(data.chip_id).content_params
                if not contents[data.req_id]:
                    del contents[data.req_id]
            self.used_bytes -= entry.size_bytes
            del self.entries[key]
            self.evictions += 1
            return True
        return False

    def publish(self, request, tick):
        if not request.prefix_key or request.prefix_key in self.entries:
            return
        sizes = [(index, self.model.hybrid_blocks[type_id].states * request.step.prefix_tokens)
                 for index, type_id in enumerate(self.model.block_type_sequence)]
        total = sum(size for _, size in sizes)
        if total > self.capacity:
            return
        by_chip = defaultdict(float)
        for index, size in sizes:
            by_chip[self.memory.blocks_alloc[index]] += size / 1024**2

        def fits():
            return self.used_bytes + total <= self.capacity and all(
                max(self.memory.get_mem_byid(chip)._current_allocated,
                    self.memory.get_mem_byid(chip).inuse_budget) + size <= self.memory.get_mem_byid(chip).dram_budget
                for chip, size in by_chip.items())

        while not fits():
            if not self.evict_one(tick):
                return
        data = []
        for index, size in sizes:
            item = intermediate(size=size / 1024**2, block_id=index,
                                name='Retained prefix KV', chip_id=self.memory.blocks_alloc[index],
                                req_id=f'prefix:{request.prefix_key}')
            self.memory.allocate_memchiplet(item.size, f'Block {index}')
            self.memory.load_data_byid(item.chip_id, item, tick)
            data.append(item)
        self.entries[request.prefix_key] = PrefixEntry(request.step.prefix_tokens, tuple(data), total)
        self.used_bytes += total
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        self.insertions += 1

    def complete(self, request, tick):
        # Execution has released private state; retain its prompt prefix at the
        # same timestamp and placement. This models ownership transfer, no copy.
        self.release(request)
        self.publish(request, tick)

    def snapshot(self):
        return dict(policy=self.get_name(), capacity_bytes=self.capacity,
                    resident_bytes=self.used_bytes, peak_bytes=self.peak_bytes,
                    entries=len(self.entries), pinned_references=sum(e.references for e in self.entries.values()),
                    hits=self.hits, misses=self.misses, reused_tokens=self.reused_tokens,
                    insertions=self.insertions, evictions=self.evictions,
                    scope='Completed-call whole-prefix retention; trusted identities; HBM only; no offload.')
