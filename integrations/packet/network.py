"""Thin batched C ABI binding; all packet events execute in C++."""

import ctypes as ct
import hashlib
import math
import os
from pathlib import Path


class Completion(ct.Structure):
    _fields_ = [('flow', ct.c_uint64), ('tick', ct.c_uint64)]


class Statistics(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in (
        'events', 'packets', 'submitted', 'completed', 'received_bytes',
        'peak_buffer_bytes', 'blocked_checks')]


class PacketNetwork:
    def __init__(self, config, library=None):
        self.handle = None
        self.pending = set()
        self.now = 0
        self.config = dict(config)
        root = Path(__file__).resolve().parents[2]
        library = Path(library or os.environ.get('HYDRA_PACKET_LIBRARY') or root / '.hydra-packet-build/libhydra_packet.so').resolve()
        if not library.is_file():
            raise RuntimeError('HYDRA-Packet is not built. Run: bash scripts/setup_packet.sh')
        self.library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
        self.lib = ct.CDLL(str(library))
        self._bind()
        self._validate()
        config = self.config
        nodes = len(config['chiplet_ids'])
        period = round(1e12 / config['network_frequency_hz'])
        width = config['link_width_bits'] // 8
        self.flit_bytes = width
        bandwidth = (ct.c_double * nodes)(*[
            config['memory_bandwidth_gbps'].get(str(node), 0) * 1e9 for node in range(nodes)])
        self.handle = self.lib.hp_create(
            config['mesh_rows'], nodes // config['mesh_rows'], width, period,
            (config['router_latency_cycles'] + config['link_latency_cycles']) * period,
            config['packet_quantum_bytes'], config['packet_buffer_bytes'], bandwidth,
            config.get('dma_pacing', False), config.get('dma_burst_bytes', 4096),
            round(config['hbm_access_latency_ns'] * 1000))
        if not self.handle:
            self._check(-1)

    def _bind(self):
        integer, u64, pointer = ct.c_int, ct.c_uint64, ct.c_void_p
        self.lib.hp_abi_version.restype = integer
        if self.lib.hp_abi_version() != 1:
            raise RuntimeError('Incompatible HYDRA-Packet library; rebuild it.')
        self.lib.hp_error.restype = ct.c_char_p
        self.lib.hp_create.argtypes = [integer, integer] + [u64] * 5 + [ct.POINTER(ct.c_double), integer, u64, u64]
        self.lib.hp_create.restype = pointer
        self.lib.hp_destroy.argtypes = [pointer]
        self.lib.hp_destroy.restype = None
        self.lib.hp_submit.argtypes = [pointer, u64, integer, integer, u64]
        self.lib.hp_submit.restype = integer
        self.lib.hp_advance.argtypes = [pointer, u64, ct.POINTER(Completion), u64, ct.POINTER(u64), ct.POINTER(u64)]
        self.lib.hp_advance.restype = integer
        self.lib.hp_stats.argtypes = [pointer, ct.POINTER(Statistics)]
        self.lib.hp_stats.restype = integer

    def _validate(self):
        cfg = self.config
        nodes, rows = len(cfg['chiplet_ids']), cfg['mesh_rows']
        if rows <= 0 or nodes == 0 or nodes > 4096 or nodes % rows or cfg['chiplet_ids'] != list(range(nodes)):
            raise ValueError('HYDRA-Packet requires a row-major rectangular mesh with contiguous IDs.')
        columns = nodes // rows
        package_nodes = cfg.get('nodes_per_package', nodes)
        if (not isinstance(package_nodes, int) or package_nodes <= 0 or nodes % package_nodes
                or package_nodes % columns or cfg.get('package_count', 1) != nodes // package_nodes):
            raise ValueError('Package meshes must consist of complete contiguous mesh rows.')
        if len(cfg['adjacency']) != nodes or any(len(row) != nodes for row in cfg['adjacency']):
            raise ValueError('Adjacency dimensions do not match the mesh.')
        for a in range(nodes):
            for b in range(nodes):
                adjacent = abs(a // columns - b // columns) + abs(a % columns - b % columns) == 1
                adjacent = adjacent and a // package_nodes == b // package_nodes
                if bool(cfg['adjacency'][a][b]) != adjacent:
                    raise ValueError('HYDRA-Packet requires complete rectangular meshes within packages.')
        frequency = cfg['network_frequency_hz']
        if not math.isfinite(frequency) or frequency <= 0 or frequency > 1e12:
            raise ValueError('Network frequency must be positive and representable in picoseconds.')
        period = 1e12 / frequency
        if not math.isclose(period, round(period), rel_tol=0, abs_tol=1e-6):
            raise ValueError('Network period must be an integer number of picoseconds.')
        if cfg['link_width_bits'] <= 0 or cfg['link_width_bits'] % 8:
            raise ValueError('Link width must be a positive whole number of bytes.')
        width = cfg['link_width_bits'] // 8
        for key in ('packet_quantum_bytes', 'packet_buffer_bytes'):
            if not isinstance(cfg[key], int) or not width <= cfg[key] <= 2**32:
                raise ValueError(f'{key} must hold at least one flit (at most 4 GiB).')
            cfg['requested_' + key] = cfg[key]
            cfg[key] = cfg[key] // width * width
        if cfg.get('dma_pacing', False):
            if not width <= cfg['dma_burst_bytes'] <= 2**32:
                raise ValueError('DMA burst must hold at least one flit (at most 4 GiB).')
            cfg['requested_dma_burst_bytes'] = cfg['dma_burst_bytes']
            cfg['dma_burst_bytes'] = cfg['dma_burst_bytes'] // width * width
        if cfg['router_latency_cycles'] <= 0 or cfg['link_latency_cycles'] <= 0 or cfg['hbm_access_latency_ns'] < 0:
            raise ValueError('Router/link delays must be positive; HBM access delay must be nonnegative.')

    def _check(self, status):
        if status:
            raise RuntimeError(self.lib.hp_error().decode())

    def execute(self, request):
        if not self.handle:
            raise RuntimeError('HYDRA-Packet network is closed.')
        completed = []
        if request['kind'] == 'submit':
            flow = request['flow_id']
            byte_count = request['bytes']
            if not isinstance(flow, int) or not 0 <= flow < 2**64 or flow in self.pending:
                raise ValueError('Invalid or duplicate flow ID.')
            if not math.isfinite(byte_count) or not 0 < byte_count <= 2**52:
                raise ValueError('Transfer bytes must be positive and exactly representable.')
            for name in ('source', 'destination'):
                if not isinstance(request[name], int) or not 0 <= request[name] < len(self.config['chiplet_ids']):
                    raise ValueError('Invalid transfer endpoint.')
            # The C++ XY engine allocates a rectangular queue array. Package
            # boundary queues are unreachable: every submission is local and
            # XY paths between local endpoints stay inside their row range.
            package_nodes = self.config.get('nodes_per_package', len(self.config['chiplet_ids']))
            if request['source'] // package_nodes != request['destination'] // package_nodes:
                raise ValueError('Cross-package traffic must use the package fabric.')
            rounded = math.ceil(byte_count / self.flit_bytes) * self.flit_bytes
            self._check(self.lib.hp_submit(self.handle, flow, request['source'], request['destination'], rounded))
            self.pending.add(flow)
        elif request['kind'] == 'advance':
            target = request['target_tick']
            if not isinstance(target, int) or not self.now <= target < 2**63:
                raise ValueError('Invalid target tick or reversed network time.')
            output = (Completion * len(self.pending))()
            count, reached = ct.c_uint64(), ct.c_uint64()
            self._check(self.lib.hp_advance(self.handle, target, output, len(output), ct.byref(count), ct.byref(reached)))
            self.now = reached.value
            for item in output[:count.value]:
                self.pending.remove(item.flow)
                completed.extend([item.flow, item.tick])
        else:
            raise ValueError('Unknown packet network request.')
        return {'request_id': request.get('request_id'), 'tick': self.now, 'completed': completed}

    def statistics(self):
        result = Statistics()
        self._check(self.lib.hp_stats(self.handle, ct.byref(result)))
        return {name: getattr(result, name) for name, _ in result._fields_}

    def close(self):
        if self.handle:
            self.lib.hp_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
