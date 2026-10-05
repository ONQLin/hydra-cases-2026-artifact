"""Disconnected local meshes, contiguous decoder stages and local weights."""

from abc import ABC, abstractmethod
from copy import deepcopy

import networkx as nx

import Sim.common as common
from Sim.config.utils import get_all_subclasses
from Sim.entities.chips_network import chip_graph
from Sim.entities.mem_sys import mem_sys
from Sim.entities.mem_chiplet import Params
from Sim.entities.static_mapper import static_mapper
from Sim.metrics.monitor import mem_monitor


class BasePackagePartition(ABC):
    @classmethod
    def create_from_name(cls, name):
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass()
        raise ValueError(f'Unknown package partition: {name}')

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def assign(self, model, count):
        raise NotImplementedError


class ContiguousPackagePartition(BasePackagePartition):
    @staticmethod
    def get_name():
        return 'contiguous'

    def assign(self, model, count):
        layers = len(model.block_type_sequence)
        if not 1 <= count <= layers:
            raise ValueError('Each package must own at least one decoder layer.')
        quotient, remainder = divmod(layers, count)
        return {layer: package for package in range(count)
                for layer in range(package * quotient + min(package, remainder),
                                   (package + 1) * quotient + min(package + 1, remainder))}


class PackageGraph(chip_graph):
    """Global IDs preserve local XY geometry; no inter-package NoI edges exist.

    The effective package fabric is a separate SimPy resource model. Keeping
    it out of this graph prevents weight/KV traffic from using remote HBMs.
    """
    def __init__(self, local_graph, count):
        self.package_count = count
        self.nodes_per_package = local_graph.num_nodes
        self.num_nodes = count * local_graph.num_nodes
        self.intp_width = count * local_graph.intp_width
        self.intp_height = local_graph.intp_height
        self.two_d_grid = True
        self.graph = nx.Graph()
        for package in range(count):
            offset = package * local_graph.intp_width
            for (x, y), attrs in local_graph.graph.nodes(data=True):
                attrs = deepcopy(attrs)
                attrs.update(id=package * self.nodes_per_package + attrs['id'], package_id=package)
                self.graph.add_node((x + offset, y), **attrs)
            for (x, y), (u, v), attrs in local_graph.graph.edges(data=True):
                self.graph.add_edge((x + offset, y), (u + offset, v), **deepcopy(attrs))
        self.id_to_coord = {a['id']: node for node, a in self.graph.nodes(data=True)}
        self.coord_to_id = {node: a['id'] for node, a in self.graph.nodes(data=True)}

    def package_of(self, chiplet_id):
        return self.graph.nodes[self.id_to_coord[chiplet_id]]['package_id']


class PackageMemorySystem(mem_sys):
    def __init__(self, graph, layer_packages):
        super().__init__(graph)
        self.package_graph = graph
        self.layer_packages = layer_packages

    def load_model(self, label, mod_config, group_HBM_M, group_HBM_A):
        # Plan the whole placement before mutating any allocation. Whole layers
        # remain on one HBM, matching the existing pipeline execution contract.
        remaining = {m.chiplet_id: m.dram_budget - m.inuse_budget for m in self._mem_chiplets}
        allocation = []
        for layer, block_type in enumerate(mod_config.block_type_sequence):
            block = mod_config.hybrid_blocks[block_type]
            size = common.convert_param_mB(block.parameter_count)
            candidates = [m for m in self._mem_chiplets
                          if self.package_graph.package_of(m.chiplet_id) == self.layer_packages[layer]
                          and remaining[m.chiplet_id] >= size]
            if not candidates:
                raise MemoryError(f'Package {self.layer_packages[layer]} cannot place layer {layer} '
                                  f'({size / 1024:.3f} GiB) on a local HBM; increase local capacity or packages.')
            memory = min(candidates, key=lambda m: (-remaining[m.chiplet_id], m.chiplet_id))
            remaining[memory.chiplet_id] -= size
            allocation.append((layer, memory, size))
        for layer, memory, size in allocation:
            memory.inuse_budget += size
            memory._current_allocated += size
            memory.content_params.setdefault('weights', []).append(
                Params(size=size, block_id=layer, name=f'Block {layer} weights', chip_id=memory.chiplet_id))
            self._current_used += size
            self.blocks_alloc[layer] = memory.chiplet_id
        mem_monitor.update_utilization(self._mem_chiplets, self._current_used, 0)


class PackageStaticMapper(static_mapper):
    def __init__(self, graph, layer_packages):
        self.graph = graph
        self.layer_packages = layer_packages

    def is_eligible(self, chiplet, block_id):
        return self.graph.package_of(chiplet.chiplet_id) == self.layer_packages[block_id]


class PackageSystem:
    """Compose the existing placed package without creating extra simulators."""
    def __init__(self, config, local_graph, model):
        self.graph = PackageGraph(local_graph, config.count)
        self.layer_packages = BasePackagePartition.create_from_name(config.partition).assign(model, config.count)
        self.memory = PackageMemorySystem(self.graph, self.layer_packages)
        self.mapper = PackageStaticMapper(self.graph, self.layer_packages)

    def snapshot(self):
        return {'count': self.graph.package_count, 'nodes_per_package': self.graph.nodes_per_package,
                'layer_packages': self.layer_packages,
                'memory_capacity_gib': {str(p): sum(m.dram_budget for m in self.memory._mem_chiplets
                    if self.graph.package_of(m.chiplet_id) == p) / 1024
                    for p in range(self.graph.package_count)}}
