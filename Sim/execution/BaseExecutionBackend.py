"""Execution backend contract at HYDRA's block and transfer boundaries."""

from abc import ABC, abstractmethod
import simpy
from Sim.config.utils import get_all_subclasses


class BaseExecutionBackend(ABC):
    @classmethod
    def create_from_name(cls, name: str):
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass
        raise ValueError(f"[{cls.__name__}] Invalid name: {name}")

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    def initialize(self, graph, memory_system, compute_system, mapping):
        """Bind the backend to the already placed HYDRA system."""

    def create_environment(self):
        return simpy.Environment(initial_time=0)

    @abstractmethod
    def start_block(self, env, *args):
        """Return a completion event and utilization for an admitted block."""
        raise NotImplementedError

    def start_transfer(self, env, *args):
        return env.timeout(int(self.transfer(*args)))

    def transfer(self, source, destination, size_mb, bandwidth, native_ticks):
        return native_ticks

    def close(self):
        """Release resources after success or partial initialization failure.

        Repeated calls must be safe; diagnostic writes must not prevent release.
        """

    def initialize_packages(self, env, graph, output_dir):
        """Bind the independent effective fabric after the local backend starts."""
        from Sim.execution.package_network import BasePackageNetwork
        self.package_graph = graph
        self.package_network = BasePackageNetwork.create_from_name(
            self.config.package_config.communication_model)(self.config.package_config, env, output_dir)

    def is_package_transfer(self, source, destination):
        return (getattr(self, 'package_network', None) is not None
                and self.package_graph.package_of(source) != self.package_graph.package_of(destination))

    def record_package_block(self, chiplet_id, block_id, request_ids, stage, event):
        network = getattr(self, 'package_network', None)
        if network is not None:
            network.record_block(self.package_graph.package_of(chiplet_id), block_id, request_ids, stage, event)

    def transfer_package_input(self, env, graph, memory, item, destination):
        """Reserve both local HBM endpoints until remote input is available."""
        source = memory.get_mem_byid(item.chip_id)
        target = memory.get_mem_byid(destination)
        # Reuse HYDRA's HBM bandwidth admission; the effective gateway path
        # has its own bandwidth ceiling and does not reserve local NoI links.
        import Sim.common as common
        import Sim.config.utils as utils
        bandwidth = min(int(utils.IO_bw / 3), 128)
        while not (target.is_capacity_available(item.size)
                   and graph.check_bw_availability(source.chiplet_id, bandwidth)
                   and graph.check_bw_availability(destination, bandwidth)):
            yield env.timeout(common.simulation_clk)
        memory.load_data_byid(destination, item, env.now)
        graph.load_bw_byid(source.chiplet_id, bandwidth, env.now)
        graph.load_bw_byid(destination, bandwidth, env.now)
        try:
            yield self.package_network.transfer(
                env, graph.package_of(source.chiplet_id), graph.package_of(destination), item.size * 1024**2,
                min(bandwidth, utils.NoI_bw),
                {'source_hbm': source.chiplet_id, 'destination_hbm': destination,
                 'request_id': item.req_id, 'producer_block': item.block_id,
                 'kind': 'token_feedback' if 'feedback' in item.name else 'activation'})
        except BaseException:
            # A failed/cancelled transfer leaves the producer's copy valid and
            # rolls back the destination capacity reserved for this message.
            memory.offload_data_byid(destination, item, env.now)
            raise
        finally:
            graph.offload_bw_byid(source.chiplet_id, bandwidth, env.now)
            graph.offload_bw_byid(destination, bandwidth, env.now)
        memory.offload_data_byid(source.chiplet_id, item, env.now)
        item.chip_id = destination
