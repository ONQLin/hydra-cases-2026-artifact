"""Hardware and mapping export shared by the packet and Garnet transports."""


class NetworkSystemSnapshot:
    def __init__(self, config):
        self.config = config

    def export(self, graph, memory_system, mapping):
        cfg = self.config.chipsim_config
        if cfg.network_frequency_hz <= 0:
            raise ValueError('Network frequency must be positive.')
        nodes = sorted(graph.graph.nodes, key=lambda node: graph.graph.nodes[node]['id'])
        ids = [int(graph.graph.nodes[node]['id']) for node in nodes]
        if ids != list(range(len(nodes))):
            raise ValueError("Network execution requires contiguous chiplet IDs.")
        import networkx as nx
        snapshot = {
            'schema_version': 1,
            'chiplet_ids': ids,
            'adjacency': nx.to_numpy_array(graph.graph, nodelist=nodes, weight=None).astype(int).tolist(),
            'chiplet_types': [int(graph.graph.nodes[node]['chiplet_type']) for node in nodes],
            'weight_mapping': memory_system.blocks_alloc,
            'task_mapping': mapping,
            'memory_bandwidth_gbps': {str(m.chiplet_id): float(m.BW_budget) for m in memory_system._mem_chiplets},
            'mesh_rows': graph.intp_width,
            'reuse_mesh_paths': cfg.reuse_mesh_paths,
            'link_width_bits': max(8, round(self.config.chips_config.D2D_NoI_bw * 1e9 / cfg.network_frequency_hz) * 8),
            'network_frequency_hz': cfg.network_frequency_hz,
            'router_latency_cycles': cfg.router_latency_cycles,
            'link_latency_cycles': cfg.link_latency_cycles,
            'virtual_channels_per_vnet': cfg.virtual_channels_per_vnet,
            'garnet_sim_cycles': cfg.garnet_sim_cycles,
            'garnet_ticks_per_cycle': cfg.garnet_ticks_per_cycle,
            'hbm_access_latency_ns': cfg.hbm_access_latency_ns,
            'timeout_seconds': cfg.timeout_seconds,
            'model': self.config.workload_config.model,
            'execution_contract': 'sequential kernel phases, overlapped compute/HBM/NoI, HYDRA bandwidth reservations',
            'traffic_contract': 'aggregate accelerator HBM bytes represented as HBM-to-compute bursts; inter-block HBM-to-HBM transfers explicit',
        }
        return snapshot
