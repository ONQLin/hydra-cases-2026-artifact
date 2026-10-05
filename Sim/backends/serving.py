"""Shared policy validation for network-refined serving backends."""


class StaticServingMixin:
    supports_dma_pacing = False

    def run(self):
        if self.config.chipsim_config.dma_pacing and not self.supports_dma_pacing:
            raise ValueError('DMA pacing requires the chipsim_contended or hydra_packet backend.')
        if self.config.time_limit <= 0:
            raise ValueError(f"{self.get_name()} requires a positive measurement window.")
        if not self.config.chipsim_config.use_hydra_compute_backend:
            raise ValueError(f"{self.get_name()} requires the shared HYDRA compute profiles.")
        if self.config.mapping_config.mapping_strategy != "static":
            raise ValueError(f"{self.get_name()} currently validates static task mapping only.")
        if self.config.mapping_config.task_parallelism != "pipeline":
            raise ValueError(f"{self.get_name()} currently supports pipeline parallelism only.")
        if self.config.cluster_config.local_scheduler not in ("static", "agent", "vllm_latest"):
            raise ValueError(f"{self.get_name()} requires static, agent, or vllm_latest scheduling.")
        if self.config.workload_config.bytes_per_param != 1:
            raise ValueError(f"{self.get_name()} phase exports currently use the native one-byte profiles.")
        if not self.config.placmt_config.two_d_grid:
            raise ValueError(f"{self.get_name()} currently validates uniform grid topologies only.")
        return super().run()
