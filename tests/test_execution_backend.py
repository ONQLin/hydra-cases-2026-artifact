"""Check the contract between HYDRA accelerator profiles and CHIPSIM phases."""

from pathlib import Path
import tempfile
import random
import numpy as np
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import Sim.common as common
import Sim.config.utils as utils
from Sim.config.model_config import BaseModelConfig
from Sim.config.sys_config import HPSim_Config, MetricsConfig
from Sim.entities.block import block
from Sim.execution import BaseExecutionBackend
from Sim.execution.native import NativeExecutionBackend
from Sim.backends import BaseSimulationBackend
from Sim.backends.chipsim_runtime import ChipsimRuntimeBackend
from Sim.simulator import Simulator


class ExecutionBackendTest(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.config = HPSim_Config(metrics_config=MetricsConfig(output_dir=self.output.name))
        Simulator(self.config)

    def test_exported_phases_reconstruct_native_kernel_latency(self):
        # The export must preserve max(compute, memory) and serial spill phases,
        # including SSM, attention, normalization, and both serving stages.
        backend = NativeExecutionBackend(self.config)
        for model_name in ('llama3-8b', 'nemotronh-4b'):
            model = BaseModelConfig.create_from_name(model_name)
            for block_config in model.hybrid_blocks:
                family = 'marca' if 'mamba' in block_config.type_name else 'tscs'
                for stage, suffix in (('prefill', 'p'), ('decode', 'd')):
                    chiplet = SimpleNamespace(chiplet_type=utils.chiplet_types_list.index(f'{family}_{suffix}'))
                    for length in (11, 128, 4096):
                        for bandwidth in (64, 256):
                            with self.subTest(model=model_name, block=block_config.type_name,
                                              stage=stage, length=length, bandwidth=bandwidth):
                                profiles = backend.profile_block(block(length, block_config), stage, 2, chiplet, bandwidth)
                                for profile in profiles:
                                    self.assertTrue(profile.execution_phases)
                                    reconstructed = sum(max(p.compute_ns, p.memory_bytes / bandwidth)
                                                        for p in profile.execution_phases)
                                    # Original kernels round intermediate nanosecond costs down.
                                    self.assertAlmostEqual(reconstructed, profile.total_latency,
                                                           delta=len(profile.execution_phases) + 1)

    def test_setup_applies_seed_to_placement_random_sources(self):
        Simulator(self.config)
        first = (random.random(), np.random.random())
        Simulator(self.config)
        self.assertEqual(first, (random.random(), np.random.random()))

    def test_factories_expose_native_snapshot_and_runtime(self):
        for name in ('hydra_sim', 'chipsim', 'chipsim_runtime', 'chipsim_contended', 'hydra_packet'):
            self.assertEqual(BaseSimulationBackend.create_from_name(name).get_name(), name)
        for name in ('hydra_sim', 'chipsim_runtime', 'chipsim_contended', 'hydra_packet'):
            self.assertEqual(BaseExecutionBackend.create_from_name(name).get_name(), name)
        with self.assertRaises(ValueError):
            BaseExecutionBackend.create_from_name('missing')

    def test_runtime_rejects_unsupported_policies_before_simulation(self):
        self.config.mapping_config.mapping_strategy = 'elastic'
        with patch.object(Simulator, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'static task mapping'):
                ChipsimRuntimeBackend(self.config).run()
            run.assert_not_called()

    def test_runtime_rejects_unsupported_dma_pacing_before_simulation(self):
        self.config.chipsim_config.dma_pacing = True
        with patch.object(Simulator, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'requires the chipsim_contended'):
                ChipsimRuntimeBackend(self.config).run()
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
