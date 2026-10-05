import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest

import yaml

from Sim.backends.chipsim import (
    ChipsimBackend,
    ChipsimBackendError,
    ChipsimTranslation,
)
from Sim.config.sys_config import ChipsimConfig, HPSim_Config, MetricsConfig
from integrations.chipsim.run_hydra_experiment import (
    _rewrite_garnet_endpoint_ids,
)


class ChipsimBackendTest(unittest.TestCase):
    def _config(self, root: Path) -> HPSim_Config:
        chipsim_config: ChipsimConfig = ChipsimConfig(
            submodule_dir="third_party/CHIPSIM",
            python_executable=".chipsim-venv/bin/python",
        )
        metrics_config: MetricsConfig = MetricsConfig(
            output_dir=str(root / "output"),
            label_name="chipsim-test",
        )
        config: HPSim_Config = HPSim_Config(
            chipsim_config=chipsim_config,
            metrics_config=metrics_config,
        )
        config.cluster_config.batch_size = 1
        trace_config = (
            config.workload_config.request_generator_config
            .trace_length_generator_config
        )
        trace_config.trace_file = str(
            Path(__file__).resolve().parents[1]
            / "dataset"
            / "chat"
            / "chat1m_stats_llama3.csv"
        )
        return config

    def _fixture(self, root: Path, with_garnet: bool = True) -> None:
        chipsim: Path = root / "third_party" / "CHIPSIM"
        (chipsim / "integrations" / "gem5" / "build" / "Garnet_standalone").mkdir(
            parents=True
        )
        (chipsim / "simulate.py").write_text("", encoding="utf-8")

        runner: Path = root / "integrations" / "chipsim" / "run_hydra_experiment.py"
        runner.parent.mkdir(parents=True)
        runner.write_text("", encoding="utf-8")

        python: Path = root / ".chipsim-venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' "
            "'{\"missing\": [], \"mismatched\": [], \"broken\": []}'\n",
            encoding="utf-8",
        )
        os.chmod(python, 0o755)

        requirements: Path = chipsim / "docs" / "requirements.txt"
        requirements.parent.mkdir(parents=True)
        requirements.write_text("numpy==2.0.1\n", encoding="utf-8")

        if with_garnet:
            garnet: Path = (
                chipsim
                / "integrations"
                / "gem5"
                / "build"
                / "Garnet_standalone"
                / "gem5.opt"
            )
            garnet.write_text("", encoding="utf-8")
            os.chmod(garnet, 0o755)

    def test_translates_hpsim_config_into_chipsim_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root: Path = Path(temp_dir)
            self._fixture(root)
            backend: ChipsimBackend = ChipsimBackend(
                self._config(root), repo_root=root
            )

            backend.validate_environment()
            translation: ChipsimTranslation = backend.translate_config()
            backend.validate_translation()

            with translation.generated_config.open(
                "r", encoding="utf-8"
            ) as stream:
                experiment: dict[str, Any] = yaml.safe_load(stream)
            with translation.chiplet_specs.open("r", encoding="utf-8") as stream:
                specs: list[dict[str, Any]] = json.load(stream)
            with translation.manifest.open("r", encoding="utf-8") as stream:
                manifest: dict[str, Any] = json.load(stream)
            with translation.weight_placement.open(
                "r", encoding="utf-8"
            ) as stream:
                weight_placement: dict[str, Any] = json.load(stream)
            with translation.execution_mapping.open(
                "r", encoding="utf-8"
            ) as stream:
                execution_mapping: dict[str, Any] = json.load(stream)

            self.assertEqual(translation.sequence_length, 11)
            self.assertEqual(manifest["hydra_model"], "nemotronh-4b")
            self.assertEqual(manifest["translated_num_blocks"], 1)
            self.assertEqual(manifest["hydra_block_ids"], [4])
            self.assertEqual(manifest["batch_size"], 1)
            self.assertEqual(len(specs), 10)
            self.assertEqual(
                len(
                    [
                        spec
                        for spec in specs
                        if spec.get("memory_backend") == "HYDRA_HBM"
                    ]
                ),
                6,
            )
            self.assertEqual(len(weight_placement["blocks"]), 28)
            self.assertEqual(
                set(weight_placement["layer_to_block"].values()),
                {4},
            )
            self.assertEqual(execution_mapping["stage"], "prefill")
            self.assertEqual(
                len(set(execution_mapping["layer_mapping"].values())),
                1,
            )
            self.assertEqual(
                experiment["simulation"]["hardware_parameters"]["bits_per_packet"],
                2048,
            )
            self.assertIn(
                "MHA_QKV",
                (translation.input_dir / "hydra_model_definitions.py").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertEqual(
                backend.command(),
                [
                    str(root / ".chipsim-venv" / "bin" / "python"),
                    str(root / "integrations" / "chipsim" / "run_hydra_experiment.py"),
                    "--config",
                    str(translation.generated_config),
                    "--chiplet-specs",
                    str(translation.chiplet_specs),
                    "--mesh-rows",
                    "4",
                    "--link-width-bits",
                    "2048",
                    "--router-latency",
                    "1",
                    "--link-latency",
                    "1",
                    "--vcs-per-vnet",
                    "1",
                ],
            )

    def test_reports_missing_garnet_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root: Path = Path(temp_dir)
            self._fixture(root, with_garnet=False)
            backend: ChipsimBackend = ChipsimBackend(
                self._config(root), repo_root=root
            )

            with self.assertRaisesRegex(ChipsimBackendError, "Garnet is not built"):
                backend.validate_environment()

    def test_reports_all_missing_installation_components(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root: Path = Path(temp_dir)
            backend: ChipsimBackend = ChipsimBackend(
                self._config(root), repo_root=root
            )

            with self.assertRaises(ChipsimBackendError) as raised:
                backend.validate_environment()

            message: str = str(raised.exception)
            self.assertIn("git submodule update --init --recursive", message)
            self.assertIn("bash scripts/setup_chipsim.sh", message)
            self.assertIn("HYDRA CHIPSIM runner is missing", message)

    def test_reports_broken_python_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root: Path = Path(temp_dir)
            self._fixture(root)
            python: Path = root / ".chipsim-venv" / "bin" / "python"
            python.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' "
                "'{\"missing\": [\"numpy\"], \"mismatched\": [], "
                "\"broken\": []}'\n",
                encoding="utf-8",
            )

            backend: ChipsimBackend = ChipsimBackend(
                self._config(root), repo_root=root
            )
            with self.assertRaisesRegex(
                ChipsimBackendError, "missing Python packages: numpy"
            ):
                backend.validate_environment()

    def test_rewrites_garnet_directory_offset_for_hydra_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            traffic_file: Path = Path(temp_dir) / "traffic.txt"
            traffic_file.write_text(
                "# cycle i_ni i_router o_ni o_router vnet flits\n"
                "1 3 3 107 7 0 64\n",
                encoding="utf-8",
            )

            _rewrite_garnet_endpoint_ids(str(traffic_file), node_count=24)

            expected: str = (
                "# cycle i_ni i_router o_ni o_router vnet flits\n"
                "1 3 3 31 7 0 64\n"
            )
            self.assertEqual(traffic_file.read_text(encoding="utf-8"), expected)
            with gzip.open(
                traffic_file.with_suffix(".gz"), "rt", encoding="utf-8"
            ) as stream:
                self.assertEqual(stream.read(), expected)

    def test_hydra_compute_backend_uses_exported_profile(self) -> None:
        chipsim_root: Path = (
            Path(__file__).resolve().parents[1] / "third_party" / "CHIPSIM"
        )
        sys.path.insert(0, str(chipsim_root))
        from src.integrations.hydra.compute_backend import HydraComputeBackend

        backend: HydraComputeBackend = HydraComputeBackend()
        result: dict[str, float | int] = backend.simulate(
            partitioned_layer={
                "percentage": 25.0,
                "hydra_compute_profiles": {
                    "HYDRA_TSCS_P": {
                        "latency_us": 40.0,
                        "energy_fj": 800.0,
                        "frequency_hz": 1e9,
                    }
                },
            },
            chiplet_id=2,
            chiplet_type="HYDRA_TSCS_P",
        )

        self.assertEqual(result["latency_us"], 10.0)
        self.assertEqual(result["energy_fj"], 200.0)
        self.assertEqual(result["cycles"], 10_000)

    def test_explicit_weight_traffic_uses_hydra_hbm_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root: Path = Path(temp_dir)
            self._fixture(root)
            backend: ChipsimBackend = ChipsimBackend(
                self._config(root), repo_root=root
            )
            translation: ChipsimTranslation = backend.translate_config()

            chipsim_root: Path = (
                Path(__file__).resolve().parents[1] / "third_party" / "CHIPSIM"
            )
            sys.path.insert(0, str(chipsim_root))
            from src.integrations.hydra.memory_placement import (
                HydraMemoryPlacement,
            )
            from src.integrations.hydra.traffic_calculator import (
                HydraTrafficCalculator,
            )

            with translation.weight_placement.open(
                "r", encoding="utf-8"
            ) as stream:
                exported: dict[str, Any] = json.load(stream)
            with translation.execution_mapping.open(
                "r", encoding="utf-8"
            ) as stream:
                execution: dict[str, Any] = json.load(stream)
            io_ids: set[int] = {
                int(memory["chipsim_chiplet_id"])
                for memory in exported["memories"]
            }
            fake_system: Any = SimpleNamespace(
                chiplets=[SimpleNamespace(bits_per_weight=8) for _ in range(24)],
                is_io_chiplet=lambda chiplet_id: chiplet_id in io_ids,
            )
            placement: HydraMemoryPlacement = HydraMemoryPlacement(
                translation.weight_placement,
                fake_system,
            )
            calculator: HydraTrafficCalculator = HydraTrafficCalculator(
                system=fake_system,
                memory_placement=placement,
                bits_per_packet=128,
            )
            destination_id: int = int(execution["layer_mapping"]["0"])
            traffic: dict[int, dict[int, int]] = (
                calculator._calculate_single_layer_weight_traffic(
                    layer_idx=0,
                    model_metrics=[{"name": "layer_0", "num_weights": 256}],
                    mapping=[(0, [(destination_id, 100.0)])],
                )
            )
            expected_source: int = int(
                exported["blocks"]["4"]["shards"][0][
                    "chipsim_memory_chiplet_id"
                ]
            )

            self.assertEqual(set(traffic), {expected_source})
            self.assertEqual(traffic[expected_source][destination_id], 16)
            self.assertAlmostEqual(
                placement.service_latency_us({expected_source: 600_000}),
                1.05,
            )


if __name__ == "__main__":
    unittest.main()
