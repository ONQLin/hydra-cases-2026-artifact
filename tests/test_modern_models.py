"""Verify pinned shapes, explicit memory units, and backend phase contracts."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import pandas as pd

import Sim.common as common
from Sim.config.model_config import BaseModelConfig
from Sim.config.sys_config import HPSim_Config, MetricsConfig
from Sim.entities.block import block
from Sim.entities.mem_sys import mem_sys
from Sim.entities.request import Request
from Sim.execution.native import NativeExecutionBackend
from Sim.simulator import Simulator
from analytic_profile.modern import GQAProfile, GatedDeltaNetProfile, SwiGLUProfile


class ModernModelTest(unittest.TestCase):
    def setUp(self):
        output = tempfile.TemporaryDirectory()
        self.addCleanup(output.cleanup)
        self.config = HPSim_Config(metrics_config=MetricsConfig(output_dir=output.name))
        Simulator(self.config)

    def test_shapes_match_pinned_official_configs(self):
        inventory = json.loads(Path('docs/workload_modernization/model_inventory.json').read_text())
        sources = {item['model_id']: item for item in inventory['models']}
        for name in ('qwen3-8b', 'qwen3.5-9b-text'):
            model = BaseModelConfig.create_from_name(name)
            source = sources[model.source_model]
            official = source['config'].get('text_config', source['config'])
            self.assertEqual(model.source_revision, source['revision'])
            self.assertEqual(model.num_layers, official['num_hidden_layers'])
            self.assertEqual(model.embedding_dim, official['hidden_size'])
            self.assertEqual(model.max_position_embeddings, official['max_position_embeddings'])
            self.assertEqual(len(model.block_type_sequence), model.num_layers)
            expected_types = official.get('layer_types', ['full_attention'] * model.num_layers)
            for index, kind in zip(model.block_type_sequence, expected_types):
                config = model.hybrid_blocks[index]
                self.assertEqual(config.execution_family == 'attention', kind == 'full_attention')
                if kind == 'full_attention':
                    self.assertEqual(config.attention.head_dim, official['head_dim'])
                    self.assertEqual(config.attention.q_heads, official['num_attention_heads'])
                    self.assertEqual(config.attention.kv_heads, official['num_key_value_heads'])
                else:
                    self.assertEqual(config.attention.key_heads, official['linear_num_key_heads'])
                    self.assertEqual(config.attention.value_heads, official['linear_num_value_heads'])
                    self.assertEqual(config.attention.conv_kernel_size, official['linear_conv_kernel_dim'])

    def test_independent_gqa_and_swiglu_counts(self):
        # Toy geometry catches the legacy D/KV-head assumption and missing gate FC.
        work = GQAProfile().work(embedding_dim=16, q_heads=4, kv_heads=2,
                                head_dim=8, rotary_dim=4, output_gate=True,
                                bs=3, batch_size=2, L_seq=3, stage='prefill')
        projection = next(item for item in work if item.name == 'qkv_gate_projection')
        self.assertEqual(projection.weight_bytes, 16*(32*2+16*2))
        self.assertEqual(projection.macs, 6*projection.weight_bytes)
        attention = next(item for item in work if item.name == 'causal_attention')
        self.assertEqual(attention.macs, 2*2*4*8*6)
        self.assertEqual(attention.state_bytes, 2*3*2*2*8)
        mlp = SwiGLUProfile().work(embedding_dim=16, intermediate_dim=24, bs=3, batch_size=2)
        self.assertEqual(sum(item.macs for item in mlp), 3*6*16*24)

    def test_gdn_delta_work_and_cache_traffic(self):
        config = dict(embedding_dim=16, key_heads=2, value_heads=4,
                      key_head_dim=3, value_head_dim=5, conv_kernel_size=4,
                      state_bytes=4, batch_size=2)
        prefill = GatedDeltaNetProfile().work(**config, bs=7, stage='prefill')
        decode = GatedDeltaNetProfile().work(**config, bs=1, stage='decode')
        pre = next(item for item in prefill if item.name == 'delta_recurrence')
        dec = next(item for item in decode if item.name == 'delta_recurrence')
        self.assertEqual(pre.macs, 3*2*7*4*3*5)
        self.assertEqual(dec.macs, 3*2*4*3*5)
        self.assertEqual(pre.state_bytes, 2*(4*3*5*4+(2*2*3+4*5)*4))
        self.assertEqual(dec.state_bytes, 2*pre.state_bytes)

    def test_hybrid_state_and_memory_admission(self):
        model = BaseModelConfig.create_from_name('qwen3.5-9b-text')
        recurrent, attention = model.hybrid_blocks
        self.assertNotIn('mamba', recurrent.type_name)
        self.assertEqual(recurrent.states, 32*128*128*4+(2*16*128+32*128)*4)
        self.assertEqual(attention.states, 2*4*256)
        state = block(128, recurrent)
        grown = block(1024, recurrent)
        self.assertEqual(state.states_store, grown.states_store)
        self.assertEqual(grown.additional_memory_mib(common.convert_param_mB(state.states_store)),
                         common.convert_param_mB(grown.peak_intermediate_store))
        previous = block(128, attention)
        current = block(129, attention)
        self.assertEqual(current.additional_memory_mib(common.convert_param_mB(previous.states_store)),
                         common.convert_param_mB(current.peak_intermediate_store + 2048))
        groups = mem_sys.get_global_index_map(model.block_type_sequence, model.hybrid_blocks)
        self.assertEqual(groups['transformer'], list(range(3, 32, 4)))
        self.assertEqual(len(groups['mamba']), 24)

    def test_prefill_activations_and_context_rejection(self):
        for name in ('qwen3-8b', 'qwen3.5-9b-text'):
            model = BaseModelConfig.create_from_name(name)
            request = Request(0, 32, 2, 32, model)
            request.fill_request()
            for prefill, decode in zip(request._infs['prefill'].blocks, request._infs['decode'].blocks):
                self.assertEqual(prefill.output_act, 32*decode.output_act)
                self.assertEqual(prefill.peak_intermediate_store, 32*decode.peak_intermediate_store)
            with self.assertRaisesRegex(ValueError, 'Context'):
                block(model.max_position_embeddings+1, model.hybrid_blocks[0])
            with self.assertRaisesRegex(ValueError, 'context limit'):
                Request(0, 32, 2, model.max_position_embeddings-1, model).fill_request()
        legacy = BaseModelConfig.create_from_name('llama3-8b')
        request = Request(0, 32, 2, 32, legacy)
        request.fill_request()
        self.assertEqual(request._infs['prefill'].blocks[0].output_act, 4096)

    def test_new_profiles_export_all_work_and_weight_counts(self):
        backend = NativeExecutionBackend(self.config)
        for name in ('qwen3-8b', 'qwen3.5-9b-text'):
            for config in BaseModelConfig.create_from_name(name).hybrid_blocks:
                for stage in ('prefill', 'decode'):
                    chip = SimpleNamespace(chiplet_type=config.get_accelerators(stage)[0])
                    for length in (1, 32, 2048):
                        profiles = backend.profile_block(block(length, config), stage, 2, chip, 128)
                        self.assertEqual(len(profiles), 2)
                        for result in profiles:
                            self.assertTrue(result.execution_phases)
                            reconstructed = sum(max(p.compute_ns, p.memory_bytes/128) for p in result.execution_phases)
                            self.assertAlmostEqual(result.total_latency, reconstructed, delta=1)
                        from analytic_profile.modern import BaseOperatorProfile
                        weights = 0
                        for configs, names in zip(config.layer_configs, config.layers):
                            for layer, kernel in zip(configs, names):
                                work = BaseOperatorProfile.find(kernel)().work(
                                    **dict(vars(layer), bs=length if stage == 'prefill' else 1,
                                           batch_size=2, L_seq=length, stage=stage))
                                weights += sum(item.weight_bytes for item in work)
                        self.assertEqual(weights, config.parameter_count)

    def test_heterogeneous_batches_are_rejected(self):
        lengths = pd.DataFrame({'num_prefill_tokens': [16, 32], 'context_length': [16, 32]})
        for name in ('qwen3-8b', 'qwen3.5-9b-text'):
            model = BaseModelConfig.create_from_name(name)
            with self.assertRaisesRegex(ValueError, 'batch size 1'):
                model.validate_batch_lengths(lengths, 2)
            model.validate_batch_lengths(lengths, 1)
            model.validate_batch_lengths(lengths.iloc[[0, 0]], 2)

    def test_unsupported_precision_and_policy_fail_early(self):
        self.config.workload_config.model_config = BaseModelConfig.create_from_name('qwen3-8b')
        self.config.workload_config.bytes_per_param = 2
        with self.assertRaisesRegex(ValueError, 'one-byte'):
            Simulator(self.config)
        self.config.workload_config.bytes_per_param = 1
        self.config.mapping_config.task_parallelism = 'tensor'
        with self.assertRaisesRegex(ValueError, 'pipeline'):
            Simulator(self.config)


if __name__ == '__main__':
    unittest.main()
