"""Grouped router algebra, full DeepSeek shapes and single-family placement."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import Sim.common as common
from Sim.config.model_config import BaseModelConfig
from Sim.config.sys_config import BaseChipletPlacerConfig
from tests.validation.modern_models.validate import ModernModelValidation
from Sim.config.moe_config import GroupedMoEConfig, MoEConfig
from Sim.entities.expert_routing import BaseExpertRouting, GroupedExpertRoutePlan
from Sim.entities.mem_sys import mem_sys
from Sim.entities.mem_chiplet import mem_chiplet
from analytic_profile.moe import GroupedMoEProfile, DecoderBlockProfile
from analytic_profile.modern import BaseOperatorProfile, AcceleratorRates


class GroupedMoETest(unittest.TestCase):
    def config(self):
        return GroupedMoEConfig(embedding_dim=8,intermediate_dim=4,num_experts=8,
                               top_k=3,num_groups=4,top_k_groups=2)

    def graph(self, cfg, tokens=1):
        return GroupedMoEProfile().graph(**dict(vars(cfg),bs=tokens,batch_size=1,L_seq=tokens,stage='prefill'))

    def execute_router(self, graph, cfg, logits, bias):
        """Interpret production router nodes on unquantized supplied logits."""
        values = {'logits':logits,'router.bias':bias}
        topk = lambda x,k: np.argsort(-x,axis=-1,kind='stable')[...,:k]
        for node in graph.ordered_nodes():
            if node.name=='dispatch': break
            if node.name=='router': continue  # GEMM logits are supplied independently.
            a = [values[key] for key in node.inputs]
            if node.name=='router.sigmoid': out = np.exp(-np.logaddexp(0,-a[0]))
            elif node.name=='router.correction': out = a[0]+a[1]
            elif node.name=='router.group_scores':
                grouped = a[0].reshape(-1,cfg.num_groups,cfg.num_experts//cfg.num_groups)
                out = np.take_along_axis(grouped,topk(grouped,cfg.group_score_top_k),axis=-1).sum(-1)
            elif node.name=='router.group_topk': out = topk(a[0],cfg.top_k_groups)
            elif node.name=='router.group_mask':
                allowed = np.zeros((len(logits),cfg.num_groups),dtype=bool)
                np.put_along_axis(allowed,a[1],True,axis=-1)
                out = np.where(np.repeat(allowed,cfg.num_experts//cfg.num_groups,axis=-1),a[0],-np.inf)
            elif node.name=='router.expert_topk': out = topk(a[0],cfg.top_k)
            elif node.name=='router.gather_weights': out = np.take_along_axis(a[0],a[1],axis=-1)
            elif node.name=='router.weight_sum': out = a[0].sum(-1,keepdims=True)
            elif node.name=='router.normalize_scale': out = a[0]/a[1]*cfg.routed_scaling_factor
            else: raise AssertionError(node.name)
            values[node.outputs[0]] = out
        return values['route_ids'],values['route_weights']

    def test_correction_changes_selection_but_not_combination_weights(self):
        cfg = self.config();graph = self.graph(cfg)
        scores = np.array([[.1,.2,.3,.4,.5,.6,.7,.8]])
        logits = np.log(scores/(1-scores))
        ids,weights = self.execute_router(graph,cfg,logits,np.array([0,0,1,1,0,0,0,0]))
        np.testing.assert_array_equal(ids,[[3,2,7]])
        np.testing.assert_allclose(weights,[[2/3,.5,4/3]])
        # Even negative corrected scores must not select a masked group.
        ids_negative,weights_negative = self.execute_router(graph,cfg,logits,np.array([0,0,1,1,0,0,0,0])-10)
        np.testing.assert_array_equal(ids_negative,ids)
        np.testing.assert_allclose(weights_negative,weights)

    def test_router_matches_independent_group_selection_oracle(self):
        rng = np.random.default_rng(51)
        cfg = self.config();graph = self.graph(cfg,7)
        logits = rng.normal(size=(7,8));bias = rng.normal(size=8)
        ids,weights = self.execute_router(graph,cfg,logits,bias)
        for row in range(7):
            original = [1/(1+np.exp(-x)) for x in logits[row]]
            corrected = [x+b for x,b in zip(original,bias)]
            sums = [sum(sorted(corrected[g*2:g*2+2],reverse=True)[:2]) for g in range(4)]
            groups = sorted(range(4),key=lambda g:-sums[g])[:2]
            candidates = [e for e in range(8) if e//2 in groups]
            expected = sorted(candidates,key=lambda e:-corrected[e])[:3]
            expected_weights = np.array([original[e] for e in expected])
            expected_weights *= 2.5/expected_weights.sum()
            np.testing.assert_array_equal(ids[row],expected)
            np.testing.assert_allclose(weights[row],expected_weights,rtol=1e-12)

    def test_synthetic_routes_respect_groups_and_decode_offsets(self):
        for policy in ('grouped_cyclic','grouped_hotspot'):
            cfg = replace(self.config(),routing_policy=policy,top_k=4)
            route = BaseExpertRouting.create_from_name(policy)
            plan = route.route(cfg,19,2,0)
            self.assertEqual(sum(plan.counts),19*2*4)
            for row in plan.expert_ids:
                self.assertEqual(len(set(row)),4)
                self.assertLessEqual(len({e//2 for e in row}),2)
            decode = route.route(cfg,1,2,18)
            for batch in range(2):self.assertEqual(decode.expert_ids[batch],plan.expert_ids[batch*19+18])
        with self.assertRaises(ValueError): GroupedExpertRoutePlan(((0,2,4),),8,3,'bad',4,2)
        for changes in (dict(num_groups=3),dict(top_k_groups=0),dict(group_score_top_k=3),
                        dict(top_k=5),dict(routed_scaling_factor=float('nan')),dict(routing_policy='cyclic')):
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(self.config(),**changes)
        with self.assertRaises(ValueError): MoEConfig(routing_policy='grouped_cyclic')

    def test_router_dependencies_and_weight_storage_are_explicit(self):
        cfg=self.config();graph=self.graph(cfg,3)
        self.assertIs(BaseOperatorProfile.find('ModernGroupedMoE'),GroupedMoEProfile)
        self.assertEqual(sum(t.nbytes for t in graph.tensors.values() if t.storage=='weight'),cfg.parameter_count)
        nodes={n.name:n for n in graph.nodes}
        self.assertEqual(nodes['router.gather_weights'].inputs,('scores','route_ids'))
        self.assertEqual(nodes['router.group_mask'].inputs,('selection_scores','group_ids'))
        order=graph.export()['schedule']
        self.assertLess(order.index('router.group_topk'),order.index('dispatch'))
        self.assertLess(order.index('router.normalize_scale'),order.index('gather_weight'))
        phases=graph.lower(AcceleratorRates(100,20,64,1024))
        phase=next(p for p in phases if p.name=='router.gather_weights')
        self.assertEqual(phase.read_bytes,3*8*4+3*3*4)
        self.assertEqual(phase.write_bytes,3*3*4)
        self.assertEqual(sum(n.macs for n in graph.nodes),3*8*8+3*(3+1)*3*8*4)


class DeepSeekModelTest(unittest.TestCase):
    def test_pinned_topology_and_resident_bytes(self):
        record=json.loads(Path('docs/workload_modernization/deepseek/source_config.json').read_text())
        cfg=record['config'];model=BaseModelConfig.create_from_name('deepseek-v3-text')
        self.assertEqual(model.source_revision,record['revision'])
        self.assertEqual(model.num_layers,cfg['num_hidden_layers'])
        self.assertEqual(model.block_type_sequence,[0]*cfg['first_k_dense_replace']+[1]*58)
        self.assertEqual(model.num_M_blocks,0)
        for block in model.hybrid_blocks:
            att=block.operator.attention
            self.assertEqual(att.num_heads,cfg['num_attention_heads'])
            self.assertEqual(att.q_lora_rank,cfg['q_lora_rank'])
            self.assertEqual(att.kv_lora_rank,cfg['kv_lora_rank'])
            self.assertEqual(att.cache_bytes_per_token,576)
            self.assertEqual(att.cache_layout,'absorbed')
            graph=DecoderBlockProfile().graph(**dict(vars(block.operator),bs=1,batch_size=1,L_seq=1,stage='prefill'))
            self.assertEqual(sum(t.nbytes for t in graph.tensors.values() if t.storage=='weight'),block.parameter_count)
        moe=model.hybrid_blocks[1].operator.feed_forward
        for name,key in (('num_experts','n_routed_experts'),('top_k','num_experts_per_tok'),
                         ('num_groups','n_group'),('top_k_groups','topk_group'),('routed_scaling_factor','routed_scaling_factor')):
            self.assertEqual(getattr(moe,name),cfg[key])
        size=sum(model.hybrid_blocks[i].parameter_count for i in model.block_type_sequence)
        self.assertGreater(size,660_000_000_000)
        self.assertLess(size,680_000_000_000)
        self.assertGreater(size,16*16*1024**3)  # Original hardware cannot hold this decoder.

    def test_original_hardware_rejects_all_resident_expert_weights(self):
        model=BaseModelConfig.create_from_name('deepseek-v3-text')
        system=mem_sys.__new__(mem_sys)
        system._mem_chiplets=[mem_chiplet(chiplet_type=0,dram_budget=16,chiplet_id=i,chiplet_loc=(0,i)) for i in range(16)]
        system._total_budget=sum(m.dram_budget for m in system._mem_chiplets)
        system._current_used=0;system.blocks_alloc={}
        with patch.object(common,'ByteperParam',1),patch('Sim.entities.mem_sys.mem_monitor.update_utilization'):
            with self.assertRaisesRegex(MemoryError,'weights exceed HBM'):
                system.load_model_bw(model,[],list(range(16)))

    def test_all_attention_heterogeneous_blocks_need_no_recurrent_hbm_group(self):
        model=BaseModelConfig.create_from_name('deepseek-decoder-fixture')
        system=mem_sys.__new__(mem_sys)
        system._mem_chiplets=[mem_chiplet(chiplet_type=0,dram_budget=1,chiplet_id=i,chiplet_loc=(0,i)) for i in range(2)]
        system._total_budget=sum(m.dram_budget for m in system._mem_chiplets)
        system._current_used=0;system.blocks_alloc={}
        with patch.object(common,'ByteperParam',1),patch('Sim.entities.mem_sys.mem_monitor.update_utilization'):
            system.load_model_bw(model,[],[0,1])
        names=[p.name for m in system._mem_chiplets for p in m.content_params.get('weights',[])]
        self.assertEqual(sorted(names),['Block 0 weights','Block 1 weights'])
        with self.assertRaisesRegex(ValueError,'populated block group'):
            system.load_model_bw(model,[0,1],[])


class CapacityHardwareTest(unittest.TestCase):
    def test_uniform_grid_dimensions_and_initial_mapping(self):
        default=BaseChipletPlacerConfig()
        self.assertEqual(default.chiplets_mapping.shape,(6,4))
        self.assertEqual(default.chip_graph.graph.number_of_nodes(),24)
        enlarged=BaseChipletPlacerConfig(num_nodes=72,int_width=36,int_height=2)
        self.assertEqual(enlarged.chiplets_mapping.shape,(36,2))
        self.assertEqual(enlarged.chip_graph.graph.number_of_nodes(),72)
        self.assertEqual(int(enlarged.chiplets_mapping.sum()),0)
        with self.assertRaisesRegex(ValueError,'node count'):
            BaseChipletPlacerConfig(num_nodes=72)
        with self.assertRaisesRegex(ValueError,'mapping'):
            BaseChipletPlacerConfig(num_nodes=72,int_width=36,int_height=2,chiplets_mapping=[[0]])

    def test_capacity_override_preserves_source_point_and_outer_ring_policy(self):
        with TemporaryDirectory() as directory:
            from tests.validation.prepare_inputs import ValidationInputs
            trace = ValidationInputs.write_trace(Path(directory)/'trace.csv', prefill=2)
            args=SimpleNamespace(output_dir=Path(directory),memory_chiplets=64,models=['deepseek-v3-text'],
                backends=['hydra_sim'],trace=trace,
                dataset_label='chat',arrival_interval=.01,require_complete=True)
            validator=ModernModelValidation(args)
            point=validator.select_hardware()
            record=json.loads((Path(directory)/'manifest.json').read_text())
            self.assertEqual(int(point.row['Num M']),64)
            self.assertNotEqual(record['source_row']['Num M'],point.row['Num M'])
            grid=record['hardware_overrides']['grid']
            self.assertEqual(grid['width']*grid['height'],grid['nodes'])
            perimeter=grid['nodes']-max(grid['width']-2,0)*max(grid['height']-2,0)
            self.assertGreaterEqual(perimeter,64)


if __name__ == '__main__':
    unittest.main()
