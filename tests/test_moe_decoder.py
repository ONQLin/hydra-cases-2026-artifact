"""MoE conservation, synthetic-route provenance and pinned Kimi composition."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from Sim.config.attention_operator_config import KDAConfig, MLAConfig
from Sim.config.model_config import BaseModelConfig
from Sim.config.moe_config import MoEConfig
from Sim.entities.expert_routing import BaseExpertRouting, ExpertRoutePlan
from Sim.entities.mem_sys import mem_sys
from Sim.entities.mem_chiplet import mem_chiplet, intermediate
from Sim.processing import processing
from analytic_profile.moe import MoEProfile, DecoderBlockProfile
from analytic_profile.modern import AcceleratorRates, BaseOperatorProfile


class MoEDecoderTest(unittest.TestCase):
    def config(self, **kwargs):
        return MoEConfig(embedding_dim=8,intermediate_dim=4,num_experts=8,top_k=2,**kwargs)

    def graph(self, cfg, tokens=3, batch=2, stage='prefill', context=3):
        return MoEProfile().graph(**dict(vars(cfg),bs=tokens,batch_size=batch,L_seq=context,stage=stage))

    def test_routed_assignment_conservation_and_decode_prefix_consistency(self):
        cfg = self.config()
        routing = BaseExpertRouting.create_from_name('cyclic')
        plan = routing.route(cfg,5,2,0)
        self.assertEqual(sum(plan.counts),20)
        self.assertEqual(len(plan.expert_ids),10)
        for slot in range(2):
            decode = routing.route(cfg,1,2,4)
            self.assertEqual(decode.expert_ids[slot],plan.expert_ids[slot*5+4])
        self.assertEqual(plan.export(),routing.route(cfg,5,2,0).export())

    def test_hotspot_changes_active_weights_without_changing_resident_capacity(self):
        cfg = self.config()
        balanced = self.graph(cfg,tokens=4,context=4)
        hotspot = self.graph(replace(cfg,routing_policy='hotspot'),tokens=4,context=4)
        resident = lambda g: sum(t.nbytes for t in g.tensors.values() if t.storage=='weight')
        used = lambda g: {key for node in g.nodes for key in node.inputs if g.tensors[key].storage=='weight'}
        self.assertEqual(resident(balanced),cfg.parameter_count)
        self.assertEqual(resident(hotspot),cfg.parameter_count)
        self.assertGreater(sum(balanced.tensors[k].nbytes for k in used(balanced)),
                           sum(hotspot.tensors[k].nbytes for k in used(hotspot)))
        self.assertEqual(sum(n.macs for n in balanced.nodes),sum(n.macs for n in hotspot.nodes))
        counts = hotspot.metadata['routing']['tokens_per_expert']
        self.assertEqual(sorted(counts),[0,0,0,0,0,0,8,8])

    def test_independent_moe_costs_buffers_and_join_edges(self):
        cfg = self.config()
        graph = self.graph(cfg)
        self.assertEqual(cfg.parameter_count,936)  # 8*8+8 + 3*8*4*(8+1).
        self.assertEqual(sum(n.macs for n in graph.nodes),2112)  # Router 384, routed 1152, shared 576.
        dispatch = next(n for n in graph.nodes if n.name=='dispatch')
        self.assertEqual(sum(graph.tensors[key].nbytes for key in dispatch.outputs),6*2*8)
        self.assertEqual(graph.tensors['weighted_returns'].nbytes,6*2*8*4)
        gather = next(n for n in graph.nodes if n.name=='gather_weight')
        for expert,count in enumerate(graph.metadata['routing']['tokens_per_expert']):
            self.assertEqual(f'expert{expert}.output' in gather.inputs,count>0)
        self.assertEqual(graph.metadata['return_bytes'],graph.metadata['dispatch_bytes'])
        schedule = graph.export()['schedule']
        self.assertLess(schedule.index('dispatch'),schedule.index('gather_weight'))
        self.assertLess(schedule.index('gather_weight'),schedule.index('combine'))
        phases = graph.lower(AcceleratorRates(100,20,64,100000))
        dispatch_phase = next(p for p in phases if p.name=='dispatch')
        self.assertEqual(dispatch_phase.write_bytes,96)

    def test_invalid_routes_and_unsupported_placement_fail(self):
        for rows in (((0,0),),((0,8),),((0,),)):
            with self.assertRaises(ValueError): ExpertRoutePlan(rows,8,2,'test')
        with self.assertRaises(ValueError): replace(self.config(),top_k=9)
        with self.assertRaises(ValueError): replace(self.config(),expert_placement='distributed')
        with self.assertRaises(ValueError): replace(self.config(),routing_policy='unknown')

    def test_decoder_graph_includes_norms_residuals_and_all_weights(self):
        model = BaseModelConfig.create_from_name('kimi-decoder-fixture')
        for cfg in model.hybrid_blocks:
            profile = BaseOperatorProfile.find(cfg.kernel)()
            graph = profile.graph(**dict(vars(cfg.operator),bs=17,batch_size=2,L_seq=17,stage='prefill'))
            self.assertEqual(sum(t.nbytes for t in graph.tensors.values() if t.storage=='weight'),cfg.parameter_count)
            names = graph.export()['schedule']
            self.assertLess(names.index('attention.norm'),names.index('attention.residual'))
            self.assertLess(names.index('attention.residual'),names.index('ffn.norm'))
            self.assertEqual(names[-1],'ffn.residual')
            self.assertEqual(graph.tensors['output'].shape,(34,128))
            self.assertTrue(graph.lower(AcceleratorRates(100,20,64,1024*1024)))

    def test_kimi_topology_and_parameters_match_pinned_config(self):
        inventory = json.loads(Path('docs/workload_modernization/model_inventory.json').read_text())
        model = BaseModelConfig.create_from_name('kimi-linear-48b-a3b-text')
        source = next(s for s in inventory['models'] if s['model_id']==model.source_model)
        cfg = source['config']
        self.assertEqual(model.source_revision,source['revision'])
        self.assertEqual(model.num_layers,cfg['num_hidden_layers'])
        self.assertEqual(model.embedding_dim,cfg['hidden_size'])
        self.assertEqual(model.source_context_limit,cfg['model_max_length'])
        expected_mla = set(cfg['linear_attn_config']['full_attn_layers'])
        parameters = 0
        for layer,index in enumerate(model.block_type_sequence,1):
            block = model.hybrid_blocks[index]
            parameters += block.parameter_count
            self.assertEqual(block.cache_store,layer in expected_mla)
            if layer==1:
                self.assertEqual(block.operator.feed_forward.intermediate_dim,cfg['intermediate_size'])
            else:
                ffn = block.operator.feed_forward
                self.assertEqual(ffn.num_experts,cfg['num_experts'])
                self.assertEqual(ffn.top_k,cfg['num_experts_per_token'])
                self.assertEqual(ffn.shared_experts,cfg['num_shared_experts'])
                self.assertEqual(ffn.intermediate_dim,cfg['moe_intermediate_size'])
            attention = block.operator.attention
            if isinstance(attention,KDAConfig):
                self.assertEqual(attention.head_dim,cfg['linear_attn_config']['head_dim'])
                self.assertEqual(attention.num_heads,cfg['linear_attn_config']['num_heads'])
            else:
                self.assertIsInstance(attention,MLAConfig)
                self.assertEqual(attention.cache_layout,'expanded')
                self.assertFalse(attention.rope_enabled)
        self.assertGreater(parameters,40_000_000_000)
        self.assertLess(parameters,50_000_000_000)

    def test_per_hbm_capacity_is_checked_even_when_global_capacity_fits(self):
        model = BaseModelConfig.create_from_name('kimi-decoder-fixture')
        system = mem_sys.__new__(mem_sys)
        system._mem_chiplets = [mem_chiplet(chiplet_type=0,dram_budget=1e-6 if i==0 else 1,
                                           chiplet_id=i,chiplet_loc=(0,i)) for i in range(2)]
        system._total_budget = sum(m.dram_budget for m in system._mem_chiplets)
        system._current_used = 0
        system.blocks_alloc = {}
        with patch('Sim.entities.mem_sys.common.ByteperParam',1):
            with self.assertRaisesRegex(MemoryError,'HBM 0 capacity'):
                system.load_model_bw(model,[0],[1])

    def test_drop_removes_all_state_objects_without_touching_other_requests(self):
        system = mem_sys.__new__(mem_sys)
        chiplet = mem_chiplet(chiplet_type=0,dram_budget=1,chiplet_id=0,chiplet_loc=(0,0))
        chiplet.content_params = {10:[intermediate(size=1,req_id=10),intermediate(size=2,req_id=10)],
                                  11:[intermediate(size=4,req_id=11)]}
        chiplet.inuse_budget = system._current_used = 7
        system._mem_chiplets = [chiplet]
        worker = processing.__new__(processing)
        with patch('Sim.processing.common.RequestQueue.running',[]), \
             patch('Sim.entities.mem_sys.mem_monitor.update_utilization'):
            worker.drop_request_fromsystem(10,system,0)
        self.assertEqual(chiplet.content_params[10],[])
        self.assertEqual(len(chiplet.content_params[11]),1)
        self.assertEqual(chiplet.inuse_budget,4)
        self.assertEqual(system._current_used,4)


if __name__ == '__main__':
    unittest.main()
