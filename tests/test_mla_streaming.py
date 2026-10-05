"""Online-softmax algebra, causal tiling, cache traffic and SRAM boundaries."""

from dataclasses import replace
import unittest

import numpy as np

from Sim.config.attention_operator_config import MLAConfig
from Sim.config.model_config import BaseModelConfig
from analytic_profile.attention import MLAProfile
from analytic_profile.mla import MLAStreamingSchedule, OnlineSoftmaxProgram
from analytic_profile.modern import AcceleratorRates


class MLAStreamingTest(unittest.TestCase):
    def config(self, layout='expanded'):
        return MLAConfig(embedding_dim=8,num_heads=3,q_lora_rank=0,kv_lora_rank=4,
                         qk_nope_head_dim=2,qk_rope_head_dim=2,v_head_dim=3,
                         cache_layout=layout,algorithm='streaming',query_tile_size=3,
                         key_tile_size=2,head_tile_size=2)

    def reference(self, schedule, query, key, value):
        """Interpret the production tile graph; numerical execution is test-only."""
        result = np.zeros((*query.shape[:-1],value.shape[-1]),dtype=query.dtype)
        for batch in range(schedule.batch_size):
            for head,h,q,n in schedule.query_tiles():
                state = dict(max_before=np.full((h,n,1),-np.inf,dtype=query.dtype),
                             sum_before=np.zeros((h,n,1),dtype=query.dtype),
                             acc_before=np.zeros((h,n,value.shape[-1]),dtype=query.dtype))
                for start,k in schedule.key_tiles(q,n):
                    x = dict(state,q=query[batch,head:head+h,q:q+n],
                             k=key[batch,head:head+h,start:start+k],
                             v=value[batch,head:head+h,start:start+k])
                    for node in schedule.program(h,n,k).graph.ordered_nodes():
                        a = [x[name] for name in node.inputs]
                        if node.name == 'scores':
                            out = a[0]@a[1].swapaxes(-1,-2) / np.sqrt(schedule.config.qk_nope_head_dim+schedule.config.qk_rope_head_dim)
                            visible = np.arange(start,start+k)[None,:] <= (schedule.context-schedule.tokens+np.arange(q,q+n))[:,None]
                            out = np.where(visible,out,-np.inf)
                        elif node.name == 'running_max': out = np.maximum(a[0].max(-1,keepdims=True),a[1])
                        elif node.name in ('rescale','probabilities'): out = np.exp(a[0]-a[1])
                        elif node.name == 'tile_sum': out = a[0].sum(-1,keepdims=True)
                        elif node.name == 'running_sum': out = a[0]*a[1]+a[2]
                        elif node.name == 'scaled_acc': out = a[0]*a[1]
                        elif node.name == 'weighted': out = a[0]@a[1]
                        elif node.name == 'running_acc': out = a[0]+a[1]
                        else: raise AssertionError(node.name)
                        x[node.outputs[0]] = out
                    state = {name+'_before':x[name+'_after'] for name in ('max','sum','acc')}
                result[batch,head:head+h,q:q+n] = state['acc_before']/state['sum_before']
        return result

    def dense(self, cfg, query, key, value):
        scores = query@key.swapaxes(-1,-2) / np.sqrt(cfg.qk_nope_head_dim+cfg.qk_rope_head_dim)
        t,l = query.shape[-2],key.shape[-2]
        scores = np.where(np.arange(l)[None,:] <= (l-t+np.arange(t))[:,None],scores,-np.inf)
        probs = np.exp(scores-scores.max(-1,keepdims=True))
        return (probs/probs.sum(-1,keepdims=True))@value

    def test_dense_equivalence_partial_tiles_batch_decode_and_extreme_logits(self):
        rng = np.random.default_rng(12)
        for layout in ('expanded','absorbed'):
            cfg = self.config(layout)
            for tokens,length in ((1,1),(1,17),(7,7),(17,17)):
                for dtype,scale in ((np.float64,1),(np.float32,50)):
                    with self.subTest(layout=layout,tokens=tokens,length=length,dtype=dtype):
                        schedule = MLAStreamingSchedule(cfg,2,tokens,length)
                        q = (rng.normal(size=(2,3,tokens,schedule.key_dim))*scale).astype(dtype)
                        k = (rng.normal(size=(2,3,length,schedule.key_dim))*scale).astype(dtype)
                        v = rng.normal(size=(2,3,length,schedule.value_dim)).astype(dtype)
                        np.testing.assert_allclose(self.reference(schedule,q,k,v),self.dense(cfg,q,k,v),rtol=2e-5,atol=2e-5)

    def test_absorption_matches_expanded_with_shared_latent_and_weights(self):
        rng = np.random.default_rng(43)
        cfg = self.config()
        q = rng.normal(size=(2,3,7,4)); latent = rng.normal(size=(2,7,4)); pos = rng.normal(size=(2,7,2))
        wk = rng.normal(size=(3,2,4)); wv = rng.normal(size=(3,3,4))
        key = np.concatenate((np.einsum('blc,hpc->bhlp',latent,wk),np.broadcast_to(pos[:,None],(2,3,7,2))),axis=-1)
        value = np.einsum('blc,hvc->bhlv',latent,wv)
        expanded = self.reference(MLAStreamingSchedule(cfg,2,7,7),q,key,value)
        absorbed_query = np.concatenate((np.einsum('bhqp,hpc->bhqc',q[...,:2],wk),q[...,2:]),axis=-1)
        absorbed_key = np.broadcast_to(np.concatenate((latent,pos),axis=-1)[:,None],(2,3,7,6))
        absorbed_value = np.broadcast_to(latent[:,None],(2,3,7,4))
        absorbed = self.reference(MLAStreamingSchedule(replace(cfg,cache_layout='absorbed'),2,7,7),absorbed_query,absorbed_key,absorbed_value)
        np.testing.assert_allclose(np.einsum('bhqc,hvc->bhqv',absorbed,wv),expanded,rtol=1e-12,atol=1e-12)

    def test_rectangular_work_and_cache_reads_have_independent_closed_counts(self):
        for layout,dim,value,cache in (('expanded',4,3,7),('absorbed',6,4,6)):
            cfg = replace(self.config(layout),num_heads=1,query_tile_size=2,key_tile_size=2)
            schedule = MLAStreamingSchedule(cfg,1,5,5)
            # Query blocks 2,2,1 read 2,4,5 keys: 17 rectangle pairs, 15 causal pairs.
            self.assertEqual(schedule.operation_counts()[0],17*(dim+value))
            phases = schedule.lower(AcceleratorRates(1,1,1,10000))
            self.assertEqual(sum(p.memory_bytes for p in phases),5*dim+11*cache+5*value)
            self.assertEqual(sum(p.write_bytes for p in phases),5*value)

    def test_sram_boundary_rejects_one_byte_short_without_hidden_spill(self):
        schedule = MLAStreamingSchedule(self.config(),2,17,17)
        capacity = schedule.peak_scratch_bytes()
        self.assertTrue(schedule.lower(AcceleratorRates(1,1,1,capacity)))
        with self.assertRaisesRegex(ValueError,'Reduce tile sizes'):
            schedule.lower(AcceleratorRates(1,1,1,capacity-1))

    def test_query_remains_resident_after_qk_until_the_next_key_tile(self):
        program = OnlineSoftmaxProgram(2,3,2,4,3)
        # At weighted PV: pinned Q(24), V(12), P(48), scaled/weighted accumulators(72 each).
        # Both running-state versions add 2*2*3*(3+2)*4 = 240 bytes.
        self.assertEqual(program.resident_bytes,24+12+48+72+72+240)
        with self.assertRaises(ValueError):
            program.graph.peak_activation_bytes(retained_inputs=('acc_before',))

    def test_no_quadratic_tensor_and_cache_append_precedes_stream(self):
        cfg = self.config()
        peaks = []
        for length in (64,128):
            g = MLAProfile().graph(**dict(vars(cfg),bs=length,batch_size=2,L_seq=length,stage='prefill'))
            self.assertNotIn('scores',g.tensors)
            order = g.export()['schedule']
            self.assertLess(order.index('cache_commit'),order.index('attention_stream'))
            self.assertEqual(sum(t.nbytes for n,t in g.tensors.items() if n.endswith('_append')),2*length*cfg.cache_bytes_per_token)
            peaks.append(g.peak_activation_bytes())
        self.assertLessEqual(peaks[1],2*peaks[0])
        g = MLAProfile().graph(**dict(vars(cfg),bs=1,batch_size=2,L_seq=128,stage='decode'))
        self.assertEqual(g.tensors['key_append'].nbytes+g.tensors['value_append'].nbytes,2*cfg.cache_bytes_per_token)

    def test_factory_preserves_historical_model_and_selects_streaming_variant(self):
        old = BaseModelConfig.create_from_name('kimi-linear-48b-a3b-text')
        new = BaseModelConfig.create_from_name('kimi-linear-48b-a3b-text-streaming')
        self.assertEqual(old.block_type_sequence,new.block_type_sequence)
        self.assertEqual(old.hybrid_blocks[2].operator.attention.algorithm,'materialized')
        self.assertEqual(new.hybrid_blocks[2].operator.attention.algorithm,'streaming')
        with self.assertRaises(ValueError): MLAConfig(algorithm='unknown')
        with self.assertRaises(ValueError): MLAConfig(query_tile_size=0)


if __name__ == '__main__':
    unittest.main()
